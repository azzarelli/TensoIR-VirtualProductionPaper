import os, random
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from dataLoader.ray_utils import *
from typing import NamedTuple

class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    fx: np.array
    fy: np.array
    cx: np.array
    cy: np.array
    
    k1: np.array
    k2: np.array
    p1: np.array
    p2: np.array

    image_path: str
    canon_path:str
    so_path: str
    
    image: torch.Tensor
    canon:torch.Tensor
    mask: torch.Tensor


    uid: int    
    width: int
    height: int
    time : int
    
    def update_canon(self, path):
        return self._replace(canon_path=path)

def readCamerasFromTransforms(path, transformsfile, plot=False):
    cam_infos = []

    tf_path = os.path.join(path, transformsfile)
    with open(tf_path, "r") as json_file:
        contents = json.load(json_file)

    # Global intrinsics
    g_fx = contents.get("fl_x")
    g_fy = contents.get("fl_y")
    g_cx = contents.get("cx")
    g_cy = contents.get("cy")
    g_w  = contents.get("w")
    g_h  = contents.get("h")
    g_k1 = contents.get("k1")
    g_k2 = contents.get("k2")
    g_p1 = contents.get("p1")
    g_p2 = contents.get("p2")

    # Nerfstudio normalization (transform + scale)
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        fx = frame.get("fl_x", g_fx)
        fy = frame.get("fl_y", g_fy)
        cx = frame.get("cx", g_cx)
        cy = frame.get("cy", g_cy)
        w  = frame.get("w", g_w)
        h  = frame.get("h", g_h)

        k1 = frame.get("k1", g_k1)
        k2 = frame.get("k2", g_k2)
        p1 = frame.get("p1", g_p1)
        p2 = frame.get("p2", g_p2)

        # Load and convert transform
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        R =  c2w[:3, :3]
        T = c2w[:3, 3]
        
        image_path = os.path.normpath(os.path.join(path, frame["file_path"]))

        cam_infos.append(CameraInfo(
            uid=frame.get("colmap_im_id", idx),
            R=R, T=T,
            fx=fx, fy=fy, cx=cx, cy=cy,
            k1=k1, k2=k2, p1=p1, p2=p2,
            width=w, height=h,
            image_path=image_path,
            canon_path=None,
            so_path=None,
            image=None,
            canon=None,
            mask=None,
            time=float(frame.get("time", -1.0)),
        ))
    cam_infos.sort(key=lambda c: os.path.basename(c.image_path))

    return cam_infos

class VSR_multi_lights(Dataset):
    def __init__(self,
                 root_dir,
                 hdr_dir,
                 split='train',
                 random_test=False,
                 N_vis=-1,
                 downsample=1.0,
                 sub=0,
                 light_name_list=["sunset", "snow", "courtyard"],
                 dataset=0,
                 scene=0,
                 **temp
                 ):
        """
        @param root_dir: str | Root path of dataset folder
        @param hdr_dir: str | Root path for HDR folder
        @param split: str | e.g. 'train' / 'test'
        @param random_test: bool | Whether to randomly select a test view and a lighting
        else [frames, h*w, 6]
        @param N_vis: int | If N_vis > 0, select N_vis frames from the dataset, else (-1) import entire dataset
        @param downsample: float | Downsample ratio for input rgb images
        """
        assert split in ['train', 'test']
        self.N_vis = N_vis
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_list = [x for x in self.root_dir.iterdir() if x.stem.startswith(self.split)]

        if not random_test:
            self.split_list.sort() # to render video
        if sub > 0:
            self.split_list = self.split_list[:sub]


        self.img_wh = (int(1920 / downsample), int(1080 / downsample))  
        self.white_bg = True
        self.downsample = downsample
        self.requested_light_name_list = list(light_name_list)
        
        self.transform = self.define_transforms()
        self.near_far = [0.1, 5.0]  
        
        self.scene_bbox = torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]) * self.downsample
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # HDR configs
        self.scan = self.root_dir.stem  # Scan name e.g. 'lego', 'hotdog'

        ## Load light data
        self.hdr_dir = Path(hdr_dir)
        
        self.scene = scene
        self.dataset_id = dataset
        if scene == 1:
            self.min_image_idx = 0
            self.max_image_idx = 33
        elif scene == 2:
            self.min_image_idx = 33
            self.max_image_idx = 66
        elif scene == 3:
            self.min_image_idx = 66
            self.max_image_idx = 99
        else:
            print('Scene argument incorrect')
            exit()
            
        if dataset == 1:
            self.test_cam_idx = 5
        else:
            self.test_cam_idx = 18

        self.read_lights()

        # when trainning, we will load all the rays and rgbs
        if split == 'train':
            self.read_all_frames()        
    
    
    def define_transforms(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])
        return transforms

    def read_lights(self):
        """
        Read background files from local path
        """
        self.lights_probes = dict()
        background_fps = sorted(os.listdir(self.hdr_dir))

        names = []
        subset_range = [self.min_image_idx, self.max_image_idx]
        selected_names = background_fps[subset_range[0]:subset_range[1]]

        for idx, light_name in enumerate(selected_names):
            
            if self.split == 'train' and idx > 22:
                continue
            if self.split == 'test' and idx < 23:
                continue
            
            hdr_path = self.hdr_dir / light_name
            light_rgb = self.transform(Image.open(hdr_path).convert("RGB")).permute(1, 2, 0).float()

            self.envir_map_h, self.envir_map_w = light_rgb.shape[:2]
            light_rgb = light_rgb.reshape(-1, 3)
            
            self.lights_probes[light_name] = light_rgb
            names.append(light_name)

        # Set light names list
        self.light_name_list = names
        self.light_num = len(self.light_name_list)
        print(f"Using {self.light_num} for Dataset/scene: {self.dataset_id}-{self.scene}")

    def read_all_frames(self):
        self.all_masks = []
        
        canon_cam_infos = readCamerasFromTransforms(self.root_dir, 'transforms.json')
        tot_cams = len(canon_cam_infos) - 19
        
        c_cams = canon_cam_infos[-19:]
        c_cams.pop(self.test_cam_idx)

        total_rows = 0
        for idx, cam_info in enumerate(c_cams):
            img_wh = (int(cam_info.width / self.downsample), int(cam_info.height / self.downsample))
            cam_idx = (cam_info.uid - tot_cams)

            total_rows += img_wh[0] * img_wh[1] * (self.light_num)

        estimated_bytes = total_rows * (6 * 4 + 3 * 4 + 1)
        
        print(
            f"Allocating training tensors for {total_rows} rays "
            f"(~{estimated_bytes / (1024 ** 3):.2f} GiB across rays/rgbs/light_idx)"
        )
        
        self.all_rays = torch.empty((total_rows, 6), dtype=torch.float32)
        self.all_rgbs = torch.empty((total_rows, 3), dtype=torch.float32)
        self.all_light_idx = torch.empty((total_rows, 1), dtype=torch.int8)
        row_offset = 0

        for idx in tqdm(range(len(c_cams)), desc=f'Loading {self.split} data, view number: {self.__len__()}, lighting number: {self.light_num}'):
            cam_info = c_cams[idx]
            cam_idx = (cam_info.uid - tot_cams)
            cam_name = f'cam{cam_idx:02}'

            ## Camera Intrinsic Settings ## 
            img_wh = (int(cam_info.width / self.downsample), int(cam_info.height / self.downsample))
            self.img_wh = img_wh
            # Get ray directions for all pixels, same for all images (with same H, W, focal)
            focal_x = cam_info.fx  # fov -> focal length
            focal_y = cam_info.fy  # fov -> focal length
            focal_x *= img_wh[0] / cam_info.width
            focal_y *= img_wh[1] / cam_info.height

            directions = get_ray_directions(img_wh[1], img_wh[0], [focal_x, focal_y])  # [H, W, 3]
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)

            
            ## Camera Extrinsic Settings ## 
            # (in opengl coord space)
            T = cam_info.T
            R = cam_info.R 
            cam_trans = np.eye(4, dtype=np.float32)
            cam_trans[:3, :3] = R
            cam_trans[:3, 3] = T
            
            pose = cam_trans @ self.blender2opencv # Convert back to openCV
            c2w = torch.FloatTensor(pose)  # [4, 4]
            
            # Read ray data
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

            # light_kind_to_choose = int(np.random.randint(len(self.light_name_list))) # temp
            cam_folder_path = os.path.join(self.root_dir, 'meta', 'images', cam_name)
            mask_path = os.path.join(self.root_dir, 'meta', 'masks', f"{cam_name}.png")
            for light_kind_idx in range(len(self.light_name_list)):
                
                # Read RGB data
                light_name = self.light_name_list[light_kind_idx]
                
                img_name = light_name.replace('png', 'jpg')
                
                relight_img_path = os.path.join(cam_folder_path, img_name)
                
                relight_img = Image.open(relight_img_path)
                alpha = Image.open(mask_path).split()[-1]

                if self.downsample != 1.0:
                    relight_img = relight_img.resize(img_wh, Image.Resampling.LANCZOS)
                    alpha = alpha.resize(img_wh, Image.Resampling.LANCZOS)
                relight_img = self.transform(relight_img)  # [3, H, W]
                alpha = 1.0 - self.transform(alpha)  # [1, H, W]
                relight_img = torch.cat([relight_img, alpha], dim=0)
                
                relight_img = relight_img.view(4, -1).permute(1, 0)  # [H*W, 4]
                
                
                ## Blend RGBA to RGB
                relight_rgbs = relight_img[:, :3] * relight_img[:, -1:] + (1 - relight_img[:, -1:])  # [H*W, 3]
                row_count = img_wh[0] * img_wh[1]
                next_offset = row_offset + row_count
                self.all_rays[row_offset:next_offset] = rays
                self.all_rgbs[row_offset:next_offset] = relight_rgbs
                self.all_light_idx[row_offset:next_offset] = light_kind_idx
                row_offset = next_offset

        assert row_offset == total_rows


    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def read_stack(self):
        for idx in range(self.__len__()):
            item = self.__getitem__(idx)
            rays = item['rays']
            rgbs = item['rgbs']
            self.all_rays += [rays]
            self.all_rgbs += [rgbs]
        self.all_rays = torch.stack(self.all_rays, 0)  # [len(self), H*W, 6]
        self.all_rgbs = torch.stack(self.all_rgbs, 0)  # [len(self), H*W, 3]

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx):
        print('Getting Item...', idx)
        idx = idx % 10 # TODO: need to change to len of test dataset
        self.all_masks = []
        
        canon_cam_infos = readCamerasFromTransforms(self.root_dir, 'transforms.json')
        tot_cams = len(canon_cam_infos) - 19
        
        c_cams = canon_cam_infos[-19:]
        
        cam_info = c_cams[self.test_cam_idx]
        cam_idx = (cam_info.uid - tot_cams)
        cam_name = f'cam{cam_idx:02}'
        
        ## Camera Intrinsic Settings ## 
        img_wh = (int(cam_info.width / self.downsample), int(cam_info.height / self.downsample))
        self.img_wh = img_wh
        
        # Get ray directions for all pixels, same for all images (with same H, W, focal)
        focal_x = cam_info.fx  # fov -> focal length
        focal_y = cam_info.fy  # fov -> focal length
        focal_x *= img_wh[0] / cam_info.width
        focal_y *= img_wh[1] / cam_info.height

        directions = get_ray_directions(img_wh[1], img_wh[0], [focal_x, focal_y])  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        
        ## Camera Extrinsic Settings ## 
        # (in opengl coord space)
        T = cam_info.T
        R = cam_info.R 
        cam_trans = np.eye(4, dtype=np.float32)
        cam_trans[:3, :3] = R
        cam_trans[:3, 3] = T
        
        pose = cam_trans @ self.blender2opencv # Convert back to openCV
        c2w = torch.FloatTensor(pose)  # [4, 4]
        w2c = torch.linalg.inv(c2w)  # [4, 4]
        
        # Read ray data
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

        cam_folder_path = os.path.join(self.root_dir, 'meta', 'images', cam_name)
        mask_path = os.path.join(self.root_dir, 'meta', 'masks', f"{cam_name}.png")
        
        light_kind_idx = idx
            
        # Read RGB data
        light_name = self.light_name_list[light_kind_idx % len(self.light_name_list)]
        
        img_name = light_name.replace('png', 'jpg')
        
        relight_img_path = os.path.join(cam_folder_path, img_name)

        relight_img = Image.open(relight_img_path)
        alpha = Image.open(mask_path).split()[-1]

        if self.downsample != 1.0:
            relight_img = relight_img.resize(img_wh, Image.Resampling.LANCZOS)
            alpha = alpha.resize(img_wh, Image.Resampling.LANCZOS)
        relight_img = self.transform(relight_img)  # [3, H, W]
        alpha = 1.0 - self.transform(alpha)  # [1, H, W]
        relight_img = torch.cat([relight_img, alpha], dim=0)
        
        relight_img = relight_img.view(4, -1).permute(1, 0)  # [H*W, 4]
        
        ## Blend RGBA to RGB
        rgbs = relight_img[:, :3] * relight_img[:, -1:] + (1 - relight_img[:, -1:])  # [H*W, 3]
        row_count = img_wh[0] * img_wh[1]
        next_offset = row_offset + row_count
        row_offset = next_offset

        item = {
            'img_wh': img_wh,  # (int, int)
            'light_idx': light_kind_idx,  # [rotation_num, H*W, 1]
            'rgbs': rgbs,  # [rotation_num, H*W, 3],
            'rgbs_mask': relight_img[:, -1:],  # [H*W, 1]
            'rays': rays,  # [H*W, 6]

            'c2w': c2w,  # [4, 4]
            'w2c': w2c  # [4, 4]
        }
        return item

