import numpy as np
import random
import os, imageio
from tqdm.auto import tqdm
from utils import *
from models.relight_utils import render_with_BRDF
import torch
import torchvision.utils as vutils


@torch.no_grad()
def compute_rescale_ratio(tensoIR, dataset, sampled_num=20):
    '''compute three channel rescale ratio for albedo by sampling some views
    - Args:
        tensoIR: model
        dataset: dataset containing the G.T albedo
    - Returns:
        single_channel_ratio: median of the ratio of the first channel
        three_channel_ratio: median of the ratio of the three channels
    '''
    W, H = dataset.img_wh
    data_num = len(dataset)
    interval = data_num // sampled_num
    idx_list = [i * interval for i in range(sampled_num)]
    ratio_list = list()
    gt_albedo_list = []
    reconstructed_albedo_list = []
    for idx in tqdm(idx_list, desc="compute rescale ratio"):
        item = dataset[idx]
        frame_rays = item['rays'].squeeze(0).to(tensoIR.device) # [H*W, 6]
        gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu() # [H*W]
        gt_albedo = item['albedo'].squeeze(0).to(tensoIR.device) # [H*W, 3]
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(tensoIR.device).fill_(0)
        albedo_map = list()
        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), 3000) 
        for chunk_idx in chunk_idxs:
            rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                fresnel_chunk, acc_chunk, *temp \
                = tensoIR(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)
            albedo_map.append(albedo_chunk.detach())
        albedo_map = torch.cat(albedo_map, dim=0).reshape(H, W, 3)
        gt_albedo = gt_albedo.reshape(H, W, 3)
        gt_mask = gt_mask.reshape(H, W)
        gt_albedo_list.append(gt_albedo[gt_mask])
        reconstructed_albedo_list.append(albedo_map[gt_mask])
    # ratio = torch.stack(ratio_list, dim=0).mean(dim=0)
    gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
    single_channel_ratio = (gt_albedo_all / albedo_map_all.clamp(min=1e-6))[..., 0].median()
    three_channel_ratio, _ = (gt_albedo_all / albedo_map_all.clamp(min=1e-6)).median(dim=0)
    print("single channel rescale ratio: ", single_channel_ratio)
    print("three channels rescale ratio: ", three_channel_ratio)
    return single_channel_ratio, three_channel_ratio



def Renderer_TensoIR_train(  
                            rays=None, 
                            normal_gt=None, 
                            light_idx=None, 
                            tensoIR=None, 
                            N_samples=-1,
                            ndc_ray=False, 
                            white_bg=True, 
                            is_train=False,
                            is_relight=True,
                            sample_method='fixed_envirmap',
                            chunk_size=15000,
                            device='cuda',      
                            args=None,
                        ):

   
    rays = rays.to(device)
    light_idx = light_idx.to(device, torch.int32)
    rgb_map, depth_map, normal_map, albedo_map, roughness_map, \
        fresnel_map, acc_map, normals_diff_map, normals_orientation_loss_map, \
        acc_mask, albedo_smoothness_loss, roughness_smoothness_loss \
        = tensoIR(rays, light_idx, is_train=is_train, white_bg=white_bg, is_relight=is_relight, ndc_ray=ndc_ray, N_samples=N_samples)

    # If use GT normals
    if tensoIR.normals_kind == "gt_normals" and normal_gt is not None:
        normal_map = normal_gt.to(device)

    # Physically-based Rendering(Relighting)
    if is_relight:
        rgb_with_brdf_masked = render_with_BRDF(   
                                                depth_map[acc_mask],
                                                normal_map[acc_mask],
                                                albedo_map[acc_mask],
                                                roughness_map[acc_mask].repeat(1, 3),
                                                fresnel_map[acc_mask],
                                                rays[acc_mask],
                                                tensoIR,
                                                light_idx[acc_mask],
                                                sample_method,
                                                chunk_size=chunk_size,
                                                device=device,
                                                args=args
                                               )




        rgb_with_brdf = torch.ones_like(rgb_map) # background default to be white
        rgb_with_brdf[acc_mask] = rgb_with_brdf_masked
        # rgb_with_brdf = rgb_with_brdf * acc_map[..., None]  + (1. - acc_map[..., None])
    else:
        rgb_with_brdf = torch.ones_like(rgb_map)


    ret_kw = {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "normal_map": normal_map,
        "albedo_map": albedo_map,
        "acc_map": acc_map,
        "roughness_map": roughness_map,
        "fresnel_map": fresnel_map,
        'rgb_with_brdf_map': rgb_with_brdf,
        'normals_diff_map': normals_diff_map,
        'normals_orientation_loss_map': normals_orientation_loss_map,
        'albedo_smoothness_loss': albedo_smoothness_loss,
        'roughness_smoothness_loss': roughness_smoothness_loss,
    }

    return ret_kw


@torch.no_grad()
def evaluation_iter_TensoIR_general_multi_lights(
        test_dataset,
        tensoIR,
        args,
        renderer,
        savePath=None,
        prtx='',
        N_samples=-1,
        white_bg=False,
        ndc_ray=False,
        compute_extra_metrics=True,
        device='cuda',
        logger=None,
        step=None,
        test_all=False,
        light_idx_to_test=-1,
):

    
    rgb_maps, depth_maps= [], []
    rgb_with_brdf_maps= []



    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/nvs_with_radiance_field", exist_ok=True)
    os.makedirs(savePath + "/nvs_with_brdf", exist_ok=True)
    os.makedirs(savePath + "/normal", exist_ok=True)
    os.makedirs(savePath + "/normal_vis", exist_ok=True)
    os.makedirs(savePath + "/brdf", exist_ok=True)
    os.makedirs(savePath + "/envir_map/", exist_ok=True)
    os.makedirs(savePath + "/acc_map", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh
    gt_envir_map = None

    _, view_dirs = tensoIR.generate_envir_map_dir(256, 512)

    predicted_envir_map = tensoIR.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))
    predicted_envir_map = predicted_envir_map.reshape(256 * tensoIR.light_num, 512, 3).cpu().detach().numpy()
    predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=np.inf)
    predicted_envir_map = np.uint8(np.clip(np.power(predicted_envir_map, 1./2.2), 0., 1.) * 255.)
    if gt_envir_map is not None:
        envirmap = np.concatenate((gt_envir_map, predicted_envir_map), axis=1)
    else:
        envirmap = predicted_envir_map
    # save predicted envir map
    imageio.imwrite(f'{savePath}/envir_map/{prtx}envirmap.png', envirmap)

    # compute global rescale ratio for predicted albedo
    if test_all:
        global_rescale_value_single, global_rescale_value_three = compute_rescale_ratio(tensoIR, test_dataset, sampled_num=20)
        global_rescale_value_single, global_rescale_value_three = global_rescale_value_single.cpu(), global_rescale_value_three.cpu()


    for idx in range(1):
        item = test_dataset.__getitem__(idx)
        rays = item['rays']                 # [H*W, 6]
        light_idx = item['light_idx']
        
        rgb_map, rgb_with_brdf_map, depth_map= [], [], []

        chunk_idxs = torch.split(torch.arange(rays.shape[0]), args.batch_size_test)
        for chunk_idx in chunk_idxs:
            ret_kw= renderer(   
                rays[chunk_idx], 
                None, # not used
                torch.tensor([light_idx for btchsz in range(args.batch_size_test)]).cuda(),
                tensoIR, 
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                sample_method='fixed_envirmap',
                chunk_size=args.relight_chunk_size,  
                device=device,
                args=args
            )
            
            rgb_map.append(ret_kw['rgb_map'].detach().cpu())
            depth_map.append(ret_kw['depth_map'].detach().cpu())

            rgb_with_brdf_map.append(ret_kw['rgb_with_brdf_map'].detach().cpu())
        
        rgb_map = torch.cat(rgb_map)
        depth_map = torch.cat(depth_map)

        rgb_with_brdf_map = torch.cat(rgb_with_brdf_map)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_with_brdf_map = rgb_with_brdf_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).detach().cpu(), depth_map.reshape(H, W).detach().cpu()
        rgb_with_brdf_map = rgb_with_brdf_map.reshape(H, W, 3).detach().cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_with_brdf_map = (rgb_with_brdf_map.numpy() * 255).astype('uint8')

        rgb_maps.append(rgb_map)
        rgb_with_brdf_maps.append(rgb_with_brdf_map)
        depth_maps.append(depth_map)

        if savePath is not None:
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            rgb_with_brdf_map = np.concatenate((rgb_with_brdf_map), axis=1)

            imageio.imwrite(f'{savePath}/nvs_with_radiance_field/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/nvs_with_brdf/{prtx}{idx:03d}.png', rgb_with_brdf_map)

    return 0, 0, 0, 0, 0