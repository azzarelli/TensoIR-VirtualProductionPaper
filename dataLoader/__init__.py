from .blender import BlenderDataset
from .vsr_multilight import VSR_multi_lights


dataset_dict = {'blender': BlenderDataset,
                'tensoIR_unknown_general_multi_lights': VSR_multi_lights, # Changes to VSR dataset
                }
