from .nodes.pipeline_loader import PipelineLoader,GarmentNet_IDM, TryOnNet_IDM
from .nodes.idm_vton import IDM_VTON
from .nodes.idm_vton_low_vram import IDM_VTON_low_VRAM

NODE_CLASS_MAPPINGS = {
    "PipelineLoader": PipelineLoader,
    "IDM-VTON": IDM_VTON,
    "IDM-VTON-low-VRAM":IDM_VTON_low_VRAM,
    "GarmentNet_IDM_Loader":GarmentNet_IDM,
    "TryOnNet_IDM_Loader":TryOnNet_IDM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PipelineLoader": "Load IDM-VTON Pipeline",
    "IDM-VTON": "Run IDM-VTON Inference",
    "IDM-VTON-low-VRAM": "Run IDM-VTON Inference in low VRAM Mode",
    "GarmentNet_IDM_Loader":"GarmentNet_IDM_Loader",
    "TryOnNet_IDM_Loader":"TryOnNet_IDM_Loader"
}