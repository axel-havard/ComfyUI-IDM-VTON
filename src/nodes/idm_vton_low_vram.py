import sys
sys.path.append('.')
sys.path.append('..')

import torch
from torchvision import transforms
from comfy.model_management import get_torch_device
from diffusers import DDPMScheduler
import os
DEVICE = get_torch_device()
MAX_RESOLUTION = 16384

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class IDM_VTON_low_VRAM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tryon_net": ("MODEL",),
                "garment_net": ("MODEL",),
                "noise_latent": ("LATENT",),
                "densepose_latent": ("LATENT",),
                "mask_image_latent": ("LATENT",),
                "masked_image_latent": ("LATENT",),
                "garment_image_latent": ("LATENT",),
                "garment_image_clip_embedding": ("LATENT",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "garment_positive": ("CONDITIONING", ),
                "garment_negative": ("CONDITIONING", ),
                "num_inference_steps": ("INT", {"default": 30}),
                "guidance_scale": ("FLOAT", {"default": 2.0}),
                "strength": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LATENT")
    FUNCTION = "make_inference"
    CATEGORY = "ComfyUI-IDM-VTON"

    def make_inference(self, tryon_net, garment_net,noise_latent,densepose_latent,mask_image_latent,masked_image_latent,garment_image_latent,garment_image_clip_embedding,positive,negative,garment_positive,garment_negative,num_inference_steps, guidance_scale, seed):
        
        current_dir = os.path.dirname(__file__)
        # Construct the path to the scheduler relative to the current file
        scheduler_path = os.path.join(current_dir, "idm_scheduler")
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        latents = noise_latent
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        added_cond_kwargs["image_embeds"] = tryon_net.encoder_hid_proj(garment_image_clip_embedding)

        if mask_image_latent.shape[1]>1:
            mask_image_latent = mask_image_latent[:,0:1,:,:]

        text_embeds_cloth = torch.cat([garment_negative,garment_positive],dim=0)
        prompt_embeds = torch.cat([negative,positive],dim=0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                latent_model_input = torch.cat([latent_model_input, mask_image_latent, masked_image_latent,densepose_latent], dim=1)
                
                _,reference_features = garment_net(garment_image_latent,t, text_embeds_cloth,return_dict=False)

                reference_features = list(reference_features)

                
                reference_features = [torch.cat([torch.zeros_like(d), d]) for d in reference_features]


                noise_pred = tryon_net(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    garment_features=reference_features,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return (latents,)

