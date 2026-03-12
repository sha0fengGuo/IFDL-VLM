import torch
import torch.nn as nn

import os
import numpy as np
from PIL import Image

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch.nn.functional as F

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # Region-aware attributes.
        # Set attributes before model loading.
        self.enable_region_aware = getattr(args, 'enable_region_aware', True)
        self.region_weight = getattr(args, 'region_weight', 0.5)

        print(f"🎯 CLIPVisionTower initialization:")
        print(f"   model: {self.vision_tower_name}")
        print(f"   region aware: {self.enable_region_aware}")
        print(f"   region weight α: {self.region_weight}")
        print(f"   delay_load: {delay_load}")
        print(f"   unfreeze_mm_vision_tower: {getattr(args, 'unfreeze_mm_vision_tower', False)}")

        # Debug-friendly initialization path.
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
       
            try:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
              
            except Exception as e:
              
                raise

        print("🎉 CLIPVisionTower initialized")


        # if not delay_load:
        #     self.load_model()
        # elif getattr(args, 'unfreeze_mm_vision_tower', False):
        #     self.load_model()
        # else:
        #     self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, alpha_masks=None):
        def _rank0_print(*args):
            try:
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(*args)
            except Exception:
                print(*args)
        # Debug logs.
        _rank0_print(f"🔍 CLIP forward debug:")
        _rank0_print(f"   enable_region_aware: {self.enable_region_aware}")
        _rank0_print(f"   alpha_masks is None: {alpha_masks is None}")
        if alpha_masks is not None:
            if isinstance(alpha_masks, list):
                _rank0_print(f"   alpha_masks type: list, length: {len(alpha_masks)}")
                if alpha_masks:
                    _rank0_print(f"   first mask shape: {alpha_masks[0].shape}")
            else:
                _rank0_print(f"   alpha_masks shape: {alpha_masks.shape}")
        
        if not self.enable_region_aware or alpha_masks is None:
            _rank0_print("📋 Using standard CLIP (without region-aware path)")
            return self._standard_forward(images)
        
        _rank0_print("🎯 Using region-aware CLIP (mask multiplication)")
        return self._region_aware_forward(images, alpha_masks)

    def _standard_forward(self, images):
        """Standard CLIP forward path."""
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), 
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), 
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(self.dtype)

        return image_features

    def _region_aware_forward(self, images, alpha_masks):
        """Region-aware forward: fuse full-image and masked-image features."""
        if isinstance(images, list):
            features = []
            for i, (img, mask) in enumerate(zip(images, alpha_masks)):
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(f"  processing image {i+1}...")
                feature = self._process_single_image_masked(img, mask)
                features.append(feature)
            return features
        else:
            batch_features = []
            for i in range(images.shape[0]):
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(f"  processing batch image {i+1}...")
                feature = self._process_single_image_masked(images[i], alpha_masks[i])
                batch_features.append(feature)
            return torch.stack(batch_features, dim=0)

    def _process_single_image_masked(self, image, mask):
        """Core logic: single-image masked processing."""
        try:
            # 1. Global image feature.
            global_feature = self._extract_clip_features(image)
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"    global feature shape: {global_feature.shape}")
            
            # 2. Build masked image and region feature.
            masked_image = self._apply_mask_to_image(image, mask)
            region_feature = self._extract_clip_features(masked_image)
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"    region feature shape: {region_feature.shape}")
            
            # 3. Weighted fusion: α * global + (1-α) * region.
            α = self.region_weight
            combined_feature = α * global_feature + (1 - α) * region_feature
            
            # Keep output dtype consistent with training precision.
            combined_feature = combined_feature.to(dtype=self.dtype)
            
            return combined_feature
            
        except Exception as e:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"❌ Error when processing single image: {e}")
                # Fallback to global feature on failure.
            fallback_feature = self._extract_clip_features(image)
            return fallback_feature.to(dtype=self.dtype)

    def _process_single_image_standard(self, image):
        """Standard CLIP processing (without mask)."""
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"📋 Using standard CLIP (without region-aware path)")
        feature = self._extract_clip_features(image)
        # Keep output dtype consistent with input.
        return feature.to(dtype=image.dtype)

    def _apply_mask_to_image(self, image, mask):
        """
        Apply a binarized mask to image tensor.
        - image: [3, H, W] tensor
        - mask: [1, H, W] or [H, W] tensor
        - return: [3, H, W] masked image tensor
        """
        rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0

        # 1) Normalize mask to torch.Tensor[H,W] on image device/dtype.
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 3 and mask.size(0) == 1:
                mask = mask.squeeze(0)  # [H, W]
            elif mask.dim() == 2:
                pass  # [H, W]
            else:
                raise ValueError(f"Unexpected mask dimensions: {mask.shape}")
            mask = mask.to(device=image.device, dtype=image.dtype)
        else:
            if isinstance(mask, Image.Image):
                mask_np = np.array(mask.convert('L'))
            elif isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                raise ValueError(f"Unsupported mask type: {type(mask)}")
            mask = torch.from_numpy(mask_np).to(device=image.device, dtype=image.dtype)

        # 2) Binarization.
        # mask may be in [0,1] or [0,255]
        if float(mask.max()) <= 1.0:
            mask = (mask > 0.5).to(dtype=image.dtype)
        else:
            mask = (mask > 128).to(dtype=image.dtype)

        if rank0:
            print(f"    mask stats: min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")
            active_ratio = (mask > 0.5).float().mean().item()
            if active_ratio < 1e-4:
                print("    ⚠️ Mask active region is very small; region feature may become weak.")
            elif active_ratio > 0.9999:
                print("    ℹ️ Mask is almost fully active; region feature is close to global feature.")

        # 3) Resize alignment if needed.
        if mask.shape != image.shape[-2:]:
            if rank0:
                print(f"    resizing mask: {tuple(mask.shape)} -> {tuple(image.shape[-2:])}")
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)
            mask = (mask > 0.5).to(dtype=image.dtype)

        # 4) Apply mask.
        masked_image = image * mask.unsqueeze(0).expand_as(image)

        if rank0:
            print(f"    masked image stats: min={masked_image.min():.3f}, max={masked_image.max():.3f}")

        return masked_image

    def _extract_clip_features(self, image):
        """Extract standard CLIP features."""
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, H, W]
        
        forward_out = self.vision_tower(
            image.to(device=self.device, dtype=self.dtype), 
            output_hidden_states=True
        )
        features = self.feature_select(forward_out)  # [1, 576, 1024]
        return features.squeeze(0)  # [576, 1024]
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
