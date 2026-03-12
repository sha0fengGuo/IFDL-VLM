import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch.nn.functional as F
from PIL import Image  # keep PIL branch in _apply_mask_to_image
import numpy as np     # keep numpy branch in _apply_mask_to_image



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
    # Legacy field: region_weight.
    self.global_weight = getattr(args, 'global_weight', 0.5)  # α is the global-feature weight


        print(f"🎯 CLIPVisionTower initialization:")
        print(f"   model: {self.vision_tower_name}")
        print(f"   region aware: {self.enable_region_aware}")
        print(f"   global weight α: {self.global_weight}")
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
        # Debug logs.
        print(f"🔍 CLIP forward debug:")
        print(f"   enable_region_aware: {self.enable_region_aware}")
        print(f"   alpha_masks is None: {alpha_masks is None}")
        if alpha_masks is not None:
            if isinstance(alpha_masks, list):
                print(f"   alpha_masks type: list, length: {len(alpha_masks)}")
                if alpha_masks:
                    print(f"   first mask shape: {alpha_masks[0].shape}")
            else:
                print(f"   alpha_masks shape: {alpha_masks.shape}")
        
        if not self.enable_region_aware or alpha_masks is None:
            print("📋 Using standard CLIP (without region-aware path)")
            return self._standard_forward(images)
        
        print("🎯 Using region-aware CLIP (patch-level fusion + background attenuation)")

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
                image_feature = self.feature_select(image_forward_out).squeeze(0).to(image.dtype)  # unified as [P, D]
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(self.dtype)  # [B,P,D]
        return image_features

    def _region_aware_forward(self, images, alpha_masks):
        """Region-aware forward: fuse full-image and masked-image features."""
        if isinstance(images, list):
            features = []
            for i, (img, mask) in enumerate(zip(images, alpha_masks)):
                print(f"  processing image {i+1}...")
                feature = self._process_single_image_masked(img, mask)
                features.append(feature)
            return features
        else:
            batch_features = []
            for i in range(images.shape[0]):
                print(f"  processing batch image {i+1}...")
                feature = self._process_single_image_masked(images[i], alpha_masks[i])
                batch_features.append(feature)
            return torch.stack(batch_features, dim=0)

    def _process_single_image_masked(self, image, mask):
        try:
            # 1) Global feature
            global_feat = self._extract_clip_features(image)          # [P, D]

            # 2) Region image (blurred/mean background) + aligned mask_2d
            masked_image, mask_aligned = self._apply_mask_to_image(image, mask, return_mask=True)  # [3,H,W], [H,W]
            region_feat = self._extract_clip_features(masked_image)   # [P, D]

            # 3) Downsample aligned mask to patch grid
            mask_patches = self._downsample_mask_to_patches(mask_aligned)  # [P] ∈ [0,1]

            # 4) Patch-level fusion weights (α=global_weight), clamped to [0,1]
            alpha = float(self.global_weight)
            alpha = max(0.0, min(1.0, alpha))

            w_region = (1.0 - alpha) * mask_patches                   # [P]
            w_global = 1.0 - w_region                                 # [P]

            combined = w_global.unsqueeze(1) * global_feat + w_region.unsqueeze(1) * region_feat
            return combined.to(dtype=self.dtype)

        except Exception as e:
            print(f"❌ Error when processing single image: {e}")
            fallback_feature = self._extract_clip_features(image)
            return fallback_feature.to(dtype=self.dtype)


    def _process_single_image_standard(self, image):
        """Standard CLIP processing (without mask)."""
        print(f"📋 Using standard CLIP (without region-aware path)")
        feature = self._extract_clip_features(image)
        # Keep output dtype consistent with input.
        return feature.to(dtype=image.dtype)
    def _apply_mask_to_image(self, image, mask, bg_mode="blur", return_mask=False):
        """
        Apply binarized mask to image and optionally return aligned 2D mask.
        - image: [3, H, W] tensor (已是 processor 输出尺寸)
        - mask:  [1,H,W] or [H,W] tensor / PIL / np
        - return_mask: 若为 True，额外返回对齐后的 [H,W] mask（float, {0,1}）
        """
        # 1) Normalize mask to 2D float tensor.
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 3 and mask.size(0) == 1:
                mask_2d = mask.squeeze(0)
            elif mask.dim() == 2:
                mask_2d = mask
            else:
                raise ValueError(f"Unexpected mask dimensions: {mask.shape}")
            mask_2d = mask_2d.to(device=image.device, dtype=image.dtype)
        else:
            from PIL import Image
            import numpy as np
            if isinstance(mask, Image.Image):
                mask_2d = torch.from_numpy(np.array(mask.convert('L'))).to(image.device).to(image.dtype)
            elif isinstance(mask, np.ndarray):
                mask_2d = torch.from_numpy(mask).to(image.device).to(image.dtype)
            else:
                raise ValueError(f"Unsupported mask type: {type(mask)}")

        # 2) Normalize from [0,255] if needed, then binarize.
        if mask_2d.max() > 1.0:
            mask_2d = mask_2d / 255.0
        mask_2d = (mask_2d > 0.5).float()  # {0,1}

        # 3) Resize alignment (nearest to avoid gray boundaries).
        H, W = image.shape[-2:]
        if mask_2d.shape != (H, W):
            mask_2d = F.interpolate(mask_2d.unsqueeze(0).unsqueeze(0),
                                    size=(H, W), mode='nearest').squeeze(0).squeeze(0)

        # 4) Apply to image (blur/mean background is usually more natural).
        mask_3d = mask_2d.unsqueeze(0).expand_as(image)
        if bg_mode == "blur":
            blurred = F.avg_pool2d(image, kernel_size=9, stride=1, padding=4)
            masked_image = mask_3d * image + (1 - mask_3d) * blurred
        elif bg_mode == "mean":
            mean_color = image.mean(dim=(1, 2), keepdim=True)
            masked_image = mask_3d * image + (1 - mask_3d) * mean_color
        else:
            masked_image = mask_3d * image  # less recommended: hard zeroing may introduce artifacts

        return (masked_image, mask_2d) if return_mask else masked_image


    def _downsample_mask_to_patches(self, mask_2d: torch.Tensor) -> torch.Tensor:
        """
        Downsample [H,W] binary/soft mask to ViT patch grid and return [P] in [0,1].
        Assumes input image has been preprocessed to config.image_size.
        """
        ps = self.config.patch_size            # e.g., 14
        # Use patch-average as weight (can be switched to max for conservative behavior).
        mask_patch = F.avg_pool2d(mask_2d.unsqueeze(0).unsqueeze(0),
                                kernel_size=ps, stride=ps)
        return mask_patch.flatten()  # [P]

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
