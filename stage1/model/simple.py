import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
from .segment_anything import build_sam_vit_h
from typing import List, Optional, Tuple

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

class SimpleSidaModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_hidden_size = 1024
        self.visual_model = build_sam_vit_h(config.vision_pretrained)
        self.image_size = 1024
        
    # Freeze all SAM parameters
        for param in self.visual_model.parameters():
            param.requires_grad = False
    # Train only the mask decoder if requested
        if getattr(config, "train_mask_decoder", False):
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        
        self.cls_projection = nn.Sequential(
            nn.Linear(self.clip_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    # Projection for patch tokens used in segmentation
        self.seg_projection = nn.Sequential(
            nn.Linear(self.clip_hidden_size, 256),
            nn.ReLU()
        )
        self.sida_fc1 = nn.Linear(3, 256)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        self.cls_loss_weight = getattr(config, "cls_loss_weight", 1.0)
        self.mask_loss_weight = getattr(config, "mask_loss_weight", 1.0)

    def get_visual_embs(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def model_forward(
        self,
        images: torch.FloatTensor,         # [B, 3, 1024, 1024] for SAM
        images_clip: torch.FloatTensor,    # [B, 3, 224, 224] for CLIP
        input_ids=None,
        cls_labels: torch.LongTensor = None,
        labels=None,
        attention_masks=None,
        offset=None,
        masks_list: List[torch.FloatTensor] = None,
        cls_labels_list=None,
        label_list=None,
        resize_list=None,
        inference: bool = False,
        **kwargs,
    ):
    # 1. CLIP feature extraction
        clip_outputs = self.clip_model(images_clip, output_hidden_states=True)
        last_hidden = clip_outputs.last_hidden_state  # [B, num_tokens, 1024]
        cls_token = last_hidden[:, 0]  # [B, 1024]
        patch_tokens = last_hidden[:, 1:]  # [B, num_patches, 1024]

    # 2. Classification prediction
        cls_logits = self.cls_projection(cls_token)  # [B, 3]

        outputs = {
            "logits": cls_logits
        }

        cls_loss = None
        mask_loss = None

        if cls_labels is not None:
            cls_loss = F.cross_entropy(cls_logits, cls_labels)
            outputs["cls_loss"] = cls_loss

    # 3. Generate segmentation masks (for tampered samples only)

        if (cls_labels is not None) and (masks_list is not None) and (cls_labels == 2).any():
            tampered_indices = (cls_labels == 2)
            if tampered_indices.any():
                image_embeddings = self.get_visual_embs(images[tampered_indices])
                seg_tokens = patch_tokens[tampered_indices]  # [N_t, num_patches, 1024]
                seg_features = self.seg_projection(seg_tokens)  # [N_t, num_patches, 256]
                cls_projected = self.sida_fc1(cls_logits[tampered_indices])  # [N_t, 256]
                enhanced_pred_embeddings = []
                for i in range(seg_features.shape[0]):
                    query = cls_projected[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
                    key = seg_features[i].unsqueeze(0)  # [1, num_patches, 256]
                    value = seg_features[i].unsqueeze(0)  # [1, num_patches, 256]
                    attn_output, _ = self.attention_layer(query=query, key=key, value=value)
                    enhanced = seg_features[i] + attn_output.squeeze(0).expand_as(seg_features[i])
                    enhanced_pred_embeddings.append(enhanced.mean(dim=0))  # [256]
                enhanced_pred_embeddings = torch.stack(enhanced_pred_embeddings, dim=0)  # [N_t, 256]
                pred_masks = []
                for i in range(enhanced_pred_embeddings.shape[0]):
                    sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=enhanced_pred_embeddings[i].unsqueeze(0).unsqueeze(0),
                    )
                    sparse_embeddings = sparse_embeddings.to(enhanced_pred_embeddings[i].dtype)
                    low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_mask = self.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=label_list[i].shape,
                    )
                    pred_masks.append(pred_mask[:, 0])
                # pred_masks = torch.stack(pred_masks, dim=0)  # [N_t, 1, H, W]
                outputs["pred_masks"] = pred_masks
                gt_masks = masks_list
                gt_masks = [masks_list[i] for i in torch.where(tampered_indices)[0]]

                if inference:
                    return {
                        "pred_masks": pred_masks,
                        "gt_masks": gt_masks,
                        "logits": cls_logits,
                    }

                mask_bce_loss = 0
                mask_dice_loss = 0
                num_masks = 0
                for pred_mask, gt_mask in zip(pred_masks, [masks_list[i] for i in torch.where(tampered_indices)[0]]):
                    assert pred_mask.shape == gt_mask.shape, f"pred_mask.shape: {pred_mask.shape}, gt_mask.shape: {gt_mask.shape}"
                    mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=1)
                    mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=1)
                    num_masks += 1
                if num_masks > 0:
                    mask_bce_loss = mask_bce_loss / num_masks
                    mask_dice_loss = mask_dice_loss / num_masks
                mask_loss = mask_bce_loss + mask_dice_loss

                outputs["mask_bce_loss"] = mask_bce_loss
                outputs["mask_dice_loss"] = mask_dice_loss
                outputs["mask_loss"] = mask_loss
            else:
                outputs["mask_bce_loss"] = torch.tensor(0.0, device=cls_logits.device)
                outputs["mask_dice_loss"] = torch.tensor(0.0, device=cls_logits.device)
                outputs["mask_loss"] = torch.tensor(0.0, device=cls_logits.device)
        else:
            outputs["mask_bce_loss"] = torch.tensor(0.0, device=cls_logits.device)
            outputs["mask_dice_loss"] = torch.tensor(0.0, device=cls_logits.device)
            outputs["mask_loss"] = torch.tensor(0.0, device=cls_logits.device)



            # Total loss
        if (cls_loss is not None) and (mask_loss is not None):
            outputs["loss"] = self.cls_loss_weight * cls_loss + self.mask_loss_weight * mask_loss
        elif cls_loss is not None:
            outputs["loss"] = cls_loss

        return outputs

    def forward(self, **kwargs):
        return self.model_forward(**kwargs)
    

    def evaluate(
        self,
        images_clip,
        images,
        masks_list=None,
        resize_list=None,
        label_list=None,
        inference=True,
        **kwargs,
    ):
        self.eval()
        with torch.no_grad():
            # 1. Classification prediction
            clip_outputs = self.clip_model(images_clip, output_hidden_states=True)
            last_hidden = clip_outputs.last_hidden_state  # [B, num_tokens, 1024]
            cls_token = last_hidden[:, 0]  # [B, 1024]
            patch_tokens = last_hidden[:, 1:]  # [B, num_patches, 1024]
            cls_logits = self.cls_projection(cls_token)  # [B, 3]
            preds = torch.argmax(torch.softmax(cls_logits, dim=1), dim=1)  # [B]

            outputs = {
                "logits": cls_logits,
                "pred_class": preds,
            }

            # 2. Only segment samples predicted as tampered
            tampered_indices = (preds == 2)
            if tampered_indices.any():
                # Select only the samples predicted as tampered
                image_embeddings = self.get_visual_embs(images[tampered_indices])
                seg_tokens = patch_tokens[tampered_indices]  # [N_t, num_patches, 1024]
                seg_features = self.seg_projection(seg_tokens)  # [N_t, num_patches, 256]
                cls_projected = self.sida_fc1(cls_logits[tampered_indices])  # [N_t, 256]
                enhanced_pred_embeddings = []
                for i in range(seg_features.shape[0]):
                    query = cls_projected[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
                    key = seg_features[i].unsqueeze(0)  # [1, num_patches, 256]
                    value = seg_features[i].unsqueeze(0)  # [1, num_patches, 256]
                    attn_output, _ = self.attention_layer(query=query, key=key, value=value)
                    enhanced = seg_features[i] + attn_output.squeeze(0).expand_as(seg_features[i])
                    enhanced_pred_embeddings.append(enhanced.mean(dim=0))  # [256]
                enhanced_pred_embeddings = torch.stack(enhanced_pred_embeddings, dim=0)  # [N_t, 256]
                pred_masks = []
                for idx in range(enhanced_pred_embeddings.shape[0]):
                    sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=enhanced_pred_embeddings[idx].unsqueeze(0).unsqueeze(0),
                    )
                    sparse_embeddings = sparse_embeddings.to(enhanced_pred_embeddings[idx].dtype)
                    low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[idx].unsqueeze(0),
                        image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    # Use original batch indices to fetch resize/label_list
                    orig_idx = tampered_indices.nonzero(as_tuple=True)[0][idx].item()
                    pred_mask = self.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[orig_idx],
                        original_size=label_list[orig_idx].shape if label_list is not None else None,
                    )
                    pred_masks.append(pred_mask[:, 0])
                outputs["pred_masks"] = pred_masks
                # Optional: return gt_masks only for tampered samples
                if masks_list is not None:
                    outputs["gt_masks"] = [masks_list[i] for i in tampered_indices.nonzero(as_tuple=True)[0]]
            else:
                outputs["pred_masks"] = []
                if masks_list is not None:
                    outputs["gt_masks"] = []

            return outputs