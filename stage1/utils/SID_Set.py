import glob
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide

def simple_collate_fn(batch):
    """
    Collate function for SimpleSidaDataset.
    Returns:
        {
            "image_paths": [...],
            "images": torch.stack([...]),         # [B, 3, 1024, 1024]
            "images_clip": torch.stack([...]),    # [B, 3, 224, 224]
            "masks_list": [...],                  # list of [1, H, W]
            "cls_labels": torch.tensor([...]),    # [B]
            "resize_list": [...],                 # list of (H, W)
        }
    """
    image_path_list = []
    images_list = []
    images_clip_list = []
    masks_list = []
    cls_labels_list = []
    resize_list = []
    label_list = []
    inferences = []

    for (
        image_path,
        image,
        image_clip,
        mask,
        cls_label,
        resize,
        label,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(image)
        images_clip_list.append(image_clip)
        masks_list.append(mask.float())
        cls_labels_list.append(cls_label)
        resize_list.append(resize)
        label_list.append(label)
        inferences.append(inference)

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "masks_list": masks_list,  # list of [1, H, W]
        "cls_labels": torch.tensor(cls_labels_list, dtype=torch.long),
        "resize_list": resize_list,
        "label_list": label_list,  # list of [H, W] labels
        "inference": inferences[0]
    }



class SimpleSidaDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        split="train",
        image_size=224,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.split = split
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        split_dir = os.path.join(base_image_dir, split)
        real_images = glob.glob(os.path.join(split_dir, "real", "*.jpg"))
        full_syn_images = glob.glob(os.path.join(split_dir, "full_synthetic", "*.png"))
        tampered_images = glob.glob(os.path.join(split_dir, "tampered", "*.png"))
        valid_tampered_images = []
        for img_path in tampered_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = f"{base_name}_mask.png"
            mask_path = os.path.join(split_dir, "masks", mask_name)
            if os.path.exists(mask_path):
                valid_tampered_images.append(img_path)
        self.images = real_images + full_syn_images + valid_tampered_images
        self.cls_labels = [0]*len(real_images) + [1]*len(full_syn_images) + [2]*len(valid_tampered_images)

        print(f"\nDataset Statistics for {split} split:")
        print(f"Real images: {len(real_images)}")
        print(f"Full synthetic images: {len(full_syn_images)}")
        print(f"Tampered images: {len(valid_tampered_images)} (Valid) / {len(tampered_images)} (Total)")

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        #debug
        x = x.float()
        assert not torch.any(self.pixel_std == 0), "pixel_std contains zero!"
        #debug
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        #debug
        assert not torch.isnan(x).any(), "Image has nan after preprocess"
        assert not torch.isinf(x).any(), "Image has inf after preprocess"
        #debug
        return x

    def __getitem__(self, idx):
        image_path = self.images[idx]
        cls_label = self.cls_labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        #debug
        assert not torch.isnan(image).any(), f"Image has nan: {image_path}"
        assert not torch.isinf(image).any(), f"Image has inf: {image_path}"
        assert image.max() <= 10 and image.min() >= -10, f"Image value out of range: {image_path}"
        #debug
 
        mask = torch.zeros((1, resize[0], resize[1]))
        if cls_label == 2:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_name = f"{base_name}_mask.png"
            mask_path = os.path.join(self.base_image_dir, self.split, "masks", mask_name)
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                assert mask_img is not None, f"Failed to read mask: {mask_path}"
                mask_img = self.transform.apply_image(mask_img)
                mask_img = mask_img / 255.0
                mask = torch.from_numpy(mask_img).unsqueeze(0)
                #debug
                assert not torch.isnan(mask).any(), f"Mask has nan: {mask_path}"
                assert not torch.isinf(mask).any(), f"Mask has inf: {mask_path}"
                assert mask.max() <= 1.0 and mask.min() >= 0.0, f"Mask value out of range: {mask_path}"
                #debug       
        labels = torch.ones(mask.shape[1], mask.shape[2]) * 255  # ignore all pixels

        return image_path,image, image_clip, mask, cls_label, resize, labels,None

    def __len__(self):
        return len(self.images)
    


