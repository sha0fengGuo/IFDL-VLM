#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /data/guoshaofeng/LLaVA/inference.py

import torch
import requests
from PIL import Image
from io import BytesIO
import json
import argparse
import os
import sys
from pathlib import Path

# Add current stage2 root to Python path for local imports.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
import numpy as np


FORENSIC_TRAIN_PROMPT = (
    "Analyze this image for potential digital tampering or manipulation. "
    "As a forensic image analyst, examine the image carefully for any signs of:\n\n"
    "1. Object Tampering: Artificially inserted or replaced objects\n"
    "2. Partial Tampering: Modified portions of existing objects\n\n"
    "For any suspicious regions you identify, provide detailed analysis including:\n\n"
    "Location Description:\n"
    "- Describe the location using natural language\n"
    "- Provide both relative position (in relation to other objects) and absolute position (within the image frame)\n\n"
    "Content Analysis:\n"
    "- Identify what appears to be tampered\n"
    "- Describe characteristics of the suspicious area (size, color, texture, etc.)\n\n"
    "Evidence of Manipulation:\n"
    "- Lighting and shadow inconsistencies\n"
    "- Edge artifacts and texture mismatches\n"
    "- Perspective or proportion issues\n"
    "- Resolution or quality differences\n"
    "- Contextual abnormalities\n"
    "- Missing or incorrect reflections/shadows\n\n"
    "Provide your analysis in a structured format, starting with the tampering type, "
    "then location, content, and visual inconsistencies.If the highlighted region does not contain convincing evidence of tampering, explicitly state that no clear manipulation evidence is observed and briefly explain why. Avoid speculating about non-existent forensic artifacts."
)



def expand2square(pil_img, background_color):
    """Image preprocessing aligned with training behavior."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(image_path):
    """Load an RGB image from a local path or URL."""
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image


def load_mask(mask_path):
    """Load and lightly preprocess a mask image."""
    if not os.path.exists(mask_path):
        print(f"⚠️ Mask file not found: {mask_path}")
        return None
    
    mask = Image.open(mask_path).convert('L')
    print(f"🎭 Loaded mask: {mask.size}")
    return mask


def prepare_image_and_mask(image_path, mask_path=None, image_processor=None, device='cuda'):
    """Prepare image and mask tensors for inference."""
    # Load primary image.
    image = load_image(image_path)
    print(f"📷 Original image size: {image.size}")
    
    # Process image using standard process_images.
    images = process_images([image], image_processor, None)
    if isinstance(images, list):
        images = torch.stack(images)
    
    print(f"📷 Processed image tensor shape: {images.shape}")
    
    # Process mask with a simple and robust path.
    alpha_mask_tensor = None
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')
        # Ensure mask size matches image size.
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
        # Convert to tensor and normalize to [0, 1].
        mask_array = np.array(mask)
        alpha_mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0) / 255.0

        mask_mean = alpha_mask_tensor.mean().item()
        mask_active_ratio = (alpha_mask_tensor > 0.5).float().mean().item()
        print(f"🎭 Using custom mask: {alpha_mask_tensor.shape}, white ratio: {mask_mean:.3f}, active ratio(>0.5): {mask_active_ratio:.5f}")
        if mask_active_ratio < 1e-4:
            print("⚠️ Mask is nearly all black: region branch input is close to empty, output may bias toward global semantics.")
        elif mask_active_ratio > 0.9999:
            print("ℹ️ Mask is nearly all white: region branch is close to full-image inference.")
    else:
        # Create a full-white mask for full-image focus.
        alpha_mask_tensor = torch.ones(1, 1, 336, 336)
        print(f"🎭 Using default full-image mask: {alpha_mask_tensor.shape}")
    
    return images, alpha_mask_tensor, image


def load_trained_model(model_path, device='cuda'):
    """Load a trained model with LoRA-first fallback strategy."""
    print(f"🚀 Start loading trained model...")
    print(f"   Model path: {model_path}")
    
    # Disable torch default init for faster loading.
    disable_torch_init()
    
    try:
        # Try LoRA loading flow from model_load.py first.
        import model_load
        print("🔄 Loading as LoRA model...")
        tokenizer, model, image_processor = model_load.load_lora_model(model_path)
        
        # Move model to expected device and precision.
        if torch.cuda.is_available() and device == 'cuda':
            model = model.to('cuda', dtype=torch.float16)
            print("✅ Model moved to GPU with FP16")
        else:
            model = model.to('cpu', dtype=torch.float32)
            print("✅ Model running on CPU with FP32")
            
        print("✅ LoRA model loaded")
        return tokenizer, model, image_processor
        
    except Exception as e:
        print(f"❌ LoRA load failed: {e}")
        print("🔄 Falling back to standard pretrained loading...")
        
        # Fallback to standard loading.
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            load_8bit=False,
            load_4bit=False,
            device_map="auto"
        )
        print("✅ Standard model loaded")
        return tokenizer, model, image_processor


def inference_single(args):
    """Single-image inference."""
    # Load model.
    tokenizer, model, image_processor = load_trained_model(args.model_path, args.device)
    
    # Check model type and support.
    print(f"\n🔍 Model inspection:")
    print(f"🔧 Outer model type: {type(model)}")
    
    # Unwrap to the actual LLaVA model when wrapped.
    actual_model = model
    if hasattr(model, 'base_model'):
        actual_model = model.base_model
        if hasattr(actual_model, 'model'):
            actual_model = actual_model.model
    
    # Check region-aware support.
    vision_tower = actual_model.get_vision_tower()
    if hasattr(vision_tower, 'enable_region_aware'):
        print(f"✅ Region-aware enabled: {vision_tower.enable_region_aware}")
        print(f"✅ Region weight: {vision_tower.region_weight}")
        
        # Set region weight if provided.
        if hasattr(args, 'region_weight'):
            vision_tower.region_weight = args.region_weight
            print(f"🔧 Set region weight: {args.region_weight}")
    else:
        print("❌ Model does not support region-aware encoding")
    
    # Prepare image and mask.
    images, alpha_masks, original_image = prepare_image_and_mask(
        args.image_path, args.mask_path, image_processor, args.device
    )
    
    # Ensure all tensors match model device and dtype.
    model_device = model.device
    model_dtype = next(model.parameters()).dtype
    
    images = images.to(model_device, dtype=model_dtype)
    alpha_masks = alpha_masks.to(model_device, dtype=model_dtype)
    
    print(f"\n📊 Input preparation:")
    print(f"🔧 Model device: {model_device}, dtype: {model_dtype}")
    print(f"🔧 Image shape: {images.shape}, dtype: {images.dtype}")
    print(f"🔧 Mask shape: {alpha_masks.shape}, dtype: {alpha_masks.dtype}")
    
    # Prepare conversation template.
    conv_mode = getattr(args, 'conv_mode', "llava_v1")
    conv = conv_templates[conv_mode].copy()
    
    # Build text prompt.
    question = args.query
    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
        
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"🗣️ Prompt: {prompt}")
    
    # Tokenize.
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    input_ids = input_ids.to(model_device)
    
    print(f"🔢 Input ID shape: {input_ids.shape}")
    
    # Run generation.
    print(f"\n🔮 Starting inference...")
    
    # Check whether forward supports alpha_masks.
    import inspect
    actual_forward_sig = inspect.signature(actual_model.forward)
    supports_alpha_masks = 'alpha_masks' in actual_forward_sig.parameters
    
    print(f"🔍 Model supports alpha_masks: {supports_alpha_masks}")
    
    try:
        with torch.no_grad():
            if supports_alpha_masks:
                # Use region-aware inference path.
                print("✅ Using region-aware inference")
                output_ids = model.generate(
                    inputs=input_ids,
                    images=images,
                    alpha_masks=alpha_masks,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=getattr(args, 'top_p', 0.7),
                    num_beams=getattr(args, 'num_beams', 1),
                    max_new_tokens=getattr(args, 'max_new_tokens', 512),
                    use_cache=True
                )
            else:
                # Use standard inference path.
                print("⚠️ Using standard inference (no region-aware branch)")
                output_ids = model.generate(
                    inputs=input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=getattr(args, 'max_new_tokens', 512),
                    use_cache=True
                )
    
        # Decode output.
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Strip prompt from decoded output when possible.
        try:
            if conv.sep_style == SeparatorStyle.TWO:
                outputs = outputs.split(conv.sep2)[-1].strip()
            else:
                outputs = outputs.split(conv.roles[1])[-1].strip()
                if outputs.startswith(":"):
                    outputs = outputs[1:].strip()
        except:
            pass
        
        print(f"\n{'='*50}")
        print(f"🎯 Inference result:")
        print(f"{'='*50}")
        print(f"📷 Image: {args.image_path}")
        print(f"🎭 Mask: {args.mask_path if args.mask_path else 'None'}")
        print(f"❓ Query: {args.query}")
        print(f"💬 Answer: {outputs}")
        print(f"{'='*50}")
        
        return outputs
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def inference_batch(args):
    """Batch inference."""
    # Load test metadata.
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"📋 Loaded test samples: {len(test_data)}")
    
    results = []
    
    for i, item in enumerate(test_data[:args.max_samples]):
        print(f"\n🔄 Processing sample {i+1}/{min(len(test_data), args.max_samples)}...")
        
        # Build image and mask paths.
        image_path = os.path.join(args.image_folder, item['image'])
        mask_path = os.path.join(args.image_folder, item.get('mask', '')) if item.get('mask') else None
        
        # Get query: default to JSON human prompt, or force --query with --use-arg-query.
        query = None
        if getattr(args, 'use_arg_query', False):
            query = args.query
        else:
            for conv in item.get('conversations', []):
                if conv.get('from') == 'human':
                    query = conv.get('value', '').replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    break

            # Fallback: if conversations are missing/empty, use CLI query.
            if not query:
                query = args.query
        
        if not query:
            continue
            
        # Populate args for single-sample inference.
        args.image_path = image_path
        args.mask_path = mask_path
        args.query = query
        
        try:
            output = inference_single(args)
            if output:
                results.append({
                    'image': item['image'],
                    'mask': item.get('mask'),
                    'query': query,
                    'prediction': output,
                    'ground_truth': item['conversations']
                })
        except Exception as e:
            print(f"❌ Sample failed: {e}")
            continue
    
    # Save outputs.
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {args.output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments.
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--region-weight", type=float, default=0.5,
                       help="Region weight (should match training)")
    parser.add_argument("--enable-region-aware", action="store_true", default=True,
                       help="Enable region-aware path")
    
    # Input arguments.
    parser.add_argument("--image-path", type=str, 
                       help="Image path (single-image mode)")
    parser.add_argument("--mask-path", type=str,
                       help="Mask path (optional)")
    parser.add_argument("--query", type=str, default=FORENSIC_TRAIN_PROMPT,
                       help="Prompt query")
    parser.add_argument(
        "--use-arg-query",
        action="store_true",
        help="Batch mode: ignore JSON conversations and use --query for every sample.",
    )
    
    # Batch inference arguments.
    parser.add_argument("--test-data-path", type=str,
                       help="Test metadata path (JSON)")
    parser.add_argument("--image-folder", type=str,
                       help="Image folder path")
    parser.add_argument("--output-path", type=str,
                       help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum number of samples")
    
    # Generation arguments.
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    
    # Device argument.
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Choose single-image or batch mode.
    if args.image_path:
        inference_single(args)
    elif args.test_data_path:
        inference_batch(args)
    else:
        print("Please provide --image-path or --test-data-path")


if __name__ == "__main__":
    main()