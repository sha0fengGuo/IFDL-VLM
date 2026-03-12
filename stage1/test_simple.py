import os
import sys
import argparse
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from model.simple import SimpleSidaModel
from utils.SID_Set import SimpleSidaDataset, simple_collate_fn
from utils.utils import AverageMeter, Summary, dict_to_cuda, intersectionAndUnionGPU
from torch.utils.tensorboard import SummaryWriter

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--precision", default="fp16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log_dir", default="./runs/best_exp", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model and configuration
    from model.simple_config import SimpleSidaConfig
    config = SimpleSidaConfig(
        vision_pretrained=None, 
        vision_tower=args.vision_tower,
        image_size=args.image_size,
    )
    model = SimpleSidaModel(config)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    torch_dtype = torch.float16 if args.precision == "fp16" else (
        torch.bfloat16 if args.precision == "bf16" else torch.float32
    )
    model = model.to(dtype=torch_dtype, device=device)
    model.eval()

    # Dataset
    test_dataset = SimpleSidaDataset(
        base_image_dir=args.dataset_dir,
        vision_tower=args.vision_tower,
        split="test",
        image_size=args.image_size,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        collate_fn=simple_collate_fn
    )
    print(f"Loaded test set size: {len(test_dataset)}")

    # Evaluation
    test_acc, test_giou, test_ciou, per_class_metrics = test(
        test_loader, model, writer, args
    )

    print("\nTest Finished.")
    print(f"Accuracy: {test_acc:.4f}%  gIoU: {test_giou:.4f}  cIoU: {test_ciou:.4f}")
    print(f"Per-class metrics:")
    for k, v in per_class_metrics.items():
        print(f"{k}: {v}")

import random

def test(test_loader, model_engine, writer, args, sample_ratio=None):
    model_engine.eval()
    correct = 0
    total = 0
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device='cuda')
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # Calculate total number of batches and samples to use
    total_batches = len(test_loader)
    if sample_ratio is not None:
        num_batches = max(1, int(total_batches * sample_ratio))
        # Generate random indices for sampling
        sample_indices = set(random.sample(range(total_batches), num_batches))
        print(f"\ntest on {num_batches}/{total_batches} randomly sampled batches...")

    for batch_idx, input_dict in enumerate(tqdm.tqdm(test_loader)):
        # Skip batches not in our sample if sampling is enabled
        if sample_ratio is not None and batch_idx not in sample_indices:
            continue
        if batch_idx == 0:
            print("\nFirst test batch details:")
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"{key} length: {len(value)}")

        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        # Debug first processed batch
        if total == 0:
            print("\nProcessing first batch:")
            print("Input dict keys:", input_dict.keys())

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        input_dict['inference'] = True
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # Get predictions
        logits = output_dict["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        cls_labels = input_dict["cls_labels"]
        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)

        for t, p in zip(cls_labels, preds):
            confusion_matrix[t.long(), p.long()] += 1
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0])


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if len(acc_iou_meter.avg) > 1 else 0.0

    accuracy = correct / total * 100.0
    confusion_matrix = confusion_matrix.cpu()
    class_names = ['Real', 'Full Synthetic', 'Tampered']
    per_class_metrics = {}
    for i in range(num_classes):
        tp = confusion_matrix[i, i]  
        fp = confusion_matrix[:, i].sum() - tp  
        fn = confusion_matrix[i, :].sum() - tp 
        tn = confusion_matrix.sum() - (tp + fp + fn)  

        total_class_samples = confusion_matrix[i, :].sum()

        class_accuracy = float(tp / total_class_samples) if total_class_samples > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[class_names[i]] = {
            'accuracy': class_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    pixel_correct = intersection_meter.sum[1]  
    pixel_total = union_meter.sum[1]  
    pixel_accuracy = pixel_correct / (pixel_total + 1e-10) * 100.0

    iou = ciou  
    f1_score = 2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10) if (iou + accuracy / 100) > 0 else 0.0

    avg_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])

    auc_approx = avg_precision * avg_recall

    
    writer.add_scalar("test/accuracy", accuracy)
    writer.add_scalar("test/giou", giou)
    writer.add_scalar("test/ciou", ciou)
    writer.add_scalar("test/pixel_accuracy", pixel_accuracy)
    writer.add_scalar("test/iou", iou)
    writer.add_scalar("test/f1_score", f1_score)
    writer.add_scalar("test/auc_approx", auc_approx)
    for class_name, metrics in per_class_metrics.items():
        for metric_name, value in metrics.items():
            writer.add_scalar(f"test/{class_name.lower().replace('/', '_')}_{metric_name}", value)

    test_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100}%)"
    print(f"\n{test_type} test Results:")
    print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")
    print(f"Classification Accuracy: {accuracy:.4f}%")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
    print(f"IoU: {iou:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Approximate AUC: {auc_approx:.4f}")
    print(f"Total correct classifications: {correct}")
    print(f"Total classification samples: {total}")
    print("\nPer-Class Metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"\n{class_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print("Predicted ")
    print("Actual ")
    print(f"{'':20}", end="")  
    for name in class_names:
        print(f"{name:>12}", end="") 
    print()  

    for i, class_name in enumerate(class_names):
        print(f"{class_name:20}", end="") 
        for j in range(num_classes):
            print(f"{confusion_matrix[i, j]:12.0f}", end="")
        print()      

    return accuracy, giou, ciou, per_class_metrics



if __name__ == "__main__":
    main(sys.argv[1:])
