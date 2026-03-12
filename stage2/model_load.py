import torch
import os
from transformers import AutoTokenizer
from peft import PeftModel
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import ModelArguments
import dataclasses
import typing
def create_model_args():
    """Create model arguments aligned with training config and backward compatibility."""
    @dataclasses.dataclass
    class Args:
        # vision / clip
        vision_tower: str = "openai/clip-vit-large-336"
        mm_vision_tower: str = "openai/clip-vit-large-336"  # compatible alias

        # mm projector / vision selection
        mm_projector_type: str = "mlp2x_gelu"
        mm_hidden_size: int = 1024
        mm_vision_select_layer: int = -2
        mm_vision_select_feature: str = "patch"
        mm_patch_merge_type: typing.Optional[str] = None

        # image token handling
        mm_use_im_start_end: bool = False
        mm_use_im_patch_token: bool = False
        image_aspect_ratio: str = "pad"

        # Pretrained adapter path (also passed by training scripts)
        pretrain_mm_mlp_adapter: str = "./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin"

        # region aware
        enable_region_aware: bool = True
        region_weight: float = 1.0

        # other flags often referenced
        unfreeze_mm_vision_tower: bool = False
        s2: bool = False

    return Args()

def load_lora_model(model_path: str):
    """LoRA model loading flow with explicit debug logs and deduplicated steps."""
    import sys
    print("🚀 Start loading LoRA model...", flush=True)

    BASE_MODEL_PATH = "liuhaotian/llava-v1.5-13b"
    LORA_CHECKPOINT_PATH = model_path
    NON_LORA_WEIGHTS_PATH = f"{LORA_CHECKPOINT_PATH}/non_lora_trainables.bin"

    # 1. Build model args.
    model_args = create_model_args()

    # 2. Load base LLaVA model (without LoRA).
    print(f"📦 Calling LlavaLlamaForCausalLM.from_pretrained({BASE_MODEL_PATH})", flush=True)
    try:
        # Load on CPU first to avoid potential long auto device_map stalls.
        model = LlavaLlamaForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        print("✅ Base model from_pretrained returned", flush=True)
    except Exception as e:
        print("❌ Failed to load base pretrained model:", e, flush=True)
        import traceback; traceback.print_exc()
        raise

    # 3. Initialize vision modules (single call).
    print("🎯 Calling initialize_vision_modules (delay_load supported)", flush=True)
    try:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
        print("✅ initialize_vision_modules returned", flush=True)
    except Exception as e:
        print("❌ initialize_vision_modules raised an exception:", flush=True)
        import traceback; traceback.print_exc()
        raise

    # Check vision_tower state and load explicitly if needed.
    vision_tower = None
    try:
        vision_tower = model.get_vision_tower()
        print(f"🔍 get_vision_tower() -> {type(vision_tower)}", flush=True)
        print(f"🔍 vision_tower.is_loaded: {getattr(vision_tower, 'is_loaded', 'N/A')}", flush=True)
        if not getattr(vision_tower, 'is_loaded', False):
            print("⚠️ vision_tower not loaded, calling vision_tower.load_model()", flush=True)
            try:
                vision_tower.load_model()
                print("✅ vision_tower.load_model() completed", flush=True)
            except Exception as e:
                print("❌ vision_tower.load_model() raised an exception:", flush=True)
                import traceback; traceback.print_exc()
                raise
    except Exception as e:
        print("❌ Error while checking vision_tower:", e, flush=True)
        import traceback; traceback.print_exc()
        raise

    # Acquire image processor.
    image_processor = getattr(vision_tower, 'image_processor', None)
    if image_processor is None:
        print("❌ image_processor is None (load may have failed or delay_load is incomplete)", flush=True)
        raise RuntimeError("image_processor is not initialized")

    print("📝 Loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    print("✅ Tokenizer loaded", flush=True)

    # 5. Load LoRA adapter.
    if os.path.exists(LORA_CHECKPOINT_PATH):
        print(f"🔧 Loading LoRA weights from: {LORA_CHECKPOINT_PATH}", flush=True)
        try:
            model = PeftModel.from_pretrained(
                model,
                LORA_CHECKPOINT_PATH,
                device_map=None,
                torch_dtype=torch.float16
            )
            print("✅ LoRA loaded successfully (on CPU)", flush=True)
        except Exception as e:
            print("❌ LoRA load failed:", e, flush=True)
            import traceback; traceback.print_exc()
            raise
    else:
        print("⚠️ LoRA checkpoint not found, skip LoRA loading", flush=True)

    # 6. Force using pretrained mm_projector (instead of non_lora_trainables.bin).
    # Prefer PRETRAINED_PROJECTOR_PATH; otherwise resolve from local checkpoints directory.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_projector = os.path.join(
        repo_root, "checkpoints", "llava-v1.5-13b-pretrain", "mm_projector.bin"
    )
    PRETRAINED_PROJECTOR_PATH = os.environ.get("PRETRAINED_PROJECTOR_PATH", default_projector)
    if not os.path.exists(PRETRAINED_PROJECTOR_PATH):
        raise FileNotFoundError(
            f"Pretrained mm_projector.bin not found at: {PRETRAINED_PROJECTOR_PATH}. "
            "Set PRETRAINED_PROJECTOR_PATH to override."
        )

    print(f"🎯 Using pretrained mm_projector: {PRETRAINED_PROJECTOR_PATH}", flush=True)
    try:
        sd_pre = torch.load(PRETRAINED_PROJECTOR_PATH, map_location="cpu")
        # Keep only projector-related weights.
        proj_only = {k: v for k, v in sd_pre.items() if k.startswith("model.mm_projector.")}

        # Rename into LoRA-wrapped namespace.
        renamed = {}
        for k, v in proj_only.items():
            new_k = k.replace("model.", "base_model.model.model.", 1)
            renamed[new_k] = v

        # Load only matching keys to avoid conflicts.
        model_state = model.state_dict()
        to_load = {k: v for k, v in renamed.items() if k in model_state and model_state[k].shape == v.shape}

        missing_keys = [k for k in renamed.keys() if k not in to_load]
        if missing_keys:
            print(f"⚠️ {len(missing_keys)} projector keys did not match current model and will be skipped", flush=True)

        model_state.update(to_load)
        model.load_state_dict(model_state, strict=False)
        print(f"✅ Pretrained projector loaded (updated {len(to_load)} tensors)", flush=True)

    except Exception as e:
        print("❌ Failed to load pretrained projector:", e, flush=True)
        import traceback; traceback.print_exc()
        raise


    if torch.cuda.is_available():
        print("🚀 Moving model to GPU...")
        model = model.to('cuda', dtype=torch.float16)
        print("✅ Model moved to GPU with FP16")
    else:
        # Ensure FP32 on CPU to avoid half-precision issues.
        model = model.to('cpu', dtype=torch.float32)
        print("⚠️ Using CPU with FP32")
    
    print("🎉 Model loading completed, switching to eval()")
    model.eval()
    
    return tokenizer, model, image_processor


def compare_mm_projector_with_pretrain(pretrain_path: str, model, tol: float = 1e-6):
    """Compare mm_projector weights between current model and pretrained checkpoint."""
    import torch
    from collections import defaultdict

    if not os.path.exists(pretrain_path):
        print(f"❌ pretrain file not found: {pretrain_path}")
        return

    pre = torch.load(pretrain_path, map_location='cpu')
    # Keep only mm_projector-related keys.
    pre_mm = {k: v for k, v in pre.items() if 'mm_projector' in k}

    # Get mm_projector state_dict from model.
    model_sd = {k: v.cpu() for k, v in model.get_model().state_dict().items() if 'mm_projector' in k}

    # Build suffix index to handle different key prefixes.
    by_suffix = defaultdict(list)
    for k in pre_mm.keys():
        suffix = k.split('mm_projector')[-1]
        by_suffix[suffix].append(k)

    matched = []
    unmatched_model = []
    unmatched_pre = set(pre_mm.keys())

    print(f"🔎 Comparing pretrain({pretrain_path}) vs model.mm_projector, tol={tol}")
    for mk, mv in model_sd.items():
        suffix = mk.split('mm_projector')[-1]
        candidates = by_suffix.get(suffix, [])
        chosen = None
        if mk in pre_mm:
            chosen = mk
        elif candidates:
            chosen = candidates[0]
        else:
            # Try looser suffix matching.
            for pk in pre_mm.keys():
                if pk.endswith(suffix):
                    chosen = pk
                    break

        if chosen is None:
            unmatched_model.append(mk)
            continue

        pre_tensor = pre_mm[chosen].cpu()
        cur_tensor = mv.cpu()
        if pre_tensor.shape != cur_tensor.shape:
            print(f"⚠️ SHAPE MISMATCH {mk} vs {chosen}: {cur_tensor.shape} vs {pre_tensor.shape}")
            unmatched_model.append(mk)
            continue

        diff = (cur_tensor - pre_tensor).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        l2 = torch.norm((cur_tensor - pre_tensor).view(-1)).item()
        n_diff = (diff > tol).sum().item()
        total = diff.numel()
        pct = 100.0 * n_diff / total

        print(f"  {mk}  | matched pre_key: {chosen} | max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} l2={l2:.3e} diff_count={n_diff}/{total} ({pct:.3f}%)")
        matched.append((mk, chosen))
        if chosen in unmatched_pre:
            unmatched_pre.remove(chosen)

    print("\n📌 Summary:")
    print(f"  matched layers: {len(matched)}")
    print(f"  unmatched model mm_projector keys: {len(unmatched_model)} (first 10): {unmatched_model[:10]}")
    print(f"  pretrain keys not matched: {len(unmatched_pre)} (first 10): {list(unmatched_pre)[:10]}")
    # Quick overall stats.
    if matched:
        # Compute overall max diff and number of layers above tolerance.
        overall_max = 0.0
        overall_changed_layers = 0
        for mk, pk in matched:
            cur = model_sd[mk]
            pre_t = pre_mm[pk].cpu()
            m = (cur - pre_t).abs().max().item()
            if m > overall_max: overall_max = m
            if m > tol: overall_changed_layers += 1
        print(f"  overall max abs across matched layers: {overall_max:.3e}")
        print(f"  layers with any diff > tol: {overall_changed_layers}/{len(matched)}")
    return
def test_loaded_model():
    """Test the loaded model."""
    try:
        tokenizer, model, image_processor = load_lora_model()
        
        print(f"\n📊 Final model check:")
        print(f"  tokenizer: {type(tokenizer).__name__}")
        print(f"  model: {type(model).__name__}")
        print(f"  image_processor: {type(image_processor).__name__}")
        
        # Check key components.
        base_model = model.get_model()
        
        # Vision Tower
        if hasattr(base_model, 'vision_tower'):
            vision_tower = base_model.vision_tower
            print(f"  ✅ Vision Tower: {type(vision_tower).__name__}")
            print(f"     model name: {getattr(vision_tower, 'vision_tower_name', 'Unknown')}")
            if hasattr(vision_tower, 'region_weight'):
                print(f"     region weight: {vision_tower.region_weight}")
        
        # MM Projector
        if hasattr(base_model, 'mm_projector'):
            mm_projector = base_model.mm_projector
            print(f"  ✅ MM Projector: {type(mm_projector).__name__}")
        
        # LoRA status
        print(f"  ✅ LoRA status: {hasattr(model, 'peft_config')}")

        # Check LoRA parameter names.
        lora_names = [n for n, p in model.named_parameters() if "lora" in n]
        print("LoRA layers count:", len(lora_names))
        print("LoRA examples:", lora_names[:10])

        compare_mm_projector_with_pretrain(create_model_args().pretrain_mm_mlp_adapter, model)
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    test_loaded_model()