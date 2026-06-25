#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import tempfile

import torch
import yaml
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from qwen25vl_safetensors_keys import fix_qwen25vl_visual_prefix_in_dir


def _find_adapter_weight_file(adapter_dir: str) -> str | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(adapter_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _find_adapter_config_file(start_dir: str, max_parent_levels: int = 4) -> str | None:
    """Look for adapter_config.json in start_dir and up to max_parent_levels parents."""
    cur = os.path.abspath(start_dir)
    for _ in range(max_parent_levels + 1):
        candidate = os.path.join(cur, "adapter_config.json")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def ensure_adapter_config_present(adapter_name_or_path: str, cleanup_dirs: list[str]) -> str:
    """
    PeftModel requires adapter_config.json next to adapter weights. Some trainers only
    write weights under checkpoint-N/ while adapter_config.json lives in the run root.
    If weights exist but config is missing locally, copy config from a parent directory
    into a temporary folder together with the weights.
    """
    adapter_dir = os.path.abspath(adapter_name_or_path)
    cfg_in_dir = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(cfg_in_dir):
        return adapter_name_or_path

    weight_path = _find_adapter_weight_file(adapter_dir)
    if weight_path is None:
        raise ValueError(
            f"No adapter_config.json and no adapter_model.safetensors / adapter_model.bin under {adapter_dir!r}. "
            "Pass a directory that contains a PEFT adapter (or enable checkpoint layout with config in a parent folder)."
        )

    cfg_path = _find_adapter_config_file(adapter_dir)
    if cfg_path is None:
        raise ValueError(
            f"Missing adapter_config.json for weights in {adapter_dir!r}. "
            "Looked in this folder and up to 4 parent directories. "
            "Copy adapter_config.json from your training run next to the adapter weights, or use an adapter export that includes it."
        )

    tmpdir = tempfile.mkdtemp(prefix="merge_lora_adapter_cfg_")
    cleanup_dirs.append(tmpdir)
    try:
        shutil.copy2(weight_path, os.path.join(tmpdir, os.path.basename(weight_path)))
        shutil.copy2(cfg_path, os.path.join(tmpdir, "adapter_config.json"))
        readme = os.path.join(adapter_dir, "README.md")
        if os.path.isfile(readme):
            shutil.copy2(readme, os.path.join(tmpdir, "README.md"))
        return tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_dirs.pop()
        raise


def remap_adapter_keys_and_prepare_dir(adapter_name_or_path: str) -> str:
    """
    Remap known adapter key mismatches (e.g. language_model.layers, visual.blocks,
    visual.merger, and default-adapter naming) so PeftModel.from_pretrained can
    load without 'missing adapter keys' warnings.
    Writes remapped adapter to a temp dir and returns that path.
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        raise ImportError("safetensors is required for remap_adapter_keys. pip install safetensors")
    adapter_path = os.path.abspath(adapter_name_or_path)
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(safetensors_path):
        return adapter_name_or_path
    sd = load_file(safetensors_path)
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace(".model.model.language_model.layers.", ".model.model.layers.")
        nk = nk.replace(".model.model.visual.blocks.", ".model.visual.blocks.")
        # Same extra `.model.` wrapper as blocks; PEFT expects base_model.model.visual.merger.*
        nk = nk.replace(".model.model.visual.merger.", ".model.visual.merger.")
        # Training saves lora_A.weight; Peft default adapter uses lora_A.default.weight
        if ".visual.merger." in nk:
            nk = nk.replace("lora_A.weight", "lora_A.default.weight")
            nk = nk.replace("lora_B.weight", "lora_B.default.weight")
        new_sd[nk] = v
    tmpdir = tempfile.mkdtemp(prefix="merge_lora_remap_")
    try:
        save_file(new_sd, os.path.join(tmpdir, "adapter_model.safetensors"))
        for fn in ("adapter_config.json", "README.md"):
            src = os.path.join(adapter_path, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmpdir, fn))
        if not os.path.isfile(os.path.join(tmpdir, "adapter_config.json")):
            found = _find_adapter_config_file(adapter_path)
            if found:
                shutil.copy2(found, os.path.join(tmpdir, "adapter_config.json"))
        return tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", type=str, default=None, help="Path to merge config YAML")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="Base model or merged SFT model path")
    parser.add_argument("--adapter-name-or-path", type=str, default=None, help="LoRA adapter directory")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to save merged weights")
    parser.add_argument(
        "--remap-adapter-keys",
        type=str,
        default="false",
        help=(
            "Remap adapter keys before merge (true/false). Default false - use raw adapter keys. "
            "WARNING: true often yields 0%% LoRA applied on Qwen2.5-VL GRPO checkpoints."
        ),
    )
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_processor_or_tokenizer(model_name_or_path: str, export_dir: str) -> None:
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor.save_pretrained(export_dir)
        return
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(export_dir)


def ensure_cuda_home() -> None:
    """Best-effort CUDA_HOME for tools that import DeepSpeed (merge save path avoids DS when possible)."""
    ch = os.environ.get("CUDA_HOME", "").strip()
    if ch and os.path.isdir(ch):
        return
    try:
        import torch.utils.cpp_extension as cep

        th = getattr(cep, "CUDA_HOME", None)
        if th and isinstance(th, str) and os.path.isdir(th):
            os.environ["CUDA_HOME"] = th
            return
    except Exception:
        pass
    for candidate in ("/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.4"):
        if os.path.isdir(candidate):
            os.environ["CUDA_HOME"] = candidate
            return


def save_merged_pretrained(model, export_dir: str) -> None:
    """merge_and_unload() returns an unwrapped model; skip accelerate unwrap to avoid importing deepspeed."""
    import glob
    import transformers.modeling_utils as modeling_utils

    _unwrap = modeling_utils.unwrap_model

    def _unwrap_identity(m, *args, **kwargs):
        return m

    try:
        modeling_utils.unwrap_model = _unwrap_identity  # type: ignore[assignment]
        model.save_pretrained(export_dir, safe_serialization=True)
    finally:
        modeling_utils.unwrap_model = _unwrap

    has_weights = bool(glob.glob(os.path.join(export_dir, "*.safetensors"))) or bool(
        glob.glob(os.path.join(export_dir, "pytorch_model*.bin"))
    )
    if not has_weights:
        raise RuntimeError(
            f"save_pretrained finished but no *.safetensors or pytorch_model*.bin under {export_dir!r}. "
            "Often OOM/kill on login node or full disk - re-run on a compute node with enough RAM/disk, "
            "or check quota and merge logs."
        )


def get_base_model(model_name_or_path: str):
    """Load Qwen2.5-VL base weights. No CausalLM fallback - VL config is incompatible and hides real errors."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Install a transformers build that provides Qwen2_5_VLForConditionalGeneration "
            "(Qwen2.5-VL merge is not supported via AutoModelForCausalLM)."
        ) from e
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
    )


_PROBE_KEY_SUBSTR = "layers.0.self_attn.q_proj.weight"


def _snapshot_probe_weight(model):
    """Return (key, cloned_tensor) for a representative language-model projection."""
    for name, param in model.named_parameters():
        if _PROBE_KEY_SUBSTR in name and "lora" not in name:
            return name, param.detach().float().clone()
    # Fallback: first floating-point parameter.
    for name, param in model.named_parameters():
        if param.is_floating_point():
            return name, param.detach().float().clone()
    return None, None


def _verify_merge_delta(merged_model, probe_key, probe_before, remap_adapter_keys) -> None:
    """Fail loudly if the merge did not change weights (0% LoRA applied)."""
    if probe_key is None or probe_before is None:
        print("[merge_lora] WARNING: could not snapshot a probe weight; skipping delta check.", flush=True)
        return
    after = None
    for name, param in merged_model.named_parameters():
        if name == probe_key:
            after = param.detach().float()
            break
    if after is None:
        print(f"[merge_lora] WARNING: probe key {probe_key!r} missing after merge; skipping delta check.", flush=True)
        return
    delta = (after - probe_before.to(after.device)).norm().item()
    print(f"[merge_lora] merge delta check: ||?({probe_key})||_2 = {delta:.6f}", flush=True)
    if delta <= 1e-6:
        hint = (
            "remap_adapter_keys=true likely caused 696/696 missing keys (0% LoRA applied). "
            "Re-run with --remap-adapter-keys false."
            if remap_adapter_keys
            else "Check that the adapter weights are non-zero and keys match the base model."
        )
        raise RuntimeError(
            f"[merge_lora] LoRA delta is ~0 ({delta:.2e}) on {probe_key!r}: the merged model is "
            f"identical to the SFT base (NOT a real GRPO model). {hint}"
        )


def main() -> None:
    args = parse_args()
    ensure_cuda_home()
    if args.config is not None:
        cfg = load_yaml(args.config)
    else:
        cfg = {}

    model_name_or_path = args.model_name_or_path or cfg.get("model_name_or_path")
    adapter_name_or_path = args.adapter_name_or_path or cfg.get("adapter_name_or_path")
    export_dir = args.export_dir or cfg.get("export_dir")
    remap_adapter_keys = cfg.get("remap_adapter_keys", False)
    if args.remap_adapter_keys is not None:
        remap_adapter_keys = str(args.remap_adapter_keys).strip().lower() in {"1", "true", "yes", "y", "on"}

    if remap_adapter_keys:
        print(
            "\n*** WARNING: remap_adapter_keys=true ***\n"
            "GRPO/Qwen2.5-VL adapters saved with raw keys (lora_A.weight) load correctly WITHOUT remap.\n"
            "Legacy remap (language_model.layers, lora_A.default.weight) often causes 696/696 missing keys\n"
            "and zero LoRA delta (remerged_clean ? SFT). Prefer --remap-adapter-keys false.\n"
            "Proceed only if you verified Peft missing keys = 0 after remap.\n",
            flush=True,
        )
    else:
        print("[merge_lora] remap_adapter_keys=false (default): merging raw adapter keys as stored.", flush=True)

    if not model_name_or_path or not adapter_name_or_path or not export_dir:
        raise ValueError(
            "model_name_or_path, adapter_name_or_path, and export_dir must be provided "
            "either via --config or direct CLI flags."
        )

    os.makedirs(export_dir, exist_ok=True)

    cleanup_dirs: list[str] = []
    adapter_name_or_path = ensure_adapter_config_present(adapter_name_or_path, cleanup_dirs)
    if remap_adapter_keys:
        adapter_name_or_path = remap_adapter_keys_and_prepare_dir(adapter_name_or_path)
        cleanup_dirs.append(adapter_name_or_path)

    base_model = get_base_model(model_name_or_path)

    # Snapshot a representative weight BEFORE applying the adapter, so we can verify
    # the merge actually changed weights (guards against the remap regression that
    # silently yields a 0% LoRA-applied model identical to the SFT base).
    probe_key, probe_before = _snapshot_probe_weight(base_model)

    peft_model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
    merged_model = peft_model.merge_and_unload()

    _verify_merge_delta(merged_model, probe_key, probe_before, remap_adapter_keys)

    save_merged_pretrained(merged_model, export_dir)
    n_files, n_tensors = fix_qwen25vl_visual_prefix_in_dir(export_dir)
    if n_tensors:
        print(
            f"[merge_lora] Fixed HF/vLLM keys: model.visual.* -> visual.* "
            f"({n_tensors} tensors in {n_files} shard file(s))"
        )
    save_processor_or_tokenizer(model_name_or_path, export_dir)

    for d in cleanup_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
