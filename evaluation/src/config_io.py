from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configs. Install pyyaml.") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def normalize_model_diet_config(model_diet: dict[str, Any]) -> dict[str, Any]:
    """Accept both legacy {"models": ...} files and the spec's top-level model schema."""
    if "models" in model_diet:
        return model_diet
    reserved = {"dataset", "eval", "canonical", "metric", "runtime", "output"}
    models = {
        name: cfg
        for name, cfg in model_diet.items()
        if isinstance(cfg, dict) and name not in reserved
    }
    return {"models": models}


def normalize_global_config(global_cfg: dict[str, Any]) -> dict[str, Any]:
    """Expose the new eval_global.yaml schema through the legacy runtime keys."""
    cfg = deepcopy(global_cfg)
    eval_cfg = cfg.get("eval", {})
    intensity = eval_cfg.get("intensity", {})
    metrics = eval_cfg.get("metrics", {})

    if "canonical" not in cfg and intensity:
        clip = intensity.get("clip_for_metric", [-1024.0, 3000.0])
        cfg["canonical"] = {
            "axis_order": eval_cfg.get("gt", {}).get("axis_order_after_loading", "ZYX"),
            "ct_min": float(intensity.get("norm_min", clip[0])),
            "ct_max": float(intensity.get("norm_max", clip[1])),
            "clip_before_metric": True,
            "canonical_space": intensity.get("canonical_space", "project_defined"),
        }

    if "metric" not in cfg and metrics:
        cfg["metric"] = {
            "mae": {"mae_primary": "raw", "output_both_raw_and_norm": True},
            "psnr": {"use_normalized": True, "data_range": 1.0, "eps": 1.0e-8},
            "ssim": {
                "enabled": bool(metrics.get("compute_ssim", True)),
                **metrics.get("ssim", {}),
            },
            "lpips": {
                "enabled": bool(metrics.get("compute_lpips", True)),
                "bbox_padding": metrics.get("lpips", {}).get("bbox_margin", 8),
                "backbone": metrics.get("lpips", {}).get("net", "alex"),
                **metrics.get("lpips", {}),
            },
        }

    if "output" not in cfg:
        cfg["output"] = {
            "output_dir": "./evaluation/outputs",
            "per_case_csv": "per_case_metrics.csv",
            "summary_csv": "summary_metrics.csv",
            "debug_csv": "debug_alignment.csv",
            "failed_csv": "failed_cases.csv",
        }
    return cfg


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write YAML configs. Install pyyaml.") from exc

    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_excel_config(excel_path: str | Path) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas and openpyxl are required to read Excel configs. Install pandas openpyxl."
        ) from exc

    df = pd.read_excel(excel_path)
    column_map = {
        "模型": "model",
        "GT": "gt_source",
        "归一化范围": "norm_range_text",
        "reshape": "reshape_text",
        "分辨率": "resolution_text",
        "前向投影": "projector",
        "与GT对齐的处理": "align_note",
    }
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        model = row.get("模型")
        if model is None or str(model).strip() == "" or str(model) == "nan":
            continue
        model_name = str(model).strip()
        out[model_name] = {}
        for src, dst in column_map.items():
            value = row.get(src)
            if value is None:
                value = ""
            out[model_name][dst] = "" if str(value) == "nan" else str(value)
    return out


def merge_excel_config(model_diet: dict[str, Any], excel_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(normalize_model_diet_config(model_diet))
    models = merged.setdefault("models", {})
    for excel_name, excel_model in excel_cfg.items():
        canonical = resolve_model_name(excel_name, merged, required=False) or excel_name
        model_cfg = models.setdefault(canonical, {"enabled": True, "aliases": [excel_name]})
        aliases = set(model_cfg.get("aliases", []))
        aliases.add(excel_name)
        model_cfg["aliases"] = sorted(aliases)
        model_cfg["excel_raw"] = excel_model
        if excel_model.get("gt_source") and not model_cfg.get("train_gt_source"):
            model_cfg["train_gt_source"] = excel_model["gt_source"]
        if excel_model.get("projector") and not model_cfg.get("projector"):
            model_cfg["projector"] = excel_model["projector"]
    return merged


def resolve_model_name(name: str, model_diet: dict[str, Any], required: bool = True) -> str | None:
    models = normalize_model_diet_config(model_diet).get("models", {})
    if name in models:
        return name
    lowered = name.lower()
    for model_name, cfg in models.items():
        aliases = [model_name, *cfg.get("aliases", [])]
        if any(str(alias).lower() == lowered for alias in aliases):
            return model_name
    if required:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(models)}")
    return None


def resolve_model_names(names: list[str] | None, model_diet: dict[str, Any]) -> list[str]:
    models = normalize_model_diet_config(model_diet).get("models", {})
    if not names:
        return [name for name, cfg in models.items() if cfg.get("enabled", True)]
    resolved: list[str] = []
    for name in names:
        model_name = resolve_model_name(name, model_diet)
        if model_name not in resolved:
            resolved.append(model_name)
    return resolved


def read_test_list(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        cases = data.get("test", [])
        if not isinstance(cases, list):
            raise ValueError(f"JSON test list must contain a list field named 'test': {path}")
        if not cases:
            raise ValueError(f"No cases found in JSON test list field 'test': {path}")
        return [str(case_id) for case_id in cases]

    cases: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            cases.append(case_id_from_token(text.split()[0]))
    if not cases:
        raise ValueError(f"No cases found in test list: {path}")
    return cases


def case_id_from_token(token: str) -> str:
    name = Path(token).name if "/" in token or "\\" in token else token
    known_exts = [
        ".nii.gz",
        ".nii",
        ".mha",
        ".mhd",
        ".h5",
        ".hdf5",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
    ]
    lowered = name.lower()
    for ext in known_exts:
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def build_runtime_diet(global_cfg: dict[str, Any], model_cfg: dict[str, Any], lpips_runner=None) -> dict[str, Any]:
    global_cfg = normalize_global_config(global_cfg)
    return {
        "canonical": global_cfg["canonical"],
        "metric": global_cfg["metric"],
        "model": model_cfg,
        "_runtime": {"lpips_runner": lpips_runner},
    }
