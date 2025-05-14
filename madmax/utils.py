"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

# Imports
import argparse
import json
import sys
from typing import Any, Dict

# Model imports mapping
MODEL_IMPORTS = {
    "DLRM": ("models.dlrm", "DLRM"),
    "DLRM_Transformer": ("models.dlrm_transformer", "DLRM_Transformer"),
    "DLRM_MoE": ("models.dlrm_moe", "DLRM_MoE"),
    "LLM": ("models.llm", "LLM"),
    "LLM_MoE": ("models.llm_moe", "LLM_MoE"),
    "ViT": ("models.vit", "ViT"),
}

# System imports mapping
SYSTEM_IMPORTS = {"GPU": ("systems.gpus", "GPUs")}

# Task imports mapping
TASK_IMPORTS = {
    "DLRM": ("tasks.dlrm_tasks", "DLRM_Task"),
    "DLRM_Transformer": ("tasks.dlrm_transformer_tasks", "DLRM_Transformer_Task"),
    "DLRM_MoE": ("tasks.dlrm_moe_tasks", "DLRM_MoE_Task"),
    "LLM": ("tasks.llm_tasks_preoptimized", "LLM_Task"),  # Pre-optimized version
    "LLM_MoE": ("tasks.llm_moe_tasks", "LLM_MoE_Task"),
    "ViT": ("tasks.vit_tasks", "ViT_Task"),  # Enable post-optimized version
    "Seamless": ("tasks.seamless_tasks", "SeamlessM4T_Task"),
}


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        sys.exit(f"Error loading config file {config_file}: {e}")


# Import model
def import_model(model_cfg_file: str) -> Any:
    """Import model based on configuration file."""
    model_cfg = load_config(model_cfg_file)
    model_type = model_cfg["type"]

    if model_type not in MODEL_IMPORTS:
        sys.exit(f'Model type "{model_type}" undefined!')

    module_path, class_name = MODEL_IMPORTS[model_type]
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    return model_class(model_cfg)


def import_system(system_cfg_file: str) -> Any:
    """Import system based on configuration file."""
    system_cfg = load_config(system_cfg_file)
    system_type = system_cfg["type"]

    if system_type not in SYSTEM_IMPORTS:
        sys.exit(f'System type "{system_type}" undefined!')

    module_path, class_name = SYSTEM_IMPORTS[system_type]
    module = __import__(module_path, fromlist=[class_name])
    system_class = getattr(module, class_name)

    return system_class(system_cfg)


def import_task(model: Any, system: Any, task_cfg_file: str) -> Any:
    """Import task based on model type and configuration file."""
    task_cfg = load_config(task_cfg_file)
    model_type = model.type

    if model_type not in TASK_IMPORTS:
        sys.exit(f'Task for model type "{model_type}" undefined!')

    module_path, class_name = TASK_IMPORTS[model_type]
    module = __import__(module_path, fromlist=[class_name])
    task_class = getattr(module, class_name)

    return task_class(model, system, task_cfg)


def parse_configurations() -> argparse.Namespace:
    """Parse command line configurations."""
    parser = argparse.ArgumentParser(description="Performance Model")

    # Configuration Arguments
    parser.add_argument(
        "--model-cfg-file",
        type=str,
        default="model_cfgs/dlrm/dlrm_a.json",
        help="Model architecture configuration file.",
    )
    parser.add_argument(
        "--system-cfg-file",
        type=str,
        default="system_cfgs/zionex/zionex_128.json",
        help="System configuration file.",
    )
    parser.add_argument(
        "--task-cfg-file",
        type=str,
        default="task_cfgs/dlrm/dlrm_train.json",
        help="Task configuration file.",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="figures/", help="Directory to save figures."
    )

    return parser.parse_args()
