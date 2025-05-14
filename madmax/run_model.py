"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

# Imports
import os

from utils import import_model, import_system, import_task, parse_configurations
from visualize import plot_overall_results, plot_timeline


def run_task(task, model, system):
    """Run the appropriate task based on task type and return streams."""
    if task.type == "pretrain":
        return task.build_pretrain(model, system)
    elif task.type == "inference":
        return task.build_inference(model, system)
    elif task.type == "finetune":
        return task.build_finetune(model, system)
    else:
        raise ValueError(f"Unknown task type: {task.type}")


def create_output_directory(directory):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    args = parse_configurations()
    model = import_model(args.model_cfg_file)
    system = import_system(args.system_cfg_file)
    task = import_task(model, system, args.task_cfg_file)

    computation_stream, communication_stream = run_task(task, model, system)

    create_output_directory("figures")

    plot_overall_results(task, args.figures_dir)
    plot_timeline(computation_stream, communication_stream, args.figures_dir)


if __name__ == "__main__":
    main()
