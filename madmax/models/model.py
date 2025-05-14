"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict


# Model Architecture
class Model:
    """Base model class for representing ML models."""

    def __init__(self, model_cfg: Dict[str, Any]):
        """Initialize model with configuration parameters.

        Args:
            model_cfg: Dictionary containing model architecture configuration
        """
        self.name = model_cfg["name"]
        self.type = model_cfg["type"]
        self.bytes_per_nonemb_param = model_cfg["bytes_per_nonemb_param"]
        self.bytes_per_emb_param = model_cfg["bytes_per_emb_param"]

    def print_summary_stats(self) -> None:
        """Print model summary statistics."""
        print(f"Model Name: {self.name}")

    def __str__(self) -> str:
        return f"Model(name={self.name}, type={self.type})"
