from typing import Any, Dict, Tuple

from models.model import Model


class DLRM_MoE(Model):
    """Deep Learning Recommendation Model with Mixture of Experts implementation."""

    def __init__(self, model_cfg: Dict[str, Any]):
        """Initialize DLRM_MoE model with configuration parameters.

        Args:
            model_cfg: Dictionary containing model architecture configuration
        """
        super().__init__(model_cfg)
        self.num_bot_mlp_layers = model_cfg["num_bot_mlp_layers"]
        self.num_top_mlp_layers = model_cfg["num_top_mlp_layers"]
        self.num_mlp_layers = self.num_bot_mlp_layers + self.num_top_mlp_layers
        self.mlp_dim = model_cfg["mlp_dim"]

        self.num_tables = model_cfg["num_tables"]
        self.entries_per_table = model_cfg["entries_per_table"]
        self.emb_dim = model_cfg["emb_dim"]
        self.pooling_size = model_cfg["pooling_size"]

        self.num_experts = model_cfg["num_experts"]
        self.num_active_experts = model_cfg["num_active_experts"]

        assert (
            self.num_experts >= self.num_active_experts
        ), "Incorrect number of active experts."

        self.mlp_layer_params = self.mlp_dim * self.mlp_dim
        self.total_params, self.mlp_params, self.mlp_active_params, self.emb_params = (
            self.get_num_params()
        )
        self.bot_layer_flops, self.top_layer_flops, self.total_flops = (
            self.get_num_flops()
        )
        self.lookup_bytes = self.get_lookup_bytes()

        self.print_summary_stats()

    def get_lookup_bytes(self) -> int:
        """Get bytes from sparse embedding lookups."""
        lookup_bytes = (
            self.num_tables
            * self.pooling_size
            * self.emb_dim
            * self.bytes_per_emb_param
        )
        return lookup_bytes

    def get_num_flops(self) -> Tuple[int, int, int]:
        """Get number of FLOPs per sample.

        Returns:
            Tuple of (bottom MLP layer FLOPs, top MLP layer FLOPs, total FLOPs)
        """
        bot_mlp_flops_layer = 2 * self.mlp_layer_params
        top_mlp_flops_layer = 2 * self.num_active_experts * self.mlp_layer_params
        mlp_flops_total = (
            bot_mlp_flops_layer * self.num_bot_mlp_layers
            + top_mlp_flops_layer * self.num_top_mlp_layers
        )
        return bot_mlp_flops_layer, top_mlp_flops_layer, mlp_flops_total

    def get_num_params(self) -> Tuple[int, int, int, int]:
        """Get number of parameters.

        Returns:
            Tuple of (total parameters, MLP parameters, active MLP parameters, embedding parameters)
        """
        mlp_params = (self.num_bot_mlp_layers * self.mlp_layer_params) + (
            self.num_experts * self.num_top_mlp_layers * self.mlp_layer_params
        )
        mlp_active_params = (self.num_bot_mlp_layers * self.mlp_layer_params) + (
            self.num_active_experts * self.num_top_mlp_layers * self.mlp_layer_params
        )
        emb_params = self.num_tables * self.entries_per_table * self.emb_dim
        total_params = mlp_params + emb_params
        return total_params, mlp_params, mlp_active_params, emb_params

    def print_summary_stats(self) -> None:
        """Print model summary statistics."""
        total_params_b = self.total_params / 1e9
        perc_dense_params = (self.mlp_params / self.total_params) * 100.0
        perc_sparse_params = 100.0 - perc_dense_params

        dense_size_gb = (self.mlp_params * self.bytes_per_nonemb_param) / 1e9
        sparse_size_gb = (self.emb_params * self.bytes_per_emb_param) / 1e9
        total_size_gb = dense_size_gb + sparse_size_gb

        mflops_bot_layer = self.bot_layer_flops / 1e6
        mflops_top_layer = self.top_layer_flops / 1e6
        mflops_total = self.total_flops / 1e6

        lookup_bytes_mb = self.lookup_bytes / 1e6

        print("**************************************************")
        super().print_summary_stats()
        print(
            "Parameters: {:.2f} B ({:.2f}% dense, {:.2f}% sparse).".format(
                total_params_b, perc_dense_params, perc_sparse_params
            )
        )
        print(
            "\t{:.2f} B dense params, {:.2f} B ({:.2f}%) active dense params".format(
                self.mlp_params / 1e9,
                self.mlp_active_params / 1e9,
                self.mlp_active_params / self.mlp_params * 100.0,
            )
        )
        print(
            "Size: {:.2f} GB ({:.2f} GB dense, {:.2f} GB sparse).".format(
                total_size_gb, dense_size_gb, sparse_size_gb
            )
        )
        print("FLOPs: {:.2f} MFLOPs per sample.".format(mflops_total))
        print(
            "\t({:.2f} MFLOPs per Bot MLP layer, {:.2f} MFLOPs per Top MLP layer) per sample.".format(
                mflops_bot_layer, mflops_top_layer
            )
        )
        print("Lookup Bytes: {:.2f} MB per sample.".format(lookup_bytes_mb))
        print("**************************************************")
