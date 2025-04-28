from typing import Any, Dict, Tuple

from models.model import Model


class LLM_MoE(Model):
    """Large Language Model with Mixture of Experts implementation."""

    def __init__(self, model_cfg: Dict[str, Any]):
        """Initialize LLM_MoE model with configuration parameters."""
        super().__init__(model_cfg)
        self.entries_per_table = model_cfg["entries_per_table"]

        # Transformer configuration
        self.num_transformer_layers = model_cfg["num_transformer_layers"]
        self.num_transformer_heads = model_cfg["num_transformer_heads"]
        self.attention_dim = model_cfg["attention_dim"]
        self.transformer_fc_dim = model_cfg["transformer_fc_dim"]
        self.transformer_seq_len = model_cfg["transformer_seq_len"]

        # MoE configuration
        self.num_experts = model_cfg["num_experts"]
        self.num_active_experts = model_cfg["num_active_experts"]

        assert (
            self.num_experts >= self.num_active_experts
        ), "Incorrect number of active experts."

        self.attention_head_dim = self.attention_dim / self.num_transformer_heads

        # Parameter calculations
        self.word_emb_params = self.entries_per_table * self.attention_dim
        self.position_emb_params = self.transformer_seq_len * self.attention_dim
        self.attention_layer_params = 4 * self.attention_dim * self.attention_dim
        self.transformer_fc_layer_params = (
            2 * self.attention_dim * self.transformer_fc_dim
        )

        # Calculate model statistics
        (
            self.total_params,
            self.emb_params,
            self.transformer_params,
            self.active_transformer_params,
        ) = self.get_num_params()
        (
            self.attention_layer_flops,
            self.transformer_fc_layer_flops,
            self.total_flops,
        ) = self.get_num_flops()
        self.lookup_bytes = self.get_lookup_bytes()

        self.print_summary_stats()

    def get_lookup_bytes(self) -> int:
        """Get bytes from sparse embedding lookups."""
        lookup_bytes = (
            2 * self.transformer_seq_len * self.attention_dim * self.bytes_per_emb_param
        )
        return lookup_bytes

    def get_num_flops(self) -> Tuple[int, int, int]:
        """Get number of FLOPs per sample.

        Returns:
            Tuple of (attention layer FLOPs, transformer FC layer FLOPs, total FLOPs)
        """
        attention_flops_layer = (
            2 * self.attention_layer_params * self.transformer_seq_len
        )
        transformer_fc_flops_layer = (
            2
            * self.num_active_experts
            * self.transformer_fc_layer_params
            * self.transformer_seq_len
        )
        flops_total = (
            attention_flops_layer + transformer_fc_flops_layer
        ) * self.num_transformer_layers
        return attention_flops_layer, transformer_fc_flops_layer, flops_total

    def get_num_params(self) -> Tuple[int, int, int, int]:
        """Get number of parameters.

        Returns:
            Tuple of (total parameters, embedding parameters, transformer parameters, active transformer parameters)
        """
        emb_params = self.word_emb_params + self.position_emb_params
        transformer_params = self.num_transformer_layers * (
            self.attention_layer_params
            + self.num_experts * self.transformer_fc_layer_params
        )
        active_transformer_params = self.num_transformer_layers * (
            self.attention_layer_params
            + self.num_active_experts * self.transformer_fc_layer_params
        )
        total_params = emb_params + transformer_params
        return total_params, emb_params, transformer_params, active_transformer_params

    def print_summary_stats(self) -> None:
        """Print model summary statistics."""
        total_params_b = self.total_params / 1e9
        perc_dense_params = (self.transformer_params / self.total_params) * 100.0
        perc_sparse_params = 100.0 - perc_dense_params

        attention_params_b = (
            self.attention_layer_params * self.num_transformer_layers / 1e9
        )
        perc_attention_params = (attention_params_b / total_params_b) * 100.0

        transformer_fc_params_b = (
            self.num_experts
            * self.transformer_fc_layer_params
            * self.num_transformer_layers
            / 1e9
        )
        perc_transformer_fc_params = (transformer_fc_params_b / total_params_b) * 100.0

        dense_size_gb = (self.transformer_params * self.bytes_per_nonemb_param) / 1e9
        sparse_size_gb = (self.emb_params * self.bytes_per_emb_param) / 1e9
        total_size_gb = dense_size_gb + sparse_size_gb

        mflops_attention_layer = self.attention_layer_flops / 1e6
        mflops_transformer_fc_layer = self.transformer_fc_layer_flops / 1e6
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
            "\tAttention: {:.2f} B ({:.2f}%)".format(
                attention_params_b, perc_attention_params
            )
        )
        print(
            "\tFC: {:.2f} B ({:.2f}%) -- {:.2f} B per expert ({} experts)".format(
                transformer_fc_params_b,
                perc_transformer_fc_params,
                transformer_fc_params_b / self.num_experts,
                self.num_experts,
            )
        )
        print(
            "\tActive: {:.2f} GB ({:.2f}%)".format(
                self.active_transformer_params / 1e9,
                100.0 * self.active_transformer_params / self.transformer_params,
            )
        )
        print(
            "Size: {:.2f} GB ({:.2f} GB dense, {:.2f} GB sparse).".format(
                total_size_gb, dense_size_gb, sparse_size_gb
            )
        )
        print(
            "FLOPs: {:.2f} MFLOPs per sample ({:.2f} MFLOPs per attention layer, {:.2f} MFLOPs per Transformer FC).".format(
                mflops_total, mflops_attention_layer, mflops_transformer_fc_layer
            )
        )
        print("Lookup Bytes: {:.2f} MB per sample.".format(lookup_bytes_mb))
        print("**************************************************")
