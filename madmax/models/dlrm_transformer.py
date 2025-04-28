from typing import Any, Dict, Tuple

from models.model import Model


class DLRM_Transformer(Model):
    """Deep Learning Recommendation Model with Transformer feature interaction implementation."""

    def __init__(self, model_cfg):
        """Initialize DLRM_Transformer model with configuration parameters."""
        super().__init__(model_cfg)
        # MLP Layers
        self.num_bot_mlp_layers = model_cfg["num_bot_mlp_layers"]
        self.num_top_mlp_layers = model_cfg["num_top_mlp_layers"]
        self.num_mlp_layers = self.num_bot_mlp_layers + self.num_top_mlp_layers
        self.mlp_dim = model_cfg["mlp_dim"]

        # Embedding layers
        self.num_tables = model_cfg["num_tables"]
        self.entries_per_table = model_cfg["entries_per_table"]
        self.emb_dim = model_cfg["emb_dim"]
        self.pooling_size = model_cfg["pooling_size"]

        # Transformer layers
        self.num_transformer_layers = model_cfg["num_transformer_layers"]
        self.num_transformer_heads = model_cfg["num_transformer_heads"]
        self.attention_dim = model_cfg["attention_dim"]
        self.transformer_fc_dim = model_cfg["transformer_fc_dim"]
        self.transformer_seq_len = model_cfg["transformer_seq_len"]

        self.attention_head_dim = self.attention_dim / self.num_transformer_heads

        self.mlp_layer_params = self.mlp_dim * self.mlp_dim
        self.attention_layer_params = 4 * self.attention_dim * self.attention_dim
        self.transformer_fc_layer_params = (
            2 * self.attention_dim * self.transformer_fc_dim
        )

        self.total_params, self.mlp_params, self.emb_params, self.transformer_params = (
            self.get_num_params()
        )
        (
            self.layer_flops,
            self.attention_layer_flops,
            self.transformer_fc_layer_flops,
            self.total_flops,
        ) = self.get_num_flops()
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

    def get_num_flops(self) -> Tuple[int, int, int, int]:
        """Get number of FLOPs per sample."""
        mlp_flops_layer = 2 * self.mlp_layer_params
        attention_flops_layer = (
            2 * self.attention_layer_params * self.transformer_seq_len
        )
        transformer_fc_flops_layer = (
            2 * self.transformer_fc_layer_params * self.transformer_seq_len
        )
        flops_total = (mlp_flops_layer * self.num_mlp_layers) + (
            (attention_flops_layer + transformer_fc_flops_layer)
            * self.num_transformer_layers
        )
        return (
            mlp_flops_layer,
            attention_flops_layer,
            transformer_fc_flops_layer,
            flops_total,
        )

    def get_num_params(self) -> Tuple[int, int, int, int]:
        """Get number of parameters."""
        mlp_params = self.num_mlp_layers * self.mlp_layer_params
        emb_params = self.num_tables * self.entries_per_table * self.emb_dim
        transformer_params = self.num_transformer_layers * (
            self.attention_layer_params + self.transformer_fc_layer_params
        )
        total_params = mlp_params + emb_params + transformer_params
        return total_params, mlp_params, emb_params, transformer_params

    def print_summary_stats(self) -> None:
        """Print model summary statistics."""
        total_params_b = self.total_params / 1e9
        perc_dense_params = (
            (self.mlp_params + self.transformer_params) / self.total_params
        ) * 100.0
        perc_transformer_dense_params = (
            self.transformer_params
            / (self.mlp_params + self.transformer_params)
            * 100.0
        )
        perc_base_dense_params = 100.0 - perc_transformer_dense_params
        perc_sparse_params = 100.0 - perc_dense_params

        dense_size_gb = (
            (self.mlp_params + self.transformer_params) * self.bytes_per_nonemb_param
        ) / 1e9
        sparse_size_gb = (self.emb_params * self.bytes_per_emb_param) / 1e9
        total_size_gb = dense_size_gb + sparse_size_gb

        mflops_layer = self.layer_flops / 1e6
        mflops_attention_layer = self.attention_layer_flops / 1e6
        mflops_transformer_fc_layer = self.transformer_fc_layer_flops / 1e6
        mflops_total = self.total_flops / 1e6

        perc_flops_base = (
            (self.layer_flops * self.num_mlp_layers) / self.total_flops
        ) * 100.0
        perc_flops_transformer = 100.0 - perc_flops_base

        lookup_bytes_mb = self.lookup_bytes / 1e6

        print("**************************************************")
        super().print_summary_stats()
        print(
            "Parameters: {:.2f} B ({:.2f}% dense, {:.2f}% sparse).".format(
                total_params_b, perc_dense_params, perc_sparse_params
            )
        )
        print(
            "Size: {:.2f} GB ({:.2f} GB dense, {:.2f} GB sparse).".format(
                total_size_gb, dense_size_gb, sparse_size_gb
            )
        )
        print(
            "\t Dense parameters: {:.2f}% base MLPs, {:.2f}% transformer".format(
                perc_base_dense_params, perc_transformer_dense_params
            )
        )
        print(
            "FLOPs: {:.2f} MFLOPs per sample ({:.2f} MFLOPs per MLP layer, {:.2f} MFLOPs per attention layer, {:.2f} MFLOPs per Transformer FC).".format(
                mflops_total,
                mflops_layer,
                mflops_attention_layer,
                mflops_transformer_fc_layer,
            )
        )
        print(
            "\t{:.2f}% base MLPs, {:.2f}% transformer".format(
                perc_flops_base, perc_flops_transformer
            )
        )
        print("Lookup Bytes: {:.2f} MB per sample.".format(lookup_bytes_mb))
        print("**************************************************")
