"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import sys
from typing import Any

from tasks.tasks import Task


# LLM Tasks
class LLM_Task(Task):

    def __init__(self, model, system, task_cfg):
        super().__init__(model, system, task_cfg)

        self.emb_parallel = task_cfg["emb_parallel"]
        self.emb_parallel_degree = task_cfg["emb_parallel_degree"]
        self.transformer_parallel = task_cfg["transformer_parallel"]
        self.transformer_parallel_degree = task_cfg["transformer_parallel_degree"]

        self.local_emb_bs = task_cfg["local_emb_bs"]
        self.local_transformer_bs = task_cfg["local_transformer_bs"]

        if self.type == "finetune":
            self.freeze_emb = task_cfg["freeze_emb"]
            self.freeze_transformer = task_cfg["freeze_transformer"]
            self.min_frozen_layer = task_cfg["min_frozen_layer"]

        self._set_effective_flops(model)

        self.check_parallelization_strats(
            model, system.num_devices, system.num_intra_node_devices, restrict2d=True
        )

        (
            self.emb_duplciate_factor,
            self.emb_shard_factor,
            self.transformer_duplicate_factor,
            self.transformer_shard_factor,
        ) = self.get_parallelization_factors()

        self.global_emb_bs = self.local_emb_bs * self.emb_duplciate_factor
        self.global_transformer_bs = (
            self.local_transformer_bs * self.transformer_duplicate_factor
        )

        assert self.global_emb_bs == self.global_transformer_bs
        self.global_bs = self.global_emb_bs

        self.emb_cap_per_device, self.transformer_cap_per_device = self.get_mem_usage(
            model
        )
        (
            self.local_attention_layer_flops,
            self.global_attention_layer_flops,
            self.local_transformer_fc_layer_flops,
            self.global_transformer_fc_layer_flops,
            self.local_total_flops,
            self.global_total_flops,
        ) = self.get_task_flops(model)
        (
            self.local_lookup_bytes,
            self.global_lookup_bytes,
            self.lookup_bytes_per_device,
        ) = self.get_task_lookup_bytes(model)

        self.print_summary_stats()

    def _set_effective_flops(self, model: Any) -> None:
        """Set effective FLOPS based on parameter precision."""
        if model.bytes_per_nonemb_param == 8:
            self.eff_flops = self.system.eff_f64_flops
        elif model.bytes_per_nonemb_param == 4:
            self.eff_flops = self.system.eff_f32_flops
        elif model.bytes_per_nonemb_param == 2:
            self.eff_flops = self.system.eff_f16_flops
        elif model.bytes_per_nonemb_param == 1:
            self.eff_flops = self.system.eff_i8_ops
        else:
            sys.exit(
                "Invalid dense parameter specification with respect to system specs."
            )

    # Build forward-pass streams
    def build_fwd(self, model, system):
        computation_stream = []
        communication_stream = []
        t_comp = 0
        t_comm = 0

        # --- Handle embedding table ---
        # Handle computation (lookups)
        t_emb = self.lookup_bytes_per_device / system.eff_mem_bw
        t_comp, t_comm = self.add_trace(
            "EMB_f.",
            t_emb,
            {},
            "comp",
            computation_stream,
            communication_stream,
            t_comp,
            t_comm,
        )
        self.t_emb_total += t_emb

        # Handle communication
        for parallel_lvl, (strat, deg) in enumerate(
            zip(self.emb_parallel, self.emb_parallel_degree)
        ):
            if strat == "mp":
                t_emb_c = (
                    2
                    * (self.lookup_bytes_per_device / deg)
                    / system.eff_all2all_bw[-len(self.emb_parallel) + parallel_lvl]
                )
                t_comp, t_comm = self.add_trace(
                    "EMB_f_c_all2all.",
                    t_emb_c,
                    {"EMB_f."},
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_all2all_total += t_emb_c
            elif self.emb_parallel[0] == "ddp":
                pass  # no need to communicate if DDP

        for transformer_layer_num in range(model.num_transformer_layers):
            # --- Handle Transformer - 1) Attention ---
            # If FSDP, need to gather weights before computation
            if self.transformer_parallel[0] == "fsdp":
                # FSDP allgather time proportional to output buffer size.
                attn_layer_bytes = (
                    model.attention_layer_params
                    / self.transformer_shard_factor
                    * self.transformer_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_attn_c = (
                    attn_layer_bytes
                    / system.eff_allgather_bw[-len(self.transformer_parallel)]
                )  # FSDP is always first level of parallelism.
                attn_deps = {"Attn{}_f.".format(transformer_layer_num - 1)}
                t_comp, t_comm = self.add_trace(
                    "Attn{}_f_c_wgt_ag.".format(transformer_layer_num),
                    t_attn_c,
                    attn_deps,
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_attn_c
            # Handle computation
            t_attn = self.local_attention_layer_flops / self.eff_flops
            if transformer_layer_num == 0:
                attn_deps = {
                    "EMB_f_c",
                    "Attn{}_f_c_wgt_ag.".format(transformer_layer_num),
                }
            else:
                attn_deps = {
                    "TransformerFC{}_f.".format(transformer_layer_num - 1),
                    "TransformerFC{}_f_c_act_ar.".format(transformer_layer_num - 1),
                    "Attn{}_f_c_wgt_ag.".format(transformer_layer_num),
                }
            t_comp, t_comm = self.add_trace(
                "Attn{}_f.".format(transformer_layer_num),
                t_attn,
                attn_deps,
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_gemm_total += t_attn
            # Handle communication
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.transformer_parallel, self.transformer_parallel_degree)
            ):
                if strat == "ddp":
                    pass  # no need to communicate if 1D DDP
                elif strat == "tp":
                    # Reduce activations after even layers.
                    activations_bytes = (
                        self.local_transformer_bs
                        * model.transformer_seq_len
                        * model.attention_dim
                        * model.bytes_per_nonemb_param
                    )
                    t_activations_c = (
                        activations_bytes
                        / system.eff_allreduce_bw[
                            -len(self.transformer_parallel) + parallel_lvl
                        ]
                    )
                    t_comp, t_comm = self.add_trace(
                        "Attn{}_f_c_act_ar.".format(transformer_layer_num),
                        t_activations_c,
                        {"Attn{}_f.".format(transformer_layer_num)},
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_allreduce_total += t_activations_c

            # --- Handle Transformer - 2) FC ---
            # If FSDP, need to gather weights before computation
            if self.transformer_parallel[0] == "fsdp":
                # FSDP allgather time proportional to output buffer size.
                fc_layer_bytes = (
                    model.transformer_fc_layer_params
                    / self.transformer_shard_factor
                    * self.transformer_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_fc_c = (
                    fc_layer_bytes
                    / system.eff_allgather_bw[-len(self.transformer_parallel)]
                )  # FSDP is always first level of parallelism.
                fc_deps = {"TransformerFC{}_f.".format(transformer_layer_num - 1)}
                t_comp, t_comm = self.add_trace(
                    "TransformerFC{}_f_c_wgt_ag.".format(transformer_layer_num),
                    t_fc_c,
                    fc_deps,
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_fc_c
            # Handle computation
            t_fc = self.local_transformer_fc_layer_flops / self.eff_flops
            fc_deps = {
                "Attn{}_f.".format(transformer_layer_num),
                "Attn{}_f_c_act_ar.".format(transformer_layer_num),
                "TransformerFC{}_f_c_wgt_ag.".format(transformer_layer_num),
            }
            t_comp, t_comm = self.add_trace(
                "TransformerFC{}_f.".format(transformer_layer_num),
                t_fc,
                fc_deps,
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_gemm_total += t_fc
            # Handle communication
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.transformer_parallel, self.transformer_parallel_degree)
            ):
                if strat == "ddp":
                    pass  # no need to communicate if 1D DDP
                elif strat == "tp":
                    # Reduce activations after even layers.
                    activations_bytes = (
                        self.local_transformer_bs
                        * model.transformer_seq_len
                        * model.attention_dim
                        * model.bytes_per_nonemb_param
                    )
                    t_activations_c = (
                        activations_bytes
                        / system.eff_allreduce_bw[
                            -len(self.transformer_parallel) + parallel_lvl
                        ]
                    )
                    t_comp, t_comm = self.add_trace(
                        "TransformerFC{}_f_c_act_ar.".format(transformer_layer_num),
                        t_activations_c,
                        {"TransformerFC{}_f.".format(transformer_layer_num)},
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_allreduce_total += t_activations_c

        return computation_stream, communication_stream, t_comp, t_comm

    # Build backward-pass streams
    def build_bwd(
        self,
        model,
        system,
        t_comp_start,
        t_comm_start,
        freeze_emb,
        freeze_transformer,
        min_frozen_layer,
    ):
        computation_stream = []
        communication_stream = []
        t_comp = t_comp_start
        t_comm = t_comm_start

        assert (
            not freeze_emb or not freeze_transformer
        ), "need to have at least one component that needs to be trained."

        for transformer_layer_num in range(model.num_transformer_layers - 1, -1, -1):
            # --- Handle Transformer - 2) FC ---

            # Special Case: Prefetch first layer
            if transformer_layer_num == model.num_transformer_layers - 1:
                fc_layer_bytes = (
                    model.transformer_fc_layer_params
                    / self.transformer_shard_factor
                    * self.transformer_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_fc_c = (
                    fc_layer_bytes
                    / system.eff_allgather_bw[-len(self.transformer_parallel)]
                )  # FSDP is always first level of parallelism.
                t_comp, t_comm = self.add_trace(
                    "TransformerFC{}_b_c_wgt_ag.".format(transformer_layer_num),
                    t_fc_c,
                    {},
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_fc_c

            # If FSDP, need to gather weights before computation
            if self.transformer_parallel[0] == "fsdp":
                # FSDP allgather time proportional to output buffer size.
                attn_layer_bytes = (
                    model.attention_layer_params
                    / self.transformer_shard_factor
                    * self.transformer_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_attn_c = (
                    attn_layer_bytes
                    / system.eff_allgather_bw[-len(self.transformer_parallel)]
                )  # FSDP is always first level of parallelism.
                t_comp, t_comm = self.add_trace(
                    "Attn{}_b_c_wgt_ag.".format(transformer_layer_num),
                    t_attn_c,
                    {},
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_attn_c

            # Handle computation
            t_fc = 2 * self.local_transformer_fc_layer_flops / self.eff_flops
            fc_deps = {
                "Attn{}_b.".format(transformer_layer_num + 1),
                "Attn{}_b_c_act_ar.".format(transformer_layer_num + 1),
                "TransformerFC{}_b_c_wgt_ag.".format(transformer_layer_num),
            }
            t_comp, t_comm = self.add_trace(
                "TransformerFC{}_b.".format(transformer_layer_num),
                t_fc,
                fc_deps,
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_gemm_total += t_fc
            # Handle communication
            shard_factor = self.transformer_shard_factor
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.transformer_parallel, self.transformer_parallel_degree)
            ):
                if strat == "ddp":
                    if (
                        not freeze_transformer
                        and transformer_layer_num < min_frozen_layer
                    ):
                        fc_layer_bytes = (
                            model.transformer_fc_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_fc_c = (
                            fc_layer_bytes
                            / system.eff_allreduce_bw[
                                -len(self.transformer_parallel) + parallel_lvl
                            ]
                        )
                        t_comp, t_comm = self.add_trace(
                            "TransformerFC{}_b_c_wg_ar".format(transformer_layer_num),
                            t_fc_c,
                            {"TransformerFC{}_b.".format(transformer_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_allreduce_total += t_fc_c
                elif strat == "fsdp":
                    shard_factor /= deg
                    if (
                        not freeze_transformer
                        and transformer_layer_num < min_frozen_layer
                    ):
                        fc_layer_bytes = (
                            model.transformer_fc_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_fc_c = (
                            fc_layer_bytes
                            / system.eff_reducescatter_bw[
                                -len(self.transformer_parallel)
                            ]
                        )  # FSDP is always first level of parallelism.
                        t_comp, t_comm = self.add_trace(
                            "TransformerFC{}_b_c_wg_rs".format(transformer_layer_num),
                            t_fc_c,
                            {"TransformerFC{}_b.".format(transformer_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_reducescatter_total += t_fc_c
                elif strat == "tp":
                    shard_factor /= deg
                    activations_bytes = (
                        self.local_transformer_bs
                        * model.transformer_seq_len
                        * model.attention_dim
                        * model.bytes_per_nonemb_param
                    )
                    t_activations_c = (
                        activations_bytes
                        / system.eff_allreduce_bw[
                            -len(self.transformer_parallel) + parallel_lvl
                        ]
                    )
                    t_comp, t_comm = self.add_trace(
                        "TransformerFC{}_b_c_act_ar.".format(transformer_layer_num),
                        t_activations_c,
                        {"TransformerFC{}_b.".format(transformer_layer_num)},
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_allreduce_total += t_activations_c

            # --- Handle Transformer - 1) Attention ---
            # If FSDP, need to gather weights before computation
            if self.transformer_parallel[0] == "fsdp" and transformer_layer_num > 0:
                # FSDP allgather time proportional to output buffer size.
                fc_layer_bytes = (
                    model.transformer_fc_layer_params
                    / self.transformer_shard_factor
                    * self.transformer_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_fc_c = (
                    fc_layer_bytes
                    / system.eff_allgather_bw[-len(self.transformer_parallel)]
                )  # FSDP is always first level of parallelism.
                t_comp, t_comm = self.add_trace(
                    "TransformerFC{}_b_c_wgt_ag.".format(transformer_layer_num - 1),
                    t_fc_c,
                    {},
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_fc_c

            # Handle computation
            t_attn = 2 * self.local_attention_layer_flops / self.eff_flops
            attn_deps = {
                "TransformerFC{}_b.".format(transformer_layer_num),
                "TransformerFC{}_b_c_act_ar.".format(transformer_layer_num),
                "Attn{}_b_c_wgt_ag.".format(transformer_layer_num),
            }
            t_comp, t_comm = self.add_trace(
                "Attn{}_b.".format(transformer_layer_num),
                t_attn,
                attn_deps,
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_gemm_total += t_attn
            # Handle communication
            shard_factor = self.transformer_shard_factor
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.transformer_parallel, self.transformer_parallel_degree)
            ):
                if strat == "ddp":
                    if (
                        not freeze_transformer
                        and transformer_layer_num < min_frozen_layer
                    ):
                        attn_layer_bytes = (
                            model.attention_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_attn_c = (
                            attn_layer_bytes
                            / system.eff_allreduce_bw[
                                -len(self.transformer_parallel) + parallel_lvl
                            ]
                        )
                        t_comp, t_comm = self.add_trace(
                            "Attn{}_b_c_wg_ar".format(transformer_layer_num),
                            t_attn_c,
                            {"Attn{}_b.".format(transformer_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_allreduce_total += t_attn_c
                elif strat == "fsdp":
                    shard_factor /= deg
                    if (
                        not freeze_transformer
                        and transformer_layer_num < min_frozen_layer
                    ):
                        attn_layer_bytes = (
                            model.attention_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_attn_c = (
                            attn_layer_bytes
                            / system.eff_reducescatter_bw[
                                -len(self.transformer_parallel)
                            ]
                        )  # FSDP is always first level of parallelism.
                        t_comp, t_comm = self.add_trace(
                            "Attn{}_b_c_wg_rs".format(transformer_layer_num),
                            t_attn_c,
                            {"Attn{}_b.".format(transformer_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_reducescatter_total += t_attn_c
                elif strat == "tp":
                    shard_factor /= deg
                    activations_bytes = (
                        self.local_transformer_bs
                        * model.transformer_seq_len
                        * model.attention_dim
                        * model.bytes_per_nonemb_param
                    )
                    t_activations_c = (
                        activations_bytes
                        / system.eff_allreduce_bw[
                            -len(self.transformer_parallel) + parallel_lvl
                        ]
                    )
                    t_comp, t_comm = self.add_trace(
                        "Attn{}_b_c_act_ar.".format(transformer_layer_num),
                        t_activations_c,
                        {"Attn{}_b.".format(transformer_layer_num)},
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_allreduce_total += t_activations_c

        if not freeze_emb:
            # --- Handle embedding table (communication) ---
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.emb_parallel, self.emb_parallel_degree)
            ):
                if strat == "mp":
                    t_emb_c = (
                        2
                        * (self.lookup_bytes_per_device / deg)
                        / system.eff_all2all_bw[-len(self.emb_parallel) + parallel_lvl]
                    )
                    t_comp, t_comm = self.add_trace(
                        "EMB_b_c_all2all.",
                        t_emb_c,
                        {
                            "Attn{}_b.".format(transformer_layer_num),
                            "Attn{}_b_c_act_ar.".format(transformer_layer_num),
                        },
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_all2all_total += t_emb_c
                elif self.emb_parallel[0] == "ddp":
                    t_emb_c = (
                        self.lookup_bytes_per_device
                        / system.eff_allreduce_bw[
                            -len(self.emb_parallel) + parallel_lvl
                        ]
                    )
                    t_comp, t_comm = self.add_trace(
                        "EMB_b_c_ar.",
                        t_emb_c,
                        {
                            "Attn{}_b.".format(transformer_layer_num),
                            "Attn{}_b_c_act_ar.".format(transformer_layer_num),
                        },
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_allreduce_total += t_emb_c

            # --- Handle embedding table (computation) ---
            t_emb = self.lookup_bytes_per_device / system.eff_mem_bw
            t_comp, t_comm = self.add_trace(
                "EMB_b.",
                t_emb,
                {"EMB_b_c"},
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_emb_total += t_emb

        return computation_stream, communication_stream, t_comp, t_comm

    # Build inference streams
    def build_inference(self, model, system):
        f_computation_stream, f_communication_stream, f_t_comp, f_t_comm = (
            self.build_fwd(model, system)
        )
        t_end = max(
            f_t_comp, f_t_comm
        )  # change to account for overlapping forward and backward pass

        self.update_experiment_stats(t_end)
        return f_computation_stream, f_communication_stream

    # Build pre-training streams
    def build_pretrain(self, model, system):
        f_computation_stream, f_communication_stream, f_t_comp, f_t_comm = (
            self.build_fwd(model, system)
        )
        t_end = max(
            f_t_comp, f_t_comm
        )  # change to account for overlapping forward and backward pass

        b_computation_stream, b_communication_stream, b_t_comp, b_t_comm = (
            self.build_bwd(
                model,
                system,
                t_end,
                t_end,
                freeze_emb=False,
                freeze_transformer=False,
                min_frozen_layer=model.num_transformer_layers,
            )
        )
        t_end = max(b_t_comp, b_t_comm)

        self.update_experiment_stats(t_end)
        return (
            f_computation_stream + b_computation_stream,
            f_communication_stream + b_communication_stream,
        )

    # Build fine-tuning streams
    def build_finetune(self, model, system):
        f_computation_stream, f_communication_stream, f_t_comp, f_t_comm = (
            self.build_fwd(model, system)
        )
        t_end = max(
            f_t_comp, f_t_comm
        )  # change to account for overlapping forward and backward pass

        b_computation_stream, b_communication_stream, b_t_comp, b_t_comm = (
            self.build_bwd(
                model,
                system,
                t_end,
                t_end,
                self.freeze_emb,
                self.freeze_transformer,
                self.min_frozen_layer,
            )
        )
        t_end = max(b_t_comp, b_t_comm)

        self.update_experiment_stats(t_end)
        return (
            f_computation_stream + b_computation_stream,
            f_communication_stream + b_communication_stream,
        )

    # Check if the specified parallelization strategies are legal.
    def check_parallelization_strats(
        self, model, num_devices, num_intra_node_devices, restrict2d
    ):
        # Embedding Tables Parallelization Strategy Checks
        assert len(self.emb_parallel) == len(
            self.emb_parallel_degree
        ), "Mismatch in EMB parallelism specification"
        total_emb_degree = 1
        seen_strats = set()
        for strat, deg in zip(self.emb_parallel, self.emb_parallel_degree):
            assert (
                strat not in seen_strats
            ), "repeat EMB parallelization strategies are refactorable"
            seen_strats.add(strat)
            assert deg > 1, "EMB parallelization degree of 1 is redundant"
            total_emb_degree *= deg
        assert (
            total_emb_degree == num_devices
        ), "Mismatch in EMB parallelization and number of devices"

        # Transformer Layers Parallelization Strategy Checks
        assert len(self.transformer_parallel) == len(
            self.transformer_parallel_degree
        ), "Mismatch in MLP parallelism specification"
        total_transformer_degree = 1
        seen_strats = set()
        for strat, deg in zip(
            self.transformer_parallel, self.transformer_parallel_degree
        ):
            assert (
                strat not in seen_strats
            ), "repeat MLP parallelization strategies are refactorable"
            if strat == "fsdp":
                assert (
                    len(seen_strats) == 0
                ), "FSDP can only be applied as first degree of parallelism!"
            seen_strats.add(strat)
            assert deg > 1, "Transformer parallelization degree of 1 is redundant"
            total_transformer_degree *= deg
        assert (
            total_transformer_degree == num_devices
        ), "Mismatch in MLP parallelization and number of devices"

        # Restrict to 2D parallelism
        if restrict2d:
            assert (
                len(self.emb_parallel) < 3 and len(self.transformer_parallel) < 3
            ), "Only allow for up to 2D parallelism for now"
            assert (
                self.emb_parallel_degree[0] == num_intra_node_devices
                or self.emb_parallel_degree[0] == num_devices
            ), "Only allow for up to 2D parallelism for now"
            assert (
                self.transformer_parallel_degree[0] == num_intra_node_devices
                or self.transformer_parallel_degree[0] == num_devices
            ), "Only allow for up to 2D parallelism for now"

    # Get per-device memory usage
    def get_mem_usage(self, model):
        emb_cap_per_device = (
            model.emb_params * model.bytes_per_emb_param / self.emb_shard_factor
        )
        transformer_cap_per_device = (
            model.transformer_params
            * model.bytes_per_nonemb_param
            / self.transformer_shard_factor
        )

        return emb_cap_per_device, transformer_cap_per_device

    # Get duplication and sharding factors of MLPs and Embedding Tables
    def get_parallelization_factors(self):
        # Embedding Table Parameters
        emb_duplicate_factor = 1
        emb_shard_factor = 1
        for strat, deg in zip(self.emb_parallel, self.emb_parallel_degree):
            if strat == "ddp":
                emb_duplicate_factor *= deg
            elif strat in ["mp"]:
                emb_shard_factor *= deg
            else:
                sys.exit(
                    "Undefined parallelization strategy for EMBs: {}".format(strat)
                )

        # Transformer Parameters
        transformer_duplicate_factor = 1
        transformer_shard_factor = 1
        for strat, deg in zip(
            self.transformer_parallel, self.transformer_parallel_degree
        ):
            if strat == "ddp":
                transformer_duplicate_factor *= deg
            elif strat in ["tp", "pp"]:
                transformer_shard_factor *= deg
            elif (
                strat == "fsdp"
            ):  # "Duplicate" across devices but also sharded before actual computation
                transformer_duplicate_factor *= deg
                transformer_shard_factor *= deg
            else:
                sys.exit(
                    "Undefined parallelization strategy for Transformer Layers: {}".format(
                        strat
                    )
                )

        return (
            emb_duplicate_factor,
            emb_shard_factor,
            transformer_duplicate_factor,
            transformer_shard_factor,
        )

    # Get Task FLOPs
    def get_task_flops(self, model):
        # If Tensor Parallel is present, layer-level FLOPs are divided accordingly.
        transformer_tp_factor = 1
        for strat, deg in zip(
            self.transformer_parallel, self.transformer_parallel_degree
        ):
            if strat == "tp":
                transformer_tp_factor = deg
        local_attention_layer_flops = (
            model.attention_layer_flops
            * self.local_transformer_bs
            / transformer_tp_factor
        )
        global_attention_layer_flops = (
            model.attention_layer_flops * self.global_transformer_bs
        )
        local_transformer_fc_layer_flops = (
            model.transformer_fc_layer_flops
            * self.local_transformer_bs
            / transformer_tp_factor
        )
        global_transformer_fc_layer_flops = (
            model.transformer_fc_layer_flops * self.global_transformer_bs
        )

        local_total_flops = (
            local_attention_layer_flops + local_transformer_fc_layer_flops
        ) * model.num_transformer_layers
        global_total_flops = (
            global_attention_layer_flops + global_transformer_fc_layer_flops
        ) * model.num_transformer_layers

        return (
            local_attention_layer_flops,
            global_attention_layer_flops,
            local_transformer_fc_layer_flops,
            global_transformer_fc_layer_flops,
            local_total_flops,
            global_total_flops,
        )

    # Get Task Lookup Bytes
    def get_task_lookup_bytes(self, model):
        local_lookup_bytes = model.lookup_bytes * self.local_emb_bs
        global_lookup_bytes = model.lookup_bytes * self.global_emb_bs
        lookup_bytes_per_device = local_lookup_bytes / self.emb_shard_factor
        return local_lookup_bytes, global_lookup_bytes, lookup_bytes_per_device

    # Print task summary statistics
    def print_summary_stats(self):
        total_cap_per_device = self.emb_cap_per_device + self.transformer_cap_per_device

        print("**************************************************")
        super().print_summary_stats()
        if self.type == "finetune":
            print("Frozen Components:")
            print(
                "\tEMB: {}, Transformer: {} (Min Frozen Layer: {})".format(
                    self.freeze_emb, self.freeze_transformer, self.min_frozen_layer
                )
            )
        print("Task Memory Usage:")
        print(
            "\tModel Weights: {:.2f} ({:.2f} EMB, {:.2f} Transformer) GB per device.".format(
                total_cap_per_device / 1e9,
                self.emb_cap_per_device / 1e9,
                self.transformer_cap_per_device / 1e9,
            )
        )
        print("Task FLOPs:")
        print(
            "\tAttention Layer FLOPs per local batch: {:.2f} TFLOPs.".format(
                self.local_attention_layer_flops / 1e12
            )
        )
        print(
            "\tAttention Layer FLOPs per global batch: {:.2f} TFLOPs.".format(
                self.global_attention_layer_flops / 1e12
            )
        )
        print(
            "\tTransformer FC Layer FLOPs per local batch: {:.2f} TFLOPs.".format(
                self.local_transformer_fc_layer_flops / 1e12
            )
        )
        print(
            "\tTransformer FC Layer FLOPs per global batch: {:.2f} TFLOPs.".format(
                self.global_transformer_fc_layer_flops / 1e12
            )
        )
        print(
            "\tModel FLOPs per local batch: {:.2f} TFLOPs.".format(
                self.local_total_flops / 1e12
            )
        )
        print(
            "\tModel FLOPs per global batch: {:.2f} TFLOPs.".format(
                self.global_total_flops / 1e12
            )
        )
        print("Task Lookup Bytes:")
        print(
            "\tLookup bytes per local batch: {:.2f} GB".format(
                self.local_lookup_bytes / 1e9
            )
        )
        print(
            "\tLookup bytes per global batch: {:.2f} GB".format(
                self.global_lookup_bytes / 1e9
            )
        )
        print(
            "\tLookup bytes per device (per global batch): {:.2f} GB".format(
                self.lookup_bytes_per_device / 1e9
            )
        )
        print("**************************************************")
