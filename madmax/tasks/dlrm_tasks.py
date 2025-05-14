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


class DLRM_Task(Task):
    """Deep Learning Recommendation Model Task implementation."""

    def __init__(self, model, system, task_cfg):
        super().__init__(model, system, task_cfg)

        # Parallelization configuration
        self.mlp_parallel = task_cfg["mlp_parallel"]
        self.mlp_parallel_degree = task_cfg["mlp_parallel_degree"]
        self.emb_parallel = task_cfg["emb_parallel"]
        self.emb_parallel_degree = task_cfg["emb_parallel_degree"]

        # Batch size configuration
        self.local_mlp_bs = task_cfg["local_mlp_bs"]
        self.local_emb_bs = task_cfg["local_emb_bs"]

        # Fine-tuning configuration
        if self.type == "finetune":
            self.freeze_top = task_cfg["freeze_top_mlp"]
            self.freeze_bot = task_cfg["freeze_bot_mlp"]
            self.freeze_emb = task_cfg["freeze_emb"]

        # Set effective FLOPS based on parameter precision
        self._set_effective_flops(model)

        # Validate parallelization strategies
        self.check_parallelization_strats(
            model, system.num_devices, system.num_intra_node_devices, restrict2d=True
        )

        # Calculate parallelization factors
        (
            self.mlp_duplicate_factor,
            self.mlp_shard_factor,
            self.emb_duplciate_factor,
            self.emb_shard_factor,
        ) = self.get_parallelization_factors()

        # Calculate global batch sizes
        self.global_mlp_bs = self.local_mlp_bs * self.mlp_duplicate_factor
        self.global_emb_bs = self.local_emb_bs * self.emb_duplciate_factor

        assert (
            self.global_mlp_bs == self.global_emb_bs
        ), "Global MLP and EMB batch sizes must match"
        self.global_bs = self.global_mlp_bs

        # Calculate memory usage and computational requirements
        self.mlp_cap_per_device, self.emb_cap_per_device = self.get_mem_usage(model)
        (
            self.local_layer_flops,
            self.global_layer_flops,
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

        # --- Handle Bottom MLP ---
        for mlp_bot_layer_num in range(model.num_bot_mlp_layers):
            # If FSDP, need to gather weights before computation
            if self.mlp_parallel[0] == "fsdp":
                # FSDP allgather time proportional to output buffer size.
                mlp_layer_bytes = (
                    model.mlp_layer_params
                    / self.mlp_shard_factor
                    * self.mlp_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_mlp_c = (
                    mlp_layer_bytes / system.eff_allgather_bw[-len(self.mlp_parallel)]
                )  # FSDP is always first level of parallelism.
                t_comp, t_comm = self.add_trace(
                    "MLPBot{}_f_c_wgt_ag.".format(mlp_bot_layer_num),
                    t_mlp_c,
                    {"MLPBot{}_f".format(mlp_bot_layer_num - 2)},
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_mlp_c
            # Handle computation
            t_mlp = self.local_layer_flops / self.eff_flops
            if mlp_bot_layer_num == 0:
                mlp_deps = {"MLPBot{}_f_c_wgt_ag.".format(mlp_bot_layer_num)}
            else:
                mlp_deps = {
                    "MLPBot{}_f.".format(mlp_bot_layer_num - 1),
                    "MLPBot{}_f_c_act_ar.".format(mlp_bot_layer_num - 1),
                    "MLPBot{}_f_c_wgt_ag.".format(mlp_bot_layer_num),
                }
            t_comp, t_comm = self.add_trace(
                "MLPBot{}_f.".format(mlp_bot_layer_num),
                t_mlp,
                mlp_deps,
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_gemm_total += t_mlp
            # Handle communication
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.mlp_parallel, self.mlp_parallel_degree)
            ):
                if strat == "ddp":
                    pass  # no need to communicate if 1D DDP
                elif strat == "tp":
                    # Reduce activations after even layers.
                    if mlp_bot_layer_num % 2 == 1:
                        activations_bytes = (
                            self.local_mlp_bs
                            * model.mlp_dim
                            * model.bytes_per_nonemb_param
                        )
                        t_activations_c = (
                            activations_bytes
                            / system.eff_allreduce_bw[
                                -len(self.mlp_parallel) + parallel_lvl
                            ]
                        )
                        t_comp, t_comm = self.add_trace(
                            "MLPBot{}_f_c_act_ar.".format(mlp_bot_layer_num),
                            t_activations_c,
                            {"MLPBot{}_f.".format(mlp_bot_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_allreduce_total += t_activations_c

        # --- Handle Top MLP ---
        for mlp_top_layer_num in range(model.num_top_mlp_layers):
            # If FSDP, need to gather weights before computation
            if self.mlp_parallel[0] == "fsdp":
                # FSDP allgather time proportional to output buffer size.
                mlp_layer_bytes = (
                    model.mlp_layer_params
                    / self.mlp_shard_factor
                    * self.mlp_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_mlp_c = (
                    mlp_layer_bytes / system.eff_allgather_bw[-len(self.mlp_parallel)]
                )  # FSDP is always first level of parallelism.
                if mlp_top_layer_num == 0:
                    mlp_deps = {"MLPBot{}_f.".format(mlp_bot_layer_num - 1)}
                elif mlp_top_layer_num == 1:
                    mlp_deps = {"MLPBot{}_f.".format(mlp_bot_layer_num)}
                else:
                    mlp_deps = {"MLPTop{}_f.".format(mlp_top_layer_num - 2)}
                t_comp, t_comm = self.add_trace(
                    "MLPTop{}_f_c_wgt_ag.".format(mlp_top_layer_num),
                    t_mlp_c,
                    mlp_deps,
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_mlp_c
            # Handle computation
            t_mlp = self.local_layer_flops / self.eff_flops
            if mlp_top_layer_num == 0:
                mlp_deps = {
                    "EMB_f_c",
                    "MLPBot{}_f.".format(mlp_bot_layer_num),
                    "MLPBot{}_f_c_act_ar.".format(mlp_bot_layer_num),
                    "MLPTop{}_f_c_wgt_ag.".format(mlp_top_layer_num),
                }
            else:
                mlp_deps = {
                    "MLPTop{}_f.".format(mlp_top_layer_num - 1),
                    "MLPTop{}_f_c_act_ar.".format(mlp_top_layer_num - 1),
                    "MLPTop{}_f_c_wgt_ag.".format(mlp_top_layer_num),
                }
            t_comp, t_comm = self.add_trace(
                "MLPTop{}_f.".format(mlp_top_layer_num),
                t_mlp,
                mlp_deps,
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_gemm_total += t_mlp
            # Handle communication
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.mlp_parallel, self.mlp_parallel_degree)
            ):
                if strat == "ddp":
                    pass  # no need to communicate if 1D DDP
                elif strat == "tp":
                    # Reduce activations after even layers.
                    if mlp_top_layer_num % 2 == 1:
                        activations_bytes = (
                            self.local_mlp_bs
                            * model.mlp_dim
                            * model.bytes_per_nonemb_param
                        )
                        t_activations_c = (
                            activations_bytes
                            / system.eff_allreduce_bw[
                                -len(self.mlp_parallel) + parallel_lvl
                            ]
                        )
                        t_comp, t_comm = self.add_trace(
                            "MLPTop{}_f_c_act_ar.".format(mlp_top_layer_num),
                            t_activations_c,
                            {"MLPTop{}_f.".format(mlp_top_layer_num)},
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
        freeze_top_mlp,
        freeze_bot_mlp,
        freeze_emb,
    ):
        computation_stream = []
        communication_stream = []
        t_comp = t_comp_start
        t_comm = t_comm_start

        assert (
            not freeze_top_mlp or not freeze_bot_mlp or not freeze_emb
        ), "need to have at least one component that needs to be trained."

        # --- Handle Top MLP ---
        for mlp_top_layer_num in range(model.num_top_mlp_layers - 1, -1, -1):
            # If FSDP, need to gather weights before computation
            if self.mlp_parallel[0] == "fsdp":
                # FSDP allgather time proportional to output buffer size.
                mlp_layer_bytes = (
                    model.mlp_layer_params
                    / self.mlp_shard_factor
                    * self.mlp_parallel_degree[0]
                ) * model.bytes_per_nonemb_param
                t_mlp_c = (
                    mlp_layer_bytes / system.eff_allgather_bw[-len(self.mlp_parallel)]
                )  # FSDP is always first level of parallelism.
                t_comp, t_comm = self.add_trace(
                    "MLPTop{}_b_c_wgt_ag.".format(mlp_top_layer_num),
                    t_mlp_c,
                    {"MLPTop{}_b.".format(mlp_top_layer_num + 2)},
                    "comm",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_allgather_total += t_mlp_c
            # Handle computation
            t_mlp = 2 * self.local_layer_flops / self.eff_flops
            if mlp_top_layer_num == model.num_top_mlp_layers - 1:
                mlp_deps = {"MLPTop{}_b_c_wgt_ag.".format(mlp_top_layer_num)}
            else:
                mlp_deps = {
                    "MLPTop{}_b.".format(mlp_top_layer_num + 1),
                    "MLPTop{}_b_c_act_ar.".format(mlp_top_layer_num + 1),
                    "MLPTop{}_b_c_wgt_ag.".format(mlp_top_layer_num),
                }
            t_comp, t_comm = self.add_trace(
                "MLPTop{}_b.".format(mlp_top_layer_num),
                t_mlp,
                mlp_deps,
                "comp",
                computation_stream,
                communication_stream,
                t_comp,
                t_comm,
            )
            self.t_gemm_total += t_mlp
            # Handle communication
            shard_factor = self.mlp_shard_factor
            for parallel_lvl, (strat, deg) in enumerate(
                zip(self.mlp_parallel, self.mlp_parallel_degree)
            ):
                if strat == "ddp":
                    if not freeze_top_mlp:
                        mlp_layer_bytes = (
                            model.mlp_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_mlp_c = (
                            mlp_layer_bytes
                            / system.eff_allreduce_bw[
                                -len(self.mlp_parallel) + parallel_lvl
                            ]
                        )
                        t_comp, t_comm = self.add_trace(
                            "MLPTop{}_b_c_wg_ar".format(mlp_top_layer_num),
                            t_mlp_c,
                            {"MLPTop{}_b.".format(mlp_top_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_allreduce_total += t_mlp_c
                elif strat == "fsdp":
                    shard_factor /= deg
                    if not freeze_top_mlp:
                        mlp_layer_bytes = (
                            model.mlp_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_mlp_c = (
                            mlp_layer_bytes
                            / system.eff_reducescatter_bw[-len(self.mlp_parallel)]
                        )  # FSDP is always first level of parallelism.
                        t_comp, t_comm = self.add_trace(
                            "MLPTop{}_b_c_wg_rs.".format(mlp_top_layer_num),
                            t_mlp_c,
                            {"MLPTop{}_b.".format(mlp_top_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_reducescatter_total += t_mlp_c
                elif strat == "tp":
                    shard_factor /= deg
                    # Reduce activations after odd layers.
                    if mlp_top_layer_num % 2 == 0:
                        activations_bytes = (
                            self.local_mlp_bs
                            * model.mlp_dim
                            * model.bytes_per_nonemb_param
                        )
                        t_activations_c = (
                            activations_bytes
                            / system.eff_allreduce_bw[
                                -len(self.mlp_parallel) + parallel_lvl
                            ]
                        )
                        t_comp, t_comm = self.add_trace(
                            "MLPTop{}_b_c_act_ar.".format(mlp_top_layer_num),
                            t_activations_c,
                            {"MLPTop{}_b.".format(mlp_top_layer_num)},
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
                            "MLPTop{}_b.".format(mlp_top_layer_num),
                            "MLPTop{}_b_c_act_ar.".format(mlp_top_layer_num),
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
                            "MLPTop{}_b.".format(mlp_top_layer_num),
                            "MLPTop{}_b_c_act_ar.".format(mlp_top_layer_num),
                        },
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_allreduce_total += t_emb_c

        if not freeze_bot_mlp:
            # --- Handle Bot MLP ---
            for mlp_bot_layer_num in range(model.num_bot_mlp_layers - 1, -1, -1):
                # If FSDP, need to gather weights before computation
                if self.mlp_parallel[0] == "fsdp":
                    # FSDP allgather time proportional to output buffer size.
                    mlp_layer_bytes = (
                        model.mlp_layer_params
                        / self.mlp_shard_factor
                        * self.mlp_parallel_degree[0]
                    ) * model.bytes_per_nonemb_param
                    t_mlp_c = (
                        mlp_layer_bytes
                        / system.eff_allgather_bw[-len(self.mlp_parallel)]
                    )  # FSDP is always first level of parallelism.
                    if mlp_bot_layer_num == model.num_bot_mlp_layers - 1:
                        mlp_deps = {"MLPTop1_b."}
                    elif mlp_top_layer_num == model.num_bot_mlp_layers - 2:
                        mlp_deps = {"MLPTop0_b.".format(mlp_bot_layer_num)}
                    else:
                        mlp_deps = {"MLPBot{}_b".format(mlp_bot_layer_num + 2)}
                    t_comp, t_comm = self.add_trace(
                        "MLPBot{}_b_c_wgt_ag.".format(mlp_bot_layer_num),
                        t_mlp_c,
                        {"MLPBot{}_b".format(mlp_bot_layer_num + 2)},
                        "comm",
                        computation_stream,
                        communication_stream,
                        t_comp,
                        t_comm,
                    )
                    self.t_allgather_total += t_mlp_c
                # Handle computation
                t_mlp = 2 * self.local_layer_flops / self.eff_flops
                if mlp_bot_layer_num == model.num_bot_mlp_layers - 1:
                    mlp_deps = {
                        "MLPTop{}_b.".format(mlp_top_layer_num),
                        "MLPTop{}_b_c_act_ar.".format(mlp_top_layer_num),
                        "MLPBot{}_b_c_wgt_ag.".format(mlp_bot_layer_num),
                    }
                else:
                    mlp_deps = {
                        "MLPBot{}_b.".format(mlp_bot_layer_num + 1),
                        "MLPBot{}_b_c_act_ar.".format(mlp_bot_layer_num + 1),
                        "MLPBot{}_b_c_wgt_ag.".format(mlp_bot_layer_num),
                    }
                t_comp, t_comm = self.add_trace(
                    "MLPBot{}_b.".format(mlp_bot_layer_num),
                    t_mlp,
                    mlp_deps,
                    "comp",
                    computation_stream,
                    communication_stream,
                    t_comp,
                    t_comm,
                )
                self.t_gemm_total += t_mlp
                # Handle communication
                shard_factor = self.mlp_shard_factor
                for parallel_lvl, (strat, deg) in enumerate(
                    zip(self.mlp_parallel, self.mlp_parallel_degree)
                ):
                    if strat == "ddp":
                        mlp_layer_bytes = (
                            model.mlp_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_mlp_c = (
                            mlp_layer_bytes
                            / system.eff_allreduce_bw[
                                -len(self.mlp_parallel) + parallel_lvl
                            ]
                        )
                        t_comp, t_comm = self.add_trace(
                            "MLPBot{}_b_c_wg_ar".format(mlp_bot_layer_num),
                            t_mlp_c,
                            {"MLPBot{}_b.".format(mlp_bot_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_allreduce_total += t_mlp_c
                    elif strat == "fsdp":
                        shard_factor /= deg
                        mlp_layer_bytes = (
                            model.mlp_layer_params / shard_factor
                        ) * model.bytes_per_nonemb_param
                        t_mlp_c = (
                            mlp_layer_bytes
                            / system.eff_reducescatter_bw[-len(self.mlp_parallel)]
                        )  # FSDP is always first level of parallelism.
                        t_comp, t_comm = self.add_trace(
                            "MLPBot{}_b_c_wg_rs.".format(mlp_bot_layer_num),
                            t_mlp_c,
                            {"MLPBot{}_b.".format(mlp_bot_layer_num)},
                            "comm",
                            computation_stream,
                            communication_stream,
                            t_comp,
                            t_comm,
                        )
                        self.t_reducescatter_total += t_mlp_c
                    elif strat == "tp":
                        shard_factor /= deg
                        # Reduce activations after odd layers.
                        if mlp_bot_layer_num % 2 == 0:
                            activations_bytes = (
                                self.local_mlp_bs
                                * model.mlp_dim
                                * model.bytes_per_nonemb_param
                            )
                            t_activations_c = (
                                activations_bytes
                                / system.eff_allreduce_bw[
                                    -len(self.mlp_parallel) + parallel_lvl
                                ]
                            )
                            t_comp, t_comm = self.add_trace(
                                "MLPBot{}_b_c_act_ar.".format(mlp_bot_layer_num),
                                t_activations_c,
                                {"MLPBot{}_b.".format(mlp_bot_layer_num)},
                                "comm",
                                computation_stream,
                                communication_stream,
                                t_comp,
                                t_comm,
                            )
                            self.t_allreduce_total += t_activations_c

        if not freeze_emb:
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
                freeze_top_mlp=False,
                freeze_bot_mlp=False,
                freeze_emb=False,
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
                self.freeze_top,
                self.freeze_bot,
                self.freeze_emb,
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
        # MLPs Parallelization Strategy Checks
        assert len(self.mlp_parallel) == len(
            self.mlp_parallel_degree
        ), "Mismatch in MLP parallelism specification"
        total_mlp_degree = 1
        seen_strats = set()
        for strat, deg in zip(self.mlp_parallel, self.mlp_parallel_degree):
            assert (
                strat not in seen_strats
            ), "repeat MLP parallelization strategies are refactorable"
            if strat == "fsdp":
                assert (
                    len(seen_strats) == 0
                ), "FSDP can only be applied as first degree of parallelism!"
            if strat == "tp":
                assert (
                    model.num_bot_mlp_layers % 2 == 0
                    and model.num_top_mlp_layers % 2 == 0
                )
            seen_strats.add(strat)
            assert deg > 1, "MLP parallelization degree of 1 is redundant"
            total_mlp_degree *= deg
        assert (
            total_mlp_degree == num_devices
        ), "Mismatch in MLP parallelization and number of devices"

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

        # Restrict to 2D parallelism
        if restrict2d:
            assert (
                len(self.mlp_parallel) < 3 and len(self.emb_parallel) < 3
            ), "Only allow for up to 2D parallelism for now"
            assert (
                self.mlp_parallel_degree[0] == num_intra_node_devices
                or self.mlp_parallel_degree[0] == num_devices
            ), "Only allow for up to 2D parallelism for now"
            assert (
                self.emb_parallel_degree[0] == num_intra_node_devices
                or self.emb_parallel_degree[0] == num_devices
            ), "Only allow for up to 2D parallelism for now"

    # Get per-device memory usage
    def get_mem_usage(self, model):
        mlp_cap_per_device = (
            model.mlp_params * model.bytes_per_nonemb_param / self.mlp_shard_factor
        )
        emb_cap_per_device = (
            model.emb_params * model.bytes_per_emb_param / self.emb_shard_factor
        )

        return mlp_cap_per_device, emb_cap_per_device

    # Get duplication and sharding factors of MLPs and Embedding Tables
    def get_parallelization_factors(self):
        # MLP Parameters
        mlp_duplicate_factor = 1
        mlp_shard_factor = 1
        for strat, deg in zip(self.mlp_parallel, self.mlp_parallel_degree):
            if strat == "ddp":
                mlp_duplicate_factor *= deg
            elif strat in ["tp", "pp"]:
                mlp_shard_factor *= deg
            elif (
                strat == "fsdp"
            ):  # "Duplicate" across devices but also sharded before actual computation
                mlp_duplicate_factor *= deg
                mlp_shard_factor *= deg
            else:
                sys.exit(
                    "Undefined parallelization strategy for MLPs: {}".format(strat)
                )

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

        return (
            mlp_duplicate_factor,
            mlp_shard_factor,
            emb_duplicate_factor,
            emb_shard_factor,
        )

    # Get Task FLOPs
    def get_task_flops(self, model):
        # If Tensor Parallel is present, layer-level FLOPs are divided accordingly.
        tp_factor = 1
        for strat, deg in zip(self.mlp_parallel, self.mlp_parallel_degree):
            if strat == "tp":
                tp_factor = deg
        local_layer_flops = model.layer_flops * self.local_mlp_bs / tp_factor
        global_layer_flops = model.layer_flops * self.global_mlp_bs
        local_total_flops = model.total_flops * self.local_mlp_bs / tp_factor
        global_total_flops = model.total_flops * self.global_mlp_bs
        return (
            local_layer_flops,
            global_layer_flops,
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
        total_cap_per_device = self.mlp_cap_per_device + self.emb_cap_per_device

        print("**************************************************")
        super().print_summary_stats()
        if self.type == "finetune":
            print("Frozen Components:")
            print(
                "\tTop MLP: {}, Bot MLP: {}, EMB: {}".format(
                    self.freeze_top, self.freeze_bot, self.freeze_emb
                )
            )
        print("Task Memory Usage:")
        print(
            "\tModel Weights: {:.2f} ({:.2f} MLP, {:.2f} EMB) GB per device.".format(
                total_cap_per_device / 1e9,
                self.mlp_cap_per_device / 1e9,
                self.emb_cap_per_device / 1e9,
            )
        )
        print("Task FLOPs:")
        print(
            "\tMLP Layer FLOPs per local batch: {:.2f} TFLOPs.".format(
                self.local_layer_flops / 1e12
            )
        )
        print(
            "\tMLP Layer FLOPs per global batch: {:.2f} TFLOPs.".format(
                self.global_layer_flops / 1e12
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
