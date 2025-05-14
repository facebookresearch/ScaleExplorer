"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from systems.system import System


class GPUs(System):
    # Get effective unidirectional bandwidths
    def get_eff_bw(self):
        eff_intra_bw = self.num_nvlinks * (self.bw_per_nvlink / 2) * self.nvlink_util
        eff_inter_bw = (
            self.num_ibroce_ports
            * (self.ibroce_bw / 2)
            * self.ibroce_util
            / self.num_intra_node_devices
        )
        return eff_intra_bw, eff_inter_bw

    # Get effective collective-specific bandwidths
    def get_eff_comm_bw(self):
        if self.num_nodes > 1:
            eff_all2all_bw = [self.eff_intra_bw, self.eff_inter_bw]
            eff_allreduce_bw = [
                self.eff_intra_bw,
                self.ar_perc_inter * self.eff_inter_bw
                + self.ar_perc_intra * self.eff_intra_bw,
            ]
            eff_allgather_bw = [bw * 2 for bw in eff_allreduce_bw]
            eff_reducescatter_bw = [bw * 2 for bw in eff_allreduce_bw]
        else:
            assert (
                self.ar_perc_intra == 1.0
            ), "For one node, only use intra-node networking."

            eff_all2all_bw = [self.eff_intra_bw]
            eff_allreduce_bw = [
                self.ar_perc_inter * self.eff_inter_bw
                + self.ar_perc_intra * self.eff_intra_bw
            ]
            eff_allgather_bw = [bw * 2 for bw in eff_allreduce_bw]
            eff_reducescatter_bw = [bw * 2 for bw in eff_allreduce_bw]
        return eff_all2all_bw, eff_allreduce_bw, eff_allgather_bw, eff_reducescatter_bw

    # Print system summary statistics
    def print_summary_stats(self):
        eff_intra_bw_gb = self.eff_intra_bw / 1e9
        eff_inter_bw_gb = self.eff_inter_bw / 1e9

        eff_all2all_bw_gb = self.eff_all2all_bw[-1] / 1e9
        eff_allreduce_bw_gb = self.eff_allreduce_bw[-1] / 1e9
        eff_allgather_bw_gb = self.eff_allgather_bw[-1] / 1e9
        eff_reducescatter_bw_gb = self.eff_reducescatter_bw[-1] / 1e9

        print("**************************************************")
        super().print_summary_stats()
        print("Effective Unidirectional BW per GPU:")
        print("\tIntra-Node: {:.2f} GB/s".format(eff_intra_bw_gb))
        print("\tInter-Node: {:.2f} GB/s".format(eff_inter_bw_gb))
        print("Effective Communication Collectives BW:")
        print("\tAll to All: {:.2f} GB/s".format(eff_all2all_bw_gb))
        print("\tAll Reduce: {:.2f} GB/s".format(eff_allreduce_bw_gb))
        print("\tAll Gather: {:.2f} GB/s".format(eff_allgather_bw_gb))
        print("\tReduce Scatter: {:.2f} GB/s".format(eff_reducescatter_bw_gb))
        print("**************************************************")

    def __init__(self, system_cfg):
        super().__init__(system_cfg)

        self.num_nvlinks = system_cfg["num_nvlinks"]
        self.bw_per_nvlink = system_cfg["bw_per_nvlink"]
        self.nvlink_util = system_cfg["nvlink_util"]

        self.num_ibroce_ports = system_cfg["num_ibroce_ports"]
        self.ibroce_bw = system_cfg["ibroce_bw"]  # in bytes per second
        self.ibroce_util = system_cfg["ibroce_util"]

        self.ar_perc_intra = system_cfg["ar_perc_intra"]
        self.ar_perc_inter = 1 - self.ar_perc_intra

        self.eff_intra_bw, self.eff_inter_bw = self.get_eff_bw()
        (
            self.eff_all2all_bw,
            self.eff_allreduce_bw,
            self.eff_allgather_bw,
            self.eff_reducescatter_bw,
        ) = self.get_eff_comm_bw()

        self.print_summary_stats()
