from typing import Any, Dict, Tuple


# Distributed System
class System:
    """Base class for distributed computing systems."""

    def __init__(self, system_cfg: Dict[str, Any]):
        """Initialize system with configuration parameters."""
        self.name = system_cfg["name"]
        self.num_devices = system_cfg["num_devices"]
        self.num_nodes = system_cfg["num_nodes"]
        assert (
            self.num_devices % self.num_nodes == 0
        ), "Number of devices must be evenly divisible by number of nodes!"
        self.num_intra_node_devices = int(self.num_devices / self.num_nodes)

        # FLOPS configuration
        self.f64_flops = system_cfg["f64_flops"]
        self.f32_flops = system_cfg["f32_flops"]
        self.f16_flops = system_cfg["f16_flops"]
        self.i8_ops = system_cfg["i8_ops"]
        self.flops_util = system_cfg["flops_util"]
        self.eff_f64_flops, self.eff_f32_flops, self.eff_f16_flops, self.eff_i8_ops = (
            self.get_eff_flops()
        )

        # Memory configuration
        self.mem_cap = system_cfg["mem_cap"]
        self.mem_bw = system_cfg["mem_bw"]
        self.mem_bw_util = system_cfg["mem_bw_util"]
        self.eff_mem_bw = self.get_eff_mem_bw()

    def get_eff_flops(self) -> Tuple[float, float, float, float]:
        """Calculate effective FLOPS based on utilization.

        Returns:
            Tuple of (FP64 FLOPS, FP32 FLOPS, FP16 FLOPS, INT8 OPS)
        """
        eff_f64_flops = self.f64_flops * self.flops_util
        eff_f32_flops = self.f32_flops * self.flops_util
        eff_f16_flops = self.f16_flops * self.flops_util
        eff_i8_ops = self.i8_ops * self.flops_util
        return eff_f64_flops, eff_f32_flops, eff_f16_flops, eff_i8_ops

    def get_eff_mem_bw(self) -> float:
        """Calculate effective memory bandwidth based on utilization.

        Returns:
            Effective memory bandwidth
        """
        eff_mem_bw = self.mem_bw * self.mem_bw_util
        return eff_mem_bw

    def print_summary_stats(self) -> None:
        """Print system summary statistics."""
        eff_f64_tflops = self.eff_f64_flops / 1e12
        eff_f32_tflops = self.eff_f32_flops / 1e12
        eff_f16_tflops = self.eff_f16_flops / 1e12
        eff_i8_tops = self.eff_i8_ops / 1e12

        mem_cap_gb = self.mem_cap / 1e9
        mem_bw_gbps = self.mem_bw / 1e9

        print("System Name: {}".format(self.name))
        print(
            "{} nodes with {} devices each".format(
                self.num_nodes, self.num_intra_node_devices
            )
        )
        print("Effective FLOPs:")
        print(
            "\tFP64: {:.2f} TFLOPS per device / {:.2f} PFLOPS system-wide".format(
                eff_f64_tflops, eff_f64_tflops * self.num_devices / 1000
            )
        )
        print(
            "\tFP/TF32: {:.2f} TFLOPS per device / {:.2f} PFLOPS system-wide".format(
                eff_f32_tflops, eff_f32_tflops * self.num_devices / 1000
            )
        )
        print(
            "\tFP/BF16: {:.2f} TFLOPS per device / {:.2f} PFLOPS system-wide".format(
                eff_f16_tflops, eff_f16_tflops * self.num_devices / 1000
            )
        )
        print(
            "\tINT8: {:.2f} TOPS per device / {:.2f} POPS system-wide".format(
                eff_i8_tops, eff_i8_tops * self.num_devices / 1000
            )
        )
        print("Memory:")
        print(
            "\tCapacity: {:.2f} GB per device / {:.2f} TB system-wide".format(
                mem_cap_gb, mem_cap_gb * self.num_devices / 1000
            )
        )
        print(
            "\tBandwidth: {:.2f} GB/s per device / {:.2f} TB/s system-wide".format(
                mem_bw_gbps, mem_bw_gbps * self.num_devices / 1000
            )
        )
