import sys

from tasks.trace import Trace


# General Task (Pre-training, Fine-tuning, Inference)
class Task:
    def __init__(self, model, system, task_cfg):
        self.model = model
        self.system = system
        self.name = task_cfg["name"]
        self.type = task_cfg["type"]
        self.global_bs = 0

        # Initialize timing metrics
        self.t_emb_total = 0
        self.t_gemm_total = 0
        self.t_all2all_total = 0
        self.t_allreduce_total = 0
        self.t_allgather_total = 0
        self.t_reducescatter_total = 0
        self.exposed_comms = 0
        self.overlapped_comms = 0
        self.throughput = 0

    def add_trace(
        self,
        trace_name,
        t_trace,
        trace_deps,
        stream_type,
        computation_stream,
        communication_stream,
        t_comp,
        t_comm,
    ):
        """Add trace to respective stream and update timing."""
        # Find out earliest entry point for trace
        if stream_type == "comp":
            t_start = t_comp
            search_stream = communication_stream  # stream to search for dependencies
            target_stream = computation_stream  # stream to add trace to
        elif stream_type == "comm":
            t_start = t_comm
            search_stream = computation_stream
            target_stream = communication_stream
        else:
            sys.exit(f"Stream type {stream_type} not supported")
        for dep in trace_deps:
            for trace in search_stream:
                if dep in trace["name"]:
                    t_start = max(trace["t_end"], t_start)

        # Create and add trace
        op_trace = Trace(trace_name, t_trace, t_start, trace_deps).trace
        target_stream.append(op_trace)

        # Update home stream's latest checkpoint
        if stream_type == "comp":
            t_comp = op_trace["t_end"]
        elif stream_type == "comm":
            t_comm = op_trace["t_end"]

        return t_comp, t_comm

    # Print task summary statistics
    def print_summary_stats(self):
        """Print basic task summary statistics."""
        print("Task Type: {}".format(self.type))

    def update_experiment_stats(self, t_end):
        """Calculate final statistics and print detailed report."""

        total_comm_time = (
            self.t_all2all_total
            + self.t_allreduce_total
            + self.t_allgather_total
            + self.t_reducescatter_total
        )
        total_compute_time = self.t_gemm_total + self.t_emb_total
        self.exposed_comms = t_end - total_compute_time
        self.overlapped_comms = total_comm_time - self.exposed_comms
        self.throughput = (1 / t_end) * self.global_bs

        print("**************************************************")
        print("Aggregate Compute Times [ms]:")
        print("\tGEMM: {:.2f}".format(self.t_gemm_total * 1000))
        print("\tEMB: {:.2f}".format(self.t_emb_total * 1000))

        print("Aggregate Communication Times [ms]:")
        print("\tAll-to-All: {:.2f}".format(self.t_all2all_total * 1000))
        print("\tAllReduce: {:.2f}".format(self.t_allreduce_total * 1000))
        print("\tAllGather: {:.2f}".format(self.t_allgather_total * 1000))
        print("\tReduceScatter: {:.2f}".format(self.t_reducescatter_total * 1000))

        print("Communication Overlap Breakdown [ms]:")
        total_comm = self.overlapped_comms + self.exposed_comms

        if total_comm > 0:
            overlap_percent = 100 * self.overlapped_comms / total_comm
            exposed_percent = 100 * self.exposed_comms / total_comm
            print(
                f"\tExposed Communication: {self.exposed_comms*1000:.2f} ({exposed_percent:.2f} %)"
            )
            print(
                f"\tOverlapped Communication: {self.overlapped_comms*1000:.2f} ({overlap_percent:.2f} %)"
            )
        else:
            print("\tExposed Communication: 0 (0 %)")
            print("\tOverlapped Communication: 0 (0 %)")
        print("Task Iteration Time [ms]: {:.2f}".format(t_end * 1e3))
        if self.throughput / 1e6 > 0.1:
            print("Task Throughput: {:.2f} MQPS".format(self.throughput / 1e6))
        else:
            print("Task Throughput: {:.2f} QPS".format(self.throughput))
        print("**************************************************")
