# Imports
import os

from matplotlib import pyplot as plt

# Consistent coloring
COLORS = {
    "gemm": "tab:blue",
    "emb": "tab:orange",
    "all2all": "tab:green",
    "allreduce": "tab:red",
    "allgather": "tab:purple",
    "reducescatter": "tab:brown",
    "exposed": "tab:pink",
}


def plot_overall_results(task, figures_dir):

    filename = os.path.join(figures_dir, "overall.png")
    plt.figure(figsize=(6.5, 6.5))

    # Serialized results
    t_accum = 0
    operations = [
        ("GEMM", task.t_gemm_total, COLORS["gemm"]),
        ("EMB", task.t_emb_total, COLORS["emb"]),
        ("All2All", task.t_all2all_total, COLORS["all2all"]),
        ("AllReduce", task.t_allreduce_total, COLORS["allreduce"]),
        ("AllGather", task.t_allgather_total, COLORS["allgather"]),
        ("ReduceScatter", task.t_reducescatter_total, COLORS["reducescatter"]),
    ]

    for label, duration, color in operations:
        plt.bar(
            0,
            duration * 1e3,
            width=0.8,
            bottom=t_accum,
            color=color,
            edgecolor="black",
            label=label,
        )
        t_accum += duration * 1e3

    # Overlapped results
    t_accum = 0
    overlapped_ops = [
        ("GEMM", task.t_gemm_total, COLORS["gemm"]),
        ("EMB", task.t_emb_total, COLORS["emb"]),
        ("Exposed Comm.", task.exposed_comms, COLORS["exposed"]),
    ]

    for label, duration, color in overlapped_ops:
        plt.bar(
            1,
            duration * 1e3,
            width=0.8,
            bottom=t_accum,
            color=color,
            edgecolor="black",
            label=None if label != "Exposed Comm." else label,
        )
        t_accum += duration * 1e3

    plt.xlim([-0.5, 1.5])
    plt.xticks([0, 1], ["Serialized", "Overlapped"], fontsize=14)
    plt.ylabel("Execution Time [ms]", fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(filename)


def plot_timeline(computation_stream, communication_stream, figures_dir):

    filename = os.path.join(figures_dir, "timeline.png")
    plt.figure(figsize=(15, 5), dpi=300)

    label_set = set()

    # Mapping for operation types
    comp_mapping = {
        "EMB": ("EMB", COLORS["emb"]),
        "MLP": ("GEMM", COLORS["gemm"]),
        "Attn": ("GEMM", COLORS["gemm"]),
        "FC": ("GEMM", COLORS["gemm"]),
    }

    comm_mapping = {
        "all2all": ("All2All", COLORS["all2all"]),
        "ar": ("AllReduce", COLORS["allreduce"]),
        "ag": ("AllGather", COLORS["allgather"]),
        "rs": ("ReduceScatter", COLORS["reducescatter"]),
    }

    # Plot computation stream
    for trace in computation_stream:
        lbl, clr = None, None
        for key, (label, color) in comp_mapping.items():
            if key in trace["name"]:
                lbl, clr = label, color
                break

        if lbl and lbl not in label_set:
            label_set.add(lbl)
        else:
            lbl = None

        plt.barh(
            1,
            trace["duration"] * 1e3,
            height=0.8,
            left=trace["t_start"] * 1e3,
            color=clr,
            edgecolor="black",
            label=lbl,
        )

    # Plot communication stream
    for trace in communication_stream:
        lbl, clr = None, None
        for key, (label, color) in comm_mapping.items():
            if key in trace["name"]:
                lbl, clr = label, color
                break

        if lbl and lbl not in label_set:
            label_set.add(lbl)
        else:
            lbl = None

        plt.barh(
            0,
            trace["duration"] * 1e3,
            height=0.8,
            left=trace["t_start"] * 1e3,
            color=clr,
            edgecolor="black",
            label=lbl,
        )

    plt.ylim([-0.5, 1.5])
    plt.yticks(
        [0, 1], ["Communication", "Computation"], fontsize=14, rotation=90, va="center"
    )
    plt.xlabel("Execution Time [ms]", fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(filename)
