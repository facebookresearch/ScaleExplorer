# Imports
from matplotlib import pyplot as plt

# Consistent coloring
clr_gemm = 'tab:blue'
clr_emb = 'tab:orange'
clr_all2all = 'tab:green'
clr_allreduce = 'tab:red'
clr_allgather = 'tab:purple'
clr_reducescatter = 'tab:brown'
clr_exposed = 'tab:pink'

def plot_overall_results(task, figures_dir):

    filename = figures_dir + 'overall.png'
    plt.figure(figsize=(6.5, 6.5))

    # Serialized results
    t_accum = 0
    plt.bar(0, task.t_gemm_total*1e3, width = 0.8, bottom = t_accum, color = clr_gemm, edgecolor = 'black', label = 'GEMM')
    t_accum += task.t_gemm_total*1e3
    plt.bar(0, task.t_emb_total*1e3, width = 0.8, bottom = t_accum, color = clr_emb, edgecolor = 'black', label = 'EMB')
    t_accum += task.t_emb_total*1e3
    plt.bar(0, task.t_all2all_total*1e3, width = 0.8, bottom = t_accum, color = clr_all2all, edgecolor = 'black', label = 'All2All')
    t_accum += task.t_all2all_total*1e3
    plt.bar(0, task.t_allreduce_total*1e3, width = 0.8, bottom = t_accum, color = clr_allreduce, edgecolor = 'black', label = 'AllReduce')
    t_accum += task.t_allreduce_total*1e3
    plt.bar(0, task.t_allgather_total*1e3, width = 0.8, bottom = t_accum, color = clr_allgather, edgecolor = 'black', label = 'AllGather')
    t_accum += task.t_allgather_total*1e3
    plt.bar(0, task.t_reducescatter_total*1e3, width = 0.8, bottom = t_accum, color = clr_reducescatter, edgecolor = 'black', label = 'ReduceScatter')
    t_accum += task.t_reducescatter_total*1e3

    # Exposed results
    t_accum = 0
    plt.bar(1, task.t_gemm_total*1e3, width = 0.8, bottom = t_accum, color = clr_gemm, edgecolor = 'black')
    t_accum += task.t_gemm_total*1e3
    plt.bar(1, task.t_emb_total*1e3, width = 0.8, bottom = t_accum, color = clr_emb, edgecolor = 'black')
    t_accum += task.t_emb_total*1e3
    plt.bar(1, task.exposed_comms*1e3, width = 0.8, bottom = t_accum, color = clr_exposed, edgecolor = 'black', label = 'Exposed Comm.')
    t_accum += task.exposed_comms*1e3

    plt.xlim([-0.5,1.5])
    plt.xticks([0, 1], ['Serialized', 'Overlapped'], fontsize=14)
    plt.ylabel('Execution Time [ms]', fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(filename)

def plot_timeline(computation_stream, communication_stream, figures_dir):

    filename = figures_dir + 'timeline.png'
    plt.figure(figsize=(15, 5), dpi=300)

    label_set = set()

    # Plot computation stream
    for trace in computation_stream:
        if 'EMB' in trace['name']:
            clr = clr_emb
            lbl = 'EMB'
        elif 'MLP' in trace['name'] or 'Attn' in trace['name'] or 'FC' in trace['name']:
            clr = clr_gemm
            lbl = 'GEMM'
        if lbl in label_set:
            lbl = None
        else:
            label_set.add(lbl)
        plt.barh(1, trace['duration']*1e3, height=0.8, left = trace['t_start']*1e3, color = clr, edgecolor='black', label = lbl)
    
    # Plot communication stream
    for trace in communication_stream:
        if 'all2all' in trace['name']:
            clr = clr_all2all
            lbl = 'All2All'
        if 'ar' in trace['name']:
            clr = clr_allreduce
            lbl = 'AllReduce'
        if 'ag' in trace['name']:
            clr = clr_allgather
            lbl = 'AllGather'
        if 'rs' in trace['name']:
            clr = clr_reducescatter
            lbl = 'ReduceScatter'
        if lbl in label_set:
            lbl = None
        else:
            label_set.add(lbl)
        plt.barh(0, trace['duration']*1e3, height=0.8, left = trace['t_start']*1e3, color = clr, edgecolor='black', label = lbl)

    plt.ylim([-0.5,1.5])
    plt.yticks([0, 1], ['Communication', 'Computation'], fontsize=14, rotation = 90, va='center')
    plt.xlabel('Execution Time [ms]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(filename)