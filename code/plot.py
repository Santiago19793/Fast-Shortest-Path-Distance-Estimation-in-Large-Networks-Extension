import matplotlib.pyplot as plt

import settings

# GENERATE NICELY VISUALIZED RESULTS

def plot(results_all):
    """
    Aims to nicely plot the results that it gets passed in results_all.
    """

    fig, axes = plt.subplots(2, settings.NR_DATASETS, figsize=(23, 9), sharex=True)
    axes = axes.flatten()

    for i, dataset in enumerate(settings.DATASETS):

        results_curr = results_all[i]

        ax = axes[i]

        ax.set_title(f"({chr(ord('a') + i)}) {dataset} w/ random/local", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

        ax.plot(settings.LANDMARK_SIZES, results_curr[0][0], linestyle="dashed", color="gold", label=f"Random")
        ax.plot(settings.LANDMARK_SIZES, results_curr[0][1], linestyle="dotted", label=f"Degree")
        ax.plot(settings.LANDMARK_SIZES, results_curr[0][2], marker="o", label=f"Betweenness")
        ax.plot(settings.LANDMARK_SIZES, results_curr[0][3], linestyle="dashdot", label=f"Closeness")
        ax.plot(settings.LANDMARK_SIZES, results_curr[1][0], linestyle="dashed", color="brown", label=f"Random/1")
        ax.plot(settings.LANDMARK_SIZES, results_curr[1][1], linestyle="dotted", label=f"Degree/1")
        ax.plot(settings.LANDMARK_SIZES, results_curr[1][2], marker="o", label=f"Betweenness/1")
        ax.plot(settings.LANDMARK_SIZES, results_curr[1][3], linestyle="dashdot", label=f"Closeness/1")
        ax.plot(settings.LANDMARK_SIZES, results_curr[2][0], linestyle="dashed", color="royalblue", label=f"Random/P")
        ax.plot(settings.LANDMARK_SIZES, results_curr[2][1], linestyle="dotted", label=f"Degree/P")
        ax.plot(settings.LANDMARK_SIZES, results_curr[2][2], linestyle="dashdot", label=f"Closeness/P")
        ax.plot(settings.LANDMARK_SIZES, results_curr[2][3], linestyle="solid", label=f"Border/P")
        ax.plot(settings.LANDMARK_SIZES, results_curr[3], marker="D", color="olivedrab", label=f"Local")
        ax.plot(settings.LANDMARK_SIZES, results_curr[4][0], marker="P", label=f"Linear(1,1,1)")
        ax.plot(settings.LANDMARK_SIZES, results_curr[4][1], marker="P", color="magenta", label=f"Linear(1,2,3)")
        ax.plot(settings.LANDMARK_SIZES, results_curr[4][2], marker="P", color="lime", label=f"Linear(2,1,3)")
        
        ax = axes[i+settings.NR_DATASETS]
        ax.set_title(f"({chr(ord('a') + i)}) {dataset} w/o random/local", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)        
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

        ax.plot(settings.LANDMARK_SIZES, results_curr[0][1], linestyle="dotted", label=f"Degree")
        ax.plot(settings.LANDMARK_SIZES, results_curr[0][2], marker="o", label=f"Betweenness")
        ax.plot(settings.LANDMARK_SIZES, results_curr[0][3], linestyle="dashdot", label=f"Closeness")
        ax.plot(settings.LANDMARK_SIZES, results_curr[1][1], linestyle="dotted", label=f"Degree/1")
        ax.plot(settings.LANDMARK_SIZES, results_curr[1][2], marker="o", label=f"Betweenness/1")
        ax.plot(settings.LANDMARK_SIZES, results_curr[1][3], linestyle="dashdot", label=f"Closeness/1")
        ax.plot(settings.LANDMARK_SIZES, results_curr[2][1], linestyle="dotted", label=f"Degree/P")
        ax.plot(settings.LANDMARK_SIZES, results_curr[2][2], linestyle="dashdot", label=f"Closeness/P")
        ax.plot(settings.LANDMARK_SIZES, results_curr[2][3], linestyle="solid", label=f"Border/P")
        ax.plot(settings.LANDMARK_SIZES, results_curr[4][0], marker="P", label=f"Linear(1,1,1)")
        ax.plot(settings.LANDMARK_SIZES, results_curr[4][1], marker="P", color="magenta", label=f"Linear(1,2,3)")
        ax.plot(settings.LANDMARK_SIZES, results_curr[4][2], marker="P", color="lime", label=f"Linear(2,1,3)")

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc='upper center',
            ncol=8,
            fontsize=16,
            frameon=True)

    # Shared axis labels
    fig.text(0.5, 0.04, 'Size of landmark set', ha='center', fontsize=16)
    fig.text(-0.01, 0.5, 'Proportional error',
         va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0, 0.06, 1, 0.9])
    plt.savefig(f"../results/plot.pdf", bbox_inches="tight")