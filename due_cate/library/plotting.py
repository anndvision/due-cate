import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 14,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    "font.size": 24,
}
plt.rcParams.update(params)


def errorbar(
    x,
    y,
    y_err,
    x_label,
    y_label,
    marker_label=None,
    x_pad=-20,
    y_pad=-45,
    legend_loc="upper left",
    file_path=None,
):
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    plt.errorbar(
        x,
        y,
        yerr=y_err,
        linestyle="None",
        marker="o",
        elinewidth=1.0,
        capsize=2.0,
        label=marker_label,
    )
    lim = max(np.abs(x.min()), np.abs(x.max())) * 1.1
    r = np.arange(-lim, lim, 0.1)
    _ = plt.plot(r, r, label="Ground Truth")
    _ = plt.tick_params(axis="x", direction="in", pad=x_pad)
    _ = plt.tick_params(axis="y", direction="in", pad=y_pad)
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.ylim([-lim, lim])
    _ = plt.legend(loc=legend_loc)
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()
