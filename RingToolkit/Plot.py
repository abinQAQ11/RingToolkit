import mplcursors
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from typing import List, Any
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
def magnet(ax, height, lattice):
    current_s = 0.0
    for item in lattice.ring:
        l = item.length
        if item.type == 'drift':
            current_s += l
        elif item.type == 'quadrupole':
            ax.add_patch(plt.Rectangle((current_s, -height*1.1), l, height, color='red', alpha=0.8))
            current_s += l
        elif item.type == 'bending':
            ax.add_patch(plt.Rectangle((current_s, -height*1.1), l, height * 0.8, color='blue', alpha=1.0))
            current_s += l
        elif item.type == 'sextupole':
            ax.add_patch(plt.Rectangle((current_s, -height*1.1), l, height, color='green', alpha=1.0))
            current_s += l
# ----------------------------------------------------------------------------------------------------------------------
def plot_resonance_lines(ax, x_min, x_max, y_min, y_max):
    if not type(x_min) == int:
        x_min = int(x_min)
        x_max = int(x_max) + 1
        y_min = int(y_min)
        y_max = int(y_max) + 1

    """一阶共振"""
    for i in range(x_min, x_max + 1):
        """x=N"""
        ax.axvline(x=i, color='black', alpha=1.0, zorder=1, linewidth=4.0)

    for i in range(y_min, y_max + 1):
        """y=N"""
        ax.axhline(y=i, color='black', alpha=1.0, zorder=1, linewidth=4.0)

    """二阶共振"""
    for i in np.arange(x_min - 0.5, x_max + 0.5, 1):
        """2x=N"""
        ax.axvline(x=float(i), color='gray', alpha=1.0, zorder=2, linewidth=3.0)

    for i in np.arange(y_min - 0.5, y_max + 0.5, 1):
        """2y=N"""
        ax.axhline(y=float(i), color='gray', alpha=1.0, zorder=2, linewidth=3.0)

    for i in range(x_min + y_min, x_max + y_max + 1):
        """x+y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), i - np.linspace(x_min, x_max, 1000),
                color='gray', alpha=1.0, zorder=2, linewidth=3.0)

    for i in range(x_min - y_max, x_max - y_min + 1):
        """x-y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), np.linspace(x_min, x_max, 1000) - i,
                color='gray', alpha=1.0, zorder=2, linewidth=3.0)

    """三阶共振"""
    for i in np.arange(x_min - 1 / 3, x_max + 1 / 3, 1 / 3):
        """3x=N"""
        ax.axvline(x=float(i), color='red', alpha=0.9, zorder=3, linewidth=2.0)

    for i in np.arange(y_min - 1 / 3, y_max + 1 / 3, 1 / 3):
        """3y=N"""
        ax.axhline(y=float(i), color='red', alpha=0.9, zorder=3, linewidth=2.0)

    for i in range(x_min + 2 * y_min, x_max + 2 * y_max + 1):
        """x+2y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (i - np.linspace(x_min, x_max, 1000)) / 2,
                color='red', alpha=0.9, zorder=3, linewidth=2.0)

    for i in range(x_min - 2 * y_max, x_max - 2 * y_min + 1):
        """x-2y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (np.linspace(x_min, x_max, 1000) - i) / 2,
                color='red', alpha=0.9, zorder=3, linewidth=2.0)

    # for i in range(2 * x_min + y_min, 2 * x_max + y_max + 1):
    #     """2x+y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), i - 2 * np.linspace(x_min, x_max, 1000),
    #             color='red', alpha=1.0, zorder=3, linewidth=2.0)

    # for i in range(2 * x_min - y_max, 2 * x_max - y_min + 1):
    #     """2x-y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), 2 * np.linspace(x_min, x_max, 1000) - i,
    #             color='red', alpha=1.0, zorder=3, linewidth=2.0)

    """四阶共振"""
    for i in np.arange(x_min - 0.25, x_max + 0.25, 0.5):
        """4x=N"""
        ax.axvline(x=float(i), color='green', alpha=0.9, zorder=4, linewidth=1.0)

    for i in np.arange(y_min - 0.25, y_max + 0.25, 0.5):
        """4y=N"""
        ax.axhline(y=float(i), color='green', alpha=0.9, zorder=4, linewidth=1.0)

    for i in range(2 * x_min + 2 * y_min, 2 * x_max + 2 * y_max + 1):
        """2x+2y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (i - 2 * np.linspace(x_min, x_max, 1000)) / 2,
                color='green', alpha=0.9, zorder=4, linewidth=1.0)

    for i in range(2 * x_min - 2 * y_max, 2 * x_max - 2 * y_min + 1):
        """2x-2y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (2 * np.linspace(x_min, x_max, 1000) - i) / 2,
                color='green', alpha=0.9, zorder=4, linewidth=1.0)

    # for i in range(x_min + 3 * y_min, x_max + 3 * y_max + 1):
    #     """x+3y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), (i - np.linspace(x_min, x_max, 1000)) / 3,
    #             color='green', alpha=0.9, zorder=4, linewidth=1.0)

    # for i in range(x_min - 3 * y_max, x_max - 3 * y_min + 1):
    #     """x-3y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), (np.linspace(x_min, x_max, 1000) - i) / 3,
    #             color='green', alpha=0.9, zorder=4, linewidth=1.0)

    # for i in range(3 * x_min + y_min, 3 * x_max + y_max + 1):
    #     """3x+y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), i - 3 * np.linspace(x_min, x_max, 1000),
    #             color='green', alpha=0.9, zorder=4, linewidth=1.0)

    # for i in range(3 * x_min - y_max, 3 * x_max - y_min + 1):
    #     """3x-y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), 3 * np.linspace(x_min, x_max, 1000) - i,
    #             color='green', alpha=0.9, zorder=4, linewidth=1.0)

    """五阶共振"""
    for i in np.arange(x_min - 0.2, x_max + 0.2, 0.2):
        """5x=N"""
        ax.axvline(x=float(i), color='blue', alpha=0.8, zorder=5, linewidth=1.0)
    #
    # for i in np.arange(y_min - 0.2, y_max + 0.2, 0.2):
    #     """5y=N"""
    #     ax.axhline(y=float(i), color='blue', alpha=0.8, zorder=5, linewidth=1.0)

    for i in range(x_min + 4 * y_min, x_max + 4 * y_max + 1):
        """x+4y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (i - np.linspace(x_min, x_max, 1000)) / 4,
                color='blue', alpha=0.8, zorder=5, linewidth=1.0)

    for i in range(x_min - 4 * y_max, x_max - 4 * y_min + 1):
        """x-4y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (np.linspace(x_min, x_max, 1000) - i) / 4,
                color='blue', alpha=0.8, zorder=5, linewidth=1.0)

    # for i in range(2 * x_min + 3 * y_min, 2 * x_max + 3 * y_max + 1):
    #     """2x+3y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), (i - 2 * np.linspace(x_min, x_max, 1000)) / 3,
    #             color='blue', alpha=0.8, zorder=5, linewidth=1.0)
    #
    # for i in range(2 * x_min - 3 * y_max, 2 * x_max - 3 * y_min + 1):
    #     """2x-3y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), (2 * np.linspace(x_min, x_max, 1000) - i) / 3,
    #             color='blue', alpha=0.8, zorder=5, linewidth=1.0)
    #
    for i in range(3 * x_min + 2 * y_min, 3 * x_max + 2 * y_max + 1):
        """3x+2y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (i - 3 * np.linspace(x_min, x_max, 1000)) / 2,
                color='blue', alpha=0.8, zorder=5, linewidth=1.0)

    for i in range(3 * x_min - 2 * y_max, 3 * x_max - 2 * y_min + 1):
        """3x-2y=N"""
        ax.plot(np.linspace(x_min, x_max, 1000), (3 * np.linspace(x_min, x_max, 1000) - i) / 2,
                color='blue', alpha=0.8, zorder=5, linewidth=1.0)

    # for i in range(4 * x_min + y_min, 4 * x_max + y_max + 1):
    #     """4x+y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), i - 4 * np.linspace(x_min, x_max, 1000),
    #             color='blue', alpha=0.8, zorder=5, linewidth=1.0)

    # for i in range(4 * x_min - y_max, 4 * x_max - y_min + 1):
    #     """4x-y=N"""
    #     ax.plot(np.linspace(x_min, x_max, 1000), 4 * np.linspace(x_min, x_max, 1000) - i,
    #             color='blue', alpha=0.8, zorder=5, linewidth=1.0)
# ----------------------------------------------------------------------------------------------------------------------
def plot_lattice(lattice: Any, cell_or_ring: str = "cell"):
    plt.ion()
    plt.rcParams.update({
        "font.size": 45,
        "font.family": "Times New Roman",
        "axes.titlesize": 45,
        "axes.labelsize": 45,
        "xtick.labelsize": 42,
        "ytick.labelsize": 42,
        "legend.fontsize": 38,
        "mathtext.fontset": "stix"
    })
    fig, ax1 = plt.subplots(figsize=(12, 12 * 0.618))
    plt.subplots_adjust(left=0.14, right=0.86, bottom=0.17, top=0.95)
    for spine in ax1.spines.values():
        spine.set_linewidth(1)

    data = None
    if cell_or_ring == "cell":
        data = lattice.twiss
    elif cell_or_ring == "ring":
        data = lattice.d_terms_m

    s = np.array([item[0] for item in data])
    beta_x = np.array([item[1]['Beta_X'] for item in data])
    beta_y = np.array([item[1]['Beta_Y'] for item in data])
    disp_x = np.array([item[1]['Disp_X'] for item in data])

    ax1.plot(s, beta_x, color='blue', label=r'$\beta_x$', linewidth=5, alpha=1.0)
    ax1.plot(s, beta_y, color='red', label=r'$\beta_y$', linewidth=5, alpha=1.0)

    ax1.set_xlabel('s [m]')
    ax1.set_ylabel('Beta functions [m]', labelpad=10, color='black')
    ax1.tick_params(axis='y', labelcolor='black', direction="out", length=6, width=1)
    ax1.tick_params(axis='x', labelcolor='black', direction="out", length=6, width=1)
    high_1 = 30
    height_1 = high_1 / 20
    ax1.set_xlim(min(s), max(s))
    ax1.set_ylim(-height_1, high_1)
    # ax1.xaxis.set_major_locator(MultipleLocator(4))
    # ax1.yaxis.set_major_locator(MultipleLocator(5))

    ax2 = ax1.twinx()
    ax2.plot(s, disp_x, color='green', label=r'$\eta_x$', linewidth=5, alpha=1.0)
    ax2.set_ylabel('Dispersion [m]', labelpad=10, color='black')
    ax2.tick_params(axis='y', labelcolor='black', direction="out", length=6, width=1)
    high_2 = 0.6
    height_2 = high_2 / 20
    ax2.set_ylim(-height_2, high_2)
    # ax2.yaxis.set_major_locator(MultipleLocator(0.1))

    magnet(ax1, height_1, lattice)

    ax1.legend(ncol=2, loc='upper left', frameon=False)
    ax2.legend(loc='upper right', frameon=False)

    # fig.suptitle(r'$\beta_x$, $\beta_y$ and Disp_x', fontsize=20)
    # plt.savefig("lattice.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.ioff()
# ----------------------------------------------------------------------------------------------------------------------
def plot_driving_terms(lattice: Any):
    data = lattice.d_terms_m
    plt.ion()
    s = np.array([item[0] for item in data])
    h_21000 = np.array([item[1]['h_21000'] for item in data])
    h_30000 = np.array([item[1]['h_30000'] for item in data])
    h_10110 = np.array([item[1]['h_10110'] for item in data])
    h_10020 = np.array([item[1]['h_10020'] for item in data])
    h_10200 = np.array([item[1]['h_10200'] for item in data])

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(s, h_21000, color= 'green', linewidth=3, label=r'$h_{21000}$')
    ax1.plot(s, h_30000, color=  'blue', linewidth=3, label=r'$h_{30000}$')
    ax1.plot(s, h_10110, color='purple', linewidth=3, label=r'$h_{10110}$')
    ax1.plot(s, h_10020, color=   'red', linewidth=3, label=r'$h_{10020}$')
    ax1.plot(s, h_10200, color=  'cyan', linewidth=3, label=r'$h_{10200}$')

    ax1.set_xlabel('s (m)', fontsize=20)
    ax1.set_ylabel(r'$Amplitude\ of\ Terms\ in\ f_3\ [\ m^{-1/2}\ ]$', fontsize=20, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=18)
    high = max(max(h_21000), max(h_30000), max(h_10110), max(h_10020), max(h_10200))
    height = high / 40
    ax1.set_ylim(-height, high * 1.2)
    ax1.set_xlim(-1, max(s)+1)

    magnet(ax1, height, lattice)

    ax1.legend(loc='best', fontsize=20, ncol=2)
    fig.tight_layout()
    plt.show()
    plt.ioff()
# ----------------------------------------------------------------------------------------------------------------------
def plot_rdts(lattice: Any):
    plt.ion()
    plt.rcParams.update({
        "font.size": 45,
        "font.family": "Times New Roman",
        "axes.titlesize": 45,
        "axes.labelsize": 42,
        "xtick.labelsize": 42,
        "ytick.labelsize": 42,
        "legend.fontsize": 34,
        "mathtext.fontset": "stix"
    })
    fig, ax = plt.subplots(figsize=(12, 12 * 0.618))
    plt.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.97)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    data = lattice.rdts

    s = np.array([item[0] for item in lattice.d_term])
    f_21000 = np.array([item[1]['C_t'] for item in data['h_21000']])
    f_30000 = np.array([item[1]['C_t'] for item in data['h_30000']])
    f_10110 = np.array([item[1]['C_t'] for item in data['h_10110']])
    f_10020 = np.array([item[1]['C_t'] for item in data['h_10020']])
    f_10200 = np.array([item[1]['C_t'] for item in data['h_10200']])

    ax.plot(s, f_21000, color='green', linewidth=4, label=r'$f_{21000}$')
    ax.plot(s, f_30000, color='blue', linewidth=4, label=r'$f_{30000}$')
    ax.plot(s, f_10110, color='purple', linewidth=4, label=r'$f_{10110}$')
    ax.plot(s, f_10020, color='red', linewidth=4, label=r'$f_{10020}$')
    ax.plot(s, f_10200, color='cyan', linewidth=4, label=r'$f_{10200}$')

    ax.set_xlabel('s [m]')
    ax.set_ylabel(r'3rd-order RDTs [$\text{m}^{-1/2}$]', color='black', labelpad=10)
    ax.tick_params(axis='y', labelcolor='black', direction="out", length=4, width=1)
    ax.tick_params(axis='x', labelcolor='black', direction="out", length=4, width=1)
    high = max(max(f_21000), max(f_30000), max(f_10110), max(f_10020), max(f_10200))
    height = high / 20
    ax.set_ylim(-height, high * 1.45)
    ax.set_xlim(-0.5, max(s) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    magnet(ax, height, lattice)

    ax.legend(ncol=3, loc='upper center', frameon=False)
    # plt.savefig("rdts.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.ioff()
# ----------------------------------------------------------------------------------------------------------------------
def plot_tune(lattice: Any):
    plt.ion()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 50

    fig, ax = plt.subplots(figsize=(12, 9))

    x_min, x_max = 7, 8
    y_min, y_max = 3, 4

    x0 = np.array([lattice.tune_x])
    y0 = np.array([lattice.tune_y])
    ax.scatter(x0, y0, color='red', s=150, marker='^')

    ax.set_xlabel(r'$\nu_{x}$')
    ax.set_ylabel(r'$\nu_{y}$')
    ax.tick_params(axis='both')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plot_resonance_lines(ax, x_min, x_max, y_min, y_max)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    fig.tight_layout()
    plt.show()
    plt.ioff()
# ----------------------------------------------------------------------------------------------------------------------
def plot_fma(fma_name: str, mode: str='da', off_mom = 0):
    plt.ion()
    plt.rcParams.update({
        "font.size": 24,
        "font.family": "Times New Roman",
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "mathtext.fontset": "stix"
    })
    fig, ax = plt.subplots(figsize=(8, 7 * 0.618))
    plt.subplots_adjust(left=0.15, right=0.90, bottom=0.20, top=0.95)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    import pysdds
    fma = pysdds.read(fma_name)

    def da(_fma):
        x = _fma.col('x')[0]
        y = _fma.col('y')[0]
        c = _fma.col('diffusionRate')[0]
        if off_mom:
            p_idx = np.where(np.abs(fma.col('y')[0] - (off_mom / 100)) < 1e-4)[0]
            scatter = ax.scatter(x[p_idx] * 1000, y[p_idx] * 1000, c=c[p_idx], s=30, marker='s', cmap='jet', alpha=1.0,
                             vmax=-3, vmin=-12)
        else:
            scatter = ax.scatter(x * 1000, y * 1000, s=30, c=c, marker='s', cmap='jet', alpha=1.0,
                             vmax=-3, vmin=-12)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_xlim(-50, 50)
        ax.set_ylim(0, 10)
        ax.tick_params(axis='x', labelcolor='black', direction="out", length=4, width=1)
        ax.tick_params(axis='y', labelcolor='black', direction="out", length=4, width=1)
        ax.xaxis.set_major_locator(MultipleLocator(20))

        cbar = fig.colorbar(scatter, pad=0.01)
        cbar.ax.tick_params(length=4, width=1)
        cbar.set_label('Diffusion rate')
        cbar.ax.yaxis.set_major_locator(MultipleLocator(2))

    def tune(_fma):
        # plt.subplots_adjust(left=0.12, right=0.94, bottom=0.18, top=0.95)
        x = _fma.col('nux')[0]
        y = _fma.col('nuy')[0]
        c = _fma.col('diffusionRate')[0]
        if off_mom:
            p_idx = np.where(np.abs(fma.col('y')[0] - off_mom / 100) < 1e-4)[0]
            scatter = ax.scatter(x[p_idx], y[p_idx], c=c[p_idx], s=5, marker='s', cmap='jet', alpha=1.0,
                             vmax=-3, vmin=-12)
        else:
            scatter = ax.scatter(x, y, c=c, s=5, marker='s', cmap='jet', alpha=1.0,
                             vmax=-3, vmin=-12)

        ax.set_xlabel(r'$\nu_{x}$')
        ax.set_ylabel(r'$\nu_{y}$')
        ax.tick_params(axis='x', labelcolor='black', direction="out", length=4, width=1)
        ax.tick_params(axis='y', labelcolor='black', direction="out", length=4, width=1)

        cbar = fig.colorbar(scatter, pad=0.01)
        cbar.ax.tick_params(length=4, width=1)
        cbar.set_label('Diffusion rate')
        cbar.ax.yaxis.set_major_locator(MultipleLocator(2))

        plot_resonance_lines(ax, min(x), max(x), min(y), max(y))

    match mode:
        case "da":
            da(fma)
        case "tune":
            tune(fma)

    # plt.savefig("fma.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.ioff()
# ----------------------------------------------------------------------------------------------------------------------
def plot_fma_x_delta(fma_name: str, mode: str='hda', off_mom = 0):
    plt.ion()
    plt.rcParams.update({
        "font.size": 24,
        "font.family": "Times New Roman",
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "mathtext.fontset": "stix"
    })
    fig, ax = plt.subplots(figsize=(8, 7 * 0.618))
    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.20, top=0.95)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    import pysdds
    fma_x_delta = pysdds.read(fma_name)

    def hda(_fma_x_delta):
        x = _fma_x_delta.col('delta')[0]
        y = _fma_x_delta.col('x')[0]
        c = _fma_x_delta.col('diffusionRate')[0]
        if off_mom:
            p_idx = np.where(np.abs(x) - off_mom / 100 < 1e-4)[0]
            scatter = ax.scatter(x[p_idx] * 100, y[p_idx] * 1000, c=c[p_idx], s=30, marker='s', cmap='jet', alpha=1.0,
                             vmax=-3, vmin=-12)
        else:
            scatter = ax.scatter(x * 100, y * 1000, c=c, s=30, marker='s', cmap='jet', alpha=1.0,
                             vmax=-3, vmin=-12)

        ax.set_xlabel('$\\delta$ [%]')
        ax.set_ylabel('x [mm]')
        ax.set_xlim(-3, 3)
        ax.tick_params(axis='x', labelcolor='black', direction="out", length=4, width=1)
        ax.tick_params(axis='y', labelcolor='black', direction="out", length=4, width=1)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(20))

        cbar = fig.colorbar(scatter, pad=0.01)
        cbar.ax.tick_params(length=4, width=1)
        cbar.set_label('Diffusion rate')
        cbar.ax.yaxis.set_major_locator(MultipleLocator(2))

    def tune(_fma_x_delta):
        plt.subplots_adjust(left=0.12, right=0.94, bottom=0.18, top=0.95)

        x = _fma_x_delta.col('nux')[0]
        y = _fma_x_delta.col('nuy')[0]
        c = _fma_x_delta.col('diffusionRate')[0]

        if off_mom:
            p_idx = np.where(np.abs(_fma_x_delta.col('delta')[0]) - off_mom / 100 < 1e-4)[0]
            scatter = ax.scatter(x[p_idx], y[p_idx], c=c[p_idx], s=5, marker='s', cmap='jet', alpha=1.0,
                                 vmax=-3, vmin=-12)
        else:
            scatter = ax.scatter(x, y, c=c, s=5, marker='s', cmap='jet', alpha=1.0,
                                 vmax=-3, vmin=-12)

        ax.set_xlabel(r'$\nu_{x}$')
        ax.set_ylabel(r'$\nu_{y}$')
        ax.tick_params(axis='x', labelcolor='black', direction="out", length=4, width=1)
        ax.tick_params(axis='y', labelcolor='black', direction="out", length=4, width=1)

        cbar = fig.colorbar(scatter, pad=0.01)
        cbar.ax.tick_params(length=4, width=1)
        cbar.set_label('Diffusion rate')
        cbar.ax.yaxis.set_major_locator(MultipleLocator(2))

        plot_resonance_lines(ax, min(x), max(x), min(y), max(y))

    match mode:
        case "hda":
            hda(fma_x_delta)
        case "tune":
            tune(fma_x_delta)

    # plt.savefig("fma.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.ioff()
# ----------------------------------------------------------------------------------------------------------------------
def plot_bunch_charge(data):
    plt.ion()
    plt.rcParams.update({
        "font.size": 45,
        "font.family": "Times New Roman",
        "axes.titlesize": 45,
        "axes.labelsize": 42,
        "xtick.labelsize": 42,
        "ytick.labelsize": 42,
        "legend.fontsize": 38,
        "mathtext.fontset": "stix"
    })
    fig, ax1 = plt.subplots(figsize=(12, 9.0))
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.165, top=0.95)
    for spine in ax1.spines.values():
        spine.set_linewidth(1)

    x = np.array([item[0] for item in data])
    y_1 = np.array([item[1] for item in data]) * 1e9
    y_2 = np.array([item[2] for item in data]) * 1e4

    ax1.scatter(x, y_1, s=80, color='red', marker='s', label='Horizontal emittance', alpha=1.0)
    ax1.set_xlabel('Bunch charge [nC]')
    ax1.set_ylabel('Horizontal emittance [nm·rad]', color='red')
    ax1.tick_params(axis='y', labelcolor='red', direction="out", length=4, width=1)
    ax1.tick_params(axis='x', labelcolor='black', direction="out", length=4, width=1)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(2.6, 3.4)
    y1_ticks = np.arange(2.6, 3.41, 0.2)
    ax1.set_yticks(y1_ticks)
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.tick_params(axis='both', which='both', pad=10)

    ax1.grid('on')

    ax2 = ax1.twinx()
    ax2.scatter(x, y_2, s=80, color='blue', marker='s', label='Energy spread', alpha=1.0)
    ax2.set_ylabel(r'Energy spread [$\times10^{-4}$]', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue', direction="out", length=4, width=1)
    ax2.set_ylim(4.9, 5.7)
    y2_ticks = np.arange(4.9, 5.71, 0.2)
    ax2.set_yticks(y2_ticks)
    ax2.tick_params(axis='y', pad=10)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    legend = ax1.legend(all_handles, all_labels,
                loc='upper left',
                handlelength=1, handletextpad=0.5,
                columnspacing=1, labelspacing=0.3,
                frameon=False,
                ncol=1)
    plt.setp(legend.get_texts(), ha='left')  # ha=horizontal alignment

    # fig.tight_layout()
    plt.show()
    plt.ioff()
# ----------------------------------------------------------------------------------------------------------------------
