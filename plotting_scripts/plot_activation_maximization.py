#!/usr/bin/env python3
"""
plot_activation_maximization.py

Three modular figures from activation-maximisation results:

  plot_data1  — system-state heatmap  (channel × microservice, time-averaged)
  plot_data2  — latency-history heatmap (lat-pct-channel × time-step, annotated)
  plot_data3  — CPU-allocation bar chart (one bar per microservice)

Run (from plotting_scripts/ with the venv active):
    .venv/bin/python plot_activation_maximization.py \
        --results ../activation_maximization_results_3.npy \
        --outdir  ./plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── metadata ──────────────────────────────────────────────────────────────────

PERCENTILES = ['p90', 'p95', 'p98', 'p99', 'p99.9']
COLORS      = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

# data1 channel labels (order matches data_parser_socialml_next_k.py)
CHANNEL_LABELS = [
    'RPS\n(req/s)',
    'Replica\ncount',
    'CPU limit\n(vCPUs)',
    'CPU usage\n(vCPUs)',
    'RSS\n(MB)',
    'Cache mem\n(MB)',
]

# 28 microservices (order matches Services list in data parser)
SERVICES = [
    'compose-post-redis',        'compose-post-svc',
    'home-timeline-redis',       'home-timeline-svc',
    'nginx-thrift',              'post-storage-memcached',
    'post-storage-mongodb',      'post-storage-svc',
    'social-graph-mongodb',      'social-graph-redis',
    'social-graph-svc',          'text-svc',
    'text-filter-svc',           'unique-id-svc',
    'url-shorten-svc',           'media-svc',
    'media-filter-svc',          'user-mention-svc',
    'user-memcached',            'user-mongodb',
    'user-svc',                  'user-timeline-mongodb',
    'user-timeline-redis',       'user-timeline-svc',
    'write-home-timeline-svc',   'write-home-timeline-rmq',
    'write-user-timeline-svc',   'write-user-timeline-rmq',
]

# data2 row labels: latency-percentile channels in the history window
LAT_HISTORY_LABELS = ['p90 hist', 'p95 hist', 'p98 hist', 'p99 hist', 'p99.9 hist']

# data2 column labels: time steps (oldest → most recent)
TIME_LABELS = ['t − 4', 't − 3', 't − 2', 't − 1', 't']


# ── fig 1 · data1 heatmap ─────────────────────────────────────────────────────

def plot_data1(results, outdir):
    """
    For each maximised latency percentile: a heatmap grid showing the optimal
    system state (data1) across all 5 time steps (not averaged).

    Axes (per subplot)
    ------------------
    y  —  6 resource-metric channels (RPS, replicas, CPU limit, …)
    x  —  28 Social-Network microservices

    Layout: 5 rows (percentiles) × 5 columns (time steps)

    Each subplot is normalised to [0, 1] relative to its own maximum so that
    the spatial pattern is visible even when raw channel scales differ.
    """
    n_percentiles = len(PERCENTILES)
    n_timesteps = 5

    fig, axes = plt.subplots(n_percentiles, n_timesteps,
                              figsize=(6 * n_timesteps, 3.5 * n_percentiles))
    fig.suptitle(
        'Fig 1  —  Optimal system-state (data1) that maximises each latency percentile\n'
        'Columns = time steps (t−4 → t), Rows = latency percentiles. '
        'Each subplot normalised to [0 = min, 1 = max]',
        fontsize=13, y=1.01,
    )

    # Ensure axes is 2D even if n_percentiles=1
    if n_percentiles == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (pname, color) in enumerate(zip(PERCENTILES, COLORS)):
        d1 = results[pname]['data1'][0]  # (6, 28, 5)

        for col_idx in range(n_timesteps):
            ax = axes[row_idx, col_idx]
            mat = d1[:, :, col_idx]  # (6, 28) — single time step

            vmax = mat.max()
            mat_norm = mat / vmax if vmax > 1e-9 else mat

            im = ax.imshow(mat_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

            # Add subtle vertical grid lines to help track x-axis columns
            ax.set_xticks(np.arange(len(SERVICES)) - 0.5, minor=True)
            ax.grid(which='minor', axis='x', color='gray', linewidth=0.5, alpha=0.3)

            # Title only on first column
            if col_idx == 0:
                ax.set_title(
                    f'{pname} (activation={results[pname]["activation"]:.1f}ms)',
                    fontsize=9, fontweight='bold', loc='left', color=color,
                )
            else:
                ax.set_title('')

            # Y labels only on first column
            if col_idx == 0:
                ax.set_yticks(range(len(CHANNEL_LABELS)))
                ax.set_yticklabels(CHANNEL_LABELS, fontsize=6)
            else:
                ax.set_yticks([])

            # X labels on bottom row only
            if row_idx == n_percentiles - 1:
                ax.set_xticks(range(len(SERVICES)))
                ax.set_xticklabels(SERVICES, rotation=45, ha='right', fontsize=5)
            else:
                ax.set_xticks([])

            # Time step label on top
            time_label = TIME_LABELS[col_idx]
            ax.text(0.5, 1.02, time_label, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalised value (1 = highest)', fontsize=8)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    path = os.path.join(outdir, 'fig1_data1_system_state_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved → {path}')
    plt.close()


# ── fig 2 · data2 heatmap ─────────────────────────────────────────────────────

def plot_data2(results, outdir):
    """
    For each maximised latency percentile: a 5×5 annotated heatmap of the
    optimal historical latency pattern (data2).

    Axes
    ----
    y  —  5 latency-percentile channels in the history window (p90 … p99.9)
    x  —  5 time steps (t−4 = oldest  …  t = most recent)

    Raw millisecond values are shown in each cell.  Each subplot has its own
    colour scale so low-activation percentiles (e.g. p99.9 = all zeros) are
    not washed out by the scale of others.

    Key question this answers: does the model need to see *sustained* high
    latency across multiple past seconds, or just a single recent spike?
    And which past-latency percentile matters most?
    """
    n = len(PERCENTILES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(
        'Fig 2  —  Optimal historical latency pattern (data2) that maximises each latency percentile\n'
        'Cell values are raw milliseconds.  Each subplot has its own colour scale.',
        fontsize=13, y=1.04,
    )

    for ax, pname, color in zip(axes, PERCENTILES, COLORS):
        d2   = results[pname]['data2'][0]          # (5, 5)  lat-pct × time
        vmax = d2.max() if d2.max() > 0 else 1.0

        im = ax.imshow(d2, aspect='auto', cmap='Blues', vmin=0, vmax=vmax)

        ax.set_title(
            f'Maximising  {pname}\n'
            f'({results[pname]["activation"]:.1f} ms)',
            fontsize=9, fontweight='bold', color=color,
        )

        ax.set_yticks(range(5))
        ax.set_yticklabels(LAT_HISTORY_LABELS, fontsize=8)
        ax.set_ylabel('Past latency percentile channel', fontsize=8)

        ax.set_xticks(range(5))
        ax.set_xticklabels(TIME_LABELS, fontsize=8)
        ax.set_xlabel('Time step\n(t = most recent)', fontsize=8)

        # annotate every cell with its ms value
        for r in range(5):
            for c in range(5):
                val = d2[r, c]
                text_color = 'white' if val > 0.55 * vmax else 'black'
                ax.text(
                    c, r, f'{val:.0f}',
                    ha='center', va='center',
                    fontsize=7, color=text_color,
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Latency (ms)', fontsize=7)

    plt.tight_layout()
    path = os.path.join(outdir, 'fig2_data2_latency_history_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved → {path}')
    plt.close()


# ── fig 3 · data3 bar chart ───────────────────────────────────────────────────

def plot_data3(results, outdir):
    """
    For each maximised latency percentile: a bar chart of the optimal
    next-step CPU allocation (data3) across all 28 microservices.

    Axes
    ----
    x  —  28 Social-Network microservices
    y  —  CPU limit the optimiser assigned to that service (vCPUs)

    This is the most directly actionable plot: data3 is the control input —
    the CPU allocation Sinan is about to apply.  Tall bars identify which
    services' CPU assignments the model associates with worst-case latency
    for each percentile.
    """
    n = len(PERCENTILES)
    fig, axes = plt.subplots(n, 1, figsize=(22, 4 * n))
    fig.suptitle(
        'Fig 3  —  Optimal next-step CPU allocation (data3) that maximises each latency percentile\n'
        'Bar height = CPU limit (vCPUs) assigned to each microservice by the optimiser',
        fontsize=13, y=1.01,
    )

    x = np.arange(len(SERVICES))

    for ax, pname, color in zip(axes, PERCENTILES, COLORS):
        d3 = results[pname]['data3'][0]            # (28,)

        bars = ax.bar(x, d3, color=color, alpha=0.85, width=0.7)

        ax.set_title(
            f'Maximising  {pname}   '
            f'(best activation = {results[pname]["activation"]:.1f} ms)',
            fontsize=10, fontweight='bold', loc='left', color=color,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(SERVICES, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('Microservice', fontsize=9)

        ax.set_ylabel('CPU limit (vCPUs)', fontsize=9)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=5))

        # label bars that are non-trivially tall (top 20 % of the max)
        threshold = d3.max() * 0.2 if d3.max() > 0 else 1
        for bar, val in zip(bars, d3):
            if val >= threshold:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + d3.max() * 0.01,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6.5,
                )

        ax.set_xlim(-0.5, len(SERVICES) - 0.5)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(outdir, 'fig3_data3_cpu_allocation_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved → {path}')
    plt.close()


# ── fig 4 · loss curves ───────────────────────────────────────────────────────

def plot_loss_curves(results, outdir):
    """
    One subplot per neuron showing how the optimisation progressed over steps.

    Two lines per subplot:
      - 'current'  (solid)   : raw activation at each logged step — shows the
                               true trajectory including any oscillation
      - 'best'     (dashed)  : running maximum up to that step — monotonically
                               non-decreasing, shows when the optimum was found

    The gap between the two lines reveals oscillation: a large gap means the
    optimizer overshot the best point and is wandering below it.  A small gap
    means the trajectory is well-behaved.  If 'best' flattens while 'current'
    is still noisy, more steps won't help — a smaller LR is needed instead.
    """
    # skip percentiles that have no curve saved (e.g. older result files)
    has_curves = all('curve' in results[p] for p in PERCENTILES)
    if not has_curves:
        print('  Skipping loss curves — no curve data in this results file.')
        return

    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
    fig.suptitle(
        'Fig 4  —  Optimisation loss curves: activation value vs step\n'
        'Solid = current trajectory;  Dashed = running best',
        fontsize=13, y=1.04,
    )

    for ax, pname, color in zip(axes, PERCENTILES, COLORS):
        curve = results[pname]['curve']
        steps   = curve['steps']
        current = curve['current']
        best    = curve['best']

        ax.plot(steps, current, color=color, alpha=0.5, linewidth=1.0,
                label='current')
        ax.plot(steps, best,    color=color, alpha=1.0, linewidth=1.5,
                linestyle='--', label='best')

        # mark the final best activation with a horizontal reference line
        final_best = results[pname]['activation']
        ax.axhline(final_best, color='black', linewidth=0.8, linestyle=':',
                   alpha=0.6)
        ax.text(steps[-1], final_best, f' {final_best:.0f} ms',
                va='bottom', fontsize=7, color='black')

        ax.set_title(
            f'{pname}',
            fontsize=10, fontweight='bold', color=color,
        )
        ax.set_xlabel('Optimisation step', fontsize=9)
        ax.set_ylabel('fc4 activation (ms)', fontsize=9)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'fig4_loss_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved → {path}')
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot activation-maximisation results (Figs 1–4)'
    )
    parser.add_argument(
        '--results',
        default='../activation_maximization_results.npy',
        help='Path to .npy results file',
    )
    parser.add_argument(
        '--outdir',
        default='./plots',
        help='Directory for saved figures (created if absent)',
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f'Loading {args.results} …')
    results = np.load(args.results, allow_pickle=True).item()
    for pname in PERCENTILES:
        print(f'  {pname}: activation = {results[pname]["activation"]:.4f} ms')

    print('\nGenerating figures …')
    plot_data1(results, args.outdir)
    plot_data2(results, args.outdir)
    plot_data3(results, args.outdir)
    plot_loss_curves(results, args.outdir)
    print('\nDone.')


if __name__ == '__main__':
    main()
