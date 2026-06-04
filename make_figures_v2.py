"""
make_figures_v2.py — Generate all publication figures for the AE-NS paper revision (v2 data).

Figures produced (saved to figures/):
  fig1_pareto_adult.pdf        — Pareto frontier (accuracy vs DPD) for Adult
  fig1_pareto_compas.pdf       — Pareto frontier (accuracy vs DPD) for COMPAS
  fig1_pareto_credit.pdf       — Pareto frontier (accuracy vs DPD) for Credit
  fig2_sensitivity_alpha.pdf   — Sensitivity to alpha (DPD, EOD, AOD on y-axis)
  fig2_sensitivity_beta.pdf    — Sensitivity to beta  (DPD, EOD, AOD on y-axis)
  fig3_ablation.pdf            — Ablation bar chart (DPD) across 7 variants
  fig4_tsne_adult.pdf          — t-SNE: Z (AE-NS) vs X (raw features) on Adult
  fig4_tsne_compas.pdf         — t-SNE: Z vs X on COMPAS
  fig4_tsne_credit.pdf         — t-SNE: Z vs X on Credit
  fig5_nmi_classify.pdf        — NMI + classify-on-Z summary (new for revision)

Requires: matplotlib, seaborn, numpy, json (all in .venv_gpu)
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
PALETTE = sns.color_palette('Set2', 10)

# ── Helpers ──────────────────────────────────────────────────────────────
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  saved {path}')


# ====================================================================
# FIGURE 1: Pareto frontier  —  accuracy vs DPD (3 subplots)
# ====================================================================
def fig1_pareto():
    datasets = ['adult', 'compas', 'credit']
    ds_labels = {'adult': 'Adult', 'compas': 'COMPAS', 'credit': 'Credit'}

    # Baselines from v1 (excluding old AE-NS)
    baselines = {
        'Adversarial Debiasing':         {'color': '#377eb8', 'marker': 'o',  'zorder': 5,  'size': 80},
        'Prejudice Remover':             {'color': '#4daf4a', 'marker': 's',  'zorder': 5,  'size': 80},
        'LAFAN (Lagrangian w/o AE)':     {'color': '#984ea3', 'marker': 'D',  'zorder': 5,  'size': 80},
        'Standard NN (No Fairness)':     {'color': '#ff7f00', 'marker': '^',  'zorder': 5,  'size': 80},
        'XGBoost (No Fairness)':         {'color': '#a65628', 'marker': 'v',  'zorder': 5,  'size': 80},
        'Fairlearn-DP + XGBoost':        {'color': '#f781bf', 'marker': 'P',  'zorder': 5,  'size': 80},
        'Fairlearn-EO + XGBoost':        {'color': '#999999', 'marker': 'X',  'zorder': 5,  'size': 80},
    }
    # v2 additions (filled below per dataset)
    v2_style = {
        'AE-NS v2 (Ours)':              {'color': '#e41a1c', 'marker': '*',  'zorder': 10, 'size': 220},
        'AE-NS v2 + Fairpost':          {'color': '#daa520', 'marker': 'p',  'zorder': 10, 'size': 200},
    }
    short_labels = {
        'AE-NS v2 (Ours)':              'AE-NS v2',
        'AE-NS v2 + Fairpost':          'AE-NS v2+FP',
        'Adversarial Debiasing':         'Adv. Debias.',
        'Prejudice Remover':             'Prej. Rem.',
        'LAFAN (Lagrangian w/o AE)':     'LAFAN',
        'Standard NN (No Fairness)':     'Std NN',
        'XGBoost (No Fairness)':         'XGBoost',
        'Fairlearn-DP + XGBoost':        'FL-DP',
        'Fairlearn-EO + XGBoost':        'FL-EO',
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]

        # Load v1 baselines
        data = json.load(open(f'AE_NS_Results_v1/{ds}/baseline_results.json'))
        accs, dpds = [], []
        names_all = []
        styles_all = []

        for name, style in baselines.items():
            m = data[name]
            accs.append(m['accuracy']['mean'])
            dpds.append(m['dpd']['mean'])
            names_all.append(name)
            styles_all.append(style)

        # Load AE-NS v2 standalone
        v2_res = json.load(open(f'AE_NS_Results_v2/v2_{ds}_results.json'))
        accs.append(v2_res['aggregate']['accuracy']['mean'])
        dpds.append(v2_res['aggregate']['dpd']['mean'])
        names_all.append('AE-NS v2 (Ours)')
        styles_all.append(v2_style['AE-NS v2 (Ours)'])

        # Load AE-NS v2 + Fairpost
        fp_res = json.load(open(f'AE_NS_Results_v2/fairpost_3seeds_{ds}.json'))
        accs.append(fp_res['aggregate']['accuracy']['mean'])
        dpds.append(fp_res['aggregate']['dpd']['mean'])
        names_all.append('AE-NS v2 + Fairpost')
        styles_all.append(v2_style['AE-NS v2 + Fairpost'])

        # Compute convex hull for Pareto front
        pts = np.array(list(zip(dpds, accs)))
        order = np.argsort(pts[:, 0])
        pts_sorted = pts[order]
        front_dpd, front_acc = [pts_sorted[0, 0]], [pts_sorted[0, 1]]
        for i in range(1, len(pts_sorted)):
            if pts_sorted[i, 1] > front_acc[-1]:
                front_dpd.append(pts_sorted[i, 0])
                front_acc.append(pts_sorted[i, 1])
        ax.plot(front_dpd, front_acc, '-', color='gray', alpha=0.4, lw=1.0,
                zorder=1, label='Pareto front')

        for i, (name, style) in enumerate(zip(names_all, styles_all)):
            ax.scatter(dpds[i], accs[i],
                       c=[style['color']], marker=style['marker'],
                       s=style['size'], zorder=style['zorder'],
                       edgecolors='white', linewidths=0.5,
                       label=short_labels[name])
            if name == 'AE-NS v2 (Ours)':
                ax.annotate('AE-NS v2',
                            (dpds[i], accs[i]),
                            textcoords='offset points', xytext=(8, -4),
                            fontsize=10, fontweight='bold', color='#e41a1c')

        ax.set_xlabel('Demographic Parity Diff. (DPD)')
        if ax_idx == 0:
            ax.set_ylabel('Accuracy')
        ax.set_title(ds_labels[ds])
        ax.set_xlim(-0.01, max(dpds) * 1.25)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=True,
               fontsize=8.5, bbox_to_anchor=(0.52, -0.08))
    fig.suptitle('Pareto Frontier: Accuracy vs Demographic Parity Difference',
                 fontsize=13, fontweight='bold', y=1.02)
    savefig(fig, 'fig1_pareto.pdf')
    savefig(fig, 'fig1_pareto.png')


# ====================================================================
# FIGURE 2: Sensitivity analysis  —  alpha and beta heatmaps
# ====================================================================
def fig2_sensitivity():
    datasets = ['adult', 'compas', 'credit']
    ds_labels = {'adult': 'Adult', 'compas': 'COMPAS', 'credit': 'Credit'}
    fair_metrics = ['dpd', 'eod', 'aod']
    metric_labels = {'dpd': 'DPD', 'eod': 'EOD', 'aod': 'AOD'}
    metric_colors = {'dpd': '#e41a1c', 'eod': '#377eb8', 'aod': '#4daf4a'}

    # --- 2a: Accuracy vs alpha (beta fixed at 0.05) ---
    for ds in datasets:
        alpha_data = json.load(open(f'AE_NS_Results_v2/{ds}_sensitivity_alpha.json'))
        alphas = sorted(alpha_data.keys(), key=float)
        acc_vals = [alpha_data[a]['accuracy']['mean'] for a in alphas]
        fair_vals = {m: [alpha_data[a][m]['mean'] for a in alphas] for m in fair_metrics}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        ax1.plot([float(a) for a in alphas], acc_vals, 'o-', color='black', lw=1.5, label='Accuracy')
        ax1.set_xlabel(r'$\alpha$ (adversarial weight)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Utility — {ds_labels[ds]}')
        ax1.grid(True, alpha=0.3)

        for m in fair_metrics:
            ax2.plot([float(a) for a in alphas], fair_vals[m], 'o-',
                     color=metric_colors[m], lw=1.5, label=metric_labels[m])
        ax2.set_xlabel(r'$\alpha$ (adversarial weight)')
        ax2.set_ylabel('Fairness Violation')
        ax2.set_title(f'Fairness — {ds_labels[ds]}')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Sensitivity to $\\alpha$',
                     fontsize=12, fontweight='bold')
        savefig(fig, f'fig2_sensitivity_alpha_{ds}.pdf')
        savefig(fig, f'fig2_sensitivity_alpha_{ds}.png')

    # --- 2b: Accuracy vs beta (alpha fixed at 1.0) ---
    for ds in datasets:
        beta_data = json.load(open(f'AE_NS_Results_v2/{ds}_sensitivity_beta.json'))
        betas = sorted(beta_data.keys(), key=float)
        acc_vals = [beta_data[b]['accuracy']['mean'] for b in betas]
        fair_vals = {m: [beta_data[b][m]['mean'] for b in betas] for m in fair_metrics}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        ax1.plot([float(b) for b in betas], acc_vals, 's-', color='black', lw=1.5)
        ax1.set_xlabel(r'$\beta$ (tolerance)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Utility — {ds_labels[ds]}')
        ax1.grid(True, alpha=0.3)

        for m in fair_metrics:
            ax2.plot([float(b) for b in betas], fair_vals[m], 's-',
                     color=metric_colors[m], lw=1.5, label=metric_labels[m])
        ax2.set_xlabel(r'$\beta$ (tolerance)')
        ax2.set_ylabel('Fairness Violation')
        ax2.set_title(f'Fairness — {ds_labels[ds]}')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Sensitivity to $\\beta$',
                     fontsize=12, fontweight='bold')
        savefig(fig, f'fig2_sensitivity_beta_{ds}.pdf')
        savefig(fig, f'fig2_sensitivity_beta_{ds}.png')

    # --- 2c: Combined heatmap (alpha x metric) ---
    for ds in datasets:
        alpha_data = json.load(open(f'AE_NS_Results_v2/{ds}_sensitivity_alpha.json'))
        alphas = sorted(alpha_data.keys(), key=float)
        metric_keys = ['accuracy', 'dpd', 'eod', 'aod']
        heat = np.zeros((len(alphas), len(metric_keys)))
        for i, a in enumerate(alphas):
            for j, mk in enumerate(metric_keys):
                heat[i, j] = alpha_data[a][mk]['mean']

        fig, ax = plt.subplots(figsize=(6, 3.5))
        im = ax.imshow(heat, aspect='auto', cmap='RdYlGn_r')
        ax.set_xticks(range(len(metric_keys)))
        ax.set_xticklabels(['Accuracy', 'DPD', 'EOD', 'AOD'])
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([r'$\alpha=$' + a for a in alphas])
        for i in range(len(alphas)):
            for j in range(len(metric_keys)):
                val = heat[i, j]
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                        color='white' if heat[i, j] > 0.5 else 'black')
        ax.set_title(f'AE-NS v2 Performance — {ds_labels[ds]}  (sweeping ' + r'$\alpha$' + ')')
        plt.colorbar(im, ax=ax, shrink=0.8)
        savefig(fig, f'fig2_heatmap_alpha_{ds}.pdf')
        savefig(fig, f'fig2_heatmap_alpha_{ds}.png')


# ====================================================================
# FIGURE 3: Ablation bar chart
# ====================================================================
def fig3_ablation():
    datasets = ['adult', 'compas', 'credit']
    ds_labels = {'adult': 'Adult', 'compas': 'COMPAS', 'credit': 'Credit'}
    variants = [
        'Full AE-NS',
        'w/o Reconstruction',
        'w/o Orthogonality',
        'w/o Lagrangian (Fixed Penalty)',
        'MLP Encoder',
        'Decoder Depth=2',
        'Decoder Depth=0 (Linear)',
    ]
    short = [
        'Full AE-NS',
        r'$\backslash$ Recon.',
        r'$\backslash$ Orth.',
        r'$\backslash$ Lagr.',
        'MLP Enc.',
        'Dec=2',
        'Dec=0',
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    bar_colors = ['#e41a1c'] + [PALETTE[i] for i in range(1, len(variants))]

    for row, ds in enumerate(datasets):
        abl = json.load(open(f'AE_NS_Results_v2/{ds}_ablation.json'))
        acc_vals = [abl[v]['accuracy']['mean'] for v in variants]
        acc_std  = [abl[v]['accuracy']['std']  for v in variants]
        dpd_vals = [abl[v]['dpd']['mean'] for v in variants]
        dpd_std  = [abl[v]['dpd']['std']  for v in variants]
        eod_vals = [abl[v]['eod']['mean'] for v in variants]
        eod_std  = [abl[v]['eod']['std']  for v in variants]

        x = np.arange(len(variants))
        w = 0.55

        # Accuracy subplot
        ax = axes[row, 0]
        bars = ax.bar(x, acc_vals, w, yerr=acc_std, color=bar_colors,
                      edgecolor='white', capsize=3, error_kw={'linewidth': 0.8})
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{ds_labels[ds]}')
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha='right', fontsize=8.5)
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis='y', alpha=0.3)
        bars[0].set_edgecolor('#e41a1c')
        bars[0].set_linewidth(2)

        # Fairness subplot (DPD + EOD grouped)
        ax = axes[row, 1]
        x2 = x - 0.18
        ax.bar(x2, dpd_vals, w * 0.4, yerr=dpd_std, color='#e41a1c',
               edgecolor='white', capsize=2, label='DPD', error_kw={'linewidth': 0.8})
        ax.bar(x2 + 0.36, eod_vals, w * 0.4, yerr=eod_std, color='#377eb8',
               edgecolor='white', capsize=2, label='EOD', error_kw={'linewidth': 0.8})
        ax.set_ylabel('Fairness Violation')
        ax.set_title(f'{ds_labels[ds]}')
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha='right', fontsize=8.5)
        ax.set_ylim(0, max(max(dpd_vals), max(eod_vals)) * 1.35)
        ax.grid(True, axis='y', alpha=0.3)
        if row == 0:
            ax.legend(frameon=True, fontsize=8, loc='upper right')

    fig.suptitle('Ablation Study: Impact of Each Component',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    savefig(fig, 'fig3_ablation.pdf')
    savefig(fig, 'fig3_ablation.png')


# ====================================================================
# FIGURE 4: t-SNE — Z vs X for 3 datasets
# ====================================================================
def fig4_tsne():
    datasets = ['adult', 'compas', 'credit']
    ds_labels = {'adult': 'Adult', 'compas': 'COMPAS', 'credit': 'Credit'}
    SEED = 42

    for ds in datasets:
        print(f'\n  t-SNE: {ds} ...')
        data = np.load(f'AE_NS_Results_v2/tsne_data_{ds}.npz')
        Z = data['Z']
        X = data['X']
        A = data['A']

        # Subsample if too large (>5000)
        n_test = Z.shape[0]
        if n_test > 5000:
            idx = np.random.RandomState(SEED).choice(n_test, 5000, replace=False)
            Z_plot, X_plot, A_plot = Z[idx], X[idx], A[idx]
        else:
            Z_plot, X_plot, A_plot = Z, X, A

        # t-SNE
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(Z_plot) // 4))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED,
                    n_iter=1000, init='pca', learning_rate='auto')
        Z2d = tsne.fit_transform(Z_plot)
        X2d = tsne.fit_transform(X_plot)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, emb, title in [
            (ax1, Z2d, f'Latent Z  (d={Z.shape[1]})'),
            (ax2, X2d, f'Raw Features X  (d={X.shape[1]})'),
        ]:
            for g, label, color in [(0, 'Group 0', '#377eb8'), (1, 'Group 1', '#e41a1c')]:
                mask = A_plot == g
                ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=12,
                           alpha=0.55, label=label, edgecolors='none')
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('t-SNE dim 1')
            ax.set_ylabel('t-SNE dim 2')
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(frameon=True, fontsize=9, markerscale=1.5)

        fig.suptitle(f't-SNE Projection — {ds_labels[ds]}',
                     fontsize=13, fontweight='bold', y=1.02)
        savefig(fig, f'fig4_tsne_{ds}.pdf')
        savefig(fig, f'fig4_tsne_{ds}.png')
    print('  t-SNE done.')


# ====================================================================
# FIGURE 5: NMI + classify-on-Z summary (new for revision)
# ====================================================================
def fig5_nmi_classify():
    datasets = ['adult', 'compas', 'credit']
    ds_labels = {'adult': 'Adult', 'compas': 'COMPAS', 'credit': 'Credit'}
    nmi_vals, acc_on_Z, acc_on_X, dpd_on_Z, dpd_on_X = [], [], [], [], []

    for ds in datasets:
        rev = json.load(open(f'AE_NS_Results_v2/{ds}_revision.json'))
        nmi_vals.append(rev['nmi'])
        cz = rev['classify_on_Z']
        acc_on_Z.append(cz['acc_on_Z'])
        acc_on_X.append(cz['acc_on_X'])
        dpd_on_Z.append(cz['dpd_on_Z'])
        dpd_on_X.append(cz['dpd_on_X'])

    x = np.arange(len(datasets))
    w = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) NMI bar chart
    bars = ax1.bar(x, nmi_vals, 0.55, color=[PALETTE[i] for i in range(3)],
                   edgecolor='white', capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([ds_labels[d] for d in datasets])
    ax1.set_ylabel('NMI(Z, A)')
    ax1.set_title('(a) Disentanglement\n(NMI lower = better)')
    ax1.set_ylim(0, max(nmi_vals) * 1.4)
    for i, v in enumerate(nmi_vals):
        ax1.text(i, v + max(nmi_vals) * 0.03, f'{v:.4f}', ha='center', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)

    # (b) Accuracy on Z vs X
    ax2.bar(x - w/2, acc_on_Z, w, color='#e41a1c', edgecolor='white', label='On Z (latent)')
    ax2.bar(x + w/2, acc_on_X, w, color='#377eb8', edgecolor='white', label='On X (raw features)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([ds_labels[d] for d in datasets])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('(b) Predictive Signal\n(Classify-on-Z)')
    ax2.legend(frameon=True, fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis='y', alpha=0.3)
    for i in range(len(datasets)):
        delta = acc_on_Z[i] - acc_on_X[i]
        sign = '+' if delta >= 0 else ''
        ax2.text(i, max(acc_on_Z[i], acc_on_X[i]) + 0.02,
                 f'Δ={sign}{delta:.3f}', ha='center', fontsize=9,
                 color='green' if delta >= 0 else 'red')

    # (c) DPD on Z vs X
    ax3.bar(x - w/2, dpd_on_Z, w, color='#e41a1c', edgecolor='white', label='On Z')
    ax3.bar(x + w/2, dpd_on_X, w, color='#377eb8', edgecolor='white', label='On X')
    ax3.set_xticks(x)
    ax3.set_xticklabels([ds_labels[d] for d in datasets])
    ax3.set_ylabel('Demographic Parity Diff.')
    ax3.set_title('(c) Fairness Transfer\n(DPD lower = better)')
    ax3.legend(frameon=True, fontsize=9)
    ax3.set_ylim(0, max(max(dpd_on_Z), max(dpd_on_X)) * 1.4)
    ax3.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Latent Representation Quality: Disentanglement and Predictive Signal',
                 fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()
    savefig(fig, 'fig5_nmi_classify.pdf')
    savefig(fig, 'fig5_nmi_classify.png')


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print('Generating figures (v2 data)...')
    print('\n[1/5] Pareto frontier ...')
    fig1_pareto()
    print('[2/5] Sensitivity analysis ...')
    fig2_sensitivity()
    print('[3/5] Ablation bar chart ...')
    fig3_ablation()
    print('[4/5] t-SNE (from pre-computed npz) ...')
    fig4_tsne()
    print('[5/5] NMI + classify-on-Z summary ...')
    fig5_nmi_classify()
    print(f'\nAll figures saved to {OUT}/')
