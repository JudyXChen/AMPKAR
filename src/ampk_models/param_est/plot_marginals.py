import arviz as az
import pandas as pd
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys

sys.path.append("../")
from plotting_helper_funcs import *

mpl.rcParams['figure.autolayout'] = True

samplers = ["Pathfinder", "ADVI"]
sampler_colors = {"Pathfinder": sns.color_palette("colorblind")[0],
                  "ADVI": sns.color_palette("colorblind")[1]}

data_dir = '../../../results/param_est/'
save_dir_base = '../../../results/param_est/figs/'

# Per-model configuration: free params, labels, JSON path
model_configs = {
    "MA_nonessential": {
        'info_file': '../models/MA_nonessential.json',
        'free_params': ["kOffAMP", "kOffADP", "kOffATP", "kOffCaMKK", "kPhosCaMKK",
                        "kOffLKB1", "alphaLKB1", "kPhosLKB1", "kOffPP", "alphaPP",
                        "kDephosPP", "kOffAMPK", "kPhosAMPK", "betaAMP", "kOffPP1",
                        "kDephosPP1"],
        'param_labels': {
            "kOffAMP": r'$k_{\mathrm{Off,AMP}}$',
            "kOffADP": r'$k_{\mathrm{Off,ADP}}$',
            "kOffATP": r'$k_{\mathrm{Off,ATP}}$',
            "kOffCaMKK": r'$k_{\mathrm{Off,CaMKK}}$',
            "kPhosCaMKK": r'$k_{\mathrm{Phos,CaMKK}}$',
            "kOffLKB1": r'$k_{\mathrm{Off,LKB1}}$',
            "alphaLKB1": r'$\alpha_{\mathrm{LKB1}}$',
            "kPhosLKB1": r'$k_{\mathrm{Phos,LKB1}}$',
            "kOffPP": r'$k_{\mathrm{Off,PP}}$',
            "alphaPP": r'$\alpha_{\mathrm{PP}}$',
            "kDephosPP": r'$k_{\mathrm{Dephos,PP}}$',
            "kOffAMPK": r'$k_{\mathrm{Off,AMPK}}$',
            "kPhosAMPK": r'$k_{\mathrm{Phos,AMPK}}$',
            "betaAMP": r'$\beta_{\mathrm{AMP}}$',
            "kOffPP1": r'$k_{\mathrm{Off,PP1}}$',
            "kDephosPP1": r'$k_{\mathrm{Dephos,PP1}}$',
        },
    },
    "MA_nonessential_phos": {
        'info_file': '../models/MA_nonessential_phos.json',
        'free_params': ["kOffAMP", "kOffADP", "kOffATP", "kOffCaMKK", "kPhosCaMKK",
                        "betaCaMKK", "kOffLKB1", "betaLKB1", "kPhosLKB1", "kOffPP",
                        "betaPP", "kDephosPP", "kOffAMPK", "kPhosAMPK", "betaAMP",
                        "kOffPP1", "kDephosPP1"],
        'param_labels': {
            "kOffAMP": r'$k_{\mathrm{Off,AMP}}$',
            "kOffADP": r'$k_{\mathrm{Off,ADP}}$',
            "kOffATP": r'$k_{\mathrm{Off,ATP}}$',
            "kOffCaMKK": r'$k_{\mathrm{Off,CaMKK}}$',
            "kPhosCaMKK": r'$k_{\mathrm{Phos,CaMKK}}$',
            "betaCaMKK": r'$\beta_{\mathrm{CaMKK}}$',
            "kOffLKB1": r'$k_{\mathrm{Off,LKB1}}$',
            "betaLKB1": r'$\beta_{\mathrm{LKB1}}$',
            "kPhosLKB1": r'$k_{\mathrm{Phos,LKB1}}$',
            "kOffPP": r'$k_{\mathrm{Off,PP}}$',
            "betaPP": r'$\beta_{\mathrm{PP}}$',
            "kDephosPP": r'$k_{\mathrm{Dephos,PP}}$',
            "kOffAMPK": r'$k_{\mathrm{Off,AMPK}}$',
            "kPhosAMPK": r'$k_{\mathrm{Phos,AMPK}}$',
            "betaAMP": r'$\beta_{\mathrm{AMP}}$',
            "kOffPP1": r'$k_{\mathrm{Off,PP1}}$',
            "kDephosPP1": r'$k_{\mathrm{Dephos,PP1}}$',
        },
    },
    "MM_nonessential": {
        'info_file': '../models/MM_nonessential.json',
        'free_params': ["kOffAMP", "kOffADP", "kOffATP", "kCaMKK", "KmCaMKK",
                        "kLKB1", "KmLKB1", "alphaLKB1", "kPP", "KmPP",
                        "alphaPP", "betaAMP"],
        'param_labels': {
            "kOffAMP": r'$k_{\mathrm{Off,AMP}}$',
            "kOffADP": r'$k_{\mathrm{Off,ADP}}$',
            "kOffATP": r'$k_{\mathrm{Off,ATP}}$',
            "kCaMKK": r'$k_{\mathrm{CaMKK}}$',
            "KmCaMKK": r'$K_{m,\mathrm{CaMKK}}$',
            "kLKB1": r'$k_{\mathrm{LKB1}}$',
            "KmLKB1": r'$K_{m,\mathrm{LKB1}}$',
            "alphaLKB1": r'$\alpha_{\mathrm{LKB1}}$',
            "kPP": r'$k_{\mathrm{PP}}$',
            "KmPP": r'$K_{m,\mathrm{PP}}$',
            "alphaPP": r'$\alpha_{\mathrm{PP}}$',
            "betaAMP": r'$\beta_{\mathrm{AMP}}$',
        },
    },
    "MM_nonessential_phos": {
        'info_file': '../models/MM_nonessential_phos.json',
        'free_params': ["kOffAMP", "kOffADP", "kOffATP", "kCaMKK", "KmCaMKK",
                        "betaCaMKK", "kLKB1", "KmLKB1", "betaLKB1", "kPP",
                        "KmPP", "betaPP", "betaAMP"],
        'param_labels': {
            "kOffAMP": r'$k_{\mathrm{Off,AMP}}$',
            "kOffADP": r'$k_{\mathrm{Off,ADP}}$',
            "kOffATP": r'$k_{\mathrm{Off,ATP}}$',
            "kCaMKK": r'$k_{\mathrm{CaMKK}}$',
            "KmCaMKK": r'$K_{m,\mathrm{CaMKK}}$',
            "betaCaMKK": r'$\beta_{\mathrm{CaMKK}}$',
            "kLKB1": r'$k_{\mathrm{LKB1}}$',
            "KmLKB1": r'$K_{m,\mathrm{LKB1}}$',
            "betaLKB1": r'$\beta_{\mathrm{LKB1}}$',
            "kPP": r'$k_{\mathrm{PP}}$',
            "KmPP": r'$K_{m,\mathrm{PP}}$',
            "betaPP": r'$\beta_{\mathrm{PP}}$',
            "betaAMP": r'$\beta_{\mathrm{AMP}}$',
        },
    },
}

for model, config in model_configs.items():
    print(f"\n{'='*50}\nModel: {model}\n{'='*50}")

    save_dir = save_dir_base + model + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    free_params = config['free_params']
    param_labels = config['param_labels']

    # Load idata for each sampler
    idatas = {}
    for sampler in samplers:
        fname = data_dir + model + '_Iono_mcmc_samples_' + sampler + '.nc'
        if os.path.exists(fname):
            idatas[sampler] = az.from_netcdf(fname)
            print(f"Loaded {sampler}")
        else:
            print(f"File {fname} not found, skipping {sampler}")

    if len(idatas) == 0:
        print(f"No data found for {model}, skipping.")
        continue

    # Load model info for nominal parameter values
    with open(config['info_file'], 'r') as f:
        model_info = json.load(f)
    nominal_values = model_info['nominal_params']

    ############ Grid plot: all parameters in a single figure ############
    n_params = len(free_params)
    ncols = 4
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()

    for i, param in enumerate(free_params):
        ax = axes[i]

        for sampler, idata in idatas.items():
            samples = idata.posterior[param].values.flatten()
            sns.kdeplot(samples, ax=ax, color=sampler_colors[sampler],
                        fill=True, alpha=0.3, linewidth=1.5, label=sampler,
                        log_scale=(True, False))

        # Add nominal value as vertical line
        if param in nominal_values:
            ax.axvline(nominal_values[param], color='0.3', linestyle='--',
                       linewidth=1.0, label='Nominal')

        ax.set_title(param_labels.get(param, param), fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=8)

    # Remove unused subplots
    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    # Export legend separately
    handles, labels = axes[0].get_legend_handles_labels()
    leg_fig, leg_ax = plt.subplots(figsize=(4, 1))
    leg_ax.axis('off')
    leg = leg_ax.legend(handles, labels, loc='center', fontsize=10, ncol=len(labels))
    leg_fig.savefig(save_dir + f'marginals_legend_{model}.pdf',
                    transparent=True, bbox_inches='tight')
    plt.close(leg_fig)

    # Remove per-subplot legends
    for ax in axes[:n_params]:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    fig.suptitle(f'Parameter Marginal Posteriors — {model}', fontsize=14, y=1.01)
    fig.tight_layout()

    plt.savefig(save_dir + f'marginals_grid_{model}.pdf',
                transparent=True, bbox_inches='tight')
    plt.savefig(save_dir + f'marginals_grid_{model}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved marginals grid plot.")

    ############ Summary table ############
    summary_rows = []
    for param in free_params:
        row = {'Parameter': param_labels.get(param, param),
               'Nominal': nominal_values.get(param, None)}
        for sampler, idata in idatas.items():
            samples = idata.posterior[param].values.flatten()
            row[f'{sampler} median'] = np.median(samples)
            row[f'{sampler} 5%'] = np.percentile(samples, 5)
            row[f'{sampler} 95%'] = np.percentile(samples, 95)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_dir + f'param_summary_{model}.csv', index=False)
    print(f"Saved parameter summary table.")
    print(summary_df.to_string(index=False))

    ############ Convergence diagnostics ############
    for sampler, idata in idatas.items():
        print(f"\n--- Convergence diagnostics: {model} / {sampler} ---")
        convergence = az.summary(idata, var_names=free_params,
                                 kind="diagnostics")
        convergence.to_csv(save_dir + f'convergence_{model}.csv')
        print(convergence.to_string())

        # Flag any parameters with poor convergence
        if 'r_hat' in convergence.columns:
            bad_rhat = convergence[convergence['r_hat'] > 1.05]
            if len(bad_rhat) > 0:
                print(f"\n  WARNING: {len(bad_rhat)} params with Rhat > 1.05:")
                for idx, row in bad_rhat.iterrows():
                    print(f"    {idx}: Rhat = {row['r_hat']:.3f}")
            else:
                print(f"  All Rhat <= 1.05")

        if 'ess_bulk' in convergence.columns:
            low_ess = convergence[convergence['ess_bulk'] < 100]
            if len(low_ess) > 0:
                print(f"  WARNING: {len(low_ess)} params with ESS_bulk < 100:")
                for idx, row in low_ess.iterrows():
                    print(f"    {idx}: ESS_bulk = {row['ess_bulk']:.0f}")
            else:
                print(f"  All ESS_bulk >= 100")

print("\nDone!")
