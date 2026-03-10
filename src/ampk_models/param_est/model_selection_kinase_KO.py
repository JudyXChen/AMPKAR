import arviz as az
import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
import sys
from scipy.stats import norm

sys.path.append("../")
from utils import load_data
from plotting_helper_funcs import *

mpl.rcParams['figure.autolayout'] = True

############ Configuration ############
# Samplers to compare
samplers = ["Pathfinder"] #, "ADVI"]
sampler_colors = {
    "Pathfinder": sns.color_palette("colorblind")[0],
    "ADVI": sns.color_palette("colorblind")[1],
    # Add more samplers here, e.g.:
    # "NUTS": sns.color_palette("colorblind")[2],
    # "Nutpie": sns.color_palette("colorblind")[3],
}

# Models to compare
models = {
    "MA_nonessential": {
        'info_file': '../models/MA_nonessential.json',
        'label': r'MA $\alpha$',
    },
    "MM_nonessential": {
        'info_file': '../models/MM_nonessential.json',
        'label': r'MM $\alpha$',
    },
    "MA_nonessential_phos": {
        'info_file': '../models/MA_nonessential_phos.json',
        'label': r'MA $\beta$',
    },
    "MM_nonessential_phos": {
        'info_file': '../models/MM_nonessential_phos.json',
        'label': r'MM $\beta$',
    },
}

# Conditions and their data files
conditions = {
    'WT': {
        'data_file': '../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_Iono.npz',
        'llike_name': 'llike_WT',
        'det_name': 'WT',
    },
    'LKB1_KO': {
        'data_file': '../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1kd_Iono.npz',
        'llike_name': 'llike_LKB1_KO',
        'det_name': 'LKB1_KO',
    },
}

data_dir = '../../../results/param_est/'
save_dir = '../../../results/param_est/figs/model_comparison/'
data_std_max = 0.1  # must match what was used in inference

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

############ Load data std for log-likelihood computation ############
data_stds = {}
for cond, cond_info in conditions.items():
    _, data_std, _ = load_data(cond_info['data_file'], to_seconds=True,
                               constant_std=False, exclude_zero_std=True)
    scale = data_std_max / np.max(data_std)
    data_stds[cond] = data_std.reshape(1, len(data_std)) * scale

############ Load idata and compute log-likelihood ############
idata_dict = {}  # {(model_name, sampler): idata}

for model_name, model_info in models.items():
    for sampler in samplers:

        fname = data_dir + model_name + '_Iono_mcmc_samples_' + sampler + '.nc'
        if not os.path.exists(fname):
            print(f"  File not found: {fname}, skipping.")
            continue

        idata = az.from_netcdf(fname)
        print(f"Loaded {model_name} / {sampler}")

        # Compute log_likelihood from posterior Deterministic trajectories
        # log_lik(chain, draw, obs) = log Normal(y_obs | mu, sigma)
        log_lik_dict = {}
        for cond, cond_info in conditions.items():
            det_name = cond_info['det_name']
            llike_name = cond_info['llike_name']

            if det_name not in idata.posterior:
                print(f"  {det_name} not in posterior for {model_name}/{sampler}, skipping.")
                continue

            # mu: model predictions, shape (nchains, ndraws, 1, ntime)
            mu = idata.posterior[det_name].values
            # observed data, shape (1, ntime)
            y_obs = idata.observed_data[llike_name].values
            # data_std, shape (1, ntime)
            sigma = data_stds[cond]

            # Compute log-likelihood: shape (nchains, ndraws, 1, ntime)
            log_lik = norm.logpdf(y_obs, loc=mu, scale=sigma)
            log_lik_dict[llike_name] = (["chain", "draw", "__obs__0", "__obs__1"], log_lik)

        # Add log_likelihood group to idata
        import xarray as xr
        ll_ds = xr.Dataset({k: xr.DataArray(data=v[1], dims=v[0])
                            for k, v in log_lik_dict.items()})
        idata.add_groups({"log_likelihood": ll_ds})

        idata_dict[(model_name, sampler)] = idata

############ Compute ELPD (LOO) per condition ############
print("\n" + "="*60)
print("ELPD (LOO) Results")
print("="*60)

elpd_results = []

for cond, cond_info in conditions.items():
    llike_name = cond_info['llike_name']
    print(f"\n--- Condition: {cond} ({llike_name}) ---")

    for (model_name, sampler), idata in idata_dict.items():
        if llike_name not in idata.log_likelihood:
            continue

        try:
            loo = az.loo(idata, var_name=llike_name)
            elpd = loo.elpd_loo
            se = loo.se
            p_loo = loo.p_loo
            print(f"  {model_name} / {sampler}: ELPD = {elpd:.2f} +/- {se:.2f}, p_loo = {p_loo:.2f}")

            elpd_results.append({
                'model': model_name,
                'model_label': models[model_name]['label'],
                'sampler': sampler,
                'condition': cond,
                'elpd_loo': elpd,
                'se': se,
                'p_loo': p_loo,
            })
        except Exception as e:
            print(f"  {model_name} / {sampler}: LOO failed - {e}")

elpd_df = pd.DataFrame(elpd_results)

############ ELPD comparison using az.compare() ############
# Compare across samplers for each condition (and later across models too)
print("\n" + "="*60)
print("Model Comparison (az.compare)")
print("="*60)

compare_results = {}
for cond, cond_info in conditions.items():
    llike_name = cond_info['llike_name']

    compare_dict = {}
    for (model_name, sampler), idata in idata_dict.items():
        if llike_name in idata.log_likelihood:
            label = models[model_name]['label']
            compare_dict[label] = idata

    if len(compare_dict) >= 2:
        try:
            comp = az.compare(compare_dict, var_name=llike_name)
            compare_results[cond] = comp
            print(f"\n--- {cond} ---")
            print(comp[['rank', 'elpd_loo', 'se', 'dse', 'weight']].to_string())
        except Exception as e:
            print(f"  Compare failed for {cond}: {e}")

############ Plot: ELPD bar chart per condition ############
# Enforce model ordering to match dict insertion order (Models 3-6)
model_order = list(models.keys())

if len(elpd_df) > 0:
    for cond in elpd_df['condition'].unique():
        cond_df = elpd_df[elpd_df['condition'] == cond].copy()
        cond_df['model'] = pd.Categorical(cond_df['model'], categories=model_order, ordered=True)
        cond_df = cond_df.sort_values('model')

        fig, ax = plt.subplots(figsize=(max(3, len(cond_df) * 1.2), 3))

        x_labels = [row['model_label'] for _, row in cond_df.iterrows()]
        x_pos = np.arange(len(cond_df))
        colors_list = [sns.color_palette("colorblind")[i] for i in range(len(cond_df))]

        bars = ax.bar(x_pos, cond_df['elpd_loo'].values, yerr=cond_df['se'].values,
                       capsize=4, color=colors_list, edgecolor='black', linewidth=0.8, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('ELPD (LOO)', fontsize=11)
        ax.set_title(f'{cond} — Expected Log Pointwise Predictive Density', fontsize=12)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        plt.savefig(save_dir + f'elpd_{cond}.pdf', transparent=True, bbox_inches='tight')
        plt.savefig(save_dir + f'elpd_{cond}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved ELPD plot for {cond}")

    ############ Plot: Combined ELPD (averaged over conditions) ############
    avg_elpd = elpd_df.groupby(['model', 'model_label', 'sampler'], sort=False).agg(
        avg_elpd=('elpd_loo', 'mean'),
        avg_se=('se', 'mean')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(max(3, len(avg_elpd) * 1.2), 3))

    x_labels = [row['model_label'] for _, row in avg_elpd.iterrows()]
    x_pos = np.arange(len(avg_elpd))
    colors_list = [sns.color_palette("colorblind")[i] for i in range(len(avg_elpd))]

    bars = ax.bar(x_pos, avg_elpd['avg_elpd'].values, yerr=avg_elpd['avg_se'].values,
                   capsize=4, color=colors_list, edgecolor='black', linewidth=0.8, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel('ELPD (LOO)', fontsize=11)
    ax.set_title('Average ELPD across Conditions', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.tick_params(labelsize=9)

    # Legend for samplers
    handles = [mpatches.Patch(facecolor=sampler_colors[s], edgecolor='black',
               alpha=0.8, label=s) for s in samplers if s in sampler_colors]
    ax.legend(handles=handles, fontsize=9, loc='best')

    fig.tight_layout()
    plt.savefig(save_dir + f'elpd_combined.pdf', transparent=True, bbox_inches='tight')
    plt.savefig(save_dir + f'elpd_combined.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved combined ELPD plot.")

    # Save results to CSV
    elpd_df.to_csv(save_dir + 'elpd_results.csv', index=False)
    print(f"Saved ELPD results to {save_dir}elpd_results.csv")

print("\nDone!")
