"""
Plot ODE posterior trajectories for kinase KO predictions.

Uses posterior samples from inference (fitted to WT + LKB1kd) to predict
what the model expects under CaMKK2 KO (only LKB1 active).
No re-inference needed — just forward ODE solves.
"""
import arviz as az
import json

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfrx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys

sys.path.append("../")
from utils import solve_traj, load_data
from plotting_helper_funcs import *

sys.path.append("../models/")

jax.config.update("jax_enable_x64", True)
mpl.rcParams['figure.autolayout'] = True

############ Configuration ############
models = {
    "MA_nonessential": {
        'label': r'MA $\alpha$',
        'lkb1_kd_params': ['kOnLKB1', 'kPhosLKB1'],
        'camkk2_ko_params': ['kOnCaMKK', 'kPhosCaMKK'],
    },
    "MM_nonessential": {
        'label': r'MM $\alpha$',
        'lkb1_kd_params': ['LKB1tot'],
        'camkk2_ko_params': ['CaMKKtot'],
    },
    "MA_nonessential_phos": {
        'label': r'MA $\beta$',
        'lkb1_kd_params': ['kOnLKB1', 'kPhosLKB1'],
        'camkk2_ko_params': ['kOnCaMKK', 'kPhosCaMKK'],
    },
    "MM_nonessential_phos": {
        'label': r'MM $\beta$',
        'lkb1_kd_params': ['LKB1tot'],
        'camkk2_ko_params': ['CaMKKtot'],
    },
}

data_dir = '../../../results/param_est/'
save_dir_base = '../../../results/param_est/figs/'
metab_params_file = '../models/metabolism_params_Coccimiglio.json'

fig_width = 5
fig_height = 3.5
fontsize_label = 10
fontsize_tick = 8
fontsize_title = 11
n_traj = 200

# Colors
wt_color = sns.color_palette("colorblind")[0]
lkb1kd_color = sns.color_palette("colorblind")[1]
camkk2ko_color = sns.color_palette("colorblind")[2]

############ Load data (for overlay) ############
wt_data, _, wt_times = load_data(
    '../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_Iono.npz',
    to_seconds=False, constant_std=False, exclude_zero_std=True)
lkb1_data, _, lkb1_times = load_data(
    '../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1kd_Iono.npz',
    to_seconds=False, constant_std=False, exclude_zero_std=True)
_, _, times_sec = load_data(
    '../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_Iono.npz',
    to_seconds=True, constant_std=False, exclude_zero_std=True)

with open(metab_params_file, 'r') as f:
    metab_params = json.load(f)
basal_params = list(metab_params["metab_params_basal"].values())
stress_params = list(metab_params["metab_params_stress"].values())


############ Forward solve helper ############
def simulate_condition(param_samples, zero_idxs, times_sec, y0,
                       rhs, rhs_stress, ampkar_idxs, pampkar_idxs):
    """Run ODE forward for each posterior sample with specified params zeroed."""
    trajectories = []
    for i in range(len(param_samples)):
        params = list(param_samples[i])
        for idx in zero_idxs:
            params[idx] = 0.0
        params = tuple(jnp.array(p, dtype=jnp.float64) for p in params)

        try:
            sol, _ = solve_traj(rhs, rhs_stress, y0, params, times_sec,
                                pcoeff=0.3, icoeff=0.4)
            ampkar = sol[jnp.array(ampkar_idxs), :].sum(axis=0)
            pampkar = sol[jnp.array(pampkar_idxs), :].sum(axis=0)
            traj = pampkar / ampkar
            trajectories.append(np.array(traj))
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue

    return np.array(trajectories)


############ Loop over models ############
for model_name, model_cfg in models.items():
    print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")

    save_dir = save_dir_base + model_name + '/'

    # Check if inference results exist
    nc_file = data_dir + model_name + '_Iono_mcmc_samples_Pathfinder.nc'
    try:
        idata = az.from_netcdf(nc_file)
    except FileNotFoundError:
        print(f"  File not found: {nc_file}, skipping.")
        continue

    # Load model info
    model_info_file = '../models/' + model_name + '.json'
    with open(model_info_file, 'r') as f:
        model_info = json.load(f)

    param_names = model_info['params']
    state_names = list(model_info["init_conds"].keys())
    y0 = list(model_info["init_conds"].values())
    ampkar_idxs = [state_names.index(s) for s in model_info['ampkar_states']]
    pampkar_idxs = [state_names.index(s) for s in model_info['pampkar_states']]

    # Build ODE RHS
    exec('from ' + model_name + '_diffrax import *')
    rhs = eval(model_name + '(' + ','.join(str(p) for p in basal_params) + ')')
    rhs_stress = eval(model_name + '(' + ','.join(str(p) for p in stress_params) + ')')
    rhs = dfrx.ODETerm(rhs)
    rhs_stress = dfrx.ODETerm(rhs_stress)

    # Load posterior samples
    posterior_dict = idata.posterior.to_dict()
    n_total = np.array(list(posterior_dict['data_vars'].values())[0]['data']).reshape(-1).shape[0]
    rng = np.random.default_rng(seed=42)
    sample_idxs = rng.choice(n_total, size=n_traj, replace=False)

    param_samples = np.zeros((n_traj, len(param_names)))
    for j, pname in enumerate(param_names):
        if pname in posterior_dict['data_vars']:
            all_samples = np.array(posterior_dict['data_vars'][pname]['data']).reshape(-1)
            param_samples[:, j] = all_samples[sample_idxs]
        else:
            param_samples[:, j] = float(idata.constant_data[pname].values.flatten()[0])

    print(f"  Loaded {n_traj} posterior samples")

    # Define KO conditions
    lkb1_kd_idxs = [param_names.index(p) for p in model_cfg['lkb1_kd_params']]
    camkk2_ko_idxs = [param_names.index(p) for p in model_cfg['camkk2_ko_params']]

    conditions = {
        'WT': {
            'zero_idxs': [],
            'color': wt_color,
            'label': 'LKB1wt',
            'data': wt_data,
            'times_min': wt_times,
        },
        'LKB1kd': {
            'zero_idxs': lkb1_kd_idxs,
            'color': lkb1kd_color,
            'label': 'LKB1kd',
            'data': lkb1_data,
            'times_min': lkb1_times,
        },
        'CaMKK2_KO': {
            'zero_idxs': camkk2_ko_idxs,
            'color': camkk2ko_color,
            'label': 'CaMKK2 KO',
            'data': None,
            'times_min': wt_times,
        },
    }

    # Forward solve for each condition
    results = {}
    for cond_name, cond_info in conditions.items():
        print(f"  Simulating {cond_name}...")
        results[cond_name] = simulate_condition(
            param_samples, cond_info['zero_idxs'], times_sec, y0,
            rhs, rhs_stress, ampkar_idxs, pampkar_idxs)
        print(f"    {results[cond_name].shape[0]} successful trajectories")

    # Plot: all conditions overlaid
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for cond_name, cond_info in conditions.items():
        traj = results[cond_name]
        if len(traj) == 0:
            continue

        times_min = cond_info['times_min']
        median = np.median(traj, axis=0)
        q05 = np.percentile(traj, 5, axis=0)
        q95 = np.percentile(traj, 95, axis=0)

        t_plot = np.hstack(([0], times_min))
        median_plot = np.hstack(([0], median))
        q05_plot = np.hstack(([0], q05))
        q95_plot = np.hstack(([0], q95))

        ax.fill_between(t_plot, q05_plot, q95_plot,
                        color=cond_info['color'], alpha=0.2)
        ax.plot(t_plot, median_plot, color=cond_info['color'],
                linewidth=1.5, label=cond_info['label'])

    # Overlay data as dashed lines
    ax.plot(wt_times, wt_data, color=wt_color, linestyle='--',
            linewidth=1.0, zorder=5)
    ax.plot(lkb1_times, lkb1_data, color=lkb1kd_color, linestyle='--',
            linewidth=1.0, zorder=5)

    ax.set_xlabel('Time (min)', fontsize=fontsize_label)
    ax.set_ylabel('Fraction active AMPKAR', fontsize=fontsize_label)
    ax.set_title(f"{model_cfg['label']} — Kinase KO Predictions", fontsize=fontsize_title)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    plt.savefig(save_dir + 'KO_predictions_overlay.pdf',
                transparent=True, bbox_inches='tight')
    plt.savefig(save_dir + 'KO_predictions_overlay.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved overlay prediction plot.")

print("\nDone!")
