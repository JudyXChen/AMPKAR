import arviz as az
import pandas as pd
import json
import os

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import jax
import sys

sys.path.append("../")
from utils import *
from plotting_helper_funcs import *

sys.path.append("../models/")

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

mpl.rcParams['figure.autolayout'] = True

samplers = ["Pathfinder", "ADVI"] #, "Nutpie", "NUTS"]
n_trajectories = 10

wt_color = sns.color_palette("colorblind")[0]
lkb1_color = sns.color_palette("colorblind")[1]
data_color = '0.2'  # dark gray for data points

model = "MA_nonessential"

data_dir = '../../../results/param_est/'
save_dir_base = '../../../results/param_est/figs/'

fig_width = 3.5
fig_height = 2.5
fontsize_label = 10
fontsize_tick = 8
fontsize_title = 11

## Load data (times in minutes for plotting)
wt_data, _, wt_times = load_data('../../../AMPKARkey_data/HeLaAMPKAR3_RCamp_Iono.npz',
                                     to_seconds=False, constant_std=False, exclude_zero_std=True)
lkb1_data, _, lkb1_times = load_data('../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz',
                                     to_seconds=False, constant_std=False, exclude_zero_std=True)

for sampler in samplers:

    save_dir = save_dir_base + model + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load the idata
    fname = data_dir + model + '_Iono_mcmc_samples_' + sampler + '.nc'
    if not os.path.exists(fname):
        print(f"File {fname} does not exist. Skipping.")
        continue

    idata = az.from_netcdf(fname)

    ############ plot posterior predictive for each condition
    dat = {
        'WT':{'data': wt_data, 'times': wt_times, 'color': wt_color,
              'llike': 'llike_WT', 'det': 'WT', 'label': 'WT'},
        'LKB1_KD':{'data': lkb1_data, 'times': lkb1_times, 'color': lkb1_color,
                    'llike': 'llike_LKB1_KO', 'det': 'LKB1_KO', 'label': 'LKB1 KD'},
    }

    for cond in dat.keys():
        llike = dat[cond]['llike']

        if hasattr(idata, 'posterior_predictive') and llike in idata.posterior_predictive:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            fig, ax, leg = plot_predictive(idata, dat[cond]['data'], dat[cond]['times'],
                            plot_prior=False, add_t_0=True, n_traces=0, figsize=None,
                            prior_color='', post_color=dat[cond]['color'], data_color=data_color,
                            data_marker_size=20, fig_ax = (fig, ax), llike_name=llike)

            if n_trajectories > 0:
                for i in range(n_trajectories):
                    ax.plot(dat[cond]['times'],
                        jnp.squeeze(idata.posterior_predictive[llike][0,i,:].values),
                        color=dat[cond]['color'], alpha=0.15, linewidth=0.8)

            leg.remove()

            ax.set_xlabel('Time (min)', fontsize=fontsize_label)
            ax.set_ylabel('Fraction active AMPKAR', fontsize=fontsize_label)
            ax.set_title(dat[cond]['label'] + ' — Posterior Predictive', fontsize=fontsize_title)
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_ylim(0, 1.5)
            ax.set_xlim(0, None)

            fig.tight_layout()
            plt.savefig(save_dir + f'{cond}_ppc_{sampler}.pdf', transparent=True, bbox_inches='tight')
            plt.savefig(save_dir + f'{cond}_ppc_{sampler}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {cond} posterior predictive plot.")
        else:
            print(f"  {llike} not in posterior_predictive, skipping PPC plot.")

    ############ plot posterior (Deterministic) for each condition
    for cond in dat.keys():
        det = dat[cond]['det']

        if det in idata.posterior:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            trajectories = np.squeeze(idata['posterior'][det].values)

            # credible band via plot_predictive
            fig, ax, leg = plot_predictive(trajectories, dat[cond]['data'], dat[cond]['times'],
                            plot_prior=False, add_t_0=True, n_traces=0, figsize=None,
                            prior_color='', post_color=dat[cond]['color'], data_color=data_color,
                            data_marker_size=20, fig_ax = (fig, ax), llike_name=det)

            # overlay individual trajectories
            if n_trajectories > 0:
                for i in range(n_trajectories):
                    ax.plot(np.hstack(([0], dat[cond]['times'])),
                        np.hstack(([0], trajectories[i,:])),
                        color=dat[cond]['color'], alpha=0.15, linewidth=0.8)

            export_legend(leg, save_dir + f'{cond}_posterior_legend_' + sampler + '.pdf')
            leg.remove()

            ax.set_xlabel('Time (min)', fontsize=fontsize_label)
            ax.set_ylabel('Fraction active AMPKAR', fontsize=fontsize_label)
            ax.set_title(dat[cond]['label'] + ' — Model Posterior', fontsize=fontsize_title)
            ax.tick_params(labelsize=fontsize_tick)
            ax.set_ylim(0, 1.3)
            ax.set_xlim(0, None)

            fig.tight_layout()
            plt.savefig(save_dir + f'{cond}_posterior_{sampler}.pdf',
                        transparent=True, bbox_inches='tight')
            plt.savefig(save_dir + f'{cond}_posterior_{sampler}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {cond} model posterior plot.")
        else:
            print(f"  {det} not in posterior, skipping.")

print("Done!")
