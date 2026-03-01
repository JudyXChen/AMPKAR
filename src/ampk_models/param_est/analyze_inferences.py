import arviz as az
import pandas as pd
import json
import os

import numpy as np
import diffrax
import matplotlib.pyplot as plt
import matplotlib as mpl
import met_brewer as mb
import seaborn as sns
import jax
import sys

sys.path.append("../")
from utils import *
from plotting_helper_funcs import *

sys.path.append("../models/")

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

plt.style.use('custom')
mpl.rcParams['figure.autolayout'] = True
colors = mb.met_brew(name="Egypt", n=3)

print(colors)

samplers = ["Nutpie"] #["ADVI"] #, "NUTS"] # list of samplers used
n_trajectories = 10 # number of trajectories to add to PPC plot

cyto_color = colors[0]
lyso_color = colors[1]
mito_color = colors[2]


models_free_params = { 
        # "ampk_Coccimiglio": {'free':["Km_ADP_pAMPK","k_ADP_pAMPK","Km16",
        #                              "Km_AMP_pAMPK","VmaxkinaseADP","k10r",
        #                              "Km_pAMPK","Km13","Km_ATP_pAMPK","k_AMP_pAMPK",
        #                              "Km17","Vmaxppase","k_ATP_pAMPK","Km15","k11r",
        #                              "k9r","Km14","VmaxppaseATP","Km18","k7r","Km12",
        #                              "k_pAMPK"],
        #                     'names':[r'$K_{m,ADP,pAMPK}$',r'$k_{ADP,pAMPK}$',r'$K_{m16}$',
        #                              r'$K_{m,AMP,pAMPK}$', r'$V_{max,kinase,ADP}$',
        #                              r'$k_{10r}$',r'$K_{m,pAMPK}$',r'$K_{m13}$',
        #                              r'$K_{m,ATP,pAMPK}$',r'$K_{m,AMP,pAMPK}$',r'$K_{m17}$',r'$V_{max,ppase}$',
        #                              r'$k_{ATP,pAMPK}$',r'$K_{m15}$',r'$k_{11r}$',r'$k_{9r}$',
        #                              r'$K_{m14}$',r'$V_{max,ppase,ATP}$',r'$K_{m18}$',
        #                              r'$k_{7r}$',r'$K_{m12}$',r'$k_{pAMPK}$'],
        #                     'info_file': '../models/ampk_Coccimiglio.json',
        #                     },
        # "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffAMPK",
        #                       "kPhosAMPK","kOffPP1","kDephosPP1"],
        #             'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                      r'$k_{\text{OffATP}}$',r'$k_{\text{OffAMPK}}$',
        #                      r'$k_{\text{PhosAMPK}}$',r'$k_{\text{OffPP1}}$',
        #                      r'$k_{\text{Dephos,PP1}}$'],
        #             'info_file':'../models/MA_single.json',
        #             },
        # "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffPP","kPhosAMPK",
        #                       "kOffAMPK","kOffPP1","kDephosPP1"],
        #             'names':[r'$k_{\text{Off,AMP}}$',r'$k_{\text{Off,ADP}}$',
        #                      r'$k_{\text{Off,ATP}}$', r'$k_{\text{Off,PP}}$',
        #                      r'$k_{\text{Phos,AMPK}}$', r'$k_{\text{Off,AMPK}}$',
        #                      r'$k_{\text{Off,PP1}}$',r'$k_{\text{Dephos,PP1}}$'],
        #             'info_file':'../models/MA_single.json',
        #             },
        "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP"],
                    'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                             r'$k_{\text{OffATP}}$',r'$k_{\text{PhosCaMKK}}$',
                             r'$k_{\text{PhosLKB1}}$',r'$K_{\text{m,LKB1}}$'],
                    'info_file': '../models/MM_single.json',
                      },
        # "MA_nonessential": {'free':["kOffAMP","kOffADP","kOffATP","kPhosCaMKK",
        #                             "kOffPP","alphaPP","kOffAMPK","kPhosAMPK",
        #                             "betaAMP","kOffPP1","kDephosPP1"],
        #             'names':[r'$k_{\text{Off,AMP}}$',r'$k_{\text{Off,ADP}}$',
        #                      r'$k_{\text{Off,ATP}}$',r'$k_{\text{Phos,CaMKK}}$',
        #                      r'$k_{\text{Off,PP}}$',r'$\alpha_{\text{PP}}$',
        #                      r'$k_{\text{Off,AMPK}}$', r'$k_{\text{Phos,AMPK}}$',
        #                      r'$\beta_{\text{AMP}}$',r'$k_{\text{OffPP1}}$',
        #                      r'$k_{\text{Dephos,PP1}}$'],
        #             'info_file': '../models/MA_nonessential.json',
        #                 },
        # "MM_nonessential":  {'free':["kOffAMP","kOffADP","kOffATP","betaAMP"],
        #             'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                      r'$k_{\text{OffATP}}$',r'$\beta_{\text{AMP}}$'],
        #             'info_file': '../models/MM_nonessential.json',
        #            }
        }

data_dir = '../../../results/param_est/'
save_dir_base = '../../../results/param_est/figs/'

## Load data
cyto_data, _, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', 
                                     to_seconds=False, constant_std=False)
lyso_data, _, lyso_times = load_data('../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz', 
                                     to_seconds=False, constant_std=False)
mito_data, _, mito_times = load_data('../../../Schmitt_et_al_2022_data/fig_2c_mito.npz', 
                                     to_seconds=False, constant_std=False)

# dictionary to store idata
idata_dict = {
    'cyto':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()},
    'lyso':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()},
    'mito':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()}
}

for i, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {i+1}: {model}")

    for sampler in samplers:
    
        save_dir = save_dir_base + model + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load the idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')
        idata_lyso = az.from_netcdf(data_dir + model + '_lyso_mcmc_samples_' + sampler + '.nc')
        idata_mito = az.from_netcdf(data_dir + model + '_mito_mcmc_samples_' + sampler + '.nc')
        # prior_pred_idata = az.from_netcdf(data_dir + model + '_cyto_prior_samples.nc')

        # store idata
        idata_dict['cyto'][model][sampler] = idata_cyto
        idata_dict['lyso'][model][sampler] = idata_lyso
        idata_dict['mito'][model][sampler] = idata_mito

        # ########### plot traces
        az.plot_trace(idata_cyto)
        plt.savefig(save_dir + 'cyto_trace_' + sampler + '.png', dpi=500)
        az.plot_trace(idata_lyso)
        plt.savefig(save_dir + 'lyso_trace_' + sampler + '.png', dpi=500)
        az.plot_trace(idata_mito)
        plt.savefig(save_dir + 'mito_trace_' + sampler + '.png', dpi=500)

        ########### convergence metrics
        with open(model + '_convergence_' + sampler + '.txt', 'w') as file:
            summary = az.summary(idata_cyto)
            file.write("Cytosol:\n")
            file.write(summary.to_string())
            file.write("\n\n")

            summary = az.summary(idata_lyso)
            file.write("Lysosome:\n")
            file.write(summary.to_string())
            file.write("\n\n")

            summary = az.summary(idata_mito)
            file.write("Mitochondria:\n")
            file.write(summary.to_string())
            file.write("\n\n")

        ############ plot 1D marginals for each model param colored by compartment
        posterior_idata_cyto = idata_cyto.posterior.to_dataframe()
        posterior_idata_cyto['compartment'] = 'cyto'
        posterior_idata_lyso = idata_lyso.posterior.to_dataframe()
        posterior_idata_lyso['compartment'] = 'lyso'
        posterior_idata_mito = idata_mito.posterior.to_dataframe()
        posterior_idata_mito['compartment'] = 'mito'

        # Combine the three dataframes into one
        combined_posterior = pd.concat([posterior_idata_cyto, posterior_idata_lyso, posterior_idata_mito])
        # Reset index to avoid duplicate indices
        combined_posterior.reset_index(drop=True, inplace=True)

        # make folder to 1D marginal plots
        if not os.path.exists(save_dir + 'marginals/'):
            os.makedirs(save_dir + 'marginals/')
    
        # make the plots
        for i, param in enumerate(models_free_params[model]['free']): #combined_posterior.columns:
            if param != 'compartment':
                fig, ax = get_sized_fig_ax(0.6, 0.4)
                sns.kdeplot(data=combined_posterior, x=param, hue='compartment', ax=ax,
                            fill=True, palette={'cyto':cyto_color, 'lyso':lyso_color, 'mito':mito_color},
                            legend=False, log_scale=(True, False), linewidth=1.0)
                ax.set_xlabel("", fontsize=8.0)
                ax.set_ylabel("", fontsize=8.0)
                # ax.set_xscale("log")
                # for label in ax.get_xticklabels() + ax.get_yticklabels():
                #     label.set_fontsize(8)
                ax.tick_params(axis='both', which='major', labelsize=8)

                leg = ax.legend(['Mitochondria', 'Lysosome', 'Cytoplasm'], loc='upper right', fontsize=8.0, bbox_to_anchor=(3, 1))
                leg.set_title("")
                export_legend(leg, save_dir + 'marginals/' + f'{param}_dist_legend_' + sampler + '.pdf')
                leg.remove()

                # ax.ticklabel_format(style='plain', axis='x')
                # ax.ticklabel_format(style='plain', axis='y')

                ax.set_title(models_free_params[model]['names'][i], fontsize=10.0,
                             fontweight='normal', pad=0)

                plt.savefig(save_dir + 'marginals/' + f'{param}_dist_' + sampler + '.pdf', transparent=True,
                            bbox_inches='tight')
                plt.close()

        # ############ plot prior predictive
        # fig, ax = get_sized_fig_ax(2,1)

        # fig, ax, leg = plot_predictive(prior_pred_idata, cyto_data, cyto_times, 
        #                 plot_post=False, add_t_0=True, n_traces=0, figsize=None, 
        #                 prior_color=cyto_color, post_color=cyto_color, data_color=cyto_color, 
        #                 data_marker_size=10, fig_ax = (fig, ax))
    
        # # add n_trajectories to the plot if n_trajectories > 0
        # if n_trajectories > 0:
        #     for i in range(n_trajectories):
        #         ax.plot(cyto_times, 
        #             jnp.squeeze(prior_pred_idata.prior_predictive["llike"][0,i,:].values), 
        #             color=cyto_color, alpha=0.2, linewidth=1.0)
                
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # ax.set_ylim(0, 1.5)

        # plt.savefig(save_dir + 'prior_pred_' + '.pdf', transparent=True, bbox_inches='tight')


        # # ############ plot sims with prior_samples for each model
        # # run prior simulations
        # if os.path.isfile(save_dir + 'prior_sims.npy'):
        #     print(f"File {save_dir + 'prior_sims.npy'} already exists.")
        #     prior_sims_cyto = np.load(save_dir + 'prior_sims.npy')
        # else:
        #     model_info_file = json.load(open(models_free_params[model]['info_file']))
        #     param_samples = get_param_subsample(model_info_file['params'], prior_pred_idata, 1000, prior_or_post="prior", rng=np.random.default_rng(seed=1234))
            
        #     prior_sims_cyto = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
        #                     '../models/metabolism_params_Coccimiglio.json',
        #                     cyto_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
        #                     icoeff=0.4)
        #     np.save(save_dir + 'prior_sims.npy', np.array(prior_sims_cyto))

        # # plot
        # fig, ax = get_sized_fig_ax(2,1)

        # # fig, ax, leg = plot_predictive(prior_sims_cyto, cyto_data, cyto_times, 
        # #                 plot_prior=False, plot_post=True, add_t_0=True, n_traces=0, figsize=None, 
        # #                 prior_color='', post_color=cyto_color, data_color=cyto_color, 
        # #                 data_marker_size=10, fig_ax = (fig, ax))
    
        # # # add n_trajectories to the plot if n_trajectories > 0
        # if n_trajectories > 0:
        #     for i in range(n_trajectories):
        #         ax.plot(cyto_times, 
        #             jnp.squeeze(prior_sims_cyto[i,:]), 
        #             color=cyto_color, alpha=0.2, linewidth=1.0)
                
        # # export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
        # leg.remove()
                
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # ax.set_ylim(0, 1.5)

        # plt.savefig(save_dir + f'prior.pdf', transparent=True, bbox_inches='tight')

#         ############ plot posterior predictive for each model
        dat = {
            'cyto':{'idata': idata_cyto, 'data': cyto_data, 'times': cyto_times, 'color': cyto_color},
            'lyso':{'idata': idata_lyso, 'data': lyso_data, 'times': lyso_times, 'color': lyso_color},
            'mito':{'idata': idata_mito, 'data': mito_data, 'times': mito_times, 'color': mito_color}
        }
        for comp in dat.keys():
            fig, ax = get_sized_fig_ax(2,1)
    
            fig, ax, leg = plot_predictive(dat[comp]['idata'], dat[comp]['data'], dat[comp]['times'], 
                            plot_prior=False, add_t_0=True, n_traces=0, figsize=None, 
                            prior_color='', post_color=dat[comp]['color'], data_color=dat[comp]['color'], 
                            data_marker_size=10, fig_ax = (fig, ax))
        
            # add n_trajectories to the plot if n_trajectories > 0
            if n_trajectories > 0:
                for i in range(n_trajectories):
                    ax.plot(dat[comp]['times'], 
                        jnp.squeeze(dat[comp]['idata'].posterior_predictive["llike"][0,i,:].values), 
                        color=dat[comp]['color'], alpha=0.2, linewidth=1.0)
                    
            export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
            leg.remove()
                   
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_ylim(0, 1.5)

            plt.savefig(save_dir + f'{comp}_ppc_' + sampler + '.pdf', transparent=True, bbox_inches='tight')


        # ############ plot posterior for each model
        # run posterior simulations
        # cyto
        if os.path.isfile(save_dir + 'post_sims_cyto.npy') & \
            os.path.isfile(save_dir + 'post_sims_cyto_LKB1_KD.npy') & \
            os.path.isfile(save_dir + 'post_sims_cyto_CaMKK_KD.npy'):
            print(f"File {save_dir + 'post_sims_cyto.npy'} already exists.")
            post_sims_cyto = np.load(save_dir + 'post_sims_cyto.npy')
            post_sims_cyto_LKB1_KD = np.load(save_dir + 'post_sims_cyto_LKB1_KD.npy')
            post_sims_cyto_CaMKK_KD = np.load(save_dir + 'post_sims_cyto_CaMKK_KD.npy')
        else:
            model_info_file = json.load(open(models_free_params[model]['info_file']))
            param_samples = get_param_subsample(model_info_file['params'], idata_cyto, 400, prior_or_post="post", rng=np.random.default_rng(seed=1234))
            post_sims_cyto = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
                            '../models/metabolism_params_Coccimiglio.json',
                            cyto_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_cyto.npy', np.array(post_sims_cyto))

            # LKB1 KD
            if 'MA' in model:
                k_cat_lkb1_idx = model_info_file['params'].index('kPhosLKB1')
            elif 'MM' in model:
                k_cat_lkb1_idx = model_info_file['params'].index('kLKB1')

            param_samples_LKB1_KD = param_samples.copy()
            param_samples_LKB1_KD[:,k_cat_lkb1_idx] = 0.0

            post_sims_cyto_LKB1_KD = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
                            '../models/metabolism_params_Coccimiglio.json',
                            cyto_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_cyto_LKB1_KD.npy', np.array(post_sims_cyto_LKB1_KD))

            # CaMKK KD
            if 'MA' in model:
                k_cat_camkk_idx = model_info_file['params'].index('kPhosCaMKK')
            elif 'MM' in model:
                k_cat_camkk_idx = model_info_file['params'].index('kCaMKK')

            param_samples_CaMKK_KD = param_samples.copy()
            param_samples_CaMKK_KD[:,k_cat_camkk_idx] = 0.0

            post_sims_cyto_CaMKK_KD = run_simulations(param_samples, model, models_free_params[model]['info_file'],
                            '../models/metabolism_params_Coccimiglio.json',
                            cyto_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_cyto_CaMKK_KD.npy', np.array(post_sims_cyto_CaMKK_KD))

        # lyso
        if os.path.isfile(save_dir + 'post_sims_lyso.npy') & \
            os.path.isfile(save_dir + 'post_sims_lyso_LKB1_KD.npy') & \
            os.path.isfile(save_dir + 'post_sims_lyso_CaMKK_KD.npy'):
            print(f"File {save_dir + 'post_sims_lyso.npy'} already exists.")
            post_sims_lyso = np.load(save_dir + 'post_sims_lyso.npy')
            post_sims_lyso_LKB1_KD = np.load(save_dir + 'post_sims_lyso_LKB1_KD.npy')
            post_sims_lyso_CaMKK_KD = np.load(save_dir + 'post_sims_lyso_CaMKK_KD.npy')
        else:
            model_info_file = json.load(open(models_free_params[model]['info_file']))
            param_samples = get_param_subsample(model_info_file['params'], idata_lyso, 400, prior_or_post="post", rng=np.random.default_rng(seed=1234))
            post_sims_lyso = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
                            '../models/metabolism_params_Coccimiglio.json',
                            lyso_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_lyso.npy', np.array(post_sims_lyso))

            # LKB1 KD
            if 'MA' in model:
                k_cat_lkb1_idx = model_info_file['params'].index('kPhosLKB1')
            elif 'MM' in model:
                k_cat_lkb1_idx = model_info_file['params'].index('kLKB1')

            param_samples_LKB1_KD = param_samples.copy()
            param_samples_LKB1_KD[:,k_cat_lkb1_idx] = 0.0

            post_sims_lyso_LKB1_KD = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
                            '../models/metabolism_params_Coccimiglio.json',
                            lyso_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_lyso_LKB1_KD.npy', np.array(post_sims_lyso_LKB1_KD))

            # CaMKK KD
            if 'MA' in model:
                k_cat_camkk_idx = model_info_file['params'].index('kPhosCaMKK')
            elif 'MM' in model:
                k_cat_camkk_idx = model_info_file['params'].index('kCaMKK')

            param_samples_CaMKK_KD = param_samples.copy()
            param_samples_CaMKK_KD[:,k_cat_camkk_idx] = 0.0

            post_sims_lyso_CaMKK_KD = run_simulations(param_samples, model, models_free_params[model]['info_file'],
                            '../models/metabolism_params_Coccimiglio.json',
                            lyso_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_lyso_CaMKK_KD.npy', np.array(post_sims_lyso_CaMKK_KD))

        # mito
        if os.path.isfile(save_dir + 'post_sims_mito.npy') & \
            os.path.isfile(save_dir + 'post_sims_mito_LKB1_KD.npy') & \
            os.path.isfile(save_dir + 'post_sims_mito_CaMKK_KD.npy'):
            print(f"File {save_dir + 'post_sims_mito.npy'} already exists.")
            post_sims_mito = np.load(save_dir + 'post_sims_mito.npy')
            post_sims_mito_LKB1_KD = np.load(save_dir + 'post_sims_mito_LKB1_KD.npy')
            post_sims_mito_CaMKK_KD = np.load(save_dir + 'post_sims_mito_CaMKK_KD.npy')
        else:
            model_info_file = json.load(open(models_free_params[model]['info_file']))
            param_samples = get_param_subsample(model_info_file['params'], idata_mito, 400, prior_or_post="post", rng=np.random.default_rng(seed=1234))
            post_sims_mito = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
                            '../models/metabolism_params_Coccimiglio.json',
                            mito_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_mito.npy', np.array(post_sims_mito))

            # LKB1 KD
            if 'MA' in model:
                k_cat_lkb1_idx = model_info_file['params'].index('kPhosLKB1')
            elif 'MM' in model:
                k_cat_lkb1_idx = model_info_file['params'].index('kLKB1')

            param_samples_LKB1_KD = param_samples.copy()
            param_samples_LKB1_KD[:,k_cat_lkb1_idx] = 0.0

            post_sims_mito_LKB1_KD = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
                            '../models/metabolism_params_Coccimiglio.json',
                            mito_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_mito_LKB1_KD.npy', np.array(post_sims_mito_LKB1_KD))

            # CaMKK KD
            if 'MA' in model:
                k_cat_camkk_idx = model_info_file['params'].index('kPhosCaMKK')
            elif 'MM' in model:
                k_cat_camkk_idx = model_info_file['params'].index('kCaMKK')

            param_samples_CaMKK_KD = param_samples.copy()
            param_samples_CaMKK_KD[:,k_cat_camkk_idx] = 0.0

            post_sims_mito_CaMKK_KD = run_simulations(param_samples, model, models_free_params[model]['info_file'],
                            '../models/metabolism_params_Coccimiglio.json',
                            mito_times*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                            icoeff=0.4)
            np.save(save_dir + 'post_sims_mito_CaMKK_KD.npy', np.array(post_sims_mito_CaMKK_KD))

        ############ plots
        dat = {
            'cyto':{'sims': post_sims_cyto, 'data': cyto_data, 'times': cyto_times, 'color': cyto_color},
            'lyso':{'sims': post_sims_lyso, 'data': lyso_data, 'times': lyso_times, 'color': lyso_color},
            'mito':{'sims': post_sims_mito, 'data': mito_data, 'times': mito_times, 'color': mito_color}
        }
        for comp in dat.keys():
            fig_width, fig_height = 1.75, 0.5
            fig, ax = get_sized_fig_ax(fig_width, fig_height)
    
            fig, ax, leg = plot_predictive(dat[comp]['sims'], dat[comp]['data'], dat[comp]['times'], 
                            plot_prior=False, add_t_0=True, n_traces=0, figsize=None, 
                            prior_color='', post_color=dat[comp]['color'], data_color=dat[comp]['color'], 
                            data_marker_size=10, fig_ax = (fig, ax))
        
            # add n_trajectories to the plot if n_trajectories > 0
            if n_trajectories > 0:
                for i in range(n_trajectories):
                    ax.plot(dat[comp]['times'], 
                        jnp.squeeze(dat[comp]['sims'][i,:]), 
                        color=dat[comp]['color'], alpha=0.2, linewidth=1.0)
                    
            export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
            leg.remove()
                   
            ax.set_xlabel("time (min)", fontsize=10.0)
            ax.set_ylabel("")
            ax.set_ylim(0, 1.5)

            plt.savefig(save_dir + f'{comp}_posterior_' + sampler + '.pdf', transparent=True, bbox_inches='tight')
    

        ############ plot response to LKB1 and CaMKK KD

# ############ compare models using ELPD

# # repeat for each sampler
# for sampler in samplers:
#     cyto_dat = {model:idata_dict['cyto'][model][sampler] for model in idata_dict['cyto'].keys()}
#     lyso_dat = {model:idata_dict['lyso'][model][sampler] for model in idata_dict['lyso'].keys()}
#     mito_dat = {model:idata_dict['mito'][model][sampler] for model in idata_dict['mito'].keys()}

#     # run az.compare
#     cyto_compare = az.compare(cyto_dat)
#     lyso_compare = az.compare(lyso_dat)
#     mito_compare = az.compare(mito_dat)

#     print(cyto_compare)
#     print(lyso_compare)
#     print(mito_compare)