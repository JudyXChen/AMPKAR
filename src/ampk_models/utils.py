# Nathaniel Linden 2023
# Utilities to help with the project
import json
import os
import sys

import numpy as np
import jax.numpy as jnp
import pandas as pd
import jax

import pymc as pm
import arviz as az
import preliz as pz
import diffrax as dfrx
import equinox as eqx
import seaborn as sns
import met_brewer as mb
from scipy.spatial import cKDTree

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(seed=1234)

# load DIFFRAX PYTENSOR OP for JAX ODE
from pymc_jax_ode import *
from plotting_helper_funcs import *
from matplotlib.patches import Patch

###############################################################################
#### General Utilities ####
###############################################################################
def load_smc_samples_to_idata(samples_json, sample_time=False):
    """ Load SMC samples from json file to arviz InferenceData object """
    with open(samples_json, 'r') as f:
        data = json.load(f)
    
    # create idata object from dictionary
    # ignore sample stats because that changes with each SMC chain
    idata = az.from_dict(
        posterior =  data['posterior'],
        posterior_attrs = data['posterior_attrs'],
        # sample_stats = data['sample_stats'],
        observed_data = data['observed_data'],
        observed_data_attrs = data['observed_data_attrs'],
        log_likelihood = data['log_likelihood'],
        log_likelihood_attrs = data['log_likelihood_attrs'],
        constant_data = data['constant_data'],
        constant_data_attrs = data['constant_data_attrs'],
        attrs = data['attrs'],
    )

    sample_stats = data['sample_stats']

    if sample_time:
        return idata, sample_stats, data['sample_stats_attrs']['_t_sampling']
    else:
        return idata, sample_stats
 
def get_color_pallette(n_colors=11, append_colors=['#363737','#929591','#d8dcd6']):
    """Function to get standard colors for the project.
    
    Uses a desaturated version of the seaborn colorblind pallette.
    """
    colors = sns.color_palette("colorblind", n_colors, desat=0.65)
    return colors + append_colors

def load_data(data_file, to_seconds=False, constant_std=False, exclude_zero_std=False):
    """ Loads the data from the specified file.

    Parameters
    ----------
    exclude_zero_std : bool
        If True, exclude data points where std == 0 (e.g. the t=0 normalization
        reference point). This prevents logp = -inf in the Normal likelihood.
    """
    data = np.load(data_file) # read data npz file

    # handle time, convert to seconds if specified
    times = data['times']
    zero_idx = int(np.where(times==0.0)[0][0]) # we only want values after the 2-DG stimulus
    if to_seconds:
        mult = 60
    else:
        mult = 1
    times = mult*times[zero_idx:]

    # handle data
    mean_data = data['mean'][zero_idx:]

    if constant_std:
        std_data = data['std_constant']*np.ones_like(mean_data)
    else:
        std_data = data['std'][zero_idx:]

    # Exclude data points with zero std (e.g. the t=0 normalization reference)
    # These cause logp = -inf in the Normal likelihood since log(1/0) = -inf
    if exclude_zero_std:
        valid = std_data > 0
        mean_data = mean_data[valid]
        std_data = std_data[valid]
        times = times[valid]

    return mean_data, std_data, times

def get_param_subsample(param_names, idata, n_traj, prior_or_post="post", rng=np.random.default_rng(seed=1234)):
    if prior_or_post == "post":
        dat = idata.posterior.to_dict() # convert to dictionary
    elif prior_or_post == "prior":
        dat = idata.prior.to_dict()
    else:
        raise ValueError("prior_or_post should be either 'prior' or 'post'.")
    
    # get total number of MCMC samples
    n_samples = np.array(dat['data_vars'][list(dat['data_vars'].keys())[0]]['data']).reshape(-1).shape[0]

    # extract samples for free params ot dict of numpy arrays
    free_param_samples_dict = {}
    fixed_params_value = {}
    for param in param_names:
        if param in dat['data_vars'].keys(): # free parameter
            free_param_samples_dict[param] = np.array(dat['data_vars'][param]['data']).reshape(-1)
        else: # fixed parameter
            fixed_params_value[param] = idata.constant_data[param].values

    # randomly select n_traj samples
    param_samples = []
    idxs = rng.choice(np.arange(n_samples), size=n_traj, replace=False)
    for i in idxs:
        tmp = []
        for param in param_names:
            if param in free_param_samples_dict.keys():
                tmp.append(free_param_samples_dict[param][i])
            else:
                tmp.append(fixed_params_value[param])
        param_samples.append(tmp)
 
    return np.array(param_samples)

def kl_divergence_knn(X, Y, k=1):
    """
    Estimate the KL divergence D_KL(P || Q) using k-Nearest Neighbors.
    
    Parameters:
    X : numpy array, shape (n_samples_p, n_features)
        Samples from distribution P.
    Y : numpy array, shape (n_samples_q, n_features)
        Samples from distribution Q.
    k : int
        Number of nearest neighbors to use (default is 1).
        
    Returns:
    float
        Estimated KL divergence.

    Estimator derived in: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4595271
        (Eqn: 13)
    """
    n, d = X.shape
    m, _ = Y.shape

    # Build KD-Trees for efficient neighbor search
    tree_p = cKDTree(X)
    tree_q = cKDTree(Y)

    # Distances to k-th nearest neighbor in P (excluding self)
    rho = tree_p.query(X, k + 1, p=float('inf'))[0][:, -1]  # Skip the first neighbor (itself)
    
    # Distances to k-th nearest neighbor in Q
    if k==1:
        nu = tree_q.query(X, k, p=float('inf'))[0][:,]
    else:
        nu = tree_q.query(X, k, p=float('inf'))[0][:, -1]

    # Avoid log(0) by replacing zero distances with a small value
    nu[nu == 0] = 1e-16
    rho[rho == 0] = 1e-16

    # KL Divergence estimation
    kl_estimate = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1))
    return kl_estimate
###############################################################################
#### Solving ODEs ####
###############################################################################
@jax.jit
def solve_traj(rhs, rhs_stress, y0, params, times, rtol=1e-6, atol=1e-6, 
               evnt_rtol = 1e-12, evnt_atol = 1e-12, tmax_init = 1e3, 
               pcoeff=0, icoeff=1, dcoeff=0, solver = dfrx.Kvaerno5(), dt0=1e-10):
    """ simulates a model over the specified time interval and returns the 
    calculated values.
    Returns an array of shape (n_species, 1) 
    TODO add way to specify autodiff method
    """
    stepsize_controller=dfrx.PIDController(rtol, atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)
    cond_fn=dfrx.steady_state_event(rtol=evnt_rtol, atol=evnt_atol)
    event = dfrx.Event(cond_fn=cond_fn)
    t0 = 0.0
    t1 = times[-1]
    saveat=dfrx.SaveAt(ts=times)
    max_steps=int(1e6)

    # first solve the basal model to SS
    sol = dfrx.diffeqsolve(
        rhs, solver,
        t0, tmax_init, dt0, y0,
        args=params,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=max_steps, throw=False)

    # then use that solution as the initial condition for the stressed setting
    sol_stressed = dfrx.diffeqsolve(
        rhs_stress, solver,
        t0, t1, dt0,
        sol.ys, # use basal SS at IC
        args=params, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps, throw=False)

    return jnp.squeeze(jnp.array(sol_stressed.ys)), jnp.squeeze(jnp.array(sol.ys))

@jax.jit
def solve_traj_forwardAdj(rhs, rhs_stress, y0, params, times, rtol=1e-6, atol=1e-6, 
               evnt_rtol = 1e-12, evnt_atol = 1e-12, tmax_init = 1e3, 
               pcoeff=0, icoeff=1, dcoeff=0, solver = dfrx.Kvaerno5(), dt0=1e-10):
    """ simulates a model over the specified time interval and returns the 
    calculated values.
    Returns an array of shape (n_species, 1) 
    TODO add way to specify autodiff method
    """
    stepsize_controller=dfrx.PIDController(rtol, atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)
    cond_fn=dfrx.steady_state_event(rtol=evnt_rtol, atol=evnt_atol)
    event = dfrx.Event(cond_fn=cond_fn)
    t0 = 0.0
    t1 = times[-1]
    saveat=dfrx.SaveAt(ts=times)
    max_steps=int(1e7)

    # first solve the basal model to SS
    sol = dfrx.diffeqsolve(
        rhs, solver, 
        t0, tmax_init, dt0, y0, 
        args=params,
        stepsize_controller=stepsize_controller,
        event=event, adjoint=dfrx.ForwardMode(),
        max_steps=max_steps, throw=True)
    
    # then use that solution as the initial condition for the stressed setting
    sol_stressed = dfrx.diffeqsolve(
        rhs_stress, solver, 
        t0, t1, dt0, 
        sol.ys, # use basal SS at IC
        args=params, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps, throw=True)
    
    return jnp.squeeze(jnp.array(sol_stressed.ys)), jnp.squeeze(jnp.array(sol.ys))


@jax.jit
def solve_SS(rhs, rhs_stress, y0, params, rtol=1e-6, atol=1e-6, 
             evnt_rtol = 1e-12, evnt_atol = 1e-12, tmax = 1e3,
             pcoeff=0, icoeff=1, dcoeff=0, solver = dfrx.Kvaerno5()):
    """ simulates a model over the specified time interval and returns the 
    calculated values.
    Returns an array of shape (n_species, 1) 
    TODO add way to specify autodiff method
    """
    dt0=1e-10
    solver = dfrx.Kvaerno5()
    cond_fn=dfrx.steady_state_event(rtol=evnt_rtol, atol=evnt_atol)
    event = dfrx.Event(cond_fn=cond_fn)
    stepsize_controller=dfrx.PIDController(rtol, atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)
    t0 = 0.0

    # first solve the basal model
    sol = dfrx.diffeqsolve(
        rhs, solver, 
        t0, tmax, dt0, 
        y0, args=params,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=1000000, throw=True)
    
    # then use that solution as the initial condition for the stressed setting
    sol_stressed = dfrx.diffeqsolve(
        rhs_stress, solver, 
        t0, tmax, dt0, 
        sol.ys, args=params,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=1000000, throw=True)
    
    return jnp.squeeze(jnp.array(sol_stressed.ys)), jnp.squeeze(jnp.array(sol.ys))

def run_simulations(param_samples, model_name, model_info_file, metab_params_file, times, rtol=1e-6,atol=1e-6,pcoeff=0,icoeff=1,dcoeff=0,tmax_init=1e3, y0=None):
    """ Run simulations for the specified model and return the results.
    """
    ####################################################
    # set up model info and priors #
    ####################################################
    # import the model
    try:
        exec('from ' + model_name + '_diffrax import *')
    except:
        print('Warning Model {} not found. Quitting.'.format(model_name))
        quit()

    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack loaded model data dictionary
    state_names = list(model_info["init_conds"].keys())
    ampkar_states = model_info['ampkar_states']
    pampkar_states = model_info['pampkar_states']

    if y0 is None:
        y0 = list(model_info["init_conds"].values())

    # get the indices of the states
    ampkar_idxs = [state_names.index(item) for item in ampkar_states]
    pampkar_idxs = [state_names.index(item) for item in pampkar_states]

    # parameters for the metabolic model
    with open(metab_params_file, 'r') as file:
           metab_params = json.load(file)

    basal_params = list(metab_params["metab_params_basal"].values())
    stress_params = list(metab_params["metab_params_stress"].values())

    ###############################################
    #                   Model RHS                  #
    ################################################
    try:
        rhs = eval(model_name + '(' + ','.join(str(elm) for elm in basal_params) \
            + ')')
        rhs_stress = eval(model_name + '(' + ','.join(str(elm) for elm in stress_params) \
             + ')')
        rhs = dfrx.ODETerm(rhs)
        rhs_stress = dfrx.ODETerm(rhs_stress)
    except:
        print('Warning Model {} not found. Quitting.'.format(model_name))
        quit()

    def simulator(params):
        # solve model
        sol_stressed, _ = solve_traj(rhs, rhs_stress, y0, params, times, tmax_init=tmax_init, rtol=rtol,atol=atol,pcoeff=pcoeff, icoeff=icoeff,dcoeff=dcoeff)

        # compute delta pAMPKAR/AMPKAR_tot
        AMPKAR_stressed = sol_stressed[jnp.array(ampkar_idxs), :].sum(axis=0)
        pAMPKAR_stressed = sol_stressed[jnp.array(pampkar_idxs), :].sum(axis=0)
        
        return pAMPKAR_stressed/AMPKAR_stressed
    
    # run the simulations
    sim_results = []
    for param_sample in param_samples:
        sim_results.append(simulator(param_sample))
    
    return np.array(sim_results)
###############################################################################
#### PyMC Inference Utils ####
###############################################################################
def set_prior_params(param_names, free_params, nominal_params_dict,bounds_dict, prior_family="[['Gamma()',['alpha', 'beta']]]", prob_mass_bounds=0.95, log_transform_bounds=False):
    """ Sets the prior parameters by finding parameters of the specified prior such that the specified probability mass is between the upper and lower bound.

    Uses the preliz maximum entropy function.
        Inputs:
            - param_names (list): list of parameter names
            - nominal_params (np.ndarray): array of nominal parameter values
            - free_param_idxs (list): list of indices of the free parameters
            - prior_family (str): prior family to use for the parameters. If a string will use that family for all free parameters, otherwise should be a list of strings of the same length as free_param_idxs. Each string should correspond to a pm.Distribution and pz.Distribution object, e.g., Gamma which is the default familly.
            - upper_mult (float): multiplier for the upper bound of the prior
            - lower_mult (float): multiplier for the lower bound of the prior
        Returns:
            - prior_param_dict (dict): dictionary of prior parameters for the model in syntax to use exec to set them in a pymc model object
    """

    # get the indices of the free parameters
    free_param_idxs = [param_names.index(param) for param in free_params]

    # determine if a string or list of strings was passed for the prior family
    prior_family = eval(prior_family)
    if len(prior_family) == 1:
        prior_family_list = prior_family*len(free_param_idxs)
    else:
        prior_family_list = prior_family
    
    # set the prior parameters
    prior_param_dict = {}
    for i, param in enumerate(param_names):
        if param in free_params: # check if we are dealing with a free parameter
            # get the nominal value
            nominal_val = nominal_params_dict[param]
            if nominal_val == 0:
                upper = 1.0
                lower = 1e-4
            else:
                # get the upper and lower bounds
                if log_transform_bounds:
                    upper = np.exp(bounds_dict[param][1])
                    lower = np.exp(bounds_dict[param][0])
                else:
                    upper = bounds_dict[param][1]
                    lower = bounds_dict[param][0]

            # use preliz.maxent to find the prior parameters for the specified family
            prior_fam = prior_family_list[free_param_idxs.index(i)]

            if "Truncated" in prior_fam[0]:
                print('as', prior_fam)
                tmp = prior_fam[0].strip(')').split('(')[1].split(',')
                print(tmp)
                for item in tmp:
                    if 'lower' in item:
                        lower = float(item.split('=')[1])
                    elif 'upper' in item:
                        upper = float(item.split('=')[1])
        
            dist_family = eval('pz.' + prior_fam[0])
            result = pz.maxent(dist_family, lower, upper, prob_mass_bounds, plot=False)
            result_dict = {result.param_names[i]: result.params[i] for i in range(len(result.params))}

            # set the prior parameters
            prior_fam_name = prior_fam[0].strip(')').split('(')[0]
            fixed_params = prior_fam[0].strip(')').split('(')[1].split(',')

            tmp = 'pm.' + prior_fam_name + '("' + param + '",'
            for i, hyper_param in enumerate(prior_fam[1]):
                tmp += hyper_param + '=' + str(result_dict[hyper_param]) + ', '
            
            for fixed_param in fixed_params:
                if len(fixed_param) > 0:
                    tmp += (fixed_param + ', ')
                        
            prior_param_dict[param] = tmp + ')'
            print(prior_param_dict[param])
        else: # fixed parameter
            # set the prior parameters to the nominal value
            prior_param_dict[param] = 'pm.ConstantData("' + param + '", ' + str(nominal_params_dict[param]) + ')'

    return prior_param_dict

def set_lognormal_priors(param_names, free_params, nominal_params_dict,prior_param_dict):
    """ Constructs dict of priors for the specified model using lognormal priors.

    Assums prior_params_dict contains the mu and tau values for the lognormal priors.

    Returns:
        - prior_param_dict (dict): dictionary of prior parameters for the model in syntax to use exec to set them in a pymc model object
    """

    # # get the indices of the free parameters
    # free_param_idxs = [param_names.index(param) for param in free_params]

    # set the prior parameters
    prior_dict = {}
    for i, param in enumerate(param_names):
        if param in free_params: # check if we are dealing with a free parameter
            
            # set the prior parameters string to be evaluated in the pymc model constructor
            tmp = 'pm.LogNormal("' + param + '", mu=' + \
                str(prior_param_dict[param]['mu']) + ', sigma=' + \
                str(prior_param_dict[param]['sigma']) + ')'
            
            print(tmp)
            prior_dict[param] = tmp

        else: # fixed parameter
            # set the prior parameters to the nominal value
            if len(prior_param_dict[param]) > 0:
                val = pz.LogNormal(mu=prior_param_dict[param]['mu'], sigma=prior_param_dict[param]['sigma']).mean()
            else:
                val = nominal_params_dict[param]
            prior_dict[param] = 'pm.Data("' + param + '", jnp.array(' + str(val) + '))'

    return prior_dict

def build_pymc_model(param_names, prior_param_dict, data, sol_op, data_sigma=0.1):
    """ Builds a pymc model object for the AMPK models.
    
    If model is None, the function will use the default model. If a model is 
    specified, it will use that model_func function to create a PyMC model.

    """

    # Construct the PyMC model #   
    with pm.Model() as model:
        # loop over free params and construct the priors
        priors = {}
        # for key, value in prior_param_dict.items():
        for param in param_names:
            # create PyMC variables for each parameters in the model
            prior = eval(prior_param_dict[param])
            priors[param] = prior

        # predict response
        prediction = pm.Deterministic('prediction', sol_op(*[priors[param] for param in param_names]))

        # assume a normal model for the data
        # sigma specified by the data_sigma param to this function
        llike = pm.Normal("llike", mu=prediction, sigma=data_sigma, observed=data)

    return model

###############################################################################
#### Plotting Utils ####
###############################################################################
def plot_predictive(inf_data, data, times, plot_prior=True, plot_post=True,
                    add_t_0=True, n_traces=200, figsize=(6, 4), prior_color='blue',
                    post_color='black', data_color='red', data_marker_size=10, 
                    cred_int=95, fig_ax = (None, None), linestyle='-',llike_name='llike'):
    """"plots prior and posterior predictive checks for the given model 
    along with the data supplied for inference"""

    # first create data frames for plotting of prior and posterior predictive checks
    if plot_prior: # if plotting prior, then convert prior predictive into a dataframe
        prior_sims = inf_data.prior_predictive[llike_name].values
        # Convert the nchains x ndraws x ntime Prior predictive into a dataframe
        nchains, ndraws, _, ntime = prior_sims.shape
        prior_sims_df = pd.DataFrame({
            'chain': np.repeat(np.arange(nchains), ndraws * ntime),
            'draw': np.tile(np.repeat(np.arange(ndraws), ntime), nchains),
            'time': np.tile(times, nchains * ndraws),
            'y': prior_sims.flatten()
        })
        # Add rows with time 0 and y 0 for each draw
        if add_t_0:
            zero_rows = []
            for (c, d) in prior_sims_df.groupby(['chain', 'draw']).groups.keys():
                zero_rows.append({'chain': c, 'draw': d, 'time': 0, 'y': 0})
            zero_time_rows = pd.DataFrame(zero_rows)
            prior_sims_df = pd.concat([prior_sims_df, zero_time_rows],
                                        ignore_index=True).sort_values(by=['chain',
                                        'draw', 'time']).reset_index(drop=True)
    
    if plot_post:
        # convert posterior predictive into a dataframe
        if type(inf_data) == az.InferenceData:
            post_sims = inf_data.posterior_predictive[llike_name].values
            nchains, ndraws, _, ntime = post_sims.shape
        elif type(inf_data) == np.ndarray:
            post_sims = inf_data
            nchains = 1
            ndraws, ntime = post_sims.shape
            
        post_sims_df = pd.DataFrame({
            'chain': np.repeat(np.arange(nchains), ndraws * ntime),
            'draw': np.tile(np.repeat(np.arange(ndraws), ntime), nchains),
            'time': np.tile(times, nchains * ndraws),
            'y': post_sims.flatten()
        })
        # Add rows with time 0 and y 0 for each draw
        if add_t_0:
            zero_rows = []
            for (c, d) in post_sims_df.groupby(['chain', 'draw']).groups.keys():
                zero_rows.append({'chain': c, 'draw': d, 'time': 0, 'y': 0})
            zero_time_rows = pd.DataFrame(zero_rows)
            post_sims_df = pd.concat([post_sims_df, zero_time_rows],
                                        ignore_index=True).sort_values(by=['chain',
                                    'draw', 'time']).reset_index(drop=True)

    # plot predictive checks
    if fig_ax[0] is not None and fig_ax[1] is not None:
        fig, ax = fig_ax
    else:
        fig, ax = get_sized_fig_ax(figsize[0], figsize[1])
    if n_traces > 0:
        for i in range(n_traces):
            if plot_prior:
                if i == 0:
                    label = 'Prior'
                else:
                    label = None
                ax.plot(np.hstack((np.array([0]), times)), 
                        np.hstack((np.array([0]), np.squeeze(prior_sims[0, i, 0, :]))), 
                        color=prior_color, alpha=0.05, linewidth=0.5, label=label)
            elif plot_post:
                if i == 0:
                    label = 'Posterior'
                else:
                    label = None
                ax.plot(np.hstack((np.array([0]), times)), 
                        np.hstack((np.array([0]), np.squeeze(post_sims[0, i, 0, :]))), 
                        color=post_color, alpha=0.05, linewidth=0.5, label=label)
            
    # plot predictive densities
    if plot_prior:
        sns.lineplot(data=prior_sims_df, x='time', y='y', 
                    errorbar=("pi", cred_int), ax=ax, color=prior_color, 
                    label='Prior', linewidth=1.0)
    elif plot_post:
        sns.lineplot(data=post_sims_df, x='time', y='y',
                    errorbar=("pi", cred_int), ax=ax, color=post_color, 
                    label='Posterior predictive', linewidth=2.0, linestyle=linestyle)
    
    # plot data
    ax.plot(times, data, color=data_color, linestyle='--', linewidth=1.5,
            label='Data', zorder=5)

    # label formatting
    ax.set_xlabel('')
    ax.set_ylabel('')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8)

    # set min lims to 0 for x and y axes
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    
    # Build legend handles manually
    custom_handles = []
    custom_labels = []
    if plot_prior:
        custom_handles.append(Patch(facecolor=prior_color, edgecolor='none', alpha=0.3))
        custom_labels.append('Prior')
    if plot_post:
        line_handle = mpl.lines.Line2D([], [], color=post_color, linewidth=2.0, linestyle=linestyle)
        custom_handles.append((line_handle, Patch(facecolor=post_color, edgecolor='none', alpha=0.3)))
        custom_labels.append('Posterior predictive')
    custom_handles.append(mpl.lines.Line2D([], [], color=data_color, linestyle='--', linewidth=1.5))
    custom_labels.append('Data')
    leg = ax.legend(handles=custom_handles, labels=custom_labels, fontsize=8,
                    bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig, ax, leg

def get_compartment_colors(compartment_names=['cyto', 'lyso', 'mito'], mb_pallete='Egypt'):
    n_compartment = len(compartment_names)

    # get the colors for the compartments
    colors = mb.met_brew(name=mb_pallete, n=n_compartment)

    # create a dictionary of compartment colors
    compartment_colors = {compartment_names[i]: colors[i] for i in range(n_compartment)}

    return compartment_colors