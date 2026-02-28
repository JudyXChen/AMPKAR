import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import diffrax as dfrx
import equinox as eqx
import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts, sample_blackjax_nuts, get_jaxified_logp
import nutpie
from pymc.variational.callbacks import CheckParametersConvergence
from pytensor.link.jax.dispatch import jax_funcify
from pymc.stats.log_density import compute_log_likelihood

from jax import random
import arviz as az
from numpyro.infer import Predictive
import sys, argparse, json, os
import matplotlib.pyplot as plt

sys.path.append("../")
from utils import *
from pymc_jax_ode import *

sys.path.append("../models/")

# tell jax to use 64bit floats
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Run MCMC for AMPK models.")
    # model info and general setup
    parser.add_argument("-model", type=str, help="model to process.")
    parser.add_argument("-compartment", type=str, help="compartment to which data belongs.")
    parser.add_argument("-free_params", type=str, help="parameters to estimate")
    parser.add_argument("-data_file", type=str, help="path to the data file. Should be a NPZ with the following objects: \
                        'times', 'mean', 'std_constant', and 'std'.")
    parser.add_argument("-model_info_file", type=str, help="JSON file with relevant info. Model params, initial conditions, and AMPKAR states.")
    parser.add_argument("-metab_params_file", type=str, help="Metabolism model parameters. Should be a JSON")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    # MCMC sampling
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Family of priors to use. Defaults to 'lognormal'.")
    parser.add_argument("-normalization", type=str, default='ratio', help="Normalization to use for the data. Defaults to 'ratio'.")
    parser.add_argument("-nwarmup", type=int, default=1000, help="Number of MCMC tuning samples. Defaults to 1000.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of posterior samples to draw per MCMC chain. Defaults to 1000.")
    parser.add_argument("-nchains", type=int, default=1, help="Number of chains to run. Defaults to 1.")
    parser.add_argument("-sampler", type=str, default='NUTS', help="Name of the MCMC sampler to use ['NUTS', 'NUTS-ADVI', 'NumpyroNUTS', 'Nutpie']. Defaults to 'NUTS'")
    parser.add_argument("-chain_method_numpyro", type=str, default='vectorized', help="Method to use for running chains in NumpyroNUTS. Defaults to 'vectorized'.")
    parser.add_argument("-ncores_nutpie", type=int, default=1, help="Number of cores to use for Nutpie. Defaults to 1 in which case sampling is sequential over the chains. If ncores > 1 then sampling is parallel over the chains.")
    # simulation parameters
    parser.add_argument("-tmax_init", type=float, default=1e3, help="Maximum time to run the simulation. Defaults to 1e3.")
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-evnt_rtol", type=float,default=1e-12)
    parser.add_argument("-evnt_atol", type=float,default=1e-12)
    parser.add_argument('-pcoeff', type=float, default=0, help='pcoeff for PID time stepper')
    parser.add_argument('-dcoeff', type=float, default=0, help='dcoeff for PID time stepper')
    parser.add_argument('-icoeff', type=float, default=1.0, help='icoeff for PID time stepper')
    # other
    parser.add_argument("-seed", type=int, default=0, help="Random seed to use. Defaults to 0.")
    parser.add_argument("--sample_prior", action='store_true', help="Flag to sample from the prior.")
    parser.add_argument("--sample_posterior", action='store_true', help="Flag to sample from the posterior.")
    parser.add_argument("--compute_llike", action='store_true', help="Flag to resample the posterior predictive using previous param samples.")
    parser.add_argument("-n_advi_iter", type=int, default=1000, help="Number of iterations for ADVI. Defaults to 1000.")

    
    args=parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    # jax.config.update("jax_enable_x64", True)
    """ Main function to execute command line script functionality. See the args parser for arguments
    """
    args = parse_args(raw_args) # parse the arguments
    print('Processing model {}.'.format(args.model))

    # add savedir if it does not exist
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    ####################################################
    # set up model info and priors #
    ####################################################
    # import the model
    try:
        exec('from ' + args.model + '_diffrax import *')
    except:
        print('Warning Model {} not found. Quitting.'.format(args.model))
        quit()

    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(args.model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack loaded model data dictionary
    state_names = list(model_info["init_conds"].keys())
    ampkar_states = model_info['ampkar_states']
    pampkar_states = model_info['pampkar_states']
    y0 = list(model_info["init_conds"].values())

    # process the free parameters
    free_params = args.free_params.split(',')

    # get the indices of the states
    ampkar_idxs = [state_names.index(item) for item in ampkar_states]
    pampkar_idxs = [state_names.index(item) for item in pampkar_states]

    # parameters for the metabolic model
    with open(args.metab_params_file, 'r') as file:
           metab_params = json.load(file)

    basal_params = list(metab_params["metab_params_basal"].values())
    stress_params = list(metab_params["metab_params_stress"].values())
    ###############################################
    #                   Model RHS                  #
    ################################################
    try:
        rhs = eval(args.model + '(' + ','.join(str(elm) for elm in basal_params) \
            + ')')
        rhs_stress = eval(args.model + '(' + ','.join(str(elm) for elm in stress_params) \
             + ')')
        rhs = dfrx.ODETerm(rhs)
        rhs_stress = dfrx.ODETerm(rhs_stress)
    except:
        print('Warning Model {} not found. Quitting.'.format(args.model))
        quit()

    ############################################
    # Data #
    ############################################
    # load the data
    # converts from min to seconds
    data, data_std, times = load_data(args.data_file, to_seconds=True, constant_std=False)
    data = data.reshape(1, len(data))
    data_std = data_std.reshape(1, len(data_std))

    ############################################
    # Simulator func #
    ############################################
    # normaliztion func
    if args.normalization == 'ratio':
        def norm_func(pAMPKAR_stressed, AMPKAR_stressed, pAMPKAR_basal, AMPKAR_basal):
            return (pAMPKAR_stressed / AMPKAR_stressed)
    elif args.normalization == 'delta_ratio':
        def norm_func(pAMPKAR_stressed, AMPKAR_stressed, pAMPKAR_basal, AMPKAR_basal):
            return (pAMPKAR_stressed / AMPKAR_stressed) - (pAMPKAR_basal / AMPKAR_basal)
    # def simulation function that solves ODE and computes proper qoi
    # the solve_traj function first runs the model to SS in the basal energy state, and then 
    # runs the model in the stressed energy state using the SS from the basal state as the initial condition
    def simulator(params):
        # solve model
        sol_stressed, sol_basal = solve_traj(rhs, rhs_stress, y0, params, times, tmax_init=args.tmax_init, rtol=args.rtol, atol=args.atol, evnt_atol=args.evnt_atol, evnt_rtol=args.evnt_rtol, pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff, dt0=1e-10)

        # compute delta pAMPKAR/AMPKAR_tot
        AMPKAR_basal = sol_basal[jnp.array(ampkar_idxs)].sum(axis=0)
        pAMPKAR_basal = sol_basal[jnp.array(pampkar_idxs)].sum(axis=0)
        AMPKAR_stressed = sol_stressed[jnp.array(ampkar_idxs), :].sum(axis=0)
        pAMPKAR_stressed = sol_stressed[jnp.array(pampkar_idxs), :].sum(axis=0)
        result = norm_func(pAMPKAR_stressed, AMPKAR_stressed, pAMPKAR_basal, AMPKAR_basal)
        
        return jnp.reshape(result, (1, len(result)))
    
    # construct PyTensor Op for simulator
    def sol_op_jax(*params):
        return simulator(params)
    
    sol_op_jax_jitted = eqx.filter_jit(sol_op_jax)
    
    def vjp_sol_op_jax(gz, *params):
        _, vjp_fn = jax.vjp(sol_op_jax, *params)
        return vjp_fn(gz)

    vjp_sol_op_jax_jitted = eqx.filter_jit(vjp_sol_op_jax)

    if args.sampler in ['NUTS', 'NUTS-ADVI', 'Nutpie', 'ADVI']:
        # if using Pymc or Nutpie samplers, then we need the Pytensor op for the grads
        vjp_sol_op = VJPSolOp(vjp_sol_op_jax_jitted)
        sol_op = SolOp(sol_op_jax_jitted, vjp_sol_op)

        # register the ops with PyTensor
        @jax_funcify.register(SolOp)
        def sol_op_jax_funcify(op, **kwargs):
            return sol_op_jax

        @jax_funcify.register(VJPSolOp)
        def vjp_sol_op_jax_funcify(op, **kwargs):
            return vjp_sol_op_jax
    elif args.sampler in ['NumpyroNUTS']:
        # using Jax-based sampler, so we do not need the Pytensor ops
        sol_op = SolOp_noGrad(sol_op_jax_jitted)

        @jax_funcify.register(SolOp_noGrad)
        def sol_op_jax_funcify(op, **kwargs):
            return sol_op_jax

    ####################################################
    # PyMC model #
    ####################################################
    prior_dict = set_lognormal_priors(list(model_info["nominal_params"].keys()), 
                                  free_params, model_info["nominal_params"], 
                                  model_info['prior_params'])
    
    pm_model = build_pymc_model(model_info['params'], prior_dict, data, sol_op, data_sigma=data_std)

    ###################################################
    # prior sampling #
    ###################################################
    if args.sample_prior:
        with pm_model:
            prior_pred = pm.sample_prior_predictive(samples=2000, random_seed=args.seed)

            prior_pred.to_netcdf(os.path.join(args.savedir, args.model + '_' \
                                            + args.compartment + '_prior_samples.nc'))
    
    #####################################################
    # Posterior sampleing (MCMC or ADVI) #
    #####################################################
    if args.sample_posterior:
        print('Running MCMC for model {} with sampler {}'.format(args.model, args.sampler))
        if args.sampler == 'NUTS':
            with pm_model:
                posterior = pm.sample(args.nsamples, tune=args.nwarmup, chains=args.nchains, 
                                cores=1, random_seed=args.seed, 
                                idata_kwargs={'log_likelihood': True})
            
        elif args.sampler == 'NUTS-ADVI':
            with pm_model:
                posterior = pm.sample(args.nsamples, tune=args.nwarmup, chains=args.nchains, 
                                cores=1, init='advi+adapt_diag', random_seed=args.seed, 
                                idata_kwargs={'log_likelihood': True})
        elif args.sampler == 'NumpyroNUTS':
            with pm_model:
                posterior = sample_numpyro_nuts(draws=args.nsamples, tune=args.nwarmup, 
                                jitter=False, chains=args.nchains, 
                                random_seed=args.seed, chain_method=args.chain_method_numpyro, 
                                progressbar=True, idata_kwargs={'log_likelihood': True})
        elif args.sampler == "ADVI":
            # Set initial values to nominal params to avoid inf loss at default starting point
            initvals = {p: model_info['nominal_params'][p] for p in free_params}
            with pm_model:
                mean_field = pm.fit(n=args.n_advi_iter, method='advi',
                                callbacks=[CheckParametersConvergence(diff='absolute', tolerance=1e-3)],
                                obj_optimizer=pm.adam,
                                start=initvals)
                
                

            # make convergence plot
            fig, ax = plt.subplots()
            ax.plot(mean_field.hist)
            fig.savefig(args.savedir + args.model + '_' + \
                                    args.compartment + '_advi_converg.png', dpi=300)
            plt.close()

            # sample from the mean field approximation
            posterior = mean_field.sample(draws=args.nsamples)
        elif args.sampler == "Nutpie":
            nutpie_compiled_model = nutpie.compile_pymc_model(pm_model)
            posterior = nutpie.sample(nutpie_compiled_model, draws=args.nsamples, 
                                      tune=args.nwarmup, chains=args.nchains, 
                                      cores=args.ncores_nutpie, seed=args.seed)
            
        ####################################################
        # posterior predictive sampling #
        ####################################################
        print('Running posterior predictive sampling for model {}'.format(args.model))
        post_pred = pm.sample_posterior_predictive(posterior, model=pm_model)

        # ####################################################
        # # save the samples #
        # ####################################################
        if args.sample_prior:
            posterior.extend(prior_pred)
        posterior.extend(post_pred)

        # save as netcdf file
        posterior.to_netcdf(os.path.join(args.savedir, args.model + '_' + \
                                        args.compartment + '_mcmc_samples_' + args.sampler + '.nc'))
        
    ####################################################
    # posterior predictive REsampling #
    ####################################################
    # Block to generate new posterior predictive samples using the stored posterior samples
    if args.compute_llike:
        fname = os.path.join(args.savedir, args.model + '_' + args.compartment + '_mcmc_samples_' + args.sampler + '.nc')
        print('Evaluating log likelihood using posterior samples stored in {}'.format(fname))
        posterior = az.from_netcdf(fname)

        posterior = compute_log_likelihood(posterior, model=pm_model, progressbar=True,
                                           extend_inferencedata=True)

        # save as netcdf file
        posterior.to_netcdf(os.path.join(args.savedir, args.model + '_' + \
                                        args.compartment + '_mcmc_samples_llike_' + args.sampler + '.nc'))
                              
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()