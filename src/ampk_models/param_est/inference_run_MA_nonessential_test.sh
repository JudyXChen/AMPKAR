#! /bin/zsh
# Script to run inference for MA_nonessential model with HeLaAMPKAR3 data.
# Run from: cd src/ampk_models/param_est
#
# Data note: load_data() with exclude_zero_std=True drops the t=0 normalization
# reference point (std=0) which would otherwise cause logp=-inf in the Normal likelihood.

# MA nonessential

# # NUTS
# python inference_pymc.py -model MA_nonessential -compartment LKB1kd_Iono -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,kOffPP,alphaPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -normalization minmax_ratio -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# # ADVI
# python inference_pymc.py -model MA_nonessential -compartment LKB1kd_Iono -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,kOffPP,alphaPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 4 -sampler ADVI -normalization minmax_ratio -pcoeff 0.3 -icoeff 0.4 -rtol 1e-4 -atol 1e-4 -n_advi_iter 3000 -advi_learning_rate 0.01 --sample_posterior

# Joint WT + LKB1kd Iono fit with shared parameters (Pathfinder)
# Parameters are shared between WT and LKB1kd; only kOnLKB1 and kPhosLKB1 are zeroed in the KO condition.
python inference_lkb1kd.py -model MA_nonessential -compartment Iono -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,alphaLKB1,kPhosLKB1,kOffPP,alphaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_RCamp_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 1.0 -maxiter_pathfinder 100 --compute_llike --sample_posterior
