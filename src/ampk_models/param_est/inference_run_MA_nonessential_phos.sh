#! /bin/zsh
# Script to run joint WT + LKB1kd inference for MA_nonessential_phos (Model 5, beta parameterization).
# Run from: cd src/ampk_models/param_est
#
# Supported samplers: NUTS, NUTS-ADVI, NumpyroNUTS, Nutpie, Pathfinder
# Data note: load_data() with exclude_zero_std=True drops the t=0 normalization
# reference point (std=0) which would otherwise cause logp=-inf in the Normal likelihood.

# Joint WT + LKB1kd Pathfinder (parallel paths)
python inference_lkb1kd.py -model MA_nonessential_phos -compartment Iono -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,betaCaMKK,kOffLKB1,betaLKB1,kPhosLKB1,kOffPP,betaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_RCamp_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential_phos.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 8 -jitter_pathfinder 1.0 -maxiter_pathfinder 300 -pathfinder_concurrent thread --compute_llike --sample_posterior

# Joint WT + LKB1kd Pathfinder (original paper settings: low jitter, fewer paths, higher maxiter)
python inference_lkb1kd.py -model MA_nonessential_phos -compartment Iono -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,betaCaMKK,kOffLKB1,betaLKB1,kPhosLKB1,kOffPP,betaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_RCamp_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential_phos.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 0.01 -maxiter_pathfinder 1000 --compute_llike --sample_posterior
