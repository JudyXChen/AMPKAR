#! /bin/zsh
# Script to run joint WT + LKB1kd inference for MA_nonessential model with HeLaAMPKAR3 Iono data.
# Run from: cd src/ampk_models/param_est
#
# Supported samplers: NUTS, NUTS-ADVI, NumpyroNUTS, Nutpie, Pathfinder
# Data note: load_data() with exclude_zero_std=True drops the t=0 normalization
# reference point (std=0) which would otherwise cause logp=-inf in the Normal likelihood.

# Joint WT + LKB1kd Pathfinder
python inference_lkb1kd.py -model MA_nonessential -compartment Iono -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,alphaLKB1,kPhosLKB1,kOffPP,alphaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 0.01 -maxiter_pathfinder 1000 --compute_llike --sample_posterior