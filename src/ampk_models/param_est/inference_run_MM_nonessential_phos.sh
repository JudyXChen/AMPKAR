#! /bin/zsh
# Script to run joint WT + LKB1kd inference for MM_nonessential_phos (Model 6, beta parameterization).
# Run from: cd src/ampk_models/param_est
#
# Supported samplers: NUTS, NUTS-ADVI, NumpyroNUTS, Nutpie, Pathfinder
# Data note: load_data() with exclude_zero_std=True drops the t=0 normalization
# reference point (std=0) which would otherwise cause logp=-inf in the Normal likelihood.

# Joint WT + LKB1kd Pathfinder — 2-DG data (glycolysis inhibition, pure nucleotide stimulus)
python inference_lkb1kd.py -model MM_nonessential_phos -compartment 2DG -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,betaCaMKK,kLKB1,KmLKB1,betaLKB1,kPP,KmPP,betaPP,betaAMP -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_2DG.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_HeLaWT_2DG.npz -model_info_file ../models/MM_nonessential_phos.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 0.01 -maxiter_pathfinder 1000 --compute_llike --sample_posterior

# Joint WT + LKB1kd Pathfinder — Iono data (calcium stimulus, requires CaMKK2 model + Ca²⁺ stress mechanism)
# python inference_lkb1kd.py -model MM_nonessential_phos -compartment Iono -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,betaCaMKK,kLKB1,KmLKB1,betaLKB1,kPP,KmPP,betaPP,betaAMP -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1kd_Iono.npz -model_info_file ../models/MM_nonessential_phos.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 0.01 -maxiter_pathfinder 1000 --compute_llike --sample_posterior
