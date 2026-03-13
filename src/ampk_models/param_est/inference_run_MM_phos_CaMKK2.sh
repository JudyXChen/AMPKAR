#! /bin/zsh
# Script to run joint LKB1wt + HeLa WT inference for MM_phos_CaMKK2 (MM beta + CaMKK2 cascade).
# Run from: cd src/ampk_models/param_est
#
# This model extends MM_nonessential_phos with mechanistic Ca -> CaM -> CaMKK2 activation.
# For Iono mode: uses -calcium_signal_file to provide time-dependent Ca(t) input.
# rhs_stress uses basal metab params + ca_func (Iono doesn't change metabolism).

# Joint LKB1wt + HeLa WT Pathfinder — Iono data with calcium signal
python inference_lkb1kd.py -model MM_phos_CaMKK2 -compartment Iono -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,betaCaMKK,kLKB1,KmLKB1,betaLKB1,kPP,KmPP,betaPP,betaAMP -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1kd_Iono.npz -model_info_file ../models/MM_phos_CaMKK2.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -calcium_signal_file ../models/calcium_signal_iono.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 0.01 -maxiter_pathfinder 1000 --compute_llike --sample_posterior
