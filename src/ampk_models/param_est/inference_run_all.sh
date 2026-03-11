#! /bin/zsh
# Run joint WT + LKB1kd Pathfinder inference for all 4 models sequentially.
# Run from: cd src/ampk_models/param_est

# 2-DG data (glycolysis inhibition, pure nucleotide stimulus)
COMMON_ARGS="-compartment 2DG -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_2DG.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_HeLaWT_2DG.npz -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 0.01 -maxiter_pathfinder 1000 --compute_llike --sample_posterior"

# Iono data (calcium stimulus, requires CaMKK2 model + Ca²⁺ stress mechanism — not yet implemented)
# COMMON_ARGS="-compartment Iono -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1wt_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_260307_LKB1kd_Iono.npz -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -data_std_max 0.1 -num_paths_pathfinder 2 -jitter_pathfinder 0.01 -maxiter_pathfinder 1000 --compute_llike --sample_posterior"

echo "=== Model 3: MA alpha ==="
python inference_lkb1kd.py -model MA_nonessential -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,alphaLKB1,kPhosLKB1,kOffPP,alphaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -model_info_file ../models/MA_nonessential.json $COMMON_ARGS

echo "=== Model 4: MM alpha ==="
python inference_lkb1kd.py -model MM_nonessential -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,alphaLKB1,kPP,KmPP,alphaPP,betaAMP -model_info_file ../models/MM_nonessential.json $COMMON_ARGS

echo "=== Model 5: MA beta ==="
python inference_lkb1kd.py -model MA_nonessential_phos -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,betaCaMKK,kOffLKB1,betaLKB1,kPhosLKB1,kOffPP,betaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -model_info_file ../models/MA_nonessential_phos.json $COMMON_ARGS

echo "=== Model 6: MM beta ==="
python inference_lkb1kd.py -model MM_nonessential_phos -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,betaCaMKK,kLKB1,KmLKB1,betaLKB1,kPP,KmPP,betaPP,betaAMP -model_info_file ../models/MM_nonessential_phos.json $COMMON_ARGS

echo "All 4 models finished."
