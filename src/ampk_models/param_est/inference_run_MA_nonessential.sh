#! /bin/zsh
# Script to run single-condition inference for MA_nonessential model with HeLaAMPKAR3 WT Iono data.
# Run from: cd src/ampk_models/param_est
#
# Supported samplers: NUTS, NUTS-ADVI, NumpyroNUTS, Nutpie, ADVI, DEMetropolisZ, SMC
# Data note: load_data() with exclude_zero_std=True drops the t=0 normalization
# reference point (std=0) which would otherwise cause logp=-inf in the Normal likelihood.

# Nutpie
python inference_pymc.py -model MA_nonessential -compartment RCamp_Iono -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,alphaLKB1,kPhosLKB1,kOffPP,alphaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_RCamp_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI
python inference_pymc.py -model MA_nonessential -compartment RCamp_Iono -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,alphaLKB1,kPhosLKB1,kOffPP,alphaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_RCamp_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 150000 --sample_posterior

python inference_lkb1kd.py -model MA_nonessential -compartment Iono -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,alphaLKB1,kPhosLKB1,kOffPP,alphaPP,kDephosPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_RCamp_Iono.npz -LKB1_KO_data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -n_advi_iter 5000 --sample_posterior