#! /bin/zsh
# script to run inference for MA_nonessential model with HeLaAMPKAR3 LKB1kd Iono data
# cd src/ampk_models/param_est

# MA nonessential, LKB1kd Iono
# NUTS with default init
python inference_pymc.py -model MA_nonessential -compartment LKB1kd_Iono -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,kOffPP,alphaPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -normalization minmax_ratio -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with minmax normalization
python inference_pymc.py -model MA_nonessential -compartment LKB1kd_Iono -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,kOffPP,alphaPP,kOffAMPK,kPhosAMPK,betaAMP,kOffPP1,kDephosPP1 -data_file ../../../AMPKARkey_data/HeLaAMPKAR3_LKB1kd_Iono.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 4 -sampler ADVI -normalization minmax_ratio -pcoeff 0.3 -icoeff 0.4 -rtol 1e-4 -atol 1e-4 -n_advi_iter 3000 -advi_learning_rate 0.01 --sample_posterior