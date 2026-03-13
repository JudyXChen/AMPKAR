"""
Calcium signal loading and JAX-compatible Ca(t) function for CaMKK2 ODE models.

Reads fitted parameters from calcium_signal_iono.json (produced by calcium_signal_fitting.ipynb)
and returns a JIT-compatible function Ca(t_sec) -> [Ca2+] in uM.
"""
import json
import jax.numpy as jnp


def load_calcium_signal(json_path):
    """Load fitted calcium signal and return a JAX-compatible Ca(t) function.

    Args:
        json_path: path to calcium_signal_iono.json

    Returns:
        ca_func: callable(t_sec) -> [Ca2+] in uM, compatible with jax.jit
        params_dict: the full parameter dictionary from JSON
    """
    with open(json_path, 'r') as f:
        params = json.load(f)

    fp = params['fit_params']
    Ca_plat = fp['Ca_plat_uM']
    A_ca = fp['A_ca_uM']
    tau_rise = fp['tau_rise_s'] / 60.0   # seconds -> minutes (model uses minutes internally)
    tau_decay = fp['tau_decay_s'] / 60.0
    Ca_basal = params['Ca_basal_uM']
    Ca_offset = Ca_plat - Ca_basal  # gap between fit t=0 value and true basal
    tau_onset = 5.0 / 60.0  # 5s onset in minutes — fast rise from basal to Ca_plat

    def ca_func(t_sec):
        """[Ca2+] in uM at time t (seconds). t=0 is drug addition."""
        t_min = jnp.maximum(t_sec / 60.0, 0.0)
        # Start at Ca_basal, rise to Ca_plat over ~5s, then follow original bi-exponential
        onset = 1.0 - jnp.exp(-t_min / tau_onset)
        transient = A_ca * (1.0 - jnp.exp(-t_min / tau_rise)) * jnp.exp(-t_min / tau_decay)
        ca = Ca_basal + Ca_offset * onset + transient
        return jnp.where(t_sec >= 0, ca, Ca_basal)

    return ca_func, params
