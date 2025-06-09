import jax.numpy as np
import numpy as onp 
from jax import Array



# ============================================================
# Orbit input validation
# ============================================================

def check_scalar(x, name):
    """Check that the input is a scalar (float or int)."""
    if not np.isscalar(x):
        raise TypeError(f"'{name}' must be a scalar (float or int).")

def check_1d_array(x, name):
    """Ensure input is a 1D array (NumPy or JAX). Converts NumPy to JAX array if needed."""
    if isinstance(x, onp.ndarray):
        x = np.array(x)  # convert NumPy to JAX
    elif not isinstance(x, Array):
        raise TypeError(f"'{name}' must be a JAX or NumPy array.")

    if x.ndim != 1:
        raise ValueError(f"'{name}' must be a 1D array (ndim=1).")

    return x



# ============================================================
# Waveform input validation
# ============================================================

def check_event_dict(event_dict, required_keys):
    """Check that all required keys are present in the event dictionary."""
    missing = [k for k in required_keys if k not in event_dict]
    if missing:
        raise KeyError(f"Missing required parameters in event dictionary: {missing}")

def validate_event_parameters(event_dict):
    """
    Validate waveform input parameters needed for the Fisher matrix.
    Ensures each value is a 1D JAX array.
    Raises errors if invalid; does not return anything.
    """
    required_keys = [
        'Mc', 'dL', 'theta', 'phi', 'iota', 'psi',
        'tcoal', 'eta', 'Phicoal', 'chi1z', 'chi2z'
    ]
    check_event_dict(event_dict, required_keys)

    for key in required_keys:
        _ = check_1d_array(event_dict[key], key)




# ============================================================
# Response input validation
# ============================================================

def validate_response_inputs(f, lambd, beta):
    """
    Validate basic input types and shapes for the LISA response class.
    """
    f = check_1d_array(f, "f")
    check_scalar(lambd, "lambd_SSB")
    check_scalar(beta, "beta_SSB")
    return f  # in case it was converted to jax array