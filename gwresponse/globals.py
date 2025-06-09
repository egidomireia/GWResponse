import jax.numpy as np
import os

# ============================================================
# Data paths 
# ============================================================

# Path to the current package's folder (where globals.py lives)
package_path = os.path.dirname(os.path.abspath(__file__))

# Go one level up (to the repo root where WFfiles lives)
root_path = os.path.abspath(os.path.join(package_path, ".."))

# Path to waveform files
WFfilesPath = os.path.join(root_path, "WFfiles")





# ============================================================
# Global physical constants used in the LISA response model
# ============================================================

"""
G * M_sun / c^3 — Conversion factor from solar masses to seconds.
Used in waveform models to express mass in geometrized units.
Type: float 
"""
GMsun_over_c3 = 4.925491025543575903411922162094833998e-6 # seconds

"""
G * M_sun / c^2 — Conversion factor from solar masses to meters.
Type: float 
"""
GMsun_over_c2 = 1.476625061404649406193430731479084713e3 # meters

"""
Gigaparsec (Gpc) in meters. 
Used to convert luminosity distances to SI units for waveform generation.
Type: float (dimensionful scalar)
"""
uGpc = 3.085677581491367278913937957796471611e25 # meters

"""
G * M_sun / c^2 — Conversion factor from solar masses to gigaparsec
Type: float 
"""
GMsun_over_c2_Gpc = GMsun_over_c2/uGpc # Gpc

"""
Astronomical Unit (AU) — average orbital radius of the Earth around the Sun.
Used as the radius of LISA’s guiding center orbit.
Type: float 
"""
R = 1.4959787068362817e11 # meters

"""
Speed of light in vacuum.
Type: float 
"""
c = 299792458 # m/s

"""
Length of each arm of the LISA triangular constellation.
Type: float 
"""
L = 2.5e9 # meters

"""
Orbital eccentricity of each LISA spacecraft around the Sun.
Computed from L and R assuming equilateral triangle configuration.
Type: float 
"""
e = L / (R * 2 * np.sqrt(3))
