

import jax

from jax import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import jax.numpy as np

from .orbit import LISA_spacecraft
from .waveform import IMRPhenomD_JAX
from . import globals as glob
from .utils import validate_event_parameters, validate_response_inputs

#: Spacecraft indices, as a (3,) ndarray.
#SC = np.array([1, 2, 3])

# Initial angular positions for each spacecraft
"""SC = {1: [lambd1,beta1],
      2: [lambd2,beta2],
      3: [lambd3,beta3],}"""

#: Link (or MOSA) indices, as a (6,) ndarray.
LINKS = np.array([12, 23, 31, 13, 32, 21])



class LISA_response():
    """
    LISA_response(f, lambd_SSB, beta_SSB, kappa_SSB, theta_GW, phi_GW, psi_polarization=None, **kwargs)

    Class to compute the time-delay interferometry (TDI) response of the LISA detector 
    to a gravitational wave signal from a compact binary coalescence.

    This class implements the response formalism described in arXiv:2003.00357,
    including the Doppler modulation, orbital motion of the constellation, 
    and the application of transfer functions to the gravitational waveform in frequency space.

    Parameters
    ----------
    f : array_like
        Frequency array (in Hz) at which the response is evaluated.
    lambd_SSB : float
        Ecliptic longitude of the source in the SSB frame (in radians).
    beta_SSB : float
        Ecliptic latitude of the source in the SSB frame (in radians).
    kappa_SSB : float
        Phase shift in constellation orbit (in radians).
    theta_GW : float
        Inclination of the source in the source frame (in radians).
    phi_GW : float
        Phase of the source in the source frame (in radians).
    psi_polarization : float, optional
        Polarization angle (in radians). If None, it is computed automatically 
        from the projection of source frame and SSB frame vectors.
    **kwargs : dict
        Parameters required by the waveform model, such as total mass, mass ratio, spin, etc.

    Attributes
    ----------
    h : array_like
        Frequency-domain gravitational waveform h(f) computed with IMRPhenomD.
    fMRD : float
        Frequency at which the merger-ringdown transition occurs (Hz).
    phiR : array_like
        Doppler phase Φ_R = 2π f k · p₀(t), related to the orbital delay.
    delay_phase : array_like
        Relative Doppler phase ΔΦ_R = Φ_R - Φ_R(t_merger), useful for isolating Doppler modulation.
    p0 : array_like
        Barycenter position of the LISA constellation as a function of time.

    Methods
    -------
    polarization_tensor :
        Compute the complex polarization tensor in the detector frame.
    transfer_function(f, t, link_SC=None) :
        Compute the LISA transfer function for one or all links at time t.
    TDI_transfer(f) :
        Compute the detector-only transfer functions T_ha, T_he, T_ht.
    TDI(f) :
        Apply the transfer functions to the waveform h(f), yielding h_a, h_e, h_t in each TDI channel.
    """

    def __init__(self, f, lambd_SSB, beta_SSB,kappa_orbit = 0.0, lambd_orbit = 0.0, t0=0, **kwargs):
        self.f = validate_response_inputs(f, lambd_SSB, beta_SSB)
        self.t0 = t0
        validate_event_parameters(kwargs)

        # SSB frame
        self.lambd = lambd_SSB
        self.beta = beta_SSB
        self.kappa = kappa_orbit
        self.lambd_orbit = lambd_orbit

        self.theta = kwargs['theta'][0]
        self.phi = kwargs['phi'][0]
        self.psi = kwargs['psi'][0]


        # Spherical orthonormal basis vectors in the DETECTOR FRAME
        self.n = np.array([np.cos(self.beta)*np.cos(self.lambd), np.cos(self.beta)*np.sin(self.lambd),np.sin(self.beta)])
        self.u = np.array([np.sin(self.lambd), -np.cos(self.lambd),0.0])
        self.v = np.array([-np.sin(self.beta)*np.cos(self.lambd), -np.sin(self.beta)*np.sin(self.lambd),np.cos(self.beta)])

        # Spherical orthonormal basis vectors in the SOURCE FRAME
        self.k = - self.n
        self.p = np.cos(self.psi) * self.u + np.sin(self.psi) * self.v
        self.q = -np.sin(self.psi) * self.u + np.cos(self.psi) * self.v


        #To make explicit the dependence in polarization, we can define polarization tensors for 0 polarization (Eq.(14) in [1])
        self.epsilon_plus =  (np.outer(self.u, self.u) - np.outer(self.v, self.v)) # u⊗u - v⊗v
        self.epsilon_x =  (np.outer(self.u, self.v) + np.outer(self.v, self.u)) # u⊗v + v⊗u

        self.e_plus = (np.outer(self.p, self.p) - np.outer(self.q, self.q)) # p⊗p - q⊗q
        self.e_cross =  (np.outer(self.p, self.q) + np.outer(self.q, self.p)) # p⊗q + q⊗p 
                

        # GW waveform model with PhenomD
        self.dMf_df = (kwargs['Mc'] / kwargs['eta']**(3./5.) * glob.GMsun_over_c3)
        self.h_GW = IMRPhenomD_JAX()
        self.phase, self.dphase = self.h_GW.Phi(f, **kwargs)
        self.ampl = self.h_GW.Ampl(f, **kwargs)
        self.h = self.ampl*np.exp(-2j*self.phase)

        MfMRD = self.h_GW.PHI_fjoin_MRD  #reduced frequency (Mf)
        self.fMRD = (MfMRD/(self.dMf_df))

        # Time-delay from phase derivative (from SPA)
        # The term self.dMf_df converts the reduced frequency (Mf) used in IMRPhenomD 
        # into physical frequency f [Hz], since PhenomD evaluates in Mf = M * f * (G M_sun / c^3)
        self.tf = -self.dphase / (2 * np.pi) * self.dMf_df

        #self.z = np.exp(2j*np.pi*f*self.L)
        self.p0 = None
        self.phiR = None
        self.delay_phase = None



    @property
    def polarization_tensor(self):
        """
        Compute the polarization tensor of the gravitational wave projected
        onto the detector frame, following Eq. (16) of arXiv:2003.00357.

        Returns
        -------
        P : ndarray
            The complex polarization tensor (3x3) that encodes the response
            of the detector to a gravitational wave with given orientation
            and polarization.

        Notes
        -----
        This tensor includes the angular dependence of the spin-weighted
        spherical harmonics Y_{2,±2} and the detector's response via the
        polarization basis tensors (e₊ and eₓ). It is used in the projection
        of the wave onto each interferometer link.
        """
        # Spin-weighted spherical harmonics for modes (l, m) = (2, ±2)
        Y_22 = 0.5 * np.sqrt(5 / np.pi) * np.cos(self.theta / 2)**4 * np.exp(2j * self.phi)
        Y_2minus2 = 0.5 * np.sqrt(5 / np.pi) * np.sin(self.theta / 2)**4 * np.exp(-2j * self.phi)

        # Polarization tensor including angular and polarization dependence
        K_plus = 1/2 * (Y_22 + np.conj(Y_2minus2))
        k_cross = 1j/2 * (Y_22 - np.conj(Y_2minus2))
        P = K_plus*self.e_plus + k_cross*self.e_cross     

        return P
        

    def transfer_function(self, f, t, link_SC=None):
        """
        Compute the LISA transfer function for a given set of links and times.

        Parameters
        ----------
        f : array_like
            Frequency array (in Hz) at which to evaluate the response.
        link : list or array_like
            List of integers representing the spacecraft links (12, 23, 31, 13, 32, 21).
            The first digit is the receiver, the second is the sender.
        t : array_like
            Array of time samples (in seconds) at which to evaluate the spacecraft positions.
        link_SC : str or None, optional
            If specified, must be one of: '12', '23', '31', '13', '32', '21'.
            When provided, returns the transfer function only for that link.

        Returns
        -------
        G : ndarray
            Complex array of shape (number of links, number of times)
            containing the transfer function G(f, t) for each link.
        """

        # Initialize spacecraft constellation at given times
        spacecrafts = LISA_spacecraft(self.lambd_orbit, self.kappa, t, self.t0)
        
        # Determine which links to compute
        if link_SC is None:
            links = np.array([12, 23, 31, 13, 32, 21])
            n = spacecrafts.n_link()  # shape: (6, N, 3)
        else:
            links = np.array([int(link_SC)])
            n = spacecrafts.n_link(link=link_SC)  # shape: (1, N, 3)

        # Decode receiver and sender spacecraft indices from link IDs
        receivers = np.array([n // 10 for n in links])  # e.g., 12 → 1
        senders   = np.array([n % 10  for n in links])  # e.g., 12 → 2

        # Get spacecraft positions (shape: (L, N, 3))
        pr = spacecrafts.SC_position[receivers - 1, :, :]  # receiver positions
        ps = spacecrafts.SC_position[senders - 1, :, :]    # sender positions

        # Center of the constellation (shape: (N, 3))
        self.p0 = spacecrafts.p0


        # Reshape fL to broadcast over links
        fL = f * glob.L /glob.c      # shape: (N,)
        #fL = fL[None, :]             # shape: (1, N) → broadcast over links

        # Compute scalar products for each link and time
        k_dot_n = np.einsum("i,lki->lk", self.k, n)               # k · n  ;  (3,)·(links,N,3) -> (links,N)
        k_dot_r = np.einsum("i,lki->lk", self.k, pr + ps)         # k · (pr + ps)  ;  (3,)·(links,N,3) -> (links,N)
        nPn = np.einsum("lki,ij,lkj->lk", n, self.polarization_tensor, n)  # nᵀ · P · n  ; (3,N,links)·(3,3)·(links,N,3) -> (links,N)

        # Doppler phase computation
        k_dot_p0 = np.einsum("i,li->l", self.k, self.p0)        # k · p0  ;  (3,)·(N,3) -> (N)
        self.phiR = 2 * np.pi * f * k_dot_p0 / glob.c

        # Evaluate the transfer function
        G = 1j * fL * np.pi * np.sinc(fL * (1 - k_dot_n)) * np.exp(1j * fL * np.pi * (1 + k_dot_r / glob.L)) * nPn
        return G
    

    
    def TDI_transfer(self, f, reduced_scale = False, rescaled = True):
        """
        Compute the detector-only transfer functions for the A, E, T TDI observables
        for a given frequency array f.

        Parameters
        ----------
        f : array_like
            Frequency array (Hz)
        reduced_scale : bool, default: False
            If True, applies the normalization factor defined in Eq. (31b) of arXiv:2003.00357.
            T_ha,e,t = ( -6i π f L )^{-1} × T_a,e,t.
            This rescaling expresses the TDI response functions in units of gravitational-wave
            strain, consistent with standard representations in ground-based interferometers.
        rescaled : bool,
            If False, applies the frequency-dependent scaling factors from
            Eqs. (29a) and (29b) of arXiv:2003.00357 to convert the response from the physical TDI variables
            \tilde{a}, \tilde{e}, \tilde{t} to the scaled ones \tilde{A}, \tilde{E}, \tilde{T}.
            If True (default), these factors are not applied and the output corresponds directly to
            the unscaled physical TDI responses.

        Returns
        -------
        T_ha, T_he, T_ht : complex arrays
            Detector transfer functions for A, E, T channels (no GW included), shape (N,)
        """

        # Time-delay from phase derivative (from SPA)
        # The term self.dMf_df converts the reduced frequency (Mf) used in IMRPhenomD 
        # into physical frequency f [Hz], since PhenomD evaluates in Mf = M * f * (G M_sun / c^3)
        #tf = -self.dphase / (2 * np.pi) * self.dMf_df
        tf = self.tf

        # Evaluate individual link transfer functions
        transfer = self.transfer_function(f, tf)
        T_12, T_23, T_31, T_13, T_32, T_21 = transfer

        # Exponential delay factor z = exp(2pi i f L / c)
        fL = np.pi * f * glob.L / glob.c
        z = np.exp(2j * fL)

        # Raw TDI combinations
        T_a = (1 + z)*(T_31 + T_13) - T_23 - z*T_32 - T_21 - z*T_12
        T_e = (1 / np.sqrt(3)) * ((1 - z)*(T_13 - T_31) + (2 + z)*(T_12 - T_32) + (1 + 2*z)*(T_21 - T_23))
        T_t = np.sqrt(2 / 3) * (T_21 - T_12 + T_32 - T_23 + T_13 - T_31)



        # Normalization
        if reduced_scale:
            scale = -1 / (6j * fL)
        else:
            scale = 1

        if rescaled:
            scale_AE = 1
            scale_T = 1
        else:
            scale_AE = z * (1j * np.sqrt(2) * np.sin(2 * fL))
            scale_T = np.exp(3j*fL) * (2 * np.sqrt(2) * np.sin(2 * fL) * np.sin(fL))

        T_ha = T_a * scale * scale_AE
        T_he = T_e * scale * scale_AE
        T_ht = T_t * scale * scale_T


        return T_ha, T_he, T_ht
    
    
    def TDI(self, f, reduced_scale= False, rescaled = True):
        """
        Compute the detector response in the TDI A, E, T channels to a gravitational wave signal h(f).

        Parameters
        ----------
        f : array_like
            Frequency array (in Hz) at which to evaluate the response.
        reduced_scale : bool, default: False
            If True, applies the normalization factor defined in Eq. (31b) of arXiv:2003.00357.
            T_ha,e,t = ( -6i π f L )^{-1} × T_a,e,t.
            This rescaling expresses the TDI response functions in units of gravitational-wave
            strain, consistent with standard representations in ground-based interferometers.
        rescaled : bool,
            If False, applies the frequency-dependent scaling factors from
            Eqs. (29a) and (29b) of arXiv:2003.00357 to convert the response from the physical TDI variables
            \tilde{a}, \tilde{e}, \tilde{t} to the scaled ones \tilde{A}, \tilde{E}, \tilde{T}.
            If True (default), these factors are not applied and the output corresponds directly to
            the unscaled physical TDI responses.

        Returns
        -------
        ha, he, ht : array_like (complex)
            Gravitational wave strains measured in the TDI channels A, E, and T,
            after applying the detector transfer functions T_ha, T_he, T_ht to the waveform h(f).
        """

        # Compute the detector-only transfer functions for TDI channels A, E, and T
        T_ha, T_he, T_ht = self.TDI_transfer(f,reduced_scale, rescaled)


        # Apply the transfer functions to the waveform h(f) to get the measured strains
        ha = T_ha * self.h
        he = T_he * self.h
        ht = T_ht * self.h

        # Return the channel-specific signals
        return ha, he, ht


        



    def doppler_phase(self, f, return_delay=False):
        """
        Compute the Doppler phase Φ_R(f) and, optionally, its variation ΔΦ_R(f) 
        relative to the phase at the merger.

        This phase encodes the delay in arrival time of the gravitational wave 
        at the LISA constellation due to its orbit around the Sun. The delay is 
        projected along the propagation direction of the wave.

        Parameters
        ----------
        f : array_like
            Array of frequencies (in Hz) at which to evaluate the Doppler phase.
        return_delay : bool, optional
            If True, also returns the Doppler phase variation ΔΦ_R(f) = Φ_R(f) - Φ_R(f_merger).
            Default is False.

        Returns
        -------
        phiR : ndarray
            Doppler phase Φ_R(f) = 2π f · (k · p₀(f)), shape (N,).
        delay_phase : ndarray, optional
            Doppler phase variation ΔΦ_R(f), only if `return_delay=True`.

        Notes
        -----
        This implementation assumes the orbital position vector of the constellation 
        center `p0` has shape (N, 3), and uses the propagation direction `self.k`.

        Equation reference: arXiv:2003.00357, Eq. (36).
        """
       
        if not return_delay:
            return self.phiR
        else:
            # Ensure p0 is initialized
            if self.p0 is None:
                print("[INFO] Computing p0 inside TDI_transfer")
                spacecrafts = LISA_spacecraft(self.lambd, self.kappa, self.tf, self.t0)
                spacecrafts.SC_position 
                self.p0 = spacecrafts.p0  # shape: (N, 3)
            idx_merger = np.argmin(np.abs(f-self.fMRD))
            p0_merger = self.p0[idx_merger, :]  # (3,)
            phiR_peak = (2 * np.pi * np.dot(self.k, p0_merger) / glob.c) * f # shape (N,)
            return self.phiR, self.phiR - phiR_peak






