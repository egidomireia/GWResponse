import jax

from jax import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import jax.numpy as np
from . import globals as glob
from .utils import check_scalar, check_1d_array

#: Spacecraft indices, as a (3,) ndarray.
#SC = np.array([1, 2, 3])

#: Link (or MOSA) indices, as a (6,) ndarray.
LINKS = np.array([12, 23, 31, 13, 32, 21])


class LISA_spacecraft():
    """
    Class representing the motion of the LISA spacecrafts in the heliocentric Solar System Barycentric (SSB) frame,
    based on a first-order approximation in the orbital eccentricity.

    References:
        [1] arXiv:2003.00357 (https://arxiv.org/abs/2003.00357)

    Parameters:
        lambd (float): Initial phase offset for the spacecraft angular separation β = 2π(n−1)/3 + λ.
        kappa (float): Initial phase offset for the constellation rotation α(t) = ω(t−t₀) + κ.
        t (array): Time array over which positions are computed.
        t0 (float): Reference time (e.g. when the gravitational wave passes the solar system barycenter).

    Attributes:
        L (float): Arm length of the LISA constellation in km.
        R (float): Radius of the heliocentric orbit in km.
        e (float): Orbital eccentricity of the guiding center orbit.
        alpha (array): Rotational phase of the constellation as a function of time.

    Methods:
        spacecrafts_SSB:
            Returns the positions of the three spacecrafts in the SSB frame at each time step.
            Output shape: (3, N, 3), where 3 corresponds to the number of spacecrafts,
            N is the number of time steps, and the last dimension corresponds to the (x, y, z) coordinates.

        n_link:
            Returns the unit vectors pointing along the six inter-spacecraft links at each time step.
            Output shape: (6, N, 3), corresponding to the ordered links:
            [12, 23, 31, 13, 32, 21], where e.g. 12 is the unit vector from SC2 to SC1.
    """

    def __init__(self, lambd, kappa, t, t0=0.0):
        check_scalar(lambd, "lambd")
        check_scalar(kappa, "kappa")
        check_scalar(t0, "t0")
        check_1d_array(t, "t")
        
        # SSB frame
        self.lambd = lambd
        self.kappa = kappa


        # Angular velocity around the Sun
        T_orbit = 31557600 
        self.omega = 2*np.pi/(T_orbit) 
        self.alpha = self.omega*(t-t0) + self.kappa # shape (N,)


    
    @property
    def SC_position(self):
        """
        :return: Array containing the position of each spacecraft at each instant of time.
            Example:
                p = LISA_spacecraft(lambd, kappa, t, t0=0).spacecrafts_SSB
                p[0, i]     → position vector of SC1 at time t[i]
                p[0, i, 0]  → x-coordinate of SC1 at time t[i]
                p[0, i, 1]  → y-coordinate of SC1 at time t[i]
                p[0, i, 2]  → z-coordinate of SC1 at time t[i]
        """
        alpha = self.alpha[np.newaxis, :]  # (1, N)
        s = np.sin(alpha)
        c = np.cos(alpha)

        A = np.array([1, 2, 3]) # for each arm
        beta = (2 * (A - 1) * np.pi / 3 + self.lambd)[:, np.newaxis]  # (3, 1)

        x = glob.R * glob.e * (s * c * np.sin(beta) - (1 + s**2) * np.cos(beta))
        y = glob.R * glob.e * (s * c * np.cos(beta) - (1 + c**2) * np.sin(beta))
        z = -glob.R * glob.e * np.sqrt(3) * np.cos(alpha - beta)

        # Final shape: (3, N, 3), one row per S/C, one column per time t, third axis for x, y, z
        pL = np.stack([x, y, z], axis=-1)  # → (3, N, 3)

        # Center of the constellation
        p0 = glob.R * np.stack([np.cos(self.alpha), np.sin(self.alpha), np.zeros_like(self.alpha)], axis=-1)  # (N, 3)
        self.p0 = p0

        # Add the displacement to each S/C
        return p0[None, :, :] + pL  # → (3, N, 3)
    

    def get_link_vector(self,link_id: str):
        """
        Return the unit vector associated with a given link.

        Parameters:
            link_id (str): String representing the link, e.g. '12', '23', '31', '13', '32', '21'.

        Returns:
            array: Unit vector(s) corresponding to the selected link.
                Shape: (N, 3), one vector per time step.

        Raises:
            ValueError: If the link_id is not valid.
        """
        link_map = {
            '12': 0,
            '23': 1,
            '31': 2,
            '21': 3,
            '32': 4,
            '13': 5,
        }
        if link_id not in link_map:
            raise ValueError(f"Invalid link ID '{link_id}'. Valid options are: {list(link_map.keys())}")

        return link_map[link_id]
    
    


    def n_link(self,link=None):
        """
        Compute the unit vectors between spacecrafts for all time steps.

        Parameters:
            link (str, optional): If specified, returns only the unit vector for that link.
                                Must be one of '12', '23', '31', '13', '32', '21'.

        Returns:
            array:
                - If link is None: shape (6, N, 3), one array per link
                - If link is given: shape (1, N, 3), for the specified link only

        Notes:
            The order of links is:
                [12, 23, 31, 13, 32, 21],
            where e.g. '12' is the unit vector pointing from SC2 to SC1.
        """
    
        p = self.SC_position  # (3, N, 3)

        p1, p2, p3 = p[2], p[0], p[1]  # each: (N, 3)

        def unit_vector(v):
            """
            Normalize a time-dependent vector to obtain unit direction vectors.

            Parameters:
                v (array): Input array of shape (N, 3), representing a vector as a 
                        function of time (one vector per time step).

            Returns:
                array: Unit vectors of shape (N, 3), where each row is the normalized 
                    version of the corresponding row in v.
            """
            # Compute the norm of each vector along the last axis (x, y, z)
            norms = np.linalg.norm(v, axis=1)[:, np.newaxis]  # shape: (N, 1)
            # Divide each vector by its norm to get unit vectors
            return v / norms  # shape: (N, 3)

        n12 = unit_vector(p1 - p2)
        n23 = unit_vector(p2 - p3)
        n31 = unit_vector(p3 - p1)
        n13 = -n31
        n32 = -n23
        n21 = -n12

        n = np.array([n12, n23, n31, n13, n32, n21])

        if link==None:
            return n  # shape: (links, N, 3)
        else:
            idx = self.get_link_vector(link)
            return n[idx][np.newaxis, :, :]  # shape: (1, N, 3)
   