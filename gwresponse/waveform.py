"""
To calculate de response of the GW we'll use the PhenomD model implemented with JAX
"""




import os
import jax


from jax import config
config.update("jax_enable_x64", True)

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
import jax.numpy as np

import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(SCRIPT_DIR)


from . import globals as glob


##############################################################################
# IMRPhenomD WAVEFORM
##############################################################################


class IMRPhenomD_JAX():
    """
    IMRPhenomD waveform model.
    
    Relevant references:
        [1] `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_
        
        [2] `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_
    
    :param float, optional fRef: Reference frequency of the waveform, in :math:`\\rm Hz`. If not provided, the minimum of the frequency grid will be used.
    :param kwargs: Optional arguments to be passed to the parent class :py:class:`WaveFormModel`, such as ``is_chi1chi2``.
        
    """
    # All is taken from LALSimulation and arXiv:1508.07250, arXiv:1508.07253
    def __init__(self, objType = 'BBH', fRef=None, apply_fcut=True, **kwargs):
        """
        Constructor method
        """
        # The kind of system the wf model is made for, can be 'BBH', 'BNS' or 'NSBH'
        self.objType = objType 
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        fcutPar = 0.2

        # The cut frequency factor of the waveform, in Hz, to be divided by Mtot (in units of Msun). The method fcut can be redefined, as e.g. in the IMRPhenomD implementation, and fcutPar can be passed as an adimensional frequency (Mf)
        self.fcutPar = fcutPar
        self.apply_fcut = apply_fcut

        # Dictionary containing the order in which the parameters will appear in the Fisher matrix
        self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chiS':9,  'chiA':10}
        self.ParNums = dict(sorted(self.ParNums.items(), key=lambda item: item[1]))


        # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        self.AMP_fJoin_INS = 0.014
        # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
        self.PHI_fJoin_INS = 0.018
        self.PHI_fjoin_MRD= None
        self.fpeak= None
        
        self.fRef = fRef
        
        self.QNMgrid_a     = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_a.txt'))
        self.QNMgrid_fring = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fring.txt'))
        self.QNMgrid_fdamp = onp.loadtxt(os.path.join(glob.WFfilesPath, 'QNMData_fdamp.txt'))

    
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # These can speed up a bit, we call them multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        
        QuadMon1, QuadMon2 = np.ones(M.shape), np.ones(M.shape)
        
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)

        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        # We work in dimensionless frequency M*f, not f
        fgrid = np.transpose(np.array([M,])) * glob.GMsun_over_c3*f
        fgrid_trans = np.transpose(fgrid)
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = - 1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff.real, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff.real, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        
        # Compute sigma coefficients appearing in arXiv:1508.07253 eq. (28)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        sigma1 = 2096.551999295543 + 1463.7493168261553*eta + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
        sigma2 = -10114.056472621156 - 44631.01109458185*eta + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
        sigma3 = 22933.658273436497 + 230960.00814979506*eta + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
        sigma4 = -14621.71522218357 - 377812.8579387104*eta + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
        
        # Compute beta coefficients appearing in arXiv:1508.07253 eq. (16)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        beta1 = 97.89747327985583 - 42.659730877489224*eta + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
        beta2 = -3.282701958759534 - 9.051384468245866*eta + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
        
        # Compute alpha coefficients appearing in arXiv:1508.07253 eq. (14)
        # They derive from a fit, whose numerical coefficients are in arXiv:1508.07253 Tab. 5
        alpha1 = 43.31514709695348 + 638.6332679188081*eta + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
        alpha3 = 9.5988072383479 - 397.05438595557433*eta + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
        alpha4 = -0.02989487384493607 + 1.4022106448583738*eta + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
        
        # Compute the TF2 phase coefficients and put them in a dictionary (spin effects are included up to 3.5PN)
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -6848./21.
        TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        # Remove this part since it was not available when IMRPhenomD was tuned
        TF2coeffs['six'] = TF2coeffs['six'] - ((326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + ((4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM) + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM))*m1ByM*m1ByM*chi12 + ((4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM) + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM))*m2ByM*m2ByM*chi22)
        # Now translate into inspiral coefficients, label with the power in front of which they appear
        PhiInspcoeffs = {}
        
        PhiInspcoeffs['initial_phasing'] = TF2coeffs['five']*TF2OverallAmpl
        PhiInspcoeffs['two_thirds'] = TF2coeffs['seven']*TF2OverallAmpl*(np.pi**(2./3.))
        PhiInspcoeffs['third'] = TF2coeffs['six']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['third_log'] = TF2coeffs['six_log']*TF2OverallAmpl*(np.pi**(1./3.))
        PhiInspcoeffs['log'] = TF2coeffs['five_log']*TF2OverallAmpl
        PhiInspcoeffs['min_third'] = TF2coeffs['four']*TF2OverallAmpl*(np.pi**(-1./3.))
        PhiInspcoeffs['min_two_thirds'] = TF2coeffs['three']*TF2OverallAmpl*(np.pi**(-2./3.))
        PhiInspcoeffs['min_one'] = TF2coeffs['two']*TF2OverallAmpl/np.pi
        PhiInspcoeffs['min_four_thirds'] = TF2coeffs['one']*TF2OverallAmpl*(np.pi**(-4./3.))
        PhiInspcoeffs['min_five_thirds'] = TF2coeffs['zero']*TF2OverallAmpl*(np.pi**(-5./3.))
        PhiInspcoeffs['one'] = sigma1
        PhiInspcoeffs['four_thirds'] = sigma2 * 0.75
        PhiInspcoeffs['five_thirds'] = sigma3 * 0.6
        PhiInspcoeffs['two'] = sigma4 * 0.5
        
        #Now compute the coefficients to align the three parts
        
        fInsJoin = self.PHI_fJoin_INS
        fMRDJoin = 0.5*fring
        self.PHI_fjoin_MRD = fring
        
        
        # Time shift so that peak amplitude is approximately at t=0
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), np.fabs(fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2))
        
        t0 = (alpha1 + alpha2/(fpeak*fpeak) + alpha3/(fpeak**(1./4.)) + alpha4/(fdamp*(1. + (fpeak - alpha5*fring)*(fpeak - alpha5*fring)/(fdamp*fdamp))))/eta
        
        # Define the function of each phase (inspiral, intermediate and merger-ringdown)
        PhiIns = lambda fJoin: PhiInspcoeffs['initial_phasing'] + PhiInspcoeffs['two_thirds']*(fJoin**(2./3.)) + PhiInspcoeffs['third']*(fJoin**(1./3.)) + PhiInspcoeffs['third_log']*(fJoin**(1./3.))*np.log(np.pi*fJoin)/3. + PhiInspcoeffs['log']*np.log(np.pi*fJoin)/3. + PhiInspcoeffs['min_third']*(fJoin**(-1./3.)) + PhiInspcoeffs['min_two_thirds']*(fJoin**(-2./3.)) + PhiInspcoeffs['min_one']/fJoin + PhiInspcoeffs['min_four_thirds']*(fJoin**(-4./3.)) + PhiInspcoeffs['min_five_thirds']*(fJoin**(-5./3.)) + (PhiInspcoeffs['one']*fJoin + PhiInspcoeffs['four_thirds']*(fJoin**(4./3.)) + PhiInspcoeffs['five_thirds']*(fJoin**(5./3.)) + PhiInspcoeffs['two']*fJoin*fJoin)/eta
        PhiInt= jax.jit(lambda fJoin: (beta1*fJoin - beta3/(3.*fJoin*fJoin*fJoin) + beta2*np.log(fJoin))/eta)
        PhiMR  = jax.jit(lambda fJoin: (-(alpha2/fJoin) + (4.0/3.0) * (alpha3 * (fJoin**(3./4.))) + alpha1 * fJoin + alpha4 * np.arctan((fJoin - alpha5 * fring)/fdamp))/eta)

        # Vectorize the functions with jax.vmap
        PhiIns_v = jax.jit(jax.vmap(PhiIns,in_axes=0,out_axes=0))
        PhiInt_v = jax.jit(jax.vmap(PhiInt,in_axes=0,out_axes=0))
        PhiMR_v = jax.jit(jax.vmap(PhiMR,in_axes=0,out_axes=0))


        # Define the derivative function with batch support
        def DPhi(Phi, f, batch_size=1000):
            """
            Compute the Jacobian of Phi for both scalar and array frequencies.

            Args:
                Phi: A callable representing the function to differentiate.
                f: Frequencies, which can be a scalar or an array.
                batch_size: Size of the batch when processing large arrays.

            Returns:
                The Jacobian of Phi with respect to f, matching the dimensions of f.
            """
            diagonal_indices = np.diag_indices(M.size)
            jacfwd_single = jax.jacfwd(Phi, argnums=0)

            def compute_batch(carry, batch):
                """Compute the Jacobian for a single batch."""
                diagonal_batch = np.diag_indices(batch.size)
                jacobian_batch = jax.vmap(jacfwd_single)(batch)[diagonal_batch]
                return carry, jacobian_batch  # Carry the results forward
            

            if isinstance(f, np.ndarray) or hasattr(f, "shape"):  # Handle arrays
                if f.size > 10000:
                    # Use scan to iterate through the batches and accumulate results
                    _, jacobians = jax.lax.scan(compute_batch, None, f.reshape(-1, batch_size), length=f.size // batch_size)
                    full_jacobian = np.concatenate(jacobians, axis=0)
                else:
                    full_jacobian = jax.vmap(jacfwd_single)(f)[diagonal_indices]

                # Return the derivated function
                return full_jacobian

            else:  # Handle scalar input
                return jax.jit(jacfwd_single)(f)


    
        C2Int_new1 = DPhi(PhiIns,fInsJoin) - DPhi(PhiInt, fInsJoin)
        C1Int_new1 = PhiIns(fInsJoin) - PhiInt(fInsJoin) - C2Int_new1*fInsJoin
                
        C2MRD_new1 = DPhi(PhiInt, fMRDJoin) + C2Int_new1 - DPhi(PhiMR, fMRDJoin)
        #C2MRD_new1 = np.diag(DPhi(PhiInt, fMRDJoin)) + C2Int_new1 - np.diag(DPhi(PhiMR, fMRDJoin))
        C1MRD_new1 = PhiInt(fMRDJoin) + C1Int_new1 + C2Int_new1*fMRDJoin - PhiMR(fMRDJoin) - C2MRD_new1*fMRDJoin
        
        # Set fRef as the minimum frequency
        fRef   = np.amin(fgrid, axis=1)
        if self.fRef is not None:
            fRef =  np.transpose(np.array([M,])) * glob.GMsun_over_c3*self.fRef

        if self.apply_fcut:
            phiRef = ((fRef < self.PHI_fJoin_INS) * PhiIns(fRef) + \
                ((fRef >= self.PHI_fJoin_INS) & (fRef < fMRDJoin)) * (PhiInt(fRef) + C1Int_new1 + C2Int_new1 * fRef) + \
                ((fRef >= fMRDJoin) & (fRef < self.fcutPar)) * (PhiMR(fRef) + C1MRD_new1 + C2MRD_new1 * fRef) + \
                (fRef >= self.fcutPar) * 0)

            # Define a function that returns an array of ones matching the shape of input frequency 'f'
            f_shape = lambda f: np.ones_like(f)
            # Evaluate the phase at the cutoff frequency (fcutPar)
            #phi_cut = PhiMR_v(np.array([self.fcutPar])) + C1MRD_new1 + C2MRD_new1 * self.fcutPar
            phi_cut = PhiMR_v(np.array([max(fgrid_trans)])) + C1MRD_new1 + C2MRD_new1 * max(fgrid_trans)

            # Piecewise definition of the GW phase for different frequency intervals:
            # - Below PHI_fJoin_INS: Inspiral phase (PhiIns_v)
            # - Between PHI_fJoin_INS and fMRDJoin: Intermediate phase (PhiInt_v + linear terms)
            # - Between fMRDJoin and fcutPar: Merger-Ringdown phase (PhiMR_v + linear terms)
            # - Above fcutPar: Constant phase (phi_cut)
            phis = lambda f: np.where(f < self.PHI_fJoin_INS, PhiIns_v(f), np.where(f<fMRDJoin, PhiInt_v(f) + C1Int_new1 + C2Int_new1*f, np.where(f < self.fcutPar, PhiMR_v(f) + C1MRD_new1 + C2MRD_new1*f,f_shape(f) * phi_cut)))  # Constant phase for f >= fcutPar
            # Add time shift and reference phase corrections to the GW phase
            phase_result = lambda f: np.transpose(phis(f) + np.where(f < self.fcutPar,-t0 * (f - fRef) - phiRef,0. * f)) 

            # Define a scalar version of phase_result to use with JAX differentiation
            phase_scalar = lambda f_scalar: phase_result(f_scalar.T)[0, 0]
            # Compute the phase derivative using automatic differentiation (JAX), adding correction terms
            dphase_result = jax.vmap(jax.grad(phase_scalar))(fgrid_trans) + np.where(fgrid_trans < self.fcutPar,-t0 * (fgrid_trans - fRef) - phiRef,0. * fgrid_trans)  
            # Transpose the derivative result from shape (N, 1) to (1, N)
            dphase_result = dphase_result.T 

            return phase_result(fgrid_trans)[0], dphase_result[0]
        else:
            #phiRef = np.where(fRef < self.PHI_fJoin_INS, PhiIns_jit(fRef), np.where(fRef<fMRDJoin, PhiInt_jit(fRef) + C1Int_new1 + C2Int_new1*fRef, PhiMR_jit(fRef) + C1MRD_new1 + C2MRD_new1*fRef))
            phiRef = (fRef < self.PHI_fJoin_INS) * PhiIns(fRef) + \
                            ((fRef >= self.PHI_fJoin_INS) & (fRef < fMRDJoin)) * (PhiInt(fRef) + C1Int_new1 + C2Int_new1 * fRef) + \
                            ((fRef >= fMRDJoin) )* (PhiMR(fRef) + C1MRD_new1 + C2MRD_new1 * fRef)
            phis = lambda f: np.where(f < self.PHI_fJoin_INS, PhiIns_v(f), np.where(f<fMRDJoin, PhiInt_v(f) + C1Int_new1 + C2Int_new1*f, PhiMR_v(f) + C1MRD_new1 + C2MRD_new1*f))
            phase_result = lambda f: np.transpose(phis(f) - t0*(f - fRef) - phiRef)[0]

            # Piecewise definition of the GW phase for different frequency intervals
            phis = lambda f: np.where(f < self.PHI_fJoin_INS, PhiIns_v(f), np.where(f<fMRDJoin, PhiInt_v(f) + C1Int_new1 + C2Int_new1*f, PhiMR_v(f) + C1MRD_new1 + C2MRD_new1*f))
            # Add time shift and reference phase corrections to the GW phase
            phase_result = lambda f: np.transpose(phis(f) - t0*(f - fRef) - phiRef) 

            # Define a scalar version of phase_result to use with JAX differentiation
            phase_scalar = lambda f_scalar: phase_result(f_scalar.T)[0, 0]
            # Compute the phase derivative using automatic differentiation (JAX), adding correction terms
            dphase_result = jax.vmap(jax.grad(phase_scalar))(fgrid_trans) + (-t0 * (- fRef) - phiRef)  
            # Transpose the derivative result from shape (N, 1) to (1, N)
            dphase_result = dphase_result.T 

            return phase_result(fgrid_trans)[0], dphase_result[0]
    

         

        
    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        # Useful quantities
        M = kwargs['Mc']/(kwargs['eta']**(3./5.))
        eta = kwargs['eta']
        eta2 = eta*eta # This can speed up a bit, we call it multiple times
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
        SetaPlus1 = 1.0 + Seta
        chi_s     = 0.5 * (chi1 + chi2)
        chi_a     = 0.5 * (chi1 - chi2)
        # We work in dimensionless frequency M*f, not f
        fgrid = np.transpose(np.array([M,])) * glob.GMsun_over_c3*f
        fgrid_trans = np.transpose(fgrid)
        # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
        chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
        xi = -1.0 + chiPN
        # Compute final spin and radiated energy
        aeff = self._finalspin(eta, chi1, chi2)
        Erad = self._radiatednrg(eta, chi1, chi2)
        # Compute ringdown and damping frequencies from interpolators
        fring = np.interp(aeff.real, self.QNMgrid_a, self.QNMgrid_fring) / (1.0 - Erad)
        fdamp = np.interp(aeff.real, self.QNMgrid_a, self.QNMgrid_fdamp) / (1.0 - Erad)
        # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
        gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
        gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
        gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
        # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
        fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
        self.fpeak = fpeak
        # Compute coefficients rho appearing in arXiv:1508.07253 eq. (30), the numerical coefficients are in Tab. 5
        rho1 = 3931.8979897196696 - 17395.758706812805*eta + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
        rho2 = -40105.47653771657 + 112253.0169706701*eta + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
        rho3 = 83208.35471266537 - 191237.7264145924*eta + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
        # Compute coefficients delta appearing in arXiv:1508.07253 eq. (21)
        f1Interm = np.repeat(a = self.AMP_fJoin_INS, repeats = M.size)
        f3Interm = fpeak
        dfInterm = 0.5*(f3Interm - f1Interm)
        f2Interm = f1Interm + dfInterm
        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3

        # First write the inspiral coefficients, we put them in a dictionary and label with the power in front of which they appear
        amp0 = np.sqrt(2.0*eta/3.0)*(np.pi**(-1./6.))
        Acoeffs = {}
        Acoeffs['two_thirds'] = ((-969. + 1804.*eta)*(np.pi**(2./3.)))/672.
        Acoeffs['one'] = ((chi1*(81.*SetaPlus1 - 44.*eta) + chi2*(81. - 81.*Seta - 44.*eta))*np.pi)/48.
        Acoeffs['four_thirds'] = ((-27312085.0 - 10287648.*chi22 - 10287648.*chi12*SetaPlus1 + 10287648.*chi22*Seta+ 24.*(-1975055. + 857304.*chi12 - 994896.*chi1*chi2 + 857304.*chi22)*eta+ 35371056*eta2)* (np.pi**(4./3.)))/8.128512e6
        Acoeffs['five_thirds'] = ((np.pi**(5./3.)) * (chi2*(-285197.*(-1. + Seta) + 4.*(-91902. + 1579.*Seta)*eta - 35632.*eta2) + chi1*(285197.*SetaPlus1 - 4.*(91902. + 1579.*Seta)*eta - 35632.*eta2) + 42840.*(-1.0 + 4.*eta)*np.pi)) / 32256.
        Acoeffs['two'] = - ((np.pi**2.)*(-336.*(-3248849057.0 + 2943675504.*chi12 - 3339284256.*chi1*chi2 + 2943675504.*chi22)*eta2 - 324322727232.*eta2*eta - 7.*(-177520268561. + 107414046432.*chi22 + 107414046432.*chi12*SetaPlus1 - 107414046432.*chi22*Seta + 11087290368.*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi ) + 12.*eta*(-545384828789. - 176491177632.*chi1*chi2 + 202603761360.*chi22 + 77616.*chi12*(2610335. + 995766.*Seta) - 77287373856.*chi22*Seta + 5841690624.*(chi1 + chi2)*np.pi + 21384760320.*np.pi*np.pi)))/6.0085960704e10
        Acoeffs['seven_thirds'] = rho1
        Acoeffs['eight_thirds'] = rho2
        Acoeffs['three'] = rho3


        # Define the function of each amplitude (inspiral, intermediate and merger-ringdown)
        AmpIns = jax.jit(lambda fJoin: 1. + (fJoin**(2./3.))*Acoeffs['two_thirds'] + (fJoin**(4./3.)) * Acoeffs['four_thirds'] + (fJoin**(5./3.)) *  Acoeffs['five_thirds'] + (fJoin**(7./3.)) * Acoeffs['seven_thirds'] + (fJoin**(8./3.)) * Acoeffs['eight_thirds'] + fJoin * (Acoeffs['one'] + fJoin * Acoeffs['two'] + fJoin*fJoin * Acoeffs['three']))
        AmpInt = jax.jit(lambda fJoin: delta0 + fJoin*delta1 + fJoin**2*(delta2 + fJoin*delta3 + fJoin**2*delta4))
        AmpMR = jax.jit(lambda fJoin: np.exp(-(fJoin - fring)*gamma2/(fdamp*gamma3))* (fdamp*gamma3*gamma1) / ((fJoin - fring)**2 + (fdamp*gamma3)**2))

                    
        # Define the function with batch support
        def DAmp(Amp, f, batch_size=1000):
            """
            Compute the derivative of the amplitude function Amp with respect to frequencies f.

            Args:
                Amp: A callable or array-like representing the amplitude function. 
                    It can return shapes (n,), (5, n, n), or scalar.
                f: Frequencies, which can be a scalar or an array.
                batch_size: Size of the batch when processing large arrays.

            Returns:
                The derivative of Amp with respect to f, matching the input structure.
            """
            # Use a small test input to determine output structure
            test_input = np.array([0.1])  # Small test input
            test_output = Amp(test_input)  # Evaluate Amp with the small input
            output_shape = np.shape(test_output)  # Get the shape of the output

            # Helper: Simple jacobian computation
            diagonal_indices = np.diag_indices(M.size)
            jacfwd_single = jax.jacfwd(Amp, argnums=0)
            
            if isinstance(f, np.ndarray) and f.ndim == 1 and len(output_shape) == 1:  # Case 1: Amp returns (n,)
                if M.size > 10000:  # Use batching for large arrays
                    def compute_batch(carry, batch):
                        """Compute the Jacobian for a single batch."""
                        diagonal_batch = np.diag_indices(batch.size)
                        jacobian_batch = jax.vmap(jacfwd_single)(batch)[diagonal_batch]
                        return carry, jacobian_batch  # Carry the results forward
                    
                    # Use scan to iterate through the batches and accumulate results
                    _, jacobians = jax.lax.scan(compute_batch, None, f.reshape(-1, batch_size), length=f.size // batch_size)
                    full_jacobian = np.concatenate(jacobians, axis=0)

                else:  # Direct computation for small arrays
                    full_jacobian = jax.jit(jacfwd_single)(f)[diagonal_indices]

                return full_jacobian 
            
            elif len(output_shape) == 2 and output_shape[0] == 5:  # Case 2: AmpInt_coeffs is (5, n)
                if f.size > 10000:  # Use batching for large arrays
                        # Precompute the start indices for batches
                    n = f.size
                    batch_index = np.arange(0, n, batch_size)
                    # Ensure all batches have a fixed size by padding the input if necessary
                    pad_size = (batch_index[-1] + batch_size) - n if n % batch_size != 0 else 0
                    f_padded = np.pad(f, (0, pad_size))
                    def compute_batch(carry, start_idx):
                        # Slice the input for this batch
                        f_batch = jax.lax.dynamic_slice(f_padded, (start_idx,), (batch_size,))
                        # Compute the derivatives and extract the diagonal
                        jacobian_batch = np.diagonal(jacfwd_single(f_batch), axis1=1, axis2=2)
                        return carry, jacobian_batch
                    
                    # Process batches with jax.lax.scan
                    _, jacobians = jax.lax.scan(compute_batch, None, batch_index)
                    full_jacobian = np.concatenate(jacobians, axis=1)

                else:  # Direct computation for small arrays
                    full_jacobian = np.diagonal(jax.jit(jacfwd_single)(f), axis1=1, axis2=2)

                return full_jacobian

            else:  # Case 3: f is a scalar
                return jax.jit(jacfwd_single)(f)



        # Compute the coefficients only once
        # v1 is the inspiral model evaluated at f1Interm
        v1 = AmpIns(f1Interm)
        # d1 is the derivative of the inspiral model evaluated at f1
        #d1 = DPhi(AmpIns,f1Interm, diagonal_only=True)
        d1 = DAmp(AmpIns,f1Interm)
        # v3 is the merger-ringdown model (eq. (19) of arXiv:1508.07253) evaluated at f3
        v3 = AmpMR(f3Interm)
        # d2 is the derivative of the merger-ringdown model evaluated at f3
        #d2 = DPhi(AmpMR,f3Interm, diagonal_only=True)
        d2 = DAmp(AmpMR,f3Interm)
        # v2 is the value of the amplitude evaluated at f2. They come from the fit of the collocation points in the intermediate region
        v2 = 0.8149838730507785 + 2.5747553517454658*eta + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi**2)*xi

        # To determine the delta coefficients, we have to solve the following system of equations:
        # AmpInt(f1Interm) = v1
        # AmpInt(f2Interm) = v2
        # AmpInt(f3Interm) = v3
        # AmpInt(f1Interm) = d1
        # AmpInt(f3Interm) = d3



        # Encapsulate the entire process in a single JIT-compiled function
        @jax.jit
        def solve_system(f1Interm, f2Interm, f3Interm, v1, v2, v3, d1, d2):
            # Compute the coefficients only once
            AmpInt_coeffs = lambda fJoin: np.array([np.ones_like(fJoin), fJoin, fJoin**2, fJoin**3, fJoin**4])

            # Calculate eq_coeffs using AmpInt_coeffs and DAmpInt_coeffs
            eq_coeffs = np.stack([AmpInt_coeffs(f1Interm), AmpInt_coeffs(f2Interm), AmpInt_coeffs(f3Interm), DAmp(AmpInt_coeffs, f1Interm), DAmp(AmpInt_coeffs, f3Interm)])

            # Prepare eq_result matrix
            eq_result = np.stack([v1, v2, v3, d1, d2])       

            # Solve the system for each sub-matrix
            delta = jax.vmap(np.linalg.solve, in_axes=(2, 1))(eq_coeffs, eq_result)

            # Return the result vectors ujnpacked
            return np.transpose(delta)
        
        # Results
        delta0, delta1, delta2, delta3, delta4 = solve_system(f1Interm, f2Interm, f3Interm, v1, v2, v3, d1, d2)
        


        # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
        Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M * glob.GMsun_over_c2_Gpc * M * glob.GMsun_over_c3 / kwargs['dL']
        if self.apply_fcut:
            amplitudeIMR = np.where(fgrid_trans < self.AMP_fJoin_INS, AmpIns(fgrid_trans), np.where(fgrid_trans < fpeak, AmpInt(fgrid_trans), np.where(fgrid_trans < self.fcutPar, AmpMR(fgrid_trans), 0.)))
        else:
            amplitudeIMR = np.where(fgrid_trans < self.AMP_fJoin_INS, AmpIns(fgrid_trans), np.where(fgrid_trans < fpeak, AmpInt(fgrid_trans), AmpMR(fgrid_trans)))

        return np.transpose(Overallamp*amp0*(fgrid_trans**(-7./6.))*amplitudeIMR)[0]
        
    def _finalspin(self, eta, chi1, chi2):
        """
        Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142, which is taken from `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ eq. (3.6).
        
        :param array or float eta: Symmetric mass ratio of the objects.
        :param array or float chi1: Spin of the primary object.
        :param array or float chi2: Spin of the secondary object.
        :return: The spin of the final object.
        :rtype: array or float
        
        """
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2)
        
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def _radiatednrg(self, eta, chi1, chi2):
        """
        Compute the total radiated energy, as in `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ eq. (3.7) and (3.8).
        
        :param array or float eta: Symmetric mass ratio of the objects.
        :param array or float chi1: Spin of the primary object.
        :param array or float chi2: Spin of the secondary object.
        :return: Total energy radiated by the system.
        :rtype: array or float
        
        """
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def tau_star(self, f, **kwargs):
        """
        Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.
        
        We use the expression in `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_ eq. (3.8b).
        
        :param array f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
        :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
        :rtype: array
        
        """
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcutPar(self, **kwargs):
        """
        Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.
        
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
        :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
        :rtype: array
        
        """
        return self.fcutPar/(kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.)))



