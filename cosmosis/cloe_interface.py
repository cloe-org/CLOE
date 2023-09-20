import numpy as np
import os, sys
from astropy import constants as const
from scipy.interpolate import RectBivariateSpline, interp1d
import camb

from cloe.cosmo.cosmology import Cosmology
from cloe.like_calc.euclike import Euclike
from cloe.non_linear.nonlinear import Nonlinear
from cloe.auxiliary.likelihood_yaml_handler import *
from cloe.auxiliary.yaml_handler import *
from cloe.auxiliary.observables_dealer import *
from cloe.cobaya_interface import EuclidLikelihood

from cosmosis.datablock import names 
from cosmosis.datablock import option_section

cosmo = names.cosmological_parameters
likes = names.likelihoods
distances = names.distances
growth = names.growth_parameters
cloe_params = names.cloe_parameters

def setup(options):
    r"""Setup and initialize CLOE likelihood,
    based on the config yaml file and user input
    from the CosmoSIS ini file.

    Parameters
    ----------
    options: CosmoSIS datablock
        options read in from the CosmoSIS ini file
        
    Returns
    -------
    likefinal: Class
        The initialized Euclike class
    cloe_cosmo: dict
        Dictionary of the cosmological quantities as required
        by the Euclike class
    fid_cosmo: dict
        Dictionary of the fiducial cosmological quantities
        when Euclike class is initialised
    """
    #Get user input from ini file
    config_file = options.get_string(option_section,\
                                     'config_file')
    #Load the info on Euclid Likelihood from yaml file
    config_path = config_file
    config_dict = yaml_read(config_path)
    # obtain data and observables dict to pass to Euclike class
    info = config_dict['Cobaya']['likelihood']['Euclid']
    data = info['data']
    observables = observables_selection_specifications_checker(
                info['observables_selection'],
                info['observables_specifications'])
    if info['plot_observables_selection']:
        observables_pf = observables_visualization(
             observables['selection'])
    observables['selection']['add_phot_RSD'] = info['add_phot_RSD']
    observables['selection']['matrix_transform_phot'] = \
            info['matrix_transform_phot']
    # initialize Euclike class
    likefinal = Euclike(data, observables)
    # initialize cosmo dict
    cloe_cosmo = Cosmology()
    # add z_win and k_win to cosmo_dict based on yaml dict
    cloe_cosmo.cosmo_dic['z_win'] = np.linspace(
                                info['z_min'], info['z_max'], info['z_samp'])
    cloe_cosmo.cosmo_dic['k_win'] = np.logspace(np.log10(info['k_min_extrap']),
                                 np.log10(info['k_max_extrap']), info['k_samp'])
    # read in options from yaml dict, store in cosmo dict
    cloe_cosmo.cosmo_dic['NL_flag_phot_matter'] = \
            info['NL_flag_phot_matter']
    cloe_cosmo.cosmo_dic['NL_flag_spectro'] = info['NL_flag_spectro']
    cloe_cosmo.cosmo_dic['bias_model'] = info['bias_model']
    cloe_cosmo.cosmo_dic['use_gamma_MG'] = info['use_gamma_MG']
    # Adding GCphot and GCspectro redshift bins to cosmo dictionary and setting up
    # the internal class for Pgg_spectro with this information.
    cloe_cosmo.cosmo_dic['redshift_bins_means_phot'] = \
            data['photo']['redshifts']
    cloe_cosmo.nonlinear = Nonlinear(cloe_cosmo.cosmo_dic)
    cloe_cosmo.nonlinear.theory['redshift_bins_means_spectro'] = \
        data['spectro']['edges']
    cloe_cosmo.nonlinear.set_Pgg_spectro_model()
    cloe_cosmo.cosmo_dic['redshift_bins_means_spectro'] = \
        data['spectro']['edges']
    # Initialize the fiducial model
    fid_cosmo = set_fiducial_cosmology(likefinal, info)
    # Here we add the fiducial angular diameter distance and Hubble factor
    # to the cosmo dictionary. In this way we can avoid passing the whole
    # fiducial dictionary
    cloe_cosmo.cosmo_dic['fid_d_z_func'] = fid_cosmo.cosmo_dic['d_z_func']
    cloe_cosmo.cosmo_dic['fid_H_z_func'] = fid_cosmo.cosmo_dic['H_z_func']
    cloe_cosmo.cosmo_dic['fid_r_z_func'] = fid_cosmo.cosmo_dic['r_z_func']
    # Create a separate dictionary with fiducial cosmo quantities that are
    # available at initialization, before cosmo_dic is created.
    likefinal.fiducial_cosmo_quantities_dic.update(
        fid_cosmo.cosmo_dic)
    # Compute the data vectors
    # and initialize possible matrix transforms
    likefinal.get_masked_data()
    # Add the luminosity_ratio_z_func to the cosmo_dic after data has been
    # read and stored in the data_ins attribute of Euclike
    cloe_cosmo.cosmo_dic['luminosity_ratio_z_func'] = \
        likefinal.data_ins.luminosity_ratio_interpolator
    cloe_cosmo.cosmo_dic['obs_selection'] = observables['selection']

    return likefinal, cloe_cosmo, fid_cosmo

def set_fiducial_cosmology(likefinal, info):
    r"""Creates the fiducial cosmology class

    This function reads the input fiducial cosmology from the instance of
    the Euclike class, and sets up a dedicated Cosmology class.
    
    Parameters
    ----------
    likefinal: Class
        The initialized Euclike class
    info: dict
        Dictionary read in based on the config yaml file
        
    Returns
    -------
    fid_cosmo: dict
        Dictionary of the fiducial cosmological quantities
        when Euclike class is initialised
    """
    # This will work if CAMB is installed globally
    fid_cosmo = Cosmology()
    fid_cosmo.cosmo_dic['z_win'] = np.linspace(
                                info['z_min'], info['z_max'], info['z_samp'])
    fid_cosmo.cosmo_dic['k_win'] = np.logspace(np.log10(info['k_min_extrap']),
                                 np.log10(info['k_max_extrap']), info['k_samp'])
    # Update fiducial cosmo dic with fiducial info from reader
    fid_cosmo.cosmo_dic.update(
        likefinal.data_spectro_fiducial_cosmo)
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams(WantTransfer=True,
                           Want_CMB=False, Want_CMB_lensing=False, DoLensing=False,
                           NonLinear="NonLinear_none",
                           WantTensors=False, WantVectors=False,
                           WantCls=False,
                           WantDerivedParameters=False,
                           want_zdrag=False, want_zstar=False,
                           z_outputs=fid_cosmo.cosmo_dic['z_win'])
    #This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=fid_cosmo.cosmo_dic['H0'], 
                       ombh2=fid_cosmo.cosmo_dic['ombh2'], 
                       omch2=fid_cosmo.cosmo_dic['omch2'],  
                       mnu=fid_cosmo.cosmo_dic['mnu'], 
                       omk=fid_cosmo.cosmo_dic['Omk'], 
                       tau=fid_cosmo.cosmo_dic['tau'],
                       nnu=fid_cosmo.cosmo_dic['nnu'],
                       num_massive_neutrinos=1)
    pars.InitPower.set_params(As=fid_cosmo.cosmo_dic['As'], 
                              ns=fid_cosmo.cosmo_dic['ns'], 
                              r=0)
    pars.set_matter_power(redshifts=fid_cosmo.cosmo_dic['z_win'],
                          kmax=fid_cosmo.cosmo_dic['k_win'][-1])
    #calculate results for these parameters
    results = camb.get_results(pars)
    # Update fiducial cosmology dictionary
    # Update fiducial cosmology dictionary
    fid_cosmo.cosmo_dic['Omm'] = \
        pars.omegam
    fid_cosmo.cosmo_dic['Omk'] = \
        pars.omk
    fid_cosmo.cosmo_dic['comov_dist'] = \
        results.comoving_radial_distance(
        fid_cosmo.cosmo_dic['z_win'])
    fid_cosmo.cosmo_dic['angular_dist'] = \
        results.angular_diameter_distance(
        fid_cosmo.cosmo_dic['z_win'])
    fid_cosmo.cosmo_dic['H'] = \
        results.hubble_parameter(
            fid_cosmo.cosmo_dic['z_win'])
    fid_cosmo.cosmo_dic['H_Mpc'] = \
        results.h_of_z(
        fid_cosmo.cosmo_dic['z_win'])
    fid_cosmo.cosmo_dic['Pk_delta'] = \
        results.get_matter_power_interpolator(
            nonlinear=False, 
            var1='delta_tot', var2='delta_tot',
            extrap_kmax=fid_cosmo.cosmo_dic['k_win'][-1])
    fid_cosmo.cosmo_dic['Pk_cb'] = \
        results.get_matter_power_interpolator(
            nonlinear=False, 
            var1='delta_nonu', var2='delta_nonu',
            extrap_kmax=fid_cosmo.cosmo_dic['k_win'][-1])
    fid_cosmo.cosmo_dic['Pk_weyl'] = \
        results.get_matter_power_interpolator(
            nonlinear=False, 
            var1='Weyl', var2='Weyl',
            extrap_kmax=fid_cosmo.cosmo_dic['k_win'][-1])
    fid_cosmo.cosmo_dic['fsigma8'] = \
        results.get_fsigma8()[::-1]
    fid_cosmo.cosmo_dic['sigma8'] = \
        results.get_sigma8()[::-1]
    # In order to make the update_cosmo_dic method to work, we need to
    # specify also in this case the information on the GCspectro bins
    fid_cosmo.cosmo_dic['redshift_bins_means_phot'] = \
            info['data']['photo']['redshifts']
    fid_cosmo.nonlinear.theory['redshift_bins_means_spectro'] = \
        info['data']['spectro']['edges']
    fid_cosmo.nonlinear.set_Pgg_spectro_model()
    # Update dictionary with interpolators
    fid_cosmo.cosmo_dic['luminosity_ratio_z_func'] = \
        likefinal.data_ins.luminosity_ratio_interpolator
    fid_cosmo.update_cosmo_dic(
        fid_cosmo.cosmo_dic['z_win'], 0.05)
    
    return fid_cosmo

def execute(block, config):
    r"""Updates CLOE cosmo dictionary with parameter 
    values from CAMB/CLASS within the COSMOSIS
    datablock, and calculates the log-likelihood
    
    Parameters
    ----------
    block: CosmoSIS datablock
        contains all the information pertaining
        to the CosmoSIS sampling pipeline
    config: dict
        configuration options specific to 
        CLOE likelihood, as initialized 
    """
    likefinal, cloe_cosmo, fid_cosmo = config
    # define Pk Interpolator class (adapted from Cobaya's
    # PowerSpectrumInterpolator class) to pass to
    # cosmo_dict
    class PowerSpectrumInterpolator(RectBivariateSpline):
        r"""Class for 2D spline interpolation object, 
        subclass of scipy.interpolate.RectBivariateSpline
        """
        def __init__(self, z, k, P):
            r"""Class constructor
            
            Initialises Interpolator class

            Parameters
            ----------
            z: numpy.ndarray
                Array of redshifts at which the power spectrum was evaluated
            k: numpy.ndarray
                Array of k scales at which the power spectrum was evaluated
            P: numpy.ndarray
                2D Array of the power spectrum as a function of redshift and k
            """
            super().__init__(z, k, P)
            
        def P(self, z, k, grid=None):
            r"""Returns value of matter power spectrum for given z and k. 
            Method is defined as P in order to be consistent with the 
            corresponding quantity that is retrieved from Cobaya 
            in cosmology.py

            Parameters
            ----------
            z: float
                Value of redshift at which the power spectrum is desired
            k: float
                Value of k at which the power spectrum is desired
            
            Returns
            -------
            P: float
                Value of the power spectrum for given redshift and k
            """
            if grid is None:
                grid = not np.isscalar(z) and not np.isscalar(k)
            P = self(z, k, grid=grid)
            return P
    # proceed to update cosmo_dict from datablock                  
    try:
        cloe_cosmo.cosmo_dic['H0'] = block[cosmo,'h0']*100
        cloe_cosmo.cosmo_dic['H0_Mpc'] = \
            cloe_cosmo.cosmo_dic['H0'] / const.c.to('km/s').value
        cloe_cosmo.cosmo_dic['tau'] = block[cosmo,'tau']
        cloe_cosmo.cosmo_dic['As'] = block[cosmo,'a_s']
        cloe_cosmo.cosmo_dic['ns'] = block[cosmo,'n_s']
        cloe_cosmo.cosmo_dic['omch2'] = block[cosmo,'omch2']
        cloe_cosmo.cosmo_dic['ombh2'] =block[cosmo,'ombh2']
        cloe_cosmo.cosmo_dic['Omk'] = block[cosmo,'omega_k']
        try:
            cloe_cosmo.cosmo_dic['mnu'] = block[cosmo,'mnu']
        except KeyError:
            cloe_cosmo.cosmo_dic['omnuh2'] = block[cosmo,'omnuh2']
        cloe_cosmo.cosmo_dic['nnu'] = block[cosmo,'nnu']
        cloe_cosmo.cosmo_dic['w'] = block[cosmo,'w']
        cloe_cosmo.cosmo_dic['wa'] = block[cosmo,'wa']
        if cloe_cosmo.cosmo_dic['use_gamma_MG']:
            cloe_cosmo.cosmo_dic['gamma_MG'] = block[cosmo,'gamma_mg']
        cloe_cosmo.cosmo_dic['Omm'] = block[cosmo,'omega_m']
        cloe_cosmo.cosmo_dic['Omc'] = block[cosmo,'omega_c']
        cloe_cosmo.cosmo_dic['omnuh2'] = block[cosmo,'omnuh2']
        cloe_cosmo.cosmo_dic['Omnu'] = block[cosmo,'omega_nu']
        # Extract distance and growth parameters from datablock,
        # calculated by CAMB in previous step
        cloe_cosmo.cosmo_dic['comov_dist'] = block[distances,'D_C']
        cloe_cosmo.cosmo_dic['angular_dist'] = block[distances,'D_A']
        cloe_cosmo.cosmo_dic['H_Mpc'] = block[distances,'H']
        cloe_cosmo.cosmo_dic['H'] = block[growth,'H']
        cloe_cosmo.cosmo_dic['sigma8'] = block[growth,'sigma_8']
        cloe_cosmo.cosmo_dic['fsigma8'] = block[growth,'fsigma_8']
        # Extract matter power spectrum from datablock, transform into
        # interpolator class for cosmo_dict
        h3 = block[cosmo,'h0']**3
        z, k, pk_delta = block.get_grid('matter_power_lin', 'z', 'k_h', 'p_k')
        cloe_cosmo.cosmo_dic['Pk_delta'] = PowerSpectrumInterpolator(z, k * block[cosmo,'h0'], 
                                                                     pk_delta / h3)
        z, k, pk_cb = block.get_grid('cdm_baryon_power_lin', 'z', 'k_h', 'p_k')
        cloe_cosmo.cosmo_dic['Pk_cb'] = PowerSpectrumInterpolator(z, k * block[cosmo,'h0'], 
                                                                  pk_cb / h3)
        z, k, pk_weyl = block.get_grid('weyl_curvature_power_lin', 'z', 'k_h', 'p_k')
        cloe_cosmo.cosmo_dic['Pk_weyl'] = PowerSpectrumInterpolator(z, k * block[cosmo,'h0'], 
                                                                    pk_weyl / h3)
        if cloe_cosmo.cosmo_dic['NL_flag_phot_matter'] > 0:
            z, k, pk_delta_nl = block.get_grid('matter_power_nl', 'z', 'k_h', 'p_k')
            cloe_cosmo.cosmo_dic['Pk_halomodel_recipe'] = PowerSpectrumInterpolator(
                                                                z, k * block[cosmo,'h0'], 
                                                                pk_delta_nl / h3)
            z, k, pk_weyl_nl = block.get_grid('weyl_curvature_power_nl', 'z', 'k_h', 'p_k')
            cloe_cosmo.cosmo_dic['Pk_weyl_NL'] = PowerSpectrumInterpolator(
                                                                z, k * block[cosmo,'h0'], 
                                                                pk_weyl_nl / h3)

    except (TypeError, AttributeError):
        cloe_cosmo.cosmo_dic['H0'] = block[cosmo,'h0']*100
        cloe_cosmo.cosmo_dic['H0_Mpc'] = \
            cloe_cosmo.cosmo_dic['H0'] / const.c.to('km/s').value
        cloe_cosmo.cosmo_dic['tau'] = block[cosmo,'tau']
        cloe_cosmo.cosmo_dic['As'] = block[cosmo,'a_s']
        cloe_cosmo.cosmo_dic['ns'] = block[cosmo,'n_s']
        cloe_cosmo.cosmo_dic['omch2'] = block[cosmo,'omch2']
        cloe_cosmo.cosmo_dic['ombh2'] =block[cosmo,'ombh2']
        cloe_cosmo.cosmo_dic['Omk'] = block[cosmo,'omega_k']
        try:
            cloe_cosmo.cosmo_dic['mnu'] = block[cosmo,'mnu']
        except KeyError:
            cloe_cosmo.cosmo_dic['omnuh2'] = block[cosmo,'omnuh2']
        cloe_cosmo.cosmo_dic['w'] = block[cosmo,'w']
        cloe_cosmo.cosmo_dic['wa'] = block[cosmo,'wa']
        cloe_cosmo.cosmo_dic['Omm'] = block[cosmo,'omega_m']
        if cloe_cosmo.cosmo_dic['use_gamma_MG']:
            cloe_cosmo.cosmo_dic['gamma_MG'] = block[cosmo,'gamma_mg']
        cloe_cosmo.cosmo_dic['Omc'] = block[cosmo,'omega_c']
        cloe_cosmo.cosmo_dic['omnuh2'] = block[cosmo,'omnuh2']
        cloe_cosmo.cosmo_dic['Omnu'] = block[cosmo,'omega_nu']
        cloe_cosmo.cosmo_dic['nnu'] = block[cosmo,'nnu']
        # Extract distance and growth parameters from datablock,
        # calculated by CAMB in previous step
        cloe_cosmo.cosmo_dic['comov_dist'] = block[distances,'D_C']
        cloe_cosmo.cosmo_dic['angular_dist'] = block[distances,'D_A']
        cloe_cosmo.cosmo_dic['H_Mpc'] = block[distances,'H']
        cloe_cosmo.cosmo_dic['H'] = block[growth,'H']
        cloe_cosmo.cosmo_dic['sigma8'] = block[growth,'sigma_8']
        cloe_cosmo.cosmo_dic['fsigma8'] = block[growth,'fsigma_8']
        # Extract matter power spectrum from datablock, transform into
        # interpolator class for cosmo_dict
        h3 = block[cosmo,'h0']**3
        z, k, pk_delta = block.get_grid('matter_power_lin', 'z', 'k_h', 'p_k')
        cloe_cosmo.cosmo_dic['Pk_delta'] = PowerSpectrumInterpolator(z, k * block[cosmo,'h0'], 
                                                                     pk_delta / h3)
        z, k, pk_cb = block.get_grid('cdm_baryon_power_lin', 'z', 'k_h', 'p_k')
        cloe_cosmo.cosmo_dic['Pk_cb'] = PowerSpectrumInterpolator(z, k * block[cosmo,'h0'], 
                                                                  pk_cb / h3)
        z, k, pk_weyl = block.get_grid('weyl_curvature_power_lin', 'z', 'k_h', 'p_k')
        cloe_cosmo.cosmo_dic['Pk_weyl'] = PowerSpectrumInterpolator(z, k * block[cosmo,'h0'], 
                                                                    pk_weyl / h3)
        if cloe_cosmo.cosmo_dic['NL_flag_phot_matter'] > 0:
            z, k, pk_delta_nl = block.get_grid('matter_power_nl', 'z', 'k_h', 'p_k')
            cloe_cosmo.cosmo_dic['Pk_halomodel_recipe'] = PowerSpectrumInterpolator(
                                                                z, k * block[cosmo,'h0'], 
                                                                pk_delta_nl / h3)
            z, k, pk_weyl_nl = block.get_grid('weyl_curvature_power_nl', 'z', 'k_h', 'p_k')
            cloe_cosmo.cosmo_dic['Pk_weyl_NL'] = PowerSpectrumInterpolator(
                                                                z, k * block[cosmo,'h0'], 
                                                                pk_weyl_nl / h3)
    # update nuisance parameters, classified under the section called cloe_parameters
    section, nuisance_params = zip(*block.keys(section=cloe_params))
    for nuisance_param in nuisance_params:
        # catch capitalisation errors
        if 'wl' in nuisance_param:
            nuisance_param = nuisance_param.replace('wl','WL')
        if 'gc' in nuisance_param:
            nuisance_param = nuisance_param.replace('gc','GC')
        if 'ap' in nuisance_param:
            nuisance_param = nuisance_param.replace('ap','aP')
        if 'psn' in nuisance_param:
            nuisance_param = nuisance_param.replace('psn','Psn')
        cloe_cosmo.cosmo_dic['nuisance_parameters'].update(
                {nuisance_param: block[cloe_params, nuisance_param]})
    # Update cosmo_dic
    cloe_cosmo.update_cosmo_dic(cloe_cosmo.cosmo_dic['z_win'], 0.05)
    # Calculate log-likelihood 
    loglike = likefinal.loglike(cloe_cosmo.cosmo_dic)
    # keep value in CosmoSIS datablock
    block[likes, 'CLOE_LIKE'] = loglike
    #signal that everything went fine
    return 0

def cleanup(config):
    r"""Cleanup function native to CosmoSIS,
    does nothing
    
    Parameters
    ----------
    config: dict
        configuration options from setup function
    """
    return 0

