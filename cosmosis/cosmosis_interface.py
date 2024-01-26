#! /usr/bin/env python
from cloe.cobaya_interface import EuclidLikelihood
from cobaya.model import get_model
from cloe.auxiliary.likelihood_yaml_handler import *
from cloe.auxiliary.yaml_handler import *

from cosmosis.datablock import names 
from cosmosis.datablock import option_section

cosmo = names.cosmological_parameters
likes = names.likelihoods

def setup(options):
    r"""Sets up and initialises CLOE likelihood,
    based on the config yaml file and user input
    defined within the CosmoSIS ini file.

    Parameters
    ----------
    options: CosmoSIS datablock
        options read in from the CosmoSIS ini file
    Returns
    -------
    info: dictionary
        Cobaya info dictionary
    """
    #Get user input from ini file
    config_file = options.get_string(option_section,\
                                     'config_file')
    output_path = options.get_string(option_section,\
                                     'cobaya_output')
    #Load the info on Euclid Likelihood from yaml file
    config_path = config_file
    config_dict = yaml_read(config_path)
    # define Cobaya protected info dictonary 
    info = config_dict['Cobaya']
    # sampler: MUST BE 'EVALUATE'
    info['sampler'] = {'evaluate': None}
    # set output path for Cobaya's chains
    info['output'] = output_path
    
    return info

def execute(block, config):
    r"""Runs CLOE through Cobaya, calculates
    the logposterior of one sampled point and
    puts it back into the CosmoSIS datablock

    Parameters
    ----------
    block: CosmoSIS datablock
        contains all the information pertaining
        to the CosmoSIS sampling pipeline
    config: dict
        configuration options specific to 
        CLOE likelihood, as initialised 
    """
    info = config
    # update the params key of the info dictionary
    # with the current sampled points
    section, cosmo_params = zip(*block.keys(section=cosmo))
    for cosmo_param in cosmo_params:
        # Catch case-sensitive errors (CosmoSIS auto converts all parameters
        # into lowercase, which CLOE does not recognise, so we need to 
        # convert them back to uppercase manually)
        if cosmo_param=='h0':
            cosmo_param = cosmo_param.replace('h0','H0')
        if cosmo_param=='as':
            cosmo_param = cosmo_param.replace('as','As')
        if cosmo_param=='loga':
            cosmo_param = cosmo_param.replace('loga','logA')
        if 'bg' in cosmo_param:
            cosmo_param = cosmo_param.replace('bg','bG')
        if 'mg' in cosmo_param:
            cosmo_param = cosmo_param.replace('mg','MG')
        if 'wl' in cosmo_param:
            cosmo_param = cosmo_param.replace('wl','WL')
        if 'gc' in cosmo_param:
            cosmo_param = cosmo_param.replace('gc','GC')
        if 'ap' in cosmo_param:
            cosmo_param = cosmo_param.replace('ap','aP')
        if 'psn' in cosmo_param:
            cosmo_param = cosmo_param.replace('psn','Psn')
        if 'm1' in cosmo_param:
            cosmo_param = cosmo_param.replace('m1','M1')
        if 'm_' in cosmo_param:
            cosmo_param = cosmo_param.replace('m_','M_')
        if 'log10mc' in cosmo_param:
            cosmo_param = cosmo_param.replace('log10mc','log10Mc')
        if 'hmcode_logt_agn' in cosmo_param:
            cosmo_param = cosmo_param.replace('hmcode_logt_agn','HMCode_logT_AGN')
        if 'hmcode_eta_baryon' in cosmo_param:
            cosmo_param = cosmo_param.replace('hmcode_eta_baryon','HMCode_eta_baryon')
        if 'hmcode_a_baryon' in cosmo_param:
            cosmo_param = cosmo_param.replace('hmcode_a_baryon','HMCode_A_baryon')
        info['params'].update({cosmo_param: block[cosmo, cosmo_param]})
    
    # Set the correct nonlinear halofit version
    set_halofit_version(info, info['likelihood']['Euclid']['NL_flag_phot_matter'],
                        info['likelihood']['Euclid']['NL_flag_phot_baryon'])
    # create updated model instance based on updated sampled point
    model = get_model(info)
    # calculate the logposterior of the point
    logposterior = model.logposterior({})
    # keep value in CosmoSIS datablock
    block[likes, 'CLOE_LIKE'] = logposterior.logpost
    #signal that everything went fine
    return 0

def cleanup(config):
    r"""Does nothing, is native to CosmoSIS
    
    Parameters
    ----------
    config: dictionary
        configuration options from setup function
    """
    return 0
