# General import
import numpy as np
import matplotlib.pyplot as plt

# Cobaya import of general Likelihood class
from cobaya.likelihood import Likelihood
# Import cobaya model wrapper for fiducial model (TEMPORARY)
from cobaya.model import get_model

# Import likelihoods classes
from likelihood.cosmo.cosmology import Cosmology
from likelihood.like_calc.euclike import Euclike

# Error classes


class CobayaInterfaceError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class EuclidLikelihood(Likelihood):
    r"""
    Class to define Euclid Likelihood

    Inherits from the Likelihood class
    of Cobaya
    """

    # (GCH): alias for cov-mat
    aliases = ["myEuclid"]
    # Speed in evaluations/second (after theory inputs calculated).
    speed = 500
    # (GCH): which parameters are required by likelihood?
    # Define them here!
    # For the moment, only 14 different bias terms
    params = {"like_selection": None,
              'full_photo': None,
              'b1_photo': None,
              'b2_photo': None,
              'b3_photo': None,
              'b4_photo': None,
              'b5_photo': None,
              'b6_photo': None,
              'b7_photo': None,
              'b8_photo': None,
              'b9_photo': None,
              'b10_photo': None,
              'b1_spec': None,
              'b2_spec': None,
              'b3_spec': None,
              'b4_spec': None,
              'aia': None,
              'nia': None,
              'bia': None}

    def initialize(self):
        r""" initialize

        Set up values for initial variables
        and create instance of Cosmology class

        """

        # SJ: For now, example sampling in wavenumber (k)
        self.k_min = 0.001
        # ATTENTION: The k_min is not passed to cobaya to build
        # the matter power spectrum interpolator !!
        # The k_min is internally chosen by cobaya.
        # This needs to be changed

        self.k_max = 10.0
        self.k_samp = 100
        self.k_win = np.logspace(np.log10(self.k_min), np.log10(self.k_max),
                                 self.k_samp)

        # SJ: For now, example sampling in redshift (z)
        self.z_min = 0.0
        self.z_max = 4.0
        self.z_samp = 100
        self.z_win = np.linspace(self.z_min, self.z_max, self.z_samp)

        # Initialize Euclike module
        self.likefinal = Euclike()

        # (GCH): initialize Cosmology class for sampling
        self.cosmo = Cosmology()

        # (GCH): initialize the fiducial model
        # ATTENTION: This will work if CAMB is installed globally
        self.fiducial_cosmology = Cosmology()
        # (GCH): update fiducial cosmo dic with fiducial info from reader
        self.fiducial_cosmology.cosmo_dic.update(
            self.likefinal.data_spec_fiducial_cosmo)
        self.info_fiducial = {'params': {
            'ombh2': self.fiducial_cosmology.cosmo_dic['ombh2'],
            'omch2': self.fiducial_cosmology.cosmo_dic['omch2'],
            'omnuh2': self.fiducial_cosmology.cosmo_dic['omnuh2'],
            'H0': self.fiducial_cosmology.cosmo_dic['H0'],
            'tau': self.fiducial_cosmology.cosmo_dic['tau'],
            'mnu': self.fiducial_cosmology.cosmo_dic['mnu'],
            'nnu': self.fiducial_cosmology.cosmo_dic['nnu'],
            'ns': self.fiducial_cosmology.cosmo_dic['ns'],
            'As': self.fiducial_cosmology.cosmo_dic['As'],
        },
            'theory': {'camb':
                       {'stop_at_error': True,
                        'extra_args': {'num_massive_neutrinos': 1}}},
            # Likelihood: we load the likelihood as an external function
            'likelihood': {'one': None}}
        # Update fiducial cobaya dictionary with the IST-f
        # fiducial values of biases
        self.info_fiducial['params'].update(
            self.fiducial_cosmology.cosmo_dic['nuisance_parameters'])
        # (GCH): use get_model wrapper for fiducial
        model_fiducial = get_model(self.info_fiducial)
        model_fiducial.add_requirements({"Pk_interpolator":
                                         {"z": self.z_win,
                                          "k_max": self.k_max,
                                          "nonlinear": False,
                                          "vars_pairs": ([["delta_tot",
                                                           "delta_tot"]])},
                                         "comoving_radial_distance":
                                         {"z": self.z_win},
                                         "angular_diameter_distance":
                                         {"z": self.z_win},
                                         "Hubble": {"z": self.z_win,
                                                    "units": "km/s/Mpc"},
                                         "fsigma8": {"z": self.z_win,
                                                     "units": None},
                                         "sigma_R": {"z": self.z_win,
                                                     "vars_pairs":
                                                     [["delta_tot",
                                                       "delta_tot"]],
                                                     "R": [8 / 0.67]}})
        # (GCH): ATTENTION: in sigma_R, R is in Mpc, but we want it in
        # Mpc/h. A solution could be to make an interpolation
        # There is also a problem with the k (at which k fsigma_8 is
        # evaluated?
        # Still, for the fiducial IST cosmology, fsigma8/sigma8
        # where R=8/0.67 does not agree. Something else happens

        # (GCH): evaluation of posterior, required by Cobaya
        model_fiducial.logposterior({})

        # (GCH): update fiducial cosmology dictionary
        self.fiducial_cosmology.cosmo_dic['z_win'] = self.z_win
        self.fiducial_cosmology.cosmo_dic['k_win'] = self.k_win
        self.fiducial_cosmology.cosmo_dic['comov_dist'] = \
            model_fiducial.provider.get_comoving_radial_distance(
            self.z_win),
        self.fiducial_cosmology.cosmo_dic['angular_dist'] = \
            model_fiducial.provider.get_angular_diameter_distance(
            self.z_win),
        self.fiducial_cosmology.cosmo_dic['H'] = \
            model_fiducial.provider.get_Hubble(
            self.z_win),
        self.fiducial_cosmology.cosmo_dic['Pk_interpolator'] = \
            model_fiducial.provider.get_Pk_interpolator(
            nonlinear=False),
        self.fiducial_cosmology.cosmo_dic['Pk_delta'] = \
            model_fiducial.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), nonlinear=False)
        self.fiducial_cosmology.cosmo_dic['fsigma8'] = \
            model_fiducial.provider.get_fsigma8(
            self.z_win)
        R_fiducial, z_fiducial, sigma_R_fiducial = \
            model_fiducial.provider.get_sigma_R()
        self.fiducial_cosmology.cosmo_dic['sigma_8'] = \
            sigma_R_fiducial[:, 0]
        # (GCH): update dictionary with interpolators
        self.fiducial_cosmology.update_cosmo_dic(
            self.fiducial_cosmology.cosmo_dic['z_win'],
            0.05)

    def get_requirements(self):
        r""" get_requirements

        New 'theory needs'. Asks for the theory
        requirements to the theory code via
        Cobaya.

        Returns
        ----------
        dictionary specifying quantities i
        calculated by a theory code are needed

        """

        return {"Pk_interpolator":
                {"z": self.z_win,
                 "k_max": self.k_max,
                 "nonlinear": False,
                 "vars_pairs": ([["delta_tot",
                                  "delta_tot"]])},
                "comoving_radial_distance": {"z": self.z_win},
                "angular_diameter_distance": {"z": self.z_win},
                "Hubble": {"z": self.z_win, "units": "km/s/Mpc"},
                "sigma_R": {"z": self.z_win,
                            "vars_pairs": [["delta_tot",
                                            "delta_tot"]],
                            "R": [8 / 0.67]},
                "fsigma8": {"z": self.z_win, "units": None}}

    def passing_requirements(self, **params_dic):
        r""" passing_requirements

        Gets cosmological quantities from the theory code
        from COBAYA and passes them to an instance of the
        Cosmology class.

        Cosmological quantities are saved in the cosmo_dic
        attribute of the Cosmology class.

        """

        try:
            self.cosmo.cosmo_dic['H0'] = self.provider.get_param("H0")
            self.cosmo.cosmo_dic['omch2'] = self.provider.get_param('omch2')
            self.cosmo.cosmo_dic['ombh2'] = self.provider.get_param('ombh2')
            self.cosmo.cosmo_dic['mnu'] = self.provider.get_param('mnu')
            # GCH: ATTENTION! THIS IS A TEMPORAL SOLUTION
            # as we cannot retrieve num_massive_neutrinos
            self.cosmo.cosmo_dic['omnuh2'] = \
                self.cosmo.cosmo_dic['mnu'] / 94.07 * (1. / 3)**0.75
            self.cosmo.cosmo_dic['comov_dist'] = \
                self.provider.get_comoving_radial_distance(self.z_win)
            self.cosmo.cosmo_dic['angular_dist'] = \
                self.provider.get_angular_diameter_distance(self.z_win)
            self.cosmo.cosmo_dic['H'] = self.provider.get_Hubble(self.z_win)
            self.cosmo.cosmo_dic['Pk_interpolator'] = \
                self.provider.get_Pk_interpolator(nonlinear=False)
            self.cosmo.cosmo_dic['Pk_delta'] = \
                self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False)
            self.cosmo.cosmo_dic['z_win'] = self.z_win
            self.cosmo.cosmo_dic['k_win'] = self.k_win
            R, z, sigma_R = self.provider.get_sigma_R()
            self.cosmo.cosmo_dic['sigma_8'] = sigma_R[:, 0]
            self.cosmo.cosmo_dic['fsigma8'] = self.provider.get_fsigma8(
                self.cosmo.cosmo_dic['z_win'])
            self.cosmo.cosmo_dic['nuisance_parameters'].update(
                **params_dic)

        except CobayaInterfaceError:
            print('Cobaya theory requirements \
                  could not be pass to cosmo module')

    def logp(self, **params_values):
        r""" logp

        Executes passing_requirements,
        updates cosmology dictionary,
        calls log_likelihood

        Parameters
        ----------
        **params_values: tuple
              List of (sampled) parameters obtained from
              the theory code or asked by the likelihood
        Returns
        ----------
        loglike: float
            value of the function log_likelihood
        """
        self.passing_requirements(**params_values)
        # (GCH): update cosmo_dic to interpolators
        self.cosmo.update_cosmo_dic(self.cosmo.cosmo_dic['z_win'], 0.05)
        loglike = self.likefinal.loglike(self.cosmo.cosmo_dic,
                                         self.fiducial_cosmology.cosmo_dic)

        return loglike
