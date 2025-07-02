"""EUCLIKE

Contains class to compute the Euclid likelihood.
"""

import numpy as np
from cloe.photometric_survey.photo import Photo
from cloe.spectroscopic_survey.spectro import Spectro
from cloe.clusters_of_galaxies.CG import CG
from cloe.data_reader import reader
from cloe.masking.masking import Masking
from cloe.masking.data_handler import Data_handler
from cloe.cmbx_p.cmbx import CMBX
from cloe.photometric_survey.redshift_distribution \
    import RedshiftDistribution
from cloe.auxiliary.matrix_transforms import BNT_transform
import copy


class Euclike:
    """
    Class to compute the Euclid likelihood from the theory, data, covariance.
    """

    def __init__(self, data, observables):
        """Initialise.

        Constructor of the class Euclike. The data and covariance are
        read and arranged into their final format only once here.

        Parameters
        ----------
        data: dict
            Dictionary containing specifications for data loading and handling,
            to be passed to the data reader module
        observables: dict
            Dictionary containing specification for the chosen observables by
            the user
        """
        self.observables = observables
        self.do_photo = (any(observables['selection']['WL'].values()) or
                         any(observables['selection']['GCphot'].values()))
        try:
            self.do_cmbx = (
                any(observables['selection']['CMBlens'].values()) or
                any(observables['selection']['CMBisw'].values())
            )
        except KeyError:
            self.do_cmbx = False

        # Get the photo probes even when doing CMBX analysis only
        self.do_photo = self.do_photo or self.do_cmbx
        self.do_spectro = any(observables['selection']['GCspectro'].values())
        self.do_clusters = any(observables['selection']['CG'].values())

        self.data = data
        if self.do_spectro:
            if observables['specifications']['GCspectro']['statistics'] in \
                    ('multipole_power_spectrum',
                     'convolved_multipole_power_spectrum'):
                self.do_fourier_spectro = True
                self.str_start_spectro = 'pk'
                self.scale_var_spectro = 'k_pk'
                spec_str = 'GCspectro'
                if observables['specifications'][spec_str]['statistics'] == \
                        'convolved_multipole_power_spectrum':
                    self.do_convolved_multipole = True
                else:
                    self.do_convolved_multipole = False
            elif observables['specifications']['GCspectro']['statistics'] == \
                    'multipole_correlation_function':
                self.do_fourier_spectro = False
                self.str_start_spectro = 'xi'
                self.scale_var_spectro = 'r_xi'
            else:
                raise ValueError('Unknown statistics_spectro choice. '
                                 'Use multipole_power_spectrum, '
                                 'convolved_multipole_power_spectrum, '
                                 'or multipole_correlation_function')
            if self.data['spectro']['Fourier'] != self.do_fourier_spectro:
                raise ValueError('Inconsistent choice of statistics '
                                 'between the spectroscopic data and '
                                 'theory vectors')

        if self.do_photo:
            # Determine if to use Fourier or configuration space
            photo_obs = ['WL', 'GCphot', 'WL-GCphot']
            for obs_name in photo_obs:
                if obs_name in observables['specifications'].keys():
                    key = obs_name
                    break
            if observables['specifications'][key]['statistics'] in \
                    ('angular_power_spectrum', 'pseudo_cl'):
                self.do_fourier_photo = True
                self.scale_var_photo = 'ells'
                self.num_wl_obs = 1
                if observables['specifications'][key]['statistics'] == \
                        'pseudo_cl':
                    self.do_pseudo_cl = True
                else:
                    self.do_pseudo_cl = False
            elif observables['specifications'][key]['statistics'] == \
                    'angular_correlation_function':
                self.do_fourier_photo = False
                self.scale_var_photo = 'thetas'
                self.num_wl_obs = 2
            else:
                raise ValueError('Unknown statistics_photo choice. '
                                 'Use angular_power_spectrum, '
                                 'pseudo_cl, or '
                                 'angular_correlation_function')
            if self.data['photo']['Fourier'] != self.do_fourier_photo:
                raise ValueError('Inconsistent choice of statistics '
                                 'between the photometric data and '
                                 'theory vectors')

        self.data_ins = reader.Reader(self.data)
        self.data_ins.compute_luminosity_ratio()
        self.data_spectro_fiducial_cosmo = \
            self.data_ins.data_spectro_fiducial_cosmo
        self.fiducial_cosmo_quantities_dic = {}
        # Read data, instantiate Photo and Spectro classes
        # and compute pre-computed quantities
        if self.do_photo:
            self.data_ins.compute_nz()
            # Read photo
            self.data_ins.read_phot()

            # Calculate permutations i,j bins for WL, GC-Phot, XC.
            # This refers to the non-redundant bin combinations for
            # which we have measurements (i.e. 1-1, 1-2, ..., 1-10,
            # 2-2, 2-3, ..., 2-10, 3-3, 3-4, etc, in the case of ten
            # tomographic bins for WL and GC-Phot. Meanwhile, all bin
            # combinations exist for XC, i.e. for example both 1-2
            # and 2-1, both 1-3 and 3-1, etc).
            numtomo_wl = self.data_ins.numtomo_wl
            numtomo_gcphot = self.data_ins.numtomo_gcphot

            x_diagonal_wl = np.array(np.triu_indices(numtomo_wl)) + 1
            x_diagonal_gcphot = np.array(np.triu_indices(numtomo_gcphot)) + 1
            x_full_xc = np.indices((numtomo_gcphot,
                                    numtomo_wl)).reshape(2, -1) + 1
            self.indices_diagonal_wl = \
                tuple(zip(x_diagonal_wl[0], x_diagonal_wl[1]))
            self.indices_diagonal_gcphot = \
                tuple(zip(x_diagonal_gcphot[0], x_diagonal_gcphot[1]))
            self.indices_all = tuple(zip(x_full_xc[0], x_full_xc[1]))
            self.scales_WL = \
                self.data_ins.data_dict['WL'][self.scale_var_photo]
            self.scales_XC = \
                self.data_ins.data_dict['XC-Phot'][self.scale_var_photo]
            self.scales_GC_phot = \
                self.data_ins.data_dict['GC-Phot'][self.scale_var_photo]

            # Flag to select cases with or without RSD for photometric probes
            add_RSD = observables['selection']['add_phot_RSD']
            # default value of matrix_transform_phot, gets modified by cobaya
            # interface
            self.matrix_transform_phot = \
                self.observables['selection']['matrix_transform_phot']

            self.mixing_matrix_dict_phot = \
                self.data_ins.read_phot_mixing_matrix()

            # Photo class instance
            self.phot_ins = Photo(None,
                                  self.data_ins.nz_dict_WL,
                                  self.data_ins.nz_dict_GC_Phot,
                                  self.mixing_matrix_dict_phot,
                                  add_RSD=add_RSD)

            # Temporary placeholder for theta vector
            # (will be read from file eventually)
            theta_min_arcmin = 0.6
            theta_max_arcmin = 500.0
            nbins_theta = 20
            theta_arcmin = np.logspace(np.log10(theta_min_arcmin),
                                       np.log10(theta_max_arcmin),
                                       nbins_theta)
            theta_rad = np.deg2rad(theta_arcmin / 60.)

            # Sets the precomputed Bessel functions as an attribute of the
            # Photo class
            self.phot_ins._set_bessel_tables(theta_rad)

            if self.do_fourier_photo:
                # set the precomputed prefactors for the WL, XC and GCphot Cls
                self.phot_ins.set_prefactor(ells_WL=self.scales_WL,
                                            ells_XC=self.scales_XC,
                                            ells_GC_phot=self.scales_GC_phot)

                if self.do_cmbx:
                    # Read CMB data
                    self.data_ins.read_cmbx()
                    # CMBX class instance
                    self.cmbx_ins = CMBX(self.phot_ins)

        if self.do_spectro:
            # Read spectro
            self.data_ins.read_GC_spectro()
            self.data_spectro_fiducial_cosmo = \
                self.data_ins.data_spectro_fiducial_cosmo
            self.mixing_matrix_dict_spectro = \
                self.data_ins.read_GC_spectro_mixing_matrix()
            self.zkeys = self.data_ins.data_dict['GC-Spectro'].keys()
            # Spectro class instance
            self.spec_ins = Spectro(None, list(self.zkeys),
                                    self.mixing_matrix_dict_spectro)

        # Read data, instantiate galaxy cluster classes
        # and compute pre-computed quantities
        if self.do_clusters:

            # Read CG
            self.data_ins.read_CG()
            # Transforming data
            self.CGCCdatafinal = self.create_CG_data()[0]
            self.CGcovCCfinal = self.create_CG_cov_external()[0]
            self.CGinvcovCCfinal = 1.0 / self.CGcovCCfinal
            self.CGMoRdatafinal = self.create_CG_data()[1]
            self.CGcovMoRfinal = self.create_CG_cov_external()[1]
            self.CGinvcovMoRfinal = 1.0 / self.CGcovMoRfinal
            self.CGxi2datafinal = self.create_CG_data()[2]
            self.CGcovxi2final = self.create_CG_cov_external()[2]
            self.CGinvcovxi2final = 1.0 / self.CGcovxi2final

        # Create data vectors and covariances and mask them

    def get_masked_data(self):
        """Gets masked data.

        Creates the data vectors and covariances for photometric and
        spectroscopic probes, creates instances of the masking class for each
        and applies the masking to the data and covariances.
        """
        if self.do_photo:
            # precompute matrix transforms needed for photo data
            self.precompute_matrix_transform_phot()
            phot_data = self.create_photo_data()
            if self.do_cmbx:
                phot_data.update(self.create_photoxcmb_data())
        if self.do_spectro:
            spectrodata = self.create_spectro_data()
            spectrocov = self.create_spectro_cov()
        # Reshaping the data vectors and covariance matrices
        # into dictionaries to be passed to the data_handler class
        if self.do_photo and self.do_spectro:
            datafinal = {**phot_data,
                         'GC-Spectro': spectrodata}
            covfinal = {'3x2pt': self.data_ins.data_dict['3x2pt_cov'],
                        'GC-Spectro': spectrocov}
        elif self.do_spectro:
            datafinal = {'GC-Spectro': spectrodata}
            covfinal = {'GC-Spectro': spectrocov}
        elif self.do_photo:
            datafinal = phot_data
            covfinal = {'3x2pt': self.data_ins.data_dict['3x2pt_cov']}
        if self.do_cmbx:
            covfinal['7x2pt'] = self.data_ins.data_dict['7x2pt_cov']

        self.data_handler_ins = Data_handler(datafinal,
                                             covfinal,
                                             self.observables,
                                             self.data_ins)

        # Mask data vectors and covariances
        # for the photometric and spectroscopic probes separately
        if self.do_photo:
            self.data_vector_phot, self.cov_matrix_phot, \
                self.masking_vector_phot =               \
                self.data_handler_ins.get_data_and_masking_vector_phot()
            self.mask_ins_phot = Masking()
            self.mask_ins_phot.set_data_vector(self.data_vector_phot)
            self.mask_ins_phot.set_masking_vector(self.masking_vector_phot)
            self.masked_data_vector_phot = (
                self.mask_ins_phot.get_masked_data_vector())
            self.mask_ins_phot.set_covariance_matrix(self.cov_matrix_phot)
            self.masked_cov_matrix_phot = (
                self.mask_ins_phot.get_masked_covariance_matrix())
            self.ndata_phot = self.masked_data_vector_phot.size
            if (self.data['photo']['cov_is_num']):
                self.nsim_phot = self.data['photo']['cov_nsim']
                if (self.nsim_phot <= self.ndata_phot + 1.0):
                    raise ValueError(
                        "The photo data covariance is not invertible "
                        "because cov_nsim is too low")
                elif (self.nsim_phot <= self.ndata_phot + 4.0):
                    raise ValueError("Cannot apply Percival et al. 2022 "
                                     "likelihood shape for photo "
                                     "because cov_nsim is too low")
            self.masked_invcov_matrix_phot = (
                np.linalg.inv(self.masked_cov_matrix_phot))
            # Check for inversion issues
            if not np.allclose(np.dot(self.masked_cov_matrix_phot,
                               self.masked_invcov_matrix_phot),
                               np.eye(self.masked_cov_matrix_phot.shape[0]),
                               atol=1e-7):
                raise ValueError("Problem with the inversion of the "
                                 "photo covariance")
        if self.do_spectro:
            self.data_vector_spectro, self.cov_matrix_spectro, \
                self.masking_vector_spectro =               \
                self.data_handler_ins.get_data_and_masking_vector_spectro()
            self.mask_ins_spectro = Masking()
            self.mask_ins_spectro.set_data_vector(self.data_vector_spectro)
            self.mask_ins_spectro.set_covariance_matrix(
                self.cov_matrix_spectro)
            self.mask_ins_spectro.set_masking_vector(
                self.masking_vector_spectro)
            self.masked_data_vector_spectro = (
                self.mask_ins_spectro.get_masked_data_vector())
            self.masked_cov_matrix_spectro = (
                self.mask_ins_spectro.get_masked_covariance_matrix())
            self.ndata_spectro = self.masked_data_vector_spectro.size
            if (self.data['spectro']['cov_is_num']):
                self.nsim_spectro = self.data['spectro']['cov_nsim']
                if (self.nsim_spectro <= self.ndata_spectro + 1.0):
                    raise ValueError(
                        "The spectro data covariance is not invertible "
                        "because cov_nsim is too low")
                elif (self.nsim_spectro <= self.ndata_spectro + 4.0):
                    raise ValueError("Cannot apply Percival et al. 2022 "
                                     "likelihood shape for spectro "
                                     "because cov_nsim is too low")
            self.masked_invcov_matrix_spectro = (
                np.linalg.inv(self.masked_cov_matrix_spectro))
            # Check for inversion issues
            if not np.allclose(np.dot(self.masked_cov_matrix_spectro,
                               self.masked_invcov_matrix_spectro),
                               np.eye(
                               self.masked_cov_matrix_spectro.shape[0])):
                raise ValueError("Problem with the inversion of the "
                                 "spectro covariance")

    def create_photo_data(self):
        """Arranges the photometric data.

        Arranges the photometric data vector for the likelihood
        into its final format.

        Returns
        -------
        Photometric data vector: dict
            Dictionary of arrays with the transformed photo data
        """

        datavec_dict = {'GC-Phot': [], 'WL': [], 'XC-Phot': [], 'all': []}
        for index in list(self.data_ins.data_dict['WL'].keys()):
            if 'B' in index:
                del (self.data_ins.data_dict['WL'][index])
        for index in list(self.data_ins.data_dict['XC-Phot'].keys()):
            if 'B' in index:
                del (self.data_ins.data_dict['XC-Phot'][index])
        # Transform GC-Phot
        # We ignore the first value (scales)
        self.tomo_ind_GC_phot = \
            list(self.data_ins.data_dict['GC-Phot'].keys())[1:]
        datavec_dict['GC-Phot'] = np.array(
                [self.data_ins.data_dict['GC-Phot'][key][scale]
                 for scale in range(len(self.scales_GC_phot))
                 for key in self.tomo_ind_GC_phot])

        self.tomo_ind_WL = list(self.data_ins.data_dict['WL'].keys())[1:]
        datavec_dict['WL'] = np.array(
                [self.data_ins.data_dict['WL'][key][scale]
                 for scale in range(len(self.scales_WL))
                 for key in self.tomo_ind_WL])

        self.tomo_ind_XC = list(self.data_ins.data_dict['XC-Phot'].keys())[1:]
        datavec_dict['XC-Phot'] = np.array(
                [self.data_ins.data_dict['XC-Phot'][key][scale]
                 for scale in range(len(self.scales_XC))
                 for key in self.tomo_ind_XC])

        datavec_dict['WL'] = \
            self.transform_photo_theory_data_vector(datavec_dict['WL'],
                                                    obs='WL')
        datavec_dict['XC-Phot'] = \
            self.transform_photo_theory_data_vector(datavec_dict['XC-Phot'],
                                                    obs='XC-phot')
        datavec_dict['GC-Phot'] = \
            self.transform_photo_theory_data_vector(datavec_dict['GC-Phot'],
                                                    obs='GC-phot')
        datavec_dict['all'] = np.concatenate((datavec_dict['WL'],
                                              datavec_dict['XC-Phot'],
                                              datavec_dict['GC-Phot'],
                                              ), axis=0)

        return datavec_dict

    def create_photo_theory(self, dictionary):
        """Creates the photometric theory.

        Obtains the photometric theory for the likelihood.
        The theory is evaluated only for the probes specified in the masking
        vector. For the probes for which the theory is not evaluated, an array
        of zeros is included in the returned dictionary.

        Parameters
        ----------
        dictionary: dict
            Cosmology dictionary from the :obj:`cosmology` class which
            is updated at each sampling step

        Returns
        -------
        Photometric theory vector: numpy.ndarray
            Array with the photo theory vector.
            The elements of the array corresponding to probes for which the
            theory is not evaluated, are set to zero
        """

        self.phot_ins.update(dictionary)

        # Obtain the theory for WL
        if self.data_handler_ins.use_wl:
            if (self.do_fourier_photo and (not self.do_pseudo_cl)):
                wl_array = np.array(
                    [self.phot_ins.Cl_WL(self.scales_WL,
                                         element[0], element[1])
                     for element in self.indices_diagonal_wl]
                ).flatten('F')
            elif (self.do_fourier_photo and self.do_pseudo_cl):
                wl_array = np.array(
                    [self.phot_ins.pseudo_Cl_3x2pt('shear-shear',
                                                   self.scales_WL,
                                                   element[0], element[1])
                     for element in self.indices_diagonal_wl]
                ).flatten('F')
            else:
                wl_xi_plus = np.array(
                    [self.phot_ins.corr_func_3x2pt('shear-shear_plus',
                                                   self.scales_WL,
                                                   element[0], element[1])
                     for element in self.indices_diagonal_wl]
                ).flatten('F')
                wl_xi_minus = np.array(
                    [self.phot_ins.corr_func_3x2pt('shear-shear_minus',
                                                   self.scales_WL,
                                                   element[0], element[1])
                     for element in self.indices_diagonal_wl]
                ).flatten('F')
                wl_array = np.concatenate((wl_xi_plus, wl_xi_minus), axis=0)
        else:
            wl_array = np.zeros(
                 self.num_wl_obs * len(self.scales_WL) *
                 len(self.indices_diagonal_wl)
            )

        # Obtain the theory for XC-Phot
        if self.data_handler_ins.use_xc_phot:
            if (self.do_fourier_photo and (not self.do_pseudo_cl)):
                xc_phot_array = np.array(
                    [self.phot_ins.Cl_cross(self.scales_XC,
                                            element[1], element[0])
                     for element in self.indices_all]
                ).flatten('F')
            elif (self.do_fourier_photo and self.do_pseudo_cl):
                xc_phot_array = np.array(
                    [self.phot_ins.pseudo_Cl_3x2pt('shear-position',
                                                   self.scales_XC,
                                                   element[1], element[0])
                     for element in self.indices_all]
                ).flatten('F')
            else:
                xc_phot_array = np.array(
                    [self.phot_ins.corr_func_3x2pt('shear-position',
                                                   self.scales_XC,
                                                   element[1], element[0])
                     for element in self.indices_all]
                ).flatten('F')
        else:
            xc_phot_array = np.zeros(
                 len(self.scales_XC) *
                 len(self.indices_all)
            )

        # Obtain the theory for GC-Phot
        if self.data_handler_ins.use_gc_phot:
            if (self.do_fourier_photo and (not self.do_pseudo_cl)):
                gc_phot_array = np.array(
                    [self.phot_ins.Cl_GC_phot(self.scales_GC_phot,
                                              element[0], element[1])
                     for element in self.indices_diagonal_gcphot]
                ).flatten('F')
            elif (self.do_fourier_photo and self.do_pseudo_cl):
                gc_phot_array = np.array(
                    [self.phot_ins.pseudo_Cl_3x2pt('position-position',
                                                   self.scales_GC_phot,
                                                   element[0], element[1])
                     for element in self.indices_diagonal_gcphot]
                ).flatten('F')
            else:
                gc_phot_array = np.array(
                    [self.phot_ins.corr_func_3x2pt('position-position',
                                                   self.scales_GC_phot,
                                                   element[0], element[1])
                     for element in self.indices_diagonal_gcphot]
                ).flatten('F')
        else:
            gc_phot_array = np.zeros(
                len(self.scales_GC_phot) *
                len(self.indices_diagonal_gcphot)
            )

        # Apply any matrix transform activated with a switch
        wl_array = \
            self.transform_photo_theory_data_vector(wl_array,
                                                    obs='WL')
        xc_phot_array = \
            self.transform_photo_theory_data_vector(xc_phot_array,
                                                    obs='XC-phot')
        gc_phot_array = \
            self.transform_photo_theory_data_vector(gc_phot_array,
                                                    obs='GC-phot')

        photo_theory_vec = np.concatenate(
            (wl_array, xc_phot_array, gc_phot_array),
            axis=0)

        return photo_theory_vec

    def create_photoxcmb_data(self):
        """
        Create data for photoxcmb

        Arranges the photoxcmbx data vector for the
        likelihood into its final format

        Returns
        -------
        CMBX data vector: dict
            returns a dictionary of arrays with the transformed photoxcmbx data
        """
        CMBX_dict = {}
        CMBX_dict['kCMB'] = self.data_ins.data_dict['kCMB']['kCMB-kCMB']

        self.tomo_ind_kCMBxWL = list(
            self.data_ins.data_dict['kCMBxWL'].keys())[1:]
        CMBX_dict['kCMBxWL'] = np.array(
            [self.data_ins.data_dict['kCMBxWL'][key][scale] for scale in range(
                len(self.data_ins.data_dict['kCMBxWL']['ells']))
             for key in self.tomo_ind_kCMBxWL])

        self.tomo_ind_kCMBxGC = list(
            self.data_ins.data_dict['kCMBxGC'].keys())[1:]
        CMBX_dict['kCMBxGC'] = np.array(
            [self.data_ins.data_dict['kCMBxGC'][key][scale] for scale in range(
                len(self.data_ins.data_dict['kCMBxGC']['ells']))
             for key in self.tomo_ind_kCMBxGC])

        self.tomo_ind_ISWxGC = list(
            self.data_ins.data_dict['ISWxGC'].keys())[1:]
        CMBX_dict['ISWxGC'] = np.array(
            [self.data_ins.data_dict['ISWxGC'][key][scale] for scale in range(
                len(self.data_ins.data_dict['ISWxGC']['ells']))
             for key in self.tomo_ind_ISWxGC])

        return CMBX_dict

    def create_photoxcmb_theory(self):
        r"""Create theory vector

        Create theory vector for CMB lensing auto and cross
        with WL and GC-photo and ISW cross GCphot as well
        The cosmology dictionnary is not given as input as it
        was already updated in the instanciation of the
        create_photo_theory function

        Returns
        -------
        cmbx_theory_vec: numpy.ndarray
            returns an array  with entries being
            [kCMBxkCMB, kCMBxWL, kCMBxGC, iSWxGC]
        """
        # The binning in ell of CMB lensing is the same
        # as for the Euclid Photo WL
        # we could update this for more optimal binning

        # CMB lens class instance
        self.cmbx_ins.cmbx_update(self.phot_ins)

        # Obtain the theory for kCMB
        if self.data_handler_ins.use_kCMB:
            kCMB_array = self.cmbx_ins.Cl_kCMB(
                self.data_ins.data_dict['kCMB']['ells'])
        else:
            kCMB_array = np.zeros(self.data_handler_ins._kCMB_size)

        # Obtain the theory for WL X kCMB
        # (binning indices start at one)
        if self.data_handler_ins.use_kCMB_wl:
            kCMBxWL_array = np.array(
                [self.cmbx_ins.Cl_kCMB_X_WL(
                    self.data_ins.data_dict['kCMBxWL']['ells'], bin_i + 1)
                    for bin_i in range(self.data_ins.numtomo_wl)]
                                    ).flatten('F')
        else:
            kCMBxWL_array = np.zeros(self.data_handler_ins._kCMBxWL_size)

        # Obtain the theory for GC-Phot X kCMB
        if self.data_handler_ins.use_kCMB_gc:
            kCMBxGC_array = np.array(
                [self.cmbx_ins.Cl_kCMB_X_GC_phot(
                    self.data_ins.data_dict['kCMBxGC']['ells'], bin_i + 1)
                    for bin_i in range(self.data_ins.numtomo_gcphot)]
                                    ).flatten('F')
        else:
            kCMBxGC_array = np.zeros(self.data_handler_ins._kCMBxGC_size)

        # Obtain the theory for ISWxGC
        if self.data_handler_ins.use_iswxgc:
            iswxgc_array = np.array(
                [self.cmbx_ins.Cl_ISWxGC(
                    self.data_ins.data_dict['ISWxGC']['ells'], bin_i + 1)
                    for bin_i in range(self.data_ins.numtomo_gcphot)]
                                    ).flatten('F')
        else:
            iswxgc_array = np.zeros(
                len(self.data_ins.data_dict['ISWxGC']['ells']) *
                self.data_ins.numtomo_gcphot
            )

        cmbx_theory_vec = np.concatenate(
                (kCMB_array, kCMBxWL_array, kCMBxGC_array, iswxgc_array),
                axis=0)

        return cmbx_theory_vec

    def precompute_matrix_transform_phot(self):
        """Precompute Matrix Transform Phot

        Precompute matrices needed for matrix transforms of data and
        theory vectors, using fiducial quantities at initialization.
        Matrices are stored in the object corresponding to the chosen
        transform specified in the self.matrix_transform_phot key.

        """
        if self.do_photo:
            if not self.matrix_transform_phot:
                return None
            elif 'BNT' in self.matrix_transform_phot:
                print("** Pre-computing BNT matrix **")
                zwin = self.fiducial_cosmo_quantities_dic['z_win']
                nuisance_dict = \
                    self.fiducial_cosmo_quantities_dic['nuisance_parameters']
                nz_WL = RedshiftDistribution('WL', self.phot_ins.nz_dic_WL,
                                             nuisance_dict)
                chiwin = self.fiducial_cosmo_quantities_dic['r_z_func'](zwin)
                Nz = nz_WL.get_num_tomographic_bins()
                ni_list = np.array([nz_WL.interpolates_n_i(ni + 1, zwin)(zwin)
                                   for ni in range(Nz)])
                if 'test' in self.matrix_transform_phot:
                    test_BNT = True
                    print("** Testing BNT with unity matrix **")
                else:
                    test_BNT = False
                self.BNT_transformation = BNT_transform(zwin, chiwin, ni_list,
                                                        test_unity=test_BNT)
            else:
                print("Warning: specified matrix transform not implemented.")
        return None

    def transform_photo_theory_data_vector(self, obs_array,
                                           obs='WL'):
        """Transform Photo Theory Data Vector

        Transform the photometric theory and data vector
        with a generic matrix transformation
        specified in the 'matrix_transform_phot' key
        of the info dictionary

        Parameters
        ----------
        obs_array: array
            Array containing the original (untransformed)
            photometric theory/data vector
        dictionary: dict
            cosmology dictionary from the Cosmology class
        obs: string
            String specifying the photometric
            observable which will be transformed.
            Default: 'WL'

        Returns
        -------
        transformed_array: array
            Returns an array with the transformed
            photometric theory/data vector
            If the requested transform is unapplicable, returns the original
            photometric theory/data vector
        """

        transformed_array = obs_array
        if not self.matrix_transform_phot:
            # Not doing any matrix transform
            return transformed_array
        elif 'BNT' in self.matrix_transform_phot:
            if obs == 'WL':
                N_scales = len(self.scales_WL)
                transformed_array = \
                    self.BNT_transformation\
                    .apply_vectorized_symmetric_BNT(
                                                    N_scales,
                                                    obs_array)
            elif obs == 'XC-phot':
                N_scales = len(self.scales_XC)
                transformed_array = \
                    self.BNT_transformation\
                    .apply_vectorized_nonsymmetric_BNT(
                                                       N_scales,
                                                       obs_array)
            elif obs == 'GC-phot':
                transformed_array = obs_array
            else:
                print("In method:transform_photo_theory_data_vector,  \
                     observable passed will not be transformed")
                transformed_array = obs_array
            return transformed_array
        else:
            raise ValueError("Specified matrix_transform_phot \
                             is not available in CLOE")

    def create_spectro_theory(self, dictionary):
        """Creates the spectroscopic theory.

        Obtains the theory for the likelihood.
        The theory is evaluated only if the GCspectro probe is enabled in the
        masking vector.

        Parameters
        ----------
        dictionary: dict
            Cosmology dictionary from the :obj:`cosmology`
            class which is updated at each sampling step


        Returns
        -------
        Spectroscopic theory vector: list
            Theory array with same indexing/format as the data.
            If the GCspectro probe is not enabled in the masking vector,
            an array of zeros of the same size is returned.
        """

        self.spec_ins.update(dictionary)
        k_m_matrices = []
        for z_ins in self.zkeys:
            m_ins = [int(str(key)[-1]) for key in
                     self.data_ins.data_dict['GC-Spectro'][z_ins].keys()
                     if key.startswith(self.str_start_spectro)]
            if (self.do_fourier_spectro and (not self.do_convolved_multipole)):
                k_m_matrix = []
                for scale_ins in (
                        self.data_ins.data_dict['GC-Spectro'][z_ins][
                            self.scale_var_spectro]):
                    k_m_matrix.append(
                        self.spec_ins.multipole_spectra(
                            float(z_ins),
                            scale_ins,
                            ms=m_ins)
                    )
                k_m_matrices.append(k_m_matrix)
                theoryvec = np.hstack(k_m_matrices).T.flatten()
            elif (self.do_fourier_spectro and self.do_convolved_multipole):
                k_m_matrices.append(
                    self.spec_ins.convolved_power_spectrum_multipoles(
                        float(z_ins)))
                theoryvec = np.array(k_m_matrices).flatten()
            else:
                k_m_matrices.append(
                    self.spec_ins.multipole_correlation_function(
                        self.data_ins.data_dict['GC-Spectro'][z_ins][
                            self.scale_var_spectro],
                        float(z_ins),
                        m_ins)
                )
                theoryvec = np.array(k_m_matrices).flatten()

        return theoryvec

    def create_spectro_data(self):
        """Arranges the spectroscopic data.

        Arranges the data vector for the likelihood into its final format.

        Returns
        -------
        Spectroscopic data vector: list
            Data organised as a single array
            across redshift, cosine and wavenumber
        """

        datavec = []
        for z_ins in self.zkeys:
            multipoles = (
                [k for k in
                 self.data_ins.data_dict['GC-Spectro'][z_ins].keys()
                 if k.startswith(self.str_start_spectro)])
            for m_ins in multipoles:
                datavec = np.append(datavec, self.data_ins.data_dict[
                              'GC-Spectro'][z_ins][m_ins])
        return datavec

    def create_spectro_cov(self):
        """Arranges the spectroscopic covariance.

        Arranges the spectroscopic covariance for the likelihood
        into its final format.

        Returns
        -------
        Spectroscopic covariance: numpy.ndarray
            Single NXN covariance from sub-covariances
            (split in redshift)
        """

        # covnumsc generalizes so that each z can have different binning
        covnumsc = [0]
        for z_ins in self.zkeys:
            num_multipoles = len(
                [k for k in
                 self.data_ins.data_dict['GC-Spectro'][z_ins].keys()
                 if k.startswith(self.str_start_spectro)])
            covnumsc.append(
                num_multipoles *
                len(self.data_ins.data_dict['GC-Spectro'][z_ins][
                    self.scale_var_spectro]))

        # Put all covariances into a single/larger covariance.
        # As no cross-covariances, this takes on a block-form
        # along the diagonal.
        covfull = np.zeros([sum(covnumsc), sum(covnumsc)])
        kc = 0
        c1 = 0
        c2 = 0
        for z_ins in self.zkeys:
            c1 = c1 + covnumsc[kc]
            c2 = c2 + covnumsc[kc + 1]
            covfull[c1:c2, c1:c2] = self.data_ins.data_dict['GC-Spectro'][
                                        z_ins]['cov']
            kc = kc + 1

        return covfull

    # Create data vectors and covariances

    def create_CG_theory(self, dictionary):
        """Create CG Theory

        Obtains the theory for the likelihood.

        Parameters
        ----------
        dictionary: dict
            Cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        dictionary_fiducial: dict
            Cosmology dictionary from the Cosmology class
            at the fiducial cosmology

        Returns
        -------
        theoryvec: list
            Returns the theory list with same indexing/format as the data
        """

        CG_ins = CG(dictionary)

        CG_like_selection = \
            self.observables['specifications']['CG']['CG_probe']

        self.theoryvecbuf = CG_ins.N_zbin_Lbin_Rbin()

        if CG_like_selection in ['CC', 'CC_CWL', 'CC_Cxi2', 'CC_CWL_Cxi2']:
            theoryvec_buf = copy.deepcopy(self.theoryvecbuf[0])
            theoryvec_CC = np.zeros(
                theoryvec_buf.shape[0] * theoryvec_buf.shape[1]
            )
            n = -1
            for i in range(theoryvec_buf.shape[0]):
                for j in range(theoryvec_buf.shape[1]):
                    n = n + 1
                    theoryvec_CC[n] = theoryvec_buf[i][j]

        if CG_like_selection in ['CC_CWL', 'CC_CWL_Cxi2']:
            theoryvec_buf = copy.deepcopy(self.theoryvecbuf[1])
            theoryvec_MoR = np.zeros(
                theoryvec_buf.shape[0] * theoryvec_buf.shape[1] *
                theoryvec_buf.shape[2]
            )
            n = -1
            for i in range(theoryvec_buf.shape[0]):
                for j in range(theoryvec_buf.shape[1]):
                    for k in range(theoryvec_buf.shape[2]):
                        n = n + 1
                        theoryvec_MoR[n] = theoryvec_buf[i][j][k]

        if CG_like_selection in ['CC_Cxi2', 'CC_CWL_Cxi2']:
            theoryvec_buf = copy.deepcopy(self.theoryvecbuf[2])
            theoryvec_xi2 = np.zeros(
                theoryvec_buf.shape[0] * theoryvec_buf.shape[1] *
                theoryvec_buf.shape[2]
            )
            n = -1
            for i in range(theoryvec_buf.shape[0]):
                for j in range(theoryvec_buf.shape[1]):
                    for k in range(theoryvec_buf.shape[2]):
                        n = n + 1
                        theoryvec_xi2[n] = theoryvec_buf[i][j][k]

        if CG_like_selection == 'CC':
            return theoryvec_CC
        elif CG_like_selection == 'CC_CWL':
            return theoryvec_CC, theoryvec_MoR
        elif CG_like_selection == 'CC_Cxi2':
            return theoryvec_CC, theoryvec_xi2
        elif CG_like_selection == 'CC_CWL_Cxi2':
            return theoryvec_CC, theoryvec_MoR, theoryvec_xi2

    def create_CG_cov_analytic(self, CG_xi2_cov_selection):
        """Create CG cov Theory

        Computes the analytic covariance for the likelihood.

        Parameters
        ----------
        dictionary: dict
            Cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        dictionary_fiducial: dict
            Cosmology dictionary from the Cosmology class
            at the fiducial cosmology

        Returns
        -------
        theoryvec: list
            Returns the theory list with same indexing/format as the data
        """

        if CG_xi2_cov_selection in ['covCC', 'covCC_covCxi2']:
            theoryvec_cov_buf = copy.deepcopy(self.theoryvecbuf[3])
            theoryvec_CC_cov = np.zeros((
                theoryvec_cov_buf.shape[0] * theoryvec_cov_buf.shape[2],
                theoryvec_cov_buf.shape[1] * theoryvec_cov_buf.shape[3]
            ))
            for i in range(theoryvec_cov_buf.shape[0]):
                for j in range(theoryvec_cov_buf.shape[2]):
                    for k in range(theoryvec_cov_buf.shape[1]):
                        for ll in range(theoryvec_cov_buf.shape[3]):
                            idx1 = i * theoryvec_cov_buf.shape[2] + j
                            idx2 = k * theoryvec_cov_buf.shape[3] + ll
                            theoryvec_CC_cov[idx1][idx2] =\
                                theoryvec_cov_buf[i, k, j, ll]

        if CG_xi2_cov_selection in ['covxi', 'covCC_covCxi2']:
            theoryvec_cov_buf = copy.deepcopy(self.theoryvecbuf[4])
            theoryvec_xi2_cov = np.zeros((
                theoryvec_cov_buf.shape[0] * theoryvec_cov_buf.shape[2] *
                theoryvec_cov_buf.shape[4],
                theoryvec_cov_buf.shape[1] * theoryvec_cov_buf.shape[3] *
                theoryvec_cov_buf.shape[5]
            ))
            for i in range(theoryvec_cov_buf.shape[0]):
                for j in range(theoryvec_cov_buf.shape[1]):
                    for k in range(theoryvec_cov_buf.shape[2]):
                        for ll in range(theoryvec_cov_buf.shape[3]):
                            for m in range(theoryvec_cov_buf.shape[4]):
                                for n in range(theoryvec_cov_buf.shape[5]):
                                    theoryvec_xi2_cov[
                                        i * theoryvec_cov_buf.shape[2] *
                                        theoryvec_cov_buf.shape[4] +
                                        k * theoryvec_cov_buf.shape[4] + m,
                                        j * theoryvec_cov_buf.shape[3] *
                                        theoryvec_cov_buf.shape[5] +
                                        ll * theoryvec_cov_buf.shape[5] + n
                                    ] = theoryvec_cov_buf[i, j, k, ll, m, n]

        if CG_xi2_cov_selection == 'covCC':
            return np.linalg.inv(theoryvec_CC_cov)
        if CG_xi2_cov_selection == 'covCxi2':
            return np.linalg.inv(theoryvec_xi2_cov)
        if CG_xi2_cov_selection == 'covCC_covCxi2':
            return np.linalg.inv(theoryvec_CC_cov), \
                np.linalg.inv(theoryvec_xi2_cov)

    def create_CG_data(self):
        """Create CG Data

        Arranges the data vector for the likelihood into its final format

        Returns
        -------
        datavec: list
            Returns the data as a single list
        """

        datavec_CC = np.zeros(
            self.data_ins.data_dict['CG_CC'].shape[0] *
            self.data_ins.data_dict['CG_CC'].shape[1]
        )
        n = -1
        for i in range(self.data_ins.data_dict['CG_CC'].shape[0]):
            for j in range(self.data_ins.data_dict['CG_CC'].shape[1]):
                n = n + 1
                datavec_CC[n] = self.data_ins.data_dict['CG_CC'][i][j]

        datavec_MoR = np.zeros(
            self.data_ins.data_dict['CG_MoR'].shape[0] *
            self.data_ins.data_dict['CG_MoR'].shape[1]
        )
        n = -1
        for i in range(self.data_ins.data_dict['CG_MoR'].shape[0]):
            for j in range(self.data_ins.data_dict['CG_MoR'].shape[1]):
                n = n + 1
                datavec_MoR[n] = self.data_ins.data_dict['CG_MoR'][i][j]

        datavec_xi2 = np.zeros(
            self.data_ins.data_dict['CG_xi2'].shape[0] *
            self.data_ins.data_dict['CG_xi2'].shape[1] *
            self.data_ins.data_dict['CG_xi2'].shape[2]
        )
        n = -1
        for i in range(self.data_ins.data_dict['CG_xi2'].shape[0]):
            for j in range(self.data_ins.data_dict['CG_xi2'].shape[1]):
                for k in range(self.data_ins.data_dict['CG_xi2'].shape[2]):
                    n = n + 1
                    datavec_xi2[n] = self.data_ins.data_dict['CG_xi2'][i][j][k]

        return datavec_CC, datavec_MoR, datavec_xi2

    def create_CG_cov_external(self):
        """Create CG Cov

        Arranges the external covariance matrix for the likelihood
        into its final format

        Returns
        -------
        covfull: float N x N matrix
            Returns a single covariance from sub-covariances (split in z)
        """
        covfull_CC = np.zeros((
            self.data_ins.data_dict['CG_cov_CC'].shape[0] *
            self.data_ins.data_dict['CG_cov_CC'].shape[1]
        ))
        n = -1
        for i in range(self.data_ins.data_dict['CG_cov_CC'].shape[0]):
            for j in range(self.data_ins.data_dict['CG_cov_CC'].shape[1]):
                n = n + 1
                covfull_CC[n] = self.data_ins.data_dict['CG_cov_CC'][i][j]

        covfull_MoR = np.zeros(
            self.data_ins.data_dict['CG_MoR'].shape[0] *
            self.data_ins.data_dict['CG_MoR'].shape[1]
        )
        n = -1
        for i in range(self.data_ins.data_dict['CG_MoR'].shape[0]):
            for j in range(self.data_ins.data_dict['CG_MoR'].shape[1]):
                n = n + 1
                covfull_MoR[n] = self.data_ins.data_dict['CG_cov_MoR'][i][j]

        covfull_xi2 = np.zeros(
            self.data_ins.data_dict['CG_xi2'].shape[0] *
            self.data_ins.data_dict['CG_xi2'].shape[1] *
            self.data_ins.data_dict['CG_xi2'].shape[2]
        )
        n = -1
        for i in range(self.data_ins.data_dict['CG_xi2'].shape[0]):
            for j in range(self.data_ins.data_dict['CG_xi2'].shape[1]):
                for k in range(self.data_ins.data_dict['CG_xi2'].shape[2]):
                    n = n + 1
                    covfull_xi2[n] = \
                        self.data_ins.data_dict['CG_cov_xi2'][i][j][k]

        return covfull_CC, covfull_MoR, covfull_xi2

    def loglike(self, dictionary, npar=None):
        """Natural logarithm of the likelihood.

        Calculates the log-likelihood for a given model.

        Parameters
        ----------
        dictionary: dict
            Cosmology dictionary from the :obj:`cosmology` class
            which is updated at each sampling step
        npar: int
            Number of sampled parameters (needed in case of
            numerical covariances, optional, default None)

        Returns
        -------
        Likelihood: float
            loglike = Ln(likelihood) for the Euclid observables
        """

        print_theory = dictionary['print_theory']
        if self.do_photo:
            photo_theory_vec = self.create_photo_theory(dictionary)
            if self.do_cmbx:
                cmbx_theory_vec = self.create_photoxcmb_theory()
                photo_theory_vec = np.concatenate(
                    (photo_theory_vec, cmbx_theory_vec), axis=0)
            self.mask_ins_phot.set_theory_vector(photo_theory_vec)
            masked_data_minus_theory_phot = (
                    self.masked_data_vector_phot -
                    self.mask_ins_phot.get_masked_theory_vector())
            chi2_phot = np.dot(
                    np.dot(
                        masked_data_minus_theory_phot,
                        self.masked_invcov_matrix_phot),
                    masked_data_minus_theory_phot)
            # If the covariance is numerical we use the non-Gaussian likelihood
            # from Percival et al. 2022 Eq. 52
            if self.data['photo']['cov_is_num']:
                B_phot = (self.nsim_phot - self.ndata_phot - 2.0) / \
                    ((self.nsim_phot - self.ndata_phot - 1.0) *
                        (self.nsim_phot - self.ndata_phot - 4.0))
                m_phot = npar + 2.0 + (self.nsim_phot - 1.0 +
                                       B_phot * (self.ndata_phot - npar)) / \
                                      (1.0 + B_phot * (self.ndata_phot - npar))
                loglike_phot = - m_phot / 2.0 * np.log(
                        1.0 + chi2_phot / (self.nsim_phot - 1.0))
            else:
                loglike_phot = -0.5 * chi2_phot
        else:
            loglike_phot = 0.0
        if self.do_spectro:
            spectro_theory_vec = self.create_spectro_theory(dictionary)
            self.mask_ins_spectro.set_theory_vector(spectro_theory_vec)
            masked_data_minus_theory_spectro = (
                    self.masked_data_vector_spectro -
                    self.mask_ins_spectro.get_masked_theory_vector())
            chi2_spectro = np.dot(
                    np.dot(
                        masked_data_minus_theory_spectro,
                        self.masked_invcov_matrix_spectro),
                    masked_data_minus_theory_spectro)
            # If the covariance is numerical we use the non-Gaussian likelihood
            # from Percival et al. 2022 Eq. 52
            if self.data['spectro']['cov_is_num']:
                B_spectro = (self.nsim_spectro - self.ndata_spectro - 2.0) / \
                    ((self.nsim_spectro - self.ndata_spectro - 1.0) *
                     (self.nsim_spectro - self.ndata_spectro - 4.0))
                m_spectro = npar + 2.0 + (
                            self.nsim_spectro - 1.0 +
                            B_spectro * (self.ndata_spectro - npar)) / \
                    (1.0 + B_spectro * (self.ndata_spectro - npar))
                loglike_spectro = - m_spectro / 2.0 * np.log(
                        1.0 + chi2_spectro / (self.nsim_spectro - 1.0))
            else:
                loglike_spectro = -0.5 * chi2_spectro
        else:
            loglike_spectro = 0.0

        if self.do_clusters:
            obs = self.observables['specifications']['CG']
            CG_like_selection = obs['CG_probe']
            CG_xi2_cov_selection = obs['CG_xi2_cov_selection']

            if CG_like_selection == 'CC':
                self.CGCCthvec = self.create_CG_theory(
                    dictionary
                )
                dmt = self.CGCCdatafinal - self.CGCCthvec
                if CG_xi2_cov_selection == 'covCC':
                    self.CGinvcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)
                    loglike_clusters = -0.5 * np.dot(
                        np.dot(dmt, self.CGinvcovCCfinal), dmt
                    )
                elif CG_xi2_cov_selection == 'CG_nonanalytic_cov':
                    loglike_clusters = \
                        -0.5 * np.sum((dmt * self.CGinvcovCCfinal)**2.0)
                else:
                    err_msg = "Choose CG covariance selection "
                    err_msg += "\'CG_nonanalytic_cov\' or \'covCC\'"
                    raise CobayaInterfaceError(err_msg)
            elif CG_like_selection == 'CC_CWL':
                createCGtheory = self.create_CG_theory(
                    dictionary
                )
                self.CGCCthvec = createCGtheory[0]
                self.CGMoRthvec = createCGtheory[1]
                dmtCC = self.CGCCdatafinal - self.CGCCthvec
                dmtMoR = self.CGMoRdatafinal - self.CGMoRthvec
                if CG_xi2_cov_selection == 'covCC':
                    self.CGinvcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)
                    loglike_clusters = - 0.5 * np.dot(
                        np.dot(dmtCC, self.CGinvcovCCfinal), dmtCC
                    ) - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0)
                elif CG_xi2_cov_selection == 'CG_nonanalytic_cov':
                    loglike_clusters = - 0.5 * np.sum(
                        (dmtCC * self.CGinvcovCCfinal)**2.0
                    ) - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0)
                else:
                    err_msg = "Choose CG covariance selection "
                    err_msg += "\'CG_nonanalytic_cov\' or \'covCC\'"
                    raise CobayaInterfaceError(err_msg)
            elif CG_like_selection == 'CC_Cxi2':
                createCGtheory = self.create_CG_theory(
                    dictionary
                )
                self.CGCCthvec = createCGtheory[0]
                self.CGxi2thvec = createCGtheory[1]
                dmtCC = self.CGCCdatafinal - self.CGCCthvec
                dmtxi2 = self.CGxi2datafinal - self.CGxi2thvec
                if CG_xi2_cov_selection == 'covCC':
                    self.CGinvcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)
                    loglike_clusters = - 0.5 * np.dot(
                        np.dot(dmtCC, self.CGinvcovCCfinal), dmtCC
                    ) - 0.5 * np.sum((dmtxi2 * self.CGinvcovxi2final)**2.0)
                elif CG_xi2_cov_selection == 'covxi2':
                    self.xi2invcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)
                    loglike_clusters = - 0.5 * np.sum(
                        (dmtCC * self.CGinvcovCCfinal)**2.0
                    ) - 0.5 * np.dot(
                        np.dot(dmtxi2, self.xi2invcovCCfinal), dmtxi2
                    )
                elif CG_xi2_cov_selection == 'covCC_covCxi2':
                    self.CGinvcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)[0]
                    self.xi2invcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)[1]
                    loglike_clusters = - 0.5 * np.dot(
                        np.dot(dmtCC, self.CGinvcovCCfinal), dmtCC
                    ) - 0.5 * np.dot(
                        np.dot(dmtxi2, self.xi2invcovCCfinal), dmtxi2
                    )
                elif CG_xi2_cov_selection == 'CG_nonanalytic_cov':
                    loglike_clusters = - 0.5 * np.sum(
                        (dmtCC * self.CGinvcovCCfinal)**2.0
                    ) - 0.5 * np.sum((dmtxi2 * self.CGinvcovxi2final)**2.0)
                else:
                    err_msg = "Choose CG covariance selection "
                    err_msg += "\'CG_nonanalytic_cov\' or" + \
                        " \'covCxi2\' or \'covCC\'"
                    err_msg += "or \'covCC_covCxi2\'"
                    raise CobayaInterfaceError(err_msg)
            elif CG_like_selection == 'CC_CWL_Cxi2':
                createCGtheory = self.create_CG_theory(
                    dictionary
                )
                self.CGCCthvec = createCGtheory[0]
                self.CGMoRthvec = createCGtheory[1]
                self.CGxi2thvec = createCGtheory[2]
                dmtCC = self.CGCCdatafinal - self.CGCCthvec
                dmtMoR = self.CGMoRdatafinal - self.CGMoRthvec
                dmtxi2 = self.CGxi2datafinal - self.CGxi2thvec
                if CG_xi2_cov_selection == 'covCC':
                    self.CGinvcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)
                    loglike_clusters = - 0.5 * np.dot(
                        np.dot(dmtCC, self.CGinvcovCCfinal), dmtCC
                    ) - 0.5 * np.sum(
                        (dmtMoR * self.CGinvcovMoRfinal)**2.0
                    ) - 0.5 * np.sum((dmtxi2 * self.CGinvcovxi2final)**2.0)
                elif CG_xi2_cov_selection == 'covCxi2':
                    self.xi2invcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)
                    loglike_clusters = - 0.5 * np.sum(
                        (dmtCC * self.CGinvcovCCfinal)**2.0
                    ) - 0.5 * np.sum(
                        (dmtMoR * self.CGinvcovMoRfinal)**2.0
                    ) - 0.5 * np.dot(
                        np.dot(dmtxi2, self.xi2invcovCCfinal),
                        dmtxi2
                    )
                elif CG_xi2_cov_selection == 'covCC_covCxi2':
                    self.CGinvcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)[0]
                    self.xi2invcovCCfinal = \
                        self.create_CG_cov_analytic(CG_xi2_cov_selection)[1]
                    loglike_clusters = -0.5 * np.dot(
                        np.dot(dmtCC, self.CGinvcovCCfinal), dmtCC
                    ) - 0.5 * np.sum(
                        (dmtMoR * self.CGinvcovMoRfinal)**2.0
                    ) - 0.5 * np.dot(
                        np.dot(dmtxi2, self.xi2invcovCCfinal), dmtxi2
                    )
                elif CG_xi2_cov_selection == 'CG_nonanalytic_cov':
                    loglike_clusters = \
                        - 0.5 * np.sum((dmtCC * self.CGinvcovCCfinal)**2.0) \
                        - 0.5 * np.sum(
                            (dmtMoR * self.CGinvcovMoRfinal)**2.0
                        ) - 0.5 * np.sum(
                            (dmtxi2 * self.CGinvcovxi2final)**2.0
                        )
                else:
                    err_msg = "Choose CG covariance selection "
                    err_msg += "\'CG_nonanalytic_cov\' or \'covCxi2\' or "
                    err_msg += "\'covCC\' or \'covCC_covCxi2\'"
                    raise CobayaInterfaceError(err_msg)
            else:
                err_msg = "Choose CG like selection "
                err_msg += "\'CC\' or \'CC_CWL\' or \'CC_Cxi2\' or "
                err_msg += "\'CC_CWL_Cxi2\'"
                raise CobayaInterfaceError(err_msg)

        else:
            loglike_clusters = 0.0

        # Total likelihood
        loglike = loglike_phot + loglike_spectro + loglike_clusters

        if print_theory:
            print('Printing the theory vector and exiting!')
            if self.do_photo:
                np.savetxt('photo_theory.dat', photo_theory_vec)
            if self.do_spectro:
                np.savetxt('spectro_theory.dat', spectro_theory_vec)
            if self.do_clusters:
                if CG_like_selection == 'CC':
                    np.savetxt('CC_theory.dat', self.CGCCthvec)
                if CG_like_selection == 'CC_CWL':
                    np.savetxt('CC_theory.dat', self.CGCCthvec)
                    np.savetxt('CWL_theory.dat', self.CGMoRthvec)
                if CG_like_selection == 'CC_Cxi2':
                    np.savetxt('CC_theory.dat', self.CGCCthvec)
                    np.savetxt('Cxi2_theory.dat', self.CGxi2thvec)
                if CG_like_selection == 'CC_CWL_Cxi2':
                    np.savetxt('CC_theory.dat', self.CGCCthvec)
                    np.savetxt('CWL_theory.dat', self.CGMoRthvec)
                    np.savetxt('Cxi2_theory.dat', self.CGxi2thvec)
            exit()

        return loglike
