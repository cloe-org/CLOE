"""EUCLIKE

Contains class to compute the Euclid likelihood.
"""

import numpy as np
from cloe.photometric_survey.photo import Photo
from cloe.spectroscopic_survey.spectro import Spectro
from cloe.data_reader import reader
from cloe.masking.masking import Masking
from cloe.masking.data_handler import Data_handler
from cloe.photometric_survey.redshift_distribution \
    import RedshiftDistribution
from cloe.auxiliary.matrix_transforms import BNT_transform


class EuclikeError(Exception):
    r"""
    Class to define Exception Error.
    """

    pass


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
        self.do_spectro = any(observables['selection']['GCspectro'].values())

        self.data = data
        if self.do_spectro:
            if observables['specifications']['GCspectro']['statistics'] == \
                    'legendre_multipole_power_spectrum':
                self.do_fourier_spectro = True
                self.str_start_spectro = 'pk'
                self.scale_var_spectro = 'k_pk'
            elif observables['specifications']['GCspectro']['statistics'] == \
                    'legendre_multipole_correlation_function':
                self.do_fourier_spectro = False
                self.str_start_spectro = 'xi'
                self.scale_var_spectro = 'r_xi'
            else:
                raise ValueError('Unknown statistics_spectro choice. '
                                 'Use legendre_multipole_power_spectrum '
                                 'or legendre_multipole_correlation_function')
            self.data['spectro']['Fourier'] = self.do_fourier_spectro
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

            self.ells_WL = self.data_ins.data_dict['WL']['ells']
            self.ells_XC = self.data_ins.data_dict['XC-Phot']['ells']
            self.ells_GC_phot = self.data_ins.data_dict['GC-Phot']['ells']

            # Flag to select cases with or without RSD for photometric probes
            add_RSD = observables['selection']['add_phot_RSD']
            # default value of matrix_transform_phot, gets modified by cobaya
            # interface
            self.matrix_transform_phot = \
                self.observables['selection']['matrix_transform_phot']

            # Photo class instance
            self.phot_ins = Photo(None,
                                  self.data_ins.nz_dict_WL,
                                  self.data_ins.nz_dict_GC_Phot,
                                  add_RSD=add_RSD)

            # Temporary placeholder for theta vector
            # (will be read from file eventually)
            theta_min = 0.005
            theta_max = 20.0
            nbins_theta = 30
            theta_deg = np.logspace(np.log10(theta_min),
                                    np.log10(theta_max),
                                    nbins_theta)
            theta_rad = np.deg2rad(theta_deg)

            # Sets the precomputed Bessel functions as an attribute of the
            # Photo class
            self.phot_ins._set_bessel_tables(theta_rad)

            # set the precomputed prefactors for the WL, XC and GCphot Cl's
            self.phot_ins.set_prefactor(ells_WL=self.ells_WL,
                                        ells_XC=self.ells_XC,
                                        ells_GC_phot=self.ells_GC_phot)

        if self.do_spectro:
            # Read spectro
            self.data_ins.read_GC_spectro()
            self.zkeys = self.data_ins.data_dict['GC-Spectro'].keys()

            # Spectro class instance
            self.spec_ins = Spectro(None, list(self.zkeys))

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
        # We ignore the first value (ells)
        self.tomo_ind_GC_phot = \
            list(self.data_ins.data_dict['GC-Phot'].keys())[1:]
        datavec_dict['GC-Phot'] = np.array(
                [self.data_ins.data_dict['GC-Phot'][key][ell]
                 for ell in range(len(self.ells_GC_phot))
                 for key in self.tomo_ind_GC_phot])

        self.tomo_ind_WL = list(self.data_ins.data_dict['WL'].keys())[1:]
        datavec_dict['WL'] = np.array(
                [self.data_ins.data_dict['WL'][key][ell]
                 for ell in range(len(self.ells_WL))
                 for key in self.tomo_ind_WL])

        self.tomo_ind_XC = list(self.data_ins.data_dict['XC-Phot'].keys())[1:]
        datavec_dict['XC-Phot'] = np.array(
                [self.data_ins.data_dict['XC-Phot'][key][ell]
                 for ell in range(len(self.ells_XC))
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
                                              datavec_dict['GC-Phot']), axis=0)

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
            wl_array = np.array(
                [self.phot_ins.Cl_WL(self.ells_WL, element[0], element[1])
                 for element in self.indices_diagonal_wl]).flatten('F')
        else:
            wl_array = np.zeros(
                 len(self.ells_WL) * len(self.indices_diagonal_wl))

        # Obtain the theory for XC-Phot
        if self.data_handler_ins.use_xc_phot:
            xc_phot_array = np.array(
                [self.phot_ins.Cl_cross(self.ells_XC, element[1], element[0])
                 for element in self.indices_all]).flatten('F')
        else:
            xc_phot_array = np.zeros(
                 len(self.ells_XC) * len(self.indices_all))

        # Obtain the theory for GC-Phot
        if self.data_handler_ins.use_gc_phot:
            gc_phot_array = np.array(
                [self.phot_ins.Cl_GC_phot(self.ells_GC_phot,
                                          element[0], element[1])
                 for element in self.indices_diagonal_gcphot]).flatten('F')
        else:
            gc_phot_array = np.zeros(
                len(self.ells_GC_phot) * len(self.indices_diagonal_gcphot))

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
            (wl_array, xc_phot_array, gc_phot_array), axis=0)

        return photo_theory_vec

    def precompute_matrix_transform_phot(self):
        """Precompute Matrix Transform Phot

        Precompute matrices needed for matrix transforms of data and
        theory vectors, using fiducial quantities at initialization.
        Matrices are stored in the object corresponding to the chosen
        transform specified in the self.matrix_transform_phot key.

        """
        if self.do_photo:
            print(self.matrix_transform_phot)
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
                N_ells = len(self.ells_WL)
                transformed_array = \
                    self.BNT_transformation\
                    .apply_vectorized_symmetric_BNT(
                                                    N_ells,
                                                    obs_array)
            elif obs == 'XC-phot':
                N_ells = len(self.ells_XC)
                transformed_array = \
                    self.BNT_transformation\
                    .apply_vectorized_nonsymmetric_BNT(
                                                       N_ells,
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
            if self.do_fourier_spectro:
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

        if self.do_photo:
            photo_theory_vec = self.create_photo_theory(dictionary)
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

        # Total likelihood
        loglike = loglike_phot + loglike_spectro

        return loglike
