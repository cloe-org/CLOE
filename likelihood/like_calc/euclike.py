# -*- coding: utf-8 -*-
"""Euclike

Contains class to compute the Euclid likelihood
"""

import numpy as np
from likelihood.cosmo.cosmology import Cosmology
from likelihood.photometric_survey.photo import Photo
from likelihood.spectroscopic_survey.spectro import Spectro
from likelihood.data_reader import reader
from likelihood.masking.masking import Masking
from likelihood.masking.data_handler import Data_handler
from likelihood.masking.masking_vector_wrapper import MaskingVectorWrapper


class EuclikeError(Exception):
    r"""
    Class to define Exception Error
    """

    pass


class Euclike:
    """
    Class to compute the Euclid likelihood from the theory, data, covariance.
    """

    def __init__(self, data, observables):
        """Initialize

        Constructor of the class Euclike. The data and covariance are
        read and arranged into their final format only once here.

        Parameters
        ----------
        data: dict
            Dictionary containing specifications for data loading and handling,
            to be passed to the data reader module.
        observables: dict
            Dictionary containing specification for the chosen observables by
            the user.
        """
        # Flag to allow printing  of likelihood selection only once.
        self.l_select_print_flag = True

        self.data = data
        self.observables = observables
        self.masking_vector = np.ones(5700)
        self.masking_vector_wrapper = MaskingVectorWrapper()
        self.masking_vector_wrapper.set_masking_vector(self.masking_vector)

        self.data_ins = reader.Reader(self.data)
        self.data_ins.compute_nz()
        # Read spectro
        self.data_ins.read_GC_spectro()
        self.zkeys = self.data_ins.data_dict['GC-Spectro'].keys()
        self.data_spectro_fiducial_cosmo = \
            self.data_ins.data_spectro_fiducial_cosmo
        # Transforming data
        self.spectrodatafinal = self.create_spectro_data()
        self.spectrocovfinal = self.create_spectro_cov()
        self.spectroinvcovfinal = np.linalg.inv(self.spectrocovfinal)
        # Read photo
        self.data_ins.read_phot()
        # Tranforming data
        self.photodatafinal = self.create_photo_data()
        self.photoinvcovfinal_GC = np.linalg.inv(
            self.data_ins.data_dict['GC-Phot']['cov'])
        self.photoinvcovfinal_WL = np.linalg.inv(
            self.data_ins.data_dict['WL']['cov'])
        self.photoinvcovfinal_XC = np.linalg.inv(
            self.data_ins.data_dict['XC-Phot']['cov_XC_only'])
        # Order of this matrix is WL, XC, GC
        self.photoinvcovfinal_all = np.linalg.inv(
            self.data_ins.data_dict['XC-Phot']['cov'])

        # Calculate permutations i,j bins for WL, GC-Phot, XC.
        # This refers to the non-redundant bin combinations for
        # which we have measurements (i.e. 1-1, 1-2, ..., 1-10,
        # 2-2, 2-3, ..., 2-10, 3-3, 3-4, etc, in the case of ten
        # tomographic bins for WL and GC-Phot. Meanhile, all bin
        # combinations exist for XC, i.e. for example both 1-2
        # and 2-1, both 1-3 and 3-1, etc).
        numtomo_wl = self.data_ins.numtomo_wl
        numtomo_gcphot = self.data_ins.numtomo_gcphot
        x_diagonal_wl = np.triu(np.ones((numtomo_wl, numtomo_wl)))
        self.indices_diagonal_wl = []
        for i in range(0, len(x_diagonal_wl)):
            for j in range(0, len(x_diagonal_wl)):
                if x_diagonal_wl[i, j] == 1:
                    self.indices_diagonal_wl.append([i + 1, j + 1])
        x_diagonal_gcphot = np.triu(np.ones((numtomo_gcphot, numtomo_gcphot)))
        self.indices_diagonal_gcphot = []
        for i in range(0, len(x_diagonal_gcphot)):
            for j in range(0, len(x_diagonal_gcphot)):
                if x_diagonal_gcphot[i, j] == 1:
                    self.indices_diagonal_gcphot.append([i + 1, j + 1])
        x = np.ones((numtomo_gcphot, numtomo_wl))
        self.indices_all = []
        for i in range(0, len(x)):
            for j in range(0, len(x)):
                self.indices_all.append([i + 1, j + 1])

        # Identifies the number of elements of the spectroscopic part of the
        # theory vector. It is evaluated once the first time it is needed
        # in create_spectro_theory() and its value is then cached for future
        # use.
        self.num_spectro_elements = None

        # Reshaping the data vectors and covarinace matrices
        # into dictionaries to be passed to the data_handler class
        self.datafinal = {**self.photodatafinal,
                          'GC-Spectro': self.spectrodatafinal}
        self.covfinal = {'WL': self.data_ins.data_dict['WL']['cov'],
                         'XC-Phot': self.data_ins.data_dict['XC-Phot'][
                            'cov_XC_only'],
                         'GC-Phot': self.data_ins.data_dict['GC-Phot']['cov'],
                         'GC-Spectro': self.spectrocovfinal}

        data_handler_ins = Data_handler(self.datafinal,
                                        self.covfinal,
                                        self.observables)
        self.data_vector, self.invcov_matrix, self.masking_vector = \
            data_handler_ins.get_data_and_masking_vector()

        mask_ins = Masking()
        mask_ins.set_data_vector(self.data_vector)
        mask_ins.set_inverse_covariance_matrix(self.invcov_matrix)
        mask_ins.set_masking_vector(self.masking_vector)
        self.masked_data_vector = mask_ins.get_masked_data_vector()
        self.masked_invcov_matrix = (
            mask_ins.get_masked_inverse_covariance_matrix())

    def create_photo_data(self):
        """Create Photo Data

        Arranges the photo data vector for the likelihood into its final format

        Returns
        -------
        datavec_dict: dict
            returns a dictionary of arrays with the transformed photo data
        """

        datavec_dict = {'GC-Phot': [], 'WL': [], 'XC-Phot': [], 'all': []}
        for index in list(self.data_ins.data_dict['WL'].keys()):
            if 'B' in index:
                del(self.data_ins.data_dict['WL'][index])
        for index in list(self.data_ins.data_dict['XC-Phot'].keys()):
            if 'B' in index:
                del(self.data_ins.data_dict['XC-Phot'][index])
        # Transform GC-Phot
        # We ignore the first value (ells) and last (cov matrix)
        datavec_dict['GC-Phot'] = np.array(
                [self.data_ins.data_dict['GC-Phot'][key][ind]
                 for ind
                 in range(len(self.data_ins.data_dict['GC-Phot']['ells']))
                 for key, v
                 in list(self.data_ins.data_dict['GC-Phot'].items())[1:-1]])

        datavec_dict['WL'] = np.array(
                [self.data_ins.data_dict['WL'][key][ind]
                 for ind in range(len(self.data_ins.data_dict['WL']['ells']))
                 for key, v
                 in list(self.data_ins.data_dict['WL'].items())[1:-1]])

        datavec_dict['XC-Phot'] = np.array(
                [self.data_ins.data_dict['XC-Phot'][key][ind]
                 for ind
                 in range(len(self.data_ins.data_dict['XC-Phot']['ells']))
                 for key, v
                 in list(self.data_ins.data_dict['XC-Phot'].items())[1:-2]])

        datavec_dict['all'] = np.concatenate((datavec_dict['WL'],
                                              datavec_dict['XC-Phot'],
                                              datavec_dict['GC-Phot']), axis=0)

        return datavec_dict

    def create_photo_theory(self, phot_ins, full_photo):
        """Create Photo Theory

        Obtains the photo theory for the likelihood.
        The theory is evaluated only for the probes specified in the masking
        vector. For the probes for which the theory is not evaluated, an array
        of zeros is included in the returned dictionary.

        Parameters
        ----------
        phot_ins: object
            initialized instance of the class Photo
        full_photo: boolean
            selects whether to use full photometric
            data (with XC) or not

        Returns
        -------
        theoryvec_dict: dict
            returns a dictionary of arrays with the transformed photo theory
            vector.
            For the probes for which the theory is not evaluated, an array of
            zeros of the same size is included in the returned dictionary.
        """
        theoryvec_dict = {'GC-Phot': None, 'WL': None, 'XC-Phot': None,
                          'all': None}
        # Obtain the theory for GC-Phot
        if self.masking_vector_wrapper.get_gc_phot_enabled():
            theoryvec_dict['GC-Phot'] = np.array(
                [phot_ins.Cl_GC_phot(ell, element[0], element[1])
                 for ell in self.data_ins.data_dict['GC-Phot']['ells']
                 for element in self.indices_diagonal_gcphot]
            )
        else:
            theoryvec_dict['GC-Phot'] = np.zeros(
                len(self.data_ins.data_dict['GC-Phot']['ells']) *
                len(self.indices_diagonal_gcphot)
            )

        # Obtain the theory for WL
        if self.masking_vector_wrapper.get_wl_enabled():
            theoryvec_dict['WL'] = np.array(
                [phot_ins.Cl_WL(ell, element[0], element[1])
                 for ell in self.data_ins.data_dict['WL']['ells']
                 for element in self.indices_diagonal_wl]
            )
        else:
            theoryvec_dict['WL'] = np.zeros(
                 len(self.data_ins.data_dict['WL']['ells']) *
                 len(self.indices_diagonal_wl)
            )

        # Obtain the theory for XC-Phot
        if self.masking_vector_wrapper.get_xc_phot_enabled():
            theoryvec_dict['XC-Phot'] = np.array(
                [phot_ins.Cl_cross(ell, element[0], element[1])
                 for ell in self.data_ins.data_dict['XC-Phot']['ells']
                 for element in self.indices_all]
            )
        else:
            theoryvec_dict['XC-Phot'] = np.zeros(
                 len(self.data_ins.data_dict['XC-Phot']['ells']) *
                 len(self.indices_all)
            )

        theoryvec_dict['all'] = np.concatenate(
            (theoryvec_dict['WL'],
             theoryvec_dict['XC-Phot'],
             theoryvec_dict['GC-Phot']),
            axis=0)

        return theoryvec_dict

    def create_spectro_theory(self, dictionary, dictionary_fiducial):
        """Create Spectro Theory

        Obtains the theory for the likelihood.
        The theory is evaluated only if the GC-Spectro probe is enabled in the
        masking vector.

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        dictionary_fiducial: dict
            cosmology dictionary from the Cosmology class
            at the fiducial cosmology

        Returns
        -------
        theoryvec: list
            returns the theory array with same indexing/format as the data.
            If the GC-Spectro probe is not enabled in the masking vector,
            an array of zeros of the same size is returned.
        """
        if self.masking_vector_wrapper.get_gc_spectro_enabled():
            spec_ins = Spectro(dictionary, dictionary_fiducial)
            m_ins = [v for k, v in dictionary['nuisance_parameters'].items()
                     if k.startswith('multipole_')]
            k_m_matrices = []
            for z_ins in self.zkeys:
                k_m_matrix = []
                for k_ins in (
                        self.data_ins.data_dict['GC-Spectro'][z_ins]['k_pk']):
                    k_m_matrix.append(
                        spec_ins.multipole_spectra(
                            float(z_ins),
                            k_ins,
                            ms=m_ins
                        )
                    )
                k_m_matrices.append(k_m_matrix)
            theoryvec = np.hstack(k_m_matrices).T.flatten()
            return theoryvec

        else:
            if self.num_spectro_elements is None:
                self.num_spectro_elements = (
                    sum(
                        [k.startswith('multipole_')
                         for k in dictionary['nuisance_parameters'].keys()]
                    ) *
                    sum(
                        len(self.data_ins.data_dict['GC-Spectro'][z]['k_pk'])
                        for z in self.zkeys
                    )
                )
            theoryvec = np.zeros(num_spectro_elements)
            return theoryvec

    def create_spectro_data(self):
        """Create Spectro Data

        Arranges the data vector for the likelihood into its final format

        Returns
        -------
        datavec: list
            returns the data as a single array across z, mu, k
        """

        datavec = []
        for z_ins in self.zkeys:
            for m_ins in [0, 2, 4]:
                datavec = np.append(datavec, self.data_ins.data_dict[
                              'GC-Spectro'][z_ins]['pk' + str(m_ins)])
        return datavec

    def create_spectro_cov(self):
        """Create Spectro Cov

        Arranges the covariance for the likelihood into its final format

        Returns
        -------
        covfull: float N x N matrix
            returns a single covariance from sub-covariances (split in z)
        """

        self.covnumz = len(self.zkeys)
        # covnumk generalizes so that each z can have different k binning
        self.covnumk = []
        self.covnumk.append(0)
        for z_ins in self.zkeys:
            self.covnumk.append(
                3 * len(self.data_ins.data_dict['GC-Spectro'][z_ins]['k_pk']))

        # Put all covariances into a single/larger covariance.
        # As no cross-covariances, this takes on a block-form
        # along the diagonal.
        covfull = np.zeros([sum(self.covnumk), sum(self.covnumk)])
        kc = 0
        c1 = 0
        c2 = 0
        for z_ins in self.zkeys:
            c1 = c1 + self.covnumk[kc]
            c2 = c2 + self.covnumk[kc + 1]
            covfull[c1:c2, c1:c2] = self.data_ins.data_dict['GC-Spectro'][
                                        z_ins]['cov']
            kc = kc + 1

        return covfull

    def loglike_photo(self, dictionary, full_photo):
        """Loglike Photo

        Calculates loglike photometric based on
        the flag 'full_photo'. If True, calculates
        all probes. If false, only calculates GC+WL
        assuming they are independent.

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        full_photo: boolean
            selects whether to use full photometric
            data (with XC) or not

        Returns
        -------
        loglike_photo: float
            returns photo-z chi2
        """

        # Photo class instance
        phot_ins = Photo(
                dictionary,
                self.data_ins.nz_dict_WL,
                self.data_ins.nz_dict_GC_Phot)
        # Obtain the theory vector
        theoryvec_dict = self.create_photo_theory(phot_ins, full_photo)
        loglike_photo = 0
        if not full_photo:
            # Construct data minus theory
            dmt_GC = self.photodatafinal['GC-Phot'] - \
                    theoryvec_dict['GC-Phot']
            dmt_WL = self.photodatafinal['WL'] - \
                theoryvec_dict['WL']
            # Obtain loglike
            loglike_GC = -0.5 * \
                np.dot(np.dot(dmt_GC, self.photoinvcovfinal_GC), dmt_GC)
            loglike_WL = -0.5 * \
                np.dot(np.dot(dmt_WL, self.photoinvcovfinal_WL), dmt_WL)
            # Save loglike
            loglike_photo = loglike_GC + loglike_WL
        # If True, calls massive cov mat
        elif full_photo:
            # Construct data minus theory
            dmt_all = self.photodatafinal['all'] - \
                    theoryvec_dict['all']

            # Obtain loglike
            loglike_photo = -0.5 * np.dot(
                np.dot(dmt_all, self.photoinvcovfinal_all),
                dmt_all)
        else:
            print('ATTENTION: full_photo has to be either True/False')
        return loglike_photo, theoryvec_dict

    def loglike(self, dictionary, dictionary_fiducial):
        """Loglike

        Calculates the log-likelihood for a given model

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        dictionary_fiducial: dict
            cosmology dictionary from the Cosmology class
            at the fiducial cosmology

        Returns
        -------
        loglike_tot: float
            loglike = Ln(likelihood) for the Euclid observables
        """
        like_selection = dictionary['nuisance_parameters']['like_selection']
        full_photo = dictionary['nuisance_parameters']['full_photo']
        if like_selection == 1:
            self.loglike_tot, self.photothvec = \
                    self.loglike_photo(dictionary, full_photo)
        elif like_selection == 2:
            self.specthvec = self.create_spectro_theory(
                             dictionary, dictionary_fiducial)
            dmt = self.spectrodatafinal - self.specthvec
            self.loglike_tot = -0.5 * np.dot(
                np.dot(dmt, self.spectroinvcovfinal), dmt)
        elif like_selection == 12:
            self.specthvec = self.create_spectro_theory(
                             dictionary, dictionary_fiducial)
            dmt = self.spectrodatafinal - self.specthvec
            self.loglike_spectro = -0.5 * np.dot(np.dot(
                                    dmt, self.spectroinvcovfinal), dmt)
            self.loglike_photo, self.photothvec = \
                self.loglike_photo(dictionary, full_photo)
            # Only addition below if no cross-covariance
            self.loglike_tot = self.loglike_photo + self.loglike_spectro
        else:
            raise CobayaInterfaceError(
                r"Choose like selection '1' or '2' or '12'")
        # The first time the likelihood is called, this will explicitly print
        # the choice of likelihood.
        if self.l_select_print_flag:
            def_msg = ''
            if like_selection == 1:
                choice = 'PHOTOMETRIC'
            elif like_selection == 2:
                choice = 'SPECTROSCOPIC'
            else:
                choice = '3x2 PT'
                def_msg = (' (IF like_selection HAS NOT BEEN EXPLICITLY' +
                           ' SPECIFIED, THE 3x2 PT IS CHOSEN BY DEFAULT.')
            print('NOTE: ', choice, ' LIKELIHOOD REQUESTED' + def_msg)
            self.l_select_print_flag = False

        return self.loglike_tot
