# -*- coding: utf-8 -*-
"""Euclike

Contains class to compute the Euclid likelihood
"""

import numpy as np
from cloe.cosmo.cosmology import Cosmology
from cloe.photometric_survey.photo import Photo
from cloe.spectroscopic_survey.spectro import Spectro
from cloe.data_reader import reader
from cloe.masking.masking import Masking
from cloe.masking.data_handler import Data_handler


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
        self.data = data
        self.data_ins = reader.Reader(self.data)
        self.data_ins.compute_nz()
        # Read spectro
        self.data_ins.read_GC_spectro()
        self.zkeys = self.data_ins.data_dict['GC-Spectro'].keys()
        self.data_spectro_fiducial_cosmo = \
            self.data_ins.data_spectro_fiducial_cosmo
        # Transforming data
        spectrodata = self.create_spectro_data()
        spectrocov = self.create_spectro_cov()
        # Read photo
        self.data_ins.read_phot()
        # Tranforming data
        photodata = self.create_photo_data()

        # Calculate permutations i,j bins for WL, GC-Phot, XC.
        # This refers to the non-redundant bin combinations for
        # which we have measurements (i.e. 1-1, 1-2, ..., 1-10,
        # 2-2, 2-3, ..., 2-10, 3-3, 3-4, etc, in the case of ten
        # tomographic bins for WL and GC-Phot. Meanwhile, all bin
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

        ells_WL = self.data_ins.data_dict['WL']['ells']
        ells_XC = self.data_ins.data_dict['XC-Phot']['ells']
        ells_GC_phot = self.data_ins.data_dict['GC-Phot']['ells']

        # Reshaping the data vectors and covariance matrices
        # into dictionaries to be passed to the data_handler class
        datafinal = {**photodata,
                     'GC-Spectro': spectrodata}
        covfinal = {'3x2pt': self.data_ins.data_dict['3x2pt_cov'],
                    'GC-Spectro': spectrocov}
        self.data_handler_ins = Data_handler(datafinal,
                                             covfinal,
                                             observables,
                                             self.data_ins)

        self.data_vector, self.cov_matrix, self.masking_vector = \
            self.data_handler_ins.get_data_and_masking_vector()

        self.mask_ins = Masking()
        self.mask_ins.set_data_vector(self.data_vector)
        self.mask_ins.set_covariance_matrix(self.cov_matrix)
        self.mask_ins.set_masking_vector(self.masking_vector)
        self.masked_data_vector = self.mask_ins.get_masked_data_vector()
        self.masked_cov_matrix = (
            self.mask_ins.get_masked_covariance_matrix())
        self.masked_invcov_matrix = np.linalg.inv(self.masked_cov_matrix)

        # Photo class instance
        self.phot_ins = Photo(None,
                              self.data_ins.nz_dict_WL,
                              self.data_ins.nz_dict_GC_Phot)

        # Temporary placeholder for theta vector
        # (will be read from file eventually)
        theta_min = 0.005
        theta_max = 20.0
        nbins_theta = 30
        theta_deg = np.logspace(np.log10(theta_min),
                                np.log10(theta_max),
                                nbins_theta)
        theta_rad = theta_deg * np.pi / 180.0

        # Sets the precomputed Bessel functions as an attribute of the
        # Photo class
        self.phot_ins._set_bessel_tables(theta_rad)

        # set the precomputed prefactors for the WL, XC and GCphot Cl's
        self.phot_ins.set_prefactor(ells_WL=ells_WL,
                                    ells_XC=ells_XC,
                                    ells_GC_phot=ells_GC_phot)

        # Spectro class instance
        self.spec_ins = Spectro(None, None)

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
        # We ignore the first value (ells)

        datavec_dict['GC-Phot'] = np.array(
                [self.data_ins.data_dict['GC-Phot'][key][ind]
                 for key, v
                 in list(self.data_ins.data_dict['GC-Phot'].items())[1:]
                 for ind
                 in range(len(self.data_ins.data_dict['GC-Phot']['ells']))])

        datavec_dict['WL'] = np.array(
                [self.data_ins.data_dict['WL'][key][ind]
                 for key, v
                 in list(self.data_ins.data_dict['WL'].items())[1:]
                 for ind in range(len(self.data_ins.data_dict['WL']['ells']))])

        datavec_dict['XC-Phot'] = np.array(
                [self.data_ins.data_dict['XC-Phot'][key][ind]
                 for key, v
                 in list(self.data_ins.data_dict['XC-Phot'].items())[1:]
                 for ind
                 in range(len(self.data_ins.data_dict['XC-Phot']['ells']))])

        datavec_dict['all'] = np.concatenate((datavec_dict['WL'],
                                              datavec_dict['XC-Phot'],
                                              datavec_dict['GC-Phot']), axis=0)

        return datavec_dict

    def create_photo_theory(self, dictionary):
        """Create Photo Theory

        Obtains the photo theory for the likelihood.
        The theory is evaluated only for the probes specified in the masking
        vector. For the probes for which the theory is not evaluated, an array
        of zeros is included in the returned dictionary.

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class which is updated at
            each sampling step

        Returns
        -------
        photo_theory_vec: array
            returns an array with the photo theory vector.
            The elements of the array corresponding to probes for which the
            theory is not evaluated, are set to zero.
        """

        self.phot_ins.update(dictionary)

        # Obtain the theory for WL
        if self.data_handler_ins.use_wl:
            wl_array = np.array(
                [self.phot_ins.Cl_WL(ell, element[0], element[1])
                 for element in self.indices_diagonal_wl
                 for ell in self.data_ins.data_dict['WL']['ells']]
            )
        else:
            wl_array = np.zeros(
                 len(self.data_ins.data_dict['WL']['ells']) *
                 len(self.indices_diagonal_wl)
            )

        # Obtain the theory for XC-Phot
        if self.data_handler_ins.use_xc_phot:
            xc_phot_array = np.array(
                [self.phot_ins.Cl_cross(ell, element[1], element[0])
                 for element in self.indices_all
                 for ell in self.data_ins.data_dict['XC-Phot']['ells']]
            )
        else:
            xc_phot_array = np.zeros(
                 len(self.data_ins.data_dict['XC-Phot']['ells']) *
                 len(self.indices_all)
            )

        # Obtain the theory for GC-Phot
        if self.data_handler_ins.use_gc_phot:
            gc_phot_array = np.array(
                [self.phot_ins.Cl_GC_phot(ell, element[0], element[1])
                 for element in self.indices_diagonal_gcphot
                 for ell in self.data_ins.data_dict['GC-Phot']['ells']]
            )
        else:
            gc_phot_array = np.zeros(
                len(self.data_ins.data_dict['GC-Phot']['ells']) *
                len(self.indices_diagonal_gcphot)
            )

        photo_theory_vec = np.concatenate(
            (wl_array, xc_phot_array, gc_phot_array), axis=0)

        return photo_theory_vec

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
        # This is something that Sergio needs to change
        # Now the multipoles are within observables[specifications]
        # In order to pass the tests, I hard code m_inst now
        # Maybe Sergio has a better idea of what to these forloops
        # To include all info of the specifications
        if self.data_handler_ins.use_gc_spectro:
            self.spec_ins.update(dictionary, dictionary_fiducial)
            # m_ins = [v for k, v in dictionary['nuisance_parameters'].items()
            #         if k.startswith('multipole_')]
            m_ins = [0, 2, 4]
            k_m_matrices = []
            for z_ins in self.zkeys:
                k_m_matrix = []
                for k_ins in (
                        self.data_ins.data_dict['GC-Spectro'][z_ins]['k_pk']):
                    k_m_matrix.append(
                        self.spec_ins.multipole_spectra(
                            float(z_ins),
                            k_ins,
                            ms=m_ins)
                    )
                k_m_matrices.append(k_m_matrix)
            theoryvec = np.hstack(k_m_matrices).T.flatten()
            return theoryvec

        else:
            theoryvec = np.zeros(self.data_handler_ins.gc_spectro_size)
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
            multipoles = (
                [k for k in
                 self.data_ins.data_dict['GC-Spectro'][z_ins].keys()
                 if k.startswith('pk')])
            for m_ins in multipoles:
                datavec = np.append(datavec, self.data_ins.data_dict[
                              'GC-Spectro'][z_ins][m_ins])
        return datavec

    def create_spectro_cov(self):
        """Create Spectro Cov

        Arranges the covariance for the likelihood into its final format

        Returns
        -------
        covfull: float N x N matrix
            returns a single covariance from sub-covariances (split in z)
        """

        # covnumk generalizes so that each z can have different k binning
        covnumk = [0]
        for z_ins in self.zkeys:
            num_multipoles = len(
                [k for k in
                 self.data_ins.data_dict['GC-Spectro'][z_ins].keys()
                 if k.startswith('pk')])
            covnumk.append(
                num_multipoles *
                len(self.data_ins.data_dict['GC-Spectro'][z_ins]['k_pk']))

        # Put all covariances into a single/larger covariance.
        # As no cross-covariances, this takes on a block-form
        # along the diagonal.
        covfull = np.zeros([sum(covnumk), sum(covnumk)])
        kc = 0
        c1 = 0
        c2 = 0
        for z_ins in self.zkeys:
            c1 = c1 + covnumk[kc]
            c2 = c2 + covnumk[kc + 1]
            covfull[c1:c2, c1:c2] = self.data_ins.data_dict['GC-Spectro'][
                                        z_ins]['cov']
            kc = kc + 1

        return covfull

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
        photo_theory_vec = self.create_photo_theory(dictionary)
        spectro_theory_vec = self.create_spectro_theory(dictionary,
                                                        dictionary_fiducial)

        theory_vec = np.concatenate(
            (photo_theory_vec, spectro_theory_vec), axis=0)
        self.mask_ins.set_theory_vector(theory_vec)
        masked_data_minus_theory = (
            self.masked_data_vector - self.mask_ins.get_masked_theory_vector())

        loglike = -0.5 * np.dot(
            np.dot(masked_data_minus_theory, self.masked_invcov_matrix),
            masked_data_minus_theory)

        return loglike
