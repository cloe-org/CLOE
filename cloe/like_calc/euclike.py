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
from cloe.auxiliary.matrix_transforms import BNT_transform


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
        self.data_ins.compute_luminosity_ratio()
        # Read spectro
        self.data_ins.read_GC_spectro()
        self.zkeys = self.data_ins.data_dict['GC-Spectro'].keys()
        self.data_spectro_fiducial_cosmo = \
            self.data_ins.data_spectro_fiducial_cosmo
        # Read photo
        self.data_ins.read_phot()
        # Read observables
        self.observables = observables
        # Photo class instance
        self.phot_ins = Photo(None,
                              self.data_ins.nz_dict_WL,
                              self.data_ins.nz_dict_GC_Phot)

        # Calculate permutations i,j bins for WL, GC-Phot, XC.
        # This refers to the non-redundant bin combinations for
        # which we have measurements (i.e. 1-1, 1-2, ..., 1-10,
        # 2-2, 2-3, ..., 2-10, 3-3, 3-4, etc, in the case of ten
        # tomographic bins for WL and GC-Phot. Meanwhile, all bin
        # combinations exist for XC, i.e. for example both 1-2
        # and 2-1, both 1-3 and 3-1, etc).
        numtomo_wl = self.data_ins.numtomo_wl
        numtomo_gcphot = self.data_ins.numtomo_gcphot
        x_diagonal_wl = np.array(np.triu_indices(n=numtomo_wl,
                                 m=numtomo_wl)) + 1
        x_diagonal_gc = np.array(np.triu_indices(n=numtomo_gcphot,
                                 m=numtomo_gcphot)) + 1
        self.indices_diagonal_wl = \
            list(zip(x_diagonal_wl[0], x_diagonal_wl[1]))
        self.indices_diagonal_gc = \
            list(zip(x_diagonal_gc[0], x_diagonal_gc[1]))
        x_full_xc = np.indices((numtomo_gcphot, numtomo_wl))
        self.indices_all = tuple(zip(x_full_xc[0].flatten() + 1,
                                 x_full_xc[1].flatten() + 1))
        self.ells_WL = self.data_ins.data_dict['WL']['ells']
        self.ells_XC = self.data_ins.data_dict['XC-Phot']['ells']
        self.ells_GC_phot = self.data_ins.data_dict['GC-Phot']['ells']

        # Reshaping the data vectors and covariance matrices
        # into dictionaries to be passed to the data_handler class

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

        # Spectro class instance
        self.spec_ins = Spectro(None, list(self.zkeys))

    def create_masked_photo_data(self, dictionary):
        """
        Create Masked Photo Data

        Computes the masked photometric data vector
        and masked input covariance matrix

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class
        """

        # Tranforming data
        self.photodata = self.create_photo_data(dictionary)
        datafinal = {**self.photodata,
                     'GC-Spectro': self.spectrodata}
        covfinal = {'3x2pt': self.data_ins.data_dict['3x2pt_cov'],
                    'GC-Spectro': self.spectrocov}
        self.data_handler_ins = Data_handler(datafinal,
                                             covfinal,
                                             self.observables,
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
        return None

    def create_photo_data(self, dictionary):
        """Create Photo Data

        Arranges the photo data vector for the likelihood into its final format

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class

        Returns
        -------
        datavec_dict: dict
            returns a dictionary of arrays with the transformed photo data
        """

        datavec_dict = {'GC-Phot': [], 'WL': [], 'XC-Phot': [], 'all': []}
        for index in list(self.data_ins.data_dict['WL'].keys()):
            if 'B' in index:
                del (self.data_ins.data_dict['WL'][index])
        for index in list(self.data_ins.data_dict['XC-Phot'].keys()):
            if 'B' in index:
                del (self.data_ins.data_dict['XC-Phot'][index])
        # Transform GC-Phot
        # We ignore the first key (ells)
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
                                                    dictionary, obs='WL')
        datavec_dict['XC-Phot'] = \
            self.transform_photo_theory_data_vector(datavec_dict['XC-Phot'],
                                                    dictionary, obs='XC-phot')
        datavec_dict['GC-Phot'] = \
            self.transform_photo_theory_data_vector(datavec_dict['GC-Phot'],
                                                    dictionary, obs='GC-phot')
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
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

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
                 for ell in self.ells_WL
                 for element in self.indices_diagonal_wl]
            )
        else:
            wl_array = np.zeros(
                 len(self.indices_diagonal_wl) *
                 len(self.ells_WL)
            )

        # Obtain the theory for XC-Phot
        if self.data_handler_ins.use_xc_phot:
            xc_phot_array = np.array(
                [self.phot_ins.Cl_cross(ell, element[1], element[0])
                 for ell in self.ells_XC
                 for element in self.indices_all]
            )
        else:
            xc_phot_array = np.zeros(
                 len(self.indices_all) *
                 len(self.ells_XC)
            )

        # Obtain the theory for GC-Phot
        if self.data_handler_ins.use_gc_phot:
            gc_phot_array = np.array(
                [self.phot_ins.Cl_GC_phot(ell, element[0], element[1])
                 for ell in self.ells_GC_phot
                 for element in self.indices_diagonal_gc]
            )
        else:
            gc_phot_array = np.zeros(
                len(self.indices_diagonal_gc) *
                len(self.ells_GC_phot)
            )

        # Apply any matrix transform activated with a switch
        wl_array = \
            self.transform_photo_theory_data_vector(wl_array,
                                                    dictionary, obs='WL')
        xc_phot_array = \
            self.transform_photo_theory_data_vector(xc_phot_array,
                                                    dictionary, obs='XC-phot')
        gc_phot_array = \
            self.transform_photo_theory_data_vector(gc_phot_array,
                                                    dictionary, obs='GC-phot')

        self.photo_theory_vec = np.concatenate(
            (wl_array, xc_phot_array, gc_phot_array), axis=0)

        return self.photo_theory_vec

    def transform_photo_theory_data_vector(self, obs_array,
                                           dictionary, obs='WL'):
        """Transform Photo Theory Data Vector

        Transform the photo theory vector with a generic matrix transformation
        specified in the 'matrix_transform_phot' key
        of the cosmology dictionary

        Parameters
        ----------
        obs_array: array
            Array containing the original (untransformed)
            photo theory/data vector
        dictionary: dict
            cosmology dictionary from the Cosmology class
        obs: string
            String specifying the photo observable which will be transformed.
            Default: 'WL'

        Returns
        -------
        transformed_array: array
            Returns an array with the transformed photo theory/data vector
            If the requested transform is unapplicable, returns the original
            photo theory/data vector
        """
        self.phot_ins.calc_nz_distributions(dictionary)
        self.matrix_transform = dictionary['matrix_transform_phot']
        if self.matrix_transform == 'BNT':
            zwin = dictionary['z_win']
            chiwin = dictionary['r_z_func'](zwin)
            Nz = self.phot_ins.nz_WL.get_num_tomographic_bins()
            ni_list = np.zeros((Nz, len(zwin)))
            for ni in range(Nz):
                ni_list[ni] = \
                    self.phot_ins.nz_WL.interpolates_n_i(ni + 1, zwin)(zwin)
            BNT_transformation = BNT_transform(zwin, chiwin, ni_list)
            if obs == 'WL':
                N_ells = len(self.ells_WL)
                transformed_array = \
                    BNT_transformation\
                    .apply_vectorized_symmetric_BNT(
                                                    Nz,
                                                    N_ells,
                                                    obs_array)
            elif obs == 'XC-phot':
                N_ells = len(self.ells_XC)
                transformed_array = \
                    BNT_transformation\
                    .apply_vectorized_nonsymmetric_BNT(
                                                     Nz,
                                                     N_ells,
                                                     obs_array)
            elif obs == 'GC-phot':
                transformed_array = obs_array
            else:
                print("In method:transform_photo_theory_data_vector,  \
                     observable passed will not be transformed")
                transformed_array = obs_array
        elif self.matrix_transform is False:
            # Not doing any matrix transform
            transformed_array = obs_array
        else:
            raise ValueError("Matrix Transform not implemented yet into CLOE")
        return transformed_array

    def create_spectro_theory(self, dictionary):
        """Create Spectro Theory

        Obtains the theory for the likelihood.
        The theory is evaluated only if the GC-Spectro probe is enabled in the
        masking vector.

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

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
            self.spec_ins.update(dictionary)
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

    def loglike(self, dictionary):
        """Loglike

        Calculates the log-likelihood for a given model

        Parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        Returns
        -------
        loglike_tot: float
            loglike = Ln(likelihood) for the Euclid observables
        """
        self.spectrodata = self.create_spectro_data()
        self.spectrocov = self.create_spectro_cov()
        self.create_masked_photo_data(dictionary)
        photo_theory_vec = self.create_photo_theory(dictionary)
        spectro_theory_vec = self.create_spectro_theory(dictionary)
        theory_vec = np.concatenate(
            (photo_theory_vec, spectro_theory_vec), axis=0)
        self.mask_ins.set_theory_vector(theory_vec)
        masked_data_minus_theory = (
            self.masked_data_vector - self.mask_ins.get_masked_theory_vector())
        loglike = -0.5 * np.dot(
            np.dot(masked_data_minus_theory, self.masked_invcov_matrix),
            masked_data_minus_theory)
        return loglike
