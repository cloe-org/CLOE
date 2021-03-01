# -*- coding: utf-8 -*-
"""Euclike

Contains class to compute the Euclid likelihood
"""

import numpy as np
from likelihood.cosmo.cosmology import Cosmology
from likelihood.photometric_survey.photo import Photo
from likelihood.spectroscopic_survey.spec import Spec
from likelihood.data_reader import reader


class EuclikeError(Exception):
    r"""
    Class to define Exception Error
    """

    pass


class Euclike:
    """
    Class to compute the Euclid likelihood from the theory, data, covariance.
    """

    def __init__(self):
        """Initialize

        Constructor of the class Euclike. The data and covariance are
        read and arranged into their final format only once here.
        """
        self.data_ins = reader.Reader()
        self.data_ins.compute_nz()
        # Read spec
        self.data_ins.read_GC_spec()
        self.zkeys = self.data_ins.data_dict['GC-Spec'].keys()
        self.data_spec_fiducial_cosmo = \
            self.data_ins.data_spec_fiducial_cosmo
        # Transforming data
        self.specdatafinal = self.create_spec_data()
        self.speccovfinal = self.create_spec_cov()
        self.specinvcovfinal = np.linalg.inv(self.speccovfinal)
        # Read photo
        self.data_ins.read_phot()
        self.photoinvcovfinal_GC = np.linalg.inv(
            self.data_ins.data_dict['GC-Phot']['cov'])
        self.photoinvcovfinal_WL = np.linalg.inv(
            self.data_ins.data_dict['WL']['cov'])
        self.photoinvcovfinal_XC = np.linalg.inv(
            self.data_ins.data_dict['XC-Phot']['cov_XC_only'])
        # GCH: order of this matrix is WL, XC, GC
        self.photoinvcovfinal_all = np.linalg.inv(
            self.data_ins.data_dict['XC-Phot']['cov'])
        # Tranforming data
        self.photodatafinal = self.create_photo_data()
        # Calculate permutations i,j bins for WL, GC-Phot, XC.
        # This refers to the non-redundant bin combinations for
        # which we have measurements (i.e. 1-1, 1-2, ..., 1-10,
        # 2-2, 2-3, ..., 2-10, 3-3, 3-4, etc, in the case of ten
        # tomographic bins for WL and GC-phot. Meanhile, all bin
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
        # GCH: transform GC-Phot
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

        Parameters
        ----------
        photo_ins: object
            initialized instance of the class Photo
        full_photo: boolean
            selects whether to use full photometric
            data (with XC) or not

        Returns
        -------
        theoryvec_dict: dict
            returns a dictionary of arrays with the transformed photo theory
            vector
        """
        theoryvec_dict = {'GC-Phot': None, 'WL': None, 'XC-Phot': None,
                          'all': None}
        # GCH: compute theory for GC-Phot
        theoryvec_dict['GC-Phot'] = np.array([phot_ins.Cl_GC_phot(ell,
                                                                  element[0],
                                                                  element[1])
                                              for ell in
                                              self.data_ins.data_dict[
                                              'GC-Phot']
                                              ['ells']
                                              for
                                              element in
                                              self.indices_diagonal_gcphot])

        # GCH: getting theory WL
        theoryvec_dict['WL'] = np.array([phot_ins.Cl_WL(ell,
                                                        element[0],
                                                        element[1])
                                         for ell in
                                         self.data_ins.data_dict['WL']
                                         ['ells']
                                         for
                                         element in
                                         self.indices_diagonal_wl])
        # GCH: getting theory XC-Phot
        if full_photo:
            theoryvec_dict['XC-Phot'] = np.array([phot_ins.Cl_cross(ell,
                                                                    element[0],
                                                                    element[1])
                                                  for
                                                  element in
                                                  self.indices_all
                                                  for ell in
                                                  self.data_ins.data_dict[
                                                  'XC-Phot']
                                                  ['ells']])
            theoryvec_dict['all'] = np.concatenate(
                (theoryvec_dict['WL'],
                 theoryvec_dict['XC-Phot'],
                 theoryvec_dict['GC-Phot']),
                axis=0)

        return theoryvec_dict

    def create_spec_theory(self, dictionary, dictionary_fiducial):
        """Create Spec Theory

        Obtains the theory for the likelihood.

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
            returns the theory array with same indexing/format as the data
        """

        spec_ins = Spec(dictionary, dictionary_fiducial)
        theoryvec = []
        # (SJ): k_ins seemingly in h/Mpc units, tentative transformation here.
        # (SJ): Commented line below is pre-transformation, kept for now
        for z_ins in self.zkeys:
            for m_ins in [0, 2, 4]:
                for k_ins in self.data_ins.data_dict['GC-Spec'][z_ins]['k_pk']:
                    theoryvec = np.append(
                                    theoryvec, spec_ins.multipole_spectra(
                                        float(z_ins), k_ins, m_ins))

        return theoryvec

    def create_spec_data(self):
        """Create Spec Data

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
                              'GC-Spec'][z_ins]['pk' + str(m_ins)])

        return datavec

    def create_spec_cov(self):
        """Create Spec Cov

        Arranges the covariance for the likelihood into its final format

        Returns
        -------
        covfull: float N x N matrix
            returns a single covariance from sub-covariances (split in z)
        """

        self.covnumz = len(self.zkeys)
        # (SJ): covnumk generalizes so that each z can have different k binning
        self.covnumk = []
        self.covnumk.append(0)
        for z_ins in self.zkeys:
            self.covnumk.append(
                3 * len(self.data_ins.data_dict['GC-Spec'][z_ins]['k_pk']))

        # (SJ): Put all covariances into a single/larger covariance.
        # (SJ): As no cross-covariances, this takes on a block-form
        # (SJ): along the diagonal.
        covfull = np.zeros([sum(self.covnumk), sum(self.covnumk)])
        kc = 0
        c1 = 0
        c2 = 0
        for z_ins in self.zkeys:
            c1 = c1 + self.covnumk[kc]
            c2 = c2 + self.covnumk[kc + 1]
            covfull[c1:c2, c1:c2] = self.data_ins.data_dict['GC-Spec'][
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

        # (GCH): photo-class instance
        phot_ins = Photo(
                dictionary,
                self.data_ins.nz_dict_WL,
                self.data_ins.nz_dict_GC_Phot)
        # (GCH): theory vec cal
        theoryvec_dict = self.create_photo_theory(phot_ins, full_photo)
        loglike_photo = 0
        if not full_photo:
            # (GCH): construct dmt
            dmt_GC = self.photodatafinal['GC-Phot'] - \
                    theoryvec_dict['GC-Phot']
            dmt_WL = self.photodatafinal['WL'] - \
                theoryvec_dict['WL']
            # (GCH): cal loglike
            loglike_GC = -0.5 * \
                np.dot(np.dot(dmt_GC, self.photoinvcovfinal_GC), dmt_GC.T)
            loglike_WL = -0.5 * \
                np.dot(np.dot(dmt_WL, self.photoinvcovfinal_WL), dmt_WL.T)
            # (GCH): save loglike
            loglike_photo = loglike_GC + loglike_WL
        # If True, calls massive cov mat
        elif full_photo:
            # (GCH): construct dmt
            dmt_all = self.photodatafinal['all'] - \
                    theoryvec_dict['all']

            # (GCH): cal loglike
            loglike_photo = -0.5 * np.dot(
                np.dot(dmt_all, self.photoinvcovfinal_all),
                dmt_all.T)
        else:
            print('ATTENTION: full_photo has to be either True/False')
        return loglike_photo

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
            loglike = -2 ln(likelihood) for the Euclid observables
        """
        like_selection = dictionary['nuisance_parameters']['like_selection']
        full_photo = dictionary['nuisance_parameters']['full_photo']
        if like_selection == 1:
            self.loglike_tot = self.loglike_photo(dictionary, full_photo)
        elif like_selection == 2:
            self.thvec = self.create_spec_theory(
                             dictionary, dictionary_fiducial)
            dmt = self.specdatafinal - self.thvec
            self.loglike_tot = -0.5 * np.dot(
                np.dot(dmt, self.specinvcovfinal), dmt.T)
        elif like_selection == 12:
            self.thvec = self.create_spec_theory(
                             dictionary, dictionary_fiducial)
            dmt = self.specdatafinal - self.thvec
            self.loglike_spec = -0.5 * np.dot(np.dot(
                                    dmt, self.specinvcovfinal), dmt.T)
            self.loglike_photo = self.loglike_photo(dictionary, full_photo)
            # (SJ): only addition below if no cross-covariance
            self.loglike_tot = self.loglike_photo + self.loglike_spec
        else:
            raise CobayaInterfaceError(
                r"Choose like selection '1' or '2' or '12'")

        return self.loglike_tot
