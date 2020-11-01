# -*- coding: utf-8 -*-
"""Euclike

Contains class to compute the Euclid likelihood
"""

import numpy as np
from ..cosmo.cosmology import Cosmology
from likelihood.photometric_survey.photo import Photo
from likelihood.spectroscopic_survey.spec import Spec
from ..data_reader import reader


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
        """
        Constructor of the class Euclike. The data and covariance are
        read and arranged into their final format only once here.
        """
        self.data_ins = reader.Reader()
        self.data_ins.read_GC_spec()
        self.zkeys = self.data_ins.data_dict['GC-Spec'].keys()
        self.specdatafinal = self.create_spec_data()
        self.speccovfinal = self.create_spec_cov()
        self.speccovinvfinal = np.linalg.inv(self.speccovfinal)

    def create_spec_theory(self, dictionary, dictionary_fiducial):
        """
        Obtains the theory for the likelihood.

        Parameters
        -------
        dictionary: dictionary
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        dictionary_fiducial: dictionary
            cosmology dictionary from the Cosmology class
            at the fiducial cosmology

        Returns
        -------
        theoryvec: float array
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
                                        float(z_ins), k_ins *
                                        dictionary['H0'] / 100.0, m_ins))
        #                                 float(z_ins), k_ins, m_ins))

        return theoryvec

    def create_spec_data(self):
        """
        Arranges the data vector for the likelihood into its final format

        Returns
        -------
        datavec: float array
            returns the data as a single array across z, mu, k
        """

        datavec = []
        for z_ins in self.zkeys:
            for m_ins in [0, 2, 4]:
                datavec = np.append(datavec, self.data_ins.data_dict[
                              'GC-Spec'][z_ins]['pk' + str(m_ins)])

        return datavec

    def create_spec_cov(self):
        """
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

    def loglike(self, dictionary, dictionary_fiducial):
        """
        Calculates the log-likelihood for a given model

        Parameters
        ----------
        data_params: tuple
            List of (sampled) parameters needed by the likelihood.
            This includes nuisance parameters and settings keys.

        dictionary: dictionary
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        dictionary_fiducial: dictionary
            cosmology dictionary from the Cosmology class
            at the fiducial cosmology

        Returns
        ----------
        loglike: float
            loglike = -2 ln(likelihood) for the Euclid observables
        """
        # (SJ): We can either multiply data+cov or theory with (2pi/h)^3 factor
        # (SJ): Prefer theory as is, but more efficient to multiply it
        # datfac = (2.0 * np.pi / (dictionary['H0'] / 100.0))**3.0
        # covfac = datfac**2
        # (SJ): Not using thfac with 1/(2pi)^3 as will be removed from OU data
        # thfac = 1.0 / (2.0 * np.pi / (dictionary['H0'] / 100.0))**3.0
        thfac = (dictionary['H0'] / 100.0)**3.0
        like_selection = dictionary['nuisance_parameters']['like_selection']
        if like_selection == 1:
            # (SJ): for now, photo lines below just for fun
            phot_ins = Photo(dictionary)
            ell_ins = 100
            bin_i_ins = 1
            bin_j_ins = 1
            observable = phot_ins.Cl_WL(ell_ins, bin_i_ins, bin_j_ins)
            self.loglike = 0.0
        elif like_selection == 2:
            self.thvec = self.create_spec_theory(
                             dictionary, dictionary_fiducial)
            dmt = self.specdatafinal - self.thvec * thfac
            self.loglike = np.dot(np.dot(dmt, self.speccovinvfinal), dmt.T)
        elif like_selection == 12:
            self.thvec = self.create_spec_theory(
                             dictionary, dictionary_fiducial)
            dmt = self.specdatafinal - self.thvec * thfac
            self.loglike_spec = np.dot(np.dot(
                                    dmt, self.speccovinvfinal), dmt.T)
            self.loglike_photo = 0.0
            # (SJ): only addition below if no cross-covariance
            self.loglike = self.loglike_photo + self.loglike_spec
        else:
            raise CobayaInterfaceError(
                r"Choose like selection '1' or '2' or '12'")

        return self.loglike
