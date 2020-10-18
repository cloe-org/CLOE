# -*- coding: utf-8 -*-
"""Euclike

Contains class to compute Euclid likelihood
"""

import numpy as np
from ..cosmo.cosmology import Cosmology
from likelihood.photometric_survey.shear import Shear
from likelihood.spectroscopic_survey.spec import Spec
from ..data_reader import reader
import sys


class EuclikeError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Euclike:
    """
    Class to compute the Euclid likelihood from the theory, data, covariance.
    """

    def __init__(self, dictionary, dictionary_fiducial):
        """
        Constructor of the class Euclike. The data and covariance are
        read and arranged into their final format only once here.

        Parameters
        ----------
        dictionary: dictionary
            cosmology dictionary from Cosmology class
            which gets updated at each step of the
            sampling method

        dictionary_fiducial: dictionary
            cosmology dictionary from Cosmology class
            which includes the fiducial cosmology
        """
        self.theory = dictionary
        self.fiducial = dictionary_fiducial
        self.data_ins = reader.Reader()
        self.data_ins.read_GC_spec()
        self.zkeys = self.data_ins.data_dict['GC-Spec'].keys()
        self.specdatafinal = self.create_spec_data()
        self.speccovfinal = self.create_spec_cov()

    def create_spec_theory(self):
        """
        Obtains the theory for the likelihood

        Returns
        -------
        theoryvec: float array
            returns the theory array with same indexing/format as the data
        """

        spec_ins = Spec(self.theory, self.fiducial)
        self.theoryvec = []
        for z_ins in self.zkeys:
            for m_ins in [0, 2, 4]:
                for k_ins in self.data_ins.data_dict['GC-Spec'][z_ins]['k_pk']:
                    self.theoryvec.append(
                        spec_ins.multipole_spectra(float(z_ins), k_ins, m_ins))

        return self.theoryvec

    def create_spec_data(self):
        """
        Arranges the data vector for the likelihood into its final format

        Returns
        -------
        datavec: float array
            returns the data as a single array across z, mu, k
        """

        self.datavec = []
        for z_ins in self.zkeys:
            for m_ins in [0, 2, 4]:
                self.datavec.extend(
                    self.data_ins.data_dict['GC-Spec'][z_ins][
                        'pk' + str(m_ins)])

        return self.datavec

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
        self.covfull = np.zeros([sum(self.covnumk), sum(self.covnumk)])
        kc = 0
        c1 = 0
        c2 = 0
        for z_ins in self.zkeys:
            c1 = c1 + self.covnumk[kc]
            c2 = c2 + self.covnumk[kc + 1]
            self.covfull[c1:c2, c1:c2] = self.data_ins.data_dict['GC-Spec'][
                                             z_ins]['cov']
            kc = kc + 1

        return self.covfull

    def loglike(self, data_params):
        """
        Calculates the log-likelihood for a given model

        Parameters
        ----------
        data_params: tuple
            List of (sampled) parameters obtained from
            the theory code or asked by the likelihood

        Returns
        ----------
        loglike: float
            loglike = -2 ln(likelihood) for the Euclid observables
        """

        self.data_params = data_params
        like_selection = self.data_params['params']['like_selection']
        if like_selection == 1:
            shear_ins = Shear(self.theory)
            # (SJ): for now, shear lines below just for fun
            ell_ins = 100
            bin_i_ins = 1
            bin_j_ins = 1
            observable = shear_ins.Cl_WL(ell_ins, bin_i_ins, bin_j_ins)
            self.loglike = 0.0
        elif like_selection == 2:
            dmt_zip = zip(self.create_spec_data(), self.create_spec_theory())
            dmt = [list1_i - list2_i for (list1_i, list2_i) in dmt_zip]
            covinv = np.linalg.inv(self.speccovfinal)
            self.loglike = dmt @ covinv @ np.transpose(dmt)
        elif like_selection == 12:
            dmt_zip = zip(self.specdatafinal, self.create_spec_theory())
            dmt = [list1_i - list2_i for (list1_i, list2_i) in dmt_zip]
            covinv = np.linalg.inv(self.speccovfinal)
            self.loglike_spec = dmt @ covinv @ np.transpose(dmt)
            self.loglike_shear = 0.0
            # (SJ): only addition below if no cross-covariance
            self.loglike = self.loglike_shear + self.loglike_spec
        else:
            raise CobayaInterfaceError(
                r"Choose like selection 'shear' or 'spec' or 'both'")

        return self.loglike
