# -*- coding: utf-8 -*-
r"""MASKING VECTOR WRAPPER MODULE

Module for giving a physical interpretation to the content of the masking
vector
"""

import numpy


class MaskingVectorWrapper:
    r"""Gives an interpretation of the content of the masking vector

    This class interprets the content of the masking vector to tell whether a
    given probe is enabled.
    It assumes that the masking vector is obtained as the concatenation of
    the following probes (from lower to higher array indices)
    - WL: Weak Lensing (1100 elements, from 0 to 1100-1)
    - XC-Phot: Photometric (Weak Lensing) x (Galaxy Clustering) cross
      correlation (2000 elements, from 1100 to 3100-1)
    - GC-Phot: Photometric Galaxy Clustering (1100 elements, from 3100 to
      4200-1)
    - GC-Spectro: Spectroscopic Galaxy Clustering (1500 elements, from 4200
      to 5700-1)

    A given probe is enabled if any of the corresponding elements in the
    masking vector evaluates to True.
    """

    def __init__(self):
        r"""Constructor.

        All the data members are initialized. Those depending on the user
        input are initialized to None. Those defining the expected structure
        of the masking vector are initialized to the expected values, and are
        not supposed to change throughout the code.
        """
        self._masking_vector = None
        self._reset_enable_flags()

        # set the number of elements of the masking vector corresponding to
        # each probe
        wl_num_elements = 1100
        xc_phot_num_elements = 2000
        gc_phot_num_elements = 1100
        gc_spectro_num_elements = 1500

        # derive the first vector element of each probe starting from the
        # number of elements of the probes
        self._wl_first_element = 0
        self._xc_phot_first_element = (
            self._wl_first_element + wl_num_elements)
        self._gc_phot_first_element = (
            self._xc_phot_first_element + xc_phot_num_elements)
        self._gc_spectro_first_element = (
            self._gc_phot_first_element + gc_phot_num_elements)

        # evaluate the expected size of the masking vector, used in
        # _check_masking_vector() to check the consistency of the user input
        self._expected_vector_size = (
            wl_num_elements +
            xc_phot_num_elements +
            gc_phot_num_elements +
            gc_spectro_num_elements
        )

    def set_masking_vector(self, vector):
        r"""Set the masking vector

        Parameters
        ----------
        vector: ndarray of bool or int
          The masking vector used to identify the enabled probes.
        """
        self._check_masking_vector(vector)
        self._masking_vector = vector.astype(bool)
        self._reset_enable_flags()

    def get_wl_enabled(self):
        r"""Get whether or not the WL probe is enabled

        Returns
        -------
        bool
          True if the WL probe is enabled, False otherwise
        """
        if self._wl_enabled is None:
            self._wl_enabled = self._check_slice_enabled(
                slice_start=self._wl_first_element,
                slice_stop=self._xc_phot_first_element
            )
        return self._wl_enabled

    def get_xc_phot_enabled(self):
        r"""Get whether or not the XC-Phot probe is enabled

        Returns
        -------
        bool
          True if the XC-Phot probe is enabled, False otherwise
        """
        if self._xc_phot_enabled is None:
            self._xc_phot_enabled = self._check_slice_enabled(
                slice_start=self._xc_phot_first_element,
                slice_stop=self._gc_phot_first_element
            )
        return self._xc_phot_enabled

    def get_gc_phot_enabled(self):
        r"""Get whether or not the GC-Phot probe is enabled

        Returns
        -------
        bool
          True if the GC-Phot probe is enabled, False otherwise
        """
        if self._gc_phot_enabled is None:
            self._gc_phot_enabled = self._check_slice_enabled(
                slice_start=self._gc_phot_first_element,
                slice_stop=self._gc_spectro_first_element
            )
        return self._gc_phot_enabled

    def get_gc_spectro_enabled(self):
        r"""Get whether or not the GC-Spectro probe is enabled

        Returns
        -------
        bool
          True if the GC-Spectro probe is enabled, False otherwise
        """
        if self._gc_spectro_enabled is None:
            self._gc_spectro_enabled = self._check_slice_enabled(
                slice_start=self._gc_spectro_first_element,
                slice_stop=len(self._masking_vector)
            )
        return self._gc_spectro_enabled

    def _check_slice_enabled(self, slice_start, slice_stop):
        r"""Check whether a slice of the masking vector is enabled

        A slice is identified by its start (included) and stop (not included)
        elements, and it is enabled if any of the elements in the slice
        evaluates to True

        Parameters
        ----------
        slice_start: int
          The first element of the slice to be tested.
        slice_stop: int
          The stop element (not included) of the slice to be tested.

        Returns
        -------
        bool:
          True if any of the elements in the slice evaluates to True, False
          otherwise.

        Raises
        ------
        RuntimeError
          if the masking vector is not set
        """
        if self._masking_vector is None:
            raise RuntimeError(f'the masking vector is not set')
        return numpy.any(self._masking_vector[slice_start:slice_stop:1])

    def _reset_enable_flags(self):
        r"""Reset probe enable flags

        Set to None all the boolean flags used to tell whether a given probe
        is enabled.
        """
        self._wl_enabled = None
        self._xc_phot_enabled = None
        self._gc_phot_enabled = None
        self._gc_spectro_enabled = None

    def _check_masking_vector(self, vector):
        r"""Check the masking vector for consistency

        As a first approach we only check that the masking vector has the
        expected size, however other consistency checks might be implemented
        later.

        Parameters
        ----------
        vector: numpy.ndarray of int or bool

        Raises
        ------
        ValueError
          in case vector has an unexpected size
        """
        if (len(vector) != self._expected_vector_size):
            raise ValueError(f'The input masking vector has unexpected size:'
                             f' {len(vector)} instead of'
                             f' {self._expected_vector_size}')
