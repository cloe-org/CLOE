""" theoryvector_transforms

Contains function to transform the theory (and data) vectors regarding different matrix operations
"""
from pathlib import Path
from copy import deepcopy
import numpy as np

class BNT_transform():

    def __init__(self, z, r_z, n_i_z):

        self.z = z
        self.chi = r_z
        self.n_i_list = n_i_z
        self.nbins = len(n_i_z)


    def get_matrix(self):

        A_list = []
        B_list = []
        for i in range(self.nbins):
            nz = self.n_i_list[i]
            A_list += [np.trapz(nz, self.z)]
            B_list += [np.trapz(nz / self.chi, self.z)]


        BNT_matrix = np.eye(self.nbins)
        BNT_matrix[1,0] = -1.

        for i in range(2,self.nbins):
            mat = np.array([ [A_list[i-1], A_list[i-2]], [B_list[i-1], B_list[i-2]] ])
            A = -1. * np.array( [A_list[i], B_list[i]] )
            soln = np.dot(np.linalg.inv(mat), A)
            BNT_matrix[i,i-1] = soln[0]
            BNT_matrix[i,i-2] = soln[1]
        
        return BNT_matrix


class Theory_vector_transform():

    def __init__(self, cosmo_dic, nz_dict, transform_type='Unity'):
        
        if transform_type=='BNT':
            zwin = cosmo_dic['z_win']
            chiwin = cosmo_dic['r_z_func'](zwin)
            n_bins_list = list(nz_dict.keys())
            n_bins = len(n_bins_list)
            ni_list = np.zeros((n_bins, len(zwin)))
            for ii, ni in enumerate(n_bins_list):
                ni_list[ii] = nz_dict[ni](zwin)    
            bnt_trf = BNT_transform(zwin, chiwin, ni_list)
            self.bnt_mat = bnt_trf.get_matrix()
            
