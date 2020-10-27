from fitsio import FITS
import os
import gc

def gather_cls(path):

    # Read in the FITS PCl file
    fits = FITS(path)

    # Initialize the dictionary with multipole values
    dic = {'ELLS':fits[1][:]}

    # Loop through the hdus and create dictionary elements for each hdu
    for i in range(2,len(fits)):

        hdr = fits[i].read_header()
        fields = hdr['EXTNAME'].strip().split('-')
        bins = hdr['BIN_COMB'].strip().split('-')
        naxis = hdr['NAXIS']

        if naxis==1:
            dic['%s%s-%s%s'%(fields[0], bins[0], fields[1], bins[1])] = fits[i][:]
        elif naxis==2:
            dic['%s%s-%s%s'%(fields[0], bins[0], fields[1], bins[1])] = fits[i][:,:]

    # trying to clean
    fits.close()
    del fields
    del bins
    gc.collect()

    return dic


def gather_mixmats(path):

    # Read in the FITS mixing matrix file
    fits = FITS(path)

    dic = {}

    # Loop through the hdus and create dictionary elements for each hdu
    for i in range(1,len(fits)):

        hdr = fits[i].read_header()
        extname = hdr['EXTNAME'].strip()
        if extname[1]=='Z':
            field1 = 'ZERO'
        if extname[1]=='P':
            field1 = 'PLUS'
        if extname[1]=='M':
            field1 = 'MINUS'
        if extname[len(extname)-3]=='E':
            field2 = 'ZERO'
        if extname[len(extname)-3]=='L':
            field2 = 'PLUS'
        if extname[len(extname)-3]=='N':
            field2 = 'MINUS'

        bins = hdr['BIN_COMB'].strip().split('-')

        dic['%s%s-%s%s'%(field1, bins[0], field2, bins[1])] = fits[i][:,:]

    fits.close()
    return dic

                        
def read_cls(paths):
    """
    Reads the outputs from PK-WL's pseudo power spectra estimator.

    Call signatures:

        read_cls(["paths/to/pkwl_outputs/"])

    Parameters
    ----------
    paths:  string
            List of paths to the folder where PK-WL outputs are found. This list should can contain any combination of Pos-Pos, Pos-Shear and Shear-Shear estimates, but only one
            set of estimates for a given observable and Nls should be recovered separately PCls.


    Returns
    -------
    A dictionary with 'ELLS' being the mean multipoles for each multipole bin and power spectra labeled as 'Ki-Mj' where K and M stand
        for the probe P, E, B (for [P]osition, [E]-mode and [B]-mode), and i,j stand for the redshift tomographic bin.
        Ex: Cl['E1-B2'] is the cross of E-Modes from bin 1 with B-Modes from bin 2.
        Ex: Cl['P1-P1'] is the auto power spectrum of galaxy clustering
        Each set of Cls has shape [ClSamp, NEllBin] or [NlSamp, NellBin] for Nls.

    Examples
    --------
    >>> import pkwl_tools
    >>> import pylab as plt
    >>> Cls = pkwl_tools.create_table.read_cls(['path/to/PCl_ShearShear')
    # Plots E-Mode autopower spectrum for bin 1:
    >>> plt.plot(Cls['ELLS'], Cls['E1-E1'])
    >>> plt.show()
    
    """
    cls = {}

    for ipath in paths:
        if os.path.isfile(ipath):
            cls.update(gather_cls(ipath))
        else:
            print("File %s does not exist" %(ipath))

    return cls


def read_mixmats(paths):
    """
    Reads the outputs from PK-WL's mixing matrix.

    Call signatures:

        read_mixmats(["paths/to/pkwl_outputs/"])

    Parameters
    ----------
    paths:  string
            List of paths to the folder where PK-WL outputs are found. This list should can contain any combination of Pos-Pos, Pos-Shear and Shear-Shear estimates, but only on
            set of estimates for a given observable

    Returns
    -------
    A dictionary with the different mixing matrices for each of the readshift tomographic bins labeld as:
        Ki-Mj where K and M are the types ZERO, PLUS or MINUS and i,j stand for the redshift bin
        Ex: MM['ZERO1-PLUS1'] is the mixing matrix constructed with M^0_ll' and M^+_ll'

    Notes
    -------
    For more details on the Mixing Matrices, please check Brown, Castro, Taylor (2005) 
        https://arxiv.org/abs/astro-ph/0410394
    ZERO is usually associated with spin-0 fields such as galaxy clustering
    MINUS/PLUS are associated with spin-2 fields such as cosmic shear

    !MIXING MATRIX ARE PROVIDED FOR EACH MULTIPOLE FROM ELL = 0 to ELL = Lmax_Mm!

    Examples
    --------
    >>> import pkwl_tools
    >>> import pylab as plt
    >>> MM = pkwl_tools.create_table.read_mixmats(['path/to/MixingMatrix_PosPos'])
    # Plots M^++ for bin 1:
    >>> plt.imshow(MM['PLUS1-PLUS1'])
    >>> plt.show()
    
    """

    mm = {}

    for ipath in paths:
        if os.path.isfile(os.path.join(ipath)):
            mm.update(gather_mixmats(ipath))
        else:
            print("File %s does not exist" %(ipath))

    return mm

