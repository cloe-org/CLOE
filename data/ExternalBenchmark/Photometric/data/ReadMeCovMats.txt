*************************************************
*						*
*	Covariance Matrices Info		*
*						*
*************************************************


----------------------------------------------------------------

- Uploaded files - 

1. CovMat-PosPos-Gauss-20bins.dat: position - position (GCph) covariance matrix with Gaussian contribution only
2. CovMat-ShearShear-Gauss-20bins.dat: shear - shear (WL) covariance matrix with Gaussian contribution only
3. CovMat-3x2pt-Gauss-20bins.dat: 3x2pt (WL + XC + GCph) covariance matrix with Gaussian contribution only

4. CovMat-PosPos-GaussSSC-20bins.dat: same as 1 but including super sample covariance contribution
5. CovMat-ShearShear-GaussSSC-20bins.dat: same as 2 but including super sample covariance contribution
6. CovMat-3x2pt-GaussSSC-20bins.dat: same as 3 but including super sample covariance contribution

----------------------------------------------------------------

- Instructions on the model specifics -

- Same fiducial model as in the IST:F paper
- Linear matter power spectrum + eNLA model for intrinsic alignment
- 10 equipopulated redshift bins over the range (0.001, 2.5) with 30 gal/arcmin^2

----------------------------------------------------------------

- Instructions on corresponding data vectors -

- WL (size of the matrix = 1100 x 1100): 

shear - shear C_{ij}(ell) with i from 1 to 10 and j from i to 10 for each given ell

- GCph (size of the matrix = 1100 x 1100): 

position - position C_{ij}(ell) with i from 1 to 10 and j from i to 10 for each given ell

- WL + XC + GCph (size of the matrix = 4200 x 4200): 
 

shear - shear C_{ij}(ell) with i from 1 to 10 and j from i to 10 +
shear - position C_{ij}(ell) with i from 1 to 10 and j from 1 to 10 +
position - position C_{ij}(ell) with i from 1 to 10 and j from i to 10

for the first ell values, then for the second, and so on 

- ell values: 

20 logarithmically equispaced bins from 10 to 5000 with C_{ij} evaluated at the bin centre as from the Lee code
