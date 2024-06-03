Input Cosmology: <br>
$`\Omega_m`$ = 0.3158 <br>
h       = 0.6732 <br>
$`\Omega_b`$ = 0.0494 <br>
$`n_s`$     = 0.9661 <br>
$`\sigma_8`$ = 0.8102


Eq 19: <br>
$`A_\lambda`$ = 52.0 <br>
$`B_\lambda`$ = 0.9 <br>
$`C_\lambda`$ = 0.5


Eq 20: <br>
$`A_\sigma`$ = 0.2 <br>
$`B_\sigma`$ = -0.05 <br>
$`C_\sigma`$ = 0.001

**Truth tables for number counts and covariance:**
- NC_CG_CC_model.dat
- cov_CG_CC_model.dat

**Mock cluster catalogs:**

Mean observed richness in the bin Eq. 24 <br>
$`\langle \lambda^{\rm ob} \rangle `$= array([23.8714427 , 35.64512875, 51.13983791, 79.60670823]

Miscentering parameters Eq. 58: <br>
$`\sigma_{\rm off} = 0.42`$ <br>
$`f_{\rm off} = 0.25`$

area: <br>
$`\theta = \pi`$/3 <br>
$`\Omega_{\rm sky}`$ = 10313 deg$`^2`$


Observed Redshift bins <br>
`z_bins = np.arange(0.2, 1.61, 0.2)` = [0.2 0.4 0.6 0.8 1.  1.2 1.4 1.6]

Observed Richness bins <br>
lob_bins = [20,30,45,60,500]

To define meaningful angular bins where to compute the tangential or reduced shear (Eq. 35) I assume Omega_m=0.3 and compute $`\theta`$s corresponding to a comoving radius between 200 kpc/h and 30 Mpc/h (file: theta_bins_4gammatob_data_lam_lg20_02_z_16.npy)

```
z_bin_cent=zob_bin[:-1]+0.5*np.diff(zob_bin)
r_grid=10.**np.linspace(np.log10(0.2),np.log10(30.),21) # cMpc/h
theta_bin=np.zeros((num_z_bin,r_grid.size))
num_theta_bin=r_grid.size-1
theta_bin_cent=np.zeros((num_z_bin,num_theta_bin))
for iz,zcl in enumerate( z_bin_cent):
    D_a=d_a(zcl,0.3) # Angular Diameter Distance
    thetas=r_grid/(1.+zcl)/D_a
    theta_bin[iz,:]=thetas
    theta_bin_cent[iz,:]=thetas[:-1]+0.5*np.diff(thetas)
```
Shear profiles are computed assuming:
- Truncated NFW (BMO), with trunctaion factor $`f_t=3`$
- Halo bias from Tinker+10
- Summing the 1 halo and 2 halo term (Eq. 56) 
- $` m_{\rm shear} = 0 `$
- Shape Noise Reduced Shear: $`\sigma_{\rm shape, \gamma_t}^2 = \frac{\sigma_e^2} {n_g \pi (\theta_2^2-\theta_1^2)}`$ with $`\sigma_e=0.3`$ and $`n_g(z)`$ from Table 1 Giocoli+23 

Real space 2pt. cross/auto-correlation functions:
- Comovin distance bin edges: `rs_bins =np.logspace(np.log10(30.),np.log10(130.),25)# [cMpc/h]`
- Observed richness bin edges: `xi_lambdaob_bin=np.array([20., 30., 500. ])`
- Observed redshift bin edges `xi_zob_bin=np.arange(0.2,1.61,0.7)`
- Computed using LS estimator with $`{\tt Corrfunc}`$; comoving distances are derived using the observed redshift `Z_OB` (=`Z_TR` + photo-z noise ), and the fiducial LCDM model (see above)

Selection function tables:
- Computed assuming the model of Eq. 20 and 21 integrating $`P(\lambda^{\rm ob}|\lambda^{\rm tr},z^{\rm tr})`$ over the relevant richness bins $`\Delta\lambda^{\rm ob}_i`$: 
$`I({\tt i},\lambda^{\rm tr},z^{\rm tr})=\int_{\Delta\lambda^{\rm ob}_i} P(\lambda^{\rm ob}|\lambda^{\rm tr},z^{\rm tr})`$

- Grid of $`\lambda^{\rm tr},z^{\rm tr}`$ used to computed the tables:

`ltr_grid=10.**(np.linspace(np.log10(5.),np.log10(100.),100))` <br>
`ztr_grid=np.linspace(0.1,1.8,18)` <br>

**Data Vectors:**

Note: In the directory `catalogs` can be found other realizations of the same catalogs (file_name_cat0X.npy) produced from halo catalogs having the same underlying cosmology.

| Data   | File   |  Shape |
| ------ | ------ |------- |
| Number Counts       |  NC_data_lam_lg20_02_z_16.npy      | (num_z_bin X num_lob_bin)|
| Tangential Shear       |  gammatobob_data_lam_lg20_02_z_16.npy      | (num_z_bin X num_lob_bin X num_theta_bin)|
| Reduced Shear       |  g_tob_data_lam_lg20_02_z_16.npy      | (num_z_bin X num_lob_bin X num_theta_bin)|
| Theta Bin Edges       |  theta_bins_4gammatob_data_lam_lg20_02_z_16.npy      |(num_z_bin X num_theta_bin + 1)|
| Reduced Shear Covariance Matrix       |  cov_g_t_lg20_02_z_16.npy      |(560X560) 560=num_z_bin X num_lob_bin X num_theta_bin|
| Shape Noise Reduced Shear Covariance Matrix       |  cov_g_t_shape_noise_lg20_02_z_16.npy      |(560X560) 560=num_z_bin X num_lob_bin X num_theta_bin|
| Selection function tables       |  int_Plob_ltr_z_Dlob.npy      |(num_lob_bin X 100 X 18) |
| 2pt auto/cross-correlation functions       |  xi_r_lob_zob.npy      |(2 X 2 X 2 X 2 X 24) corresponding to `[z_bin_1,l_bin_1,z_bin_2,l_bin_2,r_bin]` |

**Catalog:**

Cluster catalog for objects having observed richness>20 and 0.2<z_ob<1.6 <br>
File: catalog_lob_Dsigma_gamma_lg20_02_z_16.fit


| Column | Description | Note |
| ------ | ------ | -------- |
|    catalog_id    |    halo ID    | 
|   M_VIR     |   virial mass w.r.t. the critical density     |
|   RA     |   RA     | neglecting Miscentering
|   DEC     |   DEC     | neglecting Miscentering
|   RA_OB     |  observed RA     | including Miscentering
|   DEC_OB     | observed DEC     | including Miscentering
|   R_OFF     | radial offset     | physical Mpc/h
|    LAMBDA_TR    |    true richness    | 
|    LAMBDA_OB    |    observed richness    | 
|    Z_TR    |    true redshift    | 
|    Z_OB    |    observed redshift    | 
|    C    |    concentration    | Derived from M_VIR assuming Duffy+2008: $`c=7.85(M/2e12)^{-0.081}(1+z)^{-0.71}`$ |
|    D_SIGMA    |    Delta Sigma    |  [ Msun h /pMpc^2] |
|    GAMMA_T    |    Tangential Shear    | 
|    GAMMA_T_OB    |    Observed Tangential Shear    | = GAMMA_T + NOISE due to c-M scatter, triaxiality and projected structures
|    G_T    |    Reduced Shear    | $`g_t = \gamma_t / (1-\langle \Sigma_{\rm cr}^{-2} \rangle /\langle \Sigma_{\rm cr}^{-1} \rangle \Sigma) `$
|    G_T_OB    |    Observed Reduced Shear    | = G_T + NOISE due to c-M scatter, triaxiality and projected structures
|    G_T_SHAPE_NS    |    Shape Noise on $`g_t`$    | Noise due to the intrinsic scatter of source ellipticity:  $`\sigma_{\rm shape, \gamma_t}^2=\frac{\sigma_e^2} {n_g \pi (\theta_2^2-\theta_1^2)}`$
