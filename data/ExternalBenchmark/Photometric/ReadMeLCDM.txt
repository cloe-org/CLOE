*************************************************
*						*
*	Photometric Folder Content		*
*						*
*************************************************


----------------------------------------------------------------

- ComDist.dat - 

1.	z
2.	r(z) in Mpc

Comoving distance for a LCDM model with (Om, h) = (0.32, 0.67)

----------------------------------------------------------------

- Pnl-TB-LCDM.dat -

1.	log k (k in Mpc^{-1})
2.	z
3.	log Pnl (in Mpc^{3})
4.	log Plin (in Mpc^{3})

Matter power spectrum for the IST:F LCDM fiducial obtained using TakaBird
Third and fourth columns are for the nonlinear and linear power spectrum

----------------------------------------------------------------

- niTabEP10-RB00.dat - 

1.	z
2.	n1(z)
3.	n2(z)
....
11.	n10(z)

Photo - z convolved redshift distribution for 10 equipopulated bins over the range (0.01, 2.5)
Bins limits are given in the 1st and 2nd column of the file PhotoZedEP10RB00.dat
ni(z) is non normalized

----------------------------------------------------------------

- WiG.dat - 

1.	z
2.	W1(z)
3.	W2(z)
....
11.	W10(z)

Photometric galaxy clustering kernel function W_{i}^{G}(z) in Mpc^{-1} (including fiducial bias)
We follow the defintion in the IST:L document on overleaf
ni(z) is now normalized

----------------------------------------------------------------

- WiGamma.dat - 

1.	z
2.	W1(z)
3.	W2(z)
....
11.	W10(z)

Shear only kernel function W_{i}^{\gamma}(z) in Mpc^{-1}
We follow the defintion in the IST:L document on overleaf
ni(z) is now normalized

----------------------------------------------------------------

- CijLL-LCDM-Lin-noIA.dat - 

1.	l
2.	C_{1 1}(l)
3.	C_{1 2}(l)
....
11.	C_{1 10}(l)
12.	C_{2 2}(l)
13.	C_{2 3}(l)
....
20.	C_{2 10}(l)
21.	C_{3 3}(l)
....
56.	C_{10 10}(l)

WL C_{ij}(l) with i from 1 to 10 and j from i to 10 (1 + 55 = 56 columns in total)
Note that C_{ji}(l) = C_{ij}(l)

----------------------------------------------------------------

- CijGG-LCDM-Lin-noIA.dat - 

1.	l
2.	C_{1 1}(l)
3.	C_{1 2}(l)
....
11.	C_{1 10}(l)
12.	C_{2 2}(l)
13.	C_{2 3}(l)
....
20.	C_{2 10}(l)
21.	C_{3 3}(l)
....
56.	C_{10 10}(l)

GCph C_{ij}(l) with i from 1 to 10 and j from i to 10 (1 + 55 = 56 columns in total)
Note that C_{ji}(l) = C_{ij}(l)