#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "math.h"
#include "vector"
#include "iostream"
#include "stdio.h"
#include "stdexcept"

#define ABSERR 0.0
#define RELERR 1e-4
#define workspace_size 8000
#define KEY 3

namespace py = pybind11;

// »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

typedef struct integrand_params{
  gsl_spline*spline;
  gsl_interp_accel*acc;
  gsl_integration_workspace*workspace;
  gsl_integration_workspace*workspace2;
  double Rp;             // R_perp (i.e. projected separation)
  double Rp2;            // R_perp^2
  double Rmis;           // Miscentering length
  double Rmis2;          // Rmis^2
  double Rmin_interp;    // R_min where Sigma_cen is computed
  double Rmax_interp;    // R_max where Sigma_cen is computed
  double Rp_cos_theta_2; // Variable to hold 2*Rp*cos(theta)
  gsl_function F_radial; // Function for the radial integrand
}integrand_params;



/** @brief Function returning the interpolated 
 *         (or extrapolated) surface density.
 *
 *  @param R Radius.
 *  @param params Container of the integrand parameters.
 *  @return The surface density computed at R.
 */
double get_Sigma(double R, void*params) {

  integrand_params*pp = (integrand_params*)params;
  
  if (R < pp->Rmin_interp) // linear extrapolation at low radii
    return gsl_spline_eval(pp->spline, log(pp->Rmin_interp), pp->acc) +
      (R - pp->Rmin_interp) / (pp->Rmin_interp * 1.1 - pp->Rmin_interp) *
      (gsl_spline_eval(pp->spline, log(pp->Rmin_interp * 1.1), pp->acc) -
       gsl_spline_eval(pp->spline, log(pp->Rmin_interp), pp->acc));
  
  else if (R > pp->Rmax_interp) // zero at high radii
    return 0.;
  
  else
    return gsl_spline_eval(pp->spline, log(R), pp->acc);

}


/** @brief Integrand of the offcentered profile of a 
 *         single halo.
 *
 *  @param theta Angle.
 *  @param params Container of the integrand parameters.
 *  @return The integrand computed at theta.
 */
double integrand_Sigma_off_radial_single(double theta, void*params) {

  integrand_params*pp = (integrand_params*)params;
  double Rmis2 = (pp->Rmis2 != pp->Rp2) ? pp->Rmis2 : pp->Rmis2 + 1.e-10;
  double arg = sqrt(pp->Rp2 + Rmis2 - 2. * pp->Rp * pp->Rmis * cos(theta));
  
  return get_Sigma(arg, params);
  
}


/** @brief Integrand of the offcentered profile of a 
 *         stack of halo, assuming a Rayleigh
 *         distribution for the offcentering scale.
 *
 *  @param Rc_ Offcentering radius.
 *  @param params Container of the integrand parameters.
 *  @return The integrand computed at Rc_.
 */
double integrand_Sigma_off_radial_Rayleigh(double Rc_, void*params) {

  integrand_params*pp = (integrand_params*)params;
  double Rc = (Rc_ != pp->Rp) ? Rc_ : Rc_ + 1.e-10;
  double Rc2 = Rc * Rc;
  double arg = sqrt(pp->Rp2 + Rc2 - Rc * pp->Rp_cos_theta_2);

  return Rc / pp->Rmis2 * exp(-0.5 * Rc2 / pp->Rmis2)
    * get_Sigma(arg, params);
  
}


/** @brief Integrand of the offcentered profile of a 
 *         stack of halo, for the integration
 *         over the angle theta.
 *
 *  @param theta Angle.
 *  @param params Container of the integrand parameters.
 *  @return The integrand computed at theta.
 */
double integrand_Sigma_off(double theta, void*params) {

  double result, err;
  integrand_params*pp = (integrand_params*)params;
  pp->Rp_cos_theta_2 = pp->Rp * cos(theta) * 2;

  gsl_integration_qag(&pp->F_radial,
		      0.,
		      pp->Rp + 5. * pp->Rmis,
		      ABSERR,
		      RELERR,
		      workspace_size,
		      KEY,
		      pp->workspace2,
		      &result,
		      &err);
  return result;
  
}


/** @brief Function computing the offcentered
 *         surface mass density profile.
 *
 *  @param R_ Radial points where the profile is computed.
 *  @param Rs_ Radial points where the input centered
 *             profile, Sigma_cen_, is computed.
 *  @param Sigma_cen_ Centered profile computed at Rs_.
 *  @param Rmis Miscentering radius in case of a single
 *              halo; rms of the misplacement
 *              distribution in case of a stack of halos.
 *  @param Sigma_mis_ output offcentered profile.
 *  @param method "single" for a single halo, "all" for a
 *                stack of halos.
 *  @return The offcentered surface mass density.
 */
void Sigma_off(py::array_t<double> R_,
	       py::array_t<double> Rs_,
	       py::array_t<double> Sigma_cen_,
	       double Rmis,
	       py::array_t<double> Sigma_mis_,
	       std::string method)
{
  const int Ns = Rs_.request().size;
  
  // Allocate
  double *R = static_cast<double *>((R_.request()).ptr);
  double *Rs = static_cast<double *>((Rs_.request()).ptr);
  double *Sigma_cen = static_cast<double *>((Sigma_cen_.request()).ptr);
  double *Sigma_mis = static_cast<double *>((Sigma_mis_.request()).ptr);
  
  static gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
  static gsl_interp_accel*acc = gsl_interp_accel_alloc();
  static gsl_integration_workspace*workspace =
    gsl_integration_workspace_alloc(workspace_size);
  static gsl_integration_workspace*workspace2 =
    gsl_integration_workspace_alloc(workspace_size);
  static integrand_params params;

  // Check the requested ranges
  std::string err_msg = "Error from Sigma_off() ";
  err_msg += "in integration_routines.cpp: ";
  
  if ( (Rs[0] > 1.e-5) || (Rs[Rs_.request().size-1] < 1.e2) )
    throw std::invalid_argument(err_msg+
				"you must provide values of centered "+
				"surface density at least in the "+
				"radial range [1.e-5,1.e2] Mpc!");
  
  // Interpolate Sigma_cen
  double*lnRs = (double*)malloc(Ns*sizeof(double));
  for(int i=0; i<Ns; i++)
    lnRs[i] = log(Rs[i]);
  gsl_spline_init(spline, lnRs, Sigma_cen, Ns);

  // Set the integrand parameters
  params.spline = spline;
  params.acc = acc;
  params.workspace = workspace;
  params.workspace2 = workspace2;
  params.Rmis = Rmis;
  params.Rmis2 = Rmis * Rmis;
  params.Rmin_interp = Rs[0];
  params.Rmax_interp = Rs[Ns-1];
  
  // Set the integrand
  gsl_function F;
  gsl_function F_radial;

  if (method == "all")
    F.function = &integrand_Sigma_off;
  else if (method == "single")
    F.function = &integrand_Sigma_off_radial_single;
  
  F_radial.function = &integrand_Sigma_off_radial_Rayleigh;
  F_radial.params = &params;
  
  params.F_radial = F_radial;
  F.params = &params;
  
  // Compute the integral
  double result, err;
  
  for (int i=0; i<int(R_.request().size); i++) {
    params.Rp  = R[i];
    params.Rp2 = R[i] * R[i];

    gsl_integration_qag(&F, 0, M_PI, ABSERR, RELERR, workspace_size,
			KEY, workspace, &result, &err);
    
    Sigma_mis[i] = result/(M_PI);
  }
  
}


/** @brief Integrand of the mean offcentered profile.
 *
 *  @param R Radius.
 *  @param params Container of the integrand parameters.
 *  @return The integrand computed at R.
 */
double integrand_Sigma_mean_off(double R, void*params) {

  integrand_params*pp = (integrand_params*)params;
  
  return gsl_spline_eval(pp->spline, log(R), pp->acc) * R;
  
}


/** @brief Function computing the offcentered
 *         excess surface mass density profile.
 *
 *  @param R_ Radial points where the profile is computed.
 *  @param Roff_ Radial points where the offcentered
 *               surface mass density profile is computed, 
 *               used to interpolate it before its integration.
 *  @param Rs_ Radial points where the input centered
 *             profile, Sigma_cen_, is computed.
 *  @param Sigma_cen_ Centered profile computed at Rs_.
 *  @param Rmis Miscentering radius in case of a single
 *              halo; rms of the misplacement
 *              distribution in case of a stack of halos.
 *  @param DeltaSigma_mis_ output offcentered profile.
 *  @param method "single" for a single halo, "all" for a
 *                stack of halos.
 *  @return The offcentered excess surface mass density.
 */
void DeltaSigma_off(py::array_t<double> R_,
		    py::array_t<double> Roff_,
		    py::array_t<double> Rs_,
		    py::array_t<double> Sigma_cen_,
		    double Rmis,
		    py::array_t<double> DeltaSigma_mis_,
		    std::string method)
{

  // Allocate
  double *R = static_cast<double *>((R_.request()).ptr);
  double *Roff = static_cast<double *>((Roff_.request()).ptr);
  double *Rs = static_cast<double *>((Rs_.request()).ptr);
  double *DeltaSigma_mis = static_cast<double *>((DeltaSigma_mis_.request()).ptr);

  // Check radial ranges
  std::string err_msg = "Error from DeltaSigma_off() ";
  err_msg += "in integration_routines.cpp: ";
  
  if ( (Rs[0] > 1.e-5) || (Rs[Rs_.request().size-1] < 1.e2) )
    throw std::invalid_argument(err_msg+
				"you must provide radius values for "+
				"the interpolation of the offcentered "+
				"surface density at least in the "+
				"radial range [1.e-10,1.e2] Mpc!");

  // Compute and interpolate Sigma_off
  const int Noff = Roff_.request().size;
  
  double*lnRoff = (double*)malloc(Noff*sizeof(double));
  for(int i=0; i<Noff; i++)
    lnRoff[i] = log(Roff[i]);

  py::array_t<double> Sigma_mis_(Noff);
  Sigma_off(Roff_, Rs_, Sigma_cen_, Rmis, Sigma_mis_, method);
  
  static gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, Noff);
  double *Sigma_mis = static_cast<double *>((Sigma_mis_.request()).ptr);  
  gsl_spline_init(spline, lnRoff, Sigma_mis, Noff);

  // Derive DeltaSigma
  gsl_function F;
  static integrand_params params;
  static gsl_interp_accel*acc = gsl_interp_accel_alloc();
  
  F.function = &integrand_Sigma_mean_off;
  params.spline = spline;
  params.acc = acc;
  F.params = &params;

  static gsl_integration_workspace*workspace =
    gsl_integration_workspace_alloc(workspace_size);
  double result, err;
  
  for (int i=0; i<int(R_.request().size); i++) {
    gsl_integration_qag(&F, Roff[0], R[i], ABSERR, RELERR, workspace_size,
    			KEY, workspace, &result, &err);
    
    DeltaSigma_mis[i] = result * 2.0 / (R[i] * R[i])
      - gsl_spline_eval(spline, log(R[i]), acc);
  }
  
}


PYBIND11_MODULE(integration_routines, m) {
    m.def("Sigma_off", &Sigma_off,
	  "Function computing the off-centered surface mass density.");
    m.def("DeltaSigma_off", &DeltaSigma_off,
	  "Function computing the off-centered excess surface mass density.");
}
