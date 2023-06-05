from PyQt5 import QtWidgets
from gui_core import Ui_MainWindow
from functools import partial
from PyQt5 import QtWidgets
import yaml 

class ConfigGUIWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ConfigGUIWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.ini_save_button.pressed.connect(partial(self.dumpToYAMLAndExit))
        self.dialog = None
        self.outfile_line_edit = None
        self.save_btn = None
    
    def read_cosmo_pars(self, latex_name, max, min, proposal, fiducial, scale):
        if max != "" and min != "" and proposal != "":
            read_values = {"latex" : latex_name, "prior" : {"max" : float(max),
                    "min" : float(min)},
                    "proposal" : float(proposal), "ref": {"dist" : "norm",
                    "loc": float(fiducial)}, "scale" : float(scale)}
        else:
            read_values = float(fiducial)
        return read_values

    def dumpToYAMLAndExit(self):
        dictionary_WL = {
             "statistics": "angular_power_spectrum",
             "bins":{"n"+str(i+1) : {"n"+str(j+1): {"ell_range":
             [[int(self.ui.Min_l_WL.text()), int(self.ui.Max_l_WL.text())]]} for j in
             range(i,10) } for i in range(10)}}
        dictionary_GCph = {
             "statistics": "angular_power_spectrum",
             "bins":{"n"+str(i+1) : {"n"+str(j+1): {"ell_range":
             [[int(self.ui.Min_l_GCph.text()), int(self.ui.Max_l_GCph.text())]]} for j in
             range(i,10) } for i in range(10)}}
             
        dictionary_GCsp = {
            "statistics": "legendre_multipole_power_spectrum", "bins":{"n1": {"n1": {"multipoles":
        {0:{"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        2:{"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        4:{"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]}}}},
        "n2": {"n2": {"multipoles":
        {0: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        2: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        4: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]}}}},
        "n3": {"n3": {"multipoles":
        {0: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        2: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        4: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]}}}},
        "n4": {"n4": {"multipoles":
        {0: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        2: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]},
        4: {"k_range": [[float(self.ui.GCsp_k_min.text()), float(self.ui.GCsp_k_max.text())]]}}}}
                                                                    }}
        dictionary_observable_selection = {"WL" : {"WL" : self.ui.include_WL.isChecked(),
                                                   "GCphot" : self.ui.GGL_include.isChecked(),
                                                   "GCSpectro" : False},
                                           "GCphot" : {"GCphot" : self.ui.include_GCph.isChecked(),
                                                   "GCSpectro" : False},
                                           "GCSpectro" : {
                                                   "GCSpectro" : self.ui.GCsp_include.isChecked()}}
        dictionary_cosmo = {"As": {"latex": "A_\mathrm{s}",
        "value": "lambda logA: 1e-10*np.exp(logA)"},
                            "H0" : self.read_cosmo_pars("H_0", self.ui.cosmo_h_max.text(),
                            self.ui.cosmo_h_min.text(), self.ui.cosmo_h_proposal.text(),
                            self.ui.cosmo_h_fiducial.text(), 1.0),
                            "logA": self.read_cosmo_pars("\log(10^{10} A_\mathrm{s})",
                            self.ui.cosmo_log10As_max.text(),
                            self.ui.cosmo_log10As_min.text(),
                            self.ui.cosmo_log10As_proposal.text(),
                            self.ui.cosmo_log10As_fiducial.text(), 0.001),
                            "mnu" : self.read_cosmo_pars("M_\\nu",
                            self.ui.cosmo_Mnu_max.text(),
                            self.ui.cosmo_Mnu_min.text(),
                            self.ui.cosmo_Mnu_proposal.text(),
                            self.ui.cosmo_Mnu_fiducial.text(), 0.0001),
                            "nnu": 3.04,
                            "ns" : self.read_cosmo_pars("n_\mathrm{s}",
                            self.ui.cosmo_ns_max.text(), self.ui.cosmo_ns_min.text(),
                            self.ui.cosmo_ns_proposal.text(),
                            self.ui.cosmo_ns_fiducial.text(), 0.004),
                            "ombh2" : self.read_cosmo_pars("\Omega_\mathrm{b} h^2",
                            self.ui.cosmo_omegab_max.text(),
                            self.ui.cosmo_omegab_min.text(),
                            self.ui.cosmo_omegab_proposal.text(),
                            self.ui.cosmo_omegab_fiducial.text(), 0.0001),
                            "omch2" : self.read_cosmo_pars("\Omega_\mathrm{c} h^2",
                            self.ui.cosmo_omegac_max.text(),
                            self.ui.cosmo_omegac_min.text(),
                            self.ui.cosmo_omegac_proposal.text(),
                            self.ui.cosmo_omegac_fiducial.text(), 0.0001),
                            "omegab": {"latex": "\Omega_\mathrm{b}"},
                            "omegam": {"latex": "\Omega_\mathrm{m}"},
                            "omk" : self.read_cosmo_pars("\Omega_\mathrm{K}",
                            self.ui.cosmo_OmegaK_max.text(),
                            self.ui.cosmo_OmegaK_min.text(),
                            self.ui.cosmo_OmegaK_proposal.text(),
                            self.ui.cosmo_OmegaK_fiducial.text(), 0.0001),
                            "sigma8": {"latex": "\sigma_8"},
                            "tau": 0.0925,
                            "w" : self.read_cosmo_pars("w_0",
                            self.ui.cosmo_w0_max.text(),
                            self.ui.cosmo_w0_min.text(),
                            self.ui.cosmo_w0_proposal.text(),
                            self.ui.cosmo_w0_fiducial.text(), 0.001),
                            "wa" : self.read_cosmo_pars("w_\mathrm{a}",
                            self.ui.cosmo_wa_max.text(),
                            self.ui.cosmo_wa_min.text(),
                            self.ui.cosmo_wa_proposal.text(),
                            self.ui.cosmo_wa_fiducial.text(), 0.01),
                            "gamma_MG" : self.read_cosmo_pars("\gamma_\mathrm{MG}",
                            self.ui.cosmo_gammaMG_max.text(),
                            self.ui.cosmo_gammaMG_min.text(),
                            self.ui.cosmo_gammaMG_proposal.text(),
                            self.ui.cosmo_gammaMG_fiducial.text(), 0.01)}

        if self.ui.Photo_SSC_include.isChecked():
            cov_model = "SuperSample"
        else:
            cov_model = "Gauss"

        dictionary_data = {"photo": { "IA_model": "zNLA",
                            "cov_3x2pt": self.ui.Photo_gaussian_covariance_filename.text(),
                            "cov_GC": "CovMat-PosPos-{:s}-20Bins.npy",
                            "cov_WL": "CovMat-ShearShear-{:s}-20Bins.npy",
                            "cov_model": cov_model,
                            "ndens_GC": "niTab-EP10-RB00.dat",
                            "ndens_WL": "niTab-EP10-RB00.dat",
                            "root_GC": self.ui.GCph_filename.text(),
                            "root_WL": self.ui.WL_filename.text(),
                            "root_XC": self.ui.GGL_filename.text()},
                            "sample": "ExternalBenchmark",
                            "spectro" :
                            { "redshifts": [self.ui.GCsp_redshifts.text().split()],
                            "edges" : [self.ui.GCsp_edges.text().split()],
                            "root": self.ui.GCsp_covariance_filename.text()}
                            }

        with open('../configs/WL.yaml', 'w') as yaml_file:
            yaml.dump(dictionary_WL, yaml_file, default_flow_style=None, sort_keys=False)
        with open('../configs/GCphot.yaml', 'w') as yaml_file:
            yaml.dump(dictionary_GCph, yaml_file, default_flow_style=None, sort_keys=False)
        with open('../configs/GCspectro.yaml', 'w') as yaml_file:
            yaml.dump(dictionary_GCsp, yaml_file, default_flow_style=None, sort_keys=False)
        with open('../configs/observables_selection.yaml', 'w') as yaml_file:
            yaml.dump(dictionary_observable_selection, yaml_file, default_flow_style=None,
            sort_keys=False)
        with open('../configs/models/cosmology.yaml', 'w') as yaml_file:
            yaml.dump(dictionary_cosmo, yaml_file, default_flow_style=None, sort_keys=False)
        with open('../configs/data.yaml', 'w') as yaml_file:
            yaml.dump(dictionary_data, yaml_file, default_flow_style=None, sort_keys=False)
    
import sys


import qtvscodestyle as qtvsc

app = QtWidgets.QApplication(sys.argv)

stylesheet = qtvsc.load_stylesheet(qtvsc.Theme.LIGHT_VS)

app.setStyleSheet(stylesheet)
window = ConfigGUIWindow()
window.show()
app.exec()
