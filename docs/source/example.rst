Example
==================

Here you can find a simple example of how to run the Euclid likelihood as an external likelihood using the Bayesian Analysis tool **cobaya** (check its `docs  <https://cobaya.readthedocs.io/en/latest/index.html>`_).

For a detailed break-down of how to run the likelihood in this and other ways, see the ``DEMO`` notebook supplied with this package.

One way for the likelihood to be run is from an interactive python interpreter (**jupyter notebook**).

From a Python interpreter
-------------------------

An external likelihood can be supplied to **cobaya** directly through a Python interpreter. In order to do this, the choice of likelihood and parameters must be stored in a Python dictionary. For example:

.. code:: python

    # We are running the Euclid-Likelihood as an external likelihood class for Cobaya
    # Cobaya needs a dictionary or yaml file as input to start running
    # This dictionary below ('info') can be modified up to some point by the user to
    # adapt it to the user's needs.
    # The options that can be modified by the user are pointed with the acronym (UC).

    info = {
    #'params': Cobaya's protected key of the input dictionary.
    # Includes the parameters that the user would like to sample over:
        'params': {
            # (UC): each parameter below (which is a 'key' of another sub-dictionary) can contain a dictionary
            # with the key 'prior', 'latex'...
            # If the prior dictionary is not passed to a parameter, this parameter is fixed.
            # In this example, we are sampling the parameter ns
            # For more information see: https://cobaya.readthedocs.io/en/latest/example.html
            'ombh2': 0.022445, #Omega density of baryons times the reduced Hubble parameter squared
            'omch2': 0.1205579307, #Omega density of cold dark matter times the reduced Hubble parameter squared
            'H0': 67, #Hubble parameter evaluated today (z=0) in km/s/Mpc
            'tau': 0.0925, #optical depth
            'mnu': 0.06, #  sum of the mass of neutrinos in eV
            'nnu': 3.046, # N_eff, or number of relativistic species
            'As': 2.12605e-9, #Amplitude of the primordial scalar power spectrum
            'ns': {'prior':{'min':0.8, 'max':1.2}}, # primordial power spectrum tilt (sampled with an uniform prior)
            'w': -1, # Present-day Dark energy equation of state parameter in PPF model
            'wa': 0, # Early-time Dark energy equation of state parameter in PPF model
            'omk': 0.0, #curvature density
            'omegam': None, #DERIVED parameter: Omega matter density
            'omegab': None, #DERIVED parameter: Omega barion density
            'omeganu': None, #DERIVED parameter: Omega neutrino density
            'omnuh2': None, #DERIVED parameter: Omega neutrino density times de reduced Hubble parameter squared
            'omegac': None, #DERIVED parameter: Omega cold dark matter density
            # (UC): change 'like_selection' based on which observational probe you would like to use.
            # Choose among:
            # 1: photometric survey
            # 2: spectroscopic survey
            # 12: both surveys
            'like_selection': 2,
            # (UC): if you selected the photometric survey (1) or both (12) in 'like_selection'
            # you may want to choose between:
            # using Galaxy Clustering photometric and Weak Lensing probes combined assuming they are independent ('full_photo': False)
            # or Galaxy Clustering photometric, Weak Lensing and the cross-correlation between them ('full_photo': True)
            # This flag is not used if 'like_selection: 2'
            'full_photo': False,
            # (UC): galaxy bias parameters:
            # The bias parameters below are currently fixed to the
            # values used by the Inter Science Taskforce: Forcast (IST:F)
            # and presented in the corresponding IST:F paper (arXiv: 1910.09273).
            # However, they can be changed by the user and even sample over them by putting a prior
            # Photometric bias parameters
            'b1_photo': 1.0997727037892875,
            'b2_photo': 1.220245876862528,
            'b3_photo': 1.2723993083933989,
            'b4_photo': 1.316624471897739,
            'b5_photo': 1.35812370570578,
            'b6_photo': 1.3998214171814918,
            'b7_photo': 1.4446452851824907,
            'b8_photo': 1.4964959071110084,
            'b9_photo': 1.5652475842498528,
            'b10_photo': 1.7429859437184225,
            # Spectroscopic bias parameters
            'b1_spec': 1.46,
            'b2_spec': 1.61,
            'b3_spec': 1.75,
            'b4_spec': 1.90},
        #'theory': Cobaya's protected key of the input dictionary.
        # Cobaya needs to ask some minimum theoretical requirements to a Boltzman Solver
        # (UC): you can choose between CAMB or CLASS
        # In this DEMO, we use CAMB and specify some CAMB arguments
        # such as the number of massive neutrinos
        # and the dark energy model
        #
        # ATTENTION: If you have CAMB/CLASS already installed and
        # you are not using the likelihood conda environment
        # or option (2) in cell (3) (Cobaya modules), you can add an extra key called 'path' within the camb dictionary
        # to point to your already installed CAMB code:
        'theory': {'camb':
                   {'stop_at_error': True, #'path': 'path/to/camb',
                    'extra_args':{'num_massive_neutrinos': 1,
                                  'dark_energy_model': 'ppf'}}},
        #'sampler': Cobaya's protected key of the input dictionary.
        # (UC): you can choose the sampler you want to use.
        # Check Cobaya's documentation to see the list of available samplers
        # In this DEMO, we use the 'evaluate' sampler to make a single computation of the posterior distributions
        # WARNING: at the moment, the only sampler that works is 'evaluate'
        'sampler': {'evaluate': None},
        # 'packages_path': Cobaya's protected key of the input dictionary.
        # This is the variable you need to update
        # if you are running Cobaya with cobaya_modules (option (2) above).
        # If you are using the conda likelihood environment or option (1),
        # please, comment the line below
        #
        'packages_path': modules_path,
        #
        #'output': Cobaya's protected key of the input dictionary.
        # Where are the results going to be stored, in case that the sampler produce output files?
        # For example: chains...
        # (UC): modify the path below within 'output' to choose a name and a directory for those files
        'output': 'chains/my_euclid_experiment',
        #'likelihood': Cobaya's protected key of the input dictionary.
        # (UC): The user can select which data wants to use for the analysis.
        # Check Cobaya's documentation to see the list of the current available data experiments
        # In this DEMO, we load the Euclid-Likelihood as an external function, and name it 'Euclid'
        'likelihood': {'Euclid': EuclidLikelihood},
        #'debug': Cobaya's protected key of the input dictionary.
        # (UC): how much information you want Cobaya to print? If debug: True, it prints every single detail
        # that is going on internally in Cobaya
        'debug': True,
        #'timing': Cobaya's protected key of the input dictionary.
        # (UC): if timing: True, Cobaya returns how much time it took it to make a computation of the posterior
        # and how much time take each of the modules to perform their tasks
        'timing': True,
        #'force': Cobaya's protected key of the input dictionary.
        # (UC): if 'force': True, Cobaya forces deleting the previous output files, if found, with the same name
        'force': True
        }

The dictionary above has several  *keys*:

- A ``params`` key: parameters that are going to be explored (or derived). Most of the time, these will be computed from the ``theory`` code (i.e: **CAMB** or **CLASS**). If sampled, you can choose their ``prior``, the Latex label for them that will be used in the plots, the reference (``ref``) starting point for the chains (optional), and the initial spread of the MCMC covariance matrix (``proposal``).
- A ``theory`` key: Boltzmann Solver we want to use (i.e: **CAMB** or **CLASS**) to compute theoretical quantities.
- A ``sampler`` key: block stating that we will use the ``mcmc`` sampler to explore the prior+likelihood described above, stating the maximum number of samples used, how many initial samples to ignore, and that we will sequentially refine our initial guess for a covariance matrix. Another samplers such as **polychord** are accepted.
- A ``modules`` key: path where your external codes (i.e: **CAMB** or **polychord**) are installed. If they are not installed using the structure **cobaya** automatically creates when using automatic installation, you can give particular paths to each of the codes in the corresponding *key*.
- An ``output`` key: path where the products will be written and a prefix for their name.
- A ``likelihood`` key: likelihood pdf's to be used. In this case, we call an external likelihood file that returns the loglike given the ``params`` values.


Once this dictionary has been set up, to run **cobaya** from the **jupyter notebook** use:

.. code:: python

    # Import cobaya run function
    from cobaya.run import run

    # Let's run cobaya
    # the function run returns
    # info_updated: an information dictionary updated with the defaults,
    # equivalent to the updated yaml file produced by the shell invocation
    # samples: a sampler object, with a sampler.products()
    # being a dictionary of results.
    # For the mcmc sampler, the dictionary contains only one chain under the key 'sampler'.

    info_updated, samples = run(info)

For further information, see the ``DEMO`` notebook provided with this package.
