# CosmoSIS implementation of CLOE

This folder contains the files neccessary to run CLOE with `CosmoSIS`, an alternate cosmological Bayesian analysis pipeline.

There are two ways to run CLOE with `CosmoSIS`: the main difference between them being whether the user wants to call the Boltzmann solver (A) from `CosmoSIS`, or (B) from `Cobaya`. 
We consider (A) to be the primary approach. In both cases, the user does not need to interact with `Cobaya`.

**Note: We first encourage you to include a minor change in the CosmoSIS source code. Add a line under the section `[nuisance parameters]`:
```
cloe_parameters = "cloe_parameters"
```
in the `cosmosis/datablock/cosmosis_py/section_names.py` file, where you have installed your CosmoSIS package.**

We next provide a brief description of the files in the `cosmosis` directory. The file used in both scenarios (A) and (B) is:

### cosmosis_validation.ipynb
This file is a python notebook which serves as a validation test for both pipelines (Scenarios A/B) against the fiducial CLOE/`Cobaya` pipeline. 
Users can also run this notebook to check that they have a working configuration of `CosmoSIS`/`Cobaya`.

The following files have been separated based on the aforementioned 2 scenarios:

## Scenario A

Should the user choose to run the pipeline as in Scenario (A), i.e., separating the Boltzmann solver and likelihood calculations, the files of concern are:

### run_cosmosis.ini
This file is the top-level parameter file containing the pipeline specifications to run CLOE through `CosmoSIS`. The user starts running the `CosmoSIS` pipeline (without MPI in this case) by executing the command 

```bash
cosmosis run_cosmosis.ini
```

This file has been arranged in the format typical of `CosmoSIS` parameter files, with options to specify the path of the file containing the parameters to sample over, 
the sampler to use, the output file path and format, and the modules that make up the pipeline. Here two modules are specified: the Boltzmann solver (we set `[camb]` as an example) and `[euclid]`, 
which is the CLOE module. For the `[camb]` module we use the pubilc release of the CAMB Boltzmann solver and the `camb_interface.py` script within this folder.

The `[euclid]` module takes in the path of the `CosmoSIS`-CLOE interface script and the path of the yaml configuration file as parameters. 
Note that the module is named `[euclid]` while the likelihood is called `cloe` to make the difference more explicit.

### config_for_cosmosis.yaml
This file, which is in the same format as the yaml files in the `config` folder, defines the parameters required by `CosmoSIS`. 
It will be read in as a dictionary. The user is free to define their desired values in this yaml file. 

### cosmosis_values.ini
This file lists the parameters and their priors to sample over. Here, the cosmological parameters fall under the section `[cosmological_parameters]`, 
and the nuisance parameters specific to CLOE have been grouped under the section `[cloe_parameters]`. Each parameter is defined as

```
param = min bestfit max
```

### cosmosis_priors.ini
This file lists the prior distributions of each of the parameters, if they have one (for example, if the parameter is known to follow a Gaussian prior). Each parameter is defined as

```
param = distribution mean sigma
```

If not defined, the prior distribution is assumed to be flat. 

### cosmosis_interface.py
This file is the interface script between `CosmoSIS` and CLOE in the case of scenario A. It takes in the configuration yaml file and desired output path of the chain as parameters. 
Within this script, the cosmological parameters calculated with the Boltzmann solver in the previous step are retrieved from the `CosmoSIS` datablock and converted into a CLOE-friendly format, 
for the logposterior to be calculated within the CLOE module. The log-likelihood is then put into the `CosmoSIS`-native likelihood datablock.  

### camb_interface.py
This file has been adapted from the original `camb_interface.py` script from the `cosmosis-standard-library` repository, such that it is ensured that the options for the Boltzmann solver follows 
that in the `config_for_cosmosis.yaml` file. Hence, it is neccessary to specify this yaml file in the `[camb]` section of the ini file. 

## Scenario B

Should the user choose to run the pipeline as in Scenario (B), i.e., calling `Cobaya` through `CosmoSIS` to carry out the cosmological and likelihood calculations, the files of concern are:

### run_cosmosis_with_cobaya.ini
This file is the top-level parameter file containing the pipeline specifications to run CLOE through `CosmoSIS`, with Cobaya being run in the background. 
The user starts running the `CosmoSIS` pipeline (without MPI in this case) by executing the command 

```bash
cosmosis run_cosmosis_with_cobaya.ini
```

This file has been arranged in the format typical of `CosmoSIS` parameter files, with options to specify the path of the file containing the parameters to sample over, the sampler to use, the output file path and format, 
and the modules that make up the pipeline. Here only one module needs to be called, i.e., `[euclid]`, since in this case both the theoretical and likelihood calculations are done within that one module. 

The `[euclid]` module takes in the path of the `CosmoSIS`-CLOE interface script, configuration file and chain output path (written by `Cobaya`, on top of the one written by `CosmoSIS`) as parameters. 
Note that the module is named `[euclid]` while the likelihood is called `cloe` to make the difference more explicit.

### cobaya_config_for_cosmosis.yaml
This file, which is in the same format as the yaml files in the `config` folder, defines the parameters required by `Cobaya` to run CLOE as an external likelihood. 
It will be read in as a dictionary. The user is free to define their desired values in this yaml file. 

### cosmosis_with_cobaya_values.ini
This file lists the parameters and their priors to sample over. Here, the cosmological parameters fall under the section `[cosmological_parameters]`, 
and the nuisance parameters specific to CLOE have been grouped under the section `[cloe_parameters]`. This file is basically identical to `cosmosis_values.ini`, 
except for a few subtleties in the naming conventions of certain parameters.  Each parameter is defined as

```
param = min bestfit max
```

### cosmosis_with_cobaya_priors.ini
This file lists the prior distributions of each of the parameters, if they have one (for example, if the parameter is known to follow a Gaussian prior). Each parameter is defined as

```
param = distribution mean sigma
```

If not defined, the prior distribution is assumed to be flat. 

### cosmosis_with_cobaya_interface.py
This file is the interface script between `CosmoSIS` and CLOE. It takes in the configuration yaml file and desired output path of the chain as parameters. 
Within this script, `CosmoSIS` does the sampling of points, but relies on CLOE within `Cobaya` to calculate the cosmology and logposterior value, which is then put back into the `CosmoSIS`-native likelihood datablock. 
