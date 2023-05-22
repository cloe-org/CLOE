This Graphic User Interface (GUI) can be used to easily modify the default initialisation files needed to run CLOE. Note that this GUI cannot be used to run an MCMC, but rather allows the user to generate the ini files needed for such a run. Please see the CLOE documentation for how to execute an MCMC run given these files.

In order to use the `gui`, run the following the command

```
python3 script_gui.py
```

After setting the values in the `gui` to the desired ones, pressing the `Produce ini files` button will save files with the requested settings in the `config` folder. This will overwrite the default files, with the chosen specifications.
