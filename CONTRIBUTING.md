# Euclid IST:L Contribution Guidelines

> Authors: Samuel Farrens, Dida Markovic  
> Last Updated: 15/12/2022

## Contents

1. [Introduction](#introduction)
1. [Git Development Guidelines](#git-development-guidelines)
1. [IST:NL Development Guidelines](#istnl-development-guidelines)
1. [Style choices](#style-choices)

## Introduction

These guidelines aim to help you contribute to the development of the CLOE IST:L package. Before attempting to make any changes to the code please do the following:

- Get authorisation from the IST:L leads to implement the changes you have in mind.
- Read and adhere to the guidelines provided in this document.

## Git Development Guidelines

This section contains some guidelines on how to contribute to and manage the development of a Git repository.

### Git

Git is a distributed version-control system for tracking changes in source code during software development.

Before working on a Git repository make sure you gave `git` installed.

```bash
$ git --version
```

You can find instructions on the [Git website](https://git-scm.com) for how to install it on your system.

### Cloning

When working with a Git repository it is essential to keep up to date with developments on the `master` branch. To this end, developers should *clone* the contents of the remote (*i.e.* online) repository rather than simply downloading it. This will ensure that the full development history is available.

To clone a repository (*i.e.* make a local copy of the remote repository) use the `git clone` command. For example to clone this IST likelihood repository:

```bash
$ git clone https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation.git
```

This will copy the contents of the `master` branch to your computer.

> **Note:** Before cloning you should set up an SSH key for the GitLab repository ([see instructions](https://docs.gitlab.com/ce/ssh/README.html)).

### Remote vs Local

After cloning the repository you will be able to make modifications on your local copy and track updates on the *remote* (*i.e.* the online GitLab repository). You can display the current remote settings using the `git remote` command. For the pipeline you should see the following output:

```bash
$ git remote -v
origin	https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation (fetch)
origin	https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation (push)
```

`origin` is the remote repository name and has the same address for both `fetch` and `push` commands.

> **Note:** Any changes made to the local repository will remain completely independent until they have been committed and pushed to the remote repository.

### Pulling Updates

Before making any modifications to the repository you should make sure your local copy is up to date with the remote. You can update your copy with the `git pull` command.

```bash
$ git pull
```

### Branching

Once you have an up-to-date copy of the repository you should create a branch, which should be linked to an open issue. In general an issue should be opened any time you believe something needs to be changed or added to the master branch (*i.e.* a bug fix, a new feature, *etc.*). This will provide context for the repository maintainers when reviewing your merge request.

**You should always avoid making direct commits to the master branch.**

You can view existing branches with the `git branch` command. For example to list all branches on the local and remote repositories:

```bash
$ git branch -a
```

To create a new branch simply specify a branch name. In general, you should choose a name linked to an issue and perhaps specify who is working on this branch, *e.g.*

```bash
$ git branch issue55_sam
```

To switch to the new branch use the `git checkout` command:

```bash
$ git checkout issue55_sam
```

These two commands can be combined into a shortcut:

```bash
$ git checkout -b issue55_sam
```

### Code Modifications

As mentioned before, code modifications should be linked to an open issue. *e.g.* If you find a bug, open an issue stating what the bug is and if known where it originates. If you plan to fix this bug yourself you can specify that you plan to open a merge request for this issue so that other developers are aware.

Any changes you make to the code should retain the current standards in terms of code style and documentation. In particular, pay attention to the code coverage (*i.e.* the number of lines covered by unit tests). If the changes you implement decrease the coverage you should aim to include additional units tests in your merge request.

Avoid including too many changes in a single merge request. Minor bug fixes can be grouped together, but new features should be merged independently. If your merge request contains hundreds of new or changed lines of code the reviewers will have difficultly spotting potential issues.

The best practice in general is to make small changes related to specific issues as often as possible and to avoid waiting several months to merge huge changes to the code.

### Staging

After modifying a file on your branch on your local repository, the file needs to be staged for a commit. Files can be added to the staging area with the `git add` command. You can add individual files:

```bash
$ git add FILE
```

or all files within a given directory:

```bash
$ git add .
```

Files can be removed from the staging area with the command `git reset`, *e.g.*

```bash
$ git reset FILE
```

Finally, you can view the status of the staging area with the command `git status`. This command will list all files currently in the staging area. In addition, it will show untracked files (*i.e.* files that have been modified but not staged).

### Committing

When you are ready to submit changes to files in the staging area to your local repository you can use the `git commit` command as follows:

```bash
$ git commit -m "Short description of the changes made"
```

Make sure to provide a clear and concise description of the changes your are committing to the repository. You should make regular commits each of which constitutes a small number of changes as the repository can always be reset to a previous commit.

### Pushing to the Remote Repository

When you have committed the changes you want you can update the remote repository with the changes made to your local repository with the `git push` command as follows:

```bash
$ git push origin BRANCH_NAME
```

where `origin` is the name of the remote repository and `BRANCH_NAME` is the name of your branch on your local repository. If this is the first push, a new branch will be uploaded to the remote, otherwise your commits will simply be synced with the remote branch.

### Merging

Once all of the changes related to a given issue have been pushed to the remote repository a merge request can be made. To do this simply select your branch on GitLab and press the merge request button. You will need to submit a short description of the changes that will be made to `master` and assign an individual and milestone (if applicable) to the request. After this, the merge request will be reviewed by one of the repository reviewers. At this stage the reviewer may request further changes or point out potential issues. The developer can simply continue working on a local branch and pushing changes to the corresponding remote branch, which remains attached to the merge request. If the proposed changes fail to meet the required standards or the reviewer believes that these changes should not be merged into the master branch then the merge request can be closed (*i.e.* rejected). If accepted, however, your changes will be merged into the master branch and your remote branch will be deleted.

### Clean Up

Following a successful merge request you should do the following:

1. Re-sync your local master with the remote master.

```bash
$ git checkout master
$ git pull
```

2. You should also delete the local branch corresponding to the merge request:

```bash
$ git branch -d BRANCH_NAME
```

> **Note:** Avoid reusing the same name for a given branch.

3. Finally, if you still see your remote branch listed even after it has been deleted run the following command:

```bash
$ git remote prune origin
```

### Bug Tracking

Upon detecting a (potential) bug, developers should open an issue in the `Development` column with the `bug` label. This should be done as soon as possible to avoid bugs being overlooked or forgotten.

In this issue, the developers should clearly specify if they:
a. Plan to open a Merge Request to fix the bug themselves. If so, they can self-assign the issue and link any corresponding Merge Request to the issue.
b. Leave the issue for someone else to resolve. In this case the IST:L leads should assign the task to another developer.

### Newline warnings

Before merging, please ensure that there are no `\No newline at end of file` warnings.


## IST:NL Development Guidelines

This section provides guidelines on how the [IST:NL repository](https://gitlab.euclid-sgs.uk/pf-ist-nonlinear/likelihood-implementation) should be managed as a *fork* (or *downstream repository*) of the IST:L repository (or *upstream repository*). These guidelines are designed to avoid issues with IST:L.

### Forking

In order to maintain a stable link to the upstream repository (*i.e.* the main IST:L repository), it recommended to create a *fork*. This can be done by simply clicking the `fork` button on the top right of the IST:L repository. This will create an identical copy of the repository with a new address.

### Mirroring

This fork will quickly diverge from the upstream if left alone. By setting up a mirror to the upstream master branch the downstream master branch can be kept in sync. This way any changes made to the upstream master will automatically flow down to the downstream master.

> **Note:** this will happen on regular intervals (every x minutes) so there may be a minor delay.

To set up a mirror to the upstream, the IST:NL leads should contact the IST:L leads and request that this be activated. The mirror must be setup by someone with maintainer rights on the upstream repository and this person must also be given access to the downstream repository.

### Protecting the Master Branch

Changes to the downstream master will not flow upstream. Therefore, it is advised that write access to the downstream master be deactivated for all users. This will avoid conflicts with the upstream master.

(The only exception to this are the maintainers, who manage the mirroring. They must retain push access to the IST:NL master in order for the mirroring to work. Even though they technically have access, they should *never* push to the IST:NL master.)

### Downstream Development

The development in the downstream repository must adhere to the following guidelines:

- New features, modules, *etc.* can be developed downstream provided they do not modify the current upstream API.
- Downstream branches can be created and merged into other downstream branches with internal reviews.
- Downstream merge requests have to be managed and reviewed by the downstream team but should maintain the same quality standards.
- Downstream issues and wiki can be managed independently by the downstream team.

### Upstream Development

Changes that should be carried out in the upstream repository include:

- Changes to the current architecture.
- Changes to the current API.
- Bug fixes.
- Generic routines that are not specific to the downstream development objectives such as file IO, basic cosmology routines, *etc.* Downstream managers should identify these cases.

The preference that these changes to be done in the upstream is to avoid duplication of issues between the upstream and the downstream.

Downstream developers that do not have access to the upstream repository should contact the IST:L leads to either request access or to request specific features.

### Upstream Merge Requests

Once downstream content is sufficiently mature a merge request can be made from a downstream branch to the upstream master. Note that the merge request will be opened in the upstream repository and reviewed by the upstream team.

Downstream managers should ensure that the merge request adheres to the following guidelines:

- Ensure that the merge request points to the upstream master and not the downstream master. This should be the default behaviour for a fork.
- Ensure that the merge request includes all the latest upstream commits. Mirroring only guarantees this until a downstream branch is created, any additional commits will need to be manually merged from the remote master to ensure all updates are included.

  ```bash
  $ git pull origin master
  ```
- New unit tests should be added by the downstream team to maintain the current code coverage.
- All existing and new CI tests must still pass.
- The same API documentation style and standards should be adopted.

Once the merge request is opened, the upstream team can carry out a *consistency* review before merging to the upstream master. After merging, the changes will flow to the downstream master. Any downstream branch still open should pull these commits.

### Repository Handling

Developers planning to work on both the upstream and the downstream repositories may wish to avoid cloning both. The easiest way to manage this is to do the following:

- Clone the downstream repository:

  ```bash
  $ git clone DOWNSTREAM_URL
  ```

- Add the upstream to the list of remotes:

  ```bash
  $ git remote add upstream UPSTREAM_URL
  $ git remote -v
  ```

- Use explicit push and pull commands *e.g.*:

  ```bash
  # Push to a downstream branch
  $ git push origin BRANCH_NAME

  # Push to an upstream branch
  $ git push upstream BRANCH_NAME
  ```

- Avoid pushing to either downstream or upstream master branches! This should be disabled for all downstream users and only possible for upstream maintainers, but should be avoided regardless.

- Pulling from either the downstream or the upstream master branches should be equivalent if the mirroring has been activated. When in doubt, however, pull from the upstream to ensure that you have all the latest commits.


## Style choices



#### **General guidelines**

- Existing python packages such as e.g. `numpy`, `scipy` or `astropy` should be used whenever possible.
- The code should follow [PEP8 rules](https://peps.python.org/pep-0008/) to pass the CI tests.
- Try to avoid (as much as possible) using hardcoded parameters. If you do so please add them to the list of hardcoded parameters ([link to the list to be created](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/issues/698#note_83137)). Reviewers should always check if there is good reason for it or not.
- Comments (i.e. lines starting with `#`) may be written in the code only if they are useful for the developers. They should never be targeted at the user and never be printed. Reviewers should check what comments are useful for a given merge request (MR).
- Comments/docstrings should be written in third person (e.g. `Returns` instead of `Return`)
- In accordance with the ECEB Style Guide for Euclid Publications, the convention for the language is British English.
- Do not use all capital letters in comments and printouts.
- Try to avoid as much as possible excluding code from coverage report with `#pragma: no cover`. It is fine to exclude certain things that cannot easily be included in unit tests (e.g. plotting routines). Reviewers should assess for a given MR if a given exclusion is justified.
- Try to avoid as much as possible using lambda functions. Developers need to justify why they are included in a piece of code. Reviewers should check that the use is indeed justified.
- Do not use relative imports and always use absolute imports

   _Example_: 
   write `import cloe.cosmo.cosmology.py` instead of `import ..cosmo.cosmology.py`


#### **Mathematics conventions in the code**

- Use e.g. `1.0` and not `1.` for floats with no decimal part.
- Square roots: always use the `numpy.sqrt()` function
- Inverse: `1.0 / x` (explicitly shows that the results is type `float`)
- Squares: `x ** 2` (space between operators, integer exponents by default)
- Scientific notation: `1.0e-9`


#### **Strings and documentation conventions**

- For string formatting use one of the following options: 

1. using f-strings (**preferred option**), *e.g.*

```python
my_string = f'My value is {my_value}'
```
2. using `.format()`, *e.g.*

```python
my_string = 'My value is {0}'.format(my_value)
```

- For string concatenation, do it as follows:                                                              
```python
my_long_string = (
   'this is a very long string ...'
   + 'that keeps going...'
   + 'and going etc.'
)
```

- External packages (e.g. CLASS, Cobaya ..) should be referred to with the same capitalisation as they appear in their original websites

    _Example_: Use Cobaya as in the [website](https://cobaya.readthedocs.io/en/latest/) and not COBAYA or     
cobaya.
- When specifying the variable type in the API docstrings follow the convention of the [PEP8 rules](https://peps.python.org/pep-0008/). 

    _Example_: use `numpy.ndarray` and not `np.ndarray` otherwise the link in the API will not work.

#### **Naming conventions**

##### Variables 

* Keep all variable names lowercase and separate words/terms with a `_` (e.g. `my_var` not `MyVar`).
* Try to avoid (as much as possible) using single letter variable names. The same applies to the corresponding docstring.

_A few examples for CLOE:_  

    - The wavenumber k has to be indicated as `wavenumber` (i.e. the wavenumber should not be called `k` but `wavenumber`).
    - The redshift z has to be indicated as `redshift`.
    - The overdensity $`\rm{\delta}`$ has to be indicated as `delta`.



* Explicit, human readable variable names are usually easier to maintain that overly condensed names (e.g. `delta_sigma_rho` is easier to understand than `dsr`) but avoid making the names too long (e.g. `this_variable_name_is_certainly_way_too_long`).
* For variables with short lifespans (i.e. those that only exist in loop) you can prepend a `_` to highlight that this variable has a limited scope. e.g.:

```python
for _value in object:
    res += myfunc(_value)
    ...
```

##### Function names

* Keep all function names lowercase and separate words/terms (e.g. `my_func` not `MyFunc`, `calculate_likelihood` is better than `clike`).
* A function starting with a `_` is considered *private* (i.e. this is only used within the module and is not intended to be used outside of this scope). 
* Longer more explicit names can be used for *internal* functions (i.e. functions used within the code, e.g. `calculate_value_per_bin`) and more easy-to-remember/user-friendly names should be used for *external* functions (i.e. functions that users will need to interact with, e.g. `getlike`).



##### Convention for naming of the different probe observables

* Weak lensing --> `WL`
* Photometric galaxy clustering --> `GCphot` 
* Spectroscopic galaxy clustering --> `GCspectro`
* Dictionary with redshift distribution for WL --> `nz_dic_WL`
* Dictionary with redshift distribution for GCphot --> `nz_dic_GCphot`
* Angular power spectrum for photometric galaxy clustering -->  `Cl_GCphot`
* Angular power spectrum for weak lensing --> `Cl_WL`
* Angular power spectrum for galaxy galaxy lensing --> `Cl_GGL`
* Multipole power spectra --> `Pk_GCspectro`
* Galaxy clustering photometric window function  --> `window_GCphot_density`
* Weak lensing window function --> `window_WL_shear`
* RSD correction to the galaxy clustering photometric window function --> `window_GCphot_RSD`
* Magnification bias kernel --> `window_GCphot_mag` 
* Intrinsic alignment (IA) weight function --> `window_WL_IA`
* Weak lensing correlation functions --> `xi_WL_plus`, `xi_WL_minus`
* Galaxy galaxy lensing correlation function --> `xi_GGL`
* Photometric galaxy clustering correlation function --> `xi_GCphot`


##### Class names

* The convention is to use CamelCase ( e.g. `MyClass` and not `myClass` or `my_class`).

##### Class instances

* Class instances can generally be written as `my_class_inst = MyClass(...)`.

##### File (module) names/directory (subpackage) names

* Try to keep these simple but descriptive.

#### **Class and functions parameters**

* Generally, all calls to functions should be made with keyword arguments (e.g. `my_func(my_first_val=3, my_second_val=4)`).
* An exception is any CLOE function/class/method that takes a single argument or that only requires a single argument (i.e. the other keyword arguments have default values) a positional argument can be used. Similarly, for any standard 3rd-party functions positional arguments (e.g. `numpy.sqrt(x=9)` or  `numpy.sqrt(9)`) can be used. 
