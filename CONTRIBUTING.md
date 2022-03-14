# Euclid IST:L Contribution Guidelines

> Authors: Samuel Farrens, Dida Markovic  
> Last Updated: 05/06/2020

## Contents

1. [Introduction](#Introduction)
1. [Git Development Guidelines](#Git-Development-Guidelines)
1. [IST:NL Development Guidelines](#IST:NL Development Guidelines)

## Introduction

These guidelines aim to help you contribute to the development of the the Euclid IST:L package. Before attempting to make any changes to the code please do the following:

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
