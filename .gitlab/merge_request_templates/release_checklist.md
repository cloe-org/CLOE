## Release Version

Release v`<VERSION NUMBER>` of CLOE.

## MR Checklist for Releases

Reviewers, please check the following points before merging.

- [ ] MR targets the `master` branch from source `develop` branch
- [ ] The `version_info` in `cloe/info.py` matches the release version above
- [ ] The values of `version` and `release` in `docs/source/conf.py` match
      the release version above
- [ ] All dependencies in `environment.yml` are pinned (i.e. with `==`) to the
      appropriate versions
- [ ] The Docker image is up to date with the current `environment.yml`
- [ ] If necessary, the notebooks in `cloe/tests/test_tools/notebooks` have
      been run to generate the pickle files required for the unit tests
- [ ] CI tests have run and passed for the latest commit on the `develop`
      branch
- [ ] Manual `Check_HTML` job run and API documentation looks good
- [ ] Manual `Profiling` job run and no unexpected performances issues found
- [ ] Manual `Verification` job run and results match expectations
- [ ] Leads have approved the release
- [ ] The link to the landing page has been updated to that of the `master` 
      branch in the Wiki page of the corresponding version
