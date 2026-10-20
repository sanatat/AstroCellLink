# Release notes — 0.3.0 (experimental)

This release combines the stable 0.2 prototype functionality with experimental fluorescence-guided spatiotemporal event growth.

- Added `astrocelllink hysteresis`, including seed-preserving voxel labeling, temporal persistence, spatial/temporal corridor constraints and ambiguous-source diagnostics.
- Kept `demo`, `run`, `prepare-h5`, `compare-events`, `segment-structural` and `evaluate-masks`.
- Documented the ExVivo event-detector agreement results separately from biological validation.
- Added public-ready README, CI and a GitHub-safe `.gitignore`.

**Status:** exploratory; 30 synthetic / interface tests pass in the current development environment. The v0.3 algorithm has not yet been run/validated against independently labeled real microscopy recordings. See README and QUICKSTART_v0.3_FA.md.
