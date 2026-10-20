# AstroCellLink 0.2.0 release notes (19 October 2026)

## Implemented

1. `astrocelllink prepare-h5`: chunkwise `::stride` copy of actual HDF5 dataset, records provenance. No registration/interpolation.
2. `astrocelllink compare-events`: streaming framewise contingency table, one-to-one greedy 3D IoU event agreement, aggregate foreground Jaccard, CSV and QC PNG.
3. `astrocelllink segment-structural`: structural-image watershed candidate instance masks + QC, with HDF5 streaming mean projection.
4. `astrocelllink evaluate-masks`: bipartite one-to-one instance matching at IoU threshold; output Precision/Recall/F1 only **relative to user-supplied reference**, not independently asserted ground truth.
5. `scripts/run_upstream_astrocast.sh`: separate official Python environment and recorded version/checksums/logs. It does not auto-install upstream and has **not been executed on user's remote ExVivo data**.
6. Previous `run`, `demo`, synthetic demo sample and 8 exploratory standalone scripts preserved.

## NOT implemented / NOT validated

- No direct result comparison from upstream astroCaST 2024 has been executed: the `ExVivoSuppRaw.h5` and official detector were only on user's computer.
- No actual deep-learning cell model trained; watershed masks are unvalidated proposals.
- No independent structural astrocyte identity labels or expert truth provided.
- No calcium wave significance or new clinical/biological conclusion.
- Pixel-stride `::2` loses information, and apparent detector agreement will depend on processing differences.
- Model outputs/metrics on synthetic tests should not be presented as real-world performance.

## Reproducibility notes

- Select a consistent physical field of view and pixel stride for detector agreement; reject mismatched shapes.
- Original candidate_map `281×251×251` came from `dff/ch0` strided every second pixel.
- Source movie `ExVivoSuppRaw.h5` `dff/ch0` was `281×502×502` on the user's machine.
- Upstream official repo: https://github.com/janreising/astroCAST; its 0.4.2 metadata requests Python >=3.9,<3.11.
- Compare detection maps as algorithms, not labels of cells or wave confirmations.
