# ExVivo detector-agreement notes (October 2026)

This page documents **observed exploratory outputs**, not ground-truth accuracy or formal replication. Results were produced locally on the astroCaST example ExVivo file; raw HDF5 data and analysis output folders are not redistributed here.

### Conditions

- Calcium input: `dff/ch0` from public `ExVivoSuppRaw.h5` after spatial stride 2; resulting HDF5 dataset shape `(281, 251, 251)`, `float32`.
- Independent candidate screen: `scripts/exploratory/exvivo_event_screen.py` with its screening settings. Label map size `(281, 251, 251)`; 335 retained candidates.
- Comparator: **official** astroCaST 0.4.2 `detect-events` (default detector parameters; `--loc dff/ch0 --parallel False --on-disk True`). 425 event labels in log.
- Comparative matching: one-to-one greedy 3D IoU ≥ 0.10, matching voxel-aligned label maps.

### Global results

- Candidate events: **335**
- astroCaST events: **425**
- Matched event pairs: **155**
- Candidate events without one-to-one match: **180**
- astroCaST events without one-to-one match: **270**
- Foreground Jaccard: **0.1680417069**
- Mean IoU for paired events: **0.2902398516**
- Foreground intersection: **72,653** voxels; union: **432,351** voxels

### Case study: candidate 194 vs. astroCaST event 212

- Candidate 194: frames 170–196, 3,692 labeled voxels.
- astroCaST 212: frames 137–204, 13,478 labeled voxels.
- Shared: **3,686** voxels; pairwise IoU **0.273361**.
- Seven candidate IDs overlapped official event 212 (194, 164, 183, 178, 190, 215, 155), jointly covering 4,489 / 13,478 voxels (**33.31%**).
- The result is consistent with boundary and/or split/merge differences but does not resolve which labels better reflect biological events.

### Boundary probes (agreement diagnostics, not accuracy)

Geometric dilation of candidate foreground, per frame, yielded Jaccard values of 0.1680 (radius 0), 0.2651 (1), 0.3263 (2), 0.3412 (3), and 0.2952 (5). Extra voxels outside astroCaST foreground increased markedly with radius: 5,255 → 103,014 at radius 3. Basic fluorescence-gated expansion did not outperform radius-1 geometric dilation on the observed agreement/cost trade-off. These probes are **diagnostic only**; selecting a radius to maximize agreement with astroCaST would not constitute independent validation.

### Caveats

- The event screen and reference use different segmentation rules and may disagree on split/merge structure.
- No expert annotated biological event or cell instance masks were used.
- The reference detector is not biological ground truth; Jaccard/IoU are agreement metrics.
- Candidate onset direction and spatial-null controls remain exploratory; no wave propagation velocity or validated p-value was established.
- Neither v0.3 performance on this dataset nor cell attribution on real annotated data has been measured.
