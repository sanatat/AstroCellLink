# Research Project — Cell-Aware Astrocyte Calcium Signaling Analysis

**Working title:** Cell-Aware Tracking and Uncertainty-Aware Attribution of Spatiotemporal Calcium Events in Astrocyte Imaging

## 1. Gap grounded in the 2024 astroCaST paper

The paper builds an efficient open-source Python workflow for detection and classification of astrocytic calcium events, but its developers explicitly state that events cannot reliably be assigned to **individual cells** (pp. 17–18). Cell overlap and morphological changes present fundamental challenges. Our code implements a useful baseline of soma mask tracking + geometric event attribution, but the gap **is not yet scientifically solved**.

## 2. Current baseline (implemented)

1. Receive upstream `event_map.tiff` (T×Y×X integer labels).
2. Receive expert masks or a secondary structural fluorescence channel. Watershed is an optional baseline, not a learned detector.
3. Link soma masks with spatial IoU and Hungarian matching.
4. For each event, aggregate overlap with each tracked cell over its actual frames.
5. Reject unsupported/ambiguous event-to-cell matches instead of making an unqualified decision.
6. Save CSV labels, tracks, QC image and synthetic-test statistics.

## 3. Proposed research phase: learned segmentation

**Not yet implemented / not validated.** Train or fine-tune Cellpose, StarDist, U-Net, or instance segmentation transformer using expertly traced astrocyte masks (soma and processes separately); evaluate against fluorescence input, tissue depth, domain shift, photobleaching, animal/sample ID, optical setups and acquisition rates. Spatial alignment with upstream astroCaST must be checked. Process masks may not be inferable from available recordings.

## 4. Proposed research phase: spatiotemporal cell identity

Model cell associations with temporal image embeddings + flow/overlap/graph assignment and explicit uncertainty. Track soma centroids and branches, model merge/split hypotheses without overwriting uncertain cases, and investigate 3D z-stacks if available. Establish where individual-cell attribution is fundamentally underdetermined from 2D images.

## 5. Proposed research phase: calcium-event attribution

Compare current geometric baseline against:
- pixelwise probabilities and co-registered structural segmentation;
- time-resolved event signature features and cell-level trace coherence;
- graphical energy or assignment models allowing `unknown` and `multi-cell` labels;
- learned fusion conditioned on mask/registration confidence.

Use calibrated confidence obtained from *held-out annotated samples* rather than geometric purity mislabeled as a calibrated probability.

## 6. Ground-truth and experimental design

- Assemble at least two imaging sources: stable astrocytic cell-identity annotations (structural marker) and event IDs/time courses. Acquire written appropriate dataset rights.
- Blind expert labeling by at least two observers; record disagreement and unassignable overlaps.
- Split train/validation/test by **animal / recording / experimental session**, not frames from the same video; avoid train/test leakage.
- Metrics: soma instance mAP/IoU, tracking IDF1 and ID switches, assigned-event precision/recall/F1, unknown rejection accuracy, calibrated ECE/Brier (if probabilistic), breakdown by cell overlap and imaging depth.
- Include controls with perturbed masks, shuffled cell identities, low signal-to-noise, shifts, bleaching, missing frames and adjacent correlated events.
- Statistical reporting: bootstrap CIs by animal/session, comparisons to fixed-mask and nearest-centroid baselines, ablation of time-dependent tracking, overlap scoring and uncertainty rules.

## 7. Completion criteria before claiming a new scientific solution

- Real-data segmented masks and ground-truth cell-event annotations exist.
- Reproducible independent holdout benchmarks against soma-overlap and nearest-centroid baselines.
- Error analysis on intertwined processes, uncertain cases and multiple acquisitions.
- Training scripts, model weights with licenses, data access protocol, model card and reproducible environment.
- Paper methods clearly mark which components extend AQuA/astroCaST and which are new.

## 8. Suggested PhD project milestones

**Months 1–3:** Dataset review, annotations, registration QC, pilot benchmarks and baseline replication.
**Months 4–8:** Astrocyte-specific segmentation and robust tracking, expert evaluation.
**Months 9–14:** Spatiotemporal event-to-cell attribution, uncertainty calibration, ablation.
**Months 15–18:** Independent lab/recording validation, documentation, preprint if results warrant.

These milestones are planning estimates, not evidence that the projected research has already occurred.
