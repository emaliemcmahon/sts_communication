# Methods

## Participants

Twenty-one adults were scanned. One participant (sub-10) was excluded prior to analysis because they fell asleep during the session, yielding a final sample of *N* = 20 participants (12 female, 8 male; mean age = 25.4 years, SD = 5.0, range = 19–37) for all analyses using the communicative-interaction task and the point-light social-interaction localizer. The theory-of-mind (ToM) localizer was acquired in 17 of these 20 participants; the remaining three completed the session without the ToM runs. All participants gave written informed consent and were compensated for their participation. All procedures were approved by the MIT Committee on the Use of Humans as Experimental Subjects, and the study was preregistered on the Open Science Framework (https://osf.io/ermyv/).

## Stimuli and Design

Each participant completed three tasks: a novel communicative-interaction experiment ("communicate"), a point-light social-interaction localizer ("pointlight"), and a false-belief/false-photograph theory-of-mind localizer ("tom"). All stimuli were presented in a blocked design using Psychtoolbox (v3.0.19–3.0.20; RRID:SCR_002881) in MATLAB.

**Communicative-interaction task.** Nine runs of the "communicate" task were acquired per participant. Each run comprised 18 blocks of 12 s duration, drawn from nine video conditions (two blocks per condition per run). The conditions were designed to factorially dissociate face-based (single-agent) communication, dyadic communication, and non-communicative social interaction:

- *com_phy*: two people engaged in verbal communication paired with matched physical (non-communicative) social interactions,
- *com_ind*: two people engaged in verbal communication paired with matched independent (non-interactive) actions,
- *phy*: two people performing social but non-communicative interactions (e.g., dancing, boxing),
- *ind*: two people performing common actions independently of one another,
- *face_first*: a single face speaking directly to the viewer (first-person address),
- *face_third*: a single face speaking to an unseen interlocutor off-screen (third-person address),
- *face_noncom*: a single face performing a non-communicative action (e.g., eating, drinking),
- *body*: moving individual bodies (from Julian et al., 2012, *NeuroImage*),
- *object*: moving objects (from Julian et al., 2012, *NeuroImage*).

Participants performed a one-back cover task, pressing a button whenever a video clip repeated within a block, to maintain attention.

**Point-light social-interaction localizer.** Four runs of the pointlight task were acquired per participant, based on the paradigm of Isik and colleagues. Each run contained 16-s blocks alternating between two conditions: two point-light figures engaged in a social interaction (*interact*) versus two point-light figures each performing independent biological motion (*noninteract*).

**Theory-of-mind localizer.** Two runs of the classic false-belief / false-photograph localizer (Dodell-Feder et al., 2011; Saxe & Kanwisher, 2003) were acquired in 17 of 20 participants. Each 14-s block presented a short story followed by a true/false question about either a character's false belief (*belief*) or an outdated photograph or map (*photo*).

## MRI Acquisition

Data were acquired on a 3 T Siemens MAGNETOM Prisma scanner at the Athinoula A. Martinos Imaging Center at MIT, using a 32-channel head coil. A whole-brain T1-weighted MPRAGE anatomical image was acquired for each participant at 1 mm isotropic resolution. Functional images were acquired with a T2\*-weighted multiband EPI sequence with the following parameters: TR = 2000 ms, TE = 30 ms, flip angle = 90°, voxel size = 2 mm isotropic, slice thickness = 2 mm with 10% gap (spacing between slices = 2.2 mm), multiband acceleration factor = 2, phase-encoding direction = posterior→anterior (j−), partial Fourier = 7/8. Slice-timing information was recorded in the BIDS sidecar files.

## Data Organization and Anonymization

DICOM images were converted to NIfTI and organized in BIDS format (v1.8.0) using `dcm2bids`. Anatomical images were defaced with `pydeface` (with FSL 6.0), and defacing was verified by visual inspection prior to release. Aborted or incomplete runs were removed from the BIDS tree before preprocessing.

## Preprocessing

Anatomical and functional data were preprocessed using fMRIPrep v24.1.1 (Esteban et al., 2019, *Nature Methods*; RRID:SCR_016216), run inside an Apptainer/Singularity container. Preprocessing was performed with the `--output-space T1w MNI152NLin2009cAsym` flag, so that outputs were resampled both into each participant's native T1w space and into MNI152NLin2009cAsym standard space. All subsequent GLM analyses used the MNI-space preprocessed BOLD (`desc-preproc`) images. Because fMRIPrep applies slice-time correction to the middle temporal volume of each acquisition, event onsets from the BIDS events files were shifted forward by 1 s (TR/2) before entering the first-level GLM, consistent with published guidance (https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/).

For each task, a common analysis mask was computed as the intersection of the fMRIPrep brain masks across all of that task's runs.

## First-level General Linear Models

All first-level GLMs were fit using Nilearn's `first_level_from_bids` (Abraham et al., 2014). Regressors were built from the BIDS events files and convolved with the SPM canonical hemodynamic response function. Data were spatially smoothed with a 5 mm FWHM Gaussian kernel prior to model fitting. Six basic head-motion parameters (three translations, three rotations) estimated by fMRIPrep were included as nuisance regressors (`confounds_strategy=("motion",)`, `confounds_motion="basic"`). Nilearn's default cosine-basis high-pass filter was used to remove low-frequency drift. Slice-timing–corrected event onsets (shifted by TR/2, as above) were used.

Two families of first-level analyses were run.

### Whole-brain first-level GLMs

For each participant and each task, a single GLM was fit jointly across all runs of that task. For the communicate task, twelve contrasts of interest were computed as *t*-statistic maps in MNI152NLin2009cAsym space:

- `face_third − object`
- `face_third + face_first + com_phy + com_ind − face_noncom + phy + ind` (communication vs. matched non-communicative controls)
- `face_third + face_first − face_noncom` (face-based communication vs. non-communicative face)
- `com_phy + com_ind − phy + ind` (dyadic communication vs. matched non-communicative dyads)
- `com_phy − phy` and `com_ind − ind` (dyadic communication vs. its matched physical or independent control)
- `face_third − face_noncom` and `face_first − face_noncom`
- `face_third + face_noncom − object` (face-selective contrast)
- `face_first − face_third`
- `body − object` (body-selective contrast)
- an all-conditions-vs.-baseline contrast used to define visually responsive regions (EVC and MT).

For the pointlight task, we computed `interact − noninteract`; for the ToM task, `belief − photo`. When multiple conditions were combined on either side of a contrast (e.g., `A + B`), the corresponding regressors were averaged (i.e., each was given weight 1/*k*, where *k* is the number of terms).

For each contrast, we also saved a positive-only single-subject "parcel" mask, obtained by thresholding the *t*-map at an uncorrected height threshold of *p* < 0.05 (one-sided, positive). These maps were used to compute cross-participant probability maps of contrast selectivity within the anatomical STS parcel (see *Group Parcel Probability*, below).

### Run-wise cross-validated GLMs for fROI analysis

To provide unbiased estimates of condition responses inside functionally defined ROIs (fROIs), we ran a separate set of run-wise GLMs following a leave-runs-out cross-validation scheme. For each task, the available runs were split into *k* equal-sized held-out groups (*k* = 9 for communicate, 4 for pointlight, 2 for ToM). For each split *i* = 1 … *k*, the GLM was fit on the runs not in group *i* and used to compute the fROI-defining *z*-statistic contrasts; the same model was then refit on the held-out runs to estimate condition-specific beta responses. This procedure ensures full statistical independence between the data used to define each fROI and the data used to estimate its response profile (Kriegeskorte et al., 2009).

## Anatomical Parcels

Anatomical/functional parcels used to constrain fROI definition were assembled from published atlases and resampled to the target MNI152NLin2009cAsym EPI grid using nearest-neighbor interpolation:

- **FFA, EBA, and posterior face-selective STS (fSTS)**: group parcels from the Kanwisher-lab "DYLOC" set (Julian et al., 2012, *NeuroImage*).
- **Temporo-parietal junction (TPJ)**: group parcel from the Saxe-lab ToM localizer (Dufour et al., 2013).
- **Early visual cortex (EVC)** (union of V1–V4) and **motion-selective area MT**: from the probabilistic visuotopic atlas of Wang et al. (Kastner lab; Wang et al., 2015).
- **Anatomical STS parcel**: from Ben Deen's anatomical STS atlas (Deen et al., 2015, *Cerebral Cortex*).

All parcels were used bilaterally.

## Functional ROIs

For each participant, hemisphere, and cross-validation split, an fROI was defined as the top-responding voxels within a given anatomical parcel for the corresponding functional contrast. Specifically, we selected the voxels with the highest positive contrast values inside the parcel until a target proportion of the parcel volume was reached (10% for FFA, EBA, fSTS, MT, and TPJ; 5% for EVC and for all STS-based communication/interaction fROIs). The functional contrasts used to define each fROI were:

- **SI-STS** (social-interaction STS): `interact − noninteract`, inside anatomical STS (from the pointlight task).
- **TPJ**: `belief − photo` (from the ToM task).
- **com-STS** (communication STS): `(face_third + face_first + com_phy + com_ind) − (face_noncom + phy + ind)`, inside anatomical STS.
- **facecom-STS** (face-communication STS): `(face_third + face_first) − face_noncom`, inside anatomical STS.
- **dyadcom-STS** (dyadic-communication STS): `(com_phy + com_ind) − (phy + ind)`, inside anatomical STS.
- **comphy-STS** and **comind-STS**: `com_phy − phy` and `com_ind − ind` respectively, inside anatomical STS.
- **phy-STS**: `phy − ind`, inside anatomical STS.
- **fSTS** (face-selective STS): `(face_third + face_noncom) − object`, inside the DYLOC STS parcel.
- **FFA**: `(face_third + face_noncom) − object`, inside the DYLOC FFA parcel.
- **EBA**: `body − object`, inside the DYLOC EBA parcel.
- **EVC** and **MT**: all-conditions-vs.-baseline contrast, inside the Wang et al. atlas parcels.

All communication-related fROIs, together with fSTS, were defined from the communicate task; SI-STS was defined from the pointlight task; and TPJ was defined from the ToM task.

## ROI Response Extraction and Statistics

For each participant, hemisphere, and ROI, we extracted the mean beta value across ROI voxels for every condition of every task, using only cross-validated data. When a condition came from the task used to define the ROI, responses were averaged across held-out runs of that task's *k*-fold cross-validation (so definition and response data were always disjoint). When a condition came from a different task than the one used to define the ROI, responses were computed in that task's runs directly (which are, by construction, independent of the definition data).

Subject-level responses were averaged across runs, and group-level statistics were computed with paired, one-tailed *t*-tests across participants (using `scipy.stats.ttest_rel`, `alternative="greater"`) on the following pre-registered contrasts of interest:

- `face_first − face_noncom`, `face_third − face_noncom` (face-communication effects),
- `com_ind − ind`, `com_phy − phy` (dyadic-communication effects),
- `phy − ind` (social-interaction effect independent of communication),
- `interact − noninteract` (point-light social-interaction effect),
- `belief − photo` (theory-of-mind effect).

*p*-values are reported uncorrected; significance thresholds are indicated as \**p* < 0.05, \*\**p* < 0.01, \*\*\**p* < 0.001.

## Whole-brain Group Analyses

For each first-level contrast of interest, second-level (group) inference was performed on the subject-wise contrast maps using Nilearn's non-parametric inference (`nilearn.glm.second_level.non_parametric_inference`). We used a one-sample intercept-only design with a one-sided test (positive direction), 10,000 sign-flip permutations, threshold-free cluster enhancement (TFCE; Smith & Nichols, 2009), and an additional 8 mm FWHM Gaussian smoothing kernel applied at the group level. We report family-wise-error-corrected –log₁₀(*p*) maps derived from the maximum-TFCE null distribution.

To characterize the spatial consistency of communication-selective, face-selective, body-selective, social-interaction, and theory-of-mind responses across participants, we additionally computed group probability maps of each single-subject positive contrast mask (uncorrected *p* < 0.05, positive tail) restricted to the union of the left and right anatomical STS parcels.

## Visualization

Group and subject-level statistical maps were projected onto the FreeSurfer `fsaverage` inflated cortical surface using Nilearn's `SurfaceImage.from_volume` (depth-weighted sampling between the white-matter and pial surfaces) and rendered with Nilearn's surface-plotting utilities and PyCortex.

## Software and Reproducibility

All analyses were implemented in Python 3, using fMRIPrep 24.1.1 (Apptainer/Singularity), Nilearn (Abraham et al., 2014), NiBabel, NumPy, SciPy, pandas, and Matplotlib/Seaborn; conda environments are versioned in `code/envs/`. Analysis code, BIDS-formatted (defaced) data, and derivatives are available at https://osf.io/79632/, and the full analysis pipeline is orchestrated by the top-level `makefile` (`preprocess → first_level_runwise → runwise_response → group_runwise → first_level_models → random_effects → surface_plots`).
