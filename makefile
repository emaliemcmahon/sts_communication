user=$(shell whoami)
project_path=$(shell pwd)
subs := 01 02 03 04 05 07 08 09 11 12 13 14 15 16 18 19 20 21 22 23
tom_subs := 01 03 04 05 07 08 09 11 12 13 14 15 16 18 19 22 23
GROUP_RUNWISE_FLAGS ?=

# generate_block.py (../communication_exp/sts_communication_experiment/), which
# determines each subject's block order AND which specific video exemplars they
# see, seeds numpy with `int(time.time())` -- one-second resolution. Subjects
# whose run files were generated within the same second got byte-identical
# sessions (same block order, same exemplar videos, same within-block shuffle,
# same 1-back repeat trial) instead of the independent-per-participant
# randomization the preregistration specifies. Diffing the source run files
# (not just the BIDS events.tsv, which can diverge on a couple of runs) found
# these exact-duplicate groups -- likely subjects whose files were generated
# back-to-back in the same batch:
#   01+02, 03+04, 05+07, 08+09, 11+12, 13+14+15, 16+18+19, 20+21+22   (23 unique)
# So the nominal n=20 sample is really only ~9 independent stimulus draws;
# treating all 20 as exchangeable overstates independence for group stats.
# independent_subs keeps only the first subject from each duplicate group.
independent_subs := 01 03 05 08 11 13 16 20 23

# Full pipeline (run stages in order)
all: preprocess first_level_runwise runwise_response group_runwise first_level_models random_effects

# Preprocess fMRI data with fMRIPrep 24.1.1. Submits one SLURM job per
# subject (code/batch_preproc.sh); see code/run_fmriprep.sh for the
# unwrapped singularity call it mirrors.
preprocess:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_preproc.sh "$$s"; \
	done

# Cross-validated, run-wise first-level GLMs (defines fROIs). Within the
# communicate task, EVC/MT/FFA/EBA/fSTS are all defined in the same fold, so
# voxels top-N%-selected by more than one of those ROIs' parcels are resolved
# via winner-take-all (nilearn_glm_runwise.py --overlap_method, default
# winner_take_all): the contested voxel goes to whichever ROI has the higher
# z-stat in its own defining contrast there, matching video_sentence_analysis's
# build_froi_masks.py. This does NOT arbitrate across tasks (e.g. SI-STS vs.
# fSTS) since pointlight/tom/communicate use different LOO fold structures --
# doing so would leak a held-out run's data into another task's fold-specific
# ROI definition. Pass --overlap_method none to restore the old behavior
# (each ROI's raw top-N% selection, independent of what other ROIs claim).
# Submits one SLURM job per subject/task (code/batch_runwise_glm.sh).
first_level_runwise:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_runwise_glm.sh "$$s" pointlight; \
		sbatch $(project_path)/code/batch_runwise_glm.sh "$$s" communicate; \
	done
	for s in $(tom_subs); do \
		sbatch $(project_path)/code/batch_runwise_glm.sh "$$s" tom; \
	done

# Extract cross-validated ROI beta responses per subject. Submits one SLURM
# job per subject (code/batch_subject_runwise_response.sh).
runwise_response:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_subject_runwise_response.sh "$$s"; \
	done

# Group ROI summaries, paired t-tests, and plots. Submitted via SLURM
# (code/batch_group_runwise.sh), which always runs with --overwrite.
group_runwise:
	sbatch $(project_path)/code/batch_group_runwise.sh $(subs) $(GROUP_RUNWISE_FLAGS)

# Same as group_runwise, but restricted to independent_subs (see comment above)
# to check whether results hold with the stimulus-duplication confound removed.
# Writes to derivatives/GroupRunwiseResults_independent/ so it does not
# overwrite the full-sample group_runwise output.
group_runwise_independent:
	sbatch $(project_path)/code/batch_group_runwise.sh $(independent_subs) --out_tag independent $(GROUP_RUNWISE_FLAGS)

# Whole-brain first-level GLMs (one t-map per contrast, plus one condition-vs-
# baseline beta map per condition, per subject). Submits one SLURM job per
# subject/task (code/batch_nilearn_glm.sh).
first_level_models:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_nilearn_glm.sh "$$s" communicate; \
		sbatch $(project_path)/code/batch_nilearn_glm.sh "$$s" pointlight; \
	done
	for s in $(tom_subs); do \
		sbatch $(project_path)/code/batch_nilearn_glm.sh "$$s" tom; \
	done

# Contrasts for the group-level (whole-brain) random-effects analyses.
# Columns are (condition_one, condition_two, task) triples.
C1S := belief interact \
face_third face_third+face_first+com_phy+com_ind \
face_third+face_first com_phy+com_ind \
com_phy com_ind face_third face_first \
face_third+face_noncom body face_first
C2S := photo noninteract \
object face_noncom+phy+ind \
face_noncom phy+ind \
phy ind face_noncom face_noncom \
object object face_third
TASKS := tom pointlight \
communicate communicate \
communicate communicate \
communicate communicate communicate communicate \
communicate communicate communicate

# Whole-brain second-level (TFCE + FWER via non-parametric inference) and
# associated surface plots for each contrast. Submits one SLURM job per
# contrast (code/batch_random_effects.sh), which runs both steps in sequence.
random_effects:
	@echo "Submitting random-effects analyses..."
	$(eval LENGTH := $(words $(C1S)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(C1S))) \
		$(eval C2 := $(word $(i),$(C2S))) \
		$(eval TASK := $(word $(i),$(TASKS))) \
		sbatch $(project_path)/code/batch_random_effects.sh $(C1) $(C2) $(TASK); \
		echo $(C1) $(C2) $(TASK); \
	)

# Voxel-level overlap (Dice + individual-subject surfaces) between dyad and
# face-perception contrasts within the STS parcel. Submits one SLURM job per
# subject (code/batch_voxel_overlap.sh); run voxel_overlap_group once all
# jobs finish.
voxel_overlap:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_voxel_overlap.sh "$$s"; \
	done

# Aggregate per-subject Dice coefficients into a group table + summary plot
voxel_overlap_group:
	python $(project_path)/code/voxel_overlap_group.py

# Split-half (odd vs even runs) reliability of the pointlight interact-noninteract
# contrast within the STS parcel, from NilearnGLMRunwise per-run outputs.
# Submits one SLURM job per subject. Group-level aggregation happens together
# with voxel_overlap_communicate_pointlight_group below (split-half serves as
# that comparison's noise ceiling), so there is no separate group target here.
voxel_overlap_pointlight_splithalf:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_voxel_overlap_pointlight_splithalf.sh "$$s"; \
	done

# Voxel-level overlap between each communicate contrast and the pointlight
# interact-noninteract contrast within the STS parcel, from NilearnGLM
# whole-brain outputs. Submits one SLURM job per subject; run the _group
# target once both this and voxel_overlap_pointlight_splithalf have finished
# for all subjects.
voxel_overlap_communicate_pointlight:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_voxel_overlap_communicate_pointlight.sh "$$s"; \
	done

# Whole-brain searchlight version of the decoding index (see
# decoding_index.py for the ROI version). Submits one SLURM job per subject
# (code/batch_searchlight_decoding_index.sh); no group-level aggregation yet.
searchlight_decoding_index:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_searchlight_decoding_index.sh "$$s"; \
	done

# Aggregates BOTH voxel_overlap_communicate_pointlight and
# voxel_overlap_pointlight_splithalf per-subject results: group Dice tables
# for each, plus a combined summary plot with the split-half reliability
# overlaid as a noise-ceiling reference on the communicate-vs-pointlight curves.
voxel_overlap_communicate_pointlight_group:
	python $(project_path)/code/voxel_overlap_communicate_pointlight_group.py
