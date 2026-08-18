user=$(shell whoami)
project_path=$(shell pwd)
subs := 01 02 03 04 05 07 08 09 11 12 13 14 15 16 18 19 20 21 22 23
tom_subs := 01 03 04 05 07 08 09 11 12 13 14 15 16 18 19 22 23
GROUP_RUNWISE_FLAGS ?=

# Full pipeline (run stages in order)
all: preprocess first_level_runwise runwise_response group_runwise first_level_models random_effects

# Preprocess fMRI data with fMRIPrep 24.1.1 (see code/run_fmriprep.sh for the
# exact singularity call). Wrap this in your local scheduler as appropriate.
preprocess:
	for s in $(subs); do \
		bash $(project_path)/code/run_fmriprep.sh "$$s"; \
	done

# Cross-validated, run-wise first-level GLMs (defines fROIs)
first_level_runwise:
	for s in $(subs); do \
		python $(project_path)/code/nilearn_glm_runwise.py -s "$$s" -t pointlight; \
		python $(project_path)/code/nilearn_glm_runwise.py -s "$$s" -t communicate; \
	done
	for s in $(tom_subs); do \
		python $(project_path)/code/nilearn_glm_runwise.py -s "$$s" -t tom; \
	done

# Extract cross-validated ROI beta responses per subject
runwise_response:
	for s in $(subs); do \
		python $(project_path)/code/runwise_response.py -s "$$s"; \
	done

# Group ROI summaries, paired t-tests, and plots
group_runwise:
	python $(project_path)/code/group_runwise_results.py $(subs) $(GROUP_RUNWISE_FLAGS)

# Whole-brain first-level GLMs (one t-map per contrast per subject)
first_level_models:
	for s in $(subs); do \
		python $(project_path)/code/nilearn_glm.py -s "$$s" -t communicate; \
		python $(project_path)/code/nilearn_glm.py -s "$$s" -t pointlight; \
	done
	for s in $(tom_subs); do \
		python $(project_path)/code/nilearn_glm.py -s "$$s" -t tom; \
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
# associated surface plots for each contrast.
random_effects:
	@echo "Running random-effects analyses..."
	$(eval LENGTH := $(words $(C1S)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(C1S))) \
		$(eval C2 := $(word $(i),$(C2S))) \
		$(eval TASK := $(word $(i),$(TASKS))) \
		python $(project_path)/code/group_random_effects.py -c1 $(C1) -c2 $(C2) -t $(TASK); \
		python $(project_path)/code/plot_surfaces.py       -c1 $(C1) -c2 $(C2) -t $(TASK); \
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

# Aggregates BOTH voxel_overlap_communicate_pointlight and
# voxel_overlap_pointlight_splithalf per-subject results: group Dice tables
# for each, plus a combined summary plot with the split-half reliability
# overlaid as a noise-ceiling reference on the communicate-vs-pointlight curves.
voxel_overlap_communicate_pointlight_group:
	python $(project_path)/code/voxel_overlap_communicate_pointlight_group.py
