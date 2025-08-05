user=$(shell whoami)
project_path=/mindhive/nklab3/users/$(user)/sts_communication
subs := 01 02 03 04 05 07 08 09 11 12 13 14 15 16

# Steps to run
all: preprocess rois first_level_runwise random_effects

# Preprocess fMRI data with fRMIPrep
preprocess:
	for s in $(subs); do \
		echo "Submitting job for subject $$s"; \
		sbatch $(project_path)/code/batch_preproc.sh "$$s"; \
	done

# Run the first level analysis for localizer tasks
tom_subs := 01 03 04 05 07 08 09 11 12 13 14 15 16
eploc_subs := 02
rois: 
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_frois.sh "$$s" pointlight; \
		sbatch $(project_path)/code/batch_frois.sh "$$s" communicate; \
	done
	for s in $(tom_subs); do \
		sbatch $(project_path)/code/batch_frois.sh "$$s" tom; \
	done
	for s in $(eploc_subs); do \
		sbatch $(project_path)/code/batch_frois.sh "$$s" eploc; \
	done

# Define the runwise ROIs and responses
first_level_runwise: 
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_runwise_glm.sh "$$s"; \
	done

# Define contrast arrays (space-separated lists in Make)
C1S := com_phy com_ind face_first face_first face_third 0.5*face_third+0.5*face_first 0.5*com_phy+0.5*com_ind 0.5*face_third+0.5*face_noncom body phy 0.33*phy+0.33*com_phy+0.33*com_ind
C2S := phy ind face_noncom face_third face_noncom face_noncom 0.5*phy+0.5*ind object object ind ind
# Task to submit all random effects jobs
random_effects:
	@echo "Submitting random effects jobs..."
	$(eval LENGTH := $(words $(C1S)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(C1S))) \
		$(eval C2 := $(word $(i),$(C2S))) \
		sbatch batch_random_effects.sh $(C1) $(C2); \
	)