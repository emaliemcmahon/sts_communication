user=$(shell whoami)
project_path=/orcd/data/ngk/001/users/$(user)/sts_communication
subs := 01 02 03 04 05 07 08 09 11 12 13 14 15 16
tom_subs := 01 03 04 05 07 08 09 11 12 13 14 15 16


# Steps to run
all: preprocess rois first_level_runwise runwise_response group_runwise random_effects

# Preprocess fMRI data with fRMIPrep
preprocess:
	for s in $(subs); do \
		echo "Submitting job for subject $$s"; \
		sbatch $(project_path)/code/batch_preproc.sh "$$s"; \
	done

# Define the runwise ROIs and responses
first_level_runwise: 
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_runwise_glm.sh "$$s" pointlight; \
		sbatch $(project_path)/code/batch_runwise_glm.sh "$$s" communicate; \
	done
	for s in $(tom_subs); do \
		sbatch $(project_path)/code/batch_runwise_glm.sh "$$s" tom; \
	done

runwise_response:
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_subject_runwise_response.sh "$$s"; \
	done

group_runwise:
	sbatch $(project_path)/code/batch_group_runwise.sh

# Define contrast arrays (space-separated lists in Make)
C1S := 0.5*face_third+0.5*face_first 0.5*com_phy+0.5*com_ind 0.25*face_third+0.25*face_first+0.25*com_phy+0.25*com_ind
C2S := face_noncom 0.5*phy+0.5*ind 0.33*face_noncom+0.33*phy+0.34*ind
# Task to submit all random effects jobs
random_effects:
	@echo "Submitting random effects jobs..."
	$(eval LENGTH := $(words $(C1S)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(C1S))) \
		$(eval C2 := $(word $(i),$(C2S))) \
		sbatch $(project_path)/code/batch_random_effects.sh $(C1) $(C2); \
	)


