user=$(shell whoami)
project_path=/orcd/data/ngk/001/users/$(user)/sts_communication
subs := 01 02 03 04 05 07 08 09 11 12 13 14 15 16 18 19 20 21 22 23
tom_subs := 01 03 04 05 07 08 09 11 12 13 14 15 16 18 19 22 23

# Steps to run
all: preprocess rois first_level_runwise runwise_response group_runwise first_level_models random_effects surface_plots

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
	sbatch $(project_path)/code/batch_group_runwise.sh $(subs)

first_level_models: 
	for s in $(subs); do \
# 		sbatch $(project_path)/code/batch_nilearn_glm.sh "$$s" pointlight; \
		sbatch $(project_path)/code/batch_nilearn_glm.sh "$$s" communicate; \
	done
# 	for s in $(tom_subs); do \
# 		sbatch $(project_path)/code/batch_nilearn_glm.sh "$$s" tom; \
# 	done

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
random_effects:
	@echo "Submitting random effects jobs..."
	$(eval LENGTH := $(words $(C1S)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(C1S))) \
		$(eval C2 := $(word $(i),$(C2S))) \
		$(eval TASK := $(word $(i),$(TASKS))) \
		sbatch $(project_path)/code/batch_random_effects.sh $(C1) $(C2) $(TASK); \
		echo $(C1) $(C2) $(TASK); \
	)
