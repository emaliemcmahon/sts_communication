user=$(shell whoami)
project_path=/orcd/data/ngk/001/users/$(user)/sts_communication
subs := 01 02 03 04 05 07 08 09 11 12 13 14 15 16 18 19 20 21 22 23
tom_subs := 01 03 04 05 07 08 09 11 12 13 14 15 16 18 19 22 23

# Steps to run
all: preprocess rois first_level_runwise runwise_response group_runwise communicate_random_effects loc_random_effects plot_surfaces

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

# Define contrast arrays (space-separated lists in Make)
COM1S := face_third+face_first+com_phy+com_ind \
face_third+face_first com_phy+com_ind \
com_phy com_ind face_third face_first \
face_third+face_noncom body face_first
COM2S := face_noncom+phy+ind \
face_noncom phy+ind \
phy ind face_noncom face_noncom \
object object face_third
communicate_random_effects:
	@echo "Submitting random effects jobs..."
	$(eval LENGTH := $(words $(COM1S)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(COM1S))) \
		$(eval C2 := $(word $(i),$(COM2S))) \
		sbatch $(project_path)/code/batch_random_effects.sh $(C1) $(C2) communicate; \
		echo $(C1) $(C2); \
	)

C1S := belief interact
C2S := photo noninteract
TASKS := tom pointlight
loc_random_effects:
	@echo "Submitting random effects jobs..."
	$(eval LENGTH := $(words $(C1S)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(C1S))) \
		$(eval C2 := $(word $(i),$(C2S))) \
		$(eval TASK := $(word $(i),$(TASKS))) \
		sbatch $(project_path)/code/batch_random_effects.sh $(C1) $(C2) $(TASK); \
		echo $(C1) $(C2) $(TASK); \
	)


CONS1 := face_third+face_first+com_phy+com_ind \
face_third+face_first com_phy+com_ind \
com_phy com_ind face_third face_first \
face_third+face_noncom body face_first \
belief interact
CONS2 := face_noncom+phy+ind \
face_noncom phy+ind \
phy ind face_noncom face_noncom \
object object face_third \
photo noninteract
plot_surfaces:
	@echo "Submitting surface plotting jobs..."
	$(eval LENGTH := $(words $(CONS1)))
	$(foreach i, $(shell seq 1 $(LENGTH)), \
		$(eval C1 := $(word $(i),$(CONS1))) \
		$(eval C2 := $(word $(i),$(CONS2))) \
		sbatch $(project_path)/code/batch_plot_surfaces.sh $(C1) $(C2); \
		echo $(C1) $(C2); \
	)