user=$(shell whoami)
project_path=/mindhive/nklab3/users/$(user)/sts_communication
subs := 1 2 3 4 5 7 8 9 11 12 13 14 15 16
tom_subs := 1 3 4 5 7 8 9 11 12 13 14 15 16
eploc_subs := 2

# Dependencies
preprocess_path=$(project_path)/derivatives/fmriprep

# Steps to run
all: preprocess


# Preprocess EEG data for regression
preprocess: $(preprocess_path)/.preprocess_done
$(preprocess_path)/.preprocess_done: 
	for s in $(subs); do \
		sbatch $(project_path)/code/batch_preproc.sh -s $$s; \
	done
	touch $(preprocess_path)/.preprocess_done