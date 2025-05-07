#!/bin/bash -l

#SBATCH
#SBATCH --job-name=random_effects
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=24
#SBATCH --exclude=node064

# c1s=(com_phy phy com_ind face_first face_first face_third)
# c2s=(phy com_phy ind face_noncom face_third face_noncom)
# length=${#c1s[@]}
# for ((i=0; i<length; i++)); do sbatch run_random_effects.sh "${c1s[$i]} ${c2s[$i]}"; done

c1=$1
c2=$2

subs=(1 2 3 4 5 7 8 9 11 12 13 14)

conda activate nilearn

python group_random_effects.py "${subs[@]}" \
    -c1 $c1 -c2 $c2

