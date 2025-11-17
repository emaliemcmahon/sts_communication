#!/bin/bash -l

#SBATCH --job-name=random_effects
#SBATCH --partition=mit_normal
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%A.out

# c1s=(com_phy com_ind face_first face_first face_third face_first+face_third com_phy+com_ind face_noncom+face_third body)
# c2s=(phy ind face_noncom face_third face_noncom face_noncom phy+ind object object)
# length=${#c1s[@]}
# for ((i=0; i<length; i++)); do sbatch batch_random_effects.sh ${c1s[$i]} ${c2s[$i]}; done

conda activate nilearn

c1=$1
c2=$2
task=${3:-"communicate"}


if [ "$task" = "tom" ]; then
    subs=(1 3 4 5 7 8 9 11 12 13 14 15 16)
else
    subs=(1 2 3 4 5 7 8 9 11 12 13 14 15 16)
fi


echo "$c1 $c2"
echo "${subs[@]}"

python code/group_random_effects.py "${subs[@]}" \
    -c1 $c1 -c2 $c2 -t $task