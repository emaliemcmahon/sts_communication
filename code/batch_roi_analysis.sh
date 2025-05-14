# /bin/bash
subs=(1 2 3 4 5 7 8 9 11 12 13 14 15 16)

for s in "${subs[@]}"; do 
    echo $s
    # for task in tom; do 
    #     python nilearn_glm.py -s $s -t $task
    #     python functional_rois.py -s $s -t $task
    # done 
    # python nilearn_glm_runwise.py -s $s --overwrite
    # python runwise_response.py -s $s
    python visualize_rois.py -s $s
done
python group_runwise_results.py "${subs[@]}" --overwrite