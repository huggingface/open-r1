

# a bash foor loop from 0 to 17,400 in chunks of 200

for i in {0..17000..200}
do
  START=$i
  END=$((i + 200))
  echo "Processing chunk from $START to $END"
  
  # Submit the job to SLURM
  sbatch slurm/compute_pass_rate.slurm recipes/dataset_filtering/filter_dapo.yaml $START $END
done

sbatch slurm/compute_pass_rate.slurm recipes/dataset_filtering/filter_dapo.yaml 17200 17398
