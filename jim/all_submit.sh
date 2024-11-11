#!/bin/bash

### List of GW event IDs (you can replace these with actual IDs)

# Define the path to the template script
template_file="template.sh"

# Loop over each GW event ID
for id in $(seq 1 5);
do
  # Create a unique SLURM script for each GW event
  new_script="slurm_scripts/submit_${id}.sh"
  cp $template_file $new_script
  
  # Replace the placeholder with the actual GW_ID
  sed -i "s/{{{ID}}}/$id/g" $new_script
  
  # Submit the job to SLURM
  sbatch $new_script
  
  echo "Submitted job for $id"
done