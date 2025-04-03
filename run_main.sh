#!/bin/bash

#SBATCH --job-name=gemma3
#SBATCH --output=slurm_outputs/%j_%x.out

#SBATCH --time=0-0:05:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G


# Slurm automatically creates a temporary directory for each job as $SLURM_TMPDIR
echo "🗂️ Local compute node directory: $SLURM_TMPDIR"

# Define a variable to point to the project directory
source_repo=$project/workspace/compute-canada-intro

# Assume this has been downloaded in $source_repo/models/$model_id
model_id="google/gemma-3-1b-it"


# Load the necessary modules
echo "🔧 Loading modules..."
module load python/3.11 cuda cudnn
source $source_repo/venv/bin/activate
echo "✅ Loaded modules"


# Copy any necessary files and directories to temp directory using tar and cp
echo "📦 Copying files to $SLURM_TMPDIR/workspace..."
files=(
    main.py
)
cd $source_repo
tar -cf $source_repo/tmp_files.tar "${files[@]}"
mkdir -p "$SLURM_TMPDIR/workspace"
cp tmp_files.tar "$SLURM_TMPDIR/workspace"
rm tmp_files.tar
cd "$SLURM_TMPDIR/workspace"
tar -xf tmp_files.tar
rm tmp_files.tar
echo "✅ Copied files to $SLURM_TMPDIR/workspace"


# Copy the model to the temporary directory
echo "🤖 Copying model to $SLURM_TMPDIR/models..."
mkdir -p $SLURM_TMPDIR/models
cp -r $source_repo/models/$model_id $SLURM_TMPDIR/models
save_id="${model_id#*/}"
model_path=$SLURM_TMPDIR/models/$save_id
echo "✅ Copied $model_id to $model_path"


# Run the Python script
echo "🚀 Running main.py with $model_id ($model_path)..."
python main.py --model_path $model_path
echo "✅ Finished running main.py"


# Copy the outputs back to the source repository
echo "📦 Copying results to $source_repo/results/$SLURM_JOB_ID..."
mkdir -p $source_repo/results
cp -r $SLURM_TMPDIR/workspace/results $source_repo/results/$SLURM_JOB_ID
echo "✅ Copied results to $source_repo/results/$SLURM_JOB_ID"


echo "🎉 Job finished successfully!"
