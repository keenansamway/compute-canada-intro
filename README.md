# Digital Research Alliance of Canada (aka Compute Canada) Introductory Guide

While my experience with CC has so far been limited to the Narval cluster (as it is currently the most capable cluster for deep learning tasks requiring large amounts of GPU memory), the following instructions should generally be applicable to other clusters as well. Starting later on in the spring of 2025, new clusters equiped with H100 80GB cards will be coming online that will match and exceed the capabilities of Narval. Keep an eye on the [2025 Intrastructure Renewal](https://docs.alliancecan.ca/wiki/Infrastructure_renewal) page for updates on these new clusters.


## Table of Contents
- [Overview and Resources](#overview-and-resources)
    - [Logging in with SSH](#logging-in-with-ssh)
    - [Directory Structure](#directory-structure)
    - [Virtual Environments](#virtual-environments)
    - [Useful tmux Commands](#useful-tmux-commands)
    - [Submitting Jobs](#submitting-jobs)
        - [Some of my experiences](#some-of-my-experiences)
        - [An example SLURM header](#an-example-slurm-header)
        - [Rough LLM Memory Requirements](#rough-llm-memory-requirements)
- [Example: Using Hugging Face for LLM inference on Narval](#example-using-hugging-face-for-llm-inference-on-narval)
    - [Setup and Installation](#setup-and-installation)
    - [Download the model](#download-the-model)
    - [Submit a job](#submit-a-job)
    - [Check the job status](#check-the-job-status)
    - [Check the output](#check-the-output)


## Overview and Resources
- [CCDB Digital Research Alliance of Canada - Login](https://ccdb.alliancecan.ca/security/login)
- [Official Wiki - Technical Documentation](https://docs.alliancecan.ca/wiki/Technical_documentation)
- [Official Wiki - Narval](https://docs.alliancecan.ca/wiki/Narval)
- [Narval Portal](https://portail.narval.calculquebec.ca)
- [A Complete Guide for using Compute Canada for Deep Learning](https://prashp.gitlab.io/post/compute-canada-tut/) (external blogpost)

### Logging in with SSH
I'm going to skip some of the early steps on how to set up your SSH keys and such. If you need help with this, check out the official wiki or external blogpost linked above. I will however include my `.ssh/config` file which I use to make it easier to log in to the cluster. Once you upload your public key to the [CCDB portal](https://docs.alliancecan.ca/wiki/SSH_Keys#Installing_your_key), it won't ask for your password when connecting. Feel free to modify this to your liking. See [https://docs.alliancecan.ca/wiki/Visual_Studio_Code/en](https://docs.alliancecan.ca/wiki/Visual_Studio_Code/en) for more information on using VSCode with the cluster.

```bash
# Digital Research Alliance of Canada #
Host beluga cedar graham narval
  ServerAliveInterval 300
  HostName %h.alliancecan.ca
  IdentityFile ~/.ssh/<your_private_key>    # specify your SSH key (also setup with GitHub)
  User <your_username>                      # specify your username
  ForwardAgent yes

Host narval1 narval2 narval3                # specify individual login nodes when using tmux
  HostName %h.alliancecan.ca
  IdentityFile ~/.ssh/<your_private_key>    # specify your SSH key (also setup with GitHub)
  User <your_username>                      # specify your username
  ForwardAgent yes

Host bc????? bg????? bl?????
  ProxyJump beluga

Host cdr*
  ProxyJump cedar

Host gra1* gra2* gra3* gra4* gra5* gra6* gra7* gra8* gra9*
  ProxyJump graham

Host nc????? ng????? nl?????
  ProxyJump narval
```

> Note that ControlMaster, which is used to reduce the need to re-authenticate with Duo within a specified time period, caused me issues with Agent Forwarding, which made git commits fail as the SSH agent was not being forwarded properly. I recommend disabling this option in your SSH config file if you are using Agent Forwarding.

### Directory Structure
The following is a quick introduction to the directory structure of the cluster (see more here [https://docs.alliancecan.ca/wiki/Storage_and_file_management](https://docs.alliancecan.ca/wiki/Storage_and_file_management)).

Once you log in to the cluster, you will be in your home directory, aka `~`. You'll notice three different directories structured as follows:
```
/home/<your_username> (50G, personal, aka $HOME or ~)
├── nearline
│   └── <def-supervisor> -> /lustreXX/nearline/<id> (1T, group)
│       ├── ...
│       ├── <your_username>
│       └── ...
├── projects
│   └── <def-supervisor> -> /lustreXX/project/<id> (1T, group)
│       ├── ...
│       ├── <your_username>
│       └── ...
└── scratch -> /lustreXX/scratch/<your_username> (20T, group)
```
Very generally: `nearline` is meant for archival storage, `scratch` is meant for temporary storage of files (wiped after ~2 months), and `projects` is a shared group directory among us all. You can check the available disk space using `diskusage_report`. Add the `--per_user` flag to see the per user breakdown.

>Note that `projects` has a group limit of 1TB, which can fill up quick with large models and such, so be mindful of this. This can be expanded up to 40TB upon request

There will also be local storage on the compute nodes you’ll submit jobs to, which you can access from within the submission script using `$SLURM_TMPDIR`. It is highly recommended that you transfer any datasets or important files which will be read/written many times to this temporary disc space and accessing the files from there (see more here [https://docs.alliancecan.ca/wiki/Transferring_data](https://docs.alliancecan.ca/wiki/Transferring_data)). Note that if your script outputs results to the local compute node directory, you’ll want to transfer those results back to the other storage options at the end of the job script.

> Note that Narval compute nodes are **NOT** connected to the internet. Anything that requires internet connection should be done prior to submitting a job (such as downloading models, datasets, etc.).

For convenience, I added the following variable to my `.bashrc` file:
```bash
export project="~/projects/<def-supervisor>/<your_username>"
```

> Note that `<def-supervisor>` is the name of your supervisor's project which can be found in the `projects` directory (i.e. by running `ls ~/projects`) or by looking it up on the CCDB [portal](https://ccdb.alliancecan.ca/security/login).


### Virtual Environments
While Conda is a popular choice for creating virtual environments, it is highly advised **against** on the clusters (see [https://docs.alliancecan.ca/wiki/Python](https://docs.alliancecan.ca/wiki/Python)). Instead, use `virtualenv` to create virtual environments.

1. First, check to see what version of Python is available on the cluster.
```bash
module avail python
```
2. Load the desired version of Python.
```bash
module load python/3.11
```
3. Create a virtual environment (optionally in your project directory).
```bash
virtualenv --no-download /path/to/your/project/.venv
```
4. Activate the virtual environment.
```bash
source /path/to/your/project/.venv/bin/activate
```
5. Install any required packages.
```bash
pip install --no-index --upgrade pip
pip install --no-index -r torch transformers accelerate
```
6. Deactivate the virtual environment when done.
```bash
deactivate
```

> Note that when installing packages using pip, prefer to use the `--no-index` flag which uses the prebuilt wheels already available and tested on the cluster. Use `avail_wheels` (either on its own or followed by a package name, e.g. `avail_wheels torch`) to see the available wheels on the cluster. If a package is not available, reach out to [technical support](https://docs.alliancecan.ca/wiki/Technical_support) to request it be added.


### Useful tmux Commands
If you are using tmux, here are some useful commands to help you manage your sessions:
```
# tmux commands from outside of tmux
tmux - create a new tmux session (named '0' by default)
tmux a - attach to most recent tmux session
tmux ls - list all tmux sessions
tmux kill-session -t <session_name> - kill a specific tmux session

# tmux commands from inside of tmux
^b d - detach from the current tmux session

^b % - split the window into two panes horizontally
^b " - split the window into two panes vertically
^b x - close pane
^b arrow keys - move between panes

^b c - create new window
^b n - move to the next window
^b p - move to the previous window

^b q - view all keybindings (q to exit)
```

tmux can be particularly useful when downloading large files or models. You can start a tmux session, start the download, and then detach from the session. The download will continue in the background, and you can reattach to the session later to check on the progress.

> Note that if if you are using tmux, make sure you ssh to the same login node that you started the tmux session on. You can check which login node you are currently on by running `echo $HOSTNAME`. On Narval, this will print something like `narval2.narval.calcul.quebec`. Next time you ssh into the cluster, use `ssh <your_username>@narval2.alliancecan.ca` and/or modify your .ssh config file to include the particular login node you want to connect to.


### Submitting Jobs
Here we'll quickly go over how to submit jobs to the cluster using SLURM. The following is a list of some commonly used commands:
- `squeue` will show all of the current queued and running jobs on the cluster (I don’t really recommend using this as it outputs *a lot* of text making it challenging really to make use of)
- `sq` will show your current queued and running jobs
- `sbatch your_script_name.sh` will queue up the program in your_script_name.sh
- `scancel <job_id>` will cancel the job associated with the particular job_id
- `sacct <argument1> <argument2> …` will request an interactive session using the particular arguments you request (this is useful for testing/debugging purposes, but generally you should run your jobs using sbatch)
    - The arguments used here are the same as those used in the header of sbatch
    - e.g., `salloc --time=0-01:00:00 --gpus-per-node=1 --cpus-per-task=1 --mem=16G`
    - Note that the instance will launch in the terminal you run the command in as soon as it is ready, so make sure you keep an eye on it.
    - From the wiki:
    - > Note that an interactive job with a duration of three hours or less will likely start very soon after submission as we have dedicated test nodes for jobs of this duration. Interactive jobs that request more than three hours run on the cluster's regular set of nodes and may wait for many hours or even days before starting, at an unpredictable (and possibly inconvenient) hour.


#### Some of my experiences
In my experience, I found that experimenting with my script with `salloc`, dialing in the necessary resources, and then submitting an `sbatch` request worked best. Once you've submitted a job, you can view all of the details regarding the number of CPU cores, system memory, GPU memory, utilization, etc. on [https://portail.narval.calculquebec.ca/](https://portail.narval.calculquebec.ca/).

The job queue times have been quite variable, but I've generally found them to be reasonable. I've had jobs that started just a couple minutes after submission (generally requesting 1 or 2 GPUs), and other jobs that took a couple hours (requesting 4 GPUs). Generally the more resources you request the longer it will take to queue, so try and request the minimum amount that are able to successfully run your job.

Your mileage may vary, but I haven't had much luck with Multi-Instance GPU jobs (see more at [https://docs.alliancecan.ca/wiki/Multi-Instance_GPU](https://docs.alliancecan.ca/wiki/Multi-Instance_GPU)), which allow you to request a fraction of a GPU. Thus far the queue times appear to be *significantly* longer than for non-MIG jobs. In fact, I have actually not been able to get a MIG job to start at all (having tested with `--gres=gpu:a100_1g.5gb:1` and `--gres=gpu:a100_2g.10gb:1`).

When a job has been submitted with `sbatch` and is currently running, while you can view some diagnostic information on the Narval portal, you also also use `srun` to attach to the instance and run certain commands. For example `srun --jobid <job_id> --pty nvidia-smi` will print the output of nvidia-smi on the node associated with the specified job_id. Other similar commands like `top` and `htop` can also be run.

I found that Hugging Face transformers only used max 1 CPU core (I don’t think it’s optimized for multithreading), but note that other applications may be able to take advantage of more a higher number of requested CPU cores.


#### An example SLURM header
```bash
#!/bin/bash

#SBATCH --mail-user=<your_email>        # Email where you want to receive notifications of jobs starting/ending
#SBATCH --mail-type=ALL                 # Remove these two lines if you don't want to receive email notifications

#SBATCH --job-name=<your_job_name>      # Name of the job
#SBATCH --output=outputs/%j_%x.out      # Where to save and name the jobs output logs (%j is the job ID and %x is the job name)

#SBATCH --time=0-01:00:00               # Time limit in the format D-HH:MM:SS
#SBATCH --gpus-per-node=1               # Number of GPUs to request (there are 4 A100 40GB GPUs per node)
#SBATCH --cpus-per-task=1               # Number of CPUs to request (there are 48 CPU cores per node)
#SBATCH --mem=16G                       # Memory to request (there is 510G of memory per node)
```


#### Rough LLM Memory Requirements
Llama 3.1 (8B, 70B, and 405B) are pretty common these days. I’ve been able to run inference using both 8B and 70B on Narval at FP16 equivalent precision using 1 A100 and 4 A100s respectively. (I’ve also run inference on a 32B model with just 2 A100s). The following is some useful info from https://huggingface.co/blog/llama31#inference-memory-requirements:
Inference Memory Requirements:
```
+-------------+--------+--------+--------+
| Model Size  | FP16   | FP8    | INT4   |
+-------------+--------+--------+--------+
| 8B          | 16 GB  | 8 GB   | 4 GB   |
| 70B         | 140 GB | 70 GB  | 35 GB  |
| 405B        | 810 GB | 405 GB | 203 GB |
+-------------+--------+--------+--------+

Additional Memory Requirements for KV Cache (FP16):
+-------------+------------+-------------+--------------+
| Model Size  | 1k tokens  | 16k tokens  | 128k tokens  |
+-------------+------------+-------------+--------------+
| 8B          | 0.125 GB   | 1.95 GB     | 15.62 GB     |
| 70B         | 0.313 GB   | 4.88 GB     | 39.06 GB     |
| 405B        | 0.984 GB   | 15.38 GB    | 123.05 GB    |
+-------------+------------+-------------+--------------+

Training Memory Requirements:
+-------------+----------------+--------+--------+
| Model Size  | Full Fine-tune |  LoRA  | Q-LoRA |
+-------------+----------------+--------+--------+
| 8B          | 60 GB          | 16 GB  |  6 GB  |
| 70B         | 500 GB         | 160 GB | 48 GB  |
| 405B        | 3.25 TB        | 950 GB | 250 GB |
+-------------+----------------+--------+--------+
```


## Example: Using Hugging Face for LLM inference on Narval
This is an example showcasing how to use Hugging Face for inference with LLMs on the Narval cluster. I’ve been running these models using Hugging Face mainly because of the convenience, although it might be useful to note that alternatives, such as vLLM, could allow for more efficient memory use and/or faster inference speeds.

We will use `~/projects/<def-supervisor>/<your_username>/workspace/compute-canada-intro` as the working directory for this example. You can change this to whatever you like, but make sure to update the paths in the scripts accordingly.

1. Create a directory for your workspace.
```bash
mkdir -p ~/projects/<def-supervisor>/<your_username>/workspace
cd ~/projects/<def-supervisor>/<your_username>/workspace
```

2. Clone the repository.
```bash
git clone https://github.com/keenansamway/compute-canada-intro.git
cd compute-canada-intro
```


### Setup and Installation
1. Load python and create a virtual environment (in this case, directly in the project directory).
```bash
module load python/3.11
virtualenv --no-download venv
source venv/bin/activate
```
2. Install the required packages.
```bash
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
```

3. Log in to Hugging Face. You'll be prompted to enter your Hugging Face token. Use this if a model requires authentication (e.g., gated models like Llama 3.1, Gemma 3, etc.).
```bash
huggingface-cli login
```
> Note that you can also use the `token` argument in `.from_pretrained()` to pass the token directly in Python.


### Download the model
You can download the model using the `download_model.py` script. This will download the model to the specified local directory. You can also specify the model ID you want to download. For this example, we will download the Gemma 3 1B instruction-tuned model.
```bash
python download_model.py --model_id "google/gemma-3-1b-it" --local_dir "models/"
```
> Note that this requires you to request access to the model on the model card page (found at [https://huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)) and log in to Hugging Face using the `huggingface-cli login` command.


### Submit a job
Here we will request 1 A100 GPU and 4GB of memory for 5 minutes. This should be enough for running inference on the model using the few test examples provided in the `main.py` script. You can adjust the time and memory as needed to try out different models and configurations.
```bash
sbatch run_main.sh
```
> Note that Gemma 3 1B is hard coded in the `run_main.sh` script, so you will need to modify that if you want to try out a different model.


### Check the job status
```bash
sq
```

### Check the output
Look for the output in the `results/<job_id>` directory. The output will be saved in a file named `responses.txt`. You can also check the slurm output file for any errors or logs. The slurm output file will be named `<job_id>_<job_name>.out` and will be located in the `slurm_outputs` directory.