# Batch runs

Below, there is some explanation about the run outdirs:

- `failed_outdir`: First attempt, but most of the runs have weird behaviours: chirp mass is not peaking at the correct location or is railing etc. Might this be because of relative binning, or perhaps we need a more agressive relative binning setting?
- `outdir`: Going to run without relative binning here. Hopefully the result will make sense.


## Guide from parallel_bilby: how to run multiple pbilby Injections
This can be useful for pp-tests or if you need to conduct an injection study with pbilby.

1. Edit and create inis:
    - Edit `pbilby_config_template.ini` and constants in `setup_multiple_pbilby_injections.py`
    - Run `$ python setup_multiple_pbilby_injections.py` 

2. Parallel bilby generation:
    - Test the constructed inis with `$ parallel_bilby_generation {LABEL}_0.ini`
    - Run `$ sbatch slurm_data_generation.sh` to generate data for all  injection's (generation logs stored in `generation_log`)
 
3. Parallel bilby analysis:
    - Test one job's analysis on the head node with `$ parallel_bilby_analysis out_{LABEL}_0/data/{LABEL}_0_data_dump.pickle --nact {NACT} --label {LABEL})_0_0 --outdir {OUTDIR} --sampling-seed 1234`
    - Run `$ bash start_data_analysis.sh` to submit all the analysis jobs  