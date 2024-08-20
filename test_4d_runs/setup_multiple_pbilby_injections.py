"""
Module to create an injection file + pbilby inis for the injections.
"""
import logging
import json
import os
# import shutil

from bilby_pipe.create_injections import create_injection_file

logging.getLogger().setLevel(logging.INFO)

N_INJECTION = 5
LABEL = "bbh"  # the main name of the injections
INJECTION_FILE = f"./datafiles/{LABEL}_injections.json"
PRIOR_FILE = "./datafiles/bbh.prior"

def create_ini(injection_idx: int,
               load_fiducial_params: bool = True):
    unique_label = f"{LABEL}_{injection_idx}"
    outdir = f"out_{unique_label}"
    ini = f"{unique_label}.ini"
    
    # Read the ini template
    with open("config.ini", "r") as f:
        txt = f.read()
        
    # Replace the placeholders with the actual values
    txt = txt.replace("{{{IDX}}}", str(injection_idx))
    txt = txt.replace("{{{LABEL}}}", unique_label)
    txt = txt.replace("{{{OUTDIR}}}", outdir)
    txt = txt.replace("{{{PRIOR_FILE}}}", PRIOR_FILE)
    txt = txt.replace("{{{INJECTION_FILE}}}", INJECTION_FILE)
        
    # Load the fiducial parameters from the JSON file
    if load_fiducial_params:
        with open(INJECTION_FILE, "r") as f:
            injections = json.load(f)
            injections = injections["injections"]["content"]
            
            chirp_mass = injections["chirp_mass"][injection_idx]
            mass_ratio = injections["mass_ratio"][injection_idx]
            chi_1 = injections["chi_1"][injection_idx]
            chi_2 = injections["chi_2"][injection_idx]
            
        txt = txt.replace("{{{CHIRP_MASS}}}", str(chirp_mass))
        txt = txt.replace("{{{MASS_RATIO}}}", str(mass_ratio))
        txt = txt.replace("{{{CHI_1}}}", str(chi_1))
        txt = txt.replace("{{{CHI_2}}}", str(chi_2))
        
    with open(ini, "w") as f:
        f.write(txt)


def create_data_generation_slurm_submission_file():
    os.makedirs("generation_log", exist_ok=True)
    with open("template_my_batch_generation.ini", "r") as f:
        txt = f.read()
        txt = txt.replace("{{{GENERATION_LOG_DIR}}}", "generation_log")
        txt = txt.replace("{{{NUM_INJECTIONS}}}", str(N_INJECTION))
        txt = txt.replace("{{{LABEL}}}", LABEL)
        # txt = txt.replace(
        #     "{{{GENERATION_EXE}}}", shutil.which("parallel_bilby_generation")
        # )
    with open("my_batch_generation.sh", "w") as f:
        f.write(txt)


def create_analysis_bash_runner():
    os.makedirs("generation_log", exist_ok=True)
    with open("template_my_batch_run.ini", "r") as f:
        txt = f.read()
        # txt = txt.replace("{{{GENERATION_LOG_DIR}}}", "generation_log")
        txt = txt.replace("{{{NUM_INJECTIONS}}}", str(N_INJECTION))
        txt = txt.replace("{{{LABEL}}}", LABEL)
        # txt = txt.replace(
        #     "{{{GENERATION_EXE}}}", shutil.which("parallel_bilby_generation")
        # )
    with open("my_batch_run.sh", "w") as f:
        f.write(txt)

def main():
    logging.info("Generating injection file")
    create_injection_file(
        filename=INJECTION_FILE,
        prior_file=PRIOR_FILE,
        n_injection=N_INJECTION,
        generation_seed=0
    )

    logging.info("Generating parallel bilby ini files + submission scripts")
    for i in range(N_INJECTION):
        create_ini(injection_idx=i, load_fiducial_params=True)

    create_data_generation_slurm_submission_file()
    logging.info("Now run bash my_batch_generation.sh")

    create_analysis_bash_runner()
    logging.info(
        "Now run bash my_batch_run.sh to run the analysis on the generated data"
    )


if __name__ == "__main__":
    main()
