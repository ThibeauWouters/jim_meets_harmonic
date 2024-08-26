"""
Module to create an injection file + pbilby inis for the injections.
"""
import logging
import numpy as np

from bilby_pipe.create_injections import create_injection_file

logging.getLogger().setLevel(logging.INFO)

N_INJECTION = 10
LABEL = "bbh"  # the main name of the injections
INJECTION_FILE = f"./datafiles/{LABEL}_injections.json"
PRIOR_FILE = "./datafiles/bbh.prior"

def create_ini(injection_idx: int):
    unique_label = f"{LABEL}_{injection_idx}"
    outdir = f"out_{unique_label}"
    ini = f"{unique_label}.ini"
    sampling_seed = np.random.randint(0, 9999)
    
    # Read the ini template
    with open("template.ini", "r") as f:
        txt = f.read()
        
    # Replace the placeholders with the actual values
    txt = txt.replace("{{{LABEL}}}", unique_label)
    txt = txt.replace("{{{OUTDIR}}}", outdir)
    txt = txt.replace("{{{SAMPLING_SEED}}}", str(sampling_seed))
        
    with open(ini, "w") as f:
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
        create_ini(injection_idx=i)

if __name__ == "__main__":
    main()