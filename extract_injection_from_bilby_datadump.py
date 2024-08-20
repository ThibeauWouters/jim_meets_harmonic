import numpy as np
import os
import copy
import json
import pickle
import argparse
import bilby

#################
### CONSTANTS ###
#################

BILBY_PARAM_NAMES = ["chirp_mass",
                     "mass_ratio",
                     "chi_1",
                     "chi_2",
                     "luminosity_distance",
                     "geocent_time",
                     "phase",
                     "theta_jn",
                     "psi",
                     "ra",
                     "dec"
]

# These are the hyperparaeters to feed into Jim that are the same for all injections
DEFAULT_JIM_CONFIG = {"seed": 21, "f_sampling": 4096, "duration": 4, "post_trigger_duration": 2, "trigger_time": 0.0, "fmin": 20, "fref": 20, "ifos": ["H1", "L1", "V1"], "outdir": "./outdir/injection_1/"}

# To translate the bilby names to suitable jim names
# TODO: update with the 15D parameters?
BILBY_TO_JIM_DICT = {"chirp_mass": "M_c",
                     "mass_ratio": "q",
                     "chi_1": "s1_z",
                     "chi_2": "s2_z",
                     "luminosity_distance": "d_L",
                     "geocent_time": "t_c",
                     "phase": "phase_c",
                     "theta_jn": "iota",
                     "psi": "psi",
                     "ra": "ra",
                     "dec": "dec"}

###########################
### EXTRACTION FUNCTION ###
###########################

def main(datadump_file: str,
         outdir: str):
    """
    Extracts the injected data from bilby to be processed into Jim.

    Args:
        datadump_file (str): The datadump pickle file from bilby
        outdir (str): The directory where the injection for Jim will be done.
    """
    
    print(f"Extracting the injection data from data file: {datadump_file}")
    
    # Open and load
    with open(datadump_file, "rb") as f:
        data = pickle.load(f)
        
        ifo_list = data["ifo_list"]
        injection_parameters = data["injection_parameters"]
    
    ### SAVE INJECTION DATA
    for ifo in ifo_list:
        # Get the strain
        strain_data = ifo.strain_data # InterferometerStrainData
        _frequency_domain_strain = strain_data._frequency_domain_strain
        _times_and_frequencies = strain_data._times_and_frequencies # CoupledTimeAndFrequencySeries object
        
        freqs = _times_and_frequencies.frequency_array
        real_strain = _frequency_domain_strain.real
        imag_strain = _frequency_domain_strain.imag
        
        # Get the psd values
        psd_values = ifo.power_spectral_density._PowerSpectralDensity__power_spectral_density_interpolated(freqs)
        
        # Assert all have same length
        assert len(freqs) == len(real_strain) == len(imag_strain) == len(psd_values), "Some shape mismatch"
            
        print(f"Saving {ifo.name} data to npz file")
        np.savez(os.path.join(outdir, f"{ifo.name}_data.npz"), freqs=freqs, real_strain=real_strain, imag_strain=imag_strain, psd_values=psd_values)
        
    ### CREATE CONFIG FILE
    config_dict = copy.deepcopy(DEFAULT_JIM_CONFIG)
    for bilby_name, jim_name in BILBY_TO_JIM_DICT.items():
        config_dict[jim_name] = float(injection_parameters[bilby_name])
        
    print(f"Saving injection and metadata to config.json file")
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config_dict, f)
        
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a datadump pickle file from bilby for Jim consumption.")
    
    # Adding arguments
    parser.add_argument("--datadump-file", type=str, help="Path to the datadump pickle file from bilby")
    parser.add_argument("--outdir", type=str, help="Output directory for processed data for Jim injection")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.datadump_file, args.outdir)
    
    print("Done!")