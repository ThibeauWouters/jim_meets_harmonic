import os
import numpy as np 
import matplotlib.pyplot as plt
import corner
import h5py

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

# file = "./outdir/final_result/test_data0_0-0_analysis_H1L1V1_result.hdf5"
file = "./outdir/result/test_data0_0-0_analysis_H1L1V1_result.hdf5"

with h5py.File(file, 'r') as f:
    print(f.keys())
    
    injection_parameters = f["injection_parameters"]
    
    injected_chirp_mass = injection_parameters["chirp_mass"][()]
    injected_mass_ratio = injection_parameters["mass_ratio"][()]
    injected_chi_1 = injection_parameters["chi_1"][()]
    injected_chi_2 = injection_parameters["chi_2"][()]
    
    posterior = f["posterior"]
    # pkeys = posterior.keys()
    # for key in pkeys:
    #     print(key)
    
    chirp_mass = posterior["chirp_mass"][()]
    mass_ratio = posterior["mass_ratio"][()]
    chi_1 = posterior["chi_1"][()]
    chi_2 = posterior["chi_2"][()]
    
    samples = np.array([chirp_mass, mass_ratio, chi_1, chi_2]).T
    truths = np.array([injected_chirp_mass, injected_mass_ratio, injected_chi_1, injected_chi_2])
    
    # Make a corner plot
    corner.corner(samples, truths = truths, labels = ["Mc", "q", "chi1", "chi2"], hist_1d_kwargs={"density": "true"}, **default_corner_kwargs)
    plt.savefig("./corner.png", bbox_inches = 'tight')
    plt.show()