import os
# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
import numpy as np
# Regular imports 
import copy
import numpy as np
from astropy.time import Time
import time
import shutil
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD # TODO: add the BNS waveforms here as well?
from jimgw.prior import Uniform, Composite
import utils
from utils import SUPPORTED_WAVEFORMS, DEFAULT_WAVEFORM
import optax

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# All uniform
PRIOR = {
        "M_c": [25.0, 100.0],
        "q": [0.125, 1.0], 
        "s1_z": [-0.99, 0.99], 
        "s2_z": [-0.99, 0.99],
        "d_L": [5e2, 4e3], 
        "t_c": [-0.01, 0.01], 
        "phase_c": [0.0, 2 * jnp.pi], 
        "iota": [0.0, 2 * jnp.pi],
        "psi": [0.0, jnp.pi], 
        "ra": [0.0, 2 * jnp.pi],
        "dec": [0.0, 2 * jnp.pi]
}
NAMING = list(PRIOR.keys())

####################
### Script setup ###
####################

def body(args):
    """
    Run an injection and recovery. To get an explanation of the hyperparameters, go to:
        - jim hyperparameters: https://github.com/ThibeauWouters/jim/blob/8cb4ef09fefe9b353bfb89273a4bc0ee52060d72/src/jimgw/jim.py#L26
        - flowMC hyperparameters: https://github.com/ThibeauWouters/flowMC/blob/ad1a32dcb6984b2e178d7204a53d5da54b578073/src/flowMC/sampler/Sampler.py#L40
    """
    
    start_time = time.time()
    # TODO: Deal with the naming, to make it automatic
    naming = NAMING
    
    # Note: if we load from bilby, then we do not need to generate, so load existing config is set to true
    if args.from_bilby:
        print("Setting load_existing_config to True, since we are loading from bilby so we don't need to generate new parameters")
        args.load_existing_config = True
    
    # Fetch waveform used
    if args.waveform_approximant not in SUPPORTED_WAVEFORMS:
        print(f"Waveform approximant {args.waveform_approximant} not supported. Supported waveforms are {SUPPORTED_WAVEFORMS}. Changing to {DEFAULT_WAVEFORM}.")
        args.waveform_approximant = DEFAULT_WAVEFORM
    
    ripple_waveform_fn = RippleIMRPhenomD

    # Check if outdir formatted correctly, then complete it with the passed identifier
    if not args.outdir.endswith(os.path.sep):
        args.outdir += os.path.sep
    outdir = f"{args.outdir}injection_{args.id}/"
    print(f"Saving results to: {outdir}")
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # First, copy over this script to the outdir for reproducibility
    shutil.copy2(__file__, outdir + "copy_injection_recovery.py")
    
    # Get the prior bounds, both as 1D and 2D arrays
    prior_ranges = jnp.array([PRIOR[name] for name in naming])
    prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]
    bounds = np.array(list(PRIOR.values()))
    
    # Now go over to creating parameters, and potentially check SNR cutoff
    network_snr = 0.0
    print(f"The SNR threshold parameter is set to {args.SNR_threshold}")
    while network_snr < args.SNR_threshold:
        
        # Generate the parameters or load them from an existing file
        if args.load_existing_config:
            config_path = f"{outdir}config.json"
            print(f"Loading existing config, path: {config_path}")
            config = json.load(open(config_path))
        else:
            print(f"Generating new config")
            config = utils.generate_config(prior_low, prior_high, naming, args.id, args.outdir)
        
        # key for noise generation and sampling
        key = jax.random.PRNGKey(args.noise_seed)
        
        # Save the given script hyperparams
        with open(f"{outdir}script_args.json", 'w') as json_file:
            json.dump(args.__dict__, json_file)
        
        # Start injections
        print("Injecting signals . . .")
        waveform = ripple_waveform_fn(f_ref=config["fref"])
        
        # Create frequency grid
        freqs = jnp.arange(
            config["fmin"],
            config["f_sampling"] / 2,  # maximum frequency being halved of sampling frequency
            1. / config["duration"]
            )
        # convert injected mass ratio to eta, and apply arccos and arcsin
        q = config["q"]
        eta = q / (1 + q) ** 2
        # Setup the timing setting for the injection
        epoch = config["duration"] - config["post_trigger_duration"]
        gmst = Time(config["trigger_time"], format='gps').sidereal_time('apparent', 'greenwich').rad
        # Array of injection parameters # TODO: add the lambdas here in case BNS waveforms?
        true_param = {
            'M_c':       config["M_c"],       # chirp mass
            'eta':       eta,                 # symmetric mass ratio 0 < eta <= 0.25
            's1_z':      config["s1_z"],      # aligned spin of priminary component s1_z.
            's2_z':      config["s2_z"],      # aligned spin of secondary component s2_z.
            'd_L':       config["d_L"],       # luminosity distance
            't_c':       config["t_c"],       # timeshift w.r.t. trigger time
            'phase_c':   config["phase_c"],   # merging phase
            'iota':      config["iota"],                # inclination angle
            'psi':       config["psi"],       # polarization angle
            'ra':        config["ra"],        # right ascension
            'dec':       config["dec"]                  # declination
            }
        
        print("True param:")
        for k, val in true_param.items():
            print(f"{k}: {val}")
        
        # # Get the true parameter values for the plots
        # truths = np.array([config[key] for key in naming])
        truths = np.array(list(true_param.values()))
        
        detector_param = {
            'ra':     config["ra"],
            'dec':    config["dec"],
            'gmst':   gmst,
            'psi':    config["psi"],
            'epoch':  epoch,
            't_c':    config["t_c"],
            }
        print(f"The injected parameters are {true_param}")
        
        
        # Setup interferometers
        # TODO: make ifos more general? Make fetching PSD files more general?
        ifos = [H1, L1, V1]
        psds: dict[str, str] = {"H1": args.psd_file_H1,
                                "L1": args.psd_file_L1,
                                "V1": args.psd_file_V1}
    
        # inject signal into ifos
        if not args.from_bilby:
            print("Injecting signal from scratch")
            
            # Generating the geocenter waveform
            h_sky = waveform(freqs, true_param)
        
            for idx, ifo in enumerate(ifos):
                key, subkey = jax.random.split(key)
                ifo.inject_signal(
                    subkey,
                    freqs,
                    h_sky,
                    detector_param,
                    psd_file=psds[ifo.name],
                )
            print("Signal injected")
            
            # TODO: needs to be modified?
            h1_snr = utils.compute_snr(H1, h_sky, detector_param)
            print(f"SNR for H1: {h1_snr}")
            l1_snr = utils.compute_snr(L1, h_sky, detector_param)
            print(f"SNR for L1: {l1_snr}")
            v1_snr = utils.compute_snr(V1, h_sky, detector_param)
            print(f"SNR for V1: {v1_snr}")
            network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)
            
            # If the SNR is too low, we need to generate new parameters
            if network_snr < args.SNR_threshold:
                print(f"Network SNR is less than {args.SNR_threshold}, generating new parameters")
                if args.load_existing_config:
                    raise ValueError("SNR is less than threshold, but loading existing config. This should not happen!")
        
        
            print(f"Saving network SNR")
            with open(outdir + 'network_snr.txt', 'w') as file:
                file.write(str(network_snr))
            
        else:
            print("Injecting signal from output generated from bilby")
            for ifo in ifos:
                name = ifo.name
                
                bilby_file = os.path.join(outdir, f"{name}_data.npz")
                ifo_data = np.load(bilby_file)
                
                # NOTE: this overwrites freqs!
                freqs, real_strain, imag_strain, psd_values = ifo_data["freqs"], ifo_data["real_strain"], ifo_data["imag_strain"], ifo_data["psd_values"]
                
                mask = (freqs >= config["fmin"]) & (freqs <= config["f_sampling"] / 2)
                
                freqs = freqs[mask]
                
                ifo.frequencies = freqs
                ifo.data = (real_strain + 1j * imag_strain)[mask]
                ifo.psd = psd_values[mask]
                
                print(f"Loaded {name} data from bilby output")
                
            # Pass the SNR threshold check
            network_snr = args.SNR_threshold + 1
                
            # TODO: can remove this? # At this point still need to generate the geocenter waveform
            # h_sky = waveform(freqs, true_param)
        


    print("Start prior setup")
    
    # Priors without transformation 
    Mc_prior       = Uniform(prior_low[0], prior_high[0], naming=['M_c'])
    q_prior        = Uniform(prior_low[1], prior_high[1], naming=['q'],transforms={'q': ('eta',lambda params: params['q'] / (1 + params['q']) ** 2)})
    s1z_prior      = Uniform(prior_low[2], prior_high[2], naming=['s1_z'])
    s2z_prior      = Uniform(prior_low[3], prior_high[3], naming=['s2_z'])
    dL_prior       = Uniform(prior_low[4], prior_high[4], naming=['d_L'])
    tc_prior       = Uniform(prior_low[5], prior_high[5], naming=['t_c'])
    phic_prior     = Uniform(prior_low[6], prior_high[6], naming=['phase_c'])
    # cos_iota_prior = Uniform(prior_low[7], prior_high[7], naming=["cos_iota"],transforms={"cos_iota": ("iota",lambda params: jnp.arccos(params["cos_iota"]))},)
    iota_prior = Uniform(prior_low[7], prior_high[7], naming=["iota"])
    psi_prior      = Uniform(prior_low[8], prior_high[8], naming=["psi"])
    ra_prior       = Uniform(prior_low[9], prior_high[9], naming=["ra"])
    # sin_dec_prior  = Uniform(prior_low[10], prior_high[10], naming=["sin_dec"],transforms={"sin_dec": ("dec",lambda params: jnp.arcsin(params["sin_dec"]))},
    dec_prior  = Uniform(prior_low[10], prior_high[10], naming=["dec"])
    
    # If this is a 4D run, then we only sample over the first 4 parameters, the other parameters are fixed:
    
    if args.is_4d_run:
        # Compose the prior
        prior_list = [
                Mc_prior,
                q_prior,
                s1z_prior,
                s2z_prior,
        ]
        
        fixing_parameters = {"d_L": config["d_L"], 
                             "t_c": config["t_c"], 
                             "phase_c": config["phase_c"], 
                             "iota": config["iota"], 
                             "psi": config["psi"], 
                             "ra": config["ra"], 
                             "dec": config["dec"]}
        
        truths = truths[:4]
        
    else:
        # Compose the prior
        prior_list = [
                Mc_prior,
                q_prior,
                s1z_prior,
                s2z_prior,
                dL_prior,
                tc_prior,
                phic_prior,
                # cos_iota_prior, #FIXME:
                iota_prior,
                psi_prior,
                ra_prior,
                # sin_dec_prior, #FIXME:
                dec_prior,
        ]
        
        fixing_parameters = {}
    
    # Combine the given list of priors
    complete_prior = Composite(prior_list)
    bounds = jnp.array([[p.xmin, p.xmax] for p in complete_prior.priors])
    print("Finished prior setup")

    print("Initializing likelihood")
    if args.use_relative_binning:
        
        raise ValueError("Please don't use relative binning for the harmonic paper (for now?)")
        
        # print("INFO using relative binning")
        # if args.relative_binning_ref_params_equal_true_params:
        #     ref_params = true_param
        #     print("Using the true parameters as reference parameters for the relative binning")
        # else:
        #     ref_params = None
        #     print("Will search for reference waveform for relative binning")
            
        # likelihood = HeterodynedTransientLikelihoodFD(
        #     ifos,
        #     prior=complete_prior,
        #     bounds=bounds,
        #     n_bins = args.relative_binning_binsize,
        #     waveform=waveform,
        #     trigger_time=config["trigger_time"],
        #     duration=config["duration"],
        #     post_trigger_duration=config["post_trigger_duration"],
        #     ref_params=ref_params,
        #     )

    else:
        print("INFO using normal likelihood")
        likelihood = TransientLikelihoodFD(
            ifos,
            waveform=waveform,
            trigger_time=config["trigger_time"],
            duration=config["duration"],
            post_trigger_duration=config["post_trigger_duration"],
            fixing_parameters = fixing_parameters
        )
        
    # Get the mass matrix of step sizes for the local sampler
    mass_matrix = jnp.eye(len(prior_list))
    for idx, prior in enumerate(prior_list):
        if prior.naming[0] in ['t_c']:
            print(f'Modified the mass matrix for {prior.naming}')
            mass_matrix = mass_matrix.at[idx, idx].set(1e-3)
    local_sampler_arg = {'step_size': mass_matrix * float(args.eps_mass_matrix)}
    
    if args.use_scheduler:
        total_epochs = args.n_epochs * args.n_loop_training
        start = int(total_epochs / 10)
        start_lr = 1e-3
        end_lr = 1e-5
        power = 4.0
        args.learning_rate = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)

        # TODO: remove me, debug this
        print("Learning rate is now:")
        print(args.learning_rate)
    
    # Create jim object
    jim = Jim(
        likelihood,
        complete_prior,
        n_loop_training=args.n_loop_training,
        n_loop_production = args.n_loop_production,
        n_local_steps=args.n_local_steps,
        n_global_steps=args.n_global_steps,
        n_chains=args.n_chains,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        max_samples = args.max_samples,
        momentum = args.momentum,
        batch_size = args.batch_size,
        use_global=args.use_global,
        keep_quantile= args.keep_quantile,
        train_thinning = args.train_thinning,
        output_thinning = args.output_thinning,
        local_sampler_arg = local_sampler_arg,
        num_layers = args.num_layers,
        seed = args.sampling_seed,
        hidden_size = args.hidden_size,
        num_bins = args.num_bins
    )
    
    ### Finally, do the sampling
    print(f"The sampling seed will be {int(args.sampling_seed)}")
    key = jax.random.PRNGKey(int(args.sampling_seed))
    jim.sample(key)
        
    # === Show results, save output ===

    # Print a summary to screen:
    jim.print_summary()

    # Save and plot the results of the run
    #  - training phase
    
    print("Producing output plots . . . ")
    
    name = outdir + f'results_training.npz'
    print(f"Saving samples to {name}")
    state = jim.Sampler.get_sampler_state(training = True)
    chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
    local_accs = jnp.mean(local_accs, axis=0)
    global_accs = jnp.mean(global_accs, axis=0)
    if args.save_training_chains:
        np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_vals, chains=chains)
    else:
        np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_vals)
    
    utils.plot_accs(local_accs, "Local accs (training)", "local_accs_training", outdir)
    utils.plot_accs(global_accs, "Global accs (training)", "global_accs_training", outdir)
    utils.plot_loss_vals(loss_vals, "Loss", "loss_vals", outdir)
    utils.plot_log_prob(log_prob, "Log probability (training)", "log_prob_training", outdir)
    
    #  - production phase
    name = outdir + f'results_production.npz'
    state = jim.Sampler.get_sampler_state(training = False)
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    local_accs = jnp.mean(local_accs, axis=0)
    global_accs = jnp.mean(global_accs, axis=0)
    np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

    utils.plot_accs(local_accs, "Local accs (production)", "local_accs_production", outdir)
    utils.plot_accs(global_accs, "Global accs (production)", "global_accs_production", outdir)
    utils.plot_log_prob(log_prob, "Log probability (production)", "log_prob_production", outdir)

    # Plot the chains as corner plots
    if args.from_bilby:
        # TODO: if we load from bilby then the true parameters don't match -- these have to be loaded!
        truths = None
    utils.plot_chains(chains, "chains_production", outdir, truths = truths)
    
    # Save the NF and show a plot of samples from the flow
    print("Saving the NF")
    jim.Sampler.save_flow(outdir + "nf_model")
    name = outdir + 'results_NF.npz'
    chains = jim.Sampler.sample_flow(10_000)
    np.savez(name, chains = chains)
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")
    
    print(f"Saving runtime")
    with open(outdir + 'runtime.txt', 'w') as file:
        file.write(str(runtime))
    
    print("Finished injection recovery successfully!")

############
### MAIN ###
############

def main(given_args = None):
    
    parser = utils.get_parser()
    args = parser.parse_args()
    
    print("given_args")
    print(given_args)
    
    # Update with given args
    if given_args is not None:
        args.__dict__.update(given_args)
        
    if args.load_existing_config and args.id == "":
        raise ValueError("If load_existing_config is True, you need to specify the --id argument to locate the existing injection.")
    
    print("------------------------------------")
    print("Arguments script:")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("------------------------------------")
        
    print("Starting main code")
    
    # If no N is given, fetch N from the structure of outdir
    if len(args.id) == 0:
        N = utils.get_N(args.outdir)
        args.id = N
    
    # Execute the script
    body(args)
    
if __name__ == "__main__":
    main()
