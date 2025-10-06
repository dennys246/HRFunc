import scipy.linalg, json, mne, random, re, os, nilearn, time
import hrfunc
import numpy as np
import matplotlib.pyplot as plt
from .hrtree import tree, HRF
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from itertools import compress
from glob import glob


def localize_hrfs(nirx_obj, max_distance = 0.01, verbose = False, **kwargs):
    """
    Localize HRFs to the optodes in a fNIRS montage given a nirx object

    Arguments:
        nirx_obj (mne raw object) - NIRS file loaded through mne
        max_distance (float) - Maximum distance in milimeter's a previously estimated HRF can be attached to an optode
        verbose (bool) - If True, print out extra information during localization
        **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)
    
    Returns:
        montage (montage object) - Montage object with localized HRF's
    """
    # Build a montage
    _montage = montage(nirx_obj, **kwargs)
    _montage.localize_hrfs(max_distance, verbose = verbose)
    return _montage

def load_montage(json_filename, rich = False, **kwargs):
    """ 
    Load montage with the given json filename 
    
    Arguments:
        json_filename (str) - Path to json file to load montage from
        rich (bool) - If True, load the full HRF information including estimates and locations, if False only load mean and std
        **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)
    
    Returns:
        montage (montage object) - Montage object with loaded HRF's
    """
    # Read in json
    with open(json_filename, 'r') as file:
        json_contents = json.load(file)

    # Initialize an empty montage object
    montage = hrfunc.montage(**kwargs)

    # Grab info from json contents
    first_hrf = json_contents[list(json_contents.keys())[0]]
    sfreq = first_hrf['sfreq']

    # Assess channel names
    ch_names = ['-'.join(key.split('-')[:-1]) for key in json_contents.keys()]

    # Assess which channels are oxygenated and deoxygenated
    montage.hbo_channels = [ch for ch in ch_names if _is_oxygenated(ch) == True]
    montage.hbr_channels = [ch for ch in ch_names if _is_oxygenated(ch) == False]

    # Update montage with saved info
    for key, channel in json_contents.items():
        key_split = key.split('-')
        doi = key_split.pop()
        ch_name = '-'.join(key_split)

        # Skip if canonical HRF
        if ch_name == 'canonical':
            continue

        if rich == False:
            channel['estimates'] = []
            channel['locations'] = []

        # create an empty HRF object
        estimated_hrf = HRF(
            doi,
            ch_name,
            channel['context']['duration'],
            channel['sfreq'],
            np.asarray(channel['hrf_mean'], dtype=np.float64),
            np.asarray(channel['hrf_std'], dtype=np.float64),
            channel['location'],
            channel['estimates'],
            channel['locations']
        )

        # Insert hrf into tree and attach pointer to channel
        oxygenation = _is_oxygenated(ch_name)
        if oxygenation:
            montage.channels[ch_name] = montage.hbo_tree.insert(estimated_hrf)
            # Add context to tree
            for context in montage.context:
                montage.hbo_tree.hasher.add(context, montage.channels[ch_name])
        elif oxygenation == False:
            montage.channels[ch_name] = montage.hbr_tree.insert(estimated_hrf)
            # Add context to tree
            for context in montage.context:
                montage.hbr_tree.hasher.add(context, montage.channels[ch_name])
    
    montage.sfreq = sfreq # Sampling frequency            

    # Load the HRtree's
    montage.hbo_tree = tree(f"{montage.lib_dir}/hrfs/hbo_hrfs.json", **kwargs)
    montage.hbr_tree = tree(f"{montage.lib_dir}/hrfs/hbr_hrfs.json", **kwargs)
    
    montage.configured = True

    return montage

class montage(tree):
    """
    Class functions:
        - localize_hrfs() - Localizes HRFs to the optodes in a fNIRS montage given a nirx object
        - estimate_hrf() - Deconvolves a fNIRS signal and impulse function to derive the underlying HRF
        - estimate_activity() - Deconvlve a fNIRS scan using estimated HRF's localized to optodes location to gain a neural activity estimate
        - generate_distribution() - Calculates an average HRF and it's standard deviation across time
        - save() - Saves the current montage HRFs
        - load() - Loads a montage of HRFs
        - _merge_montage() - Merges two montages
        - correlate_hrf() - Correlates the HRF estimates across the subject pool to assess similarity
        - correlate_canonical() - Correlates the HRF estimates with a canonical HRF to assess similarity
        - configure() - Configures the montage object to a nirx object

    
    Class attributes:
        - nirx_obj (mne raw object) - NIRX object loaded in via MNE python library
        - sfreq (float) - Sampling frequency of the fNIRS object
        - channels (list) - fNIRS montage channel names
        - subject_estimates (list) - List of subject event-wise HRF estimate
        - channel_estimates (list) - List of channel HRF distribution estimates (position 0 is mean and 1 is std)
    """

    def __init__(self, nirx_obj = None, **kwargs):

        self.root = None # Set an empty root

        # Save runtime parameters to object
        self.lib_dir = os.path.dirname(hrfunc.__file__)

        # Set data context
        self.context = {
                'method': 'toeplitz',
                'doi': 'temp',
                'study': None,
                'task': None,
                'conditions': None,
                'stimulus': None,
                'intensity': 1.0,
                'duration': 30.0,
                'protocol': None,
                'age_range': None,
                'demographics': None
        }
        self.context = {**self.context, **kwargs} # Add user input
        self.context_weights = {context: 1.0 for context in self.context.keys()}

        self.channels = {} # Create variable for holding poiners to each channel
        
        # Load the HRtree's
        self.hbo_tree = tree(f"{self.lib_dir}/hrfs/hbo_hrfs.json", **kwargs)
        self.hbr_tree = tree(f"{self.lib_dir}/hrfs/hbr_hrfs.json", **kwargs)

        self.configured = False
        if nirx_obj:
            # Configure to nirx object passed in
            self.configure(nirx_obj)
            
            # Echo the montage object
            self.__repr__()

    def __repr__(self):
        """
        String representation of the montage object
        
        Returns:
            str - String representation of the montage object
        """
        return f" - Montage object - \nNumber of channels: {len(self.channels)}\n Sampling frequency: {self.sfreq}\nHbO channels (count of {len(self.hbo_channels)}): {self.hbo_channels}\n HbR channels (count of {len(self.hbr_channels)}): {self.hbr_channels}\n - Contexts - \n{'\n'.join([f'{key} - {value} - {self.context_weights[key]}' for key, value in self.context.items()])}\n"

    def localize_hrfs(self, max_distance = 0.01, verbose = False):
        """
        Tries to find local HRFs to each of the fNIRS optodes using the tree structure
        functionality to quickly find nearby HRF's. If it can't it will default to a
        global HRF estimated.

        Arguments:
            max_distance (float) - maximum distance in milimeter's a previously estimated HRF can be attached to an optode
            verbose (bool) - If True, print out extra information during localization

        Returns:
            None
        """

        canonical_hbo = nilearn.glm.first_level.glover_hrf(tr = 1/self.sfreq, oversampling = 1, time_length = float(self.context['duration']))
        canonical_hbr = np.array([-point for point in canonical_hbo])
        canonical_std = [0 for _ in canonical_hbr]

        for ch_name, optode in self.channels.items(): # Iterate through channels apart of nirx data
            if verbose: print(f"Searching for nodes close to optode {optode.ch_name}")
            if _is_oxygenated(ch_name):
                hrf, distance = self.hbo_tree.nearest_neighbor(optode, max_distance, verbose = verbose) # Search in space for similar HRF
            else:
                hrf, distance = self.hbr_tree.nearest_neighbor(optode, max_distance, verbose = verbose)

            if hrf: # If found
                if verbose: print(f"HRF {hrf.ch_name} found at {distance} distance")

                optode.trace = hrf.trace # Add mean and std to montage for channel
                optode.trace_std = hrf.trace_std

            else: # If hrf not found locally
                if verbose: print(f"Local HRF couldn't be found for channel {ch_name}, using canonical")

                if _is_oxygenated(ch_name): # If found
                    optode.trace = canonical_hbo # Add mean and std to montage for channel
                    optode.trace_std = canonical_std
                    optode.context['method'] = 'canonical'
                else: # If global HRF not found
                    optode.trace = canonical_hbr
                    optode.trace_std = canonical_std
                    optode.context['method'] = 'canonical'


    def solve_lstsq(self, lhs, rhs, cond_thresh = None):
        # Compute condition number
        if cond_thresh:
            start = time.time()
            cond_number = np.linalg.cond(lhs)
            end = time.time()
            print(f"np.linalg.cond elapsed time: {end - start:.6f} seconds")
        
        if cond_thresh == None or cond_number < cond_thresh:
            # Stable enough to use least squares with pseudoinverse
            start = time.time()
            estimate, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            end = time.time()
            print(f"np.linalg.lstsq elapsed time: {end - start:.6f} seconds")
        else:
            # Same least squares but with smoothing
            start = time.time()
            estimate = scipy.linalg.pinv(lhs) @ rhs
            end = time.time()
            print(f"scipy.linalg.pinv elapsed time: {end - start:.6f} seconds")
        
        return estimate

    def estimate_hrf(self, nirx_obj, events, duration = 30.0, lmbda = 1e-3, edge_expansion = 0.15, preprocess = True):
        """
        Estimate an HRF subject wise given a nirx object and event impulse series using toeplitz 
        deconvolution with regularization.

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan file loaded in through mne
            events (list) - Event impulse series indicating event occurences during fNIRS scan
            duration (float) - Duration in seconds of the HRF to estimate
            lmbda (float) - Regularization parameter to apply during deconvolution
            edge_expansion (float) - Fraction of the duration to expand the events and duration by to account for toeplitz edge artifacts
            preprocess (bool) - If True, preprocess the fNIRS data before estimating the HRF
        
        Returns:
            None
        """
        if isinstance(duration, float) is False and isinstance(duration, int) is False:
            return ValueError(f"ERROR: Duration passed in must be a float or integer, duration passed in is of type {type(duration)}")
        
        if isinstance(duration, int): duration = float(duration)
        
        if isinstance(events, list) is False:
            return ValueError(f"ERROR: Events passed in must be of type list, object of type {type(events)} was passed in...")

        # Check montage still needs to be configured
        if self.configured is False:
            self.configure(nirx_obj)

        # Convert events to numpy array
        events = np.array(events) 

        # Expand event and duration to account for toeplitz edge artifacts (removed later)
        timeshift = int(round((self.sfreq * duration) * edge_expansion, 0))
        new_events = np.zeros_like(events)
        for ind in range(events.shape[0]): # Iterate through all events
            if events[ind] != 0: # if we found an event
                if (ind - timeshift) < 0: # Check if we can expand the event
                    print("WARNING: An event has been ommited due to edge expansion falling outside of the scan timeframe")
                    continue
                new_events[ind - timeshift] = 1
        
        # Update events and duration to reflect expansion
        events = new_events

        # Update new time HRF estimation duration to account for edge expansion
        duration *= (1 + 2 * edge_expansion)

        nirx_obj.load_data() # Load nirx object
        data = nirx_obj.get_data() # Grab data
        if preprocess:
            nirx_obj = preprocess_fnirs(nirx_obj, deconvolution = True)

        hrf_len = int(round(self.sfreq * duration, 0))  # Calculate HRF length
        scan_len = data.shape[1] # Grab single channel signal length

        if events.shape[0] > scan_len:
            events = events[:scan_len]
            print(f"WARNING: Shortening events for {nirx_obj}")
        elif events.shape[0] != scan_len:
            raise ValueError(f"ERROR: Expected events to be of length {scan_len} but got length {events.shape[0]}...")

        # Build Toeplitz matrix
        X = scipy.linalg.toeplitz(events, np.zeros(hrf_len))
        for fnirs_signal, channel in zip(data[:], nirx_obj.info['chs']) : # For each channel
            print(f"Deconvolving HRF from channel {channel}")
            # Grab channel data and normalize
            #Y = fnirs_signal / np.max(np.abs(fnirs_signal))
            mean = np.mean(fnirs_signal)
            std = np.std(fnirs_signal)
            Y = (fnirs_signal - mean) / std

            # Define regularized least squares equation
            lhs = X.T @ X + lmbda * np.eye(X.shape[1])
            rhs = X.T @ Y

            hrf_estimate = self.solve_lstsq(lhs, rhs)

            # Denormalize HRF estimate
            #hrf_estimate = hrf_estimate * np.max(np.abs(fnirs_signal))
            #hrf_estimate = hrf_estimate * std + mean

            # Adjust the remove the added edges from the hrf_estimate
            start = timeshift
            end = hrf_len - timeshift
            hrf_estimate = hrf_estimate[start:end]

            # Append estimate to channel estimates
            optode = self.channels[standardize_name(channel['ch_name'])]
            optode.estimates.append(list(hrf_estimate))

            # Calculate new centroid for optode given locations used for estimate
            optode.locations.append(list(channel['loc'][:3]))


    def estimate_activity(self, nirx_obj, lmbda = 1e-4, hrf_model = 'toeplitz', preprocess = True, cond_thresh = None, timeout = 1500):
        """
        Deconvlve a fNIRS scan using estimated HRF's localized to optodes location
        to gain a neural activity estimate

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan loaded through mne
            lmbda (float) - Regularization parameter to apply during deconvolution
            hrf_model (str) - HRF model to use during deconvolution, either 'toeplitz' for localized HRF's or 'canonical' for a standard canonical HRF
            preprocess (bool) - If True, preprocess the fNIRS data before estimating the neural activity
        
        Returns:
            None
        """

        # Check montage still needs to be configured
        if self.configured is False:
            self.configure(nirx_obj)
            
        nirx_obj.load_data()
        if preprocess:
            preprocess_fnirs(nirx_obj, deconvolution = True)

        # Define hrf deconvolve function to pass nirx object
        def deconvolution(nirx):

            # Normalize input z-score
            mean = np.mean(nirx)
            std = np.std(nirx)
            Y = (nirx - mean) / std
            Y = np.asarray(Y, dtype=float)

            # Pad HRF to match nirx length
            hrf_kernel = hrf.trace / np.max(np.abs(hrf.trace))
            hrf_kernel = np.asarray(hrf_kernel, dtype=float)

            # Construct Toeplitz convolution matrix (design matrix)
            n_time = len(Y)
            n_hrf = len(hrf_kernel)

            first_col = np.r_[hrf_kernel, np.zeros(n_time - n_hrf)]
            first_row = np.r_[hrf_kernel[0], np.zeros(n_time - 1)]
            A = scipy.linalg.toeplitz(first_col, first_row)
            A = np.asarray(A, dtype=float)

            # Solve the inverse problem with regularization
            lhs = A.T @ A + float(lmbda) * np.eye(A.shape[1])
            rhs = A.T @ Y
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.solve_lstsq, lhs, rhs, cond_thresh)
                try:
                    deconvolved_signal = future.result(timeout)
                    success = True

                except TimeoutError:
                    print("lstsq failed to converge and estimate neural activity, dropping channel")
                    deconvolved_signal = nirx
                    success = False
            
            """
            start = time.time()
            deconvolved_signal = self.solve_lstsq(lhs, rhs)
            end = time.time()
            print(f"Elapsed time: {end - start:.6f} seconds")
            """
            # Denormalize neural signal estimate - EXCLUDING TO KEEP IN A.U.
            #deconvolved_signal = deconvolved_signal * std + mean

            return deconvolved_signal # Return recovered neural signal

        # Apply deconvolution and return the nirx object
        for ch_name, hrf in self.channels.items():
            success = None

            if 'global' in ch_name: continue # Skip if global hrf estimate
            
            # If canonical HRF requested
            if hrf_model == 'canonical' or len(hrf.trace) == 0:
                print(f"WARNING: Using canonical HRF for channel {ch_name} in {nirx_obj}")
                estimate_hrf = hrf # Temporarily replace HRF
                if _is_oxygenated(ch_name): # with oxygenated canonical
                    hrf = self.hbo_tree.root.right
                else: # with deoxygenated canonical
                    hrf = self.hbr_tree.root.right

            # Figure out which channel to apply to
            for nirx_channel in nirx_obj.info['chs']:
                standard_ch_name = standardize_name(nirx_channel['ch_name'])
                if ch_name == standard_ch_name:
                    break
                
            print(f"Deconvolving channel {ch_name}...") # Apply deconvolution
            nirx_obj.apply_function(deconvolution, picks = [nirx_channel['ch_name']]) # Apply deconvolution for channel

            # Remove channel is neural activity estimation failed to converge
            if success == False:
                nirx_obj.drop_channels([nirx_channel['ch_name']])

            # Replace the canonical HRF estimate temporarily used with the HRF estimate
            if hrf_model == 'canonical':
                hrf = estimate_hrf # Replace the original HRF

    def generate_distribution(self, plot_dir = None):
        """
        Calculate average and standard deviation of HRF across subjects for each channel

        Arguments:
            plot_dir (str) - If provided, will save a plot of each channel's HRF to the given directory
        """
        hbr_estimates = []
        hbo_estimates = []

        for channel in self.channels.keys():
            # Grab channel optode attached to montage
            optode = self.channels[channel]

            # Calculate average HRF estimate and standard deviation
            optode.trace = np.mean(optode.estimates, axis = 0)
            optode.trace_std = np.std(optode.estimates, axis = 0)
            
            # Update centroid attached to optode with average location
            optode.update_centroid()

            if plot_dir: # Plot if requested
                optode.plot(plot_dir)
            
            if optode.oxygenation: # Append data
                hbo_estimates.append(optode.trace)
            else:
                hbr_estimates.append(optode.trace)

        # Calculate global HRF mean and standard deviation
        for oxygenation, estimates in zip([True, False], [hbo_estimates, hbr_estimates]):
            type_estimates = np.vstack(estimates)
            global_mean = np.mean(type_estimates, axis = 0)
            global_std = np.std(type_estimates, axis = 0)

            # Create a global HRF variable
            global_location = [360 + random.random(), 360 + random.random(), 360 + random.random()]
            global_hrf = HRF(
                doi = self.context['doi'],
                ch_name = ("global_hbo" if oxygenation else "global_hbr"),
                duration = self.context['duration'],
                sfreq = self.sfreq,
                trace = global_mean,
                trace_std = global_std,
                location = global_location,
                estimates = [list(global_mean)],
                locations = [list(global_location)]
            )
            #Insert global hrf into tree and attach pointer to channels dict
            if oxygenation:
                self.channels['global_hbo'] = self.insert(global_hrf)
            else:
                self.channels['global_hbr'] = self.insert(global_hrf)

    def correlate_hrf(self, plot_filename = "montage_correlation.png"):
        """
        Correlate the HRF estimates across the subject pool to assess similarity

        Arguments:
            plot_filename (str) - Filename to save correlation plot to
        
        Returns:
            corr_matrix (np.array) - Correlation matrix of HRF estimates
        """
        corr_matrix = np.zeros((len(self.hbo_channels), len(self.hbr_channels), 2))
        
        # Calculate correlation coefficients and p-values between HbO and HbR channels
        for hbo_ind, hbo_channel in enumerate(self.hbo_channels):
            hbo_hrf = self.channels[hbo_channel].trace

            for hbr_ind, hbr_channel in enumerate(self.hbr_channels):
                hbr_hrf = self.channels[hbr_channel].trace
                
                corr_coefficient, p_value = scipy.stats.spearmanr(hbo_hrf, hbr_hrf)
                
                corr_matrix[hbo_ind, hbr_ind, 0] = corr_coefficient
                corr_matrix[hbo_ind, hbr_ind, 1] = p_value

        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, :, 0], cmap='viridis', aspect='auto')
        plt.colorbar(label='Correlation Coefficient')
        plt.title('Correlation Matrix of HRF Estimates')
        plt.xlabel('HbR Channels')
        plt.ylabel('HbO Channels')
        plt.xticks(range(len(self.hbr_channels)), self.hbr_channels, rotation=90)
        plt.yticks(range(len(self.hbo_channels)), self.hbo_channels)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()

        # Plot p-values
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, :, 1], cmap='viridis', aspect='auto')
        plt.colorbar(label='P-value')
        plt.title('P-values of Correlation between HRF Estimates')
        plt.xlabel('HbR Channels')
        plt.ylabel('HbO Channels')
        plt.xticks(range(len(self.hbr_channels)), self.hbr_channels, rotation=90)
        plt.yticks(range(len(self.hbo_channels)), self.hbo_channels)
        plt.tight_layout()
        plt.savefig(plot_filename.replace(".png", "_pvalues.png"))
        plt.close()

        # Save the correlation matrix to a file
        with open("correlation_matrix.json", "w") as f:
            json.dump(corr_matrix.tolist(), f, indent=4)
        
        return corr_matrix

    def correlate_canonical(self, plot_filename = "canonical_correlation.png", duration = 30.0):
        """
        Correlate the HRF estimates with a canonical HRF to assess similarity

        Arguments:
            plot_filename (str) - Filename to save correlation plot to
            duration (float) - Duration in seconds of the canonical HRF to generate
        Returns:
            None
        """
        # Generate canonical HRF
        time_stamps = np.arange(0, len(self.root.trace), 1)

        # Parameters for the double-gamma HRF
        peak1 = scipy.stats.gamma.pdf(time_stamps, 6) # peak at ~6s
        peak2 = scipy.stats.gamma.pdf(time_stamps, 16) / 6.0 # undershoot at ~16s

        canonical_hrf = peak1 - peak2
        canonical_hrf /= np.max(canonical_hrf)  # Normalize peak to 1
        corr_matrix = np.zeros((len(self.hbo_channels) + len(self.hbr_channels), 2))
        for ind, ch_name in enumerate(self.hbo_channels + self.hbr_channels):
            hrf = self.channels[ch_name]

            corr_coefficient, p_value = scipy.stats.spearmanr(canonical_hrf, hrf.trace)
            corr_matrix[ind, 0] = corr_coefficient
            corr_matrix[ind, 1] = p_value

        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, 0][np.newaxis, :], cmap='viridis', aspect='auto')
        plt.colorbar(label='Correlation Coefficient')
        plt.title('Correlation Matrix of HRF Estimates with Cannonical HRF')
        plt.xlabel('Montage Channels')
        plt.ylabel('Cannonical HRF')
        plt.xticks(range(len(self.hbo_channels) + len(self.hbr_channels)), self.hbo_channels + self.hbr_channels, rotation=90)
        plt.yticks(range(1), ['Canonical'])
        plt.tight_layout()

        plt.savefig(plot_filename)
        plt.close()

        # Plot p-values
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, 1][np.newaxis, :], cmap='viridis', aspect='auto')
        plt.colorbar(label='P-value')
        plt.title('P-values of Correlation with Cannonical HRF')
        plt.xlabel('Montage Channels')
        plt.ylabel('Cannonical HRF')
        plt.xticks(range(len(self.hbo_channels) + len(self.hbr_channels)), self.hbo_channels + self.hbr_channels, rotation=90)
        plt.yticks(range(1), ['Canonical'])
        plt.tight_layout()
        plt.savefig(plot_filename.replace(".png", "_pvalues.png"))
        plt.close()
        return

    def configure(self, nirx_obj, **kwargs):
        """
        Configure the montage object to a nirx object

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan file loaded in through
            **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)
        
        Returns:
            None"""
        print(f"Configureding HRfunc montage...")
        self.sfreq = nirx_obj.info['sfreq'] # Sampling frequency            

        self.hbo_channels = [standardize_name(ch) for ch in nirx_obj.ch_names if _is_oxygenated(ch) == True]
        self.hbr_channels = [standardize_name(ch) for ch in nirx_obj.ch_names if _is_oxygenated(ch) == False]

        # Merge nirx object into montage
        self._merge_montages(nirx_obj) # Add empty HRF nodes to the tree for each HRF

        self.configured = True



    def save(self, filename = 'montage_hrfs.json'):
        """
        Save the hrf montage

        Arguments:
            Filename (str) - Filename to save the montage HRFs as

        Returns:
            None
        """
        hrfs = self.gather(self.root)
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return
    
    def _merge_montages(self, nirx_obj, verbose = False):
        """
        Function to merge a NIRX object montage with the HRFunc montage.
        This function should only be used when initializing an empty
        montage or if merging nirx objects with the same NIRS montage layout
        and with different channel names (useful when dealing with multiple
        data collection sites with slightly different setups in channel naming).
        
        WARNING: Merging two distinctly different montages is not recommended.
        Inaccurate HRF may be estimated depending on how the merged montage
        is used.

        Arguments:
            nirx_obj (mne NIRX object) - MNE NIRS scan recording loading in through MNE
            verbose (bool) - If True, print out extra information during localization
        
        Returns:
            None
        """
        # Add each nirx object channel to the hrfunc.montage
        for channel in nirx_obj.info['chs']:
                # (Re)set runtime variables
                results = None

                # Grab pertinent info from nirx header
                ch_name = standardize_name(channel['ch_name'])
                location = channel['loc'][:3]

                # Skip if canonical HRF
                if ch_name == 'canonical':
                    continue

                empty_hrf = HRF(
                    self.context['doi'],
                    ch_name, 
                    self.context['duration'], 
                    self.sfreq,
                    [], 
                    [], 
                    location,
                    [],
                    [],
                    []
                )

                # Check if an HRF in this area already exists 
                # NOTE: This is necessary to localize nodes with slight
                # channel name differences in the same location
                print(f"Searching for {ch_name}")
                best_node, distance = self.nearest_neighbor(empty_hrf, max_distance = 1e-9, verbose = verbose)
                if best_node and best_node.ch_name[:9] != 'canonical': # If previously defined channel hrf found
                    print(f"Local HRF found in the channel {best_node.ch_name}, merging with optode {ch_name}")
                    self.channels[ch_name] = best_node # Attach node to channel
                else: # If new channel
                    # create an empty HRF object
                    print(f"No matching HRF found, inserting new HRF")
                    self.channels[ch_name] = self.insert(empty_hrf) # Insert empty hrf into montage tree

    def _merge_trees(self, filename = 'tree_hrfs.json'):
        """
        Merge montage, HbO and HbR trees. This function is meant to be used
        by the creators of HRFunc to merge submitted HRF estimates with the
        HRF toolbox

        Arguments:
            Filename (str) - Filename to save the montage HRFs as
        """
        hrfs = self.gather(self.hbo_tree.root)
        hrfs |= self.gather(self.hbr_tree.root)
        hrfs |= self.gather(self.root)
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return

def preprocess_fnirs(scan, deconvolution = False):
    """
    Preprocess fNIRS data in an MNE Raw object.

    Steps:
    - Optical density conversion
    - Scalp coupling index evaluation and bad channel marking
    - Motion artifact correction using TDDR
    - Optional polynomial detrending for deconvolution
    - Haemoglobin conversion via Beer-Lambert Law
    - Optional bandpass filtering for GLM-based analysis

    Parameters:
    - scan (mne.io.Raw) - The raw fNIRS MNE object to preprocess.
    - deconvolution (bool) If True, performs detrending and skips filtering.

    Returns:
    - haemo (mne.io.Raw) - Preprocessed data with haemoglobin concentration channels.
    """

    scan.load_data()

    raw_od = mne.preprocessing.nirs.optical_density(scan)

    # scalp coupling index
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.95))

    if len(raw_od.info['bads']) == len(scan.ch_names):
        print("All channels are bad, skipping subject...")
        return

    if len(raw_od.info['bads']) > 0:
        print("Bad channels in subject", raw_od.info['subject_info']['his_id'], ":", raw_od.info['bads'])

    # Interpolate bad channels
    raw_od.interpolate_bads(reset_bads=False)

    # temporal derivative distribution repair (motion attempt)
    od = mne.preprocessing.nirs.tddr(raw_od)

    # If running deconvolution, polynomial detrend to remove pysiological without cutting into the frequency spectrum
    if deconvolution:
        od = polynomial_detrend(od, order=1)

    # haemoglobin conversion using Beer Lambert Law 
    haemo = mne.preprocessing.nirs.beer_lambert_law(od.copy(), ppf=0.1)

    haemo = baseline_correct(haemo, baseline=(None, 0.0))

    if not deconvolution:
        haemo.filter(0.01, 0.2)

    return haemo

def baseline_correct(raw, baseline=(None, 0.0)):
    """
    Apply baseline correction to fNIRS data.
    
    Parameters:
    - raw (mne.io.Raw) - The raw fNIRS MNE object to baseline correct.
    - baseline (tuple) - The time interval for baseline correction. 
    
    Returns:
    - raw (mne.io.Raw) - Baseline corrected fNIRS data."""
    return raw.apply_function(lambda x: mne.baseline.rescale(x, times=raw.times, baseline=baseline, mode='mean'), picks='data')

def polynomial_detrend(raw, order = 1):
    raw_detrended = raw.copy()
    times = raw.times
    times_scaled = (times - times.mean()) / times.std()  # or just (times - mean)
    X = np.vander(times_scaled, N=order + 1, increasing=True)

    for idx in range(len(raw.ch_names)):
        y = raw.get_data(picks = idx)[0]
        beta = np.linalg.lstsq(X.T @ X, X.T @ y, rcond = None)[0]
        y_detrended = y - X @ beta
        raw_detrended._data[idx] = y_detrended

    return raw_detrended
 
def standardize_name(ch_name):
    """
    Standardize channel names to a common format for processing
    
    Arguments:
        ch_name (str) - Original channel name
        
    Returns:
        ch_name (str) - Standardized channel name"""
    ch_name = re.sub(r'[_\-\s]+', '_', ch_name.lower())
    oxygenation = _is_oxygenated(ch_name)
    if oxygenation:
        ch_name = ch_name[:-3] + 'hbo'
    else:
        ch_name = ch_name[:-3] + 'hbr'
    return ch_name

def _is_oxygenated(ch_name):
    """ Check in whether the channel is HbR or HbO 
    
    Arguments:
        ch_name (str) - Channel name to check
    
    Returns:
        bool - True if oxygenated, False if deoxygenated"""
    if ch_name[-2] == 'b':
        split = ch_name.split('hb')
        if split[1][0] == 'o': # If oxygenated channel
            return True
        elif split[1][0] == 'r': # If deoxygenated channel
            return False
        else:
            raise ValueError(f"Channel {ch_name} oxygenation status could not be determines, ensure each channel has appropriate naming scheme with HbO/HbR included")
    elif ch_name[-1] == '0':
        try:
            wavelength = int(ch_name[-3:])
            if wavelength >= 760 and wavelength <= 780:
                return False
            elif wavelength >= 830 and wavelength <= 850:
                return True
            else:
                LookupError(f"Wavelength found, but failed to evaluate oxygenation status of channel {ch_name}")
        except:
            LookupError(f"Failed to evaluate oxygenation status of channel {ch_name}")