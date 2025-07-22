import scipy.linalg, json, mne, random, re, os, hrfunc
import numpy as np
import matplotlib.pyplot as plt
from .hrtree import tree, HRF
from itertools import compress
from glob import glob


def locate_hrfs(nirx_obj, similarity_threshold = 0.95, **kwargs):
    """
    Locate local HRF's for the nirx object and return a montage with found HRF's

    Arguments:
        nirx_obj (mne raw object) - NIRS file loaded through mne
    """
    # Build a montage
    _montage = montage(nirx_obj, **kwargs)
    _montage.load_hrfs()
    return _montage

class montage(tree):
    """
    Class functions:
        - localize_hrf() - Tries to find a previously derived HRFs localized to the same region
        - convolve_hrf() - Convolves an impulse function (series of 0's and 1's) with an HRF
        - deconvolve_hrf() - Deconvolves a fNIRS signal and impulse function to derive the underlying HRF
        - generate_distribution() - Calculates an average HRF and it's standard deviation across time
        - save() - Saves the current montage HRFs
        - load() - Loads a montage of HRFs
        - merge() - Merges two montages with the same layout
    
    Class attributes:
        - nirx_obj (mne raw object) - NIRX object loaded in via MNE python library
        - sfreq (float) - Sampling frequency of the fNIRS object
        - channels (list) - fNIRS montage channel names
        - subject_estimates (list) - List of subject event-wise HRF estimate
        - channel_estimates (list) - List of channel HRF distribution estimates (position 0 is mean and 1 is std)
    """

    def __init__(self, nirx_obj = None, hrfs_filename = None, **kwargs):
        if nirx_obj is None and hrfs_filename is None: # Check if enough info sent in
            raise ValueError(f"A NIRX object or a previously estimated HRFs.json must be passed in to initialize a HRFunc.montage")

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

        self.hbo_channels = [ch for ch in nirx_obj.ch_names if _is_oxygenated(ch) == True]
        self.hbr_channels = [ch for ch in nirx_obj.ch_names if _is_oxygenated(ch) == False]

        self.channels = {} # Create variable for holding poiners to each channel
        
        self.sfreq = nirx_obj.info['sfreq'] # Sampling frequency            

        self.hbo_tree = tree(f"{self.lib_dir}/hrfs/hbo_hrfs.json", **kwargs)
        self.hbr_tree = tree(f"{self.lib_dir}/hrfs/hbr_hrfs.json", **kwargs)

        # Merge nirx object into montage
        if hrfs_filename and os.path.exists(hrfs_filename):
            print(f"Loading in previous estimates {hrfs_filename}")
            self.load_montage(hrfs_filename)
        else:
            print(f"Creating new HRF montage from {nirx_obj}")
            self._merge_montages(nirx_obj) # Add empty HRF nodes to the tree for each HRF
        
        self.__repr__()

    def __repr__(self):
        return f" - Montage object - \nNumber of channels: {len(self.channels)}\n Sampling frequency: {self.sfreq}\nHbO channels (count of {len(self.hbo_channels)}): {self.hbo_channels}\n HbR channels (count of {len(self.hbr_channels)}): {self.hbr_channels}\n - Contexts - \n{'\n'.join([f'{key} - {value} - {self.context_weights[key]}' for key, value in self.context.items()])}\n"

    def localize_hrfs(self, max_distance = 0.01):
        """
        Tries to find local HRFs to each of the fNIRS optodes using the tree structure
        functionality to quickly find nearby HRF's. If it can't it will default to a
        global HRF estimated.

        Arguments:
            max_distance (float) - maximum distance in milimeter's a previously estimated HRF can be attached to an optode
        """
        
        for ch_name, optode in self.channels.items(): # Iterate through channels apart of nirx data
            if _is_oxygenated(ch_name):
                hrf = self.hbo_tree.nearest_neighbor(optode, max_distance) # Search in space for similar HRF
            else:
                hrf = self.hbr_tree.nearest_neighbor(optode, max_distance)

            if hrf: # If found
                optode.trace = hrf.trace # Add mean and std to montage for channel
                optode.trace_std = hrf.trace_std

            else: # If hrf not found locally
                print(f"Local HRF with given context couldn't be found for channel {ch_name}, searching for global HRF")
                # Adjust channel location temporarily to global node nexus 360
                ch_location = [optode.x, optode.y, optode.z] 
                optode.x, optode.y, optode.z = 360.0, 360.0, 360.0
                
                # Search for global HRF with similar context
                hrf = self.nearest_neighbor(self.channels[ch_name], max_distance)
                if hrf: # If found
                    optode.trace = hrf.trace # Add mean and std to montage for channel
                    optode.trace_std = hrf.trace_std
                else: # If global HRF not found
                    LookupError(f"Global HRF with given context could not be found for {ch_name}")
                
                # Replace global location with original optode location
                optode.x, optode.y, optode.z = ch_location[0], ch_location[1], ch_location[2]

    def convolve_hrf(self, events, hrf):
        """ Convolve an event impulse series with an hrf to create an expected signal. Both
        inputs must be the same length.

        Argument:
            events (list) - List of 0's and 1's indicating event occurances
            hrf (list) - HRF estimated trace
        """
        return np.convolve(events, hrf, mode='full')

    def deconvolve_hrf(self, nirx_obj, events, duration = 30.0, _lambda = 1e-3, edge_expansion = 0.15, preprocess = True):
        """
        Estimate an HRF subject wise given a nirx object and event impulse series using toeplitz 
        deconvolution with regularization.

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan file loaded in through mne
            events (list) - Event impulse series indicating event occurences during fNIRS scan
        """
        if isinstance(duration, float) == False:
            return ValueError(f"Duration passed in must be a float or integer, duration passed in is of type {type(duration)}")
        
        if isinstance(duration, int): duration = float(duration)
        
        if isinstance(events, list) == False:
            return ValueError(f"Events passed in must be of type list, object of type {type(events)} was passed in...")

        events = np.array(events) # Convert events to numpy array

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
            preprocess_fnirs(nirx_obj, deconvolution = True)

        hrf_len = int(round(self.sfreq * duration, 0))  # Calculate HRF length
        print(f"HRF length: {hrf_len}")
        scan_len = data.shape[1] # Grab single channel signal length

        if events.shape[0] > scan_len:
            events = events[:scan_len]
            print(f"Warning: Shortening events for {nirx_obj}")
        elif events.shape[0] != scan_len:
            raise ValueError(f"Expected events to be of length {scan_len} but got length {events.shape[0]}...")
    
        # Build Toeplitz matrix
        X = scipy.linalg.toeplitz(events, np.zeros(hrf_len))
        for fnirs_signal, channel in zip(data[:], nirx_obj.info['chs']) : # For each channel
            # Grab channel data and normalize
            #Y = fnirs_signal / np.max(np.abs(fnirs_signal))
            mean = np.mean(fnirs_signal)
            std = np.std(fnirs_signal)
            Y = (fnirs_signal - mean) / std

            # Define regularized least squares equation
            lhs = X.T @ X + _lambda * np.eye(X.shape[1])
            rhs = X.T @ Y

            try: # Try estimating with standard least squares
                hrf_estimate, *_ = np.linalg.lstsq(lhs, rhs, rcond = None)
            except np.linalg.LinAlgError: # If that fails, try applying the same with smoothing
                hrf_estimate = scipy.linalg.pinv(lhs) @ rhs

            # Denormalize HRF estimate
            #hrf_estimate = hrf_estimate * np.max(np.abs(fnirs_signal))
            #hrf_estimate = hrf_estimate * std + mean

            # Adjust the remove the added edges from the hrf_estimate
            start = timeshift
            end = hrf_len - timeshift
            hrf_estimate = hrf_estimate[start:end]
            print(f"HRF Estimate (length of {len(hrf_estimate)}): {channel['ch_name']}\n{hrf_estimate}\n{nirx_obj}")

            # Append estimate to channel estimates
            print(f"Channels: {", ".join(self.channels.keys())}")
            self.channels[channel['ch_name']].estimates.append(hrf_estimate)

            print(f"Single HRF estimate for {channel['ch_name']}: {hrf_estimate}")


    def deconvolve_activity(self, nirx_obj, _lambda = 1e-4, hrf_model = 'toeplitz', preprocess = True):
        """
        Deconvlve a fNIRS scan using estimated HRF's localized to optodes location
        to gain a neural activity estimate

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan loaded through mne
            events (list) - event impulse sequence of 0's and 1's
        """
            
        nirx_obj.load_data()
        if preprocess:
            preprocess_fnirs(nirx_obj, deconvolution = True)

        # Define hrf deconvolve function to pass nirx object
        def deconvolution(nirx):
            original_len = len(nirx)

            print("Original mean/std:", np.mean(nirx), np.std(nirx))
        
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
            lhs = A.T @ A + float(_lambda) * np.eye(A.shape[1])
            rhs = A.T @ Y
            try: # Try using standard linear least squared to solve
                deconvolved_signal, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            except np.linalg.LinAlgError as e: # If failed try to run pinv with smoothing
                print("Linear algebra error:", e)
                deconvolved_signal = scipy.linalg.pinv(lhs) @ rhs

            # Denormalize neural signal estimate
            #deconvolved_signal = deconvolved_signal * std + mean

            return deconvolved_signal # Return recovered neural signal

        # Apply deconvolution and return the nirx object
        for ch_name, hrf in self.channels.items():
            if 'global' in ch_name: continue # Skip if global hrf estimate
            
            # If canonical HRF requested
            if hrf_model == 'canonical':
                print(f"Using canonical HRF for {nirx_obj}")
                estimate_hrf = hrf # Temporarily replace HRF
                if _is_oxygenated(ch_name): # with oxygenated canonical
                    hrf = self.hbo_tree.root.right
                else: # with deoxygenated canonical
                    hrf = self.hbr_tree.root.right


            # Figure out which channel to apply to
            for nirx_channel in nirx_obj.info['chs']:
                standard_ch_name = re.sub(r'[_\-\s]+', '_', nirx_channel['ch_name'].lower())
                if ch_name == standard_ch_name:
                    break
                
            print(f"Deconvolving channel {ch_name}...") # Apply deconvolution
            nirx_obj.apply_function(deconvolution, picks = [nirx_channel['ch_name']]) # Apply deconvolution for channel
        
            if hrf_model == 'canonical':
                hrf = estimate_hrf # Replace the original HRF
        return nirx_obj

    def generate_distribution(self, plot_dir = None):
        """
        Calculate average and standard deviation of HRF across subjects for each channel

        Arguments:
            duration (float) - Duration in seconds of the HRF to estimate
        """
        hbr_estimates = []
        hbo_estimates = []

        for channel in self.channels.keys():
            optode = self.channels[channel]
            optode.trace = np.mean(optode.estimates, axis = 0)
            optode.trace_std = np.std(optode.estimates, axis = 0)
            if plot_dir:
                optode.plot(plot_dir)
            
            if optode.oxygenation:
                hbo_estimates.append(optode.trace)
            else:
                hbr_estimates.append(optode.trace)

        # Calculate global HRF mean and standard deviation
        for oxygenation, estimates in zip([True, False], [hbo_estimates, hbr_estimates]):
            type_estimates = np.vstack(estimates)
            global_mean = np.mean(type_estimates, axis = 0)
            global_std = np.std(type_estimates, axis = 0)

            # Create a global HRF variable
            global_hrf = HRF(
                doi = self.context['doi'],
                ch_name = ("global_hbo" if oxygenation else "global_hbr"),
                duration = self.context['duration'],
                sfreq = self.sfreq,
                trace = global_mean,
                trace_std = global_std,
                location = [360 + random.random(), 360 + random.random(), 360 + random.random()]
            )
            #Insert global hrf into tree and attach pointer to channels dict
            if oxygenation:
                self.channels['global_hbo'] = self.insert(global_hrf)
            else:
                self.channels['global_hbr'] = self.insert(global_hrf)
        
    def correlate_hrf(self, plot_filename = "montage_correlation.png"):
        """
        Correlate the HRF estimates across the subject pool to assess similarity
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


    def save(self, filename = 'montage_hrfs.json'):
        """
        Save the hrf montage

        Arguments:
            Filename (str) - Filename to save the montage HRFs as
        """
        hrfs = self.gather(self.root)
        print(f"Final HRF: {hrfs.keys()}")
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return
    

    def load_montage(self, json_filename):
        """ Load montage with the given json filename """
        # Read in json
        with open(json_filename, 'r') as file:
            json_contents = json.load(file)

        # Update montage with saved info
        for key, channel in json_contents.items():
            key_split = key.split('-')
            doi = key_split.pop()
            ch_name = '-'.join(key_split)

            # Skip if canonical HRF
            if ch_name == 'canonical':
                continue

            # create an empty HRF object
            empty_hrf = HRF(
                doi,
                ch_name, 
                channel['context']['duration'], 
                self.sfreq, 
                np.asarray(channel['hrf_mean'], dtype=np.float64), 
                np.asarray(channel['hrf_std'], dtype=np.float64), 
                channel['location']
            )

            # Insert empty hrf into tree and attach pointer to channel
            oxygenation = _is_oxygenated(ch_name)
            if oxygenation:
                self.channels[ch_name] = self.hbo_tree.insert(empty_hrf)
                # Add context to tree
                for context in self.context:
                    self.hbo_tree.hasher.add(context, self.channels[ch_name])
            elif oxygenation == False:
                self.channels[ch_name] = self.hbr_tree.insert(empty_hrf)
                # Add context to tree
                for context in self.context:
                    self.hbr_tree.hasher.add(context, self.channels[ch_name])
        return
    
    def _merge_montages(self, nirx_obj):
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
        """
        # Add each nirx object channel to the hrfunc.montage
        for channel in nirx_obj.info['chs']:
                # (Re)set runtime variables
                results = None

                # Grab pertinent info from nirx header
                ch_name = channel['ch_name']
                location = channel['loc'][:3]

                # Skip if canonical HRF
                if ch_name == 'canonical':
                    continue

                # create an empty HRF object
                empty_hrf = HRF(
                    self.context['doi'],
                    ch_name, 
                    self.context['duration'], 
                    self.sfreq, 
                    [], 
                    [], 
                    location,
                    []
                )

                # Check if an HRF in this area already exists 
                # NOTE: This is necessary to localize nodes with slight
                # channel name differences in the same location
                if empty_hrf.oxygenation:
                    results = self.nearest_neighbor(empty_hrf, max_distance = 1e-9)
                if results and results[0].doi != 'canonical': # If previously defined channel hrf found
                    print(f"Local HRF found in the channel {results}\n merging with optode {ch_name}")
                    self.channels[ch_name] = results # Attach node to channel
                else: # If new channel
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
    - scan: mne.io.Raw
        The raw fNIRS MNE object to preprocess.
    - deconvolution: bool
        If True, performs detrending and skips filtering.

    Returns:
    - haemo: mne.io.Raw
        Preprocessed data with haemoglobin concentration channels.
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

def _load_fnirs(nirs_filename):
    """ Load the fNIRS file based on the format found """
    if nirs_filename[-1:] == "/": # If folder passed in
        try:
            nirs_obj = mne.io.read_raw_nirx(nirs_filename)
        except:
            raise ValueError(f"NIRS folder {nirs_filename} failed to load")
    if nirs_filename[-4:] == '.fif': # If fif file format
        try:
            nirs_obj = mne.io.read_raw_fif(nirs_filename)
        except:
            raise ValueError(f"NIRX .fif file {nirs_filename} failed to load")
    if nirs_filename[-6:] == ".snirf": # If snirf format
        try:
            nirs_obj = mne.io.read_raw_snirf(nirs_filename)
        except:
            raise ValueError(f"SNIRF file {nirs_filename} passed in failed to load")
    return nirs_obj
 
def _is_oxygenated(ch_name):
    """ Check in whether the channel is HbR or HbO """
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
