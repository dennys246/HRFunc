import scipy.linalg, json, mne, random, re
from .hrtree import tree, HRF
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def estimate_hrfs(nirx_folder, nirx_identifier, events, hrfs_filename = "hrf_estimates.json", plot_dir = None, **kwargs):
    """
    This function is the primary call for estimating an HRF across a subject pool
    fNIRS data. To accomplish this, the function creates a hrfunc.montage and for
    each nirx file found using the nirx_folder and nirx_identifier estimates an 
    event wise HRF then generates a channel wise distribution after deconvolving all
    available subjects.
    """
    # Set data context
    context = {
            'type': 'global',
            'doi': 'temp',
            'study': None,
            'task': None,
            'conditions': None,
            'stimulus': None,
            'intensity': None,
            'duration': 12.0,
            'protocol': None,
            'age_range': None,
            'demographics': None
    }
    context = {**context, **kwargs} # Add user input

    # Grab all available nirx files
    nirx_files = glob(f"{nirx_folder}/**/{nirx_identifier}")
    
    _montage = montage(_load_fnirs(nirx_files[0]), hrfs_filename, **kwargs)

    for nirx_filename in nirx_files: # For each nirx object
        nirx_obj = _load_fnirs(nirx_filename) # Load the nirx

        _montage.deconvolve_hrf(nirx_obj, events) # Estimate the HRF

    _montage.generate_distribution(context['duration'], plot_dir) # Generate HRF distribution

    _montage.save(hrfs_filename, context['doi'], **kwargs) # Save montage

    return _montage


def locate_hrfs(nirx_obj, hrfs_filename = 'hrfs.json', **kwargs):
    """
    Locate local HRF's for the nirx object and return a montage with found HRF's

    Arguments:
        nirx_obj (mne raw object) - NIRS file loaded through mne
    """
    # Build a montage
    _montage = montage(nirx_obj, hrfs_filename, **kwargs)
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

    def __init__(self, nirx_obj = None, hrfs_filename = None, deconv_method = None, conv_method = None, **kwargs):
        if nirx_obj is None and hrfs_filename is None: # Check if enough info sent in
            raise ValueError(f"A NIRX object or a previously estimated HRFs.json must be passed in to initialize a HRFunc.montage")

        self.root = None # Set an empty root

        # Set data context
        self.context = {
                'type': 'global',
                'doi': 'temp',
                'study': None,
                'task': None,
                'conditions': None,
                'stimulus': None,
                'intensity': 1.0,
                'duration': 12.0,
                'protocol': None,
                'age_range': None,
                'demographics': None
        }
        self.context = {**self.context, **kwargs} # Add user input
        self.context_weights = {context: 1.0 for context in self.context.keys()}

        # Initialize an empty tree
        self.hbo_tree = tree("deoxy_hrfs.json", **kwargs)
        self.hbr_tree = tree("oxy_hrfs.json", **kwargs)

        self.hbo_channels = [ch for ch in nirx_obj.ch_names if ch.endswith('hbo')]
        self.hbr_channels = [ch for ch in nirx_obj.ch_names if ch.endswith('hbr')]

        self.channels = {} # Create variable for holding poiners to each channel
        
        self.sfreq = nirx_obj.info['sfreq'] # Sampling frequency

        if hrfs_filename: # If previously estimated hrfs provided, load in
            self.load(hrfs_filename)
        
        else: # Merge the NIRX object montage into the hrfunc.montage object
            self._merge_montages(nirx_obj) # Add empty HRF nodes to the tree for each HRF
        
        self.__repr__()

    def __repr__(self):
        return f" - Montage object - \nNumber of channels: {len(self.channels)}\n Sampling frequency: {self.sfreq}\nHbO channels (count of {len(self.hbo_channels)}): {self.hbo_channels}\n HbR channels (count of {len(self.hbr_channels)}): {self.hbr_channels}\n\n"

    def localize_hrfs(self, max_distance = 3.0):
        """
        Tries to find local HRFs to each of the fNIRS optodes using the tree structure
        functionality to quickly find nearby HRF's. If it can't it will default to a
        global HRF estimated.

        Arguments:
            max_distance (float) - maximum distance in milimeter's a previously estimated HRF can be attached to an optode
        """
        
        for ch_name, optode in self.channels.items(): # Iterate through channels apart of nirx data
            if _is_oxygenated(ch_name):
                hrf = self.hbo_tree.search_dfs(optode, max_distance) # Search in space for similar HRF
            else:
                hrf = self.hbr_tree.search_dfs(optode, max_distance)

            if hrf: # If found
                optode.trace = hrf.trace # Add mean and std to montage for channel
                optode.trace_std = hrf.trace_std

            else: # If hrf not found locally
                print(f"Local HRF with given context couldn't be found for channel {ch_name}, searching for global HRF")
                # Adjust channel location temporarily to global node nexus 360
                ch_location = [optode.x, optode.y, optode.z] 
                optode.x, optode.y, optode.z = 360, 360, 360
                
                # Search for global HRF with similar context
                hrf = self.search_bfs(self.channels[ch_name], max_distance)
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

    def deconvolve_hrf(self, nirx_obj, events, duration = 12.0, _lambda = 1e-3, edge_expansion = 0.15, plot_dir = None):
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

        hrf_len = int(round(self.sfreq * duration, 0))  # Calculate HRF length
        scan_len = data.shape[1] # Grab single channel signal length

        if events.shape[0] > scan_len:
            events = events[:scan_len]
            print(f"Warning: Shortening events for {nirx_obj}")
        elif events.shape[0] != scan_len:
            raise ValueError(f"Expected events to be of length {scan_len} but got length {events.shape[0]}...")
    
        print(f"Events: {events}")
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
            start = int(round(hrf_len * edge_expansion, 0))
            end = hrf_len - start
            hrf_estimate = hrf_estimate[start:end]

            # Append estimate to channel estimates
            self.channels[channel['ch_name']].estimates.append(hrf_estimate)
        
        self.generate_distribution(plot_dir)


    def deconvolve_activity(self, nirx_obj, _lambda = 1e-4, **kwargs):
        """
        Deconvlve a fNIRS scan using estimated HRF's localized to optodes location
        to gain a neural activity estimate

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan loaded through mne
            events (list) - event impulse sequence of 0's and 1's
        """
            
        nirx_obj.load_data()

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
            
            # Figure out which channel to apply to
            for nirx_channel in nirx_obj.info['chs']:
                standard_ch_name = re.sub(r'[_\-\s]+', '_', nirx_channel['ch_name'].lower())
                if ch_name == standard_ch_name:
                    break
            print(f"Deconvolving channel {ch_name}...") # Apply deconvolution
            nirx_obj.apply_function(deconvolution, picks = [nirx_channel['ch_name']]) # Apply deconvolution for channel
        
        return nirx_obj

    def generate_distribution(self, plot_dir = None):
        """
        Calculate average and standard deviation of HRF across subjects for each channel

        Arguments:
            duration (float) - Duration in seconds of the HRF to estimate
        """
        hbr_estimates = []
        hbo_estimates = []

        def estimate(optode, plot_dir):
            if optode is None or 'global' in optode.ch_name: # Catch base case
                return

            # Calculate the mean and standard deviation of the HRF across estimates
            optode.trace = np.mean(optode.estimates, axis = 0)
            optode.trace_std = np.std(optode.estimates, axis = 0)

            if plot_dir:
                optode.plot(plot_dir)
            
            if optode.oxygenation:
                hbo_estimates.append(optode.trace)
            else:
                hbr_estimates.append(optode.trace)

            estimate(optode.left, plot_dir)
            estimate(optode.right, plot_dir)

        estimate(self.root, plot_dir) # Call to recursive estimate

        # Calculate global HRF mean and standard deviation
        for oxygenation, estimates in zip([True, False], [hbo_estimates, hbr_estimates]):
            global_mean = np.mean(estimates)
            global_std = np.std(estimates)

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
                
                print(f"Correlating {hbo_channel} with {hbr_channel}")
                
                corr_coefficient, p_value = scipy.stats.spearmanr(hbo_hrf, hbr_hrf)
                
                corr_matrix[hbo_ind, hbr_ind, 0] = corr_coefficient
                corr_matrix[hbo_ind, hbr_ind, 1] = p_value
        
        print(f"Correlation matrix: {corr_matrix[:, :, 0]}")

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

    def correlate_canonical(self, plot_filename = "canonical_correlation.png", duration = 12.0):
        """
        Correlate the HRF estimates with a canonical HRF to assess similarity
        """
        # Generate canonical HRF
        dt = 1.0 / self.sfreq
        time_stamps = np.arange(0, duration, dt)

        # Parameters for the double-gamma HRF
        peak1 = scipy.stats.gamma.pdf(time_stamps, 6) # peak at ~6s
        peak2 = scipy.stats.gamma.pdf(time_stamps, 16) / 6.0 # undershoot at ~16s

        canonical_hrf = peak1 - peak2
        canonical_hrf /= np.max(canonical_hrf)  # Normalize peak to 1

        corr_matrix = np.zeros((len(self.hbo_channels) + len(self.hbr_channels), 2))
        for ind, ch_name in enumerate(self.hbo_channels + self.hbr_channels):
            hrf = self.channels[ch_name]
            corr_coefficient, p_value = scipy.stats.spearmanr(hrf.trace, canonical_hrf)
            corr_matrix[ind, 0] = corr_coefficient
            corr_matrix[ind, 1] = p_value

        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, :, 0], cmap='viridis', aspect='auto')
        plt.colorbar(label='Correlation Coefficient')
        plt.title('Correlation Matrix of HRF Estimates with Cannonical HRF')
        plt.xlabel('Montage Channels')
        plt.ylabel('Cannonical HRF')
        plt.xticks(range(len(self.channels.keys())), range(len(self.channels.keys())), rotation=90)
        plt.yticks(range(len(self.channels.keys())), range(len(self.channels.keys())))
        plt.tight_layout()

        plt.savefig(plot_filename)
        plt.close()

        # Plot p-values
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix[:, :, 1], cmap='viridis', aspect='auto')
        plt.colorbar(label='P-value')
        plt.title('P-values of Correlation with Cannonical HRF')
        plt.xlabel('Montage Channels')
        plt.ylabel('Cannonical HRF')
        plt.xticks(range(len(self.channels.keys())), range(len(self.channels.keys())), rotation=90)
        plt.yticks(range(len(self.channels.keys())), range(len(self.channels.keys())))
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
        print(f"Final HRF: {hrfs}")
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return
    

    def load(self, json_filename):
        """ Load montage with the given json filename """
        # Read in json
        with open(json_filename, 'r') as file:
            json_contents = json.load(file)

        # Update montage with saved info
        for key, channel in json_contents.items():
            key_split = key.split('-')
            doi = key_split.pop()
            ch_name = '-'.join(key_split)

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
            elif oxygenation == False:
                self.channels[ch_name] = self.hbr_tree.insert(empty_hrf)
            
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
                # Grab pertinent info from nirx header
                ch_name = channel['ch_name']
                location = channel['loc'][:3]
                print(f"Channel {ch_name} location: {location}")

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
                results = self.search_dfs(empty_hrf, max_distance = 1e-9)
                if results: # If previously defined channel hrf found
                    print(f"Local HRF found in the channel {results.ch_name}, merging with optode {ch_name}")
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
    split = ch_name.split('hb')
    if split[1][0] == 'o': # If oxygenated channel
        return True
    elif split[1][0] == 'r': # If deoxygenated channel
        return False
    else:
        raise ValueError(f"Channel {ch_name} oxygenation status could not be determines, ensure each channel has appropriate naming scheme with HbO/HbR included")

