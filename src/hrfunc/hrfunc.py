import scipy.linalg, hrtree, json, mne
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def estimate_hrfs(nirx_folder, nirx_identifier, events, hrfs_filename = "hrf_estimates.json", **kwargs):
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

    _montage.generate_distribution(context['duration']) # Generate HRF distribution

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

class montage:

    """
    Class functions:
        - localize_hrf() - Tries to find a previously derived HRFs localized to the same region
        - convolve_hrf() - Convolves an impulse function (series of 0's and 1's) with an HRF
        - deconvolve_hrf() - Deconvolves a fNIRS signal and impulse function to derive the underlying HRF
        - generate_distribution() - Calculates an average HRF and it's standard deviation across time
        - save() - Saves the current montage HRFs
        - load() - Loads a montage of HRFs
    
    Class attributes:
        - nirx_obj (mne raw object) - NIRX object loaded in via MNE python library
        - sfreq (float) - Sampling frequency of the fNIRS object
        - channels (list) - fNIRS montage channel names
        - subject_estimates (list) - List of subject event-wise HRF estimate
        - channel_estimates (list) - List of channel HRF distribution estimates (position 0 is mean and 1 is std)
    """

    def __init__(self, nirx_obj, hrfs_filename = None, **kwargs):
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

        # Initialize an empty tree
        self.tree = tree(**kwargs)

        self.channels = {} # Create variable for holding poiners to each channel
        self.subject_estimates = {} # Create hold variable for storing intermediary data

        self.sfreq = nirx_obj.info['sfreq']
        self.hbo_channels = [ch for ch in nirx_obj.ch_names if ch.endswith('hbo')]
        self.hbr_channels = [ch for ch in nirx_obj.ch_names if ch.endswith('hbr')]

        if hrfs_filename: # If hrfs json filename provided, load in
            self.load(hrfs_filename)
        else:
            # Add empty HRF nodes to the tree for each HRF
            for channel in nirx_obj.info['chs']:
                # Grab pertinent info from nirx header
                ch_name = channel['ch_name']
                location = channel['loc'][:3]

                # create an empty HRF object
                empty_hrf = hrtree.HRF(
                    self.context['doi'],
                    ch_name, 
                    self.context['duration'], 
                    self.sfreq, 
                    [], 
                    [], 
                    location
                )

                # Add in subject level estimate variables 
                self.subject_estimates[ch_name] = {'events': [], 'estimates': []}

                # Insert empty hrf into tree and attach pointer to channel
                self.channels[ch_name] = self.tree.insert(empty_hrf)
        self.__repr__()

    def __repr__(self):
        return f" - Montage object - \nNumber of channels: {len(self.channels)}\n Sampling frequency: {self.sfreq}\nHbO channels (count of {len(self.hbo_channels)}): {self.hbo_channels}\n HbR channels (count of {len(self.hbr_channels)}): {self.hbr_channels}\n Subject HRFs estimated: {len(self.subject_estimates)}\n\n"

    def localize_hrfs(self, max_distance = 3.0):
        """
        Tries to find local HRFs to each of the fNIRS optodes using the tree structure
        functionality to quickly find nearby HRF's. If it can't it will default to a
        global HRF estimated.

        Arguments:
            max_distance (float) - maximum distance in milimeter's a previously estimated HRF can be attached to an optode
        """
        
        for ch_name in self.channels.keys(): # Iterate through channels apart of nirx data
            
            hrf = self.tree.search_dfs(self.channels[ch_name], max_distance) # Search in space for similar HRF

            if hrf: # If found
                self.channels[ch_name].trace = hrf.trace # Add mean and std to montage for channel
                self.channels[ch_name].trace_std = hrf.trace_std

            else: # If hrf not found locally
                print(f"Local HRF with given context couldn't be found for channel {ch_name}, searching for global HRF")
                # Adjust channel location temporarily to global node nexus -0.5
                ch_location = [self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z] 
                self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z = -0.5, -0.5, -0.5
                
                # Search for global HRF with similar context
                hrf = self.tree.search_bfs(self.channels[ch_name], max_distance)
                if hrf: # If found
                    self.channels[ch_name].trace = hrf.trace # Add mean and std to montage for channel
                    self.channels[ch_name].trace_std = hrf.trace_std
                else: # If global HRF not found
                    LookupError(f"Global HRF with given context could not be found for {ch_name}")
                
                # Replace global location with original optode location
                self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z = ch_location[0], ch_location[1], ch_location[2]

    def convolve_hrf(self, events, hrf):
        """ Convolve an event impulse series with an hrf to create an expected signal. Both
        inputs must be the same length.

        Argument:
            events (list) - List of 0's and 1's indicating event occurances
            hrf (list) - HRF estimated trace
        """
        return np.convolve(events, hrf, mode='full')

    def deconvolve_hrf(self, nirx_obj, events, duration = 12.0, _lambda = 1e-3, shorten_events = False):
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

        events = np.array(events)

        nirx_obj.load_data()
        data = nirx_obj.get_data()

        hrf_len = int(round(nirx_obj.info['sfreq'] * duration, 0)) # Calculate HRF length
        scan_len = data.shape[1] # Grab single channel signal length

        if events.shape[0] > scan_len and shorten_events:
            events = events[:scan_len]
        elif events.shape[0] != scan_len:
            raise ValueError(f"Expected events to be of length {hrf_len} but got length {events.shape[0]}...\n\nAdjust your events to match the size of the NIRS scan or if you need to cut the events short to match the NIRS scan, pass the argument shorten_events as True...")

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
                hrf_estimate, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            except np.linalg.LinAlgError: # If that fails, try applying the same with smoothing
                hrf_estimate = scipy.linalg.pinv(lhs) @ rhs

            # Denormalize HRF estimate
            #hrf_estimate = hrf_estimate * np.max(np.abs(fnirs_signal))
            #hrf_estimate = hrf_estimate * std + mean

            self.subject_estimates[channel['ch_name']]['estimates'].append(hrf_estimate)
            self.subject_estimates[channel['ch_name']]['events'].append(events)
        
        self.generate_distribution(duration)


    def deconvolve_nirs(self, nirx_obj, _lambda = 1e-3, **kwargs):
        """
        Deconvlve a fNIRS scan using estimated HRF's localized to optodes location

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
                print("LinAlgError in lstsq:", e)
                deconvolved_signal = scipy.linalg.pinv(lhs) @ rhs

            print("Deconvolved signal preview:", deconvolved_signal[:10])
            print("Pre-rescale min/max:", np.min(deconvolved_signal), np.max(deconvolved_signal))

            # Denormalize neural signal estimate
            #deconvolved_signal = deconvolved_signal * std + mean

            return deconvolved_signal # Return recovered neural signal

        # Apply deconvolution and return the nirx object
        for ch_name, hrf in self.channels.items():
            
            if ch_name == 'global': continue # Skip if global hrf estimate
            
            print(f"Deconvolving channel {ch_name}...")
            nirx_obj.apply_function(deconvolution, picks = [ch_name]) # Apply deconvolution for channel
        
        return nirx_obj

    def resample_nirs(self, nirx_obj, events, std_seed, _lambda = 1e-3, hrfs_filename = "hrfs.json", verbose = True, **kwargs):
        """
        Resample fNIRs data for deep learning using the HRF confidence interval
        to establish a new HRF to deconvolve the NIRS data with.

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan loaded through mne
            events (list) - event impulse sequence of 0's and 1's
            std_seed (float) - deviation to impart on HRF
        """
        # Initialize a montage for the
        _montage = montage(nirx_obj, hrfs_filename, **kwargs)
            
        nirx_obj.load_data()
        if verbose: # Check shape if verbose
            data = nirx_obj.get_data()
            print(f"Original fNIRS Length: {data.shape}\nMax Value: {np.max(data)}\nMin Value: {np.min(data)}\nAny Nan: {np.isnan(data).any()}\nAny inf: {np.isinf(data).any()}")

        print(f"Original HRF length:{hrf.shape}\nMax Value{max(hrf)}\nMin Value: {min(hrf)}\nAny Nan: {np.isnan(hrf).any()}\nAny inf: {np.isinf(hrf).any()}")

        # Define hrf deconvolve function to pass nirx object
        def deconvolution(nirx):
            original_len = len(nirx)
        
            # Normalize input
            nirx = nirx / np.max(np.abs(nirx))

            # Pad HRF to match nirx length
            hrf.trace = hrf.trace / np.max(np.abs(hrf.trace))
            hrf_padded = np.pad(hrf.trace, (0, original_len - len(hrf.trace)), 'constant')

            # Construct Toeplitz matrix from HRF
            A = scipy.linalg.toeplitz(hrf_padded)

            # Solve the inverse problem with regularization
            lhs = A.T @ A + _lambda * np.eye(A.shape[1])
            rhs = A.T @ nirx
            try: # Try using standard linear least squared to solve
                deconvolved_signal, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            except np.linalg.LinAlgError: # If failed try to run pinv with smoothing
                deconvolved_signal = scipy.linalg.pinv(lhs) @ rhs
            return deconvolved_signal

        # Apply deconvolution and return the nirx object
        for ch_name, hrf in _montage.channels.items():
            if ch_name == 'global': # If a global HRF
                continue # skip

            # Resample HRF with passed in standard deviation seed
            resampled_hrf = hrf.resample(std_seed)

            # Convolve events with resampled HRF to assess expected activity
            expected_activity = _montage.convolve_hrf(events, resampled_hrf.trace)
            
            # Apply deconvolution to channel
            nirx_obj.apply_function(deconvolution, picks = [ch_name])
        
        return nirx_obj

    def generate_distribution(self, duration = 12.0):
        """
        Calculate average and standard deviation of HRF across subjects for each channel

        Arguments:
            duration (float) - Duration in seconds of the HRF to estimate
        """

        # Check if HRF subject wise estimates have been calculated
        hrf_means = []
        hrf_stds = []

        # Generate channel wise HRF estimates
        length = int(self.sfreq * duration)
        for ch_name, estimates in self.subject_estimates.items():
            event_estimates = []
            for sub_ind, events in enumerate(estimates['events']):
                for sample_ind in events:
                    estimate = estimates['estimates'][sub_ind]
                    if sum(estimate) == 0: # If all zeros, skip
                        continue
                    event_estimates.append(estimate)
            # Estimate hrf mean accross events
            hrf_mean = np.mean(event_estimates, axis = 0)
            hrf_std = np.std(event_estimates, axis = 0)
            
            # Add subject hrf estimates to channel
            self.channels[ch_name].trace = hrf_mean
            self.channels[ch_name].trace_std = hrf_std

            # Append mean and std of hrf estimate
            hrf_means.append(hrf_mean)
            hrf_stds.append(hrf_std)

        # Calculate global HRF mean and standard deviation
        global_mean = np.mean(hrf_means, axis = 0)
        global_std = np.mean(hrf_stds, axis = 0)

        # Create a global HRF variable
        global_hrf = hrtree.HRF(
            doi = self.channels[ch_name].context['doi'],
            ch_name = "global",
            duration = self.channels[ch_name].context['duration'],
            sfreq = self.sfreq,
            trace = global_mean,
            trace_std = global_std,
            location = [-0.5, -0.5, -0.5]
        )

        #Insert global hrf into tree and attach pointer to channels dict
        self.channels['global'] = self.tree.insert(global_hrf)
        
        # Plot all of the channel HRF estimates
        for channel, hrf in self.channels.items():
            hrf_mean = hrf.trace
            hrf_std = hrf.trace_std
            time = np.arange(len(hrf_mean))

            plt.figure(figsize=(8, 4))
            plt.plot(time, hrf_mean, label='Mean HRF', color='blue')
            plt.fill_between(time, hrf_mean - hrf_std, hrf_mean + hrf_std, color='blue', alpha=0.3, label='±1 SD')

            plt.xlabel('Samples')
            plt.ylabel('HRF amplitude')
            plt.title(f'Estimated HRF for {channel} with Standard Deviation')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/plots/{channel}_hrf_estimate.png")

        # Estimate global HRF?

    def correlate_hrf(self, plot_filename = "montage_correlation.png"):
        """
        Correlate the HRF estimates across the subject pool to assess similarity
        """
        corr_matrix = np.zeros((len(self.hbo_channels), len(self.hbr_channels), 2))
        
        # Calculate correlation coefficients and p-values between HbO and HbR channels
        for hbo_ind, hbo_channel in enumerate(self.hbo_channels):
            hbo_hrf = self.channels[hbo_channel].trace
            for hbr_ind, hbr_channel in enumerate(self.hbr_channels):
                print(f"Correlating {hbo_channel} with {hbr_channel}")
                hbr_hrf = self.channels[hbr_channel].trace
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
        canonical_hrf /= np.max(hrf)  # Normalize peak to 1

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


    def save(self, json_filename, **kwargs):
        """ Save the HRF as a json using the json_filename"""

        self.context = {**self.context, **kwargs}
        doi = self.context['doi']

        # Format json
        json_contents = {}
        for ch_name in self.channels.keys():
            channel_identifier = f"{'-'.join(ch_name.split(' '))}-{doi}"
            json_contents[channel_identifier] = {
                'hrf_mean': self.channels[ch_name].trace.tolist(),
                'hrf_std': self.channels[ch_name].trace_std.tolist(),
                'location': [self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z],
                'sfreq': self.sfreq,
                'context': self.context
        }

        # Save to a JSON file
        with open(json_filename, "w") as file:
            json.dump(json_contents, file, indent=4)  # indent is optional, just makes it pretty
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
            ch_name = ' '.join(key_split)

            # create an empty HRF object
            empty_hrf = hrtree.HRF(
                doi,
                ch_name, 
                channel['context']['duration'], 
                self.sfreq, 
                np.asarray(channel['hrf_mean'], dtype=np.float64), 
                np.asarray(channel['hrf_std'], dtype=np.float64), 
                channel['location']
            )
            empty_hrf

            # Insert empty hrf into tree and attach pointer to channel
            self.channels[ch_name] = self.tree.insert(empty_hrf)
        return
    
def _load_fnirs(nirs_filename):
    """ Load the fNIRS file based on the format found """
    if nirs_filename[-6:] == ".snirf": # If snirf format
        try:
            nirs_obj = mne.io.read_raw_snirf(nirs_filename)
        except:
            return ValueError("SNIRF file passed in failed to load")
    if nirs_filename[-1:] == "/": # If folder passed in
        try:
            nirs_obj = mne.io.read_raw_nirx(nirs_filename)
        except:
            return ValueError("NIRS folder failed to load")
    return nirs_obj
    
