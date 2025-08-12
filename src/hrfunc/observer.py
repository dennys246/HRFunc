import mne, os, math
import numpy as np
from scipy.signal import welch
from mne_nirs.preprocessing import peak_power
from mne_nirs.visualisation import plot_timechannel_quality_metric
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis


class lens:
    # This object is primarily used to calculate summary statistics of the passed in and
    # preprocessed NIRX objects. This can be used to compare your output to published results
    def __init__(self, working_directory = None):
        
        # Set working directory is not passed in
        self.working_directory = working_directory or os.getcwd()

        self.metrics = {
            'Preprocessed': {
                'kurtosis': {},
                'skewness': {},
                'snr': {}
                },
            'Deconvolved': {
                'kurtosis': {},
                'skewness': {},
                'snr': {}                
                },
            'SCI': []  
        }


    def compare_subject(self, subject_id, raw_nirx, preproc_nirx, deconv_nirx, events, channel = 0, length = 500):
        print(f"Comparing subject {subject_id}")
        self.channels = preproc_nirx.ch_names

        # Plot the preprocessed nirx with the deconvolved with event overlays
        self.plot_nirx(subject_id, preproc_nirx, deconv_nirx, events, channel, length)

        # Grab raw NIRX quality metrics scalp coupling index and peakpower
        #self.metrics['SCI'] = np.concatenate((self.metrics['SCI'], self.calc_sci(subject_id, raw_nirx, 'raw')), axis = 0)
        #self.calc_pp(subject_id, raw_nirx, 'raw')

        meters = [self.calc_skewness_and_kurtosis, self.calc_snr]
        for meter in meters: # Iterate through other metrcis comparing preprocessed and deconvolved data
            response = meter(subject_id, preproc_nirx, 'Preprocessed')
            response = meter(subject_id, deconv_nirx, 'Deconvolved')


    def compare_subjects(self):
        channel_kurtosis = {state: {channel: 0 for channel in self.channels} for state in ['Preprocessed', 'Deconvolved']}
        channel_skewness = {state: {channel: 0 for channel in self.channels} for state in ['Preprocessed', 'Deconvolved']}
        
        kurtosis = {
            'Preprocessed': [],
            'Deconvolved': []
        }
        skewness = {
            'Preprocessed': [],
            'Deconvolved': []
        }

        count = 0
        _min = -0.02
        _max = 0.07
        
        for state in ['Preprocessed', 'Deconvolved']:
            count = 0
            #Add all kurtosis across subjects per channel
            for subject_id, channels in self.metrics[state]['kurtosis'].items():
                for channel in channels:
                    channel_kurtosis[state][channel] += self.metrics[state]['kurtosis'][subject_id][channel]
                    count += 1

            # Add all skewness across subjects per channel
            for subject_id, channels in self.metrics[state]['skewness'].items():
                for channel in channels:
                    channel_skewness[state][channel] += self.metrics[state]['skewness'][subject_id][channel]
            
            # Average across subjects for each channel
            for channel in channels:
                skewness[state].append(channel_skewness[state][channel] / count)
                kurtosis[state].append(channel_kurtosis[state][channel] / count) 
        
        print(f"Average {state} skew: {sum(skewness[state]) / len(skewness[state])} \nAverage {state} kurtosis: {sum(kurtosis[state]) / len(kurtosis[state])}")
        
        preprocess_snr = [self.metrics['Preprocessed']['snr'][subject] for subject in self.metrics['Preprocessed']['snr'].keys()]
        deconvolved_snr = [self.metrics['Deconvolved']['snr'][subject] for subject in self.metrics['Deconvolved']['snr'].keys()]
        print(f"Average preprocessed SNR: {sum(preprocess_snr) / len(preprocess_snr)}\nAverage deconvolved SNR: {sum(deconvolved_snr) / len(deconvolved_snr)}")
        for metric, metric_name in zip([kurtosis, skewness], ['Kurtosis', 'Skewness']):    
            plt.figure(figsize=(10, 8))

            # Set the number of bars
            bar_width = 0.2
            x = np.arange(len(channels))  # The x locations for the groups

            # Create the bar plot
            plt.bar(x - bar_width / 2, metric['Preprocessed'], width = bar_width, label = f'Convolved Hemoglobin', color='r', align='center')
            plt.bar(x + bar_width / 2, metric['Deconvolved'], width = bar_width, label = f'Deconvolved Neural Activity', color='b', align='center')

            plt.ylim(_min, _max)

            # Adding labels and title
            plt.xlabel('Positions')
            plt.ylabel(metric_name)
            plt.title(f'Effects of Deconvolution on {metric_name}')
            plt.xticks(x, channels, rotation='vertical')  # Set the position names as x-tick labels
            plt.legend()

            # Show the plot
            plt.savefig(f'{self.working_directory}/plots/channel_wise_{metric_name.lower()}.jpeg')
            plt.close()

        #print(f"SCI: {self.metrics['SCI'].shape}")

        #plt.hist(self.metrics['SCI'])
        #plt.title(f'Scalp Coupling Index')
        #plt.savefig(f'{self.working_directory}/plots/subject_wise_sci.jpeg')
        #plt.close()

    def plot_nirx(self, subject_id, preproc_scan, deconv_scan, events, channel = 1, length = 500):

        #Load both scans
        preproc_scan.load_data()
        preproc_data = preproc_scan.get_data([channel])

        deconv_scan.load_data()
        deconv_data = deconv_scan.get_data([channel])

        if os.path.exists(f"{self.working_directory}/plots/channel_data/") == False:
            os.mkdir(f"{self.working_directory}/plots/channel_data/")

        # Prepare preprocessed Signal
        # Normalize source to 0–1
        source_norm = (preproc_data[0, :length] - preproc_data[0, :length].min()) / (preproc_data[0, :length].max() - preproc_data[0, :length].min())

        # Rescale to match target range
        target_min = deconv_data[0, :length].min()
        target_max = deconv_data[0, :length].max()
        preproc_scaled = source_norm * (target_max - target_min) + target_min

        # Plot the preprocessed and deconvolved data
        plt.figure(figsize=(14, 8)) 
        plt.plot(preproc_scaled[:length], color='red', linestyle='--', label='Convolved Hemoglobin')
        plt.plot(deconv_data[0, :length], color='blue', label='Deconvolved Neural Activity')

        plt.xlabel('fNIRS Samples')
        plt.title(f'fNIRS Channel Data')

        plt.legend(loc='best')
        
        # Add in events
        for event_ind, event in enumerate(events):
            # If outside of range we're looking for
            if event_ind > length: break
            
            if event: # If event present
                plt.axvline(x = event_ind, color = 'orange', label = 'Trial')
        
        plot_filename = f'{self.working_directory}/plots/channel_data/{subject_id}_channel_data.png'
        print(f"Saving deconv plot too {plot_filename}")
        plt.savefig(f'{self.working_directory}/plots/channel_data/{subject_id}_channel_data.png')
        plt.close()

    def calc_pp(self, subject_id, scan, state):
        print(f"Calculating peakpower for {state} data...")
        preproc_nirx = scan.load_data()

        preproc_od = mne.preprocessing.nirs.optical_density(preproc_nirx)
        preproc_od, scores, times = peak_power(preproc_od, time_window=10)

        figure = plot_timechannel_quality_metric(preproc_od, scores, times, threshold=0.1)
        plt.savefig(f'{self.working_directory}/plots/{state}_powerpeak.jpeg')
        plt.close()
        return True

    def calc_sci(self, subject_id, data, state):
        # Load the nirx object

        od = mne.preprocessing.nirs.optical_density(data)
        sci = mne.preprocessing.nirs.scalp_coupling_index(od)

        figure, axis = plt.subplots(1, 1)

        axis.hist(sci)
        axis.set_title(f'{subject_id} {state} Scalp Coupling Index')
        plt.savefig(f'{self.working_directory}/plots/{state}_sci.jpeg')
        plt.close()
        return sci

    def calc_snr(self, subject_id, nirx, state):
        # Load data
        nirx.load_data()
        data = nirx.get_data()

        # Filter the raw data to obtain the signal and noise components
        # Define the signal band (i.e., hemodynamic response function band)
        signal_band = (0.01, 0.2)
        # Define the noise band (outside of the hemodynamic response)
        noise_band = (0.2, 1.0) 

        # Extract the signal in the desired band
        preproc_signal = nirx.copy().filter(signal_band[0], signal_band[1], fir_design='firwin')

        # Extract the noise in the out-of-band frequency range
        preproc_noise = nirx.copy().filter(noise_band[0], noise_band[1], fir_design='firwin')

        # Calculate the Power Spectral Density (PSD) for signal and noise using compute_psd()
        psd_signal = preproc_signal.compute_psd(fmin = signal_band[0], fmax = signal_band[1])
        psd_noise = preproc_noise.compute_psd(fmin = noise_band[0], fmax = noise_band[1])

        # Extract the power for each component
        signal_power = psd_signal.get_data().mean(axis = -1)  # Average power across frequencies for signal
        noise_power = psd_noise.get_data().mean(axis = -1)    # Average power across frequencies for noise

        # Calculate SNR
        snr = signal_power / noise_power
        snr = sum(snr)/len(snr)
        print(f"{state} signal-to-noise ratio - {snr}")
        self.metrics[state]['snr'][subject_id] = snr
        return snr

    def calc_skewness_and_kurtosis(self, subject_id, nirx, state):
        # Load data
        nirx.load_data()
        data = nirx.get_data()

        # Compute skewness and kurtosis for each channel
        print(f"{data.shape}")
        skewness = skew(data, axis = 0)  # Calculate skewness along the time dimension
        kurtosis_vals = kurtosis(data, axis = 0)  # Calculate kurtosis along the time dimension

        # Display the results for each channel
        channel_skewness = {}
        channel_kurtosis = {}
        for ch_name, skew_val, kurt_val in zip(self.channels, skewness, kurtosis_vals):
            channel_skewness[ch_name] = skew_val
            channel_kurtosis[ch_name] = kurt_val
            print(f"{state} - Channel {ch_name}: Skewness = {skew_val:.3f}, Kurtosis = {kurt_val:.3f}")

            if subject_id not in self.metrics[state]['skewness'].keys():
                self.metrics[state]['skewness'][subject_id] = {}
                self.metrics[state]['kurtosis'][subject_id] = {}
            self.metrics[state]['skewness'][subject_id][ch_name] = skew_val
            self.metrics[state]['kurtosis'][subject_id][ch_name] = kurt_val

    def calc_heart_rate_presence(self, subject_id, nirx, state):
        # Assuming `raw_haemo` is your preprocessed fNIRS object with hemoglobin concentration data
        nirx.load_data()
        data = nirx.get_data()

        # Step 1: Define heart rate frequency range
        heart_rate_low = 0.8  # Lower bound in Hz
        heart_rate_high = 2.0  # Upper bound in Hz

        # Step 2: Calculate Power Spectral Density (PSD) for each channel
        sfreq = self.sfreq  # Sampling frequency
        n_per_seg = int(4 * sfreq)  # Length of each segment for Welch's method

        psd_list = []
        freqs, psd_all_channels = [], []

        for i, channel_data in enumerate(data):
            freqs, psd = welch(channel_data, sfreq, nperseg=n_per_seg)
            psd_all_channels.append(psd)

        # Step 3: Plot PSD for each channel with heart rate range highlighted
        plt.figure(figsize=(12, 8))

        for i, psd in enumerate(psd_all_channels):
            plt.plot(freqs, psd, label=f'Channel {i+1}')
        
        # Highlight the heart rate frequency range
        plt.axvspan(heart_rate_low, heart_rate_high, color='red', alpha=0.2, label='Heart Rate Range (0.8-2.0 Hz)')

        # Customize plot
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (PSD)')
        plt.title(f'{state} Power Spectral Density for {subject_id}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
        plt.xlim(0, 3)  # Limit to frequencies of interest
        plt.yscale('log')  # Log scale for better visualization of peaks
        plt.savefig(f'{self.working_directory}/plots/{state}_hr_presence.jpeg')
        plt.close()
    
