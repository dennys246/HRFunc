import json, random, math, hrhash
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import deque
from nilearn.glm.first_level import spm_hrf

class Tree:
    """
    This object is intended to generate a synthetic hemodynamic response function to be
    convovled with a NIRS object. You can pass in a variety of optional parameters like mean window,
    sigma and scaling factor to alter the way your hrf is generated.

    Class attributes:

    Class functions:
    
    """
    def __init__(self, hrf_filename = "hrfs.json", **kwargs):
        self.root = None

        self.hrf_filename = hrf_filename

        # Set and update context
        self.context = {
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
        self.context = {**self.context, **kwargs} 
        self.context_weights = {key: 1.0 for key in self.context.keys()}

        self.hashtable = hrhash.HashTable(self.context)

    def build(self, hrf_filename = None, sim_threshold = 0.0, context_weights = None):
        """
        Orchestrate building the HRF tree while filtering for specific context

        Arguments:
            hrf_filename (str) - Filename of the HRF json to load into the tree
            sim_threshold (float) - Threshold to allow or exclude HRF's based on context, defaults to 0.0 or no threshold
            context_weights (dict) - Weight to attach to each context during similarity comparison
        """
        if hrf_filename == None: # If no json filename provided
            hrf_filename = self.hrf_filename # Set as class default

        hrfs_json = json.load(hrf_filename) # Load HRFs from json
        
        for channel in hrfs_json:

            # If requesting a similarity comparison
            if sim_threshold > 0.0: 
                # Check if the hrf matches the context
                similarity = self.compare_context(self.context, channel['context'], context_weights)
                if similarity < sim_threshold: # If not similar enough to requested context
                    continue # Exclude derived HRF
            
            # Grab channel and doi info
            split = channel.split('-')
            doi = split.pop()
            ch_name = ' '.join(split)

            # create a new hrf node
            new_hrf = HRF(ch_name, doi, channel['duration'], channel['sfreq'], channel['hrf_mean'], channel['hrf_std'], channel['location'], **self.context)
            
            # Insert hrf node into tree
            node = self.insert(new_hrf)

            # Add newly added node into HRHash table?

    def branch(self, **kwargs):
        """
        Accepts context keyword inputs via kwargs, updates the trees context
        and then builds a new tree filtering for just the context

        Arguments:
            **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)
        """
        self.context = {**self.context, **kwargs} # Update context

        branch = Tree()

        for key, values in self.context.items(): # Iterate through all context items
            for value in values: # Iterate through each item in a context area
                # Hash on the value and iterate through the tree pointers
                context_references = self.hashtable.search(value)
                for node in context_references:
                    branch.insert(node) # Insert node pointer into branch
        return branch

    def insert(self, hrf, depth = 0, node = None):
        """Insert a new node into the 3D k-d tree based on spatial position."""

        if self.root is None:
            self.root = hrf
            return self.root

        if node is None:
            node = self.root

        axis = depth % 3  # Cycle through x, y, z

        h_val = (hrf.x, hrf.y, hrf.z)[axis]
        n_val = (node.x, node.y, node.z)[axis]

        if h_val < n_val:
            if node.left is None:
                node.left = hrf
                return node.left
            else:
                return self.insert(hrf, depth + 1, node.left)
        else:
            if node.right is None:
                node.right = hrf
                return node.right
            else:
                return self.insert(hrf, depth + 1, node.right)

    def compare_context(self, first_context, second_context, context_weights):
        """
        Compare two contexts to see how similar they are
        """
        context_similarity = []
        for key, values in first_context.items():
            # If context not mentioned in first context
            if values == None: # Exclude context in similarity comparison
                continue 

            same = 0 # Create a context specific similarity value
            for value in values:
                if value in second_context[key]:
                    if context_weights: # If a context weight provided
                        same += 1 * context_weights[key] # Weight similarity score
                    else: # add 
                        same += 1

            # Calculate context-specific similarity and append
            context_similarity.append(same/len(first_context)) 
        
        return sum(context_similarity) / len(context_similarity) # Average similarity and return
    
    def search_dfs(self, optode, max_distance = 0.5, depth=0, node = None, max_point = None, min_point = None):
        """
        Searches the tree to find a HRF node within the max distance

        Arguments:
            optode (list of floats) - The x, y, z coordinates of the optode
            max_distance (float) - The maximum distance to search for a HRF node
            depth (int) - The current depth of the search
            node (HRF) - The current node in the search
            max_point (list of floats) - The maximum x, y, z coordinates of the search
            min_point (list of floats) - The minimum x, y, z coordinates of the search
        Returns:
            node (HRF) - The HRF node within the max distance
        """
        if node is None:
            node = self.root

        if node is None: # Establish base case
            return None
        
        # Find max/min x, y and z if not calculated
        if max_point == None:
            min_point = [optode.x - max_distance, optode.y - max_distance, optode.z - max_distance]
            max_point = [optode.x + max_distance, optode.y + max_distance, optode.z + max_distance]
            

        if min_point[0] > node.x and min_point[1] > node.y and min_point[2] > node.z:
            # Check if right node
            if max_point[0] < node.x and max_point[1] < node.y and max_point[2] < node.z:
                # Check if the next node is not closer
                current_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.x, node.y, node.z])))
                if current_distance <= max_distance:
                    left_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.left.x, node.left.y, node.left.z])))
                    right_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.right.x, node.right.y, node.right.z])))
                    
                    # If the next neighbors are farther away than optode
                    if current_distance <= right_distance and current_distance <= left_distance: 
                        return node
                    elif current_distance >= right_distance: # Move to the right node
                        return self.search_dfs(optode, max_distance, depth + 1, node.right, max_point, min_point)
                    else: # Move to the left node
                        return self.search_dfs(optode, max_distance, depth + 1, node.left, max_point, min_point)

        axis = depth % 3
        if (axis == 0 and min_point[0] < node.x) or (axis == 1 and min_point[1] < node.y) or (axis == 2 and min_point[2] < node.z):
            return self.search_dfs(optode, max_distance, depth + 1, node.left, max_point, min_point)
        else:
            return self.search_dfs(optode, max_distance, depth + 1, node.right, max_point, min_point)

    def search_bfs(self, optode, max_distance):
        """
        Searches the tree to find a HRF node within the max distance using BFS
        
        Arguments:
            optode (list of floats) - The x, y, z coordinates of the optode
            max_distance (float) - The maximum distance to search for a HRF node
        Returns:
            node (HRF) - The HRF node within the max distance
        """
        if self.root is None:
            return None

        # Find max/min x, y and z if not calculated

        min_point = [optode[0] - max_distance, optode[1] - max_distance, optode[2] - max_distance]
        max_point = [optode[0] + max_distance, optode[1] + max_distance, optode[2] + max_distance]

        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            # Check if node in range
            if min_point[0] > node.x and min_point[1] > node.y and min_point[2] > node.z:
                if max_point[0] < node.x and max_point[1] < node.y and max_point[2] < node.z:
                    # Check if current node distance to optode is within max distance
                    current_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.x, node.y, node.z])))
                    if current_distance <= max_distance:
                        # Check if the next node is not closer
                        left_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.left.x, node.left.y, node.left.z])))
                        right_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.right.x, node.right.y, node.right.z])))
                        # If the next neighbors are farther away than optode
                        if current_distance <= right_distance and current_distance <= left_distance: 
                            return node
                        elif current_distance >= right_distance: # Move to the right node
                            queue.append(node.right)
                        else: # Move to the left node
                            queue.append(node.left)
                        continue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return None
      
    def save(self, hrfs_filename = 'tree_hrfs.json'):
        hrfs = self.gather(self.root)
        # Save to a JSON file
        with open(hrfs_filename, "w") as file:
            json.dump(hrfs, file, indent=4)  # indent is optional, just makes it pretty
        return

    def gather(self, node):

        hrfs = {}
        if node.left:
            hrfs |= self.gather(node.left)
        if node.right:
            hrfs |= self.gather(node.right)
        hrfs |= {
            f"{'-'.join(node.ch_name.split(' '))}-{node.doi}": {
                "hrf_mean": node.trace,
                "hrf_std": node.trace_std,
                "location": [
                    node.x,
                    node.y,
                    node.z
                ],
                "sfreq": node.sfreq,
                "context": node.context
            }
        }
        return hrfs


    def delete(self, hrf):
        """
        Delete a node from the 3D k-d tree based on spatial position.
        
        Arguments:
            hrf (HRF) - The HRF node to delete
        """
        self.root = self._delete_recursive(self.root, hrf, 0)

    def _delete_recursive(self, node, hrf, depth):
        if node is None:
            return None

        axis = depth % 3

        if node.x == hrf.x and node.y == hrf.y and node.z == hrf.z:
            if node.right:
                min_node = self._find_min(node.right, axis, depth + 1)
                node.x, node.y, node.z, node.hrf_data = min_node.x, min_node.y, min_node.z, min_node.hrf_data
                node.right = self._delete_recursive(node.right, min_node.x, min_node.y, min_node.z, depth + 1)
            elif node.left:
                min_node = self._find_min(node.left, axis, depth + 1)
                node.x, node.y, node.z, node.hrf_data = min_node.x, min_node.y, min_node.z, min_node.hrf_data
                node.right = self._delete_recursive(node.left, min_node.x, min_node.y, min_node.z, depth + 1)
                node.left = None
            else:
                return None  # No children case

        elif (axis == 0 and hrf.x < node.x) or (axis == 1 and hrf.y < node.y) or (axis == 2 and hrf.z < node.z):
            node.left = self._delete_recursive(node.left, hrf, depth + 1)
        else:
            node.right = self._delete_recursive(node.right, hrf, depth + 1)

        return node

    def _find_min(self, node, axis, depth):
        if node is None:
            return None

        if depth % 3 == axis:
            if node.left is None:
                return node
            return self._find_min(node.left, axis, depth + 1)

        left_min = self._find_min(node.left, axis, depth + 1)
        right_min = self._find_min(node.right, axis, depth + 1)

        return min([node, left_min, right_min], key=lambda n: getattr(n, ["x", "y", "z"][axis]) if n else float('inf'))

class HRF:
    def __init__(self, doi, ch_name, duration, sfreq, trace, trace_std = None, location = None, **kwargs):
        """
        Object for storing all information apart of an estimated HRF from an fNIRS optode

        Class functions:
            self.build() - Build the HRF to fit a new sampling frequency and run through processing requested
            self.spline_interp() - Resizes the HRF to new sampling frequency using spline interpolation
            self.smooth() - Smooths the HRF trace using a gaussian filter
            self.resample() - Resampled the HRF using the estimated HRF and it's standard deviation 
            self.plot() - Plots the current HRF trace attached to the class

        Class attributes:
            trace (list of floats) - A trace of the HRF
            trace_std (list of floats) - The standard deviation of the HRF over time
            duration (float) - Duration of the HRF in seconds
            sfreq (float) - Sampling frequency of the fNIRS device that the HRF estimate was recorded from
            location (list of floats) - Location of the optode the HRF was estimated from the fNIRS device
            plot (bool) - Request for whether to plot the HRF throughout it's preprocessing
            **kwargs - Context attributes to be updated, only used by class or developers

        """
        # Add doi and channel names
        self.doi = doi
        self.ch_name = ch_name

        # Attach passed into info to class 
        self.sfreq = sfreq
        self.length = int(round(self.sfreq * duration, 0))

        # Set the HRF mean and standard deviation of the trace
        self.trace = np.asarray(trace, dtype=np.float64)
        self.trace_std = np.asarray(trace_std, dtype=np.float64)

        if isinstance(location, list): # Grab location
            self.x = location[0]
            self.y = location[1]
            self.z = location[2]
        else:
            # If no location pass in, set to a random number between 0 and 1 to prevent a long tail
            self.x = -1 + random.random() 
            self.y = -1 + random.random()
            self.z = -1 + random.random()

        # Set HRF default context
        self.context = {
            'type': 'global',
            'doi': doi,
            'study': None,
            'task': None,
            'conditions': None,
            'stimulus': None,
            'intensity': None,
            'duration': duration,
            'protocol': None,
            'age_range': None,
            'demographics': None
        }
        unexpected = set(kwargs) - set(self.context)
        if unexpected:
            raise ValueError(f"Unexpected contexts cannot be added: {unexpected}\n\nMake sure the contexts your searching for are within the available contexts: {','.join(self.context.keys())}")
        self.context.update({key: value for key, value in kwargs.items() if key in self.context})

        self.left = None
        self.right = None

        self.hrf_processes = [self.spline_interp]
        self.process_names = ['spline_interpolate']
        self.process_options = []

    def __repr__(self):
        """String representation of the HRF object."""
        print(f"HRF: {self.doi} - {self.ch_name} \nSampling frequency: {self.sfreq}\nTrace length: {len(self.trace)}\nTrace standrad deviation: {len(self.trace_std)}")

    def build(self, new_sfreq, plot = False, show = False):
        # Define the processes for generating an hrf
        self.target_length = new_sfreq * float(self.context['duration'])
        for process, process_name, process_option in zip(self.hrf_processes, self.process_names, self.process_options):
            
            if process_option == None:
                self.trace = process(self.trace)
            else:
                self.trace = process(self.trace, process_option)
            
            if plot: # Plot the processing step results
                title = f"HRF - {process_name}"
                filename = f"plots/{'-'.join(process_name.split(' ')).lower()}_{self.type}_hrf_results.png"
                self.plot(title, filename, show)


    def spline_interp(self):
        """
        Use spline interpolation to resample the HRF to a new size that fits the new target length
        """
        # Original list
        hrf_indices = np.linspace(0, len(self.trace) - 1, len(self.trace))

        # Create a spline interpolation function
        spline = interp1d(hrf_indices, self.trace, kind='cubic')
        new_indices = np.linspace(0, len(self.trace) - 1, int(self.target_length))

        # Compressed list
        return spline(new_indices)

    def smooth(self, a):
        """
        Function that uses a gaussian filter to smooth the HRF trace.

        Function attributes:
            a (float) - Sigma value used in gaussian filter to dictate how much the HRF is smoothed
        """
        print(f'Smoothing HRF trace with Gaussian filter (sigma = {a})...')
        self.trace = self.gaussian_filter1d(self.trace, a)

        
    def normalize(self):
        """
        Function to normalize the trace between 0 and 1, useful for machine learning
        """
        self.trace = (self.trace - np.min(self.trace)) / (np.max(self.trace) - np.min(self.trace))

    def scale(self):
        """
        Function to scale around 1 using L2 normalization
        """
        self.trace /= np.linalg.norm(self.trace)
    
    def resample(self, std_seed = 0.0):
        """
        This resample function is an experimental resampling method for fNIRS (and potentially fMRI)
        for generating a new sample for machine learning and artificial intelligence training. The 
        general idea is to shift the HRF trace slightly within a confidence interval before deconvolving
        to generate multiple resampled fNIRS samples.

        Function attributes:
            std_seed (float) - Standard deviation seed between -3 and 3 to resample from the HRF trace deviation
        """
        if self.trace_std == None:
            raise ValueError(f"HRF does not have a trace deviation attached to it")
        # Resample trace
        return [mean + (std_seed * std) for mean, std in zip(self.trace, self.trace_std)]


    def plot(self, title = None, filepath = None, show = True):
        """
        # Function to plot the current HRF

        Function attributes:

        """

        plt.plot(self.trace) # Plot trace

        # Format plot
        plt.title(title) 
        
        if filepath: # If filepath provided
            plt.savefig(filepath) # Save

        elif show: # If plot requested to be shown
            plt.show() # Show plot

        plt.close() # Close all plots
        