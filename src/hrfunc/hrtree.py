import json, random, math, re, nilearn, scipy, os
from . import hrhash, hrfunc
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import deque
from nilearn.glm.first_level import spm_hrf

class tree:

    def __init__(self, hrf_filename = None, **kwargs):
        """
        A k-d tree data structure for storing HRF estimates across NIRX space

        Arguments:
            - hrf_filename (str) - Filepath to json file containing HRF estimates
            - context arguments - Any context item you'd like to include in the HRF search

        Functions:
            - compare() - Filter the HRF file for contexts of interest
        """
        # Find hrfunc install library
        self.lib_dir = os.path.dirname(hrfunc.__file__)
        if hrf_filename:
            self.hrf_filename = hrf_filename
        else:
            self.hrf_filename = None

        self.root = None
        self.branched = False     

        # Set and update context
        self.context = {
            'method': 'toeplitz',
            'doi': 'temp',
            'ch_name': 'global',
            'study': None,
            'task': None,
            'conditions': None,
            'stimulus': None,
            'intensity': None,
            'duration': 30.0,
            'protocol': None,
            'age_range': None,
            'demographics': None
        }
        self.context = {**self.context, **kwargs} 
        self.context_weights = {key: 1.0 for key in self.context.keys()}

        self.hasher = hrhash.hasher(self.context)

        if self.hrf_filename and os.path.exists(self.hrf_filename):
            self.load_hrfs(self.hrf_filename)

    def load_hrfs(self, hrf_filename, similarity_threshold = 0.0, oxygenated = None):
        """
        Orchestrate building the HRF tree while filtering for specific context

        Arguments:
            hrf_filename (str) - Filename of the HRF json to load into the tree
            sim_threshold (float) - Threshold to allow or exclude HRF's based on context, defaults to 0.0 or no threshold
            context_weights (dict) - Weight to attach to each context during similarity comparison
        """

        with open(hrf_filename, 'r') as json_file:
            hrfs_json = json.load(json_file) # Load HRFs from json
        
        for key, channel in hrfs_json.items():
            
            # Grab channel and doi info
            split = key.split('-')
            doi = split.pop()
            ch_name = ' '.join(split)

            # Skip if oxygenation/deoxygenation filtering requested
            oxygenation = hrfunc._is_oxygenated(ch_name)
            if oxygenated == False and oxygenation:
                continue
            if oxygenated and oxygenation == False:
                continue

            # If similarity check requested
            if similarity_threshold > 0.0:
                context_similarity = self.compare_context(self.context, channel['context'], self.context_weights)
                if context_similarity < similarity_threshold:
                    print(f"Skipping {channel}")
                    continue

            # create a new hrf node
            new_hrf = HRF(
                doi, 
                ch_name, 
                float(channel['context']['duration']), 
                float(channel['sfreq']), 
                np.asarray(channel['hrf_mean'], dtype=np.float64), 
                np.asarray(channel['hrf_std'], dtype=np.float64), 
                channel['location'], 
                channel['context'])
            
            # Insert hrf node into tree
            node = self.insert(new_hrf)

            # Add newly added node into HRHash table
            for context in self.context:
                self.hasher.add(context, node)

    def insert(self, hrf, depth = 0, node = None):
        """Insert a new node into the 3D k-d tree based on spatial position."""

        if self.root is None:
            print(f"Setting root... {hrf}")
            self.root = hrf
            return self.root

        if node is None:
            node = self.root

            canonical_hrf = nilearn.glm.first_level.glover_hrf(tr = 0.128, oversampling = 1, time_length = 30.0)
            if self.root.oxygenation:
                ch_name = 'global-hbo'
            else:
                canonical_hrf = [-point for point in canonical_hrf]
                ch_name = 'global-hbr'

            self.root.right = HRF(
                'canonical',
                ch_name,
                30.0,
                7.81,
                canonical_hrf,
                location = [359.0, 359.0, 359.0]
            )

        axis = depth % 3  # Cycle through x, y, z

        h_val = (hrf.x, hrf.y, hrf.z)[axis]
        n_val = (node.x, node.y, node.z)[axis]

        # Handle duplicates by jittering location
        if h_val == n_val and hrf.x == node.x and hrf.y == node.y and hrf.z == node.z:
            for val in (hrf.x, hrf.y, hrf.z):
                print(f"WARNING: Jittering location for {hrf.ch_name}, same location as the following node.../n{node.__repr__()}")
                val += 1e-10 # Jitter location while staying above 64-precision double threshold
        
        # If the current node is less than the new node
        if h_val < n_val: 
            if node.left is None: # If the left node is empty
                node.left = hrf
                return node.left
            else: # If the left node is not empty
                return self.insert(hrf, depth + 1, node.left)
            
        # If the current node is greater than the new node
        else: 
            if node.right is None:
                node.right = hrf
                return node.right
            else:
                return self.insert(hrf, depth + 1, node.right)

    def filter(self, similarity_threshold = 0.95, node = None, **kwargs):
        """
        Filter on experimental contexts
        """
        if node is None: # Set up filtering
            if self.root is None: # If nothing loaded yet
                raise ValueError("No HRFs loaded yet, nothing to filter")
            
            node = self.root # Set root at node
            self.context = {**self.context, **kwargs} 

            if self.branched == False: # Branch to reduce number of hard comparison
                print("Branching tree on context before filtering")
                self.branch()

        if node.left: # If there's a left node
            self.filter(similarity_threshold, node.left)

        if node.right: # If there's a right node
            self.filter(similarity_threshold, node.right)

        # Check if the hrf matches the context
        context_similarity = self.compare_context(self.context, node.context, self.context_weights)
        if context_similarity > similarity_threshold: # If not similar enough to requested context
            self.delete(node) # Exclude derived HRF

    def compare_context(self, first_context, second_context):
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
                    if self.context_weights: # If a context weight provided
                        same += 1 * self.context_weights[key] # Weight similarity score
                    else: # add 
                        same += 1

            # Calculate context-specific similarity and append
            context_similarity.append(same/len(first_context)) 
        
        return sum(context_similarity) / len(context_similarity) # Average similarity and return

    def branch(self, **kwargs):
        """
        Accepts context keyword inputs via kwargs, updates the trees context
        and then builds a new tree filtering for just the context

        Arguments:
            **kwargs - Any context keyword value pair to branch on (i.e. doi, age, etc)
        """
        if kwargs:
            self.context = {**self.context, **kwargs} # Update context

        branch = tree('hrf_branch.json')

        for key, values in self.context.items(): # Iterate through all context items
            for value in values: # Iterate through each item in a context area
                # Hash on the value and iterate through the tree pointers
                context_references = self.hasher.search(value)
                for node in context_references:
                    branch.insert(node) # Insert node pointer into branch
        self.branched = True
        return branch

    def nearest_neighbor(self, optode, max_distance, node = None, depth=0, best=None):
        """
        Find the nearest neighbor to a target point in the 3D k-d tree.
        
        Arguments:
            node (HRF) - The current node in the search
            target (HRF) - The target HRF to find the nearest neighbor for
            depth (int) - The current depth of the search
            best (tuple) - The best node and distance found so far
        Returns:
            best (tuple) - The best node and distance found so far
        """
        if node is None: # Handle base case
            return best

        k = 3 
        axis = depth % k

        #Define current and target points
        point = (node.x, node.y, node.z)
        target_point = (optode.x, optode.y, optode.z)

        # Calculate euclidian distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([optode.x, optode.y, optode.z], [node.x, node.y, node.z])))

        # Check if this node is closer than the best found so far
        if best is None or distance < best[1]:
            best = (node, distance)

        # Figure out which side needs exploring
        if target_point[axis] < point[axis]:
            near_branch = node.left
            far_branch = node.right
        else:
            near_branch = node.right
            far_branch = node.left

        # Search nearest branch
        best = self.nearest_neighbor(optode, max_distance, near_branch, depth + 1, best)

        # Check if far branch needs to be explored
        if abs(target_point[axis] - point[axis]) < best[1]:
            best = self.nearest_neighbor(optode, max_distance, far_branch, depth + 1, best)

        if best and best[1] <= max_distance:
            return best  # return the node only
        else:
            return self.root.right, 0.0  # fallback to canonical HRF

    def radius_search(self, optode, radius, node = None, depth=0, results=None):
        """
        Collect all HRFs within a radius and return them

        Arguments:
            node (HRF object) - HRF estimate to compare against
            optode (HRF object) - HRF optode object passed in
            radius (float) - Maximum euclidian distance of radius 
            depth (int) - Current depths of the search (range 0 - 2)
            results (list) - Nodes found to be within a range passed through resursions
        """
        if node is None:
            return results or []

        if results is None:
            results = []

        axis = depth % 3
        node_coords = (node.x, node.y, node.z)
        optode_coords = (optode.x, optode.y, optode.z)

        # Check distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(optode_coords, node_coords)))
        if distance <= radius:
            results.append((node, distance))

        # Decide which branches to explore
        if optode_coords[axis] - radius < node_coords[axis]:
            self.radius_search(optode, radius, node.left, depth + 1, results)
            
        if optode_coords[axis] + radius > node_coords[axis]:
            self.radius_search(optode, radius, node.right, depth + 1, results)

        return results

    def save(self, filename = 'tree_hrfs.json'):
        hrfs = self.gather(self.root)
        # Save to a JSON file
        with open(filename, "w") as file:
            json.dump(hrfs, file, indent=4)
        return

    def gather(self, node):

        hrfs = {}
        if node.left:
            hrfs |= self.gather(node.left)
        if node.right:
            hrfs |= self.gather(node.right)
        hrfs |= {
            f"{'-'.join(node.ch_name.split(' '))}-{node.doi}": {
                "hrf_mean": np.asarray(node.trace).tolist(),
                "hrf_std": np.asarray(node.trace_std).tolist(),
                "location": [
                    node.x,
                    node.y,
                    node.z
                ],
                "oxygenation":node.oxygenation,
                "sfreq": node.sfreq,
                "context": node.context
            }
        }
        print(f"Node {node.ch_name} std: {node.trace_std}")
        return hrfs
    
    def traverse(self, node = None):
        if node is None:
            node = self.root
        
        if node.left:
            self.traverse(node.left)
        if node.right:
            self.traverse(node.right)

        print(f"Node {node.ch_name}")

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
    def __init__(self, doi, ch_name, duration, sfreq, trace, trace_std = None, location = None, estimates = [], **kwargs):
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
        # Add doi
        # Add doi
        self.doi = doi

        # Clean and add channel name
        self.ch_name = re.sub(r'[_\-\s]+', '_', ch_name.lower())
        self.oxygenation = hrfunc._is_oxygenated(self.ch_name)

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
            'method': 'global',
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

        self.estimates = estimates

        self.hrf_processes = [self.spline_interp]
        self.process_names = ['spline_interpolate']
        self.process_options = []

        self.built = False

    def __repr__(self):
        """String representation of the HRF object."""
        return f"HRF: {self.doi} - {self.ch_name} \nSampling frequency: {self.sfreq}\nLocation: [{self.x}, {self.y}, {self.z}]\nTrace length: {len(self.trace)}\nTrace standrad deviation: {self.trace_std}"

    def build(self, new_sfreq, plot = False, show = False):
        """ Run through the processes requested for generating an hrf """
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
        self.built = True


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

    def plot(self, plot_dir, show_legend = True):
        """
        Function to plot the current HRF in seconds.

        Parameters:
            plot_dir (str): Path to save plots for each HRF stage
            show_legend (bool): Whether to show the HRF legend
        """
        hrf_mean = self.trace
        hrf_std = self.trace_std
        time = np.arange(len(hrf_mean)) / self.sfreq  # Convert samples to seconds

        plt.figure(figsize=(8, 4))
        plt.plot(time, hrf_mean, label='Mean HRF', color='blue')
        plt.fill_between(time, hrf_mean - hrf_std, hrf_mean + hrf_std, color='blue', alpha=0.3, label='±1 SD')

        plt.xlabel('Time (s)')
        plt.ylabel('HRF amplitude')
        plt.title(f'Estimated HRF for {self.ch_name} with Standard Deviation')
        plt.grid(True)

        # Auto-scale y-axis ticks nicely
        y_min = round(min(hrf_mean - hrf_std), 1) - 0.1
        y_max = round(max(hrf_mean + hrf_std), 1) + 0.1
        plt.yticks(np.arange(y_min, y_max, 0.1))

        # Cleaner x-axis ticks based on time
        plt.xticks(np.arange(0, max(time) + 0.3, 2))  # e.g., every 2 seconds

        if show_legend:
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{self.ch_name}_hrf_estimate.png")
            