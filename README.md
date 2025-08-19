# HRFunc
HRFunc is a Python library for estimating hemodynamic response functions and neural activity from hemoglobin concentration in functional near infrared spectroscopy (fNIRS), through modeling hemodynamic response functions through the brain recorded from fNIRS signals and deconvolving neural activity. Toeplitz deconvolution with Tikhonov regularization is employed for HRF and neural activity estimation. 

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- ✅ HRF and neural activity estimation for fNIRS
- ✅ Estimation through Toeplitz deconvolution with Tikhonov Reg.
- ✅ Community sourced HRF estimates stored in the HRtree
- ✅ Easy-to-use API compatible with MNE
- ✅ Fast computations with NumPy + SciPy
- ✅ Supports group-level aggregation
- ✅ Neural activity estimation for block task and resting state scans

---

## Installation

You can easily install hrfunc via pip, as long as you have python version 3.8 or higher.

```bash
pip install hrfunc
```

---

## HRfunc Quickstart ##

You can estimate channel-wise hemodynamic response functions and neural activity directly within your subjects fNIRS data through the hrfunc library. The hrfunc.montage() object orchestrates these estimations through three simple steps: 

1. Prepare your fNIRS and event data
2. Initialize an HRfunc montage
3. Estimate subject channel-wise HRFs
4. Calculate a subject-pool wide HRF distribution
5. Estimate neural activity in each subjects scans

## 1. Preparing Your Data ##

HRfunc leverages the MNE Python libraries standard fNIRS scan objects
to estimate HRFs and neural activity. To prepare you're data by simply
loading each fNIRS scan through MNE and creating a event impulse timeseries
representing when events occured in your

```python
import mne
# - - - -  Prepare Your fNIRS Data - - - - #
# Load in your raw fNIRS data through the MNE library
# and load into a list for easy access

# - Create a List of All of Your Subject Filepaths
scan_paths = ['path/to/sub-1.snirf', 
    'path/to/another/sub_2.snirf',
    'path/to/yet/another/sub_3.snirf']

# (Optional hack) use glob to grab them all! 
from glob import glob
# Use *, **, or ? (wildcards) to define ambiguous file patterns
# and grab all your files in one line
scan_paths = glob("path/**/sub*.snirf") 

#  - Load Raw fNIRS Data through MNE -
scans = []
for path in scan_paths: # Load through you're datatypes mne.io read call
    scans.append(mne.io.read_raw_snirf(path)) # .snirf format


# - - - -  Prepare Your Events - - - - #
# Load/create a list of 0's and 1's representing when events occur
# in your fNIRS data.

 # In this example, we're loading events in a text file
with open("task_events.txt", "r") as file:
    events = [int(line.split('/n')[0] for line in file.readlines()]

```

## HRfunc Usage Example ##

Once you're data is loaded, you 

```python

# - - - - 2. Initialize an HRfunc montage - - - - #
# Pass in one of your scans into the hrf.montage() to intialize
# an HRF estimation node for each of your montages optodes.

import hrfunc as hrf

montage = hrf.montage(scan)


# - - - - 3. Estimate Subject Level HRFs - - - - #
# Pass each of your scans and their corresponding events
# into the estimate_hrfs() function to estimate subject level 
# channel-wise estimates.

for scan in scans:
    montage.estimate_hrfs(scan, events, duration = 30.0)


# - - - - 4. Generate a Subject-Pool HRF Distribution - - - - #
# Generate channel-wise HRF estimates across the subject pool

montage.generate_distribution()
montage.save("study_HRFs.json")


# - - - - 5. Estimate Neural Activity - - - - # 
# Use the subject-pool wide HRF estimates to estimate
# channel wise neural activity for each subject

for scan in scans:
    # Estimate neural activity and replace in-place of the MNE object
    montage.estimate_activity(scan)

    # Save the scan
    scan.save(f"neural_activity_{scan.filename}")

```
## HRtree Usage Example ##
WARNING: The HRtree is currently very limited in the HRF's available and
you may need to estimate your own HRF's or rely on a canonical HRF for
estimating neural activity

HRfunc can only estimate HRFs and neural activity from fNIRS data
with events occuring during the scan. In these situations you could
skip straight to estimating neural activity and rely on the in-built
canonical HRF.

Alternatively you could search the HRtree for experimentally related
HRF's! HRfunc's hybrid tree-hash table data structure has a number of
useful functions for searching for useful HRF's.


```python

# - - - - Localize HRFs with your context of interest - - - - #
import hrfunc as hrf

# Localize any HRF's within range that contain task and age requested
montage = hrf.localize_hrfs(scan, max_distance = 0.001, task = 'flanker', age = [5, 6, 7])

# - - - - Filter again for another experimental context - - - - #
montage = montage.branch(demographic = ["black", "women"])

# - - - - Further filter the montage by percent similarity - - - - #
# Filter for specifically HRF's that meet a similarity threshold
montage.filter(similarity_threshold = 0.95)

# - - - - Estimate neuiral activity using found HRFs - - - - #
# NOTE: Relies on canonical HRF for a given optode if no 
# HRF was found for the given optode/experimental context
montage.estimate_acivity(scan)

```

---



## **Documentation**
For more comprehensive documentation on the tool, visit www.hrfunc.org

## Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
Distributed under the BSD-3 License. See [LICENSE](LICENSE) for details.

## Citation
If you use `hrfunc` in your research, please cite: