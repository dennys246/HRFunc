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
- ✅ Easy-to-use API compatible with MNE
- ✅ Fast computations with NumPy + SciPy
- ✅ Supports group-level aggregation
- ✅ Community database of HRFs called the HRtree
- ✅ Neural activity estimation for block task and resting state scans

---

## Installation

You can easily install hrfunc via pip, as long as you have python version 3.8 or higher.

```bash
pip install hrfunc
```

---

## HRfunc Quickstart ##

You can estimate neural activity directly in your fNIRS data by creating a hrfunc.montage() object with one of your scan's loaded in through MNE. 

## HRfunc Usage Example ##

```python
# ---- 1. Import MNE and HRfunc ---- #
import mne # Maybe csv or json too depending on how you store events
import hrfunc as hrf

# ---- 2. Prepare Your fNIRS Data ---- #

# - Create a List of All of Your Subject Filepaths -
scan_paths = ['path/to/sub-1.snirf', 
    'path/to/another/sub_2.snirf',
    'path/to/yet/another/sub_3.snirf']

# (Optional hack) use glob to grab them all! 
from glob import glob
# Use *, **, or ? (wildcards) to define file patterns and grab them all
scan_paths = glob("path/**/sub*.snirf") 

#  - Load Raw fNIRS Data through MNE -
scans = []
for path in scan_paths:
    # Load through you're datatypes mne.io call
    scans.append(mne.io.read_raw_snirf(path)) # .snirf format
    #scans.append(mne.io.read_raw_fif(path)) # .fif format
    #scans.append(mne.io.read_raw_nirx(path)) # NIRX format

# ---- 3. Prepare Your Events ---- #

# In this example we're using events stored in a text file
events = []
with open("task_events.txt", "r") as file:
    for line in file.readlines():
        events.append(int(line.split('/n')[0]))

# Events should be at the most as long as the fNIRS scan

# ---- 4. Initialize an HRF montage ---- #

montage = hrf.montage(scan)

# ---- 5. Estimate Subject Level HRFs ---- #

for scan in scans:
    montage.estimate_hrfs(scan, duration = 30.0)

# ---- 6. Generate a Subject-Pool HRF Distribution ---- #

montage.generate_distribution()
montage.save("study_HRFs.json")

# ---- 7. Estimate Neural Activity ----- # 

for scan in scans:

    # Data replaced in-place
    montage.estimate_activity(scan)

    # Save the scan
    scan.save(f"deconv_{scan.filename}")

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