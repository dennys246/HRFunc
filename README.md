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
- ✅ HRF deconvolution for fNIRS
- ✅ Easy-to-use API compatible with MNE
- ✅ Fast computations with NumPy + SciPy
- ✅ Supports group-level aggregation
- ✅ Community database of HRFs called the HRtree

---

## Installation

You can easily install hrfunc via pip, as long as you have python version 3.8 or higher.

```bash
pip install hrfunc
```

---

## **HRfunc Quickstart ##

You can estimate neural activity directly in your fNIRS data by creating a hrfunc.montage() object with one of your scan's loaded in through MNE. 

## HRfunc Usage Example **

```python
# ---- 1. Import MNE and HRfunc ---- #
import mne
import hrfunc as hrf

# ---- 2. Prepare Your Data ---- #

# Define all the filepaths to your fNIRS data - could be .snirf, .fif, or nirx formats
scan_filepaths = ['path/to/scan_1.snirf', 'path/to/another/scan_2.snirf']

# Python hack use glob to grab em' all
from glob import glob
scan_filepaths = glob("path/**/scan_*.snirf") # Use *, **, or ? (wildcards) to grab files by patterns

#  - Load Raw fNIRS data through MNE or MNE_NIRS -
scans = []
for filepath in scan_filepaths:
    scans.append(mne.io.read_raw_snirf(filepath))

# - Load you're events into a list -
# In this example we're using events stored in a text file
events = []
with open("task_events.txt", "r") as file:
    for line in file.readlines():
        events.append(int(line.split('/n')[0]))

# ---- 3. Initialize an HRF montage ---- #

montage = hrf.montage(scan)

# ---- 4. Estimate Subject Level HRFs ---- #

for scan in scans:
    montage.estimate_hrfs(scan, duration = 30.0)

# ---- 5. Generate a Subject-Pool HRF Distribution ---- #

montage.generate_distribution()
montage.save("study_HRFs.json")

# ---- 6. Estimate Neural Activity ----- # 

for scan in scans:

    # Data replaced in-place
    montage.estimate_activity(scan)

    # Save the scan
    scan.save(f"deconvolved_{scan.filename}")

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