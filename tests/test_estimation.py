import pytest, mne, os, time
import hrfunc as hrf
import test_loading as loader
from pathlib import Path

print("TEST: Estimating HRF and neural activity from allows filetypes")

# Load rock-paper-scissors events
events = loader.load_events()

for filetype_load, filetype in zip([loader.load_nirx, loader.load_snirf, loader.load_fif], ['NIRX', 'SNIRF', 'FIF']):
    print(f"Testing {filetype} files")
    scans = filetype_load()

    montage = hrf.montage(scans[0])

    print(f"Estimating HRF for scans: {scans}")
    for scan in scans:
        montage.estimate_hrf(scan, events, duration = 30)
    
    print(f"Generating distribution")
    montage.generate_distribution()
    print(f"Montage channels: {montage.channels}")

    print(f"Saving HRF montage")
    montage.save("test_HRFs.json")

    del montage

    print(f"Loading HRF montage")
    montage = hrf.load_montage("test_HRFs.json")

    print("Estimating neural activity from montage and saving")
    for scan in scans:
        print(f"Deconvolving neural activity from {scan}")
        scan = montage.estimate_activity(scan)

        print(f"Saving deconvolved activity")
        scan.save("temp_deconv_NIRS.fif")

        print(f"Loading scan back up...")
        scan = mne.io.read_raw_fif("temp_deconv_NIRS.fif")

        del scan

        time.sleep(2)
        
        os.remove("temp_deconv_NIRS.fif")
    os.remove("test_HRFs.json")
