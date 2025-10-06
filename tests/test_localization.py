import pytest, mne, os, time
import test_loading as loader
import hrfunc as hrf
from pathlib import Path

print("TEST: Assessing HRF localization capabilities...")

# Grab test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

for filetype_load, filetype in zip([loader.load_nirx, loader.load_snirf, loader.load_fif], ['NIRX', 'SNIRF', 'FIF']):
    print(f"Testing {filetype} files")
    scans = filetype_load()

    montage = hrf.montage(scans[0], verbose = True)
    print(scans[0])
    for optode in montage.channels.values():
        print(f"{optode.ch_name} - {optode.x} {optode.y} {optode.z}")

    montage.hbo_tree = hrf.tree(f"{TEST_DATA_DIR}/hrfs/hbo_hrtree.json")
    montage.hbr_tree = hrf.tree(f"{TEST_DATA_DIR}/hrfs/hbr_hrtree.json")

    # localize HRFs with montage object and rock-paper-scissors HRtree
    montage.localize_hrfs(0.5, verbose = True)

    # Localize HRFs with function call and true HRtree
    print(f"Localizing true HRFs...")
    montage = hrf.localize_hrfs(scans[0], 0.001, verbose = True)

    # Check that all channels were found
    for ch_name in montage.channels.keys():
        if montage.channels[ch_name].context['method'] == 'canonical':
            RuntimeError(f"ERROR: True channel {ch_name} not found in localiztion test")

    del montage

    #print("Attempting to localize sham HRFs, loading new montage...")
    montage = hrf.montage(scans[0])

    # Load HRFs with sham locations
    print(f"Loading sham edge-case HRFs into HbO and HbR tree")
    montage.hbo_tree = hrf.tree(f"{TEST_DATA_DIR}/hrfs/sham_hbo_hrtree.json")
    montage.hbr_tree = hrf.tree(f"{TEST_DATA_DIR}/hrfs/sham_hbr_hrtree.json")

    montage.localize_hrfs(max_distance = 0.001, verbose = True)

    # Check that all channels were found
    for ch_name in montage.channels.keys():
        if montage.channels[ch_name].context['method'] != 'canonical':
            RuntimeError(f"ERROR: Sham localization found an HRF, sham localization failed to fail and rely on canonical HRF")

    del montage
