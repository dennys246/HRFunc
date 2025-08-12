import pytest, mne, os, mne_nirs, csv
import hrfunc as hrf
from pathlib import Path

# Grab test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

def load_events():
    # Define runtime parameters
    sfreq = 7.81
    scan_length = 1999
    temp_events = []

    # Read in data
    with open(f"{TEST_DATA_DIR}/rps_results.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            temp_events.append([row[0], row[3]]) # each row is a list of strings
    # Remove header
    temp_events = [[float(event[0]), int(event[1])] for event in temp_events[1:]]


    # Expand events into NIRX space
    events = []
    last_position = 0
    for event in temp_events:
        # Grab event position
        position = int(round(event[0] * sfreq, 0))

        # Append 0's for non-events between events
        events += [0 for ind in range(last_position, position)]
        events += [1]

        # Preserve last position
        last_position = position
    
    events += [0 for ind in range(last_position, scan_length + 1)]
    return events
    


def load_nirx():
    subject_1 = mne.io.read_raw_nirx(f"{TEST_DATA_DIR}/NIRX_formatted/subject_1/")
    subject_2 = mne.io.read_raw_nirx(f"{TEST_DATA_DIR}/NIRX_formatted/subject_2/")
    return [subject_1, subject_2]

def load_fif():
    subject_1 = mne.io.read_raw_fif(f"{TEST_DATA_DIR}/FIF_formatted/subject_1.fif")
    subject_2 = mne.io.read_raw_fif(f"{TEST_DATA_DIR}/FIF_formatted/subject_2.fif")
    return [subject_1, subject_2]

def load_snirf():
    subject_1 = mne.io.read_raw_snirf(f"{TEST_DATA_DIR}/sNIRF_formatted/subject_1.snirf")
    subject_2 = mne.io.read_raw_snirf(f"{TEST_DATA_DIR}/sNIRF_formatted/subject_2.snirf")
    return [subject_1, subject_2]