
# Load in our fNIRS data

import mne

scans = []
paths = ['subject_1.snirf', 'subject_2.snirf']
for path in paths:
    scans.append(mne.io.read_raw_snirf(path))

# Load in our events

with open("events.txt", "r") as file:
    events = [line.split("\n")[0] for line in file.readlines()]

# Load in our HRfunc library and initialize a montage

import hrfunc as hrf

montage = hrf.montage()

# Estimate subject level HRFs

for scan in scans:
    montage.estimate_hrfs(scan, events, duration = 30.0)

# Generate a subject-pool wide HRF distribution

montage.generate_distribution()

# Save calculated HRFs

montage.save("study_HRFs.json")

# Load them up for later use!

montage = hrf.load_montage("study_HRFs.json")

#Estimate neural activity in each scan

for scan in scans:
    montage.estimate_activity(scan)

    # Save deconvolve fNIRS scans with neural activity
    scan.save(f"deconv_{scan.filename}")

# Done!
