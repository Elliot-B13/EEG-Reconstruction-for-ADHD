import mne

# Path to the EDF file
edf_path = "/sub-01_task-BT_acq-BLP_eeg.edf"

# Load the EDF file (preload=True loads data into memory for faster plotting/processing)
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Print a short summary of the Raw object (optional)
print(raw)

# Plot the raw time-series data (interactive viewer)
raw.plot(duration=10, n_channels=16, scalings="auto", block=True)

# Compute and plot the power spectral density (PSD)
raw.compute_psd(fmin=0.5, fmax=50).plot()
