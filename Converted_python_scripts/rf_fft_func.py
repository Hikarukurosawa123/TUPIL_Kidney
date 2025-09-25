import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from collections import defaultdict

from .rdataread_func import rdataread
from .ReadClariusYML_func import ReadClariusYML


def rf_roi_fft(patient_id, image_num, root_dir, frame_idx=-1, Nfft=1024):
    """
    Compute normalized FFT spectrum over all ROI lines for a given patient + image.
    """
    patient_id_str = f"P{patient_id}"
    patient_id_mask = f"{patient_id_str}_"
    files = os.listdir(root_dir)
    desired_substring, text = None, None

    for fname in files:
        if patient_id_mask in fname:
            text = fname
            match = re.search(rf"({patient_id_str}_[A-Z0-9]+_01)", fname)
            if match:
                desired_substring = match.group(1)

    if desired_substring is None:
        raise FileNotFoundError(f"Could not find patient {patient_id} data folder.")

    # --- File paths ---
    pathname = os.path.join(root_dir, text, f"{patient_id_str}_Image_{image_num}")
    filename = f"{desired_substring}_Image_{image_num}_rf"
    fname = os.path.join(pathname, filename + ".raw")

    # ROI mask
    roi_path = os.path.join(pathname, "ROIs")
    kidney_data = None
    for f in os.listdir(roi_path):
        if "raw_kidney" in f:
            kidney_data = loadmat(os.path.join(roi_path, f))
            break
    if kidney_data is None:
        raise FileNotFoundError("Kidney ROI file not found.")
    kidney_mask = kidney_data["kidney_mask"]

    # --- Load RF data ---
    with open(fname, "rb") as f:
        hinfo = np.fromfile(f, dtype=np.int32, count=5)
        num_frames = hinfo[1]

    data, header = rdataread(fname, num_frames)
    nframes, nsamples, nlines = data.shape
    if frame_idx == -1:
        frame_idx = nframes - 1

    # --- Parameters ---
    params = ReadClariusYML(fname, header["lines"])
    fs = params["SamplingRate"]  # MHz

    spectra = []

    # Loop through ROI lines
    for line_idx in range(nlines):
        if np.any(kidney_mask[:, line_idx] > 0):
            rf_line = data[frame_idx, :, line_idx]

            # Apply mask
            rf_line = rf_line * kidney_mask[:, line_idx]

            # Normalization factor = number of nonzero samples in ROI
            nonzero_count = np.count_nonzero(kidney_mask[:, line_idx])
            if nonzero_count == 0:
                continue

            spectrum = np.fft.fft(rf_line, n=Nfft) / nonzero_count
            spectrum = np.abs(spectrum[:Nfft // 2])
            spectra.append(spectrum)

    if len(spectra) == 0:
        return None, None

    avg_spectrum = np.mean(np.vstack(spectra), axis=0)
    avg_spectrum_db = 20 * np.log10(avg_spectrum + 1e-12)
    freqs = np.linspace(0, fs / 2, Nfft // 2)

    return freqs, avg_spectrum_db


def plot_group_ffts(root_dir, csv_file):
    """
    Compute FFTs for all patients and split by eGFR >= 60 vs < 60.
    Plot averaged FFT curves for each group.
    """
    # --- Load eGFR data ---
    eGFR_data = pd.read_csv(csv_file)
    eGFR_data.rename(columns={'Patient ID': 'patient_id', 'eGFR (abs/closest)': 'eGFR'}, inplace=True)
    eGFR_data['patient_id'] = eGFR_data['patient_id'].astype(int)
    eGFR_data.set_index('patient_id', inplace=True)

    spectra_high, spectra_low = [], []
    freqs_ref = None

    # Loop through patients
    for folder in os.listdir(root_dir):
        if not folder.startswith("P"):
            continue
        try:
            patient_id = int(re.findall(r"P(\d+)", folder)[0])
        except:
            continue

        if patient_id not in eGFR_data.index:
            continue

        egfr_val = eGFR_data.loc[patient_id, 'eGFR']
        egfr_label = "high" if egfr_val >= 60 else "low"

        patient_path = os.path.join(root_dir, folder)
        for subfolder in os.listdir(patient_path):
            if f"P{patient_id}_Image_" in subfolder:
                try:
                    image_num = int(re.findall(r"Image_(\d+)", subfolder)[0])
                    freqs, spectrum_db = rf_roi_fft(patient_id, image_num, root_dir)
                    if freqs is None:
                        continue
                    freqs_ref = freqs
                    if egfr_label == "high":
                        spectra_high.append(spectrum_db)
                    else:
                        spectra_low.append(spectrum_db)
                except Exception as e:
                    print(f"Skipping {subfolder}: {e}")

    # --- Average per group ---
    avg_high = np.mean(np.vstack(spectra_high), axis=0) if spectra_high else None
    avg_low = np.mean(np.vstack(spectra_low), axis=0) if spectra_low else None

    # --- Plot ---
    plt.figure(figsize=(8,5))
    if avg_high is not None:
        plt.plot(freqs_ref, avg_high, 'b', label="eGFR ≥ 60")
    if avg_low is not None:
        plt.plot(freqs_ref, avg_low, 'r', label="eGFR < 60")

    plt.title("Average ROI FFT by eGFR Group")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid(True)
    plt.legend()
    plt.show()
