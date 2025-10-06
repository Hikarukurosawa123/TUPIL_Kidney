import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from collections import defaultdict

from .rdataread_func import rdataread
from .ReadClariusYML_func import ReadClariusYML
from skimage.transform import resize


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


def rf_roi_fft_parametric(patient_id, image_num, root_dir, frame_idx=-1,
                          Nfft=1024, window_size=64, midband_range=(2.0, 6.0)):
    """
    Compute MBF and SI parametric maps from ROI RF data (per scan line),
    and return masked maps limited to the kidney ROI.
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    # --- Locate patient folder ---
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

    # --- ROI mask ---
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

    params = ReadClariusYML(fname, header["lines"])
    fs = params["SamplingRate"]  # MHz
    dz = 1 / fs  # sample spacing (μs per sample)

    # --- Sliding window setup ---
    step = window_size
    n_windows = nsamples // step
    MBF_map = np.full((n_windows, nlines), np.nan)
    SI_map = np.full((n_windows, nlines), np.nan)

    freqs = np.linspace(0, fs / 2, Nfft // 2)
    f1, f2 = midband_range
    band_mask = (freqs >= f1) & (freqs <= f2)

    # --- Process each RF line ---
    for line_idx in range(nlines):
        if np.any(kidney_mask[:, line_idx] > 0):
            rf_line = data[frame_idx, :, line_idx]
            mask_line = kidney_mask[:, line_idx]

            for w in range(n_windows):
                start = w * step
                end = start + window_size
                if end > nsamples:
                    break

                roi_segment = rf_line[start:end] * mask_line[start:end]
                if np.count_nonzero(mask_line[start:end]) == 0:
                    continue

                spectrum = np.fft.fft(roi_segment, n=Nfft)
                spectrum = np.abs(spectrum[:Nfft // 2])
                spec_db = 20 * np.log10(spectrum + 1e-12)

                f_band = freqs[band_mask]
                spec_band = spec_db[band_mask]
                slope, intercept = np.polyfit(f_band, spec_band, 1)

                fc = (f1 + f2) / 2
                MBF_map[w, line_idx] = slope * fc + intercept
                SI_map[w, line_idx] = intercept

        # --- Convert window index to depth ---
    depth_axis = np.arange(n_windows) * step * dz * 1.54 / 2 * 1e3  # mm
    width_axis = np.arange(nlines)

    # --- Resize kidney mask to match parametric map resolution ---
    from skimage.transform import resize
    mask_resized = resize(kidney_mask, MBF_map.shape, order=0, preserve_range=True)

    # --- Apply kidney mask to parametric maps ---
    # Keep ROI values; set background to 0 (not NaN)
    MBF_map_masked = MBF_map * mask_resized
    SI_map_masked  = SI_map  * mask_resized

    # Set background explicitly to 0 for clean display and saving
    MBF_map_masked[mask_resized == 0] = 0
    SI_map_masked[mask_resized == 0]  = 0

    # --- Plot masked MBF ---
    plt.figure(figsize=(10, 5))
    plt.imshow(
        MBF_map_masked,
        aspect='auto',
        cmap='turbo',
        extent=[width_axis[0], width_axis[-1], depth_axis[-1], depth_axis[0]],
        vmin=np.nanmin(MBF_map_masked[mask_resized > 0]),
        vmax=np.nanmax(MBF_map_masked[mask_resized > 0])
    )
    plt.colorbar(label="MBF [dB]")
    plt.title(f"MBF Parametric Map — P{patient_id} Image {image_num}")
    plt.xlabel("Scan Line")
    plt.ylabel("Depth [mm]")
    plt.gca().set_facecolor("black")
    plt.show()

    # --- Plot masked SI ---
    plt.figure(figsize=(10, 5))
    plt.imshow(
        SI_map_masked,
        aspect='auto',
        cmap='viridis',
        extent=[width_axis[0], width_axis[-1], depth_axis[-1], depth_axis[0]],
        vmin=np.nanmin(SI_map_masked[mask_resized > 0]),
        vmax=np.nanmax(SI_map_masked[mask_resized > 0])
    )
    plt.colorbar(label="Spectral Intercept [dB]")
    plt.title(f"Spectral Intercept (SI) Map — P{patient_id} Image {image_num}")
    plt.xlabel("Scan Line")
    plt.ylabel("Depth [mm]")
    plt.gca().set_facecolor("black")
    plt.show()

    # --- Return masked maps (background = 0) ---
    return MBF_map_masked, SI_map_masked, depth_axis, width_axis, mask_resized


def rf_parametric_maps(patient_id, image_num, root_dir, save=False, save_resized=True, out_size=(224, 224), show=True):
    """
    Generate and save MBF and SI parametric maps for a given patient/image.
    Output is saved under the CURRENT working directory:
      - ./MBF/
      - ./SI/
    """

    # Create output folders in current working directory
    os.makedirs("MBF", exist_ok=True)
    os.makedirs("SI", exist_ok=True)

    # Patient ID formatting
    patient_id_str = f"P{patient_id}"
    patient_id_mask = f"{patient_id_str}_"

    # Search for matching folder
    files = os.listdir(root_dir)
    desired_substring = None
    text = None

    for fname in files:
        if patient_id_mask in fname:
            text = fname
            pattern = rf"({patient_id_str}_[A-Z0-9]+_01)"
            match = re.search(pattern, fname)
            if match:
                desired_substring = match.group(1)

    if desired_substring is None:
        raise FileNotFoundError("Could not find patient data folder.")

    # Construct path
    pathname = os.path.join(root_dir, text, f"{patient_id_str}_Image_{image_num}")
    filename = f"{desired_substring}_Image_{image_num}_rf"

    # --- Compute MBF & SI maps ---
    MBF_map_masked, SI_map_masked, depth_axis, width_axis, mask_resized  = rf_roi_fft_parametric(patient_id, image_num, root_dir)

   
    # --- Save resized 224×224 images ---
    if save_resized:
        MBF_resized = resize(MBF_map_masked, out_size, anti_aliasing=True)
        SI_resized = resize(SI_map_masked, out_size, anti_aliasing=True)

        mbf_resized_path = os.path.join("MBF", f"{filename}_MBF_resized.png")
        si_resized_path = os.path.join("SI", f"{filename}_SI_resized.png")


        plt.imsave(mbf_resized_path, MBF_resized, cmap="gray", vmin=np.nanmin(MBF_resized), vmax=np.nanmax(MBF_resized))
        plt.imsave(si_resized_path, SI_resized, cmap="gray", vmin=np.nanmin(SI_resized), vmax=np.nanmax(SI_resized))


        print(f"Saved MBF resized → {mbf_resized_path}")
        print(f"Saved SI  resized → {si_resized_path}")

    # --- Display ---
    if show:
        extent = [width_axis[0], width_axis[-1], depth_axis[-1], depth_axis[0]]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im1 = axes[0].imshow(MBF_map_masked, cmap="jet", aspect="auto", extent=extent)
        axes[0].set_title(f"MBF Map: {filename.replace('_', '\\_')}")
        axes[0].set_xlabel("Width (mm)")
        axes[0].set_ylabel("Depth (mm)")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(SI_map_masked, cmap="jet", aspect="auto", extent=extent)
        axes[1].set_title(f"SI Map: {filename.replace('_', '\\_')}")
        axes[1].set_xlabel("Width (mm)")
        axes[1].set_ylabel("Depth (mm)")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

def plot_group_ffts(root_dir, csv_file, Nfft=1024, midband_range=(2.0, 6.0)):
    """
    Compute ROI FFTs and Mid-Band Fit (MBF) per RF line (no sliding window).
    Compare MBF between eGFR >= 60 vs < 60 groups.
    """

    # --- Load eGFR data ---
    eGFR_data = pd.read_csv(csv_file)
    eGFR_data.rename(columns={'Patient ID': 'patient_id', 'eGFR (abs/closest)': 'eGFR'}, inplace=True)
    eGFR_data['patient_id'] = eGFR_data['patient_id'].astype(int)
    eGFR_data.set_index('patient_id', inplace=True)

    spectra_high, spectra_low = [], []
    mbf_high, mbf_low = [], []
    freqs_ref = None

    # --- Loop through patient folders ---
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

        # Find patient folder that contains the expected "_01" pattern
        patient_folders = [f for f in os.listdir(root_dir) if f.startswith(f"P{patient_id}_") and "_01" in f]
        if not patient_folders:
            print(f"No valid folder found for P{patient_id}, skipping...")
            continue
        patient_folder_name = patient_folders[0]
        patient_path = os.path.join(root_dir, patient_folder_name)

        # Loop over image subfolders
        for subfolder in os.listdir(patient_path):
            if not re.search(rf"P{patient_id}_Image_\d+", subfolder):
                continue

            image_match = re.search(r"Image_(\d+)", subfolder)
            if image_match is None:
                continue
            image_num = int(image_match.group(1))

            try:
                pathname = os.path.join(patient_path, subfolder)

                # Find RF file
                rf_files = [f for f in os.listdir(pathname) if f.endswith("_rf.raw")]
                if not rf_files:
                    print(f"No RF file found in {pathname}, skipping...")
                    continue
                rf_path = os.path.join(pathname, rf_files[0])

                # Load RF data
                with open(rf_path, "rb") as f:
                    hinfo = np.fromfile(f, dtype=np.int32, count=5)
                    num_frames = hinfo[1]

                data, header = rdataread(rf_path, num_frames)
                nframes, nsamples, nlines = data.shape
                RF = data[-1, :, :]  # last frame

                # Load ROI
                roi_path = os.path.join(pathname, "ROIs")
                roi_files = [f for f in os.listdir(roi_path) if "raw_kidney" in f]
                if not roi_files:
                    print(f"No kidney ROI found in {roi_path}, skipping...")
                    continue
                kidney_data = loadmat(os.path.join(roi_path, roi_files[0]))
                kidney_mask = kidney_data["kidney_mask"]

                # Imaging parameters
                params = ReadClariusYML(rf_path, header["lines"])
                fs = params["SamplingRate"]

                # Frequency axis
                freqs = np.linspace(0, fs / 2, Nfft // 2)
                if freqs_ref is None:
                    freqs_ref = freqs

                # MBF per line
                mbf_line_vals = []
                line_spectra = []

                f1, f2 = midband_range
                band_mask = (freqs >= f1) & (freqs <= f2)
                freqs_band = freqs[band_mask]
                fc = (f1 + f2) / 2.0

                for line_idx in range(nlines):
                    mask_line = kidney_mask[:, line_idx]
                    nonzero_count = np.count_nonzero(mask_line)
                    if nonzero_count == 0:
                        continue

                    rf_line = RF[:, line_idx] * mask_line
                    spectrum = np.fft.fft(rf_line, n=Nfft) / nonzero_count
                    spectrum_db = 20 * np.log10(np.abs(spectrum[:Nfft // 2]) + 1e-12)
                    line_spectra.append(spectrum_db)

                    # MBF
                    spec_band = spectrum_db[band_mask]
                    if len(spec_band) >= 3 and not np.all(np.isnan(spec_band)):
                        a, b = np.polyfit(freqs_band, spec_band, 1)
                        mbf_val = a * fc + b
                        mbf_line_vals.append(mbf_val)

                if not line_spectra:
                    continue

                avg_spectrum_db = np.mean(np.vstack(line_spectra), axis=0)
                mbf_mean = np.nanmean(mbf_line_vals)
                mbf_std = np.nanstd(mbf_line_vals)

                # Store by group
                if egfr_label == "high":
                    spectra_high.append(avg_spectrum_db)
                    mbf_high.append((mbf_mean, mbf_std))
                else:
                    spectra_low.append(avg_spectrum_db)
                    mbf_low.append((mbf_mean, mbf_std))

            except Exception as e:
                print(f"Skipping {subfolder}: {e}")

    # --- Average spectra ---
    avg_high = np.mean(np.vstack(spectra_high), axis=0) if spectra_high else None
    avg_low = np.mean(np.vstack(spectra_low), axis=0) if spectra_low else None

    # --- Plot spectra ---
    plt.figure(figsize=(8, 5))
    if avg_high is not None:
        plt.plot(freqs_ref, avg_high, 'b', label="eGFR ≥ 60")
    if avg_low is not None:
        plt.plot(freqs_ref, avg_low, 'r', label="eGFR < 60")
    plt.title("Average ROI FFT by eGFR Group")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude [dB]")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot MBF comparison ---
    high_means, high_stds = zip(*mbf_high) if mbf_high else ([], [])
    low_means, low_stds = zip(*mbf_low) if mbf_low else ([], [])

    plt.figure(figsize=(6, 5))
    plt.boxplot([high_means, low_means], labels=["eGFR ≥ 60", "eGFR < 60"])
    plt.ylabel("Mid-Band Fit (dB)")
    plt.title("Comparison of MBF by eGFR Group")
    plt.grid(True, axis='y')
    plt.show()

    print(f"Mean MBF (eGFR ≥ 60): {np.mean(high_means):.2f} ± {np.mean(high_stds):.2f} dB")
    print(f"Mean MBF (eGFR < 60): {np.mean(low_means):.2f} ± {np.mean(low_stds):.2f} dB")

    return freqs_ref, avg_high, avg_low, mbf_high, mbf_low

from skimage.io import imread
def plot_parametric_maps(root_dir, csv_file, map_type="MBF", cmap="turbo", ncols=4):
    """
    Plot MBF or SI parametric maps from folders, grouped by eGFR classification.
    
    Args:
        root_dir (str): Root directory containing 'MBF' and 'SI' folders.
        csv_file (str): Path to eGFR CSV file.
        map_type (str): 'MBF' or 'SI' (folder name must match).
        cmap (str): Colormap for visualization.
        ncols (int): Number of columns in grid layout.
    """

    # --- Load eGFR data ---
    eGFR_data = pd.read_csv(csv_file)
    eGFR_data.rename(columns={'Patient ID': 'patient_id', 'eGFR (abs/closest)': 'eGFR'}, inplace=True)
    eGFR_data['patient_id'] = eGFR_data['patient_id'].astype(int)
    eGFR_data.set_index('patient_id', inplace=True)

    map_folder = os.path.join(root_dir, map_type)
    if not os.path.isdir(map_folder):
        raise ValueError(f"{map_folder} not found. Expected folder: 'MBF' or 'SI' inside root_dir.")

    # --- Collect image paths ---
    image_files = [f for f in os.listdir(map_folder) if f.lower().endswith(('.png', '.jpg', '.tif'))]
    image_info = []

    for img_file in image_files:
        match = re.search(r"P(\d+)", img_file)
        if not match:
            continue
        patient_id = int(match.group(1))
        if patient_id not in eGFR_data.index:
            continue

        egfr_val = eGFR_data.loc[patient_id, 'eGFR']
        group = "eGFR ≥ 60" if egfr_val >= 60 else "eGFR < 60"
        image_info.append((patient_id, group, os.path.join(map_folder, img_file)))

    if not image_info:
        print("No matching images found.")
        return

    # --- Sort by group (optional for display organization) ---
    image_info.sort(key=lambda x: (x[1], x[0]))  # group first, then patient_id

    # --- Plot grid ---
    n_images = len(image_info)
    nrows = int(np.ceil(n_images / ncols))

    plt.figure(figsize=(4 * ncols, 4 * nrows))

    for idx, (pid, group, path) in enumerate(image_info, 1):
        img = imread(path)
        plt.subplot(nrows, ncols, idx)
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
        plt.title(f"P{pid} — {group}", fontsize=10)

    plt.suptitle(f"{map_type} Parametric Maps by eGFR Group", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_group_ffts_avg(root_dir, csv_file, Nfft=1024):
    """
    Compute average ROI FFT per patient/image using rf_roi_fft
    and compare between eGFR >= 60 vs < 60 groups.

    Parameters
    ----------
    root_dir : str
        Root directory containing patient folders.
    csv_file : str
        Path to eGFR CSV file.
    Nfft : int, default=1024
        FFT length.
    """

    # --- Load eGFR data ---
    eGFR_data = pd.read_csv(csv_file)
    eGFR_data.rename(columns={'Patient ID': 'patient_id', 'eGFR (abs/closest)': 'eGFR'}, inplace=True)
    eGFR_data['patient_id'] = eGFR_data['patient_id'].astype(int)
    eGFR_data.set_index('patient_id', inplace=True)

    spectra_high, spectra_low = [], []
    freqs_ref = None

    # --- Loop through patients ---
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

        # --- Loop through images ---
        for subfolder in os.listdir(patient_path):
            if f"P{patient_id}_Image_" not in subfolder:
                continue

            try:
                image_num = int(re.findall(r"Image_(\d+)", subfolder)[0])
                freqs, avg_spectrum_db = rf_roi_fft(patient_id, image_num, root_dir, Nfft=Nfft)
                if freqs is None:
                    continue

                if freqs_ref is None:
                    freqs_ref = freqs

                if egfr_label == "high":
                    spectra_high.append(avg_spectrum_db)
                else:
                    spectra_low.append(avg_spectrum_db)

            except Exception as e:
                print(f"Skipping {subfolder}: {e}")

    # --- Average per group ---
    avg_high = np.mean(np.vstack(spectra_high), axis=0) if spectra_high else None
    avg_low = np.mean(np.vstack(spectra_low), axis=0) if spectra_low else None

    # --- Plot average spectra ---
    plt.figure(figsize=(8, 5))
    if avg_high is not None:
        plt.plot(freqs_ref, avg_high, 'b', label="eGFR ≥ 60")
    if avg_low is not None:
        plt.plot(freqs_ref, avg_low, 'r', label="eGFR < 60")
    plt.title("Average ROI FFT by eGFR Group")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude [dB]")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return freqs_ref, avg_high, avg_low