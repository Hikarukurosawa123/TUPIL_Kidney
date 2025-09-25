import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import loadmat
import yaml
from .rdataread_func import rdataread
from .ReadClariusYML_func import ReadClariusYML

def rf_image(patient_id, image_num, root_dir):
    # Patient ID formatting
    patient_id_str = f"P{patient_id}"
    patient_id_mask = f"{patient_id_str}_"

    # Search for matching folder
    files = os.listdir(root_dir)
    desired_substring = None
    text = None
    print("helloworld")

    for fname in files:
        if patient_id_mask in fname:
            text = fname
            print(text)
            pattern = rf"({patient_id_str}_[A-Z0-9]+_01)"
            match = re.search(pattern, fname)
            if match:
                desired_substring = match.group(1)

    if desired_substring is None:
        raise FileNotFoundError("Could not find patient data folder.")

    # Construct paths
    pathname = os.path.join(root_dir, text, f"{patient_id_str}_Image_{image_num}")
    filename = f"{desired_substring}_Image_{image_num}_rf"
    fname = os.path.join(pathname, filename + ".raw")
    print(fname)

    # Load kidney ROI (.mat file)
    roi_path = os.path.join(pathname, "ROIs")
    kidney_data = None
    for f in os.listdir(roi_path):
        if "raw_kidney" in f:
            kidney_data = loadmat(os.path.join(roi_path, f))
            break
    if kidney_data is None:
        raise FileNotFoundError("Kidney ROI file not found.")

    # Read raw RF data
    with open(fname, "rb") as f:
        hinfo = np.fromfile(f, dtype=np.int32, count=5)
        num_frames = hinfo[1]

    data, header = rdataread(fname, num_frames)
    print("Data shape:", data.shape)

    nframes, nsamples, nlines = data.shape
    num_frames = min(num_frames, header["frames"])

    # Imaging parameters
    params = ReadClariusYML(fname, header["lines"])
    delay = (153850 / 2) * (params["DelaySamples"] / (params["SamplingRate"] * 1e6))
    depth = (153850 / 2) * (nsamples / (params["SamplingRate"] * 1e6))

    arc_length = (params["ProbePitch"] / 1e3) * nlines
    FOV = (arc_length * 360) / (2 * np.pi * 45)
    width = 2 * (depth + delay) * np.tan(np.deg2rad(FOV / 2))

    # Display RF data image
    RF = data[-1, :, :]  # last frame
    BB = 20 * np.log10(1 + np.abs(hilbert(RF, axis=0)))
    title_name = filename.replace("_", "\\_")

    y = np.linspace(delay, depth + delay, 10)
    x = np.linspace(0, width, 10)

    plt.figure()
    plt.imshow(BB, extent=[x[0], x[-1], y[-1], y[0]],
               cmap="gray", vmin=15, vmax=70, aspect="auto")
    plt.title(f"RF Image: {title_name}")
    plt.xlabel("Width [cm]")
    plt.ylabel("Depth [cm]")
    plt.colorbar(label="dB")
    plt.show()

    # Masked kidney data
    kidney_mask = kidney_data["kidney_mask"]

    plt.figure()
    plt.imshow(kidney_mask, extent=[x[0], x[-1], y[-1], y[0]],
               cmap="gray", aspect="auto")
    plt.title(f"RF Image mask: {title_name}")
    plt.xlabel("Width [cm]")
    plt.ylabel("Depth [cm]")
    plt.show()

    plt.figure()
    plt.imshow(BB * kidney_mask, extent=[x[0], x[-1], y[-1], y[0]],
               cmap="gray", vmin=15, vmax=70, aspect="auto")
    plt.title(f"RF Image Kidney Mask: {title_name}")
    plt.xlabel("Width [cm]")
    plt.ylabel("Depth [cm]")
    plt.colorbar(label="dB")
    plt.show()

         # -------------------------
    # 2D Parametric Frequency Map (max FFT frequency per pixel window)
    # -------------------------

    # Define resolution for the parametric map
    nx = 20   # number of lateral bins
    ny = 20   # number of depth bins

    # Interpolated grid
    param_map = np.zeros((ny, nx))

    # Define frequency axis
    fft_len = 256
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1/params["SamplingRate"]))

    # Divide ROI into bins
    depth_idx = np.linspace(0, RF.shape[0]-1, ny+1, dtype=int)
    lateral_idx = np.linspace(0, RF.shape[1]-1, nx+1, dtype=int)

    for i in range(ny):
        for j in range(nx):
            # Get current bin
            d_start, d_end = depth_idx[i], depth_idx[i+1]
            l_start, l_end = lateral_idx[j], lateral_idx[j+1]

            patch = RF[d_start:d_end, l_start:l_end]
            mask_patch = kidney_mask[d_start:d_end, l_start:l_end]

            # Only analyze if mask has values here
            if np.sum(mask_patch) > 0:
                patch_vals = patch[mask_patch > 0].flatten()

                if patch_vals.size > 0:
                    # Zero-pad or trim to fft_len
                    padded = np.zeros(fft_len, dtype=np.complex64)
                    n = min(patch_vals.size, fft_len)
                    padded[:n] = patch_vals[:n]

                    spectrum = np.abs(np.fft.fftshift(np.fft.fft(padded)))

                    # Get frequency corresponding to max amplitude
                    max_freq = freqs[np.argmax(spectrum)]
                    param_map[i, j] = max_freq
                else:
                    param_map[i, j] = np.nan
            else:
                param_map[i, j] = np.nan

    # Plot parametric map
    plt.figure(figsize=(8, 6))
    plt.imshow(param_map, extent=[x[0], x[-1], y[-1], y[0]],
               aspect="auto", cmap="jet", origin="upper")
    plt.colorbar(label="Max Frequency [MHz]")
    plt.title(f"2D Parametric Frequency Map: {title_name}")
    plt.xlabel("Width [cm]")
    plt.ylabel("Depth [cm]")
    plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    root_dir = r"C:\Users\kjshah\Downloads\CLOUD_IMAGES_DATASET\MERGED"
    rf_image(patient_id=123, image_num=1, root_dir=root_dir)
