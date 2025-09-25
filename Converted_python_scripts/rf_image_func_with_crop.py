import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import loadmat
from skimage.transform import resize   # <-- for resizing
from .rdataread_func import rdataread
from .ReadClariusYML_func import ReadClariusYML


def rf_image(patient_id, image_num, root_dir, save_crops=True, crop_margin=0, target_size=144):
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

    # Display RF data image (last frame)
    RF = data[-1, :, :]
    BB = 20 * np.log10(1 + np.abs(hilbert(RF, axis=0)))
    title_name = filename.replace("_", "\\_")

    # ROI mask
    kidney_mask = kidney_data["kidney_mask"]

    # --- Cropping around ROI ---
    ys, xs = np.where(kidney_mask == 1)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    # Add margins to x only
    x1 = max(0, x1 - crop_margin)
    x2 = min(BB.shape[1], x2 + crop_margin)

    # Apply fixed y-buffer (28 top, 29 bottom)
    y1 = max(0, y1 - 0)
    y2 = min(BB.shape[0], y2 + 0)

    # Cropped images
    BB_crop = BB[y1:y2, x1:x2]
    mask_crop = kidney_mask[y1:y2, x1:x2]
    overlay_crop = BB_crop * mask_crop

    # --- Resize to target_size x target_size ---
    overlay_resized = resize(
        overlay_crop, 
        (target_size, target_size), 
        order=1,        # bilinear interpolation
        preserve_range=True, 
        anti_aliasing=True
    ).astype(np.float32)

    # Save crops if requested
    if save_crops:
        save_dir = os.path.join(pathname, "Cropped")
        os.makedirs(save_dir, exist_ok=True)

        plt.imsave(
            os.path.join(save_dir, f"{filename}_overlay_crop_resized.png"),
            overlay_resized, cmap="gray", vmin=15, vmax=70
        )

        print(f"Resized crops saved to {save_dir}")

    # Optionally show one of them
    plt.figure()
    plt.imshow(overlay_resized, cmap="gray", vmin=15, vmax=70, aspect="auto")
    plt.title(f"Cropped + Resized Kidney ROI: {title_name}")
    plt.show()
