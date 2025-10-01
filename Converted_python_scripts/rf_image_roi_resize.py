from skimage.transform import resize
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import loadmat
from .rdataread_func import rdataread
from .ReadClariusYML_func import ReadClariusYML
from skimage.transform import resize
import io
from PIL import Image
def rf_image_with_roi_resized_png_cwd(patient_id, image_num, root_dir, out_size=(224, 224), save_bmode = False, save_aspect_auto=True):
    """
    Load RF data, apply ROI mask, generate B-mode, resize, and save
    into ./Bmode folder in the current directory.

    Args:
        patient_id (int): Patient ID
        image_num (int): Image number
        root_dir (str): Root directory containing patient data
        out_size (tuple): Resize target (H, W)
        save_aspect_auto (bool): If True, also save a version with aspect='auto'
    """
    # Patient ID formatting
    patient_id_str = f"P{patient_id}"
    patient_id_mask = f"{patient_id_str}_"

    # Search for matching folder
    files = os.listdir(root_dir)
    desired_substring, text = None, None
    for fname in files:
        if patient_id_mask in fname:
            text = fname
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

    # Imaging parameters
    params = ReadClariusYML(fname, header["lines"])
    delay = (153850 / 2) * (params["DelaySamples"] / (params["SamplingRate"] * 1e6))
    depth = (153850 / 2) * (data.shape[1] / (params["SamplingRate"] * 1e6))
    arc_length = (params["ProbePitch"] / 1e3) * data.shape[2]
    FOV = (arc_length * 360) / (2 * np.pi * 45)
    width = 2 * (depth + delay) * np.tan(np.deg2rad(FOV / 2))

    # Display RF data image (last frame)
    RF = data[-1, :, :]
    BB = 20 * np.log10(1 + np.abs(hilbert(RF, axis=0)))

    # ROI mask
    kidney_mask = kidney_data["kidney_mask"]

    overlay_full = BB * kidney_mask

    # Resize to out_size (normal B-mode version)
    resized_img = resize(overlay_full, out_size, anti_aliasing=True)

    if save_bmode:
        save_dir = os.path.join(os.getcwd(), "Bmode")
        os.makedirs(save_dir, exist_ok=True)
        out_name = f"Patient_{patient_id}_Resized_Image_{image_num}.png"
        out_path = os.path.join(save_dir, out_name)
        plt.imsave(out_path, resized_img, cmap="gray", vmin=15, vmax=70)
        print(f"Saved resized Bmode to {out_path}")

    if save_aspect_auto:
        save_dir = os.path.join(os.getcwd(), "Bmode_auto_aspect")
        os.makedirs(save_dir, exist_ok=True)
        out_name_auto = f"Patient_{patient_id}_Resized_Image_{image_num}.png"
        out_path_auto = os.path.join(save_dir, out_name_auto)

        # Step 1: Render with aspect="auto" to an in-memory buffer
        fig, ax = plt.subplots()
        ax.imshow(overlay_full, cmap="gray", vmin=15, vmax=70, aspect="auto")
        ax.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Step 2: Open screenshot as PIL Image
        screenshot_img = Image.open(buf).convert("L")

        # Step 3: Resize screenshot to 224x224
        screenshot_resized = screenshot_img.resize(out_size, Image.Resampling.LANCZOS)
        screenshot_resized.save(out_path_auto)

        print(f"Saved resized 'screenshot' aspect='auto' Bmode to {out_path_auto}")