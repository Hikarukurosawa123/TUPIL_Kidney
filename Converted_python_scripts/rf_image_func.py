import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import loadmat
from .rdataread_func import rdataread
from .ReadClariusYML_func import ReadClariusYML

from skimage.transform import resize

def rf_image(patient_id, image_num, root_dir, save_full=False, save_resized=False):
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
    title_name = filename.replace("_", "\\_")

    # ROI mask
    kidney_mask = kidney_data["kidney_mask"]

    # Overlay full image
    overlay_full = BB * kidney_mask

    # Save full image if requested
    if save_full:
        save_dir = os.path.join(pathname, "Full")
        os.makedirs(save_dir, exist_ok=True)
        plt.imsave(
            os.path.join(save_dir, f"{filename}_overlay_full.png"),
            overlay_full, cmap="gray", vmin=15, vmax=70
        )
        print(f"Full overlay image saved to {save_dir}")

    # Save resized 224x224 image if requested
    if save_resized:
        save_dir = os.path.join(pathname, "Resized")
        os.makedirs(save_dir, exist_ok=True)

        # Resize with skimage (preserve intensity scaling)
        resized_img = resize(overlay_full, (224, 224), anti_aliasing=True)

        plt.imsave(
            os.path.join(save_dir, f"{filename}_overlay_resized.png"),
            resized_img, cmap="gray", vmin=15, vmax=70
        )
        print(f"Resized 224x224 image saved to {save_dir}")

    # Display
    # Display
    plt.figure()

    # Compute coordinate values for each pixel
    depth_axis = np.linspace(0, depth, overlay_full.shape[0])
    width_axis = np.linspace(0, width, overlay_full.shape[1])

    extent = [width_axis[0], width_axis[-1], depth_axis[-1], depth_axis[0]]

    plt.imshow(
        overlay_full,
        cmap="gray",
        vmin=15,
        vmax=70,
        aspect="auto",
        extent=extent
    )

    plt.xlabel("Width (mm)")
    plt.ylabel("Depth (mm)")
    plt.title(f"Full Kidney ROI Overlay: {title_name}")
    plt.show()

