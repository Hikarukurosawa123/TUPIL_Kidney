import numpy as np

def rdataread(filename, frames):
    """
    Reads and returns Clarius RF/IQ/ENV data (.raw).
    
    Parameters
    ----------
    filename : str
        Path to .raw file
    frames : int
        Number of frames to read (if > total frames, reset to max)
    
    Returns
    -------
    data : np.ndarray
        Data array [frames, samples, lines]
    header : dict
        File header information
    ts : np.ndarray
        Frame timestamps (int64)
    """

    with open(filename, "rb") as f:
        # read header info (5 int32 values)
        hinfo = np.fromfile(f, dtype=np.int32, count=5)
        header = {
            "id": int(hinfo[0]),
            "frames": int(hinfo[1]),
            "lines": int(hinfo[2]),
            "samples": int(hinfo[3]),
            "sampleSize": int(hinfo[4])
        }

        print("hinfo", hinfo[0])

        if frames > header["frames"]:
            frames = header["frames"]

        ts = np.zeros(frames, dtype=np.int64)

        if header["id"] == 3:  # pw iq
            data = np.zeros((frames, header["samples"]*2, header["lines"]), dtype=np.int16)
            for fidx in range(frames):
                ts[fidx] = np.fromfile(f, dtype=np.int64, count=1)[0]
                oneline = np.fromfile(f, dtype=np.int16,
                                      count=header["samples"]*2*header["lines"])
                data[fidx, :, :] = oneline.reshape(header["samples"]*2, header["lines"])

        elif header["id"] == 0:  # iq
            data = np.zeros((frames, header["samples"]*2, header["lines"]), dtype=np.int16)
            for fidx in range(frames):
                ts[fidx] = np.fromfile(f, dtype=np.int64, count=1)[0]
                oneline = np.fromfile(f, dtype=np.int16,
                                      count=header["samples"]*2*header["lines"])
                data[fidx, :, :] = oneline.reshape(header["samples"]*2, header["lines"])

        elif header["id"] == 1:  # env
            data = np.zeros((frames, header["samples"], header["lines"]), dtype=np.uint8)
            for fidx in range(frames):
                ts[fidx] = np.fromfile(f, dtype=np.int64, count=1)[0]
                oneline = np.fromfile(f, dtype=np.uint8,
                                      count=header["samples"]*header["lines"])
                data[fidx, :, :] = oneline.reshape(header["samples"], header["lines"])

        elif header["id"] == 2:  # rf
            data = np.zeros((frames, header["samples"], header["lines"]), dtype=np.int16)
            for fidx in range(frames):
                ts[fidx] = np.fromfile(f, dtype=np.int64, count=1)[0]
                frame_data = np.fromfile(f, dtype=np.int16,
                                         count=header["samples"]*header["lines"])
                data[fidx, :, :] = frame_data.reshape(header["samples"], header["lines"], order = "F")

        else:
            raise ValueError(f"Unsupported header id: {header['id']}")

    return data, header
