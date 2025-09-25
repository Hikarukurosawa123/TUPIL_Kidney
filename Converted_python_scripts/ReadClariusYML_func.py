import re

def ReadClariusYML(filepath, Nline):
    """
    Reads Clarius .yml metadata corresponding to .raw RF file.

    Parameters
    ----------
    filepath : str
        Path to the .raw file (expects matching .yml file)
    Nline : int
        Number of lines per frame

    Returns
    -------
    parameters : dict
        Ultrasound acquisition parameters
    """
    ymlpath = filepath.replace("raw", "yml")

    with open(ymlpath, "r") as f:
        yml = [line.strip() for line in f if line.strip()]

    def getvalue(key):
        line = next(s for s in yml if key in s)
        value = line.split(":")[1].strip().split(" ")[0]
        return float(value)

    def getRxElement(line):
        first_split = line.split(",")[0]
        second_split = first_split.split(":")[1]
        return int(second_split.strip())

    parameters = {}
    parameters["ProbeElements"]     = getvalue("elements: ")
    parameters["ProbePitch"]        = getvalue("pitch: ")
    parameters["NumFrames"]         = getvalue("frames: ")
    parameters["FrameRate"]         = getvalue("frame rate: ")
    parameters["TransmitFrequency"] = getvalue("transmit frequency: ")
    parameters["ImagingDepth"]      = getvalue("imaging depth: ")
    parameters["FocalDepth"]        = getvalue("focal depth: ")
    parameters["SamplingRate"]      = getvalue("sampling rate: ")
    parameters["DelaySamples"]      = getvalue("delay samples: ")

    # Find rx element lines
    rx_lines = [i for i, line in enumerate(yml) if "rx element:" in line]
    parameters["FirstRxElement"] = getRxElement(yml[rx_lines[0]])
    parameters["LastRxElement"]  = getRxElement(yml[rx_lines[0] + Nline - 1])
    parameters["Lineincrement"]  = getRxElement(yml[rx_lines[1]]) - getRxElement(yml[rx_lines[0]])

    return parameters
