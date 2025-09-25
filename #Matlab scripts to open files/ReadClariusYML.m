function parameters = ReadClariusYML(filepath, Nline)
%{
YAML File Reader for Clarius .raw RF Data
By David Alberico
May 5th, 2022

This function takes a .raw RF file for Clarius US data and reads
parameters from the corresponding YAML file into a structure. 'filepath' is 
a string corresponding to the path to the RF .raw file to be read.

Compatible with MATLAB 2011b and MATLAB 2016b

NOTE: the .raw and corresponding .yml files must have the same filename
NOTE: With the Clarius update in November 2022, the arrangement of entries
in the .yml files saved using the caster changed, requriing an update to 
this code in order to correctly read the settings from the .yml. 
This is the "new" version of ReadClariusYML, compatible with .yml files
created after the update.

06 FEB 2023
Updated. Now the function automatically detects whether to read the file 
using the 'old' indicies or the 'new' indicies. Intend to update again so
that the function looks at the version number in future; this is a 'stop
gap' solution.

Last Modified: Feb 2023
%}

% modify path string to point to corresponding .yml file

ymlpath = strrep(filepath, 'raw', 'yml');
%ymlpath = append(filepath(1),filepath(2),".yml");

% open file and read contents into a cell array
fid = fopen(ymlpath, 'r');
yml = textscan(fid, '%s', 'delimiter', '\n', 'whitespace', '');
fclose(fid);

% remove empty lines
yml = deblank(yml{1});
yml(cellfun('isempty', yml)) = [];

% initialize structure to hold parameters
parameters = [];

% Call 'getvalue' to extract parameters and store them
    parameters.ProbeElements = getvalue("elements: ", yml);       % # of probe elements
    parameters.ProbePitch= getvalue("pitch: ", yml);           % probe pitch
    parameters.NumFrames = getvalue("frames: ", yml);           % # of frames
    parameters.FrameRate = getvalue("frame rate: ", yml);           % framerate [Hz]
    parameters.TransmitFrequency = getvalue("transmit frequency: ", yml);  % trasmit frequency [MHz]
    parameters.ImagingDepth = getvalue("imaging depth: ", yml);       % imaging depth [mm]
    parameters.FocalDepth = getvalue("focal depth: ", yml);         % focal depth [mm]
    parameters.SamplingRate = getvalue("sampling rate: ", yml);       % sampling rate [MHz]
    parameters.DelaySamples = getvalue("delay samples: ", yml);       % # of delay samples
    
    rxLine = find(contains(yml,"rx element: "));
    parameters.FirstRxElement = getRxElement(yml{rxLine(1)}); % index of first rx element
    parameters.LastRxElement = getRxElement(yml{rxLine(1)+Nline-1}); % index of last rx element
    parameters.Lineincrement = getRxElement(yml{rxLine(2)})-getRxElement(yml{rxLine(1)});

% to add additional numeric parameters, simply check which line of the cell 
% array 'yml' the parameter is stored on.
end

function value = getvalue(thisLine, yml)
line = yml{contains(yml,thisLine)};
% split the string into a key-value pair and get the value
splitString = regexp(line,':','split');
value = strtrim(splitString{2});
% extract numeric portion of value
value = regexp(value,' ','split');
value = strtrim(value{1});
value = str2double(value);
end

function rx = getRxElement(thisLine)
% split the string twice to extract the value
splitstr1 = regexp(thisLine,',','split');
split = splitstr1{1};
splitstr2 = regexp(split,':','split');
rx = strtrim(splitstr2{2});
rx = str2num(rx);
end
