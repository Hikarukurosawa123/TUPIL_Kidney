%% Displays one RF Line and FFT of the image

% Enter path, filename, and line number [0 - 192]
pathname = "/Users/vanessahoang/Downloads/planar_reflector/water/";
filename = "water_rf";
line = 96;

% Get frame number
fname = append(pathname,filename, ".raw");
fid = fopen(fname, 'r');
hinfo = fread(fid, 5, 'int32');
sliceNum = hinfo(2);
[data, header] = rdataread(fname, sliceNum);

% Dimensions of the RF data
[nframes, nsamples, nlines] = size(data);

% Depth
parameters = ReadClariusYML(fname,header.lines);
fs = parameters.SamplingRate;
depth_increment = parameters.ImagingDepth/nsamples;
depth = (0:nsamples-1) * depth_increment/10;

% Plot RF line
rf = squeeze(data(sliceNum,:,:));
rf = rf(:,line);
name = strrep(filename, '_', '\_'); figure;
plot(depth, rf, 'color', 'b'); subtitle(name + "\_nsample=" + line);
fig = gcf; fig.Position(3) = 900; fig.Position(4) = 400; 
title('RF Line'); xlabel('Depth [cm]'); ylabel('Amplitude');
xlim([0, max(depth)]); ylim([-1e4, 1e4]);

% Plot frequnecy response
figure;
[freq,Z] = sig_fft(rf, fs, 1);
plot(freq,Z, 'color', 'b'); xlim([0, fs/2]);
fig = gcf; fig.Position(3) = 700; fig.Position(4) = 500; 
subtitle(name + "\_nsample=" + line);
title('Power Spectra');
xlabel('Frequency [MHz]'); ylabel('Power [dB]');