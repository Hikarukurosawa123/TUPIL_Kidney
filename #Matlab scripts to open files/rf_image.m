%% Displays RF data image from _rf.raw file
%run the function in command window; run at the directory where all the
%patient data is stored (e.g. //tupil/tupil data/#POCUS TRIAL DATA/#POCUS
%DATA - Shared with TMU)
function [] = rf_image(patient_id, image_num)
    
    %finding file 
    %currentFolder = pwd; %get current directory
    currentFolder = '/Users/hikaru/Desktop/TUPIL/Code/TUPIL_Kidney/data/Raw_data';

    S = dir(fullfile(currentFolder));
    
    Patient_id_mask = 'P' + string(patient_id) + '_';
    Patient_id = 'P' + string(patient_id);

    
    %get the beginning file names
    for i = 1:length(S)
        FileNames=S(i).name;
        if contains(FileNames,Patient_id_mask) 
            % Define text
            text = FileNames;
            % Define regular expression pattern to match the desired substring
            pattern = '(' + Patient_id + '_[A-Z0-9]+_01)';
            % Use regular expression to find matches
            matches = regexp(text, pattern, 'match');
            % Extract the desired substring
            desiredSubstring = matches{1};   
        end 
    end 
        
    pathname = fullfile(currentFolder) + "/" + text + "/" + Patient_id + "_Image_" + string(image_num) + "/";
    filename = string(desiredSubstring) + "_Image_" + string(image_num) + '_rf';
    
    %end find file 

    %load ROI coordinates (only kidney)
    ROI_filename = pathname + "ROIs";
    ROI_dir = dir(ROI_filename);
    ROI_dir = setdiff({ROI_dir.name},{'.','..'});

    for i = 1:length(ROI_dir)
        if contains(ROI_dir{i}, 'raw_kidney')
            ROI_pathname = ROI_filename + "/" + ROI_dir{i};
            kidney_data = load(ROI_pathname);
        end 
    end

    % Open folder
    fname = append(pathname,filename, ".raw");
    fid = fopen(fname, 'r');
    hinfo = fread(fid, 5, 'int32');
    numFrames = hinfo(2);
    
    [data, header] = rdataread(fname, numFrames);
    [nframes, nsamples, nlines] = size(data);
    
    if (numFrames > header.frames)
        numFrames = header.frames;
    end
    
    
    % depth
    parameters = ReadClariusYML(fname, header.lines);
    delay = (153850/2)*(parameters.DelaySamples/(parameters.SamplingRate*1e6));
    depth = (153850/2)*(nsamples/(parameters.SamplingRate*1e6));
    
    % width
    arc_length = (parameters.ProbePitch/1e3)*nlines;
    FOV = (arc_length * 360) / (2 * pi * 45);
    width = 2 * (depth + delay) * tand(FOV/2);
    
    % display rf data image
    figure; 
    colormap(gray)
    for j = numFrames:numFrames % numFrames is the total number of frames but ROI is only available for frame 'numframes'; by iterating through data(j,:,:), other frames can be accessed 
        RF = squeeze(data(j,:,:));
        RF(1,:)
        %length(RF(1,:))
        BB = 20.*log10(1 + abs(hilbert(RF)));
        titlename = strrep(filename, '_', '/_');
    
        y = linspace(delay, depth + delay, 10);
        x = linspace(0, width, 10);
    
        imagesc(x, y, BB, [15 70]); title('RF Image', titlename);
        ylabel('Depth [cm]'); xlabel('Width [cm]');
        
        
        % Adjust size
        % fig = gcf;  
        % fig.Position(3) = 600; 
        % fig.Position(4) = 600; 
        drawnow;

        if j == numFrames
            %plot masked kidney data 
            figure; 
            imagesc(x,y,kidney_data.kidney_mask); title('RF Image mask', titlename);
            ylabel('Depth [cm]'); xlabel('Width [cm]');
            % fig2 = gcf;  
            % fig2.Position(3) = 600; 
            % fig2.Position(4) = 600; 
            drawnow;
        end 
    end
    figure; imagesc(x, y, BB.*kidney_data.kidney_mask, [15 70]); title('RF Image Kidney Mask', titlename); drawnow;
end 