%% Known-aberration phase diversity on a 1 um fluorobead (3P).
% Center and focus the bead with ScanImage before running this script.
% Match the physical optics/FOV to cfg.optics below; keep power, gain and zoom fixed.
% This script does not move the stage or change the optical settings.
% dryRun validates/generates SLM commands and metadata without hardware.

cfg.outputRoot = 'C:\Users\ScanImage\Documents\Diego_Max\3d_phase_diversity_fluorobead_10_15';
cfg.dryRun = false;
cfg.matFileVersion = '-v7';               % Python/scipy readable; use '-v7.3' above 2 GB
cfg.systemAOFigure = '';                  % e.g. 'C:\data\sysAO_slm.fig'; empty disables
cfg.correctionFile = '';                  % loop 1: empty; later: fetched results/correction.mat
cfg.zReferenceReadoutUm = [];             % empty: enter focused-bead Z readout at first run
cfg.zPlanesUm = linspace(-2, 2, 4);         % offsets from bead focus (Z=0), in um
% Establish the bead-focus reference manually before running. Each prompt
% requests an offset from that reference, NOT from the previous plane.
% Positive/negative offsets follow your microscope Z readout convention.
cfg.imageSize = [512 512];                 % [rows columns]
cfg.framesPerImage = 5;                    % frames averaged per saved image
cfg.averageInSingleGrab = true;            % one shutter cycle per phase condition
cfg.pauseAfterSLM = 0.2;
cfg.grabTimeoutSec = 60;
cfg.channelIndex = 1;                      % index in hDisplay.lastFrame
cfg.randomSeed = 15;
cfg.imposedRMSWaves = 0.15;                 % random mode only: sampled pupil RMS
cfg.diversityWaves = 0.10;                  % normalized Zernike coefficient
cfg.centerXY = [411 457];
cfg.radiusPixels = min(376, 431)/2;
cfg.dryRunSLMSize = [1024 1024];            % [height width], hardware queries actual size
cfg.backgroundGratingPeriodPixels = 8;
cfg.lut1024 = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6748_at1300_1stOrder_031325_3.lut';
cfg.lut512 = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\512x512_linearVoltage.LUT';
% Python convention: [m n], normalized real Zernikes, coefficients in waves.
cfg.modes = [-2 2; 2 2; -3 3; -1 3; 1 3; 3 3; ...
    -4 4; -2 4; 0 4; 2 4; 4 4; -5 5; -3 5; -1 5; 1 5; 3 5; 5 5];
cfg.diversityModes = [2 2; -2 2; 0 4];     % +/- each, plus shared zero
cfg.baselineWaves = zeros(size(cfg.modes, 1), 1); % optional calibrated system correction
cfg.aberrationSource = 'manual';           % 'random' or 'manual'
% Manual table: each row is [m n coefficient_in_waves], using cfg.modes.
% Unlisted modes receive zero. Coefficients are applied exactly, no rescaling.
% Set aberrationSource = 'manual' to use this example or your own table.
cfg.manualAberration = [
     2 2  0.10;                           % cosine astigmatism
     0 4  0.08;                           % spherical aberration
];
% Laptop comparison and cluster settings travel with the acquisition metadata.
% Change these here; the notebook and submission command read them automatically.

cfg.optics = struct('excitationWavelengthNm',1300, 'nonlinearOrder',3, ...
    'NA',1.05, 'refractiveIndex',1.333, 'focalLengthMm',7.2, ...
    'magnification',4, 'beamWaistMm',3.5, 'backFocalPlaneWidthMm',15.12, ...
    'fieldOfViewUm',6.5, 'beadDiameterUm',1, 'focalPositionUm',0);
cfg.analysis = struct('size',128, 'pupilSize',64, 'background',0, ...
    'zSign',1, 'beadCenterUm',[], 'beadPlanes',21, 'useTex',true);
% Empty beadCenterUm estimates lateral position from a reference image; Z=0.
cfg.solver = struct('rho',1e46, 'samplePlanes',21, 'sampleExtentUm',4, ...
    'maxIterations',50, 'randomStarts',100);
% Phase handedness must be calibrated against the microscope relay.
% atan2(row-centerY, column-centerX) matches the starter's SLM coordinates.

if cfg.dryRun
    beadPDRun = acquireBeadPD(cfg, []);
else
    assert(exist('hSI', 'var') == 1, 'ScanImage must expose hSI in the workspace.');
    beadPDRun = acquireBeadPD(cfg, hSI);
end
fprintf('Run saved in %s\n', beadPDRun);

function folder = acquireBeadPD(cfg, hSI)
    validatestring(cfg.matFileVersion, {'-v7','-v7.3'});
    aberrationSource = validatestring(cfg.aberrationSource, {'random','manual'});
    validateattributes(cfg.zPlanesUm, {'numeric'}, {'vector','nonempty','real','finite'});
    validateattributes(cfg.framesPerImage, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(cfg.imposedRMSWaves, {'numeric'}, {'scalar','finite','nonnegative'});
    validateattributes(cfg.diversityWaves, {'numeric'}, {'scalar','finite','positive'});
    validateattributes(cfg.imageSize, {'numeric'}, {'size',[1 2],'integer','positive'});
    validateattributes(cfg.pauseAfterSLM, {'numeric'}, {'scalar','finite','nonnegative'});
    validateattributes(cfg.grabTimeoutSec, {'numeric'}, {'scalar','finite','positive'});
    validateattributes(cfg.channelIndex, {'numeric'}, {'scalar','integer','positive'});
    if ~cfg.dryRun && cfg.averageInSingleGrab && cfg.framesPerImage > 1
        assert(isprop(hSI.hDisplay,'lastAveragedFrame'), ...
            ['This ScanImage version has no lastAveragedFrame property. ' ...
             'Set cfg.averageInSingleGrab=false to use the original acquisition method.']);
    end
    slm = [];
    if cfg.dryRun
        height = cfg.dryRunSLMSize(1); width = cfg.dryRunSLMSize(2);
    else
        assert(strcmpi(hSI.acqState, 'idle'), 'Stop ScanImage acquisition first.');
        slm = SLMController('LUT1024',cfg.lut1024, 'LUT512',cfg.lut512);
        slmGuard = onCleanup(@() delete(slm)); %#ok<NASGU>
        slm.load();
        width = slm.Width; height = slm.Height;
    end
    [Z, pupil] = normalizedBasis(width, height, cfg.modes, cfg.centerXY, cfg.radiusPixels);
    systemAOPhaseSLM = loadSystemAO(cfg.systemAOFigure,height,width,pupil);
    systemAOWaves = systemAOPhaseSLM(:)/256; % raw SLM units: 256 = one full phase cycle
    baseline = cfg.baselineWaves(:);
    assert(numel(baseline) == size(Z,2) && all(isfinite(baseline)), 'Invalid baseline coefficients.');
    if strcmp(aberrationSource,'random')
        stream = RandStream('mt19937ar','Seed',cfg.randomSeed);
        raw = 2*rand(stream,size(Z,2),1)-1;
        imposed = raw * cfg.imposedRMSWaves / pupilRMS(Z*raw,pupil);
    else
        validateattributes(cfg.manualAberration, {'numeric'}, ...
            {'2d','ncols',3,'real','finite'});
        manualModes = cfg.manualAberration(:,1:2);
        assert(size(unique(manualModes,'rows'),1) == size(manualModes,1), ...
            'Manual aberration contains duplicate modes; use one coefficient per mode.');
        [foundManual, manualIndex] = ismember(manualModes,cfg.modes,'rows');
        assert(all(foundManual), 'Every manual aberration mode must be present in cfg.modes.');
        imposed = zeros(size(Z,2),1);
        imposed(manualIndex) = cfg.manualAberration(:,3);
    end
    cumulativeCorrection = zeros(size(imposed));
    loopNumber = 1; parentRunId = '';
    if ~isempty(cfg.correctionFile)
        previous = load(cfg.correctionFile);
        assert(previous.optimizerSuccess, 'The previous solver did not converge; inspect its result before continuing.');
        assert(strcmp(previous.coefficientUnits,'waves'), 'Correction must be in waves.');
        [foundCorrection, correctionIndex] = ismember(cfg.modes,previous.modes_mn,'rows');
        assert(all(foundCorrection) && size(previous.modes_mn,1) == size(cfg.modes,1), ...
            'Correction mode basis differs from this experiment.');
        assert(max(abs(imposed-previous.imposedWaves(correctionIndex))) < 1e-12 && ...
            max(abs(baseline-previous.baselineWaves(correctionIndex))) < 1e-12, ...
            'Imposed aberration/baseline changed. Clear correctionFile to begin a new experiment.');
        assert(isequaln(cfg.optics,jsondecode(previous.optics_json)), ...
            'Optics changed. Clear correctionFile to begin a new experiment.');
        previousSystemAO = zeros(height,width);
        if isfield(previous,'systemAOPhaseSLM') && ~isempty(previous.systemAOPhaseSLM)
            previousSystemAO = previous.systemAOPhaseSLM;
        end
        assert(isequal(systemAOPhaseSLM,previousSystemAO), ...
            'System AO changed. Clear correctionFile to begin a new experiment.');
        cumulativeCorrection = previous.cumulativeCorrectionWaves(correctionIndex);
        validateattributes(cumulativeCorrection,{'numeric'},{'vector','real','finite'});
        cumulativeCorrection = cumulativeCorrection(:);
        loopNumber = previous.parentLoopNumber + 1;
        parentRunId = previous.parentRunId;
        if isempty(cfg.zReferenceReadoutUm)
            cfg.zReferenceReadoutUm = previous.zReferenceReadoutUm;
        elseif ~isempty(previous.zReferenceReadoutUm)
            assert(abs(cfg.zReferenceReadoutUm-previous.zReferenceReadoutUm) < 1e-12, ...
                'Focus reference changed. Clear correctionFile to begin a new experiment.');
        end
    end
    if isempty(cfg.zReferenceReadoutUm) && ~cfg.dryRun
        cfg.zReferenceReadoutUm = str2double(input( ...
            'At bead focus, enter the microscope Z readout in um (keep this reference for all loops): ', 's'));
    end
    if ~isempty(cfg.zReferenceReadoutUm)
        validateattributes(cfg.zReferenceReadoutUm,{'numeric'},{'scalar','real','finite'});
    end
    fprintf('Acquiring correction loop %d.\n',loopNumber);
    [found, modeIndex] = ismember(cfg.diversityModes, cfg.modes, 'rows');
    assert(all(found), 'Every diversity mode must be in cfg.modes.');
    % Every image includes fixed system AO. References add only the baseline;
    % PD images also add imposed aberration + cumulative correction + diversity.
    names = {'reference_before', 'aberrated_zero'};
    diversities = zeros(2,size(Z,2));
    imposedEnabled = [false true];
    for k = 1:numel(modeIndex)
        for signValue = [1 -1]
            row = zeros(1,size(Z,2));
            row(modeIndex(k)) = signValue*cfg.diversityWaves;
            diversities(end+1,:) = row; %#ok<AGROW>
            names{end+1} = sprintf('m%+d_n%d_%+g',cfg.diversityModes(k,:),signValue*cfg.diversityWaves); %#ok<AGROW>
            imposedEnabled(end+1) = true; %#ok<AGROW>
        end
    end
    names{end+1} = 'aberrated_zero_after';
    diversities(end+1,:) = 0; imposedEnabled(end+1) = true;
    names{end+1} = 'reference_after';
    diversities(end+1,:) = 0; imposedEnabled(end+1) = false;
    % Flatten plane-major acquisition order so all per-image metadata align
    % with images(:,:,k), including for a single requested plane.
    imagesPerPlane = numel(names);
    zPlanesUm = cfg.zPlanesUm(:);
    planeIndex = repelem((1:numel(zPlanesUm)).', imagesPerPlane);
    zPositionUm = zPlanesUm(planeIndex);
    names = repmat(names, 1, numel(zPlanesUm));
    diversities = repmat(diversities, numel(zPlanesUm), 1);
    imposedEnabled = repmat(imposedEnabled, 1, numel(zPlanesUm));
    total = diversities + baseline.' + imposedEnabled(:)*(imposed+cumulativeCorrection).';
    % Use the same SDK grating and byte-to-phase mapping as grab_data.m.
    % Generate once and reuse for every phase command in this run.
    backgroundPhase8 = uint8([]);
    backgroundField = ones(width*height,1);
    if ~cfg.dryRun
        [backgroundField,backgroundPhase8] = legacyBackgroundGrating( ...
            slm,pupil,cfg.backgroundGratingPeriodPixels);
    end
    baselinePhase = Z*baseline;
    baselineField = commandField(baselinePhase,pupil,backgroundField,systemAOWaves);
    restingField = commandField(Z*(baseline+imposed+cumulativeCorrection),pupil,backgroundField,systemAOWaves);
    if ~exist(cfg.outputRoot,'dir')
        [ok,msg] = mkdir(cfg.outputRoot); assert(ok, '%s', msg);
    end
    folder = tempname(cfg.outputRoot); % separate runs, never overwrite images
    [ok,msg] = mkdir(folder); assert(ok, '%s', msg);
    [~,runId] = fileparts(folder);
    metadata = struct('runId',runId, 'loopNumber',loopNumber, 'parentRunId',parentRunId, ...
        'cumulativeCorrectionWaves',cumulativeCorrection, 'config',cfg, 'modes_mn',cfg.modes, ...
        'coefficientUnits','waves', 'phaseUnits','radians = 2*pi*waves', ...
        'normalization','sqrt(n+1) for m=0; sqrt(2*(n+1)) otherwise', ...
        'backgroundGratingMethod','generateGratings(period,3,false,WFC)', ...
        'backgroundGratingGenerated',~cfg.dryRun, 'backgroundPhase8',backgroundPhase8, ...
        'slmSize',[height width], 'pupil',reshape(pupil,height,width), ...
        'systemAOPhaseSLM',systemAOPhaseSLM, 'systemAOUnits','raw SLM phase units; 256 units = 2*pi radians', ...
        'systemAOTreatment','Fixed calibration on all images; fit residual native + system AO + imposed + cumulative phase.', ...
        'baselineWaves',baseline, 'imposedWaves',imposed, ...
        'imposedPupilRMSWaves',pupilRMS(Z*imposed,pupil), ...
        'diversityWaves',diversities, 'totalCommandWaves',total, ...
        'names',{names}, 'isPDImage',logical(imposedEnabled), ...
        'zPlanesUm',zPlanesUm, 'planeIndex',planeIndex, ...
        'zPositionUm',zPositionUm, 'focalPositionsMm',zPositionUm/1000, ...
        'zReference','Manual bead-focus reference; offsets follow microscope Z readout.', ...
        'zPositionSource','Requested offsets, manually confirmed; not measured.', ...
        'nativeSystemAberration','Unknown; estimates include residual system aberration.', ...
        'status','in_progress', 'startedAt',datestr(now,30));
    save(fullfile(folder,'metadata.mat'),'metadata');
    images = nan([cfg.imageSize numel(names)]);
    log = struct([]);
    if ~cfg.dryRun
        oldFrames = hSI.hStackManager.framesPerSlice;
        oldAverage = hSI.hDisplay.displayRollingAverageFactor;
        acquisitionGuard = onCleanup(@() finishAcquisition(hSI,slm,restingField,oldFrames,oldAverage)); %#ok<NASGU>
        configureGrab(hSI,cfg);
    end
    try
        for plane = 1:numel(zPlanesUm)
            if ~cfg.dryRun
                slm.writeEfield(baselineField);
                pause(cfg.pauseAfterSLM);
                fprintf('\nPlane %d/%d: Z = %+.6g um relative to bead focus (Z=0).\n', ...
                    plane, numel(zPlanesUm), zPlanesUm(plane));
                if ~isempty(cfg.zReferenceReadoutUm)
                    fprintf('Target microscope Z readout: %+.6g um (reference %+.6g um).\n', ...
                        cfg.zReferenceReadoutUm+zPlanesUm(plane),cfg.zReferenceReadoutUm);
                end
                reply = input(['Move manually to this Z offset, stop ScanImage acquisition, ' ...
                    'then press Enter to acquire (q to cancel): '], 's');
                if strcmpi(strtrim(reply),'q')
                    metadata.status = 'cancelled';
                    save(fullfile(folder,'metadata.mat'),'metadata');
                    save(fullfile(folder,'acquisition.mat'),'images','log',cfg.matFileVersion);
                    clear acquisitionGuard;
                    return;
                end
                assert(strcmpi(hSI.acqState,'idle'), 'Stop ScanImage before confirming the Z plane.');
                configureGrab(hSI,cfg);
            end
            for sequenceIndex = 1:imagesPerPlane
                k = (plane-1)*imagesPerPlane + sequenceIndex;
                phaseWaves = Z*total(k,:).';
                field = commandField(phaseWaves,pupil,backgroundField,systemAOWaves);
                log(k).name = names{k};
                log(k).acquired = false;
                log(k).planeIndex = plane;
                log(k).zPositionUm = zPlanesUm(plane);
                log(k).zManuallyConfirmed = ~cfg.dryRun;
                log(k).filename = sprintf('Plane_%02d_Z_%+.6gum_Img_%02d_%s.tif', ...
                    plane,zPlanesUm(plane),sequenceIndex,names{k});
                log(k).commandPupilRMSWaves = pupilRMS(phaseWaves,pupil);
                if ~cfg.dryRun
                    slm.writeEfield(field);
                    pause(cfg.pauseAfterSLM);
                    if cfg.averageInSingleGrab
                        image = grabStrict(hSI,cfg);
                    else
                        image = zeros(cfg.imageSize);
                        for frame = 1:cfg.framesPerImage
                            image = image + grabStrict(hSI,cfg)/cfg.framesPerImage;
                        end
                    end
                    images(:,:,k) = image; % retain signed floating data for inference
                    saveTiff16(image,fullfile(folder,log(k).filename));
                    log(k).acquired = true;
                    log(k).minValue = min(image(:));
                    log(k).maxValue = max(image(:));
                    log(k).tiffClippedPixels = nnz(image < 0 | image > 65535);
                end
                log(k).finishedAt = datestr(now,30);
                save(fullfile(folder,'acquisition.mat'),'images','log',cfg.matFileVersion);
                fprintf('%d/%d: Z=%+.6g um, %s\n',k,numel(names),zPlanesUm(plane),names{k});
            end
        end
        if cfg.dryRun, metadata.status = 'dry_run'; else, metadata.status = 'complete'; end
    catch ME
        metadata.status = 'failed'; metadata.error = ME.message;
        save(fullfile(folder,'metadata.mat'),'metadata');
        save(fullfile(folder,'acquisition.mat'),'images','log',cfg.matFileVersion);
        if ~cfg.dryRun, clear acquisitionGuard; end
        rethrow(ME);
    end
    save(fullfile(folder,'metadata.mat'),'metadata');
    if ~cfg.dryRun, clear acquisitionGuard; end
end

function configureGrab(hSI,cfg)
    if cfg.averageInSingleGrab
        count = cfg.framesPerImage;
    else
        count = 1;
    end
    hSI.hStackManager.framesPerSlice = count;
    hSI.hDisplay.displayRollingAverageFactor = count;
end

function image = grabStrict(hSI,cfg)
    % lastFrame is a read-only dependent property in ScanImage.
    % Wait for the grab to finish, then read it after display callbacks settle.
    assert(strcmpi(hSI.acqState,'idle'), 'ScanImage must be idle before starting a grab.');
    timer = tic;
    hSI.startGrab();
    pause(0.05); % Allow acquisition/display callbacks to run before polling.
    while ~strcmpi(hSI.acqState,'idle')
        if toc(timer) > cfg.grabTimeoutSec
            error('BeadPD:Timeout','ScanImage grab exceeded %g seconds.',cfg.grabTimeoutSec);
        end
        pause(0.05);
    end
    pause(0.1);
    if cfg.averageInSingleGrab && cfg.framesPerImage > 1
        frames = hSI.hDisplay.lastAveragedFrame; % Read averaged data, not the last raw frame.
    else
        frames = hSI.hDisplay.lastFrame;
    end
    assert(iscell(frames) && numel(frames) >= cfg.channelIndex && ~isempty(frames{cfg.channelIndex}), ...
        'ScanImage returned no image for the selected channel after the grab.');
    image = double(frames{cfg.channelIndex});
    assert(isequal(size(image),cfg.imageSize) && all(isfinite(image(:))), ...
        'Unexpected image shape or nonfinite data; check ScanImage ROI/channel settings.');
end

function finishAcquisition(hSI,slm,field,oldFrames,oldAverage)
    try
        if ~strcmpi(hSI.acqState,'idle'), hSI.abort(); end
        hSI.hStackManager.framesPerSlice = oldFrames;
        hSI.hDisplay.displayRollingAverageFactor = oldAverage;
    catch ME
        warning('BeadPD:ScanImageCleanup','Could not restore ScanImage: %s',ME.message);
    end
    try
        slm.writeEfield(field);
    catch ME
        warning('BeadPD:SLMCleanup','Could not restore the current corrected test phase: %s',ME.message);
    end
end

function field = commandField(phase,pupil,backgroundField,systemAOWaves)
    phase = phase + systemAOWaves; % Add in floating point; wrap only in the SLM writer.
    phase(~pupil) = 0;
    field = exp(2i*pi*phase) .* backgroundField;
end

function phaseSLM = loadSystemAO(path,height,width,pupil)
    phaseSLM = zeros(height,width);
    if isempty(path), return; end
    assert(isfile(path), 'System AO figure not found: %s',path);
    fig = openfig(path,'invisible');
    figureGuard = onCleanup(@() close(fig)); %#ok<NASGU>
    objects = [num2cell(findobj(fig,'Type','image')); num2cell(findobj(fig,'Type','surface'))];
    candidates = {};
    for k = 1:numel(objects)
        values = get(objects{k},'CData');
        if isnumeric(values) && ismatrix(values) && isequal(size(values),[height width])
            candidates{end+1} = values; %#ok<AGROW>
        end
    end
    assert(numel(candidates) == 1, ...
        'System AO figure must contain exactly one numeric %d-by-%d phase image/surface (rows,columns).',height,width);
    values = double(candidates{1});
    validateattributes(values,{'numeric'},{'real','finite','>=',0,'<=',255});
    % Read actual CData, not the colormap, color limits or rendered screenshot.
    % Preserve pixel orientation; do not resize, flip, transpose or normalize.
    phaseSLM(pupil) = values(pupil);
    fprintf('Loaded system AO figure: %s (applied inside pupil on every image).\n',path);
end

function [field,phase8] = legacyBackgroundGrating(slm,pupil,period)
    assert(exist('generateGratings','file') ~= 0, ...
        'Add the microscope generateGratings helper used by grab_data.m to the MATLAB path.');
    depth = calllib('Blink_C_wrapper','Get_image_depth',slm.BoardNumber);
    Bytes = depth/8;
    WFC = libpointer('uint8Ptr',zeros(slm.Width*slm.Height*Bytes,1));
    grating = generateGratings(period,3,false,WFC);
    bgVec = grating(1).value(:);
    pupilVec = pupil(:);
    assert(numel(bgVec) == numel(pupilVec), ...
        'generateGratings returned an unexpected image size.');
    phase8 = zeros(slm.Width*slm.Height,1,'uint8');
    phase8(~pupilVec) = bgVec(~pupilVec);
    field = exp(1j*double(phase8)*pi/128);
end

function value = pupilRMS(phase,pupil)
    values = phase(pupil);
    value = sqrt(mean((values-mean(values)).^2));
end

function [Z,pupil] = normalizedBasis(width,height,modes,center,radius)
    validateattributes(radius,{'numeric'},{'scalar','finite','positive'});
    assert(center(1)-radius >= 1 && center(1)+radius <= width && ...
        center(2)-radius >= 1 && center(2)+radius <= height, 'Pupil extends beyond the SLM.');
    [X,Y] = meshgrid(1:width,1:height);
    rho = hypot(X-center(1),Y-center(2))/radius;
    theta = atan2(Y-center(2),X-center(1));
    pupil = rho(:) <= 1;
    r = rho(pupil); phi = theta(pupil);
    Z = zeros(width*height,size(modes,1));
    for k = 1:size(modes,1)
        m = modes(k,1); n = modes(k,2); a = abs(m);
        assert(n >= 0 && n == fix(n) && m == fix(m) && a <= n && mod(n-a,2) == 0,'Invalid Zernike mode.');
        radial = zeros(size(r));
        for s = 0:(n-a)/2
            c = (-1)^s*factorial(n-s)/(factorial(s)*factorial((n+a)/2-s)*factorial((n-a)/2-s));
            radial = radial + c*r.^(n-2*s);
        end
        if m == 0
            values = sqrt(n+1)*radial;
        elseif m > 0
            values = sqrt(2*(n+1))*radial.*cos(a*phi);
        else
            values = sqrt(2*(n+1))*radial.*sin(a*phi);
        end
        Z(pupil,k) = values;
    end
end
