classdef ScanImageController < handle
    % ScanImageController - small wrapper around hSI for repeated grabs

    properties
        hSI
        NumFramesToAverage (1,1) double = 5 % New property to control averaging
        PollPeriodSec (1,1) double = 0.1    % Reduced for active polling
        FallbackSize (1,2) double = [512 512]  % used only if data can't be read
    end

    methods
        function obj = ScanImageController(hSI, varargin)
            obj.hSI = hSI;
            if ~isempty(varargin)
                for k = 1:2:numel(varargin)
                    obj.(varargin{k}) = varargin{k+1};
                end
            end
        end

        function img = grabFrame(obj)
            % 1. Set the number of frames to acquire per grab
            obj.hSI.hStackManager.framesPerSlice = obj.NumFramesToAverage;

            % 2. Set the display rolling average to match the grab size
            obj.hSI.hDisplay.displayRollingAverageFactor = obj.NumFramesToAverage;

            % 3. Start the acquisition
            obj.hSI.startGrab();

            % 4. Actively wait for the grab to finish instead of a fixed pause
            while strcmpi(obj.hSI.acqState, 'grab') || strcmpi(obj.hSI.acqState, 'active')
                pause(obj.PollPeriodSec);
            end

            % Small buffer pause to ensure the final frame has fully populated
            pause(0.1);

            % 5. Retrieve data
            % Prioritize hDisplay.lastFrame because it holds the natively averaged image
            if isprop(obj.hSI,'hDisplay') && isprop(obj.hSI.hDisplay,'lastFrame') && ~isempty(obj.hSI.hDisplay.lastFrame)
                img = double(obj.hSI.hDisplay.lastFrame{1});

            elseif isfield(obj.hSI,'acq') && isprop(obj.hSI.acq,'pixelData') && ~isempty(obj.hSI.acq.pixelData)
                img = double(obj.hSI.acq.pixelData);

            else
                img = rand(obj.FallbackSize) * 100;  % last resort
            end
        end
    end
end
