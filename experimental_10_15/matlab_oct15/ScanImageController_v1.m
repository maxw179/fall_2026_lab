classdef ScanImageController < handle
    % ScanImageController - small wrapper around hSI for repeated grabs

    properties
        hSI
        PollPeriodSec (1,1) double = 0.5
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
            obj.hSI.startGrab();
            %while strcmpi(obj.hSI.acqState, 'active')
%             pause(1);
            %end
            pause(obj.PollPeriodSec);

            % Retrieve data
            if isfield(obj.hSI,'acq') && isprop(obj.hSI.acq,'pixelData') && ~isempty(obj.hSI.acq.pixelData)
                img = double(obj.hSI.acq.pixelData);

            elseif isprop(obj.hSI,'hDisplay') && isprop(obj.hSI.hDisplay,'lastFrame')
                img = double(obj.hSI.hDisplay.lastFrame{1});

            else
                img = rand(obj.FallbackSize) * 100;  % last resort
            end
        end
    end
end
