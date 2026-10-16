classdef SLMController < handle
    % SLMController - Meadowlark Blink SDK wrapper for repeated writes
    %
    % Usage:
    %   slm = SLMController('BoardNumber',1,'BitDepth',12, 'LUT1024',..., 'LUT512',...);
    %   slm.load();                 % loads libs + Create_SDK + query dims + load LUT
    %   slm.setCircularMask();      % optional mask; or slm.setMask(maskVec)
    %   slm.writeEfield(E);         % E is complex vector length width*height
    %   slm.delete();               % cleanup

    properties
        % Config
        BitDepth (1,1) double = 12
        BoardNumber (1,1) double = 1
        IsNematicType (1,1) double = 1
        RAMWriteEnable (1,1) double = 1
        UseGPU (1,1) double = 0
        MaxTransients (1,1) double = 10

        LUT1024 (1,:) char = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm6748_at1300_1stOrder_031325_3.lut'
        LUT512  (1,:) char = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\512x512_linearVoltage.LUT'

        WriteTimeoutMs (1,1) double = 5000

        % Phase mapping:
        % Your original: uint8(mod(angle(E) * inv_pi_over_128, 256))
        InvPiOver128 (1,1) double = 128/pi   % set this to your inv_pi_over_128

        % Mask (uint8 vector length width*height, values 0/1)
        Mask uint8 = uint8([])

        % Derived after load()
        Width (1,1) double = NaN
        Height (1,1) double = NaN
        Depth (1,1) double = NaN
    end

    properties (Access=private)
        BytesPerPixel (1,1) double = NaN
        NBytes (1,1) double = NaN
        E_SLM_ptr
        IsLoaded (1,1) logical = false
    end

    methods
        function obj = SLMController(varargin)
            % Allow name-value construction
            if ~isempty(varargin)
                for k = 1:2:numel(varargin)
                    obj.(varargin{k}) = varargin{k+1};
                end
            end
        end

        function load(obj)
            % load libraries and initialize SDK
            if ~libisloaded('Blink_C_wrapper')
                loadlibrary('Blink_C_wrapper.dll', 'Blink_C_wrapper.h');
            end
            if ~libisloaded('ImageGen')
                loadlibrary('ImageGen.dll', 'ImageGen.h');
            end

            numBoardsFound  = libpointer('uint32Ptr', 0);
            constructedOkay = libpointer('int32Ptr', 0);
            regLut          = libpointer('string');

            calllib('Blink_C_wrapper','Create_SDK', ...
                obj.BitDepth, numBoardsFound, constructedOkay, ...
                obj.IsNematicType, obj.RAMWriteEnable, obj.UseGPU, obj.MaxTransients, regLut);

            if constructedOkay.value ~= 1
                error('Blink SDK initialization failed.');
            end

            obj.Height = calllib('Blink_C_wrapper','Get_image_height', obj.BoardNumber);
            obj.Width  = calllib('Blink_C_wrapper','Get_image_width',  obj.BoardNumber);
            obj.Depth  = calllib('Blink_C_wrapper','Get_image_depth',  obj.BoardNumber);

            obj.BytesPerPixel = obj.Depth/8;
            obj.NBytes = obj.Width * obj.Height * obj.BytesPerPixel;

            % preallocate the output buffer once
            obj.E_SLM_ptr = libpointer('uint8Ptr', zeros(obj.NBytes,1,'uint8'));

            % default mask = all ones
            obj.Mask = uint8(ones(obj.Width*obj.Height,1));

            % load LUT
            if obj.Width == 1024
                lutPath = obj.LUT1024;
            else
                lutPath = obj.LUT512;
            end
            calllib('Blink_C_wrapper','Load_LUT_file', obj.BoardNumber, lutPath);

            obj.IsLoaded = true;
        end

        function setMask(obj, maskVec)
            obj.assertLoaded();
            if ~isa(maskVec,'uint8')
                maskVec = uint8(maskVec);
            end
            if numel(maskVec) ~= obj.Width*obj.Height
                error('Mask must have length width*height (%d).', obj.Width*obj.Height);
            end
            obj.Mask = maskVec(:);
        end

        function setCircularMask(obj, varargin)
            % setCircularMask(obj) -> centered, radius = min(width,height)/2
            % setCircularMask(obj,'Center',[cx cy],'Radius',r)
            obj.assertLoaded();

            p = inputParser;
            addParameter(p,'Center',[obj.Width/2, obj.Height/2]);
            addParameter(p,'Radius',min(obj.Width,obj.Height)/2);
            parse(p,varargin{:});

            center = p.Results.Center;
            r = p.Results.Radius;

            [X,Y] = meshgrid(1:obj.Width, 1:obj.Height);
            Rnorm = sqrt((X-center(1)).^2 + (Y-center(2)).^2) / r;

            % Keep your original reshape convention (width*height,1)
            mask = uint8(reshape(Rnorm <= 1, obj.Width*obj.Height, 1));
            obj.Mask = mask;
        end

        function writeEfield(obj, EfieldVec)
            % EfieldVec: complex vector of length width*height
            obj.assertLoaded();

            if isempty(obj.InvPiOver128)
                error('Set obj.InvPiOver128 before calling writeEfield().');
            end
            if numel(EfieldVec) ~= obj.Width*obj.Height
                error('EfieldVec must have length width*height (%d).', obj.Width*obj.Height);
            end

            phase8 = uint8(mod(angle(EfieldVec(:)) * obj.InvPiOver128, 256));
            %frame  = uint8(phase8) .* uint8(obj.Mask);

            if obj.BytesPerPixel ~= 1
                error('SLMController:UnsupportedDepth', ...
                    'This phase8 writer requires an 8-bit SDK image buffer; got %g bits.', obj.Depth);
            end
            obj.E_SLM_ptr.value = phase8;

            calllib('Blink_C_wrapper','Write_image', ...
                obj.BoardNumber, obj.E_SLM_ptr, obj.NBytes, ...
                0,0,0,0, obj.WriteTimeoutMs);

            calllib('Blink_C_wrapper','ImageWriteComplete', obj.BoardNumber, obj.WriteTimeoutMs);
        end

        function delete(obj)
            % cleanup on object destruction
            if obj.IsLoaded
                calllib('Blink_C_wrapper','Delete_SDK');
                obj.IsLoaded = false;
            end
            if libisloaded('Blink_C_wrapper')
                unloadlibrary('Blink_C_wrapper');
            end
            if libisloaded('ImageGen')
                unloadlibrary('ImageGen');
            end
            fprintf('Libraries unloaded \n');
        end
    end

    methods (Access=private)
        function assertLoaded(obj)
            if ~obj.IsLoaded
                error('SLMController is not loaded. Call slm.load() first.');
            end
        end
    end
end
