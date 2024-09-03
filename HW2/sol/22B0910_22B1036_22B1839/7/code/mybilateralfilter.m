
% -----------------------------------------------------------------------------

% Takes image, sigma_s, sigma_r as input and returns an image after
% applying bilateral filter with the given parameters.

function output = mybilateralfilter(image, sigma_s, sigma_r)

    % Convert the image to double for precision
    image = double(image);

    [rows, cols] = size(image);
    
    % Kernel Size not specified so using kernel radius of 
    % 3*sigma_s to include 99.7% of the Gaussian distribution, 
    % to ensure the kernel captures most of the relevant values
    kernelR = ceil(3 * sigma_s);
    
    % Create the spatial Gaussian kernel
    [X, Y] = meshgrid(-kernelR:kernelR, -kernelR:kernelR);
    spatialKernel = exp(-(X.^2 + Y.^2) / (2 * sigma_s^2));

    % Initialize the output image
    output = zeros(rows, cols);

    % Apply the bilateral filter on all image
    for i = 1:rows
        for j = 1:cols

            % Define the local region
            iMin = max(i - kernelR, 1);
            jMin = max(j - kernelR, 1);
            iMax = min(i + kernelR, rows);
            jMax = min(j + kernelR, cols);
            localRegion = image(iMin:iMax, jMin:jMax);

            % Compute the intensity Gaussian kernel
            intensityRangeKernel = exp((-(localRegion - image(i, j)) .^ 2) / (2 * sigma_r^2));
            
            % Compute the local Spacial Gaussian kernel
            subSpacialKernel = spatialKernel((iMin:iMax) - i + kernelR + 1, (jMin:jMax) - j + kernelR + 1);
            % adding (kernelR + 1) to shift the domain from (-kernelR, +kernelR) to (0, kernelSize)

            % Combine the spatial and intensity range kernels
            combinedKernel = subSpacialKernel .* intensityRangeKernel;

            % Normalize the combined kernel
            combinedKernel = combinedKernel / sum(combinedKernel(:));
            output(i, j) = sum(sum(combinedKernel .* localRegion));
        end
    end
    % since our image intensity values range from 0 to 255
    output = uint8(output);
end


% -----------------------------------------------------------------------------
