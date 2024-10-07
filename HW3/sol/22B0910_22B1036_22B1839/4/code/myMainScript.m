
%%%%%%%%%%%%%%%%%%%% Create an image %%%%%%%%%%%%%%%%%%%%%%

N = 201;
image = zeros(N, N);
image(:, ceil(N/2)) = 255;

%%%%%%%%%%%%%%%%%%%% Perform 2D FFT %%%%%%%%%%%%%%%%%%%%%%

% Compute the 2D Fourier Transform
F = fft2(image);

% Shift the zero-frequency component to the center
F_shifted = fftshift(F);

% Compute the magnitude of the Fourier transform
magnitude = abs(F_shifted);

% Take the logarithm of the magnitude for better visualization
log_magnitude = log(1 + magnitude);


%%%%%%%%%%%%%%%%%%%% Apply filters no images %%%%%%%%%%%%%%%%%%%%%%

output_dir = '../images/output_images/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%%%%%%%%%%%%%%%%%%%% Save the images %%%%%%%%%%%%%%%%%%%%%%

imwrite(uint8(image), fullfile(output_dir, 'original_image.png'));


%%%%%%%%%%%%%%%%%%%% Display all the images %%%%%%%%%%%%%%%%%%%%%%

% Plot the original image and save it
figure;
imshow(image);
title('Original Image');
axis equal;

% Plot the result and save it
figure;
imagesc(log_magnitude);
colormap(jet); % Use a color map to enhance visualization
colorbar;
title('Logarithm of the Fourier Magnitude');
axis equal;
saveas(gcf, fullfile(output_dir, 'fourier_magnitude_plot.png'));
