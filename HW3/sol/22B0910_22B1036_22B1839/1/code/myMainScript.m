
%%%%%%%%%%%%%%%%%%%%%%% Read the images %%%%%%%%%%%%%%%%%%%%%%%%

img = imread('../images/barbara256.png');
img = double(img);

%%%%%%%%%%%%%%%%%%%% Apply filters no images %%%%%%%%%%%%%%%%%%%%%%

% Zero-pad the image
[M, N] = size(img);
padded_img = padarray(img, [M, N], 'post');

% Compute Fourier transform of the padded image
FT_img = fftshift(fft2(padded_img));

% Define frequency grid (adjust for padded image dimensions)
[u, v] = meshgrid(-N:N-1, -M:M-1);
D = sqrt(u.^2 + v.^2); % Euclidean distance from center of frequency space

% Cutoff frequencies and sigmas
D0_40 = 40; D0_80 = 80;
sigma_40 = 40; sigma_80 = 80;

% Ideal low-pass filters
H_ideal_40 = double(D <= D0_40);
H_ideal_80 = double(D <= D0_80);

% Gaussian low-pass filters
H_gaussian_40 = exp(-(D.^2) / (2 * sigma_40^2));
H_gaussian_80 = exp(-(D.^2) / (2 * sigma_80^2));
filtered_img_ideal_40 = real(ifft2(ifftshift(H_ideal_40 .* FT_img)));
filtered_img_ideal_80 = real(ifft2(ifftshift(H_ideal_80 .* FT_img)));

filtered_img_gaussian_40 = real(ifft2(ifftshift(H_gaussian_40 .* FT_img)));
filtered_img_gaussian_80 = real(ifft2(ifftshift(H_gaussian_80 .* FT_img)));

filtered_img_ideal_40 = filtered_img_ideal_40(1:M, 1:N);
filtered_img_ideal_80 = filtered_img_ideal_80(1:M, 1:N);

filtered_img_gaussian_40 = filtered_img_gaussian_40(1:M, 1:N);
filtered_img_gaussian_80 = filtered_img_gaussian_80(1:M, 1:N);

log_FT_img = log(abs(FT_img) + 1);

log_FT_ideal_40 = log(abs(H_ideal_40 .* FT_img) + 1);
log_FT_ideal_80 = log(abs(H_ideal_80 .* FT_img) + 1);

log_FT_gaussian_40 = log(abs(H_gaussian_40 .* FT_img) + 1);
log_FT_gaussian_80 = log(abs(H_gaussian_80 .* FT_img) + 1);

%%%%%%%%%%%%%%%%%%%% Create an output directory %%%%%%%%%%%%%%%%%%%%%%

output_dir = '../images/output_images/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%%%%%%%%%%%%%%%%%%%%%%%% Save Images %%%%%%%%%%%%%%%%%%%%%%%%%%

% Save original padded image's Fourier transform
imwrite(mat2gray(log_FT_img), fullfile(output_dir, 'log_FT_original.png'));

% Save filtered images
imwrite(mat2gray(filtered_img_ideal_40), fullfile(output_dir, 'filtered_img_ideal_D40.png'));
imwrite(mat2gray(filtered_img_ideal_80), fullfile(output_dir, 'filtered_img_ideal_D80.png'));
imwrite(mat2gray(filtered_img_gaussian_40), fullfile(output_dir, 'filtered_img_gaussian_sigma40.png'));
imwrite(mat2gray(filtered_img_gaussian_80), fullfile(output_dir, 'filtered_img_gaussian_sigma80.png'));

% Save log Fourier transforms of filtered images
imwrite(mat2gray(log_FT_ideal_40), fullfile(output_dir, 'log_FT_ideal_D40.png'));
imwrite(mat2gray(log_FT_ideal_80), fullfile(output_dir, 'log_FT_ideal_D80.png'));
imwrite(mat2gray(log_FT_gaussian_40), fullfile(output_dir, 'log_FT_gaussian_sigma40.png'));
imwrite(mat2gray(log_FT_gaussian_80), fullfile(output_dir, 'log_FT_gaussian_sigma80.png'));

% Save frequency domain filters
imwrite(mat2gray(H_ideal_40), fullfile(output_dir, 'ideal_LP_filter_D40.png'));
imwrite(mat2gray(H_ideal_80), fullfile(output_dir, 'ideal_LP_filter_D80.png'));
imwrite(mat2gray(H_gaussian_40), fullfile(output_dir, 'gaussian_LP_filter_sigma40.png'));
imwrite(mat2gray(H_gaussian_80), fullfile(output_dir, 'gaussian_LP_filter_sigma80.png'));


%%%%%%%%%%%%%%%%%%%% Display all the images %%%%%%%%%%%%%%%%%%%%%%

% Display Ideal and Gaussian low-pass filters in the frequency domain
figure;
subplot(1,2,1), imshow(H_ideal_40, []), title('Ideal Low-Pass Filter (D=40) in Frequency Domain');
subplot(1,2,2), imshow(H_gaussian_40, []), title('Gaussian Low-Pass Filter (σ=40) in Frequency Domain');

% Display larger filter comparison
figure;
subplot(1,2,1), imshow(H_ideal_80, []), title('Ideal Low-Pass Filter (D=80) in Frequency Domain');
subplot(1,2,2), imshow(H_gaussian_80, []), title('Gaussian Low-Pass Filter (σ=80) in Frequency Domain');

% Display results (log scale)
figure;
subplot(2,3,1), imshow(img, []), title('Original Image');
subplot(2,3,2), imshow(filtered_img_ideal_40, []), title('Ideal Low-Pass (D=40)');
subplot(2,3,3), imshow(filtered_img_ideal_80, []), title('Ideal Low-Pass (D=80)');
subplot(2,3,4), imshow(filtered_img_gaussian_40, []), title('Gaussian Low-Pass (σ=40)');
subplot(2,3,5), imshow(filtered_img_gaussian_80, []), title('Gaussian Low-Pass (σ=80)');

% Display log Fourier transforms of filtered images
figure;
subplot(2,3,1), imshow(log_FT_img, []), title('Log Fourier Transform of Original');
subplot(2,3,2), imshow(log_FT_ideal_40, []), title('Log FT (Ideal D=40)');
subplot(2,3,3), imshow(log_FT_ideal_80, []), title('Log FT (Ideal D=80)');
subplot(2,3,4), imshow(log_FT_gaussian_40, []), title('Log FT (Gaussian σ=40)');
subplot(2,3,5), imshow(log_FT_gaussian_80, []), title('Log FT (Gaussian σ=80)');



