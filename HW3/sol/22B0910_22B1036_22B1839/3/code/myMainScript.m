%% import mymeanshift.* % Assuming mymeanshift is in your working directory

tic;

% Read images
barbara_img = imread("./../images/barbara256.png");
kodak_img = imread("./../images/kodak24.png");
output_dir = createDir();

% ----------------------------------------------------------------------------- 
%                                   σ = 5 
% ----------------------------------------------------------------------------- 

% Add zero-mean Gaussian noise with standard deviation σ = 5
sigma_noise = 5;
barbara_noise_sig5 = imnoise(barbara_img, 'gaussian', 0, (sigma_noise/255)^2);
kodak_noise_sig5 = imnoise(kodak_img, 'gaussian', 0, (sigma_noise/255)^2);
disp('Added Gaussian noise (σ=5) to images.');

% Convert noisy images to uint8
barbara_noise_sig5 = uint8(barbara_noise_sig5);
kodak_noise_sig5 = uint8(kodak_noise_sig5);

% ----------------------------------------------------------------------------- 

% Perform Mean Shift filter with given parameters on barbara
% σs = 2, σr = 2
barbara_s5_ms_01 = mymeanshift(barbara_noise_sig5, 2, 2);
disp('Computed Barbara σs=2, σr=2 for σ=5.');

% σs = 15, σr = 3
barbara_s5_ms_02 = mymeanshift(barbara_noise_sig5, 15, 3);
disp('Computed Barbara σs=15, σr=3 for σ=5.');

% σs = 3, σr = 15
barbara_s5_ms_03 = mymeanshift(barbara_noise_sig5, 3, 15);
disp('Computed Barbara σs=3, σr=15 for σ=5.');


% Perform Mean Shift filter with given parameters on kodak
% σs = 2, σr = 2
kodak_s5_ms_01 = mymeanshift(kodak_noise_sig5, 2, 2);
disp('Computed Kodak σs=2, σr=2 for σ=5.');

% σs = 15, σr = 3
kodak_s5_ms_02 = mymeanshift(kodak_noise_sig5, 15, 3);
disp('Computed Kodak σs=15, σr=3 for σ=5.');

% σs = 3, σr = 15
kodak_s5_ms_03 = mymeanshift(kodak_noise_sig5, 3, 15);
disp('Computed Kodak σs=3, σr=15 for σ=5.');


% Save Images Barbara (for σ=5)
imwrite(barbara_noise_sig5, fullfile(output_dir, 'barbara_noise_sigma5.png'));
imwrite(barbara_s5_ms_01, fullfile(output_dir, 'barbara_s5_ms_01.png'));
imwrite(barbara_s5_ms_02, fullfile(output_dir, 'barbara_s5_ms_02.png'));
imwrite(barbara_s5_ms_03, fullfile(output_dir, 'barbara_s5_ms_03.png'));
disp('Saved Barbara images for σ=5.');

% Save Images Kodak (for σ=5)
imwrite(kodak_noise_sig5, fullfile(output_dir, 'kodak_noise_sigma5.png'));
imwrite(kodak_s5_ms_01, fullfile(output_dir, 'kodak_s5_ms_01.png'));
imwrite(kodak_s5_ms_02, fullfile(output_dir, 'kodak_s5_ms_02.png'));
imwrite(kodak_s5_ms_03, fullfile(output_dir, 'kodak_s5_ms_03.png'));
disp('Saved Kodak images for σ=5.');

% ----------------------------------------------------------------------------- 
%                                   σ = 10 
% ----------------------------------------------------------------------------- 

% Add zero-mean Gaussian noise with standard deviation σ = 10
sigma_noise = 10;
barbara_noise_sig10 = imnoise(barbara_img, 'gaussian', 0, (sigma_noise/255)^2);
kodak_noise_sig10 = imnoise(kodak_img, 'gaussian', 0, (sigma_noise/255)^2);
disp('Added Gaussian noise (σ=10) to images.');

% Convert noisy images to uint8
barbara_noise_sig10 = uint8(barbara_noise_sig10);
kodak_noise_sig10 = uint8(kodak_noise_sig10);

% ----------------------------------------------------------------------------- 

% Perform Mean Shift filter with given parameters on barbara
% σs = 2, σr = 2
barbara_s10_ms_01 = mymeanshift(barbara_noise_sig10, 2, 2);
disp('Computed Barbara σs=2, σr=2 for σ=10.');

% σs = 15, σr = 3
barbara_s10_ms_02 = mymeanshift(barbara_noise_sig10, 15, 3);
disp('Computed Barbara σs=15, σr=3 for σ=10.');

% σs = 3, σr = 15
barbara_s10_ms_03 = mymeanshift(barbara_noise_sig10, 3, 15);
disp('Computed Barbara σs=3, σr=15 for σ=10.');

% Perform Mean Shift filter with given parameters on kodak
% σs = 2, σr = 2
kodak_s10_ms_01 = mymeanshift(kodak_noise_sig10, 2, 2);
disp('Computed Kodak σs=2, σr=2 for σ=10.');

% σs = 15, σr = 3
kodak_s10_ms_02 = mymeanshift(kodak_noise_sig10, 15, 3);
disp('Computed Kodak σs=15, σr=3 for σ=10.');

% σs = 3, σr = 15
kodak_s10_ms_03 = mymeanshift(kodak_noise_sig10, 3, 15);
disp('Computed Kodak σs=3, σr=15 for σ=10.');

% Save Images Barbara (for σ=10)
imwrite(barbara_noise_sig10, fullfile(output_dir, 'barbara_noise_sigma10.png'));
imwrite(barbara_s10_ms_01, fullfile(output_dir, 'barbara_s10_ms_01.png'));
imwrite(barbara_s10_ms_02, fullfile(output_dir, 'barbara_s10_ms_02.png'));
imwrite(barbara_s10_ms_03, fullfile(output_dir, 'barbara_s10_ms_03.png'));
disp('Saved Barbara images for σ=10.');

% Save Images Kodak (for σ=10)
imwrite(kodak_noise_sig10, fullfile(output_dir, 'kodak_noise_sigma10.png'));
imwrite(kodak_s10_ms_01, fullfile(output_dir, 'kodak_s10_ms_01.png'));
imwrite(kodak_s10_ms_02, fullfile(output_dir, 'kodak_s10_ms_02.png'));
imwrite(kodak_s10_ms_03, fullfile(output_dir, 'kodak_s10_ms_03.png'));
disp('Saved Kodak images for σ=10.');

toc;

% ----------------------------------------------------------------------------- 
%                               Display all Images 
% ----------------------------------------------------------------------------- 

% Display Barbara images
figure;
subplot(2,2,1); imshow(barbara_noise_sig5); title('Barbara Noise (σ=5)');
subplot(2,2,2); imshow(barbara_s5_ms_01); title('Barbara σs=2, σr=2');
subplot(2,2,3); imshow(barbara_s5_ms_02); title('Barbara σs=15, σr=3');
subplot(2,2,4); imshow(barbara_s5_ms_03); title('Barbara σs=3, σr=15');

figure;
subplot(1,1,1); imshow(barbara_img); title('Barbara Original');

% Wait for user to close Barbara figure
waitForKeyPressAndCloseFigures();

% ----------------------------------------------------------------------------- 

% Display Kodak images
figure;
subplot(2,2,1); imshow(kodak_noise_sig5); title('Kodak Noise (σ=5)');
subplot(2,2,2); imshow(kodak_s5_ms_01); title('Kodak σs=2, σr=2');
subplot(2,2,3); imshow(kodak_s5_ms_02); title('Kodak σs=15, σr=3');
subplot(2,2,4); imshow(kodak_s5_ms_03); title('Kodak σs=3, σr=15');

figure;
subplot(1,1,1); imshow(kodak_img); title('Kodak Original');

% Wait for user to close Kodak figure
waitForKeyPressAndCloseFigures();

% ----------------------------------------------------------------------------- 

% Display Barbara images
figure;
subplot(2,2,1); imshow(barbara_noise_sig10); title('Barbara Noise (σ=10)');
subplot(2,2,2); imshow(barbara_s10_ms_01); title('Barbara σs=2, σr=2');
subplot(2,2,3); imshow(barbara_s10_ms_02); title('Barbara σs=15, σr=3');
subplot(2,2,4); imshow(barbara_s10_ms_03); title('Barbara σs=3, σr=15');

figure;
subplot(1,1,1); imshow(barbara_img); title('Barbara Original');

% Wait for user to close Barbara figure
waitForKeyPressAndCloseFigures();

% ----------------------------------------------------------------------------- 

% Display Kodak images
figure;
subplot(2,2,1); imshow(kodak_noise_sig10); title('Kodak Noise (σ=10)');
subplot(2,2,2); imshow(kodak_s10_ms_01); title('Kodak σs=2, σr=2');
subplot(2,2,3); imshow(kodak_s10_ms_02); title('Kodak σs=15, σr=3');
subplot(2,2,4); imshow(kodak_s10_ms_03); title('Kodak σs=3, σr=15');

figure;
subplot(1,1,1); imshow(kodak_img); title('Kodak Original');

% Wait for user to close Kodak figure
waitForKeyPressAndCloseFigures();



% ----------------------------------------------------------------------------- 
%                                   Functions 
% ----------------------------------------------------------------------------- 

% Function to close prev images and display next sets of images
function waitForKeyPressAndCloseFigures()
    % Wait for user to press 'q' to close all figures and proceed
    disp('Press ''q'' to continue to next set of images...');
    while true
        key = waitforbuttonpress;
        if key == 1
            keyPressed = get(gcf, 'CurrentCharacter');
            if keyPressed == 'q' || keyPressed == 'Q'
                close all;  % Close all figures
                break;
            end
        end
    end
end

% Mean Shift function definition
function output = mymeanshift(input_image, spatial_radius, range_radius)
    % Convert image to double precision for processing
    input_image = im2double(input_image);
    
    % Initialize output image
    output = input_image;
    [rows, cols, ~] = size(input_image);
    
    % Create a grid for spatial locations
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Iterate through each pixel
    for r = 1:rows
        for c = 1:cols
            % Get the current pixel value
            current_pixel = squeeze(input_image(r, c, :));
            
            % Define the window around the current pixel
            spatial_mask = ((X - c).^2 + (Y - r).^2) <= spatial_radius^2;
            
            % Get the neighborhood pixel values
            neighborhood = input_image(spatial_mask);
            [~, ~] = find(spatial_mask);
            
            % Initialize the Mean Shift loop
            shifting = true;
            while shifting
                % Compute the mean of the neighborhood based on range
                range_mask = vecnorm(neighborhood - current_pixel, 2, 2) <= range_radius;
                if any(range_mask)
                    weighted_mean = sum(neighborhood(range_mask), 1) / sum(range_mask);
                    if norm(weighted_mean - current_pixel) < 1e-2
                        shifting = false;
                    else
                        current_pixel = weighted_mean;
                    end
                else
                    shifting = false;
                end
            end
            
            % Assign the final pixel value to the output image
            output(r, c, :) = current_pixel;
        end
    end
    
    % Convert back to uint8 for image display
    output = im2uint8(output);
end

% Function to create output directory
function output_dir = createDir()
    output_dir = '../images/output_images/';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
end
