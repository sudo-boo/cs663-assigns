import mybilateralfilter.*

tic;

% Read images
barbara_img = imread("./../images/barbara256.png");
kodak_img = imread("./../images/kodak24.png");

% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------


% Add zero-mean Gaussian noise with standard deviation σ = 5
sigma_noise = 5;
barbara_noise_sig6 = imnoise(barbara_img, 'gaussian', 0, (sigma_noise/255)^2);
kodak_noise_sig6 = imnoise(kodak_img, 'gaussian', 0, (sigma_noise/255)^2);


% -----------------------------------------------------------------------------


% Perform bilateral filter with given parameters on barbara
% σs = 2, σr = 2
barbara_s5_bf_01 = mybilateralfilter(barbara_noise_sig6, 2, 2);
% σs = 0.1, σr = 0.1
barbara_s5_bf_02 = mybilateralfilter(barbara_noise_sig6, 0.1, 0.1);
% σs = 3, σr = 15
barbara_s5_bf_03 = mybilateralfilter(barbara_noise_sig6, 3, 15);

% Perform bilateral filter with given parameters on kodak
% σs = 2, σr = 2
kodak_s5_bf_01 = mybilateralfilter(kodak_noise_sig6, 2, 2);
% σs = 0.1, σr = 0.1
kodak_s5_bf_02 = mybilateralfilter(kodak_noise_sig6, 0.1, 0.1);
% σs = 3, σr = 15
kodak_s5_bf_03 = mybilateralfilter(kodak_noise_sig6, 3, 15);


% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------


% Add zero-mean Gaussian noise with standard deviation σ = 10
sigma_noise = 10;
barbara_noise_sig10 = imnoise(barbara_img, 'gaussian', 0, (sigma_noise/255)^2);
kodak_noise_sig10 = imnoise(kodak_img, 'gaussian', 0, (sigma_noise/255)^2);


% -----------------------------------------------------------------------------


% Perform bilateral filter with given parameters on barbara
% σs = 2, σr = 2
barbara_s10_bf_01 = mybilateralfilter(barbara_noise_sig6, 2, 2);
% σs = 0.1, σr = 0.1
barbara_s10_bf_02 = mybilateralfilter(barbara_noise_sig6, 0.1, 0.1);
% σs = 3, σr = 15
barbara_s10_bf_03 = mybilateralfilter(barbara_noise_sig6, 3, 15);

% Perform bilateral filter with given parameters on kodak
% σs = 2, σr = 2
kodak_s10_bf_01 = mybilateralfilter(kodak_noise_sig6, 2, 2);
% σs = 0.1, σr = 0.1
kodak_s10_bf_02 = mybilateralfilter(kodak_noise_sig6, 0.1, 0.1);
% σs = 3, σr = 15
kodak_s10_bf_03 = mybilateralfilter(kodak_noise_sig6, 3, 15);


% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------
toc;


% -----------------------------------------------------------------------------
% Display all Images
% -----------------------------------------------------------------------------

% Display Barbara images
figure;
subplot(2,2,1); imshow(barbara_noise_sig6); title('Barbara Noise (σ=5)');
subplot(2,2,2); imshow(barbara_s5_bf_01); title('Barbara σs=2, σr=2');
subplot(2,2,3); imshow(barbara_s5_bf_02); title('Barbara σs=0.1, σr=0.1');
subplot(2,2,4); imshow(barbara_s5_bf_03); title('Barbara σs=3, σr=15');

figure;
subplot(1,1,1); imshow(barbara_img); title('Barbara Original');

% Wait for user to close Barbara figure
waitForKeyPressAndCloseFigures();


% -----------------------------------------------------------------------------

% Display kodak images
figure;
subplot(2,2,1); imshow(kodak_noise_sig6); title('kodak Noise (σ=5)');
subplot(2,2,2); imshow(kodak_s5_bf_01); title('kodak σs=2, σr=2');
subplot(2,2,3); imshow(kodak_s5_bf_02); title('kodak σs=0.1, σr=0.1');
subplot(2,2,4); imshow(kodak_s5_bf_03); title('kodak σs=3, σr=15');

figure;
subplot(1,1,1); imshow(kodak_img); title('kodak Original');

    
% Wait for user to close Kodak figure
waitForKeyPressAndCloseFigures();

% -----------------------------------------------------------------------------


% Display Barbara images
figure;
subplot(2,2,1); imshow(barbara_noise_sig10); title('Barbara Noise (σ=10)');
subplot(2,2,2); imshow(barbara_s10_bf_01); title('Barbara σs=2, σr=2');
subplot(2,2,3); imshow(barbara_s10_bf_02); title('Barbara σs=0.1, σr=0.1');
subplot(2,2,4); imshow(barbara_s10_bf_03); title('Barbara σs=3, σr=15');

figure;
subplot(1,1,1); imshow(barbara_img); title('Barbara Original');

% Wait for user to close Barbara figure
waitForKeyPressAndCloseFigures();

% -----------------------------------------------------------------------------


% Display Kodak images
figure;
subplot(2,2,1); imshow(kodak_noise_sig10); title('kodak Noise (σ=10)');
subplot(2,2,2); imshow(kodak_s10_bf_01); title('kodak σs=2, σr=2');
subplot(2,2,3); imshow(kodak_s10_bf_02); title('kodak σs=0.1, σr=0.1');
subplot(2,2,4); imshow(kodak_s10_bf_03); title('kodak σs=3, σr=15');

figure;
subplot(1,1,1); imshow(kodak_img); title('kodak Original');

% Wait for user to close final figures
waitForKeyPressAndCloseFigures();

% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------


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


% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------