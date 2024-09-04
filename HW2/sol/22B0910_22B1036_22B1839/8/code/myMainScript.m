tic;

% Read images
lc1 = imread('../images/LC1.png');
lc2 = imread('../images/LC2.jpg');


% -----------------------------------------------------------------------------
%                   Perform Local Histogram Equalisation
% -----------------------------------------------------------------------------

% Perform Local Histogram Equalisation on LC1 with varying sizes
lc1_lhe_7x7 = performLocalHistogramEqualisation(lc1, 7);
lc1_lhe_31x31 = performLocalHistogramEqualisation(lc1, 31);
lc1_lhe_51x51 = performLocalHistogramEqualisation(lc1, 51);
lc1_lhe_71x71 = performLocalHistogramEqualisation(lc1, 71);

% Perform Global HE
lc1_ghe = histeq(lc1);

% -----------------------------------------------------------------------------

% Perform Local Histogram Equalisation on LC2 with varying sizes
lc2_lhe_7x7 = performLocalHistogramEqualisation(lc2, 7);
lc2_lhe_31x31 = performLocalHistogramEqualisation(lc2, 31);
lc2_lhe_51x51 = performLocalHistogramEqualisation(lc2, 51);
lc2_lhe_71x71 = performLocalHistogramEqualisation(lc2, 71);

% Perform Global HE
lc2_ghe = histeq(lc2);

% Save Images
output_dir = createDir();

% Save LC1s
imwrite(lc1_lhe_7x7, fullfile(output_dir, 'LC1_LHE_7x7.png'));
imwrite(lc1_lhe_31x31, fullfile(output_dir, 'LC1_LHE_31x31.png'));
imwrite(lc1_lhe_51x51, fullfile(output_dir, 'LC1_LHE_51x51.png'));
imwrite(lc1_lhe_71x71, fullfile(output_dir, 'LC1_LHE_71x71.png'));
imwrite(lc1_ghe, fullfile(output_dir, 'LC1_GHE.png'));

% Save LC2s
imwrite(lc2_lhe_7x7, fullfile(output_dir, 'LC2_LHE_7x7.png'));
imwrite(lc2_lhe_31x31, fullfile(output_dir, 'LC2_LHE_31x31.png'));
imwrite(lc2_lhe_51x51, fullfile(output_dir, 'LC2_LHE_51x51.png'));
imwrite(lc2_lhe_71x71, fullfile(output_dir, 'LC2_LHE_71x71.png'));
imwrite(lc2_ghe, fullfile(output_dir, 'LC2_GHE.png'));

toc;




% -----------------------------------------------------------------------------
%                               Display all Images
% -----------------------------------------------------------------------------



% Display LC1s
figure;
subplot(2,2,1); imshow(lc1_lhe_7x7); title('LC1 LHE 7x7');
subplot(2,2,2); imshow(lc1_lhe_31x31); title('LC1 LHE 31x31');
subplot(2,2,3); imshow(lc1_lhe_51x51); title('LC1 LHE 51x51');
subplot(2,2,4); imshow(lc1_lhe_71x71); title('LC1 LHE 71x71');

figure;
subplot(1,1,1); imshow(lc1); title('LC1 Original');

figure;
subplot(1,1,1); imshow(lc1_ghe); title('LC1 Global HE');

waitForKeyPressAndCloseFigures();

% -----------------------------------------------------------------------------

% Display LC2s
figure;
subplot(2,2,1); imshow(lc2_lhe_7x7); title('LC2 LHE 7x7');
subplot(2,2,2); imshow(lc2_lhe_31x31); title('LC2 LHE 31x31');
subplot(2,2,3); imshow(lc2_lhe_51x51); title('LC2 LHE 51x51');
subplot(2,2,4); imshow(lc2_lhe_71x71); title('LC2 LHE 71x71');

figure;
subplot(1,1,1); imshow(lc2); title('LC2 Original');

figure;
subplot(1,1,1); imshow(lc2_ghe); title('LC2 Global HE');

waitForKeyPressAndCloseFigures();





% -----------------------------------------------------------------------------
%                                   Functions
% -----------------------------------------------------------------------------

% Function performs Local Histogram Equalisation
function output = performLocalHistogramEqualisation(image, window_size)
    [rows, cols] = size(image);
    output = zeros(rows, cols, 'uint8');

    % Can do without padding too by considering only the allowed region of
    % image but would get dark regions around the borders in some cases
    % (esp in LC1) hence using a padding of size / 2 to maintain histogram
    % consistency, and have enough information near the borders

    pad_size = floor(window_size / 2);
    padded_image = padarray(image, [pad_size pad_size], 'replicate');
    
    for i = 1:rows
        for j = 1:cols
            % Define the local region (window to iterate)
            local_region = padded_image(i:(i + window_size - 1), j:(j + window_size - 1));
  
            hist_vector = imhist(local_region(:));
            
            % Compute the CDF and map to output
            cdf = cumsum(hist_vector) / numel(local_region(:));
            output(i, j) = uint8(255 * cdf(image(i, j) + 1));
        end
    end
end

% -----------------------------------------------------------------------------

% Function to close prev images and display next sets of images
function waitForKeyPressAndCloseFigures()
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

% Creates Output Images Directory
function output_dir = createDir()
    output_dir = '../images/output';
    if ~exist(output_dir, 'dir')
       mkdir(output_dir);
    end
end

% -----------------------------------------------------------------------------
% -----------------------------------------------------------------------------
