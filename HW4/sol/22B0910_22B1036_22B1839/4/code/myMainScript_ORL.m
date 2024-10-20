close all;
clc;

%==============================================================================
%                   ORL DATASET
%==============================================================================

% Parameters
orl_dir = "./../images/ORL/s%d/";
num_subjects = 32;  % First 32 subjects
train_per_subject = 6;
test_per_subject = 4;
image_height = 112;
image_width = 92;
image_size = image_height * image_width;
ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170];  % Values of k

% Predeclare the arrays for efficiency
train_data = zeros(image_size, num_subjects * train_per_subject);
test_data = zeros(image_size, num_subjects * test_per_subject);
train_labels = zeros(1, num_subjects * train_per_subject);
test_labels = zeros(1, num_subjects * test_per_subject);

% Load the images
train_index = 1;
test_index = 1;

for i = 1:num_subjects
    folder = sprintf(orl_dir, i);
    files = dir(fullfile(folder, '*.pgm'));
    
    % Load training images (first 6)
    for j = 1:train_per_subject
        img = imread(fullfile(folder, files(j).name));
        img = double(img);
        
        % Check image dimensions
        [h, w] = size(img);
        if h ~= image_height || w ~= image_width
            error('Image dimensions do not match the expected size of 92x112.');
        end
        
        % Reshape the image into a column vector
        img_vector = reshape(img, [], 1);
        train_data(:, train_index) = img_vector;
        train_labels(train_index) = i;
        train_index = train_index + 1;
    end
    
    % Load testing images (remaining 4)
    for j = train_per_subject + 1:train_per_subject + test_per_subject
        img = imread(fullfile(folder, files(j).name));
        img = double(img);
        
        % Check image dimensions
        [h, w] = size(img);
        if h ~= image_height || w ~= image_width
            error('Image dimensions do not match the expected size of 92x112.');
        end
        
        % Reshape the image into a column vector
        img_vector = reshape(img, [], 1);
        test_data(:, test_index) = img_vector;
        test_labels(test_index) = i;
        test_index = test_index + 1;
    end
end

% Compute mean face and eigenfaces
[mean_face, eig_vec] = compute_eigenfaces_orl(train_data);
train_data_centered = train_data - mean_face;

% Recognition using different k values
fprintf('===<< FACE RECOGNITION ON ORL DATASET >>===\n');
fprintf("Recognition rate for different number of eigenfaces(k) are \n");

recognition_rates = zeros(length(ks), 1);
for idx = 1:length(ks)
    k = ks(idx);
    eig_faces_k = eig_vec(:, 1:k);
    train_proj = eig_faces_k' * train_data_centered;
    num_correct = 0;
    for i = 1:size(test_data, 2)
        test_image = test_data(:, i) - mean_face;
        test_proj = eig_faces_k' * test_image;
        dists = sum((train_proj - test_proj).^2, 1);
        [~, min_idx] = min(dists);
        predicted_label = train_labels(min_idx);
        
        if predicted_label == test_labels(i)
            num_correct = num_correct + 1;
        end
    end
    
    recognition_rate = num_correct / size(test_data, 2);
    recognition_rates(idx) = recognition_rate;
    fprintf('k = %d\t:\t%.2f%%\n', k, recognition_rate * 100);
end

fprintf('\n-----------------------------------------\n\n');

% Plot recognition rates
figure;
plot(ks, recognition_rates * 100, '-o');
xlabel('Number of Eigenfaces (k)');
ylabel('Recognition Rate (%)');
title('Recognition Rate vs Number of Eigenfaces (ORL)');
grid on;

% SVD version (for comparison)
fprintf("SVD Recognition rate for different number of eigenfaces(k) are \n");

[U, S, V] = svd(train_data_centered, 'econ');
recognition_rates_svd = zeros(length(ks), 1);
for idx = 1:length(ks)
    k = ks(idx);
    svd_faces_k = U(:, 1:k);
    train_proj_svd = svd_faces_k' * train_data_centered;
    
    % Test on test data
    num_correct = 0;
    for i = 1:size(test_data, 2)
        test_image = test_data(:, i) - mean_face;
        test_proj_svd = svd_faces_k' * test_image;
        dists_svd = sum((train_proj_svd - test_proj_svd).^2, 1);
        [~, min_idx_svd] = min(dists_svd);
        predicted_label_svd = train_labels(min_idx_svd);
        
        if predicted_label_svd == test_labels(i)
            num_correct = num_correct + 1;
        end
    end
    
    recognition_rate_svd = num_correct / size(test_data, 2);
    recognition_rates_svd(idx) = recognition_rate_svd;
    fprintf('k = %d\t:\t%.2f%%\n', k, recognition_rate_svd * 100);
end

% Plot SVD recognition rates
figure;
plot(ks, recognition_rates_svd * 100, '-o');
xlabel('Number of Singular Vectors (k)');
ylabel('Recognition Rate (%)');
title('SVD Recognition Rate vs Number of Singular Vectors (ORL)');
grid on;


% -------------- Face Reconstruction -----------------

rec_ks = [2, 10, 20, 50, 75, 100, 125, 150, 175];
% Change this index to select a different face
example_face_index = 9;
example_face = train_data(:, example_face_index);
example_face_centered = example_face - mean_face;

figure;
for i = 1:length(rec_ks)
    k = rec_ks(i);
    eig_faces_k = eig_vec(:, 1:k);  % Select top k eigenfaces
    proj_example_face = eig_faces_k' * example_face_centered;
    reconstructed_face = eig_faces_k * proj_example_face + mean_face;

    subplot(3, 3, i);
    imshow(reshape(reconstructed_face, image_height, image_width), []);
    title(sprintf('Reconstruction (k=%d)', k));
end
sgtitle('Face Reconstruction using Different k Values');

fprintf('\n-----------------------------------------\n\n');
fprintf("The Reconstructed Images should be showing in a grid.\n")
fprintf('\n\n===================================================\n\n');



% -------------- Top 25 largest eigenfaces --------------------------
figure;
num_eigenfaces = 25;
tiledlayout(5, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:num_eigenfaces
    eigenface = reshape(eig_vec(:, i), [image_height, image_width]);
    nexttile;
    imshow(eigenface, []);
    title(sprintf('Eigenface %d', i), 'FontSize', 8);
end

sgtitle('Top 25 Eigenfaces');


%==============================================================================
%                   FUNCTIONS
%==============================================================================

function [mean_face, eig_vec] = compute_eigenfaces_orl(train_data)
    mean_face = mean(train_data, 2);
    train_data_centered = train_data - mean_face;
    L = train_data_centered' * train_data_centered;
    [eig_vec, eig_val] = eig(L);
    [~, sorted_indices] = sort(diag(eig_val), 'descend');
    eig_vec_sorted = eig_vec(:, sorted_indices);
    eig_vec = train_data_centered * eig_vec_sorted;  
    eig_vec = normc(eig_vec);  % Normalize eigenvectors
end

