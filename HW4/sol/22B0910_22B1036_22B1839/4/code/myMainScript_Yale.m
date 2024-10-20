close all;
clc;

%==============================================================================
%                   YALE DATASET
%==============================================================================

% Parameters
yale_dir = "./../images/Yale/yaleB%02d/";
num_subjects = 38;          % Number of subjects
train_per_subject = 40;     % Number of training images per subject
test_per_subject = 24;      % Number of testing images per subject
image_height = 192;
image_width = 168;
image_size = image_height * image_width;
ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];  % Values of k

% Modularized image loading
[train_data, test_data, train_labels, test_labels] = load_images_yale(yale_dir, num_subjects, train_per_subject, test_per_subject, image_height, image_width);

% Compute mean face and eigenfaces
[mean_face, eig_vec] = compute_eigenfaces_yale(train_data);
train_data_centered = train_data - mean_face;

% Recognition using different k values
fprintf("\nRecognition rate for different number of eigenfaces(k) are \n");
recognition_rates = zeros(length(ks), 1);
for idx = 1:length(ks)
    k = ks(idx);
    eig_faces_k = eig_vec(:, end-k+1:end);
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
    if k ~= 1000
        fprintf('k = %d\t\t:\t%.2f%%\n', k, recognition_rate*100);
    else
        fprintf('k = %d\t:\t%.2f%%\n', k, recognition_rate*100);
    end
end

fprintf('\n-----------------------------------------');

% Plot recognition rates
figure;
plot(ks, recognition_rates * 100, '-o');
xlabel('Number of Eigenfaces (k)');
ylabel('Recognition Rate (%)');
title(sprintf('Recognition Rate vs Number of Eigenfaces (Yale) for %d Subjects', num_subjects));
grid on;

fprintf('\n===================================================\n\n');


% Recognition excluding the top 3 eigenvectors
fprintf("\nRecognition rate excluding the top 3 eigenfaces (Yale) for different k values:\n");

recognition_rates_exclude_top_3 = zeros(length(ks), 1);
for idx = 1:length(ks)
    k = ks(idx);
    
    % Exclude the top 3 eigenfaces and select the next k
    eig_faces_k = eig_vec(:, end-(k+3)+1:end-3);
    
    % Project training data (after excluding the top 3 eigenfaces)
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
    recognition_rates_exclude_top_3(idx) = recognition_rate;
    if k ~= 1000
        fprintf('k = %d\t\t:\t%.2f%%\n', k, recognition_rate * 100);
    else
        fprintf('k = %d\t:\t%.2f%%\n', k, recognition_rate * 100);
    end
end

% Plot recognition rates excluding top 3 eigenfaces
figure;
plot(ks, recognition_rates_exclude_top_3 * 100, '-o');
xlabel('Number of Eigenfaces (k)');
ylabel('Recognition Rate (%)');
title('Recognition Rate (Excluding Top 3 Eigenfaces) vs Number of Eigenfaces (Yale)');
grid on;


%==============================================================================
%                   FUNCTIONS
%==============================================================================

function [mean_face, eig_vec] = compute_eigenfaces_yale(train_data)
    mean_face = mean(train_data, 2);
    train_data_centered = train_data - mean_face;
    L = train_data_centered' * train_data_centered;
    [eig_vec, ~] = eig(L);
    eig_vec = train_data_centered * eig_vec;
    eig_vec = normc(eig_vec);
end

%------------------------------------------------------

% Modularized image loading function for Yale dataset
function [train_data, test_data, train_labels, test_labels] = load_images_yale(yale_dir, num_subjects, train_per_subject, test_per_subject, image_height, image_width)
    image_size = image_height * image_width;
    train_data = zeros(image_size, num_subjects * train_per_subject);
    test_data = zeros(image_size, num_subjects * test_per_subject);
    train_labels = zeros(1, num_subjects * train_per_subject);
    test_labels = zeros(1, num_subjects * test_per_subject);
    
    fprintf('===<< FACE RECOGNITION ON YALE DATASET >>===\n\n');
    fprintf("Evaluating for %d subjects\n\n", num_subjects);
    fprintf("Loading Images....\n");

    train_index = 1; test_index = 1;

    for i = 1:num_subjects
        folder = sprintf(yale_dir, i);
        files = dir(fullfile(folder, '*.pgm'));
        
        num_images = length(files);
        if num_images < train_per_subject + test_per_subject
            fprintf('Warning: Not enough images in folder %s. (%d / %d)\n', folder, num_images, train_per_subject + test_per_subject);
            actual_train_per_subject = min(train_per_subject, num_images);
            actual_test_per_subject = max(0, num_images - actual_train_per_subject);
        else
            actual_train_per_subject = train_per_subject;
            actual_test_per_subject = test_per_subject;
        end
        
        % Load training images
        for j = 1:actual_train_per_subject
            img = imread(fullfile(folder, files(j).name));
            img = double(img);
            [h, w] = size(img);
            if h ~= image_height || w ~= image_width
                error('Image dimensions do not match the expected size of %dx%d.', image_height, image_width);
            end
            img_vector = reshape(img, [], 1);
            train_data(:, train_index) = img_vector;
            train_labels(train_index) = i;
            train_index = train_index + 1;
        end
        
        % Load testing images
        for j = actual_train_per_subject + 1:actual_train_per_subject + actual_test_per_subject
            img = imread(fullfile(folder, files(j).name));
            img = double(img);
            [h, w] = size(img);
            if h ~= image_height || w ~= image_width
                error('Image dimensions do not match the expected size of %dx%d.', image_height, image_width);
            end
            img_vector = reshape(img, [], 1);
            test_data(:, test_index) = img_vector;
            test_labels(test_index) = i;
            test_index = test_index + 1;
        end
    end
end
