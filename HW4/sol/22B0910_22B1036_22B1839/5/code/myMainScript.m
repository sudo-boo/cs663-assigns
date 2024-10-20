close all;
clc;

%============================================================================== 
%                   ORL DATASET : EXTENDED
%============================================================================== 

% Parameters 
orl_dir = "./../images/ORL/s%d/";           % Path to dataset 
num_subjects_train = 32;                    % First 32 subjects used for training 
num_subjects_test_extra = 8;                % Last 8 subjects 
train_per_subject = 6;                      % Number of training images per subject 
test_per_subject = 4;                       % Number of testing images per subject 
image_height = 112;                         % Image height 
image_width = 92;                           % Image width 
image_size = image_height * image_width;    % Total image size 
k = 50;                                     % No. of EFs (something considerable)

% Load training data (First 32 subjects, 6 images each) 
train_data = zeros(image_size, num_subjects_train * train_per_subject); 
train_labels = zeros(1, num_subjects_train * train_per_subject);  
train_index = 1; 

for i = 1:num_subjects_train 
    folder = sprintf(orl_dir, i); 
    files = dir(fullfile(folder, '*.pgm')); 
    
    % Load training images (first 6) 
    for j = 1:train_per_subject 
        img = imread(fullfile(folder, files(j).name)); 
        img = double(img); 
        img_vector = reshape(img, [], 1); 
        train_data(:, train_index) = img_vector; 
        train_labels(train_index) = i;  % Store the label 
        train_index = train_index + 1; 
    end 
end 

% Compute mean face and eigenfaces 
[mean_face, eig_vec] = compute_eigenfaces_orl(train_data); 
train_data_centered = train_data - mean_face; 

% Project the training data onto the eigenfaces 
eig_faces_k = eig_vec(:, 1:k); 
train_proj = eig_faces_k' * train_data_centered; 

% Calculate a threshold from the training data 
distances_train = pdist2(train_proj', train_proj'); 
threshold = mean(distances_train(:)) + 8000*std(distances_train(:)); % Adjust threshold as needed

fprintf("Threshold set to: %.4f\n", threshold); 

% 1. Test on the extra subjects (8 subjects not in training set) 
test_data_extra = zeros(image_size, num_subjects_test_extra * test_per_subject); 
test_labels_extra = zeros(1, num_subjects_test_extra * test_per_subject); 
test_index = 1; 

for i = num_subjects_train + 1:num_subjects_train + num_subjects_test_extra 
    folder = sprintf(orl_dir, i); 
    files = dir(fullfile(folder, '*.pgm')); 
    
    for j = 1:test_per_subject 
        img = imread(fullfile(folder, files(j).name)); 
        img = double(img); 
        test_data_extra(:, test_index) = reshape(img, [], 1); 
        test_labels_extra(test_index) = i;  % Store the label 
        test_index = test_index + 1; 
    end 
end 

fprintf("\n============ BEFORE ============\n\n");
fprintf('Results for the extra subjects (unknown identities):\n'); 
fprintf('False positives: %d\n', num_false_positives_extra); 
fprintf('False negatives: %d\n', num_false_negatives_extra); 

fprintf('\n----------------------------------------------------'); 
fprintf("\n\tIf 0, 0 for both, it means our system\n\tis misclassifying the unknowns as knowns.!!\n")
fprintf('----------------------------------------------------\n'); 

fprintf("\n\n============ AFTER ==============\n");

fprintf('Results for the extra subjects (unknown identities):\n'); 


% Initialize false positive and negative counters for extra subjects
num_false_positives_extra = 0;  % Initialize these counters
num_false_negatives_extra = 0; 

% 2. Test on the known subjects (32 subjects from training set) 
test_data_known = zeros(image_size, num_subjects_train * test_per_subject); 
test_labels_known = zeros(1, num_subjects_train * test_per_subject); 
test_index = 1; 

for i = 1:num_subjects_train 
    folder = sprintf(orl_dir, i); 
    files = dir(fullfile(folder, '*.pgm')); 
    
    for j = train_per_subject + 1:train_per_subject + test_per_subject 
        img = imread(fullfile(folder, files(j).name)); 
        img = double(img); 
        test_data_known(:, test_index) = reshape(img, [], 1); 
        test_labels_known(test_index) = i;  % Store the label 
        test_index = test_index + 1; 
    end 
end 

% Test recognition on known subjects 
[num_correct_known, num_false_positives_known, num_false_negatives_known] = ...
    test_recognition(test_data_known, test_labels_known, train_proj, eig_faces_k, mean_face, threshold, train_labels); 


fprintf('\n\n----------------------------------------------------\n\n'); 
fprintf('Results for the known subjects (correct identities):\n'); 
fprintf('Correctly identified: %d\n', num_correct_known);
fprintf('Recognition Rate: %.2f%%\n', (num_correct_known / (num_subjects_train * test_per_subject)) * 100);  
fprintf('False positives: %d\n', num_false_positives_known); 
fprintf('False negatives: %d\n', num_false_negatives_known); 


fprintf('\n\n=====================================================\n\n'); 

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

function [num_correct, false_positives, false_negatives] = test_recognition(test_data, test_labels, train_proj, eig_faces_k, mean_face, threshold, train_labels) 
    num_correct = 0; 
    false_positives = 0; 
    false_negatives = 0; 
    num_tests = size(test_data, 2); 
    
    for i = 1:num_tests 
        test_image = test_data(:, i) - mean_face; 
        test_proj = eig_faces_k' * test_image; 
        dists = sum((train_proj - test_proj).^2, 1); 
        [~, min_idx] = min(dists); 
        predicted_label = train_labels(min_idx); 

        if dists(min_idx) > threshold 
            predicted_label = -1;  % Unknown identity 
        end 

        % Update counts based on prediction
        if predicted_label == test_labels(i) 
            num_correct = num_correct + 1; 
        elseif predicted_label ~= -1 
            false_positives = false_positives + 1; % Misclassified known identity 
        else 
            false_negatives = false_negatives + 1; % Known identity not recognized 
        end 
    end 
    
    recognition_rate = num_correct / num_tests; 
    fprintf('Recognition rate: %.2f%%\n', recognition_rate * 100); 
    fprintf('False Positives: %d\n', false_positives); 
    fprintf('False Negatives: %d\n', false_negatives); 
end 
