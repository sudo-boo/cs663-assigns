
% ====== (a) =======

% Reading the images and storing them as double
J1 = double(imread('T1.jpg'));
J2 = double(imread('T2.jpg'));

% Rotating the J2
theta_initial = 28.5;
J3 = imrotate(J2, theta_initial, 'bilinear', 'crop');

% By default void pixels are set to 0 while rotation.
% Hence, not necessary to explicitly set it to 0.
% J3(J3 == 0) = 0;


% Display the original and rotated images
figure('Position', [100, 100, 1000, 500]);
subplot(1, 2, 1);
imshow(uint8(J2));
title('Original Image J2', 'FontSize', 16);

subplot(1, 2, 2);
imshow(uint8(J3));
title('Rotated Image J3 (28.5 degrees)', 'FontSize', 16);



% ====== (b) =======

range = -45:1:45;
ncc_values = zeros(size(range));
je_values = zeros(size(range));
qmi_values = zeros(size(range));

for i = 1:length(range)
    theta = range(i);
    J4 = imrotate(J3, theta, 'bilinear', 'crop');
    
    % Compute
    ncc_values(i) = compute_ncc(J1, J4);
    je_values(i) = compute_je(J1, J4, 10);
    qmi_values(i) = compute_qmi(J1, J4, 10);
end



% ====== (c) =======

% Plot the results
figure('Position', [100, 100, 800, 900]);

subplot(3, 1, 1);
plot(range, ncc_values, 'LineWidth', 2);
title('NCC', 'FontSize', 16);
xlabel('Theta (deg)', 'FontSize', 14);
ylabel('NCC', 'FontSize', 14);

subplot(3, 1, 2);
plot(range, je_values, 'LineWidth', 2);
title('Joint Entropy', 'FontSize', 16);
xlabel('Theta (deg)', 'FontSize', 14);
ylabel('JE', 'FontSize', 14);

subplot(3, 1, 3);
plot(range, qmi_values, 'LineWidth', 2);
title('QMI', 'FontSize', 16);
xlabel('Theta (deg)', 'FontSize', 14);
ylabel('QMI', 'FontSize', 14);



% ====== (d) =======

% minimum NCC value
[min_ncc, idx_ncc] = min(ncc_values);
optimal_angle_ncc = range(idx_ncc);

% minimum JE value
[min_je, idx_je] = min(je_values);
optimal_angle_je = range(idx_je);

% minimum QMI value
[min_qmi, idx_qmi] = max(qmi_values);
optimal_angle_qmi = range(idx_qmi);

% Display the results
fprintf('Optimal angle based on NCC: %.1f degrees\n', optimal_angle_ncc);
fprintf('Optimal angle based on JE: %.1f degrees\n', optimal_angle_je);
fprintf('Optimal angle based on QMI: %.1f degrees\n', optimal_angle_qmi);



% ====== (e) =======

J4 = imrotate(J1, optimal_angle_je, 'bilinear', 'crop');
J5 = imrotate(J2, optimal_angle_je, 'bilinear', 'crop');

bin_width = 10;
joint_hist = compute_joint_histogram(J4, J5, bin_width);

figure('Position', [100, 100, 1000, 800]);
imagesc(joint_hist);
colorbar;
title('Joint Histogram between J1 and J4', 'FontSize', 16);
xlabel('Intensity of J4', 'FontSize', 14);
ylabel('Intensity of J1', 'FontSize', 14);

% % Subplot 1: Display J1
% figure;
% subplot(1, 2, 1);
% imshow(uint8(J1));
% title('Image J1', 'FontSize', 16);
% 
% % Subplot 2: Display J4
% subplot(1, 2, 2);
% imshow(uint8(J4));
% title(['Image J4 (Rotated by ', num2str(theta_selected), ' degrees)'], 'FontSize', 16);



% ==========================================
% ========== Declared functions ============


% Calculate Normalized Cross-Correlation (NCC)
function ncc = compute_ncc(I1, I2)
    numerator = sum(sum((I1 - mean(I1(:))) .* (I2 - mean(I2(:)))));
    denominator = sqrt(sum(sum((I1 - mean(I1(:))).^2)) * sum(sum((I2 - mean(I2(:))).^2)));
    ncc = numerator / denominator;
end


% Calculate Joint Entropy (JE)
function je = compute_je(I1, I2, bin_width)
    joint_hist = compute_joint_histogram(I1, I2, bin_width);
    je = -sum(joint_hist(:) .* log(joint_hist(:) + 0.0000001));
end


% Calculate Quadratic Mutual Information (QMI)
function qmi = compute_qmi(I1, I2, bin_width)
    joint_hist = compute_joint_histogram(I1, I2, bin_width);
    marginal_hist_I1 = sum(joint_hist, 2); % Integrate along I2
    marginal_hist_I2 = sum(joint_hist, 1); % Integrate along I1
    qmi = sum(sum((joint_hist - marginal_hist_I1 * marginal_hist_I2).^2));
end


% Compute Joint Histogram
function joint_hist = compute_joint_histogram(I1, I2, bin_width)
    max_val = max(max(I1(:)), max(I2(:)));
    num_bins = ceil(max_val / bin_width);
    joint_hist = zeros(num_bins, num_bins);

    for i = 1:size(I1, 1)
        for j = 1:size(I1, 2)
            bin1 = floor(I1(i, j) / bin_width) + 1;
            bin2 = floor(I2(i, j) / bin_width) + 1;
            if bin1 > 0 && bin2 > 0
                joint_hist(bin1, bin2) = joint_hist(bin1, bin2) + 1;
            end
        end
    end
    joint_hist = joint_hist / sum(joint_hist(:));
end