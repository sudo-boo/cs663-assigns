im1 = imread("CS663/goi1.jpg");
im2 = imread("CS663/goi2_downsampled.jpg");

im1 = double(im1);
im2 = double(im2);

n = 12;

A = []; 
b = []; 

x1 = importdata("x1.mat");
y1 = importdata("y1.mat");
x2 = importdata("x2.mat");
y2 = importdata("y2.mat");

for i = 1:n
    
    A = [A;
         x1(i), y1(i), 1, 0, 0, 0;
         0, 0, 0, x1(i), y1(i), 1];
     
    
    b = [b; x2(i); y2(i)];
end


T = A \ b;


affine_matrix = [T(1) T(2) T(3);
                 T(4) T(5) T(6);
                 0 0 1];


disp('Affine Transformation Matrix:');
disp(affine_matrix);

[rows, cols, channels] = size(im2);


output_image = zeros(rows, cols, channels);


invT = inv(affine_matrix);

% Apply bilinear interpolation
for r = 1:rows
    for c = 1:cols
        
        out_pixel = [c; r; 1];
        
        in_pixel = invT * out_pixel;
        
        in_x = in_pixel(1);
        in_y = in_pixel(2);
        
        x1 = floor(in_x);
        x2 = ceil(in_x);
        y1 = floor(in_y);
        y2 = ceil(in_y);
        
        if x1 >= 1 && x2 <= cols && y1 >= 1 && y2 <= rows
            
            dx = in_x - x1;
            dy = in_y - y1;
            
            I11 = im1(y1, x1, :);
            I12 = im1(y2, x1, :);
            I21 = im1(y1, x2, :);
            I22 = im1(y2, x2, :);
            
            output_image(r, c, :) = (1 - dx) * (1 - dy) * I11 + ...
                                    (1 - dx) * dy * I12 + ...
                                    dx * (1 - dy) * I21 + ...
                                    dx * dy * I22;
        end
    end
end


figure(2);
subplot(1,3,1), imshow(uint8(im1)), title('Original Image 1');
subplot(1,3,2), imshow(uint8(im2)), title('Original Image 2');
subplot(1,3,3), imshow(uint8(output_image)), title('Warped Image 2');

h2 = gcf;
set(h2, 'Position', [100, 100, 1600, 800]);  