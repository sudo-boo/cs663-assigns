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


output_image = zeros(rows, cols,channels);


invT = inv(affine_matrix);


for r = 1:rows
    for c = 1:cols
        
        out_pixel = [c;r;1];
        
        
        in_pixel = invT * out_pixel;
        
        
        in_x = round(in_pixel(1));
        in_y = round(in_pixel(2));
        
        if in_x >= 1 && in_x <= cols && in_y >= 1 && in_y <= rows
            output_image(r,c,:) = im1(in_y,in_x,:);
        end
    end
end


figure(1);
subplot(1,3,1), imshow(uint8(im1)), title('Original Image 1');
subplot(1,3,2), imshow(uint8(im2)), title('Original Image 2');
subplot(1,3,3), imshow(uint8(output_image)), title('Warped Image 1');

% Adjust the size of the figure and its subplots
h1 = gcf; 
set(h1, 'Position', [100, 500, 1600, 800]);  