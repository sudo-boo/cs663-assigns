im1 = imread("CS663/goi1.jpg");
im2 = imread("CS663/goi2_downsampled.jpg");

im1 = double(im1);
im2 = double(im2);

n = 12;
x1 = zeros(1, n); y1 = zeros(1, n);
x2 = zeros(1, n); y2 = zeros(1, n);


for i = 1:n
    figure(1);
    imshow(im1/255);
    title('Select point on Image 1');
    [x1(i), y1(i)] = ginput(1);
    
    figure(2);
    imshow(im2 / 255); 
    title('Select corresponding point on Image 2');
    [x2(i), y2(i)] = ginput(1);


    disp(x1(i));
    fprintf(",");
    disp(y1(i));
    fprintf("\n");
    disp(x2(i));
    fprintf(",");
    disp(y2(i));
    fprintf("\n-----------\n");

end


close all;


