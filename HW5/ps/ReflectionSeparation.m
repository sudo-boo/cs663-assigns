clear;
close all;

im1 = double(imread('mandrill.png'));
im2 = double(imread('barbara512.png'));
[H,W] = size(im1);

sig1 = 20; sig2 = 60;
[X,Y] = meshgrid(-H/2:H/2-1,-W/2:W/2-1);
FG1 = exp(-(X.^2 + Y.^2)/(2*sig1*sig1)); 
FG2 = exp(-(X.^2 + Y.^2)/(2*sig2*sig2)); 

g1 = im1 + abs((ifft2(FG2.*fftshift(fft2(im2))))) + randn(size(im1))*10;
g2 = im2 + abs((ifft2(FG1.*fftshift(fft2(im1))))) + randn(size(im1))*10;
figure, imshow(g1/max(g1(:)));
figure, imshow(g2/max(g2(:)));

G1 = fftshift(fft2(g1));
G2 = fftshift(fft2(g2));
%FG1(FG1 == 1) = 0.9999;
%FG2(FG2 == 1) = 0.9999;
FG1FG2 = FG1.*FG2; 

eps = 0.01*exp(-(X.^2+Y.^2));
F1 = (G1 - G2.*FG2)./(1+eps-FG1FG2);
F2 = (G2 - G1.*FG1)./(1+eps-FG1FG2);
f1 = abs((ifft2(F1))); 
f2 = abs((ifft2(F2))); 

figure(3),imshow((f1-min(f1(:)))/(max(f1(:))-min(f1(:))));
figure(4),imshow((f2-min(f2(:)))/(max(f2(:))-min(f2(:))));
