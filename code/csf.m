function ret = csf(Kernel)
% CSF.m
% This function finds the contrast sensitivity of a particular set of
% difference-of-Gaussian convolution filters using the frequency response
% of the filters and the Contrast Sensitivity Function. Meant to be used
% with salmap.m to alter the weightings of the oriented edges, according
% to how sensitive the human visual system is to the frequency of those edges.
%
% USAGE: ret = csftKernel)
%
% Kernel -> the smallest convolution kernel in the Gaussian
% pyramid
% ret -> the return value: a vector of the weights to be
% applied to each level of the difference-of-Gaussian
% edge images.
% Author: Roxanne Canosa
% 3/6/03

% Increase the size of each kernel to simulate the effect of convolving a fixed
% size kernel with each level of the Gaussian pyramid
f1 = Kernel;
f2 = imresize(f1,2,'bicubic');
f3 = imresize(f2,2,'bicubic');
f4 = imresize(f3,2,'bicubic');
f5 = imresize(f4,2,'bicubic');
f6 = imresize(f5,2,'bicubic');
f7 = imresize(f6,2,'bicubic');
[r1 c1] = size(f1);
[r2 c2] = size(f2);
[r3 c3] = size(f3);

[r4 c4] = size(f4);
[r5 c5] = size(f5);
[r6 c6] = size(f6);
[r7 c7] = size(f7);

% Bring each kernel up to the size of the largest kernel
f1_new = zeros(r7,c7);
f1_new(r7/2-r1/2+1:r7/2+r1/2,c7/2-c1/2+1:c7/2+c1/2) = f1;

f2_new = zeros(r7,c7);
f2_new(r7/2-r2/2+1:r7/2+r2/2,c7/2-c2/2+1:c7/2+c2/2) = f2;

f3_new = zeros(r7,c7);
f3_new(r7/2-r3/2+1:r7/2+r3/2,c7/2-c3/2+1:c7/2+c3/2) = f3;

f4_new = zeros(r7,c7);
f4_new(r7/2-r4/2+1:r7/2+r4/2,c7/2-c4/2+1:c7/2+c4/2) = f4;

f5_new = zeros(r7,c7);
f5_new(r7/2-r5/2+1:r7/2+r5/2,c7/2-c5/2+1:c7/2+c5/2) = f5;

f6_new = zeros(r7,c7);
f6_new(r7/2-r6/2+1:r7/2+r6/2,c7/2-c6/2+1:c7/2+c6/2) = f6;

f7_new = f7;

% Normalize the new Gaussian kernels so that the total area under each curve = 1
f1_new = f1_new / sum(sum(f1_new));
f2_new = f2_new / sum(sum(f2_new));
f3_new = f3_new / sum(sum(f3_new));
f4_new = f4_new / sum(sum(f4_new));
f5_new = f5_new / sum(sum(f5_new));
f6_new = f6_new / sum(sum(f6_new));
f7_new = f7_new / sum(sum(f7_new));

% Perform the FFT and shift so that the DC component is in the center instead of
% in the upper left hand comer. 5 1 2 samples.
F1 = fftshift(fft2(f1_new,512,512));
F2 = fftshift(fft2(f2_new,512,512));
F3 = fftshift(fft2(f3_new,512,512));
F4 = fftshift(fft2(f4_new,512,512));
F5 = fftshift(fft2(f5_new,512,512));
F6 = fftshift(fft2(f6_new,512,512));
F7 = fftshift(fft2(f7_new,512,512));

figure(1), subplot( 1,7,1), imshow(f1), title('fl:5x5')
figure(1), subplot( 1,7,2), imshow(f2), title('f2: 10x10')
figure(1), subplot( 1,7,3), imshow(f3), title('f3: 20x20')
figure(1), subplot( 1,7,4), imshow(f4), title('f4:40x40')
figure(1), subplot( 1,7,5), imshow(f5), title('f5: 80x80')
figure(1), subplot( 1,7,6), imshow(f6), title('f6: 160x1 60')
figure(1), subplot( 1,7,7), imshow(f7), title('f7:320x320')

figure(2),subplot( 1,7,1),imshow(log(abs(F1)),[-1 5]),title('FT(fl)')
figure(2),subplot( 1,7,2),imshow(log(abs(F2)),[-1 5]),title('FT(f2)')
figure(2),subplot( 1,7,3),imshow(log(abs(F3)),[-1 5]),title('FT(f3)')
figure(2),subplot( 1,7,4),imshow(log(abs(F4)),[-1 5]),title('FT(f4)')
figure(2),subplot( 1,7,5),imshow(log(abs(F5)),[-1 5]),title('FT(f5)')
figure(2),subplot( 1,7,6),imshow(log(abs(F6)),[-1 5]),title('FT(f6)')
figure(2),subplot( 1,7,7),imshow(log(abs(F7)),[-1 5]),title('FT(f7)')

% Find the difference between each of the kernel responses
F12 = F1 - F2;
F23 = F2 - F3;
F34 = F3 - F4;
F45 = F4 - F5;
F56 = F5 - F6;
F67 = F6 - F7;

figure(3), subplot( 1,6,1), imshow(log((F12)),[-1 5]),title('Fl-F2')
figure(3), subplot( 1,6,2), imshow(log((F23)),[-1 5]),title('F2-F3')
figure(3), subplot( 1,6,3), imshow(log((F34)),[-1 5]),title('F3-F4')
figure(3), subplot( 1,6,4), imshow(log((F45)),[-1 5]),title('F4-F5')
figure(3), subplot( 1,6,5), imshow(log((F56)),[-1 5]),title('F5-F6')
figure(3), subplot( 1,6,6), imshow(log((F67)),[-1 5]),title('F6-F7')

% Plot a ID curve of the frequency response of each bandpass filter
x = linspace( 1,256,256);
figure(4), subplot(2,3,1),plot(x,F12(256,257:512)), axis([1 256 0 1]),title('Fl-F2')
figure(4), subplot(2,3,2),plot(x,F23(256,257:512)), axis([1 256 0 1]),title('F2-F3')
figure(4), subplot(2,3,3),plot(x,F34(256,257:512)), axis([1 256 0 1]),title('F3-F4')
figure(4), subplot(2,3,4),plot(x,F45(256,257:512)), axis([1 256 0 1]),title('F4-F5')
figure(4), subplot(2,3,5),plot(x,F56(256,257:512)), axis([1 256 0 1]),title('F5-F6')
figure(4), subplot(2,3,6),plot(x,F67(256,257:512)), axis([1 256 0 1]),title('F6-F7')

% Convert the xaxis of the frequency plot to cycles per degree. Do this by
% finding fmax, where fmax = 1/(2 * delta_x), and delta_x is the sampling
% distance in the spatial domain. Scale the xaxis from 0 to fmax, where fmax
% is in cycles per degree.
width_of_screen_in_degrees = 60;
width_of_screen_in_pixels = 1280;
delta_x = width_of_screen_in_degrees / width_of_screen_in_pixels;
fmax = 1 / (2.*delta_x);
fmin = 0;

% Get the array indices of the maximum frequency response
F12max_index = find(F12(256,257:512)==max(F12(256,257:512)));
F23max_index = find(F23(256,257:512)==max(F23(256,257:512)));
F34max_index = find(F34(256,257:512)==max(F34(256,257:512)));
F45max_index = find(F45(256,257:512)==max(F45(256,257:512)));
F56max_index = find(F56(256,257:512)==max(F56(256,257:512)));
F67max_index = find(F67(256,257:512)==max(F67(256,257:512)));

% Find the equation of the line that converts from array indices to frequency
% The result, y, is frequency in cycles per degree.
m = (fmax-fmin) / (256-1);
b = -m;
x = [F12max_index F23max_index F34max_index F45max_index F56max_index F67max_index];
y = m.*x + b;
f=y;

% CSF function from J.L. Mannos, D.J. Sakrison, "The effects of a visual fidelty
% criterion on the encoding of images", IEEE Transactions on Information Theory,
% vol. 20, number 4, pp. 525-535, 1974.
% First, plot the CSF, if so desired.
x = linspace( 1,60,60);
CSF = 2.6.*(0.0192 + 0.114.*x).*exp(-(0.114.*x).^(1.1));
figure(5), plot(x,CSF), axis([0 60 0 1]), title('CSF function')
% Find the weights to apply to the bandpass-filtered edge images of the Laplacian
% cube by finding the value from the CSF function for the corresponding frequency
% in cycles per degree
ret = 2.6.*(0.0192 + 0.114.*f).*exp(-(0.114.*f).^(1.1));
