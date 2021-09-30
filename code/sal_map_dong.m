function y = sal_map(image_name)
% SALMAP.M
% This program drives the creation of the Conspicuity Map. It augments the
% Parkhurst algorithm with rod and cone responses instead of using raw RGB
% digital counts, and uses seven levels instead of nine. Also, this algorithm
% does the spatial processing on the achromatic signal only, while separating out the
% luminance and chrominance information for higher level processing. It also
% uses a proto-object locator to simulate figure/ground segmentation of
% perceptual organization.
%
% USAGE: y = sal_map('imagename.jpg');
%
% Ref: Roxanne Canosa 2/25/03 MODELING SELECTIVE PERCEPTION OF COMPLEX,NATURAL SCENES
close All, clear All
tic;
% Read in an image to get the size

% I = imread('Herman_gird_colored.jpg');
I = imread('fruits.jpg');

ysize = size(I,1);
xsize = size(I,2);

% Transformation matrix Ml to go from XYZ to cone and rod responses.......
    M1 = [ 0.3897 0.6890 -0.0787; -0.2298 1.1834 0.0464; 0 0 1];
% Transformation matrix M2 to go from cone and rod responses to A_C1_C2...
    M2 = [2.0 1.0 0.05; 1.0 -1.09 0.09; 0.11 0.11 -0.22];
% Transform to XYZ tristimulus values.....................................
    XYZ = rgb2xyz(I);

% Convert to cone and rod response values by calculating the linear
% transform of the XYZ values
XYZ= reshape(XYZ, ysize*xsize, 3);
old_LMS = XYZ*M1';
old_Rods = XYZ * [-0.702 1.039 .433]';
long=   old_LMS(:,1);
medium= old_LMS(:,2);
short=  old_LMS(:,3);
rods=   old_Rods;

% Normalize to the maximum and minimum cone or rod responses for any RGB inputs
long_max =  51;
medium_max =54.2;
short_max = 58.4;
rod_max =   48.95;
long_min =  0.4275;
medium_min =0.4458;
short_min = 0.4120;
rod_min =   0;
a_max=  165.69;
a_min=  1.37;
Cl_max= 9.8;
Cl_min=-12.65;
C2_max= 9.11;
C2_min=-10.39;

% Opponent Color Processing.........................................
%old_LMS(:,l) = long;
%old_LMS(:,2) = medium;
%old_LMS(:,3) = short;
old_AC1C2 = old_LMS * M2';
a_total = old_AC1C2(:,1) + rods/7;  % total achromatic signal
r_g = old_AC1C2(:,2);               % red - green signal
y_b = old_AC1C2(:,3);               % yellow - blue signal

% Reshape rods, cones, and AC1C2 to original size of image
long =   reshape(long,ysize,xsize);
medium = reshape(medium,ysize,xsize);
short =  reshape(short,ysize,xsize);
rods =   reshape(old_Rods,ysize,xsize);
valid = rods > 0;
rods = rods.*valid;
a_total = reshape(a_total,ysize,xsize);
r_g = reshape(r_g,ysize,xsize);
y_b = reshape(y_b,ysize,xsize);

% Calibrate rod and cone responses to be from 0 to 1 , where 0 corresponds to
% min value and 1 corresponds to max value over all images.
% long = (long - longmin) / (longmax - longmin);
% medium = (medium - mediummin) / (medium_max - mediumjnin);
% short = (short - shortjnin) / (short_max - short_min);
% rods = (rods - rodmin) / (rodmax - rodmin);

% Normalize achromatic channel, and 2 chromatic channels to be from 0 to 1,
% where 0 corresponds to min value and 1 corresponds to max value over all images.
a_total = (a_total - a_min) / (a_max - a_min);
r_g = (r_g - Cl_min) / (Cl_max - Cl_min);
y_b = (y_b - C2_min) / (C2_max - C2_min);
r_g = abs(2.*r_g - 1);
y_b = abs(2.*y_b - 1);

% Make the Color Maps and the Achromatic Map...............................
ColMap= sqrt(r_g.^2 + y_b.^2);
InMap = a_total;
figure, subplot(2,2,1), imshow(long), title('Long')
 subplot(2,2,2), imshow(medium), title('Medium')
 subplot(2,2,3), imshow(short), title('Short')
 subplot(2,2,4), imshow(rods), title('Rods')
figure, subplot(2,3,1), imshow(I), title('Image')
 subplot(2,3,2), imshow(rgb2gray(I)), title('RGB ave')
 subplot(2,3,3), imshow(InMap./max(max(InMap))), title('A')
 subplot(2,3,4), imshow(r_g./max(max(r_g))), title('R or G')
 subplot(2,3,5), imshow(y_b./max(max(y_b))), title('Y or B')
colormap(gray)
clear long medium short rods XYZ old_LMS old_AClC2 oldrods r_g y_b valid

%% CREATE THE GAUSSIAN PYRAMID FOR THE ORIENATED EDGE CALCULATION
% Calculate the levels of the multi-resolution pyramid using the achromatic image
% Do this by first shrinking the image, then blowing it back up again using
% Gaussian interpolation.
[rows0, cols0] = size(a_total);
Lev0 = a_total;             % Level 0
Lev1 = imresize(Lev0, .5);
Lev2 = imresize(Lev1, .5);
Lev3 = imresize(Lev2, .5);
Lev4 = imresize(Lev3, .5);
Lev5 = imresize(Lev4, .5);
Lev6 = imresize(Lev5, .5);  %Level 6

% Create the kernel.......................................................
Kernel = fspecial('gaussian',2);

Lev1 = imresize(Lev1, [rows0 cols0], 'bicubic', Kernel);
Lev2 = imresize(Lev2, [rows0 cols0], 'bicubic', Kernel);
Lev3 = imresize(Lev3, [rows0 cols0], 'bicubic', Kernel);
Lev4 = imresize(Lev4, [rows0 cols0], 'bicubic', Kernel);
Lev5 = imresize(Lev5, [rows0 cols0], 'bicubic', Kernel);
Lev6 = imresize(Lev6, [rows0 cols0], 'bicubic', Kernel);

% Show the Gaussian pyramid
figure, subplot(3,3,1), imshow(Lev0), title('Level 0')
 subplot(3,3,2), imshow(Lev1), title('Level 1')
 subplot(3,3,3), imshow(Lev2), title('Level 2')
 subplot(3,3,4), imshow(Lev3), title('Level 3')
 subplot(3,3,5), imshow(Lev4), title('Level 4')
 subplot(3,3,6), imshow(Lev5), title('Level 5')
 subplot(3,3,7), imshow(Lev6), title('Level 6')
%%
% Create the Laplacian cube.................................................
% Simulate center-surround organization and lateral inhibition by subtracting
% the lower-res images from the higher-res images
L0_L1 = abs(Lev0 - Lev1);
clear Lev0;
L1_L2 = abs(Lev1 - Lev2);
clear Lev1;
L2_L3 = abs(Lev2 - Lev3);
clear Lev2;
L3_L4 = abs(Lev3 - Lev4);
clear Lev3;
L4_L5 = abs(Lev4 - Lev5);
clear Lev4;
L5_L6 = abs(Lev5 - Lev6);
clear Lev5 Lev6;

% Show the Laplacian edges before weighting by the CSF......................
figure, subplot(3,3,1), imshow(L0_L1), title('L0-Ll')
 subplot(3,3,2), imshow(L1_L2), title('Ll-L2')
 subplot(3,3,3), imshow(L2_L3), title('L2-L3')
 subplot(3,3,4), imshow(L3_L4), title('L3-L4')
 subplot(3,3,5), imshow(L4_L5), title('L4-L5')
 subplot(3,3,6), imshow(L5_L6), title('L5-L6')

% Weight the Laplacian edges by the CSF function
edge_weights = csf(Kernel);
L0_L1 = L0_L1.*edge_weights(1);
L1_L2 = L1_L2.*edge_weights(2);
L2_L3 = L2_L3.*edge_weights(3);
L3_L4 = L3_L4.*edge_weights(4);
L4_L5 = L4_L5.*edge_weights(5);
L5_L6 = L5_L6.*edge_weights(6);
% Show the Laplacian edges after weighting by the CSF
figure, subplot(3,3,1), imshow(L0_L1), title('L0-L1w')
 subplot(3,3,2), imshow(L1_L2), title('L1-L2w')
 subplot(3,3,3), imshow(L2_L3), title('L2-L3w')
 subplot(3,3,4), imshow(L3_L4), title('L3-L4w')
 subplot(3,3,5), imshow(L4_L5), title('L4-L5w')
 subplot(3,3,6), imshow(L5_L6), title('L5-L6w')

%% CREATE THE ORIENTATED EDGE LAPLACIAN CUBES
% First, generate the Gabor filters
Gabor0 =  gabor(10/2,0,'SpatialFrequencyBandwidth',1.0, 'SpatialAspectRatio', 0.6);  % 0 degrees, 8 samples, 2 cycles
Gabor45=  gabor(10/2,45,'SpatialFrequencyBandwidth',1.0, 'SpatialAspectRatio', 0.6); % 45 degrees
Gabor90=  gabor(10/2,90,'SpatialFrequencyBandwidth',1.0, 'SpatialAspectRatio', 0.6); % 90 degrees
Gabor135= gabor(10/2,135,'SpatialFrequencyBandwidth',1.0, 'SpatialAspectRatio', 0.6);% 135 degrees
f0  = imresize(Gabor0.SpatialKernel, [15 15], 'bicubic');
f45 = imresize(Gabor45.SpatialKernel, [15 15], 'bicubic');
f90 = imresize(Gabor90.SpatialKernel, [15 15], 'bicubic');
f135= imresize(Gabor135.SpatialKernel, [15 15], 'bicubic');
figure,
    subplot(2,2,1), imagesc(real(Gabor0.SpatialKernel)), axis xy, axis square, title('0 filt')
    subplot(2,2,2), imagesc(real(Gabor45.SpatialKernel)), axis xy, axis square, title('45 filt')
    subplot(2,2,3), imagesc(real(Gabor90.SpatialKernel)), axis xy, axis square, title('90 filt')
    subplot(2,2,4), imagesc(real(Gabor135.SpatialKernel)), axis xy, axis square, title('135 filt')

% Start with the 0 degrees oriented filter
f0 = Gabor0.SpatialKernel;
G0_L0_L1 = conv2(L0_L1,f0,'valid');
G0_L1_L2 = conv2(L1_L2,f0,'valid');
G0_L2_L3 = conv2(L2_L3,f0,'valid');
G0_L3_L4 = conv2(L3_L4,f0,'valid');
G0_L4_L5 = conv2(L4_L5,f0,'valid');
G0_L5_L6 = conv2(L5_L6,f0,'valid');

G0_L0_L1 = imresize(G0_L0_L1, [rows0 cols0]);
G0_L1_L2 = imresize(G0_L1_L2, [rows0 cols0]);
G0_L2_L3 = imresize(G0_L2_L3, [rows0 cols0]);
G0_L3_L4 = imresize(G0_L3_L4, [rows0 cols0]);
G0_L4_L5 = imresize(G0_L4_L5, [rows0 cols0]);
G0_L5_L6 = imresize(G0_L5_L6, [rows0 cols0]);

G0_L0_L1 =(G0_L0_L1-min(min(G0_L0_L1)))/max(max(G0_L0_L1-min(min(G0_L0_L1))));

G0_L1_L2 =(G0_L1_L2-min(min(G0_L1_L2)))/max(max(G0_L1_L2-min(min(G0_L1_L2))));
G0_L2_L3 =(G0_L2_L3-min(min(G0_L2_L3)))/max(max(G0_L2_L3-min(min(G0_L2_L3))));
G0_L3_L4 =(G0_L3_L4-min(min(G0_L3_L4)))/max(max(G0_L3_L4-min(min(G0_L3_L4))));
G0_L4_L5 =(G0_L4_L5-min(min(G0_L4_L5)))/max(max(G0_L4_L5-min(min(G0_L4_L5))));
G0_L5_L6 =(G0_L5_L6-min(min(G0_L5_L6)))/max(max(G0_L5_L6-min(min(G0_L5_L6))));

% Add them together and scale
G0 = G0_L0_L1 + G0_L1_L2 + G0_L2_L3 + G0_L3_L4 + G0_L4_L5 + G0_L5_L6;
G0 = (G0 - min(min(G0))) / max(max(G0 - min(min(G0))));
figure, subplot(3,3,1), imshow(G0_L0_L1), title('L0-Ll')
 subplot(3,3,2), imshow(G0_L1_L2), title('Ll-L2')
 subplot(3,3,3), imshow(G0_L2_L3), title('L2-L3')
 subplot(3,3,4), imshow(G0_L3_L4), title('L3-L4')
 subplot(3,3,5), imshow(G0_L4_L5), title('L4-L5')
 subplot(3,3,6), imshow(G0_L5_L6), title('L5-L6')
 subplot(3,3,9), imshow(G0), title('0 degrees Map')
clear G0_L0_L1 G0_L1_L2 G0_L2_L3 G0_L3_L4 G0_L4_L5 G0_L5_L6;

% Next do the 45 degrees oriented filter
f45 = Gabor45.SpatialKernel;
G45_L0_L1 = conv2(L0_L1,f45,'valid');
G45_L1_L2 = conv2(L1_L2,f45,'valid');
G45_L2_L3 = conv2(L2_L3,f45,'valid');
G45_L3_L4 = conv2(L3_L4,f45,'valid');
G45_L4_L5 = conv2(L4_L5,f45,'valid');
G45_L5_L6 = conv2(L5_L6,f45,'valid');
G45_L0_L1 = imresize(G45_L0_L1, [rows0 cols0]);
G45_L1_L2 = imresize(G45_L1_L2, [rows0 cols0]);
G45_L2_L3 = imresize(G45_L2_L3, [rows0 cols0]);
G45_L3_L4 = imresize(G45_L3_L4, [rows0 cols0]);
G45_L4_L5 = imresize(G45_L4_L5, [rows0 cols0]);
G45_L5_L6 = imresize(G45_L5_L6, [rows0 cols0]);

G45_L0_L1=(G45_L0_L1-min(min(G45_L0_L1)))/max(max(G45_L0_L1-min(min(G45_L0_L1))));
G45_L1_L2=(G45_L1_L2-min(min(G45_L1_L2)))/max(max(G45_L1_L2-min(min(G45_L1_L2))));
G45_L2_L3=(G45_L2_L3-min(min(G45_L2_L3)))/max(max(G45_L2_L3-min(min(G45_L2_L3))));
G45_L3_L4=(G45_L3_L4-min(min(G45_L3_L4)))/max(max(G45_L3_L4-min(min(G45_L3_L4))));
G45_L4_L5=(G45_L4_L5-min(min(G45_L4_L5)))/max(max(G45_L4_L5-min(min(G45_L4_L5))));
G45_L5_L6=(G45_L5_L6-min(min(G45_L5_L6)))/max(max(G45_L5_L6-min(min(G45_L5_L6))));

% Add them together and scale
G45 = G45_L0_L1 + G45_L1_L2 + G45_L2_L3 + G45_L3_L4 + G45_L4_L5 + G45_L5_L6;
G45 = (G45 - min(min(G45))) / max(max(G45 - min(min(G45))));
figure, subplot(3,3,1), imshow(G45_L0_L1), title('L0-L1')
 subplot(3,3,2), imshow(G45_L1_L2), title('L1-L2')
 subplot(3,3,3), imshow(G45_L2_L3), title('L2-L3')
 subplot(3,3,4), imshow(G45_L3_L4), title('L3-L4')
 subplot(3,3,5), imshow(G45_L4_L5), title('L4-L5')
 subplot(3,3,6), imshow(G45_L5_L6), title('L5-L6')
 subplot(3,3,9), imshow(G45), title('45 degrees Map')
clear G45_L0_L1 G45_L1_L2 G45_L2_L3 G45_L3_L4 G45_L4_L5 G45_L5_L6;

% Next do the 90 degrees oriented filter
f90 = Gabor90.SpatialKernel;
G90_L0_L1 = conv2(L0_L1,f90,'valid');
G90_L1_L2 = conv2(L1_L2,f90,'valid');
G90_L2_L3 = conv2(L2_L3,f90,'valid');
G90_L3_L4 = conv2(L3_L4,f90,'valid');
G90_L4_L5 = conv2(L4_L5,f90,'valid');
G90_L5_L6 = conv2(L5_L6,f90,'valid');
%G90_L6_L7= conv2(L6_L7,f90,'valid');
%G90 L7 L8= conv2(L7_L8,f90,'valid');

G90_L0_L1 = imresize(G90_L0_L1, [rows0 cols0]);
G90_L1_L2 = imresize(G90_L1_L2, [rows0 cols0]);
G90_L2_L3 = imresize(G90_L2_L3, [rows0 cols0]);
G90_L3_L4 = imresize(G90_L3_L4, [rows0 cols0]);
G90_L4_L5 = imresize(G90_L4_L5, [rows0 cols0]);
G90_L5_L6 = imresize(G90_L5_L6, [rows0 cols0]);

G90_L0_L1 =(G90_L0_L1-min(min(G90_L0_L1)))/max(max(G90_L0_L1 -min(min(G90_L0_L1))));
G90_L1_L2 =(G90_L1_L2-min(min(G90_L1_L2)))/max(max(G90_L1_L2 -min(min(G90_L1_L2))));
G90_L2_L3 =(G90_L2_L3-min(min(G90_L2_L3)))/max(max(G90_L2_L3- min(min(G90_L2_L3))));
G90_L3_L4 =(G90_L3_L4-min(min(G90_L3_L4)))/max(max(G90_L3_L4- min(min(G90_L3_L4))));
G90_L4_L5 =(G90_L4_L5-min(min(G90_L4_L5)))/max(max(G90_L4_L5- min(min(G90_L4_L5))));
G90_L5_L6 =(G90_L5_L6-min(min(G90_L5_L6)))/max(max(G90_L5_L6- min(min(G90_L5_L6))));

% Add them together and scale
G90 = G90_L0_L1 + G90_L1_L2 + G90_L2_L3 + G90_L3_L4 + G90_L4_L5 + G90_L5_L6;
G90 = (G90 - min(min(G90))) / max(max(G90 - min(min(G90))));
figure, subplot(3,3,1), imshow(G90_L0_L1), title('L0-Ll')
 subplot(3,3,2), imshow(G90_L1_L2), title('Ll-L2')
 subplot(3,3,3), imshow(G90_L2_L3), title('L2-L3')
 subplot(3,3,4), imshow(G90_L3_L4), title('L3-L4')
 subplot(3,3,5), imshow(G90_L4_L5), title('L4-L5')
 subplot(3,3,6), imshow(G90_L5_L6), title('L5-L6')
 subplot(3,3,9), imshow(G90), title('90 degrees Map')
clear G90_L0_L1 G90_L1_L2 G90_L2_L3 G90_L3_L4 G90_L4_L5 G90_L5_L6;

% And the 135 degrees oriented filter
f135 = Gabor135.SpatialKernel;
G135_L0_L1 = conv2(L0_L1,f135,'valid');
G135_L1_L2 = conv2(L1_L2,f135,'valid');
G135_L2_L3 = conv2(L2_L3,f135,'valid');
G135_L3_L4 = conv2(L3_L4,f135,'valid');
G135_L4_L5 = conv2(L4_L5,f135,'valid');
G135_L5_L6 = conv2(L5_L6,f135,'valid');
G135_L0_L1 = imresize(G135_L0_L1, [rows0 cols0]);
G135_L1_L2 = imresize(G135_L1_L2, [rows0 cols0]);
G135_L2_L3 = imresize(G135_L2_L3, [rows0 cols0]);
G135_L3_L4 = imresize(G135_L3_L4, [rows0 cols0]);
G135_L4_L5 = imresize(G135_L4_L5, [rows0 cols0]);
G135_L5_L6 = imresize(G135_L5_L6, [rows0 cols0]);

G135_L0_L1=(G135_L0_L1-min(min(G135_L0_L1)))/max(max(G135_L0_L1-min(min(G135_L0_L1))));
G135_L1_L2=(G135_L1_L2-min(min(G135_L1_L2)))/max(max(G135_L1_L2-min(min(G135_L1_L2))));
G135_L2_L3=(G135_L2_L3-min(min(G135_L2_L3)))/max(max(G135_L2_L3-min(min(G135_L2_L3))));
G135_L3_L4=(G135_L3_L4-min(min(G135_L3_L4)))/max(max(G135_L3_L4-min(min(G135_L3_L4))));
G135_L4_L5=(G135_L4_L5-min(min(G135_L4_L5)))/max(max(G135_L4_L5-min(min(G135_L4_L5))));
G135_L5_L6=(G135_L5_L6-min(min(G135_L5_L6)))/max(max(G135_L5_L6-min(min(G135_L5_L6))));

% Add them together and scale
G135 = G135_L0_L1 + G135_L1_L2 + G135_L2_L3 + G135_L3_L4 + G135_L4_L5 + G135_L5_L6;
G135 = (G135 - min(min(G135))) /max(max(G135 - min(min(G135))));
figure, subplot(3,3,1), imshow(G135_L0_L1), title('L0-Ll')
 subplot(3,3,2), imshow(G135_L1_L2), title('Ll-L2')
 subplot(3,3,3), imshow(G135_L2_L3), title('L2-L3')
 subplot(3,3,4), imshow(G135_L3_L4), title('L3-L4')
 subplot(3,3,5), imshow(G135_L4_L5), title('L4-L5')
 subplot(3,3,6), imshow(G135_L5_L6), title('L5-L6')
 subplot(3,3,7), imshow(G135), title('135 degrees Map')
clear G135_L0_L1 G135_L1_L2 G135_L2_L3 G135_L3_L4 G135_L4_L5 G135_L5_L6;

% Create the edge map by combining together all of the oriented edge maps
    OrMap = (G0 + G45 + G90 + G135) / 4;
 subplot(3,3,9), imshow(OrMap), title('Orientation Map')
% show the 4 different oritentation in one time
 figure, 
 subplot(2,2,1),imshow(G0), title('0 degrees Map')
 subplot(2,2,2),imshow(G45), title('45 degrees Map')
 subplot(2,2,3),imshow(G90), title('90 degrees Map')
 subplot(2,2,4),imshow(G135), title('135 degrees Map')
 

%% CREATE THE OBJECT MAP
% First, create a mask, then blur the mask to create the object map, then use
% the mask to inhibit the unimportant regions and enhance the important regions
% of the map.
% Process the image with a 16x16 block size
%x = rgb2gray(I);
%x = double(x);
%x = mat2gray(x);

% fun = @(block_struct) imresize(block_struct.data,0.15);
% bgl6 = blockproc(a_total,[16 16],fun);


% 
% bgl6 = blkproc(a_total,[16 16],'min(x(:))');
% 
% 
% 
% bg = imresize(bgl6, [size(a_total)],'bilinear');
% figure, subplot(2,3,1), imshow(I), title('Image')
%  subplot(2,3,2), imshow(bg, []), title('Background')
% % Subtract off the background and show what's left
% diff=a_total - bg;
%  subplot(2,3,3), imshow(diff, []), title('Image-Bg')

% % Structuring element
% SE1 = ones(9,9);
% SE2 = ones(3,3);
% 
% % Threshold the foreground image and find the edges
% bw = imbinarize(diff, .2);
% bw = edge(bw,'canny');
% figure, subplot(2,3,4), imshow(bw), title('Foreground Edges')
% 
% % Dilate the edge image to find regions
% bw = imdilate(bw,SE1);
%  subplot(2,3,5), imshow(bw), title('Dilated Edges')
% 
% % Fill in the holes of the dilated edge image
% m = bwmorph(bw,'majority', 10);
% m = bwfill(m,'holes');
% m = imerode(m,SE2);
% %m32=blkproc(m,[32,32],'mean(x(:))');
% %m2 = imresize(m32, [size(m)],'bilinear');
% %m2 = double(m2);
% %m2 = (m2 - min(min(m2))) / max(max(m2 - min(min(m2))));
%  subplot(2,3,6), imshow(m), title('Object Map')
% mask = double(m);
% ObjectMap = double(m);

%%
% Find out what the weights for the various feature maps should be, based on
% the overall mean of each feature map.
% [r c] = size(ColMap);
% ColMap2 = reshape(ColMap, r.*c, 1);
% COLmean = mean(ColMap2);
% COLstd = std(ColMap2);
% clear ColMap2
% InMap2 = reshape(InMap, r.*c, 1);
% INmean =mean(InMap2);
% INstd = std(InMap2);
% clear InMap2
% OrMap2 = reshape(OrMap, r.*c, 1);
% ORmean = mean(OrMap2);
% ORstd = std(OrMap2);
% clear OrMap2
% ObjectMap2 = reshape(ObjectMap, r.*c, 1);
% OBJmean = mean(ObjectMap2);
% OBJstd = std(ObjectMap2);
% clear ObjectMap2
% w1 = weights(1,1);
% w2 = weights(1,2);
% w3 = weights(1,3);
% w4 = weights(1,4);
% w5 = weights(1,5);
% total_weights = w1 + w2 + w3 + w4;
% w1 = w1 ./ total_weights;
% w2 = w2 ./ total_weights;
% w3 = w3 ./ total_weights;
% w4 = w4 ./ total_weights;

%% CREATE THE MAP
% Create the map by combining together the Color Map, the Intensity
% Map, the Orientation Map, and the Object Map, and scaling from 0 - 1
% Map1 = (ColMap + InMap + OrMap)/3;                               % Map 1
% Map2 = (ColMap + InMap + OrMap + ObjectMap)/4;                   % Map 2
% Map3 = (ColMap + InMap + OrMap + ObjectMap)/4.*mask;             % Map 3
% Map4 = (ColMap.*w1 + InMap.*w2 + OrMap.*w3 + ObjectMap.*w4).*mask.*w5; % Map 4
% 
% Map1 = (Mapl - min(min(Map1))) / max(max(Map1 - min(min(Map1))));
% Map2 = (Map2 - min(min(Map2))) / max(max(Map2 - min(min(Map2))));
% Map3 = (Map3 - min(min(Map3))) / max(max(Map3 - min(min(Map3))));
% Map4 = (Map4 - min(min(Map4))) / max(max(Map4 - min(min(Map4))));
% 
% figure,subplot(2,3,1),imshow(I),        title('Original Image')
%  subplot(2,3,2),imshow(InMap),    title('Intensity'), colormap(gray)
%  subplot(2,3,3),imshow(ColMap),   title('Color'), colormap(gray)
%  subplot(2,3,4),imshow(OrMap),    title('Orientation'), colormap(gray)
%  subplot(2,3,5),imshow(ObjectMap),title('Proto-object'), colormap(gray)
%  subplot(1,2,1),imshow(Map3),     title('Importance Map w/o Weights')
%  subplot(1,2,2),imshow(Map4),     title('Importance Map w/ Weights')
% colormap(jet)

%y.mapl = Mapl;
%y.map2 = Map2;
%y.map3 = Map3;

%y.map4 = Map4;
%y.features = [COLmean COLstd INmean INstd ORmean ORstd OBJmean OBJstd];
%y.weights = [wl w2 w3 w4 w5];
%y.ColMap = ColMap;
%y.InMap = InMap;
%y.OrMap = OrMap;
% y = ObjectMap;
% 
% toc
