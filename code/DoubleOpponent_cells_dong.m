%Simulating Double-opponent cells
% COSI course 2021 - by J.L. Nieves
% Code should/could be optimized and completed

clear All; close All; clc;
%cd('C:\MATLAB\ProgramasMatlab\Granada_SRI_Paintings_Yu-Jung Chen');
% cd('G:\Mi unidad\Descargas\Datos_RGB\Ilum1\flores_y_frutas');

%% Get image and their basic info
% [filename, pathname]=uigetfile('*.*','MultiSelect','on','Open images');
% cd(pathname)
% load(filename);
% im_orig= im; clear im;

im_orig= imread('colorful_card.jpg');
% im_orig= imread('Herman_gird_colored.jpg');

imshow(im_orig);
[nx,ny,nz]= size(im_orig);

% % if the image is rgb
% if nz ~= 1
%     im_lab= rgb2lab(im_orig);
%     im_layer = 'rgb';
% % if the image is only black and white
% elseif nz ==1
%     im_orig = cat(3, im_orig, im_orig, im_orig);
%     im_lab= rgb2lab(im_orig);
%     im_layer = 'bw';
% end

%% Cone layer
imCones(:,:,1)= im_orig(:,:,1);
imCones(:,:,2)= im_orig(:,:,2);
imCones(:,:,3)= im_orig(:,:,3);

imCones(:,:,4)= (im_orig(:,:,1)+im_orig(:,:,2))/2; %Pre-Yellow component

imCones(:,:,5)= im_orig(:,:,1)+im_orig(:,:,2)+imCones(:,:,3); %Luminance component

%% Retinal layer
%Considering only the type II single-opponent (SO) cells
imOrg(:,:,1)= (1/sqrt(2)).*(imCones(:,:,1)-imCones(:,:,2));
imOrg(:,:,2)= -imOrg(:,:,1);
imOyb(:,:,1)= (1/sqrt(6)).*(imCones(:,:,4)-2*imCones(:,:,3));
imOyb(:,:,2)= -imOyb(:,:,1);
imOl(:,:,1) = (1/sqrt(3)).*imCones(:,:,5);
imOl(:,:,2) = -imOl(:,:,1);

%Single Opponent SO cells
sigma= 2.5;
% hsize= 16;
% h = fspecial('gaussian',hsize,sigma);
% imSOrg(:,:,1)= conv2(imOrg(:,:,1), h);

imSOrg(:,:,1)= imgaussfilt(imOrg(:,:,1), sigma);
imSOrg(:,:,2)= imgaussfilt(imOrg(:,:,2), sigma);
imSOyb(:,:,1)= imgaussfilt(imOyb(:,:,1), sigma);
imSOyb(:,:,2)= imgaussfilt(imOyb(:,:,2), sigma);
imSOl(:,:,1) = imgaussfilt(imOl(:,:,1), sigma);
imSOl(:,:,2) = imgaussfilt(imOl(:,:,2), sigma);

%% V1 layer
%from two SO cells of type II with different scales:
%Ej: one red-on/green-off SO cell with smaller RF scale and another 
%green-on/red-off SO cell with larger RF scale

%Two free parameters, the scale (sigma) of receptive field, and the
%cone weight (k)
lambda= 3; %set λ = 3 based on the finding that the size of RF surround is roughly 3 times (in diameter) larger than that of RF center
k= 0.9; %k is a relative cone weight that controls the contribution of RF surround.
imDOrg= imSOrg(:,:,1) + k.*(imgaussfilt(imOrg(:,:,2), lambda*sigma));
imDOyb= imSOyb(:,:,1) + k.*(imgaussfilt(imOyb(:,:,2), lambda*sigma));
imDOl = imSOl(:,:,1)  + k.*(imgaussfilt(imOl(:,:,2) , lambda*sigma));

%% Higher visual cortex
%imDO(:,:,1)= imDOl; imDO(:,:,2)= imDOrg; imDO(:,:,3)= imDOyb;
%figure, imshow(imDO./max(max(max(imDO))));

%transform the output of DO cells from the double-opponent space to the RGB space
DO2rgb=[1/sqrt(2),-1/sqrt(2), 0; 1/sqrt(6),1/sqrt(6),-2/sqrt(6); 1/sqrt(3),1/sqrt(3),1/sqrt(3)];
DO=[reshape(imDOrg,nx*ny,1), reshape(imDOyb,nx*ny,1), reshape(imDOl,nx*ny,1)]';
DTrgb= (DO2rgb\double(DO))';
DTrgb= reshape(DTrgb, nx,ny,nz);

%In the final image, the equalization of the 3 channels is colored
% figure, imshow(DTrgb./max(max(max(DTrgb))));
% figure, plot(reshape(DTrgb(:,:,1),nx*ny,1), reshape(DTrgb(:,:,2),nx*ny,1),'.')

%To avoid false colors in the final image
DTrgbn(:,:,1)= DTrgb(:,:,1)./max(max(DTrgb(:,:,1)));
DTrgbn(:,:,2)= DTrgb(:,:,2)./max(max(DTrgb(:,:,2)));
DTrgbn(:,:,3)= DTrgb(:,:,3)./max(max(DTrgb(:,:,3)));
figure, imshow(DTrgbn);
figure, subplot(1,3,1), plot(reshape(DTrgbn(:,:,1),nx*ny,1), reshape(DTrgbn(:,:,2),nx*ny,1)),...
        subplot(1,3,2), plot(reshape(DTrgbn(:,:,1),nx*ny,1), reshape(DTrgbn(:,:,3),nx*ny,1)),...
        subplot(1,3,3), plot(reshape(DTrgbn(:,:,2),nx*ny,1), reshape(DTrgbn(:,:,3),nx*ny,1));