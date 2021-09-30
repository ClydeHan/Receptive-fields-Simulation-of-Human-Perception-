% Lab work Lecture 1: Simulating Receptive Fields
% COSI course 2021 - by J.L. Nieves
% Code should/could be optimized and completed

close All, clear All
%% extract single color of fruits image
img = imread('fruits.jpg'); % Read image
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel
a = zeros(size(img, 1), size(img, 2));
just_red = cat(3, red, a, a);
just_green = cat(3, a, green, a);
just_blue = cat(3, a, a, blue);
back_to_original_img = cat(3, red, green, blue);
figure, imshow(just_red)
red_img= double(red);
%% Herman_grid image
% img = imread('Herman_grid.jpg');
% figure(), imshow(img)
% red_img= double(img) % for Herman grid image, no need convert to grayscale
%% Match_band image
% img = imread('Mach_band.jpg');
% % figure(), imshow(img)
% red_img= double(rgb2gray(img)); 
%% Gaussian kernel
% Create a gaussian function with sigma spread
sigma = [1,2];
for i = 1:2 
    values= -3*sigma(1,i):0.5:3*sigma(1,i);
    [x,y]= meshgrid(values); %2-D grid coordinates based on the coordinates contained in vectors x and y
    z= x.^2 + y.^2;
%figure, surfc(x,y,z)
    gauss2D= exp(-z/(2*sigma(1,i).^2));

    figure(), surfc(gauss2D), title(['Gauss2D ',num2str(i),' sigma']);

%filter the image with the kernel
    imGaussfilter= conv2(gauss2D, red_img);
    figure(), imshow(imGaussfilter./max(max(imGaussfilter))), title(['Gaussfilter ',num2str(i),' sigma']);
%     imshow(imGaussfilter./max(max(imGaussfilter))), title(['Gaussfilter ',num2str(i),' sigma']);
    
    %use max value of filter as divisor
%% Laplacian of Gaussian kernel
    lapGauss= (1/sigma(1,i).^2).*((z./(sigma(1,i).^2))-2).*gauss2D;
%lapGauss = laplacian(gauss2D);
%lapGauss = diff(gauss2D, 2, z);
    imLapfilter= conv2(lapGauss, red_img);
    figure(), subplot(1,2,1), imshow(imLapfilter./max(max(imLapfilter)))
    subplot(1,2,2), imshow(imLapfilter,[0 0.001]),title(['Lapfilter ',num2str(i),' sigma']) %detect edge, the zero crossing of the second derivative
end 
%% Gabor kernel
% theta= [0,45,90,135]; %4 orientation 
% psi= 0;
% lambda= 4;


wavelength = 4;
orientation = [0 45 90 135];
g = gabor(wavelength,orientation,'SpatialFrequencyBandwidth',1.25, 'SpatialAspectRatio', 0.6);

outMag= imgaborfilt(red_img,g);

outSize = size(outMag);
outMag = reshape(outMag,[outSize(1:2),1,outSize(3)]);
figure, 
% subplot(2,2,i),
% imshow(outMag)
montage(outMag,'BorderSize', [20,20], 'DisplayRange',[]);
title('4 different orientations of gabor magnitude output images');

% g = g - mean(g(:));
% g = g/sqrt(sum(g(:).^2));
 



% simulate the filter
% for i=1:4
%      xr(i,:)= x(1,:).*cosd(theta(i) + y(1,:).*sind(theta(i)));
%      yr(i,:)=-x(1,:).*sind(theta(i) + y(1,:).*cosd(theta(i)));
%      [xrr,yrr]= meshgrid(xr(i,:),yr(i,:)); %create new 2D coordinates 
%      xg(i,:,:)= xrr; yg(i,:,:)= yrr;
%     for k=1:2
%         gabor(i,:,:)= exp(-(squeeze(xg(i,:,:)).^2 +squeeze(yg(i,:,:)).^2)./(2*sigma(1,k).^2))...
%             .*cos(2.*pi.*(squeeze(xg(i,:,:))./lambda)+psi);
%              
%         imGaborfilt(i,:,:)= conv2(squeeze(gabor(i,:,:)), red_img);
%         figure(k),hold on, subplot(2,2,i), imshow(squeeze(imGaborfilt(i,:,:))), title(['Gabor kernel ',num2str(k),' sigma']);
%     end
% end


   
            
