function [num_questions] = hw6()

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

%% Part A
fprintf("Part A\n")

% Load the images (assuming you have 7 grayscale images, Img1, Img2, ..., Img7)
I1 = imread('photometric_stereo_1.tiff');
I2 = imread('photometric_stereo_2.tiff');
I3 = imread('photometric_stereo_3.tiff');
I4 = imread('photometric_stereo_4.tiff');
I5 = imread('photometric_stereo_5.tiff');
I6 = imread('photometric_stereo_6.tiff');
I7 = imread('photometric_stereo_7.tiff');

% Convert images to double
I1 = double(rgb2gray(I1));
I2 = double(rgb2gray(I2));
I3 = double(rgb2gray(I3));
I4 = double(rgb2gray(I4));
I5 = double(rgb2gray(I5));
I6 = double(rgb2gray(I6));
I7 = double(rgb2gray(I7));

% Stack images into 3D matrix
[h, w] = size(I1); % Image dimensions
I = zeros(h, w, 7);
I(:,:,1) = I1;
I(:,:,2) = I2;
I(:,:,3) = I3;
I(:,:,4) = I4;
I(:,:,5) = I5;
I(:,:,6) = I6;
I(:,:,7) = I7;

% Light directions (as provided in the question)
L = readmatrix('light_directions.txt');

% Initialize matrices for normals and albedo
normals = zeros(h, w, 3);  % to store normals
albedo = zeros(h, w);      % to store albedo

% Solve for normals and albedo at each pixel
for i = 1:h
    for j = 1:w
        % Extract intensity vector at pixel (i,j) across all images
        I_vec = squeeze(I(i,j,:));  % 7x1 vector
        
        % Solve for the normal vector and albedo (Nx is the surface normal vector)
        Nx = L \ I_vec;
        
        % The length of Nx corresponds to the albedo, and we normalize Nx to get the normal
        albedo(i,j) = norm(Nx);  % Albedo is the magnitude of Nx
        normals(i,j,:) = Nx / albedo(i,j);  % Normalized normal vector
    end
end

% Rendering the canonical view (with a frontal light source in the direction [0, 0, 1])
canonical_image = zeros(h, w);

for i = 1:h
    for j = 1:w
        % The canonical light direction is [0, 0, 1], so we take the dot product with normal
        N = squeeze(normals(i,j,:));
        canonical_image(i,j) = max(0, N(3)) * 250; % We use max(0, N(3)) to ensure non-negative lighting
    end
end

% Display the canonical view image
f1 = figure('visible','off');
imshow(uint8(canonical_image), []);
title('Canonical View (Uniform Albedo and Frontal Light)','FontSize',28);
exportgraphics(f1, 'output/f1_canonical_view_of_images.png', 'Resolution', 200);

num_questions = num_questions + 1;


%% Part B
fprintf("\nPart B\n")

num_questions = num_questions + 1;


end