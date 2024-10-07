function [] = hw6()

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

%% Part A
fprintf("Part A\n")

% Load light directions
L = readmatrix('light_directions.txt');
% Load the images
[h, w, images, img_gray] = load_images('.');
% Calculate the image normal
normals = calculate_img_normals(h,w,L,img_gray);
% Calculate the image albedo
albedoImg = calculate_albedo(h,w,normals,L,images);

% Display the canonical view image
f1 = figure('visible','off');
imshow(uint8(albedoImg*255), []);
% box on; axis on;
title('Canonical View (Uniform Albedo and Frontal Light)','FontSize',10);
exportgraphics(f1, 'output/f1_canonical_view_of_images.png', 'Resolution', 200);

% Display the depth map
x = 1:1:h;
y = 1:1:w;
r = get_depth_map(h,w,normals);

f2 = figure('visible','off');
surf(x,y,r,'EdgeColor', 'none'); colormap('parula');
view(107, 64); % change the view direction
title('Surface depth map from image normals','FontSize',10);
% ax = gca;
% ax.Interactions = [rotateInteraction dataTipInteraction];
exportgraphics(f2, 'output/f2_surface_depth_mao.png', 'Resolution', 200);

[X, Y] = meshgrid(x, y); % Create grid from x and y vectors
Z = r;                   % Assuming r is a matrix corresponding to X and Y grid

% Compute surface normals from the depth map
[nx, ny, nz] = surfnorm(X, Y, Z);
snormals = cat(3, nx, ny, nz);
% Calculate the image albedo
surf_albedoImg = calculate_albedo(h,w,snormals,L,images);


f3 = figure('visible','off');
imshow(uint8(surf_albedoImg*255), []);
box on; axis on; 
title('Canonical View (Uniform Albedo and Frontal Light)','FontSize',10);
exportgraphics(f3, 'output/f3_reconstructed_image_from_surface_normals.png', 'Resolution', 200);

num_questions = num_questions + 4;

%% Part B
fprintf("\nPart B\n")

num_questions = num_questions + 1;


end


function [h, w, imgRGB, imgG] = load_images(directory)
    % Get all files in the directory
    files = dir(directory);
    tiff_files = files(startsWith({files.name}, 'photometric_stereo') & endsWith({files.name}, '.tiff'));
    [~, sortIndex] = sort({tiff_files.name});
    tiff_files = tiff_files(sortIndex);

    n_imgs = length(tiff_files);
    imgRGB  = cell(n_imgs,1);
    % Read the first file to get the image dimensions
    I1 = im2double(rgb2gray(imread(tiff_files(1).name)));
    [h, w] = size(I1);
    imgG = zeros(h, w, n_imgs);

    for n=1:n_imgs
        fname = tiff_files(n).name;
        img = imread(fname);
        imgRGB{n} = img;
        img = im2double(rgb2gray(img));
        imgG(:,:,n) = img;
    end
end

function [normals] = calculate_img_normals(h,w,LD,IMGG)
    % Initialize matrices for normals
    normals = zeros(h, w, 3);  % to store normals
    
    % Solve for normals at each pixel
    for i = 1:h
        for j = 1:w
            % Extract intensity vector at pixel (i,j) across all images
            I_vec = squeeze(IMGG(i,j,:));  % 7x1 vector
            % Solve for the normal vector
            G = (LD' * LD) \ (LD' * I_vec);
            % Normalized normal vector
            if norm(G) ~= 0
                normals(i,j,:) = G / norm(G);
            else
                normals(i,j,:) = [0;0;0];
            end
        end
    end
end

function [ch_albedo] = compute_channel_albedo(h, w, nImg, ch_int, J)
    % Function to compute albedo for a single channel (R, G, or B)
    % Inputs:
    %   intensity: Intensity values for a single channel
    %   J: Dot product between light directions and normals
    
    % Vectorized computation of albedo - flatten intensity and J to 2D arrays
    flat_ints = reshape(ch_int, nImg, []);
    J_flat = reshape(J, nImg, []);
    
    % Compute the albedo for each pixel
    albedo_flat = sum(flat_ints .* J_flat) ./ sum(J_flat .* J_flat);
    ch_albedo = reshape(albedo_flat, [h, w]);
end

function [albedoImg] = calculate_albedo(h,w,imgNorm,LD,imgs)
    n_img = length(imgs);

    rnorms = reshape(imgNorm, [], 3);
    J = LD * rnorms';
    J = reshape(J, [n_img, h, w]);

    imgIntsR = zeros(n_img, h, w);
    imgIntsG = zeros(n_img, h, w);
    imgIntsB = zeros(n_img, h, w);
    for n=1:n_img
        imgn = imgs{n};
        imgIntsR(n,:,:) = imgn(:,:,1);
        imgIntsG(n,:,:) = imgn(:,:,2);
        imgIntsB(n,:,:) = imgn(:,:,3);
    end
    
    albedoImg = zeros(h, w, 3);
    albedoImg(:, :, 1) = compute_channel_albedo(h,w,n_img,imgIntsR, J);
    albedoImg(:, :, 2) = compute_channel_albedo(h,w,n_img,imgIntsG, J);
    albedoImg(:, :, 3) = compute_channel_albedo(h,w,n_img,imgIntsB, J);

    % Normalize the albedo image to be between 0 and 1
    albedoImg = albedoImg / max(albedoImg(:));
end

function [r] = get_depth_map(h,w,normals)
    Fx = size(h,w);
    Fy = size(h,w);
    for i=1:h
        for j=1:w
            normal = reshape(normals(i,j,:),1,3);
            deriv = normal/normal(3);
            Fx(i,j) = deriv(1);
            Fy(i,j) = deriv(2);
        end
    end
    
    r=zeros(h,w);
    for i=2:h
        r(i,1)=r(i-1,1)+Fy(i,1);
    end
    
    for i=2:h
        for j=2:w
            r(i,j)=r(i,j-1)+Fx(i,j);
        end
    end
end