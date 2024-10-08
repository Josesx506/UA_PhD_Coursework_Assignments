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
[normals,~] = calculate_img_normals(h,w,L,img_gray);
% Calculate the image albedo
albedoImg = calculate_albedo(h,w,normals,L,images);

% Define the light source direction (from the camera view direction, i.e., (0, 0, 1))
light_dir = [0,0,1];
% light_dir = [0.27872325   0.34950790   0.89451528];

% Canonical image estimate
can_img = zeros(h,w);
for i = 1:h
    for j = 1:w
        nx = normals(i,j,:);
        al = albedoImg(i,j);
        can_img(i,j) = dot(squeeze(nx),light_dir)*al;
    end
end

% Display the canonical view image
f1 = figure('visible','off');
imshow(can_img);
% box on; axis on;
title('Canonical View (Uniform Albedo and Frontal Light)','FontSize',16);
exportgraphics(f1, 'output/f1_canonical_view_of_images.png', 'Resolution', 200);

% Display the depth map
x = 1:1:h;
y = 1:1:w;
% r = get_depth_map(h,w,normals);
r = get_surface_map(normals);

f2 = figure('visible','off');
surf(x,y,r,'EdgeColor', 'none'); colormap('parula');
view(107, 64); % change the view direction
title('Surface depth map from image normals','FontSize',16);
% ax = gca;
% ax.Interactions = [rotateInteraction dataTipInteraction];
exportgraphics(f2, 'output/f2_surface_depth_map.png', 'Resolution', 200);

[X, Y] = meshgrid(x, y); % Create grid from x and y vectors
Z = r;                   % Assuming r is a matrix corresponding to X and Y grid

% Compute surface normals from the depth map
[nx, ny, nz] = surfnorm(X, Y, Z);
snormals = cat(3, nx, ny, nz);
snormals = rot90(snormals,2); %fliplr(flipud(snormals));


% Calculate the image albedo
% surf_albedoImg = calculate_albedo(h,w,snormals,L,images);

% figure;
% quiver3(X, Y, Z, nx, ny, nz);  % Plot the normals

can_img = zeros(h,w);
for i = 1:h
    for j = 1:w
        nx = snormals(i,j,:);
        al = albedoImg(i,j);
        can_img(i,j) = dot(squeeze(nx),light_dir)*al;
    end
end

f3 = figure('visible','off');
imshow(can_img);
% box on; axis on; 
title('Canonical View (Reconstructed Normals)','FontSize',16);
exportgraphics(f3, 'output/f3_reconstructed_image_from_surface_normals.png', 'Resolution', 200);


num_questions = num_questions + 4;

%% Part B
fprintf("\nPart B\n")

cps1_img = imread('color_photometric_stereo_1.tiff');
cps1_col = readmatrix('color_light_colors_1.txt');
cps1_ld = readmatrix('color_light_directions_1.txt');
% cps1_illum = cps1_col .* cps1_ld;


cps1R = calculate_img_channel_normals(cps1_ld,cps1_col,cps1_img,1);
cps1G = calculate_img_channel_normals(cps1_ld,cps1_col,cps1_img,2);
cps1B = calculate_img_channel_normals(cps1_ld,cps1_col,cps1_img,3);
% cps1_normals = cat(3, cps1R, cps1G, cps1B);

% cps1_normals = cps1_normals / norm(cps1_normals);
r = get_surface_map(cps1R);

f4 = figure('visible','off');
surf(x,y,r,'EdgeColor', 'none'); colormap('parula');
view(107, 64); % change the view direction
title('Photometric Stereo 1 depth map','FontSize',16);
exportgraphics(f4, 'output/f4_cps1_depth_map.png', 'Resolution', 200);


cps2_img = imread('color_photometric_stereo_2.tiff');
cps2_col = readmatrix('color_light_colors_2.txt');
cps2_ld = readmatrix('color_light_directions_2.txt');
% cps2_illum = cps2_col .* cps2_ld;

cps2R = calculate_img_channel_normals(cps2_ld,cps2_col,cps2_img,1);
cps2G = calculate_img_channel_normals(cps2_ld,cps2_col,cps2_img,2);
cps2B = calculate_img_channel_normals(cps2_ld,cps2_col,cps2_img,3);
% cps2_normals = cat(3, cps2R, cps2G, cps2B);
cps2_normals = (cps2R .* cps2G .* cps2B);
cps2_normals = cps2_normals / PAGENORM(cps2_normals);

r = get_surface_map(cps2_normals);

f5 = figure('visible','off');
surf(x,y,r,'EdgeColor', 'none'); colormap('parula');
view(107, 64); % change the view direction
title('Photometric Stereo 2 depth map','FontSize',16);
exportgraphics(f5, 'output/f5_cps2_depth_map.png', 'Resolution', 200);



% Compute surface normals from the depth map
[nx, ny, nz] = surfnorm(X, Y, r);
snormals = cat(3, nx, ny, nz);
snormals = rot90(snormals,2);

figure;
quiver3(X, Y, r, nx, ny, nz);  % Plot the normals

can_img = zeros(h,w,3);
for c = 1:3
    for i = 1:h
        for j = 1:w
            if c == 1
                nx = cps2R(i,j,:);
                can_img(i,j,c) = dot(squeeze(nx),cps2_ld(1,:));
            elseif c == 2
                nx = cps2G(i,j,:);
                can_img(i,j,c) = dot(squeeze(nx),cps2_ld(1,:));
            else
                nx = cps2B(i,j,:);
                can_img(i,j,c) = dot(squeeze(nx),cps2_ld(1,:));
            end
        end
    end
end

figure;%('visible','off');
imagesc(snormals);
datacursormode on;

num_questions = num_questions + 2;


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
    I1 = double(rgb2gray(imread(tiff_files(1).name)));
    [h, w] = size(I1);
    imgG = zeros(h, w, n_imgs);

    for n=1:n_imgs
        fname = tiff_files(n).name;
        img = imread(fname);
        imgRGB{n} = img;
        img = double(rgb2gray(img));
        imgG(:,:,n) = img;
    end
end

function [normals, albedo] = calculate_img_normals(h,w,LD,IMGG)
    % Initialize matrices for normals
    normals = zeros(h, w, 3);  % to store normals
    albedo = zeros(h, w);   % to store albedo
    
    % Solve for normals at each pixel
    for i = 1:h
        for j = 1:w
            % Extract intensity vector at pixel (i,j) across all images
            I_vec = squeeze(IMGG(i,j,:));  % 7x1 vector
            % Solve for the normal vector
            G = (LD' * LD) \ (LD' * I_vec);
            % Normalized normal vector
            if norm(G) ~= 0
                n = G / norm(G);
            else
                n = [0;0;0]; % avoid zero division
            end
            normals(i,j,:) = n;
            albedo(i,j) = norm(G) * pi;
        end
    end
    albedo = albedo / max(albedo(:));
end

function [normals] = calculate_img_channel_normals(LD,LC,COL_IMG,CHN)
    % Initialize matrices for normals
    [h,w,~] = size(COL_IMG);
    normals = zeros(h, w, 3);  % to store normals
    LD = LD .* LC(:,CHN);   % create a light for a single channel
    nl = size(LD,1);
    
    % Solve for normals at each pixel
    for i = 1:h
        for j = 1:w
            % Extract intensity vector at pixel (i,j) across all images
            I_vec = squeeze(double(COL_IMG(i,j,CHN)));  % 5x3 vector
            % N_R = (LD' * LD) \ (LD' * I_vec(:, 1));
            % N_G = (LD' * LD) \ (LD' * I_vec(:, 2));
            % N_B = (LD' * LD) \ (LD' * I_vec(:, 3));
            % G_ij = (N_R + N_G + N_B) / 3;
            % G_ij = [N_R(1), N_G(2), N_B(3)];
            % 
            % N_R = I_vec(:, 1) ./ LD(:, 1);
            % N_G = I_vec(:, 2) ./ LD(:, 2);
            % N_B = I_vec(:, 3) ./ LD(:, 3);
            % G_ij = [N_R, N_G, N_B];

            % Since albedo is one, this step can be skipped
            % G_ij = G_ij / norm(G_ij);
               
            % disp(size(size(LD)));
            G_ij = zeros(nl,3);
            for l = 1:nl
                G_ij(l,:) = I_vec / LD(l, :)';
            end
            % G_ij = G_ij(G_ij~=0);
            % G_ij = G_ij(:, CHN);
            % disp(G_ij)
            % G_ij = (LD' * LD) \ (LD' * I_vec);
            % disp(G_ij);

            G_ij = sum(G_ij, 1) / nl;
            % G_ij = G_ij(1, :);
            if norm(G_ij) ~= 0
                G_ij = G_ij / norm(G_ij);
            else
                G_ij = [0;0;0]; % avoid zero division
            end
            normals(i,j,:) = G_ij;
        end
    end
end

function [surface] = get_surface_map(normals)
    [h,~,~] = size(normals);
    f_x = normals(:,:,1)./normals(:,:,3);
    f_x(isnan(f_x))=0;
    f_y = normals(:,:,2)./normals(:,:,3);
    f_y(isnan(f_y))=0;
    
    xsum = cumsum(f_x,2);
    ysum = cumsum(f_y,1);
    surface = repmat(xsum(1,:),h,1);
    surface = surface+ysum;
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

% function [r] = get_depth_map(h,w,normals)
%     Fx = size(h,w);
%     Fy = size(h,w);
%     for i=1:h
%         for j=1:w
%             normal = reshape(normals(i,j,:),1,3);
%             deriv = normal/normal(3);
%             Fx(i,j) = deriv(1);
%             Fy(i,j) = deriv(2);
%         end
%     end
% 
%     r=zeros(h,w);
%     for i=2:h
%         r(i,1)=r(i-1,1)+Fy(i,1);
%     end
% 
%     for i=2:h
%         for j=2:w
%             r(i,j)=r(i,j-1)+Fx(i,j);
%         end
%     end
% end