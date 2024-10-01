function [num_questions] = hw5()

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

%% Part A
fprintf("Part A\n")

% 3D world coordinates (X, Y, Z)
wdCrds = readmatrix('world_coords.txt');

% 2D image coordinates (x, y)
imCrds = readmatrix('image_coords.txt');

% Number of points
n = size(wdCrds, 1);

% Create the A matrix for homogeneous least squares
A = zeros(2*n, 12);
for i = 1:n
    X = wdCrds(i, 1);
    Y = wdCrds(i, 2);
    Z = wdCrds(i, 3);
    x = imCrds(i, 2); % Convert from click to index coordinates
    y = imCrds(i, 1); % Convert from click to index coordinates
    
    A(2*i-1, :) = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x];
    A(2*i, :)   = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y];
end

% Solve the system using SVD
[~, ~, V] = svd(A);
M = reshape(V(:, end), 4, 3)'; % Reshape last column of V to 3x4 matrix

% Display the camera matrix
disp('Camera Matrix:');
disp(M);

function [projPnts, rms_error] = project_camera_points(world_coords,calib_mat,camera_ground_truth)
    N = size(world_coords, 1);  % Number of points
    
    % Convert the world coordinates to homogeneous form (add a column of ones)
    wdHomo = [world_coords, ones(N, 1)];
    
    projHomo = (calib_mat * wdHomo')';  % Multiply by camera matrix (P), transpose to match sizes
    % Normalize the homogeneous coordinates
    projHomo = projHomo ./ repmat(projHomo(:, 3), 1, 3);
    projPnts = projHomo(:, 1:2);
    
    % Compute the error (Euclidean distance between actual and projected points)
    projPnts = projPnts(:, [2,1]); % Swap index to convert from clicking to indexing coordinates
    errors = sqrt(sum((projPnts - camera_ground_truth).^2, 2));  % Euclidean distance per point
    % Compute the RMS error
    rms_error = sqrt(mean(errors.^2));
end

% Compute the RMS error between observed and predicted points
[pred_Pnts,rms_error] = project_camera_points(wdCrds,M,imCrds);

% Display the RMS error
disp(['RMS Error: ', num2str(rms_error)]);

f1 = figure('visible','off');
imshow("IMG_0862.png"); % Also works with IMG_0861.jpeg if you encounter an error
hold on;
f(1) = plot(imCrds(:, 1), imCrds(:, 2), 'ko','MarkerSize',20,'MarkerFaceColor','red','DisplayName',"ground truth");
f(2) = plot(pred_Pnts(:, 1), pred_Pnts(:, 2), 'ko','MarkerSize',20,'MarkerFaceColor','green','DisplayName',"projected");
l = legend(f);
set(l,'FontSize',48);
hold off;
exportgraphics(f1, 'output/f1_predicted_points_HLSQR.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Part B
fprintf("\nPart B\n")


end