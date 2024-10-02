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

npnts = 100; % number of points
R=0.5;       % sphere radius 1/2 inch
plt_cent = [3,2,3];         % projected sphere centroid coords
cam_crds = [9, 14, 11];     % estimate camera  coords
vis_light = [33, 29, 44];   % Visible light position (considered at infinity)
obs_light = [-30, 0, 0];    % Obscured light position behind sphere

function [sphere_matrix] = create_sphere(R,x0,y0,z0,npnts)
    phi = linspace(-pi/2,pi/2,npnts); % phi values
    theta = linspace(0,2*pi,npnts);   % theta values
    
    % estimate all points across phi and theta using vectorized operations
    % instead of nested for loops
    [phi,theta] = meshgrid(phi,theta);
    sx = x0 + cos(phi) .* cos(theta) * R; 
    sy = y0 + cos(phi) .* sin(theta) * R;
    sz = z0 + sin(phi) * R;

    sphere_matrix = [sx(:), sy(:), sz(:)];
end

% Surface matrix for sphere
sph = create_sphere(R,plt_cent(1),plt_cent(2),plt_cent(3),npnts);

% Outward normal direction N(X) for each point on the sphere N(X) = X - center
nx = sph - plt_cent;

% Calculate vector from each sphere point to camera P - X
cam_vec = cam_crds - sph;

% Check for visible points: P - X dot N(X) > 0
dot_prod_cam = sum(cam_vec .* nx, 2);
vis_mask = dot_prod_cam > 0;

% Apply visibility mask to sphere points and normal
sph_vis = sph(vis_mask, :);
sph_vis_norm = nx(vis_mask, :);

% Light direction (assume light at infinity)
light_dir = vis_light - plt_cent;
light_dir = light_dir / norm(light_dir); % normalize the vector

% Lambertian reflectance model: N(X) dot L (only for visible points)
lambertian = max(0, sum(sph_vis_norm .* light_dir, 2)); % Set negative values to 0

% Project visible points to the image plane using camera matrix (M) from Part A
hom_sph_vis = [sph_vis, ones(size(sph_vis, 1), 1)]'; % Convert the world coordinates to homogeneous form (add a column of ones)
prj_pnts_hom = M * hom_sph_vis;
proj_sph_pnts = prj_pnts_hom(1:2, :) ./ prj_pnts_hom(3, :);
proj_sph_pnts = proj_sph_pnts([2,1],:); % Swap index to convert from clicking to indexing coordinates

fprintf("The projected sphere has %d visible points out of %d total points.\n", size(proj_sph_pnts, 2), size(sph, 1));

% Plotting the sphere on the image
f2 = figure('visible','off');
imshow('IMG_0861.jpeg'); 
hold on;
scatter(proj_sph_pnts(1, :), proj_sph_pnts(2, :), 20, lambertian, 'filled'); % Shade by Lambertian reflectance
colorbar;
% title('Projected Sphere with original light source','FontSize',28);
hold off;
exportgraphics(f2, 'output/f2_projected_sphere_with_visible_points.png', 'Resolution', 200);


% Light direction (assume light at infinity)
light_dir = obs_light - plt_cent;
light_dir = light_dir / norm(light_dir); % normalize the vector

% Lambertian reflectance model: N(X) dot L (only for visible points)
lambertian = max(0, sum(sph_vis_norm .* light_dir, 2)); % Set negative values to 0

% Project visible points to the image plane using camera matrix (M) from Part A
hom_sph_vis = [sph_vis, ones(size(sph_vis, 1), 1)]'; % Convert the world coordinates to homogeneous form (add a column of ones)
prj_pnts_hom = M * hom_sph_vis;
proj_sph_pnts = prj_pnts_hom(1:2, :) ./ prj_pnts_hom(3, :);
proj_sph_pnts = proj_sph_pnts([2,1],:); % Swap index to convert from clicking to indexing coordinates

% Plotting the sphere on the image
f3 = figure('visible','off');
imshow('IMG_0861.jpeg'); 
hold on;
scatter(proj_sph_pnts(1, :), proj_sph_pnts(2, :), 20, lambertian, 'filled'); % Shade by Lambertian reflectance
colorbar;
% title('Projected sphere for rotated light source','FontSize',28);
hold off;
exportgraphics(f3, 'output/f3_projected_sphere_with_rotated_light.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Part C
fprintf("\nPart C\n")

I = imread('IMG_0862.png');
imageSize = [size(I,1) size(I,2)];
params = estimateCameraParameters(imCrds,wdCrds, "ImageSize",imageSize);
disp(params);

% Create a random M matrix

rho = M(3,4); 
M_obs_norm = M / rho;

% Intrisic matrix using camera matrix from Part A
% Extract the upper 3x3 matrix A from M
A = M(:, 1:3);

% C = -A \ M(:, 4);
C = -inv(A) * M(:, 4);
disp(C);

function [R, Q] = rq(A)
    % Flip the matrix and perform QR decomposition on the transpose
    [Q, R] = qr(flipud(A)');
    
    % Flip the result back
    % R = flipud(R');
    % R = fliplr(R);

    R = rot90(R',2);

    Q = flipud(Q');
end

% Perform RQ decomposition to get K and R
[R, K] = rq(A);


% Ensure K has positive diagonal elements
T = diag(sign(diag(K)));
K = K * T;
R = T * R;

t = -R * C;
disp(t);

% Step 3: Extract translation vector t
t = K \ M(:, 4);

% Display the results
disp('Intrinsic matrix K:');
disp(K);
disp('Rotation matrix R:');
disp(R);
disp('Translation vector t:');
disp(t);

alpha = K(1,1); % Focal length in the x-direction (in pixels) 
beta = K(2,2); % Focal length in the y-direction (in pixels) 
u_0 = K(1,3); % Principal point in the x-direction 
v_0 = K(2,3); % Principal point in the y-direction

% t = inv(K) * M_obs_norm(:, 4);

function K = extract_intrinsic_matrix(M)
    rhoz = M(3,4); 
    M = M / rhoz;
    % Decompose M using SVD
    [U, ~, V] = svd(M);

    % Extract the 3x3 submatrix
    K = U(:, 1:3);

    % Perform QR decomposition
    [Q, R] = qr(K);

    % Extract intrinsic parameters
    ax = R(1,1);
    ay = R(2,2);
    u0 = R(1,3);
    v0 = R(2,3);

    % Construct the intrinsic matrix
    K = [
        ax    0    u0
         0    ay   v0
         0     0     1
    ];
end

K = extract_intrinsic_matrix(M);

function [R, T, C] = decompose_camera_matrix(M, K)
    rhoz = M(3,4); 
    M = M / rhoz;

    % Invert the intrinsic matrix
    K_inv = inv(K);

    % Calculate the extrinsic matrix
    [R_raw, T_raw] = qr(K_inv * M);

    % Correct the rotation matrix
    R = R_raw * diag([1, 1, det(R_raw)]);

    % Calculate the camera position
    C = -R' * T_raw;
end

R, T, C = decompose_camera_matrix(M, K);

disp(C);


num_questions = num_questions + 1;


end