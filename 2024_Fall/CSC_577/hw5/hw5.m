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

% Find the eigenvector associated with the smallest eigenvalue or
[V, D] = eig(A' * A);
[~, idx] = min(diag(D));
M = reshape(V(:, idx), 4, 3)';

% Solve the system using SVD (more concise)
% [~, ~, V] = svd(A);
% M = reshape(V(:, end), 4, 3)'; % Reshape last column of V to 3x4 matrix

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

rho = sqrt(M(3,1)^2 + M(3,2)^2+ M(3,3)^2);
disp(['Constant rho', rho]);
M = M/rho;

q1 = M(1, (1:3));
q2 = M(2, (1:3));
q3 = M(3, (1:3));

u0 = dot(q1',q3);
v0 = dot(q2',q3);
disp('center coords');
disp([u0,v0]);
alpha = sqrt(dot(q1',q1) - u0^2);
beta = sqrt(dot(q2',q2) - v0^2);

r11 = ((u0*q3(1)) - (q1(1))) / alpha;
r12 = ((u0*q3(2)) - (q1(2))) / alpha;
r13 = ((u0*q3(3)) - (q1(3))) / alpha;
r21 = ((u0*q3(1)) - (q2(1))) / beta;
r22 = ((u0*q3(2)) - (q2(2))) / beta;
r23 = ((u0*q3(3)) - (q2(3))) / beta;
r31 = q3(1);
r32 = q3(2);
r33 = q3(3);
tx = ((u0*M(3,4)) - M(1,4)) / alpha;
ty = ((v0*M(3,4)) - M(2,4)) / beta;
tz = M(3,4);


disp(['alpha = ', num2str(alpha)]);
disp(['beta = ', num2str(beta)]);
disp(['u0 = ', num2str(u0)]);
disp(['v0 = ', num2str(v0)]);

% Construct K from the intrinsic parameters
K = [alpha, 0, u0;
     0,     beta,  v0;
     0,     0,     1];
disp('Intrinsic K');
disp(K);

% Construct X from the rotation and translation matrices
R = [r11, r12, r13;
     r21, r22, r23;
     r31, r32, r33];

if det(R) < 0
    R = -R;
end

T = [tx;ty;tz];

X = [R,T];

disp('Extrinsic X');
disp(X);

% Camera location
% C = inv(R) * T;
C = R' * T;

% Camera orientation
O = R' * [0;0;1];

disp('Camera location');
disp(C);
disp('Camera orientation');
disp(O);


% Estimate alpha and beta directly
I = imread('IMG_0859.jpeg');
ph = size(I,2); % horizontal pixel dimensions
pv = size(I,1); % vertical pixel dimensions
dh = 6;         % horizontal world dimensions - 6 inches
dv = 4;         % vertical world dimensions - 4 inches
D = 11.5;       % 11.5 inches - distance of camera from object
nalpha = D * ph / dh;
nbeta = D * pv / dv;

fprintf('Direct alpha estimate %.0f \n',nalpha);
fprintf('Direct beta estimate %d \n',nbeta);


num_questions = num_questions + 1;


end


% Unused code
% % Intrinsic matrix estimate
% B = M(:, 1:3);
% b = M(:, 4);
% 
% K = B * B';  % Compute K
% % Normalize K
% K = K / K(3, 3);
% 
% % Extract elements from K. k_c is unused because gamma is set to zero
% k_u = K(1, 1);
% k_c = K(1, 2);
% k_v = K(2, 2);
% 
% % Compute intrinsic parameters
% u0 = K(1, 3);
% v0 = K(2, 3);
% beta = sqrt(k_v - v0^2);
% gamma = (k_c - u0 * v0) / beta;
% alpha = sqrt(k_u - u0^2); % - gamma^2
% 
% disp(['alpha = ', num2str(alpha)]);
% disp(['beta = ', num2str(beta)]);
% disp(['u0 = ', num2str(u0)]);
% disp(['v0 = ', num2str(v0)]);
% 
% % Estimate the camera coordinates
% C = -inv(B) * b; % or -B \ b;
% disp("Camera origin");
% disp(C);
% 
% % Construct A from the intrinsic parameters
% K = [alpha, 0, u0;
%      0,     beta,  v0;
%      0,     0,     1];
% 
% % Compute extrinsic parameters
% R = K \ B; %inv(A) * B;
% t = K \ b; %inv(A) * b;
% 
% if det(R) < 0
%     R = -R;
% end
% 
% X = [R, t];
% X(4, :) = [0,0,0,1];
% 
% % Check if M can be recovered using the estimated values as a qc step
% I = [1,0,0,0;
%      0,1,0,0;
%      0,0,1,0];
% M_est = K * I * X;
% disp("Estimated camera matrix");
% disp(M_est);
% 
% C = -R' * t;
% display(C);
% 
% 
% disp('Rotation matrix R:');
% disp(R);
% disp('Translation vector t:');
% disp(t);
% function [R, Q] = rq(M)
%     [Q,R] = qr(flipud(M)');
%     R = flipud(R');
%     R = fliplr(R);
% 
%     Q = Q';   
%     Q = flipud(Q);
% end
% 
% [R,K] = rq(B);
% disp(R);
% 
% T = diag(sign(diag(K)));
% 
% K = K * T;
% R = T * R;