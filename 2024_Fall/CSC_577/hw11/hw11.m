function [num_questions] = hw11()

close all;
clc;
num_questions = 0;


% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end


%% Part A
fprintf("Part A\n")

verbose = false;  % Print the iteration loss
spatial = false;
fs = 14;

% Question A1
ss_5 = k_means('sunset.tiff', 5, 100, 1e-4, spatial, 0, verbose);
ss_10 = k_means('sunset.tiff', 10, 100, 1e-4, spatial, 0, verbose);

figure('Position',[1, 1, 600, 250], 'visible','off');
f1 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(ss_5); title('K = 5','FontSize',fs);
nexttile; imshow(ss_10); title('K = 10','FontSize',fs);
exportgraphics(f1, 'output/f1_sunset_kdiff.png', 'Resolution', 200);

tgr1_5 = k_means('tiger-1.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr1_10 = k_means('tiger-1.tiff', 10, 100, 1e-4, spatial, 0, verbose);

figure('Position',[1, 1, 600, 250], 'visible','off');
f2 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(tgr1_5); title('K = 5','FontSize',fs);
nexttile; imshow(tgr1_10); title('K = 10','FontSize',fs);
exportgraphics(f2, 'output/f2_tiger1_kdiff.png', 'Resolution', 200);

tgr2_5 = k_means('tiger-2.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr2_10 = k_means('tiger-2.tiff', 10, 100, 1e-4, spatial, 0, verbose);

figure('Position',[1, 1, 600, 250], 'visible','off');
f3 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(tgr2_5); title('K = 5','FontSize',fs);
nexttile; imshow(tgr2_10); title('K = 10','FontSize',fs);
exportgraphics(f3, 'output/f3_tiger2_kdiff.png', 'Resolution', 200);


% Question A2
spatial = true;

ss_5_l0 = k_means('sunset.tiff', 5, 100, 1e-4, spatial, 0, verbose);
ss_5_l1 = k_means('sunset.tiff', 5, 100, 1e-4, spatial, 1, verbose);
ss_5_l10 = k_means('sunset.tiff', 5, 100, 1e-4, spatial, 10, verbose);
ss_10_l0 = k_means('sunset.tiff', 10, 100, 1e-4, spatial, 0, verbose);
ss_10_l1 = k_means('sunset.tiff', 10, 100, 1e-4, spatial, 1, verbose);
ss_10_l10 = k_means('sunset.tiff', 10, 100, 1e-4, spatial, 10, verbose);

tgr1_5_l0 = k_means('tiger-1.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr1_5_l1 = k_means('tiger-1.tiff', 5, 100, 1e-4, spatial, 1, verbose);
tgr1_5_l10 = k_means('tiger-1.tiff', 5, 100, 1e-4, spatial, 10, verbose);
tgr1_10_l0 = k_means('tiger-1.tiff', 10, 100, 1e-4, spatial, 0, verbose);
tgr1_10_l1 = k_means('tiger-1.tiff', 10, 100, 1e-4, spatial, 1, verbose);
tgr1_10_l10 = k_means('tiger-1.tiff', 10, 100, 1e-4, spatial, 10, verbose);

tgr2_5_l0 = k_means('tiger-2.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr2_5_l1 = k_means('tiger-2.tiff', 5, 100, 1e-4, spatial, 1, verbose);
tgr2_5_l10 = k_means('tiger-2.tiff', 5, 100, 1e-4, spatial, 10, verbose);
tgr2_10_l0 = k_means('tiger-2.tiff', 10, 100, 1e-4, spatial, 0, verbose);
tgr2_10_l1 = k_means('tiger-2.tiff', 10, 100, 1e-4, spatial, 1, verbose);
tgr2_10_l10 = k_means('tiger-2.tiff', 10, 100, 1e-4, spatial, 10, verbose);

% Create plots
plot_klambdas({ss_5_l0,ss_5_l1,ss_5_l10,ss_10_l0,ss_10_l1,ss_10_l10}, ...
    [5,10],[0,1,10],'output/f4_sunset_kms_spt.png');

plot_klambdas({tgr1_5_l0,tgr1_5_l1,tgr1_5_l10,tgr1_10_l0,tgr1_10_l1,tgr1_10_l10}, ...
    [5,10],[0,1,10],'output/f5_tiger1_kms_spt.png');

plot_klambdas({tgr2_5_l0,tgr2_5_l1,tgr2_5_l10,tgr2_10_l0,tgr2_10_l1,tgr2_10_l10}, ...
    [5,10],[0,1,10],'output/f6_tiger2_kms_spt.png');


% Question A3
kmeans_texture('sunset.tiff', 5, 31, 2, 4, 10, 100, 'output/f7_sunset_fvrgbspt.png', verbose);
kmeans_texture('tiger-1.tiff', 5, 31, 2, 4, 10, 100, 'output/f8_tiger1_fvrgbspt.png', verbose);
kmeans_texture('tiger-2.tiff', 5, 31, 2, 4, 10, 100, 'output/f9_tiger2_fvrgbspt.png', verbose);


fprintf("All images have been saved and exported.\n")

num_questions = num_questions + 3;

%% Part B
fprintf("\nPart B\n")

% Generate synthetic data
rng(577);

[points, true_lines] = generate_lines(4, 100, 0.05, 10);
% Apply K-means
[lines, assignments, error] = kmeans_lines(points, 4, 100, 1e-4, 'random_points');


% Visualize results
figure('Position',[1, 1, 820, 400], 'visible','off');
f10 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
x = linspace(-1, 2, 100);

nexttile; 
hold on; scatter(points(:, 1), points(:, 2), 20, 'black');
for i = 1:size(true_lines, 1)
    y = -(true_lines(i, 1) / true_lines(i, 2)) * x - true_lines(i, 3) / true_lines(i, 2);
    plot(x, y, 'LineWidth', 2);
end
ylim([-1.5 1.5]); title('Ground Truth Lines','FontSize',14);
hold off;


nexttile; 
hold on; scatter(points(:, 1), points(:, 2), 20, assignments, 'filled');
for i = 1:size(lines, 1)
    y = -(lines(i, 1) / lines(i, 2)) * x - lines(i, 3) / lines(i, 2);
    plot(x, y, 'LineWidth', 2);
end
ylim([-1.5 1.5]); title('K-means Line Clusters','FontSize',14); hold off;
exportgraphics(f10, 'output/f10_kmeans_lines.png', 'Resolution', 200);

num_questions = num_questions + 1;

end


%% Part A functions
function plot_klambdas(images,kvals,lambdas,outname)
    % Plot the images when comparing the effect of different lambda values 
    % for clustering the same image into different K clusters
    [im1,im2,im3,im4,im5,im6] = images{:};

    figure('Position',[1, 1, 900, 450], 'visible','off');
    fg = tiledlayout(2,3,'TileSpacing','Compact','Padding','Compact');
    fs = 14;
    
    % Top row for first 3 images
    nexttile; imshow(im1); 
    title(strcat('\lambda = ',num2str(lambdas(1))),'FontSize',fs,'Interpreter','tex');
    ylabel(sprintf('K = %d',kvals(1)), 'FontSize', fs);
    nexttile; imshow(im2); 
    title(strcat('\lambda = ',num2str(lambdas(2))),'FontSize',fs,'Interpreter','tex');
    nexttile; imshow(im3); 
    title(strcat('\lambda = ',num2str(lambdas(3))),'FontSize',fs,'Interpreter','tex');

    % Bottom row for next 3 images
    nexttile; imshow(im4); 
    ylabel(sprintf('K = %d',kvals(2)), 'FontSize', fs);
    nexttile; imshow(im5); 
    nexttile; imshow(im6);

    exportgraphics(fg, outname, 'Resolution', 200);
end

function [segmented_img] = k_means(image_path, K, max_iters, epsilon, spatial, lambda, verbose)
    rng(577);

     % Default args
    if nargin < 5 || isempty(spatial)
        spatial = false;
    end
    if nargin < 6 || isempty(lambda)
        lambda = 0;
    end
    if nargin < 7 || isempty(verbose)
        verbose = false;
    end

    % Read the image and reshape into 3D feature space (R, G, B)
    img = imread(image_path);
    [rows, cols, ~] = size(img);

    if spatial
        % Create 5D feature vector: [R, G, B, λX, λY]
        img = double(img) / 255; % Normalize RGB values to [0, 1]
        [X, Y] = meshgrid(linspace(0, 1, cols), linspace(0, 1, rows)); % Scale X, Y to [0, 1]
        data = [reshape(img, [], 3), lambda * reshape(X, [], 1), lambda * reshape(Y, [], 1)];
    else
        data = double(reshape(img, [], 3)); % Convert image to Nx3 data matrix
    end

    % Randomly initialize cluster centers (K x 3 matrix)
    cluster_centers = data(randperm(size(data, 1), K), :);
    
    % Initialize variables
    prev_objective = Inf;
    assignments = zeros(size(data, 1), 1);
    objective_values = zeros(max_iters, 1);
    
    if verbose, fprintf('K-means Clustering: K = %d\n', K); end
    
    for iter = 1:max_iters
        % Assign points to the nearest cluster center
        distances = pdist2(data, cluster_centers); % Pairwise distances
        [~, new_assignments] = min(distances, [], 2);
        
        % Update cluster centers
        for k = 1:K
            cluster_members = data(new_assignments == k, :);
            if ~isempty(cluster_members)
                cluster_centers(k, :) = mean(cluster_members, 1);
            end
        end
        
        % Compute objective function
        objective = sum(vecnorm(data - cluster_centers(new_assignments, :), 2, 2).^2);
        objective_values(iter) = objective;

        if verbose, fprintf('Iteration %d: Objective Function = %.2f\n', iter, objective); end
        
        % Define stopping conditions
        if abs(prev_objective - objective) < epsilon * prev_objective
            if verbose, fprintf('Converged due to small change in objective.\n'); end
            break;
        elseif all(assignments == new_assignments)
            if verbose, fprintf('Converged due to stable assignments.\n'); end
            break;
        end
        
        prev_objective = objective;
        assignments = new_assignments;
    end
    
    % Visualize segmentation
    segmented_img = zeros(rows * cols, 3);
    for k = 1:K
        segmented_img(assignments == k, :) = repmat(cluster_centers(k, 1:3), sum(assignments == k), 1);
    end

    if spatial
        segmented_img = reshape(segmented_img, rows, cols, 3);
    else
        segmented_img = reshape(uint8(segmented_img), rows, cols, 3);
    end
end

function kmeans_texture(image_path, K, W, sigma1, sigma2, lambda, max_iters, outname, verbose)
     % Default args
    if nargin < 9 || isempty(verbose)
        verbose = false;
    end
    
    % Read image and convert to grayscale
    img = imread(image_path);
    [rows, cols, ~] = size(img);
    gray_img = double(rgb2gray(img)) / 255; % Normalize grayscale to [0, 1]
    rgb_img = double(img) / 255;            % Normalize RGB to [0, 1]

    % Compute edge energy for two scales
    energy1 = edge_energy(gray_img, sigma1, W);
    energy2 = edge_energy(gray_img, sigma2, W);

    % Compute texture features using mean squared energy in window W
    texture_features = [energy1(:), energy2(:)];

    % Combine features
    [X, Y] = meshgrid(linspace(0, 1, cols), linspace(0, 1, rows));
    spatial_features = lambda * [X(:), Y(:)];
    
    % Feature cases
    feature_set_a = texture_features;                            % Case (a)
    feature_set_b = [reshape(rgb_img, [], 3), texture_features]; % Case (b)
    feature_set_c = [reshape(rgb_img, [], 3), spatial_features, texture_features]; % Case (c)

    % Normalize features
    feature_set_a = normalize_features(feature_set_a);
    feature_set_b = normalize_features(feature_set_b);
    feature_set_c = normalize_features(feature_set_c);

    % Run K-means for each feature set
    if verbose, fprintf('Running K-means for Feature Vectors (FV) Only (Case a)\n'); end
    seg_a = kmeans_clustering(feature_set_a, rows, cols, K, max_iters);

    if verbose, fprintf('Running K-means for FV + RGB (Case b)\n'); end
    seg_b = kmeans_clustering(feature_set_b, rows, cols, K, max_iters);

    if verbose, fprintf('Running K-means for FV + RGB + Spatial (Case c)\n'); end
    seg_c = kmeans_clustering(feature_set_c, rows, cols, K, max_iters);
    
    fs = 14;
    figure('Position',[1, 1, 800, 250], 'visible','off');
    fg = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
    nexttile; imshow(seg_a); title('Feature Vector (FV)','FontSize',fs);
    ylabel(sprintf('K = %d | W = %d', K, W), 'FontSize', fs);
    nexttile; imshow(seg_b); title('FV + RGB','FontSize',fs);
    nexttile; imshow(seg_c); title('FV + RGB + Spatial','FontSize',fs);
    exportgraphics(fg, outname, 'Resolution', 200);
end

function energy = edge_energy(gray_img, sigma, W)
    % Compute horizontal and vertical edge responses
    kernel_size = ceil(W * sigma);
    if mod(kernel_size, 2) == 0, kernel_size = kernel_size + 1; end
    half_size = (kernel_size - 1) / 2;
    [x, y] = meshgrid(-half_size:half_size, -half_size:half_size);
    G = exp(-(x.^2 + y.^2) / (2 * sigma^2)) / (2 * pi * sigma^2);
    Gx = conv2(G, [-1 0 1; -2 0 2; -1 0 1], 'same');
    Gy = conv2(G, [1 2 1; 0 0 0; -1 -2 -1], 'same');
    Ix = conv2(gray_img, Gx, 'same');
    Iy = conv2(gray_img, Gy, 'same');
    energy = sqrt(Ix.^2 + Iy.^2); % Edge energy
end


function normalized_features = normalize_features(features)
    % Normalize each feature dimension to zero mean, unit variance
    normalized_features = (features - mean(features)) ./ std(features);
end

function segmented_img = kmeans_clustering(features, rows, cols, K, max_iters)
    % Run K-means clustering and reshape result into segmented image
    rng(577); % Fix random seed
    [assignments, cluster_centers] = kmeans(features, K, 'MaxIter', max_iters,'Replicates', 3);

    num_features = size(cluster_centers, 2); % Number of feature dimensions
    segmented_img = zeros(rows * cols, 3);
    % Map cluster centers back to image
    for k = 1:K
        if num_features >= 3
            % Use first 3 dimensions for RGB
            segmented_img(assignments == k, :) = repmat(cluster_centers(k, 1:3), sum(assignments == k), 1);
        else
            % Replicate grayscale intensity for RGB display
            intensity = cluster_centers(k, 1); % Use first feature as grayscale
            segmented_img(assignments == k, :) = repmat(intensity, sum(assignments == k), 3);
        end
    end
    segmented_img = reshape(segmented_img, rows, cols, 3);
end



%% Part B functions

function [points, true_lines] = generate_lines(K, num_points_per_line, noise_std, num_outliers)
    % Generates synthetic data with points near K lines
    % Parameters:
    % K - Number of lines
    % num_points_per_line - Number of points per line
    % noise_std - Standard deviation of Gaussian noise for point generation
    % num_outliers - Number of uniformly distributed outliers

    points = [];
    true_lines = zeros(K, 3); % Lines in homogeneous form (ax + by + c = 0)

    for i = 1:K
        % Randomly generate two endpoints for a line within [-1, 1]
        x1 = rand() * 2 - 1;
        y1 = rand() * 2 - 1;
        x2 = rand() * 2 - 1;
        y2 = rand() * 2 - 1;

        % Compute the line equation ax + by + c = 0 from the two points
        % The line coefficients are given by:
        % a = y2 - y1, b = -(x2 - x1), c = x1*y2 - x2*y1
        a = y2 - y1;
        b = -(x2 - x1);
        c = x1 * y2 - x2 * y1;

        % Normalize the line coefficients for consistency
        norm_factor = sqrt(a^2 + b^2);
        a = a / norm_factor;
        b = b / norm_factor;
        c = c / norm_factor;

        true_lines(i, :) = [a, b, c];

        % Generate points along the line
        t = linspace(-1, 1, num_points_per_line)'; % Parameter for interpolation
        x_vals = t * (x2 - x1) + x1; % Interpolated x values
        y_vals = -(a / b) * x_vals - c / b; % Line equation for corresponding y values

        % Clip y_vals to keep them within [-1, 1] while keeping points on the line
        mask = (y_vals >= -1) & (y_vals <= 1);
        x_vals = x_vals(mask);
        y_vals = y_vals(mask);

        % Add Gaussian noise to the points
        x_vals = x_vals + noise_std * randn(size(x_vals));
        y_vals = y_vals + noise_std * randn(size(y_vals));

        points = [points; x_vals, y_vals];
    end

    % Add outliers
    if num_outliers > 0
        outliers = rand(num_outliers, 2) * 2 - 1; % Uniformly distributed points in [-1, 1]
        points = [points; outliers];
    end
end


function distances = point_to_line_dist(points, lines)
    % Computes the perpendicular distance from points to lines
    % Parameters:
    % points - Nx2 array of (x, y) coordinates
    % lines - Mx3 array of line parameters (a, b, c)

    N = size(points, 1);
    M = size(lines, 1);
    distances = zeros(N, M);

    for i = 1:M
        a = lines(i, 1);
        b = lines(i, 2);
        c = lines(i, 3);
        distances(:, i) = abs(a * points(:, 1) + b * points(:, 2) + c) ./ sqrt(a^2 + b^2);
    end
end

function line = fit_line(points)
    % Fit a line using homogeneous least squares
    % Parameters:
    % points - Nx2 array of (x, y) coordinates

    if isempty(points)
        line = [0, 0, 0];
        return;
    end

    % Convert to homogeneous coordinates
    N = size(points, 1);
    X = [points, ones(N, 1)];
    
    % Solve for the null space of X (minimizing ||Xw||^2)
    [~, ~, V] = svd(X);
    line = V(:, end)'; % Last column is the solution
end

function [lines, assignments, error] = kmeans_lines(points, K, max_iters, epsilon, init_method)
    % K-Means for line clustering
    % Parameters:
    % points - Nx2 array of points
    % K - Number of clusters
    % max_iters - Maximum number of iterations
    % epsilon - Convergence threshold
    % init_method - 'random_points' or 'random_lines'

    N = size(points, 1);
    assignments = zeros(N, 1);
    prev_error = Inf;

    % Initialization
    if strcmp(init_method, 'random_points')
        % Randomly assign points to clusters
        assignments = randi(K, N, 1);
    elseif strcmp(init_method, 'random_lines')
        % Randomly initialize lines
        lines = rand(K, 3) * 2 - 1;
    end

    for iter = 1:max_iters
        % E-Step: Assign points to nearest line
        if iter > 1 || strcmp(init_method, 'random_lines')
            distances = point_to_line_dist(points, lines);
            [~, assignments] = min(distances, [], 2);
        end

        % M-Step: Fit lines to assigned points
        lines = zeros(K, 3);
        for k = 1:K
            cluster_points = points(assignments == k, :);
            lines(k, :) = fit_line(cluster_points);
        end

        % Compute error
        distances = point_to_line_dist(points, lines);
        error = sum(min(distances, [], 2).^2);

        % Check for convergence
        if abs(prev_error - error) < epsilon
            break;
        end
        prev_error = error;
    end
end
