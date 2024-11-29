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
ss_5 = kmeans('sunset.tiff', 5, 100, 1e-4, spatial, 0, verbose);
ss_10 = kmeans('sunset.tiff', 10, 100, 1e-4, spatial, 0, verbose);

figure('Position',[1, 1, 600, 250], 'visible','off');
f1 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(ss_5); title('K = 5','FontSize',fs);
nexttile; imshow(ss_10); title('K = 10','FontSize',fs);
exportgraphics(f1, 'output/f1_sunset_kdiff.png', 'Resolution', 200);

tgr1_5 = kmeans('tiger-1.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr1_10 = kmeans('tiger-1.tiff', 10, 100, 1e-4, spatial, 0, verbose);

figure('Position',[1, 1, 600, 250], 'visible','off');
f2 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(tgr1_5); title('K = 5','FontSize',fs);
nexttile; imshow(tgr1_10); title('K = 10','FontSize',fs);
exportgraphics(f2, 'output/f2_tiger1_kdiff.png', 'Resolution', 200);

tgr2_5 = kmeans('tiger-2.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr2_10 = kmeans('tiger-2.tiff', 10, 100, 1e-4, spatial, 0, verbose);

figure('Position',[1, 1, 600, 250], 'visible','off');
f3 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(tgr2_5); title('K = 5','FontSize',fs);
nexttile; imshow(tgr2_10); title('K = 10','FontSize',fs);
exportgraphics(f3, 'output/f3_tiger2_kdiff.png', 'Resolution', 200);


% Question A2
spatial = true;

ss_5_l0 = kmeans('sunset.tiff', 5, 100, 1e-4, spatial, 0, verbose);
ss_5_l1 = kmeans('sunset.tiff', 5, 100, 1e-4, spatial, 1, verbose);
ss_5_l10 = kmeans('sunset.tiff', 5, 100, 1e-4, spatial, 10, verbose);
ss_10_l0 = kmeans('sunset.tiff', 10, 100, 1e-4, spatial, 0, verbose);
ss_10_l1 = kmeans('sunset.tiff', 10, 100, 1e-4, spatial, 1, verbose);
ss_10_l10 = kmeans('sunset.tiff', 10, 100, 1e-4, spatial, 10, verbose);

tgr1_5_l0 = kmeans('tiger-1.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr1_5_l1 = kmeans('tiger-1.tiff', 5, 100, 1e-4, spatial, 1, verbose);
tgr1_5_l10 = kmeans('tiger-1.tiff', 5, 100, 1e-4, spatial, 10, verbose);
tgr1_10_l0 = kmeans('tiger-1.tiff', 10, 100, 1e-4, spatial, 0, verbose);
tgr1_10_l1 = kmeans('tiger-1.tiff', 10, 100, 1e-4, spatial, 1, verbose);
tgr1_10_l10 = kmeans('tiger-1.tiff', 10, 100, 1e-4, spatial, 10, verbose);

tgr2_5_l0 = kmeans('tiger-2.tiff', 5, 100, 1e-4, spatial, 0, verbose);
tgr2_5_l1 = kmeans('tiger-2.tiff', 5, 100, 1e-4, spatial, 1, verbose);
tgr2_5_l10 = kmeans('tiger-2.tiff', 5, 100, 1e-4, spatial, 10, verbose);
tgr2_10_l0 = kmeans('tiger-2.tiff', 10, 100, 1e-4, spatial, 0, verbose);
tgr2_10_l1 = kmeans('tiger-2.tiff', 10, 100, 1e-4, spatial, 1, verbose);
tgr2_10_l10 = kmeans('tiger-2.tiff', 10, 100, 1e-4, spatial, 10, verbose);

% Create plots
plot_klambdas({ss_5_l0,ss_5_l1,ss_5_l10,ss_10_l0,ss_10_l1,ss_10_l10}, ...
    [5,10],[0,1,10],'output/f4_sunset_kms_spt.png');

plot_klambdas({tgr1_5_l0,tgr1_5_l1,tgr1_5_l10,tgr1_10_l0,tgr1_10_l1,tgr1_10_l10}, ...
    [5,10],[0,1,10],'output/f5_tiger1_kms_spt.png');

plot_klambdas({tgr2_5_l0,tgr2_5_l1,tgr2_5_l10,tgr2_10_l0,tgr2_10_l1,tgr2_10_l10}, ...
    [5,10],[0,1,10],'output/f6_tiger2_kms_spt.png');


% Question A3




num_questions = num_questions + 1;

%% Part B
fprintf("\nPart B\n")



num_questions = num_questions + 1;

end


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

function [segmented_img] = kmeans(image_path, K, max_iters, epsilon, spatial, lambda, verbose)
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

