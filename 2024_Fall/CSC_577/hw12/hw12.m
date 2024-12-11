function [num_questions] = hw12()

close all;
clc;
num_questions = 0;
addpath('../hw9/');


% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

rng(557);

%% Part A
fprintf("Part A\n")

% Load data
data = load('line_data_2.txt');
x = data(:,1); y = data(:,2);

% Fit a line using RANSAC
inliers = fit_ransac(data, 2, 100, 0.2, 75);

% Fit line to inliers using homogeneous least squares
x_coords = inliers(:,1); y_coords = inliers(:,2);
one = ones(size(x_coords, 1), 1);
A = [x_coords, y_coords, one];
[~, ~, V] = svd(A);
vals = V(:,end);                % Coefficients a, b, c of ax + by + c = 0
a = vals(1); b = vals(2); c = vals(3);

% Estimate best fit line
x_fit = linspace(min(x_coords), max(x_coords), 100);
y_fit = -(a * x_fit + c) / b;

% Calculate error using perpendicular distance
error = sum(abs(a * x_coords + b * y_coords + c) / sqrt(a^2 + b^2));
fprintf('Line Equation: %.3fx + %.3fy + %.3f = 0\n', a, b, c);
fprintf('Error (sum of perpendicular distances): %.2f\n', error);
fprintf('Number of inliers: %d out of %d points\n', length(inliers), length(x));

% Create the plot
f1 = figure('visible','off');
scatter(x, y); hold on;
plot(x_fit, y_fit, 'r', 'LineWidth', 2);
title('RANSAC best fit line','FontSize',14); hold off;
exportgraphics(f1, 'output/f1_ransac.png', 'Resolution', 200);


num_questions = num_questions + 1;

%% Part B
fprintf("\nPart B\n")

% Label the matching points manually
% [slideI,frameI] = loadAndPadBWImgPairs('slide3.tiff','frame3.jpg');
% figure;
% subplot(1,2,1);
% imshow(uint8(slideI));
% subplot(1,2,2);
% imshow(uint8(frameI));
% datacursormode on;


% DLT Homography Calculation and RMS Error
numTrials = 10;
pointsInPlane = [4, 5, 6]; % Different point sets
errors = zeros(length(pointsInPlane), numTrials);

for p = 1:length(pointsInPlane)
    for t = 1:numTrials
        % Generate random points in [0,1]x[0,1]
        numPoints = pointsInPlane(p);
        srcPoints = rand(numPoints, 2);
        destPoints = rand(numPoints, 2);

        % Compute homography using DLT
        H_est = computeHomographyDLT(srcPoints, destPoints);

        % Map points using the estimated homography
        mappedPoints = applyHomography(H_est, srcPoints);

        % Calculate RMS error
        errors(p, t) = sqrt(mean(sum((mappedPoints - destPoints).^2, 2)));
    end
end

% Report average RMS error
avgErrors = mean(errors, 2);
for i = 1:length(pointsInPlane)
    fprintf('RMS Error for %d points (avg over 10 trials): %.3f\n', pointsInPlane(i), avgErrors(i));
end


% Process manually clicked points
for i = 1:3
    slide_name = strcat('slide', int2str(i), '.tiff');
    frame_name = strcat('frame', int2str(i), '.jpg');
    [col_slide, col_frame] = loadAndPadColorImgPairs(slide_name, frame_name);

    % Load points
    spoints = load(strcat('matches/slide', int2str(i), '.txt'));
    fpoints = load(strcat('matches/frame', int2str(i), '.txt'));
    xoffset = size(col_slide,2);

    % Subset of 4 points for homography
    srcPoints = spoints(4:7, :);
    destPoints = fpoints(4:7, :);

    % Compute homography using DLT
    H_est = computeHomographyDLT(srcPoints, destPoints);

    % Map all points using the estimated homography
    mappedPoints = applyHomography(H_est, spoints);

    % Visualization
    fg = figure('visible','off'); 
    imshow([col_slide,col_frame],[]); hold on;
    line([spoints(:,1)';mappedPoints(:,1)'+xoffset], [spoints(:,2)';mappedPoints(:,2)'],'color','yellow','LineWidth',1);
    scatter([spoints(:,1)';mappedPoints(:,1)'+xoffset], [spoints(:,2)';mappedPoints(:,2)'],200,'rs');
    scatter(fpoints(:,1)'+xoffset, fpoints(:,2)', 'm', 'filled'); hold off;
    fgname = sprintf('output/f%d_dlt_homo_fg_%d.png',1+i,i);
    exportgraphics(fg, fgname, 'Resolution', 200);
end


num_questions = num_questions + 1;


%% Part C
fprintf("\nPart C\n")

for i = 1:3
    path1 = strcat('slide', int2str(i), '.tiff');
    path2 = strcat('frame', int2str(i), '.jpg');
    mult = 1; N = 1; lw = 1; prune = true;
    distType = 'euclidean'; lc = 'y'; pct = 0.5;

    [col_slide,col_frame] = loadAndPadColorImgPairs(path1,path2);
    [slide,frame] = loadAndPadBWImgPairs(path1, path2);
    
    % Compute the SIFT features
    [f4_slide, d128_slide] = vl_sift(slide);
    [f4_frame, d128_frame] = vl_sift(frame);

    % vertical offset that checks the height of the image.
    offset = size(slide,2);
    
    % Get the index of the shortest euclidean distance
    [idxSlide,idxFrame] = smallestPercentageDistances(d128_slide,d128_frame,pct,distType,prune);

    % Subsample 30% points every N times
    idxSlide = idxSlide(1:N:end);
    idxFrame = idxFrame(1:N:end);

    spoints = f4_slide(1:2,idxSlide);
    fpoints = f4_frame(1:2,idxFrame);
    
    % Select 6 SIFT points for DLT Homography
    srcPoints = spoints';
    destPoints = fpoints';

    % Compute homography using DLT and RANSAC
    % H_est = computeHomographyDLT(srcPoints, destPoints);
    [H_rest, inlierIdx] = homographyWithRANSAC(srcPoints, destPoints, 30, 100);

    % Compute RMS errors
    rmsWithoutRANSAC = computeRMSError(srcPoints, destPoints, computeHomographyDLT(srcPoints, destPoints));
    rmsWithRANSAC = computeRMSError(srcPoints(inlierIdx, :), destPoints(inlierIdx, :), H_rest);
    
    % Print results
    fprintf('Image pair %d\n', i);
    fprintf('RMS Error without RANSAC: %.2f\n', rmsWithoutRANSAC);
    fprintf('RMS Error with RANSAC: %.3f\n', rmsWithRANSAC);
    
    % Map all points using the estimated homography
    mappedPoints = applyHomography(H_rest, spoints(:,inlierIdx)');
    
    fg = figure('visible','off');
    imshow([col_slide,col_frame],[]); hold on;
    line([spoints(1,inlierIdx);mappedPoints(:,1)'+offset], [spoints(2,inlierIdx);mappedPoints(:,2)'],'color','yellow','LineWidth',lw);
    scatter([spoints(1,inlierIdx);mappedPoints(:,1)'+offset], [spoints(2,inlierIdx);mappedPoints(:,2)'],200,'rs');
    fgname = sprintf('output/f%d_sift_dlt_homo_fg_%d.png',4+i,i);
    exportgraphics(fg, fgname, 'Resolution', 200);
end

num_questions = num_questions + 1;

end


%% Part A
function [inliers] = fit_ransac(data, n, k, threshold, d)
    numPoints = size(data, 1);
    bestInliers = [];
    
    for i = 1:k
        % Randomly sample 2 points
        indices = randperm(numPoints, n);
        sample = data(indices, :);
        
        % Fit a line to the sampled points
        [m, c] = fitLine(sample);
        
        % Calculate distances of all points to the line
        distances = abs(m * data(:,1) - data(:,2) + c) ./ sqrt(m^2 + 1);
        
        % Find inliers
        closePoints = data(distances <= threshold, :);
        
        % Update best model if it has more inliers
        if size(closePoints, 1) > size(bestInliers, 1) && size(closePoints, 1) >= d
            bestInliers = closePoints;
        end
    end
    
    inliers = bestInliers;
end

% Function: Fit a line given two points
function [m, b] = fitLine(sample)
    point1 = sample(1, :);
    point2 = sample(2, :);
    m = (point2(2) - point1(2)) / (point2(1) - point1(1));
    b = point1(2) - m * point1(1);
end

%% Part B
function [img] = loadSiftImg(path)
    % Load image and convert to B&W for SIFT analysis
    img = imread(path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    elseif size(img, 3) == 4
        img = rgb2gray(img(:,:,1:3));
    end
    img = single(img);
end

function [padIm1,padIm2] = padSiftImg(img1,img2)
    % Pad input B&W image arrays to get equal sizes for plotting SIFT
    % results

    % Get sizes of both images
    [h1, w1] = size(img1);
    [h2, w2] = size(img2);

    % Find target dimensions (maximum in each dimension)
    th = max(h1, h2);
    tw = max(w1, w2);

    padIm1 = zeros(th, tw,'like',img1);
    padIm2 = zeros(th, tw,'like',img2);

    padIm1(1:h1,1:w1) = img1;
    padIm2(1:h2,1:w2) = img2;
end

function [padIm1,padIm2] = loadAndPadBWImgPairs(path1,path2)
    slide = loadSiftImg(path1);
    frame = loadSiftImg(path2);
    [padIm1,padIm2] = padSiftImg(slide,frame);
end

function [padIm1,padIm2] = loadAndPadColorImgPairs(path1,path2)
    % Load color images for plotting purposes only
    % Images are zero-padded to maintain equal sizes
    % SIFT works with B&W images so no analysis is performed on these
    % images
    img1 = imread(path1);
    img2 = imread(path2);

    % Fix issues with alpha and grayscale inconsistency
    if size(img1, 3) == 1
        img1 = double(repmat(img1, [1, 1, 3]));
    elseif size(img1, 3) == 4
        img1 = double(img1(:,:,1:3));
    else
        img1 = double(img1);
    end
    
    if size(img2, 3) == 1
        img2 = double(repmat(img2, [1, 1, 3]));
    elseif size(img2, 3) == 4
        img2 = double(img2(:,:,1:3));
    else
        img2 = double(img2);
    end

    % Get sizes of both images
    [h1, w1, ~] = size(img1);
    [h2, w2, c] = size(img2);

    % Find target dimensions (maximum in each dimension)
    th = max(h1, h2);
    tw = max(w1, w2);

    padIm1 = zeros(th, tw, c,'like',img1);
    padIm2 = zeros(th, tw, c,'like',img2);

    padIm1(1:h1,1:w1,:) = img1;
    padIm2(1:h2,1:w2,:) = img2;
    
    % Convert because it seems to plot weirdly for some reason
    padIm1 = uint8(padIm1);
    padIm2 = uint8(padIm2);
end

function H = generateRandomHomography()
    % Random 3x3 matrix with normalized scale
    H = rand(3, 3);
    H = H / H(3, 3); % Normalize
end

function destPoints = applyHomography(H, srcPoints)
    % Apply homography to points
    srcPointsHomog = [srcPoints, ones(size(srcPoints, 1), 1)]';
    destPointsHomog = H * srcPointsHomog;
    destPoints = (destPointsHomog(1:2, :) ./ destPointsHomog(3, :))';
end

function H = computeHomographyDLT(srcPoints, destPoints)
    % Ensure points are in homogeneous coordinates
    n = size(srcPoints, 1);
    A = zeros(2*n, 9);

    for i = 1:n
        x = srcPoints(i, 1);
        y = srcPoints(i, 2);
        xp = destPoints(i, 1);
        yp = destPoints(i, 2);

        A(2*i-1, :) = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp];
        A(2*i, :)   = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp];
    end

    [~, ~, V] = svd(A);
    H = reshape(V(:, end), [3, 3])'; % Last column of V reshaped
    H = H / H(3, 3);                 % Normalize H
end

%% Part C
function [idxs1,idxs2] = smallestEuclideanDist(feat1, feat2)
    % Compute the shortest pairwise euclidean distance matrix between feat1 and feat2
    % feat1 and feat2 are the (n,128)-element feature vectors
    % Transpose is used because `pdist2` requires the number of columns to be equal
    distances = pdist2(double(feat1'), double(feat2'), 'euclidean');
    distances(distances==0) = inf;

    % Find the minimum value and its indices
    minVal = min(distances(:));
    [idxs1, idxs2] = find(distances == minVal);
end

function [idx_ft1, idx_ft2] = smallestPercentageDistances(feat1,feat2,pct,type,prune)
    % feat1 and feat2 - SIFT 128-element vectors for both images
    % pct - top n percentage of key points that should be plotted
    % type - distance type argument. Valid args are euclidean, cosine and chisquare
    % prune - filter the points by Lowe's ratio

    % Find the best percentage of key points
    if nargin < 3 || isempty(pct)
        pct = 0.2;
    end
    if nargin < 4 || isempty(type)
        type = 'euclidean';
    end
    if nargin < 5 || isempty(prune)
        prune = false;
    end
    
    % ---------------------    
    matches = zeros(size(feat1, 2), 2);
    matches(:, 1) = 1:size(feat1, 2);
    % Compute the distance to n nearest neighbors
    if strcmp(type, 'euclidean') || strcmp(type, 'cosine')
        [idxs, dist] = knnsearch(double(feat2'), double(feat1'),'K',2,'Distance',type);
    elseif strcmp(type, 'chisquare')
        eps = 1e-5;   % Epsilon to avoid division by zero
        chi2_dist_fun = @(x, y) 0.5 * sum((x - y).^2 ./ (x + y + eps), 2);
        [idxs, dist] = knnsearch(double(feat2'), double(feat1'),'K',2,'Distance',chi2_dist_fun);
    else
        [idxs, dist] = knnsearch(double(feat2'), double(feat1'),'K',2,'NSMethod', 'kdtree');
    end
    matches(:, 2) = idxs(:, 1);

    % Calculate Lowe's ratio
    confidences = dist(:, 1) ./ dist(:, 2);
    if prune
        goodMatches = (confidences <= 0.8);
        matches = matches(goodMatches, :);
        confidences = confidences(goodMatches);
    end

    % Sort the matches so that the most confident onces are at the top of the list
    [~, ind] = sort(confidences, 'ascend');
    matches = matches(ind, :);
    
    numTop = round(pct * length(ind)); % Number of points in the top pct
    % Keep only the top percentage of matches
    selectedMatches = matches(1:numTop, :);
    idx_ft1 = selectedMatches(:,1);
    idx_ft2 = selectedMatches(:,2);
end

function [H_ransac, inlierIdx] = homographyWithRANSAC(srcPoints, destPoints, threshold, maxIter)
    % Inputs:
    %   srcPoints  - Nx2 matrix of source points
    %   destPoints - Nx2 matrix of destination points
    %   threshold  - Distance threshold for inliers
    %   maxIter    - Maximum number of RANSAC iterations
    % Outputs:
    %   H_ransac   - Homography matrix computed with inliers
    %   inlierIdx  - Logical index of inliers

    numPoints = size(srcPoints, 1);
    bestInliers = 0;
    bestH = eye(3);

    for i = 1:maxIter
        % Randomly sample 4 points
        idx = randperm(numPoints, 4);
        sampledSrc = srcPoints(idx, :);
        sampledDest = destPoints(idx, :);

        % Compute homography for the sampled points
        H = computeHomographyDLT(sampledSrc, sampledDest);

        % Map all source points using the estimated homography
        mappedPoints = applyHomography(H, srcPoints);

        % Compute reprojection error
        errors = sqrt(sum((mappedPoints - destPoints).^2, 2));

        % Find inliers
        inlierIdx = errors < threshold;
        numInliers = sum(inlierIdx);

        % Update the best model if current one has more inliers
        if numInliers > bestInliers
            bestInliers = numInliers;
            bestH = H;
        end
    end

    % Recompute homography using all inliers
    finalInlierIdx = errors < threshold;
    H_ransac = computeHomographyDLT(srcPoints(finalInlierIdx, :), destPoints(finalInlierIdx, :));
    inlierIdx = finalInlierIdx;
end

function error = computeRMSError(srcPoints, destPoints, H)
    mappedPoints = applyHomography(H, srcPoints);
    error = sqrt(mean(sum((mappedPoints - destPoints).^2, 2)));
end