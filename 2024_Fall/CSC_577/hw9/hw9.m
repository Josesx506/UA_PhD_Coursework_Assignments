function [num_questions] = hw9()

close all;
clc;
num_questions = 0;



% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end


%% Part A
fprintf("Part A\n")

% Question 1
createBestPointCollage('slide1.tiff','frame1.jpg','output/f01_img1_best_eucl_dist.png','y',2);
createBestPointCollage('slide2.tiff','frame2.jpg','output/f02_img2_best_eucl_dist.png','g',2,1);
createBestPointCollage('slide3.tiff','frame3.jpg','output/f03_img3_best_eucl_dist.png','r',2,2);

% N pairs
prb = 1;
npnts = 5;
[~] = createNPointsCollage('slide1.tiff','frame1.jpg','output/f04_img1_eucl_dist_5N.png',npnts,prb,'euclidean','y',1);
[~] = createNPointsCollage('slide2.tiff','frame2.jpg','output/f05_img2_eucl_dist_5N.png',npnts,prb,'euclidean','g',1);
[~] = createNPointsCollage('slide3.tiff','frame3.jpg','output/f06_img3_eucl_dist_5N.png',npnts,prb,'euclidean','r',1);


% Question 2 - Plot every N=5 points for the top 20% of matches
prb = 0.2;
npnts = 5;
% a - Euclidean Distance
[~] = createNPointsCollage('slide1.tiff','frame1.jpg','output/f07_img1_top20pct_eucl_dist_5N.png',npnts,prb,'euclidean','y',1);
[~] = createNPointsCollage('slide2.tiff','frame2.jpg','output/f08_img2_top20pct_eucl_dist_5N.png',npnts,prb,'euclidean','g',1);
[~] = createNPointsCollage('slide3.tiff','frame3.jpg','output/f09_img3_top20pct_eucl_dist_5N.png',npnts,prb,'euclidean','r',1);
% b - Cosine Distance
[~] = createNPointsCollage('slide1.tiff','frame1.jpg','output/f10_img1_top20pct_cosi_dist_5N.png',npnts,prb,'cosine','y',1);
[~] = createNPointsCollage('slide2.tiff','frame2.jpg','output/f11_img2_top20pct_cosi_dist_5N.png',npnts,prb,'cosine','g',1);
[~] = createNPointsCollage('slide3.tiff','frame3.jpg','output/f12_img3_top20pct_cosi_dist_5N.png',npnts,prb,'cosine','r',1);
% c - ChiSquare Distance
[~] = createNPointsCollage('slide1.tiff','frame1.jpg','output/f13_img1_top20pct_chsq_dist_5N.png',npnts,prb,'chisquare','y',1);
[~] = createNPointsCollage('slide2.tiff','frame2.jpg','output/f14_img2_top20pct_chsq_dist_5N.png',npnts,prb,'chisquare','g',1);
[~] = createNPointsCollage('slide3.tiff','frame3.jpg','output/f15_img3_top20pct_chsq_dist_5N.png',npnts,prb,'chisquare','r',1);

% Question 3 - pruning with the Lowe Ratio threshold
prb = 0.3;
npnts = 1;
[~] = createNPointsCollage('slide1.tiff','frame1.jpg','output/f16_img1_top30pct_eucl_dist_1N_prune.png',npnts,prb,'euclidean','y',1,1,true);
[~] = createNPointsCollage('slide2.tiff','frame2.jpg','output/f17_img2_top30pct_eucl_dist_1N_prune.png',npnts,prb,'euclidean','g',1,1,true);
[~] = createNPointsCollage('slide3.tiff','frame3.jpg','output/f18_img3_top30pct_eucl_dist_1N_prune.png',npnts,prb,'euclidean','r',1,1,true);

% Question 4 - Plot confusion matrix
prb = 1;
npnts = 1;
conf_mat = zeros(3,3);
for i=1:3
    for j=1:3
        slide_fl = sprintf('slide%d.tiff', i);
        frame_fl = sprintf('frame%d.jpg', j);
        if i == 1 && j==2
            conf_mat(i,j) = createNPointsCollage(slide_fl,frame_fl,'output/f19_anomalous_figure.png',npnts,prb,'euclidean','y',1,1,true);
        else
            conf_mat(i,j) = createNPointsCollage(slide_fl,frame_fl,false,npnts,prb,'euclidean','y',1,1,true);
        end
    end
end

f2 = figure('visible','off'); 
confusionchart(conf_mat,'FontSize',16,'XLabel','Frames','YLabel','Slides', ...
    'Title','Confusion Matrix for Euclidean Error with Pruning');
exportgraphics(f2, 'output/f20_confusion_matrix.png', 'Resolution', 200);


% QC Results with VLFeat plotting and matching code
% slide1 = loadSiftImg('slide3.tiff');
% frame1 = loadSiftImg('frame3.jpg');
% [slide1,frame1] = padSiftImg(slide1,frame1);
% [fs1, ds1] = vl_sift(slide1);
% [ff1, df1] = vl_sift(frame1);
% f1 = figure;
% hold on;
% imshow(uint8(slide1)); 
% h = vl_plotframe(fs1(:,212)); %360 216 212
% set(h,'color','y','linewidth',2);
% hold off;
% f2 = figure;
% hold on;
% imshow(uint8(frame1)); 
% h = vl_plotframe(ff1(:,229)); %139 249 229
% set(h,'color','y','linewidth',2);
% hold off;

num_questions = num_questions + 4;

%% Part B
fprintf("\nPart B\n")


harrisCornerDetector('checkbrd.jpg', 0.2, 2, 0.01, 36, 15, '-small-scale');
harrisCornerDetector('checkbrd.jpg', 0.2, 20, 0.01, 36, 15, '-large-scale');
harrisCornerDetector('checkbrd.jpg', 0.3, 20, 0.01, 36, 15, '-large-k');
harrisCornerDetector('indoor.jpg', 0.2, 10, 0.01, 100, 20);
harrisCornerDetector('indoor-rotate.jpg', 0.2, 10, 0.01, 100, 20);
harrisCornerDetector('outdoor.jpg', 0.2, 2, 0.01, 50, 15, '-small-scale');
harrisCornerDetector('outdoor.jpg', 0.2, 10, 0.01, 50, 15, '-large-scale');
harrisCornerDetector('downtown.jpg', 0.2, 10, 0.01, 150, 20);


% QC results with MATLAB function
% I = imread('checkbrd.jpg');
% corners = detectHarrisFeatures(rgb2gray(I));
% figure; imshow(I); hold on;
% plot(corners.selectStrongest(50));
% hold off;

num_questions = num_questions + 1;

end

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


function [idxs1,idxs2] = smallestEuclideanDist(feat1, feat2)
    % Compute the shortest pairwise euclidean distance matrix between feat1 and feat2
    % feat1 and feat2 are the (n,128)-element feature vectors
    % Transpose is used because `pdist2` requires the number of columns to be equal
    distances = pdist2(double(feat1'), double(feat2'), 'euclidean');
    distances(distances==0) = inf;

    % Find the minimum value and its indices
    minVal = min(distances(:));
    [idxs1, idxs2] = find(distances == minVal);
    
    % % Find the minimum distance and its linear index in the distance matrix
    % [~, minIndex] = min(distances(:));
    % % Convert the linear index to row and column indices
    % [idxs1, idxs2] = ind2sub(size(distances), minIndex);
end

function [idx_ft1, idx_ft2] = smallestPercentageDistances(feat1,feat2,pct,type,prune)
    % feat1 and feat2 - SIFT 128-element vectors for both images
    % pct - top n percentage of key points that should be plotted
    % type - distance type argument. Valid args are euclidean, cosine and
    % chisquare
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
    % ---------------------
    
    % Vlfeat method for QC
    % thresh = 2;
    % [matches, ~] = vl_ubcmatch(feat1, feat2, thresh);
    % idx_ft1 = matches(1,:);
    % idx_ft2 = matches(2,:);
    % ---------------------
end

function [x0,y0,x1,y1,cx,cy,dx,dy] = calculateCircularKeyPoints(fmat1,fmat2,vOffset,mult)
    % Create a matrix for plotting the directional vectors as quiver points
    % and circles. This assumes that the vertical offset is used i.e. images 
    % are stacked vertically.
    % x0,y0,x1,y1 - start and end points for directional vector
    % cx,cy - circular coordinates to plot circle
    % dx,dy - change in x and y to create magnitude and direction for
    % quiver lots
    x0 = [fmat1(1,:),fmat2(1,:)];
    y0 = [fmat1(2,:),fmat2(2,:)+vOffset;];
    scale = [fmat1(3,:),fmat2(3,:)]*mult;
    direction = [fmat1(4,:),fmat2(4,:)];

    x1 = x0 - scale .* sin(direction);
    y1 = y0 + scale .* cos(direction);

    dx = - scale/mult .* sin(direction);
    dy = scale/mult .* cos(direction);

    % Generate circle points
    t = (0:0.1:2*pi)';  % Column vector for broadcasting
    npts = length(t);
    nkeypnts = length(x0);

    % Create matrices for broadcasting
    r = 1 * scale;              % Radius for each keypoint
    X = repmat(x0, npts, 1);    % Repeat x coordinates
    Y = repmat(y0, npts, 1);    % Repeat y coordinates
    R = repmat(r, npts, 1);     % Repeat radii
    T = repmat(t, 1, nkeypnts); % Repeat angle values

    % Generate circle coordinates
    cx = R .* cos(T) + X;  % Matrix of x coordinates for circles
    cy = R .* sin(T) + Y;  % Matrix of y coordinates for circles
end

function [] = createBestPointCollage(path1,path2,out_file,lc,lw,mult)
    % Plot the Collage for only the shortest Euclidean distance based
    % on the 128-element vector from SIFR
    if nargin < 6
        mult = 2; % Default - increase keypoint plot scale by 2
    end

    [sld_col,frm_col] = loadAndPadColorImgPairs(path1,path2);

    % Load images based on filepaths
    slide = loadSiftImg(path1);  % high-res ppt slide
    frame = loadSiftImg(path2);  % low-res jpg image
    
    % Pad the images so that theyre the same size for plotting purposes
    [slide,frame] = padSiftImg(slide,frame);
    
    % Compute the SIFT features
    [f4_slide, d128_slide] = vl_sift(slide);
    [f4_frame, d128_frame] = vl_sift(frame);

    % Get the index of the shortest euclidean distance
    % [idxSlide,idxFrame] = smallestEuclideanDist(d128_slide,d128_frame);
    [idxSlide,idxFrame] = smallestPercentageDistances(d128_slide,d128_frame,0.1,'euclidean');
    idxSlide = idxSlide(1);
    idxFrame = idxFrame(1);

    disp([idxSlide,idxFrame]);

    f1match = f4_slide(:,idxSlide);
    % d1match = d128_slide(:,idxSlide);
    
    f2match = f4_frame(:,idxFrame);
    % d2match = d128_frame(:,idxFrame);

    % vertical offset that checks the height of the image.
    offset = size(slide,1);
    % Create the quiver points for the directional vectors
    [x0,y0,x1,y1,cx,cy,dx,dy] = calculateCircularKeyPoints(f1match,f2match,offset,mult);
    
    figure('visible','off');
    f1 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
    nexttile; hold on;
    imshow([sld_col;frm_col],[]);
    quiver(x0,y0,dx,dy,0.2,'LineWidth',lw,'color',lc);
    line([x0;x1],[y0;y1], 'Color', lc, 'LineWidth', lw); % Plot directional vector
    % plot circles to show orientation
    for i = 1:size(cx,2)
        plot(cx(:,i), cy(:,i), lc, 'LineWidth', lw);
    end
    hold off;

    % Plot the connecting lines showing matching points for the best euclidean distance
    nexttile;  hold on;
    imshow([sld_col;frm_col],[]);
    line([f1match(1,:);f2match(1,:)], [f1match(2,:);f2match(2,:)+offset],'color',lc,'LineWidth',lw);
    scatter([f1match(1,:);f2match(1,:)], [f1match(2,:);f2match(2,:)+offset],'ms'); % endpoints are magenta squares
    hold off;
    exportgraphics(f1, out_file, 'Resolution', 200);
end

function [nMatches] = createNPointsCollage(path1,path2,out_file,N,pct,distType,lc,lw,mult,prune)
    % Create a collage of N points. This doesn't rank the points to get the
    % best N. It only samples a subset of points with Euclidean distance <20% 
    % of the point distribution, and it plots a N subset of the to 20% points 
    % to show the key points and match them accordingly.

    if nargin < 9 || isempty(mult)
        mult = 1; % Default
    end
    if nargin < 10 || isempty(prune)
        prune = false;
    end

    [sld_col,frm_col] = loadAndPadColorImgPairs(path1,path2);
    
    % Load images based on filepaths
    slide = loadSiftImg(path1);  % high-res ppt slide
    frame = loadSiftImg(path2);  % low-res jpg image
    
    % Pad the images so that theyre the same size for plotting purposes
    [slide,frame] = padSiftImg(slide,frame);
    
    % Compute the SIFT features
    [f4_slide, d128_slide] = vl_sift(slide);
    [f4_frame, d128_frame] = vl_sift(frame);

    % vertical offset that checks the height of the image.
    offset = size(slide,1);

    % Get the index of the shortest euclidean distance
    [idxSlide,idxFrame] = smallestPercentageDistances(d128_slide,d128_frame,pct,distType,prune);
    nMatches = length(idxSlide);

    % Subsample the to 20% points every N times
    idxSlide = idxSlide(1:N:end);
    idxFrame = idxFrame(1:N:end);

    f1match = f4_slide(:,idxSlide);
    f2match = f4_frame(:,idxFrame);
    
    [x0,y0,x1,y1,cx,cy,dx,dy] = calculateCircularKeyPoints(f1match,f2match,offset,mult);
    
    figure('visible','off');
    f1 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
    nexttile; hold on;
    imshow([sld_col;frm_col],[]);
    quiver(x0,y0,dx,dy,0.2,'LineWidth',lw,'color',lc);
    line([x0;x1],[y0;y1], 'Color', lc, 'LineWidth', lw); % Plot directional vector
    % plot circles to show orientation
    for i = 1:size(cx,2)
        plot(cx(:,i), cy(:,i), lc, 'LineWidth', lw);
    end
    hold off;
    
    % Plot the connecting lines showing matching points for the best euclidean distance
    nexttile;  hold on;
    imshow([sld_col;frm_col],[]);
    line([f1match(1,:);f2match(1,:)], [f1match(2,:);f2match(2,:)+offset],'color',lc,'LineWidth',lw);
    scatter([f1match(1,:);f2match(1,:)], [f1match(2,:);f2match(2,:)+offset],'ms'); % endpoints are magenta squares
    hold off;
    if out_file
        exportgraphics(f1, out_file, 'Resolution', 200);
    end
end


function harrisCornerDetector(image_path, k, sigma, threshold, top_n, msz, suffix)
    % k - Harris corner K-parameter
    % sigma - gaussian sigma. Non max-suppression window is equivalent to
    % 2(sigma) + 1
    % threshold - designed to be percentage of max R
    % top_n - filter only the top n-points to minimize spurious points
    % msz - marker size
    % suffix - naming suffix
    if nargin < 6 || isempty(msz)
        msz = 15;
    end
    if nargin < 7 || isempty(suffix)
        suffix = '';
    end
    
    % Load and convert image to grayscale
    cimg = imread(image_path);
    if size(cimg, 3) == 3
        img = rgb2gray(cimg);
        img = double(img);
    else
        img = double(cimg);
    end
    
    % Compute gradients using Sobel filters
    Ix = imfilter(img, fspecial('sobel')', 'replicate');
    Iy = imfilter(img, fspecial('sobel'), 'replicate');
    
    % Compute products of derivatives
    Ixx = Ix .^ 2;
    Iyy = Iy .^ 2;
    Ixy = Ix .* Iy;
    
    % Apply Gaussian filter to the products
    g = fspecial('gaussian', 2 * round(3 * sigma) + 1, sigma);
    Sxx = imfilter(Ixx, g);
    Syy = imfilter(Iyy, g);
    Sxy = imfilter(Ixy, g);
    
    % Harris corner response calculation
    R = (Sxx .* Syy - Sxy .^ 2) - k * (Sxx + Syy) .^ 2;
    % disp([min(R(:)),max(R(:))]);
    threshold = threshold * max(R(:));

    % Threshold the corner response
    R(R < threshold) = 0;

    % Apply non-maximal suppression using a maximum filter
    nms_neighbor_size = (2*sigma)+1;                                   % Modify the neighborhood size
    R_max = ordfilt2(R, nms_neighbor_size^2, ones(nms_neighbor_size)); % Find local maxima in the neighborhood
    R = (R == R_max) .* R;                                             % Retain only local maxima that pass the threshold
    
    % Find the top n corners by response value
    [~, sorted_indices] = sort(R(:), 'descend');
    top_indices = sorted_indices(1:top_n);
    
    % Convert linear indices to (row, col) coordinates
    [row, col] = ind2sub(size(R), top_indices);

    % Create the output file name
    fname  = split(image_path,".");
    fname = sprintf('output/%s%s.png', char(fname(1)),suffix);
    
    % Display the results
    f3 = figure('visible','off');
    imshow(uint8(cimg)); hold on;
    plot(col, row, 'rs', 'MarkerSize', msz, 'MarkerFaceColor','r');
    th = title(['Top ', num2str(top_n), ' Harris Corners (k=', num2str(k), ', \sigma=', num2str(sigma), ', threshold=', num2str(round(threshold)), ')'], ...
        'FontSize',round(msz));
    th.Position(2) = th.Position(2) + 0.51;
    hold off;
    exportgraphics(f3, fname, 'Resolution', 200);
end
