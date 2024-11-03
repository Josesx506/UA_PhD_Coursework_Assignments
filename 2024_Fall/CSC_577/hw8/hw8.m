function [num_questions] = hw8()

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end


%% Part A
fprintf("Part A\n")

% Example 1
h1 = [0, -1, 1, 0, 0];
f1 = [0, 2, 2, 0, 0];
cnv1 = conv(h1, f1);
cor1 = conv(f1, flip(h1));
% Example 2
f2 = [0, 2, 1, 0, 0];
cnv2 = conv(h1, f2);
cor2 = conv(f2, flip(h1));
% Example 3
cnv3 = conv(f2, h1);
cor3 = conv(h1, flip(f2));

figure('Position',[1, 1, 900, 450], 'visible','off');
f0 = tiledlayout(3,4,'TileSpacing','Compact','Padding','Compact');
pad = 0.35; lw = 2; fs = 14;
% Example 1
nexttile;
stairs(1:length(h1),h1,'b-','LineWidth',lw); title('H (filter)','FontSize',fs);
xlim([1-pad, length(h1)+pad]); ylim([min(h1)-pad, max(h1)+pad]);
text(-0.5, -0.75, 'Example 1', 'Rotation', 90, 'FontSize', fs);
nexttile;
stairs(1:length(f1),f1,'color','#D95319','LineWidth',lw); title('F (function)','FontSize',fs);
xlim([1-pad, length(f1)+pad]); ylim([min(f1)-pad, max(f1)+pad]);
nexttile;
stairs(1:length(cor1),cor1,'g-','LineWidth',lw); title('Correlation','FontSize',fs);
xlim([1-pad, length(cor1)+pad]); ylim([min(cor1)-pad, max(cor1)+pad]);
nexttile;
stairs(1:length(cnv1),cnv1,'r-','LineWidth',lw); title('Convolution','FontSize',fs);
xlim([1-pad, length(cnv1)+pad]); ylim([min(cnv1)-pad, max(cnv1)+pad]);

% Example 2
nexttile;
stairs(1:length(h1),h1,'b-','LineWidth',lw);
xlim([1-pad, length(h1)+pad]); ylim([min(h1)-pad, max(h1)+pad]);
text(-0.5, -0.75, 'Example 2', 'Rotation', 90, 'FontSize', fs);
nexttile;
stairs(1:length(f2),f2,'color','#D95319','LineWidth',lw);
xlim([1-pad, length(f2)+pad]); ylim([min(f2)-pad, max(f2)+pad]);
nexttile;
stairs(1:length(cor2),cor2,'g-','LineWidth',lw);
xlim([1-pad, length(cor2)+pad]); ylim([min(cor2)-pad, max(cor2)+pad]);
nexttile;
stairs(1:length(cnv2),cnv2,'r-','LineWidth',lw);
xlim([1-pad, length(cnv2)+pad]); ylim([min(cnv2)-pad, max(cnv2)+pad]);

% Example 3
nexttile;
stairs(1:length(f2),f2,'b-','LineWidth',lw);
xlim([1-pad, length(f2)+pad]); ylim([min(f2)-pad, max(f2)+pad]);
text(-0.5, 0.25, 'Example 3', 'Rotation', 90, 'FontSize', fs);
nexttile;
stairs(1:length(h1),h1,'color','#D95319','LineWidth',lw);
xlim([1-pad, length(h1)+pad]); ylim([min(h1)-pad, max(h1)+pad]);
nexttile;
stairs(1:length(cor3),cor3,'g-','LineWidth',lw);
xlim([1-pad, length(cor3)+pad]); ylim([min(cor3)-pad, max(cor3)+pad]);
nexttile;
stairs(1:length(cnv3),cnv3,'r-','LineWidth',lw);
xlim([1-pad, length(cnv3)+pad]); ylim([min(cnv3)-pad, max(cnv3)+pad]);

exportgraphics(f0, 'output/f0_corr_vs_conv.png', 'Resolution', 200);

fprintf('1). The 3 class examples were replicated in matlab.\n');

num_questions = num_questions + 1;

%% Part B
fprintf("\nPart B\n")

% Problem 1
% Load the .tiff image
clbImg = imread('climber.tiff');

% Convert to grayscale and double
if size(clbImg, 3) == 3
    clbImg = double(rgb2gray(clbImg));
end

% Define the finite difference kernels for x and y gradients
% Simple Sobel-like operators for computing gradients
Gx = [-1 0 1; -2 0 2; -1 0 1]; % Gradient kernel in the x direction
Gy = [1 2 1; 0 0 0; -1 -2 -1]; % Gradient kernel in the y direction

% Perform convolution to calculate the gradients
Ix = conv2(clbImg, Gx, 'same'); % Gradient in the x direction
Iy = conv2(clbImg, Gy, 'same'); % Gradient in the y direction

% Calculate the gradient magnitude
gradMagnitude = sqrt(Ix.^2 + Iy.^2);

% Scale the result for better visualization
gradMagnitude = gradMagnitude / max(gradMagnitude(:)) * 255;

% Display the result
f1 = figure('visible','off');
imshow(uint8(gradMagnitude));
title('Gradient Magnitude of the Image');
exportgraphics(f1, 'output/f1_gradient_climber.png', 'Resolution', 200);


% Problem 2
% Define a threshold value
threshold = 35;

% Create a binary edge map based on the threshold
edgeMap = gradMagnitude > threshold;

% Display the binary edge map
f2 = figure('visible','off');
imshow(edgeMap);
title('Binary Edge Map');
exportgraphics(f2, 'output/f2_edge_detection.png', 'Resolution', 200);


% Problem 3
[x,y,gmask,smoothed_img] = gaussianSmooth2D(clbImg,2);

% Display the smoothed image
figure('Position',[1, 1, 600, 300],'visible','off');
f3 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; surf(x, y, gmask); title('Input Gaussian Filter with \sigma = 2');
xlabel('X'); ylabel('Y'); zlabel('Amplitude');
nexttile; imshow(uint8(smoothed_img)); title('Smoothed Image using Gaussian Filter');
exportgraphics(f3, 'output/f3_gaussian_smoothing.png', 'Resolution', 200);


% Problem 4
[~,edge_blurred] = edgeDetection(smoothed_img,70);

f4 = figure('visible','off');
imshow(edge_blurred);
title('Binary Edge Map of Smooth Image');
exportgraphics(f4, 'output/f4_gaussian_smoothing_edge_detect.png', 'Resolution', 200);


% Problem 5
[gm_combine,~] = blurredEdgeDetection(clbImg,4,20);
[~,~,~,ind_smooth] = gaussianSmooth2D(clbImg,4);
[gm_ind_sig4,~] = edgeDetection(ind_smooth,20);

figure('visible','off');
f5 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile;
imshow(uint8(gm_combine));
title('Combined Filter');
nexttile;
imshow(uint8(gm_ind_sig4));
title('Individual Filter');
title(f5, 'Gradient Magnitude of Blurred Images (\sigma = 4)'); % Suptitle
exportgraphics(f5, 'output/f5_filter_comparison.png', 'Resolution', 200);


% Problem 6
g1d = gaussianSmooth1D(clbImg,4);

figure('visible','off');
f6 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile;
imshow(uint8(g1d));
title('Convolution 1D');
nexttile;
imshow(uint8(ind_smooth));
title('Convolution 2D');
title(f6, 'Gaussian filter using different convolution (\sigma = 4)'); % Suptitle
exportgraphics(f6, 'output/f6_convolution_comparison.png', 'Resolution', 200);



f1d = @() gaussianSmooth1D(clbImg,20);
f2d = @() gaussianSmooth2D(clbImg,20);
runtime = timeit(f1d) - timeit(f2d);  % negative value indicates 1D is faster. abs() is used for display only
fprintf('6). The 1D convolution is faster than the 2D convolution for σ=20 by %.2f seconds.\n', abs(runtime));

num_questions = num_questions + 6;

%% Part C
fprintf("\nPart C\n")



[~,~,linkEdge1] = gradientEdgeDetector(clbImg,2,0.05,0.4);
[~,~,linkEdge2] = gradientEdgeDetector(clbImg,4,0.05,0.4);
[~,~,linkEdge3] = gradientEdgeDetector(clbImg,4,0.05,0.2);
fs = 14;

figure('Position',[1, 1, 750, 350],'visible','off');
f7 = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(linkEdge1); title('Thresh=0.4 & \sigma=2','FontSize',fs); 
axis on; set(gca, 'xtick', []); set(gca, 'ytick', []);
nexttile; imshow(linkEdge2); title('Thresh=0.4 & \sigma=4','FontSize',fs);
axis on; set(gca, 'xtick', []); set(gca, 'ytick', []);
nexttile; imshow(linkEdge3); title('Thresh=0.2 & \sigma=4','FontSize',fs);
axis on; set(gca, 'xtick', []); set(gca, 'ytick', []);

% Add labels to the top left corner of each tile
annotation('textbox',[0.055 0.83 0.1 0.1],'String','a)','EdgeColor','none','FontSize',fs);
annotation('textbox',[0.36 0.83 0.1 0.1],'String','b)','EdgeColor','none','FontSize',fs);
annotation('textbox',[0.665 0.83 0.1 0.1],'String','c)','EdgeColor','none','FontSize',fs);

title(f7,'Gradient edge detection using different scales and thresholds','FontSize',fs,'FontWeight','bold');
exportgraphics(f7, 'output/f7_gradient_based_edge_detec.png', 'Resolution', 200);

% disp(g1d);

[~,unlinkedEdge] = blurredEdgeDetection(clbImg,2,40);
unlinkedEdge = ~unlinkedEdge;


figure('visible','off');
f8 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; imshow(linkEdge1); title('Linked Edge Detector','FontSize',fs);
axis on; set(gca, 'xtick', []); set(gca, 'ytick', []);
nexttile; imshow(unlinkedEdge); title('Unlinked Edge Detector','FontSize',fs);
title(f8,'Effect of linking neighboring pixels on edge-detection','FontSize',fs,'FontWeight','bold');
exportgraphics(f8, 'output/f8_linked_vs_unlinked.png', 'Resolution', 200);

num_questions = num_questions + 1;


end

function [gradMag] = gaussianSmooth1D(inpImg,sigma)
    % Gaussian filter        
    kernel_size = ceil(6 * sigma);    % Reasonable size for the kernel (approx 6*sigma)
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1; % Ensure kernel size is odd
    end
    half_size = (kernel_size - 1) / 2;

    % Generate 1D Gaussian kernels for g(x) and h(y)
    x = -half_size:half_size;
    g = (1 / (sqrt(2 * pi) * sigma)) * exp(-x.^2 / (2 * sigma^2));
    g = g / sum(g); % Normalize g(x) to 1

    % Implement 1D convolution for g(x) and h(y) separately
    sx = conv2(inpImg, g, 'same');
    gradMag = conv2(sx, g', 'same');
end

function [x,y,G,gradMag] = gaussianSmooth2D(inpImg,sigma)
    % sigma - Standard deviation of the Gaussian

    % Gaussian filter        
    kernel_size = ceil(6 * sigma);    % Reasonable size for the kernel (approx 6*sigma)
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1; % Ensure kernel size is odd
    end
    half_size = (kernel_size - 1) / 2;
    
    % Generate the Gaussian mask using meshgrid
    [x, y] = meshgrid(-half_size:half_size, -half_size:half_size);
    G = (1 / (2 * pi * sigma^2)) * exp(-(x.^2 + y.^2) / (2 * sigma^2));
    
    % Normalize the Gaussian mask so it sums to 1
    G = G / sum(G(:));
    
    % Smooth the image by convolving with the Gaussian mask
    gradMag = conv2(inpImg, G, 'same');
end

function [gradMag,edgeMap] = edgeDetection(inpImg,thresh)
    % Simple Sobel-like operators for computing gradients
    Gx = [-1 0 1; -2 0 2; -1 0 1]; % Gradient kernel in the x direction
    Gy = [-1 -2 -1; 0 0 0; 1 2 1]; % Gradient kernel in the y direction

    Ix = conv2(inpImg, Gx, 'same');
    Iy = conv2(inpImg, Gy, 'same');

    gradMag = sqrt(Ix.^2 + Iy.^2);
    edgeMap = gradMag > thresh;
end

function [gradMag,edgeMap] = blurredEdgeDetection(inpImg,sigma,thresh)
    kernel_size = ceil(6 * sigma);
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
    end
    half_size = (kernel_size - 1) / 2;

    % Generate Gaussian filter
    [x, y] = meshgrid(-half_size:half_size, -half_size:half_size);
    G = (1 / (2 * pi * sigma^2)) * exp(-(x.^2 + y.^2) / (2 * sigma^2));
    G = G / sum(G(:)); % Normalize to 1

    % Define finite difference (Sobel-like) kernels
    Dx = [-1 0 1; -2 0 2; -1 0 1];
    Dy = [1 2 1; 0 0 0; -1 -2 -1];

    % Combine Gaussian and Sobel kernels
    Gx = conv2(G, Dx, 'same');
    Gy = conv2(G, Dy, 'same');

    % Apply the combined filters to estimage gradient
    Ix = conv2(inpImg, Gx, 'same');
    Iy = conv2(inpImg, Gy, 'same');
    
    % Calculate the gradient magnitude
    gradMag = sqrt(Ix.^2 + Iy.^2);
    edgeMap = gradMag > thresh;
end



function [grad_mag, nms, linked_edges] = gradientEdgeDetector(inpImg, sigma, lwthresh, hthresh)
    % GRADIENT BASED EDGE-DETECTOR: Gradient-based edge detection with non-max suppression and edge linking.
    % Input:
    %   img - Input image (grayscale).
    %   sigma - Standard deviation for Gaussian smoothing.
    %   thresh - Single threshold value for edge detection.
    % Output:
    %   grad_magnitude - Gradient magnitude of the image.
    %   non_max_suppressed - Gradient magnitude after non-maximum suppression.
    %   linked_edges - Binary edge map after edge linking.
    
    % Gaussion smoothing to reduce noise
    h = fspecial('gaussian', [6 * sigma, 6 * sigma], sigma);
    smoothImg = conv2(inpImg, h, 'same');

    % Compute image gradients using Sobel filters
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [1 2 1; 0 0 0; -1 -2 -1];
    Ix = conv2(smoothImg, Gx, 'same');
    Iy = conv2(smoothImg, Gy, 'same');

    % Calculate gradient magnitude and direction
    grad_mag = sqrt(Ix.^2 + Iy.^2);
    grad_dir = atan2(Iy, Ix) * (180 / pi); % Convert to degrees
    grad_dir(grad_dir < 0) = 360 + grad_dir(grad_dir < 0);
    
    % Non-maximum suppression
    [rows, cols] = size(grad_mag);
    
    % Find neighbors based on gradient direction angles
    neigh = zeros(rows, cols);
    for i = 1 :rows
        for j = 1:cols
            if ((grad_dir(i, j) >= 0 ) && (grad_dir(i, j) < 22.5) || (grad_dir(i, j) >= 157.5) && (grad_dir(i, j) < 202.5) || (grad_dir(i, j) >= 337.5) && (grad_dir(i, j) <= 360))
                neigh(i, j) = 0;
            elseif ((grad_dir(i, j) >= 22.5) && (grad_dir(i, j) < 67.5) || (grad_dir(i, j) >= 202.5) && (grad_dir(i, j) < 247.5))
                neigh(i, j) = 45;
            elseif ((grad_dir(i, j) >= 67.5 && grad_dir(i, j) < 112.5) || (grad_dir(i, j) >= 247.5 && grad_dir(i, j) < 292.5))
                neigh(i, j) = 90;
            elseif ((grad_dir(i, j) >= 112.5 && grad_dir(i, j) < 157.5) || (grad_dir(i, j) >= 292.5 && grad_dir(i, j) < 337.5))
                neigh(i, j) = 135;
            end
        end
    end

    nms = zeros(rows, cols);
    smag = sqrt(grad_mag);
    
    % Perform non-max suppression based on neighboring angles
    for i=2:rows-1
        for j=2:cols-1
            if (neigh(i,j)==0)
                nms(i,j) = (smag(i,j) == max([smag(i,j), smag(i,j+1), smag(i,j-1)]));
            elseif (neigh(i,j)==45)
                nms(i,j) = (smag(i,j) == max([smag(i,j), smag(i+1,j-1), smag(i-1,j+1)]));
            elseif (neigh(i,j)==90)
                nms(i,j) = (smag(i,j) == max([smag(i,j), smag(i+1,j), smag(i-1,j)]));
            elseif (neigh(i,j)==135)
                nms(i,j) = (smag(i,j) == max([smag(i,j), smag(i+1,j+1), smag(i-1,j-1)]));
            end
        end
    end

    nms = nms.*smag;

    lwthresh = lwthresh * max(nms(:));
    hthresh = hthresh * max(nms(:));

    strong_edges = nms >= hthresh;
    weak_edges = (nms >= lwthresh) & (nms < hthresh);

    % Initialize edges with strong edges
    edges = strong_edges;

    % Use a loop to link weak edges to strong edges
    for i = 2:rows-1
        for j = 2:cols-1
            if weak_edges(i, j)
                % Check if any 8-connected neighbor is a strong edge
                if any(any(strong_edges(i-1:i+1, j-1:j+1)))
                    edges(i, j) = 1; % Promote to strong edge
                end
            end
        end
    end
    
    % Flip the image to make background white
    edges = ~edges; 
    linked_edges = uint8(edges.*255);
end
