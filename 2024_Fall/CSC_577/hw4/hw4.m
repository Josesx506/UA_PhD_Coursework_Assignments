function [num_questions] = hw4(infile,infile2)

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

%% Part A
fprintf("Part A\n")
tent = imread(infile);
disp(['Image dimensions: ',num2str(size(tent))]);

f1 = figure('visible','off');
imshow(tent);

datacursormode on;

hold on;

% Get the current axes
ax = gca;

% Set the axis labels for the first and second indices
xlabel(ax, 'Column Index (Second Index) (pixels)');
ylabel(ax, 'Row Index (First Index) (pixels)');

% Set the title for clarification
title('Tent Image with Labeled Axes', 'FontSize', 16);

% Arrow properties (Length and HeadStyle)
arrowLength = 50;   % Length of the arrow
arrowHeadSize = 10; % Size of the arrowhead

% Draw an arrow for the row index (first index) direction (y-axis)
annotation('textarrow', [0.03 0.03], [0.9 0.8], 'String', 'Increasing Index', ...
    'FontSize', 12, 'Color', 'red', 'HeadLength', arrowHeadSize, 'TextRotation', 270);

% Draw an arrow for the column index (second index) direction (x-axis)
annotation('textarrow', [0.15 0.25], [0.03 0.03], 'String', 'Increasing Index', ...
    'FontSize', 12, 'Color', 'blue', 'HeadLength', arrowHeadSize);

% Adjust the axes to make the image show without distortion
axis on;
set(ax, 'XAxisLocation', 'bottom', 'YDir', 'reverse'); % Image coordinate style

hold off;
exportgraphics(f1, 'output/f1_annotated_tent.png', 'Resolution', 200);


function draw_pixel_bounds(img, row, col, box_size, pixelColor, boxColor)
    % Draws a box of size box_size around the point (row, col) with the given color.
    [rows, cols, ~] = size(img);
    
    % Update the pixels of the image
    img(row, col, :) = pixelColor;
    
    % Calculate the range for the box (ensure it doesn't go outside the image bounds)
    row_min = max(row - box_size/2, 1);
    row_max = min(row + box_size/2, rows);
    col_min = max(col - box_size/2, 1);
    col_max = min(col + box_size/2, cols);
    
    % Display the updated image
    imshow(img);
    % Draw a rectangle around the box
    drawrectangle('Position',[col_min,row_min,box_size,box_size],'Color',boxColor, 'FaceAlpha', 0);
    ax = gca;
    axis on;
    set(ax, 'XAxisLocation', 'bottom', 'YDir', 'reverse'); % Image coordinate style

end

f2 = figure('visible','off');
draw_pixel_bounds(tent, 100, 200, 2, [255 0 0], 'r');
xlabel('pixels');ylabel('pixels');
exportgraphics(f2, 'output/f2_tent_red_pixed_100_200.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Part B
fprintf("\nPart B - no output\n")
imCrds = readmatrix('image_coords.txt');


f3 = figure;
imshow(infile2);
hold on;
scatter(imCrds(:, 1), imCrds(:, 2), 30, 'w', 'filled');

datacursormode on;
hold off;
exportgraphics(f3, 'output/f3_selected_points.png', 'Resolution', 200);


%% Part C
fprintf("\nPart C\n")
wdCrds = readmatrix('world_coords.txt');
cmMat1 = readmatrix('camera_matrix_1.txt');
cmMat2 = readmatrix('camera_matrix_2.txt');

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

[cmm1Pnts,cmm1Rms] = project_camera_points(wdCrds,cmMat1,imCrds);
[cmm2Pnts,cmm2Rms] = project_camera_points(wdCrds,cmMat2,imCrds);

% Display the RMS error
fprintf('RMS Error Matrix 1: %.2f\n', cmm1Rms);
fprintf('RMS Error Matrix 2: %.2f\n', cmm2Rms);


f4 = figure('visible','off');
imshow(infile2);
hold on;
scatter(imCrds(:, 1), imCrds(:, 2), 40, 'red', 'filled');
scatter(cmm1Pnts(:, 1), cmm1Pnts(:, 2), 40, 'green', 'filled');
scatter(cmm2Pnts(:, 1), cmm2Pnts(:, 2), 40, 'blue', 'filled');
l = legend("ground truth","matrix 1","matrix 2");
set(l,'FontSize',16);
hold off;
exportgraphics(f4, 'output/f4_reprojected_points.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Part D
fprintf("\nPart D - output is limited to pdf\n")

num_questions = num_questions + 1;

%% Part E
fprintf("\nPart E\n")

tmat = [1 0 400; 0 1 600; 0 0 1]; % translation matrix
smat = [400 0 0; 0 400 0; 0 0 1]; % scaling matrix
fmat = [1 0 0; 0 -1 0; 0 0 1];    % flipping matrix
cmat = tmat * smat * fmat;        % composite transformation matrix
fprintf('The transformation matrix is \n');
disp(cmat);

p1 = cmat * [-0.5;-0.5;1];
p2 = cmat * [-0.5;0.5;1];
p3 = cmat * [0;1;1];
p4 = cmat * [1;0;1];
p5 = cmat * [0;-1;1];
fprintf("Transformed coordinates for (-0.5,-0.5) = (%d,%d)\n", p1(1), p1(2));
fprintf("Transformed coordinates for (-0.5,0.5) = (%d,%d)\n", p2(1), p2(2));
fprintf("Transformed coordinates for (0,1) = (%d,%d)\n", p3(1), p3(2));
fprintf("Transformed coordinates for additional point (1,0) = (%d,%d)\n", p4(1), p4(2));
fprintf("Transformed coordinates for additional point (0,-1) = (%d,%d)\n", p5(1), p5(2));

icmat = inv(cmat);
p6 = icmat * [1;1;1];
fprintf("Inverted coordinates for (1,1) = (%d,%d)\n", p6(1), p6(2));

xpnts = [400,p1(1),p2(1),p3(1),p4(1),p5(1)];
ypnts = [600,p1(2),p2(2),p3(2),p4(2),p5(2)];
labels = {'(0, 0)';'(-0.5, -0.5)';'(-0.5, 0.5)';'(0, 1)';'(1, 0)';'(0, -1)'};

% Plot scatter of X, Y points
f5 = figure('visible','off');
scatter(xpnts, ypnts, 100, 'filled'); % '100' sets the size of the points
xlabel('x-axis (pixels)');
ylabel('y-axis (pixels)');
title('Scatter Plot of translated coordinates');
grid on;
hold on;

% Add text labels at each point
for i = 1:length(xpnts)
    if i == 6
        text(xpnts(i), ypnts(i), labels{i}, 'VerticalAlignment', 'top', ...
            'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', 'b');
    else
        text(xpnts(i), ypnts(i), labels{i}, 'VerticalAlignment', 'bottom', ...
            'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', 'b');
    end
end

annotation('textarrow', [0.4 0.5], [0.52 0.52], 'String', ' ');
annotation('textarrow', [0.39 0.39], [0.5 0.4], 'String', ' ');
text(486, 593, 'x', 'FontSize', 14, 'Color', 'k', 'VerticalAlignment', 'baseline');
text(400, 470, 'y', 'FontSize', 14, 'Color', 'k', 'HorizontalAlignment', 'center');

hold off;
exportgraphics(f5, 'output/f5_transformed_points.png', 'Resolution', 200);


num_questions = num_questions + 1;

% close all;


end