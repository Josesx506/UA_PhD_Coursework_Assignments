function [num_questions] = hw4(infile)

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

%% Part A
tent = imread(infile);
disp(size(tent));

f1 = figure;
imshow(tent);

datacursormode on;

hold on;

% Get the current axes
ax = gca;

% Set the axis labels for the first and second indices
xlabel(ax, 'Column Index (Second Index)');
ylabel(ax, 'Row Index (First Index)');

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

f2 = figure;
draw_pixel_bounds(tent, 100, 200, 2, [255 0 0], 'r');
exportgraphics(f2, 'output/f2_tent_red_pixed_100_200.png', 'Resolution', 200);

num_questions = num_questions + 1;


end