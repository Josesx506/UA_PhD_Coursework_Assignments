
close all;
clc;

numPoints = 7;
xy1 = rand(numPoints, 20);
xy2 = rand(numPoints, 20);
distances = pdist2(xy1, xy2);
% Set 0's to inf since we don't want to find the min
% distance of a point to itself, which is 0.
distances(distances==0) = inf
% Find min distance
minDistance = min(distances(:));
% Find row and column where it occurs.
[row1, row2] = find(distances == minDistance);
% Plot all points
plot(xy1(:, 1), xy1(:, 2), 'r.', 'MarkerSize', 30); % Plot set 1.
hold on;
plot(xy2(:, 1), xy2(:, 2), 'b.', 'MarkerSize', 30); % Plot set 1.
% Plot the line
x1 = xy1(row1, 1);
y1 = xy1(row1, 2);
x2 = xy2(row2, 1);
y2 = xy2(row2, 2);
plot([x1, x2], [y1, y2], 'k-', 'LineWidth', 2);
grid on;
legend('Set 1', 'Set 2', 'Closest Pair');
caption = sprintf('Min Distance = %.4f', minDistance);
title(caption, 'fontSize', 20);