function [num_questions] = hw1(infile)
% This function reads in an image, preprocesses them as arrays, and outputs 
% several images within the active directory. Additional text is printed 
% for some internal calculations that were required. Most of the syntax is 
% focused on manipulating image arrays, but the final two questions are 
% related to PCA analysis.
% Input arg is a string path to the tent figure, and the function returns
% the total number of questions that were answered.

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end


%% Q1 - Q7
figure;
tent = imread(infile);
imshow(tent);
imwrite(tent, "output/out.jpg");
num_questions = num_questions + 7;

%% Q8
fprintf("Question 8: Basic Image Data Structures\n")
disp(whos);
disp(whos("tent"));
[num_rows, num_cols, num_channels] = size(tent);
fprintf("The tent image has %d rows, %d columns, and %d channels.\n",num_rows, num_cols, num_channels);

% min max across channels
rmin = min(tent(:, :, 1),[],"all");
gmin = min(tent(:, :, 2),[],"all");
bmin = min(tent(:, :, 3),[],"all");
omin = min(tent(:));

rmax = max(tent(:, :, 1),[],"all");
gmax = max(tent(:, :, 2),[],"all");
bmax = max(tent(:, :, 3),[],"all");
omax = max(tent(:));

red_stats = [rmin; rmax];
grn_stats = [gmin; gmax];
blu_stats = [bmin; bmax];
ovr_stats = [omin; omax];

T = table(red_stats, grn_stats, blu_stats, ovr_stats, ...
    RowNames={'Min','Max'}, VariableNames={'Red','Green','Blue','Overall'});
fprintf("Min-Max across channels\n")
disp(T);


bw_tent = rgb2gray(tent);
fprintf("B & W image dimensions are %d rows and %d columns.\n\n\n",size(bw_tent));
f1 = figure;
imshow(bw_tent);
imwrite(bw_tent, "output/bw_tent.jpg");

num_questions = num_questions + 1;

%% Q9
figure;
f2 = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');

nexttile;
imshow(tent(:, :, 1));
title('Red');

nexttile;
imshow(tent(:, :, 2));
title('Green');

nexttile;
imshow(tent(:, :, 3)); 
title('Blue');

set(gca, 'LooseInset', get(gca, 'TightInset')); % Minimize whitespace
exportgraphics(f2, 'output/tent_rgb_channels.png', 'Resolution', 200);

flip_chans = tent;

flip_chans(:, :, 1) = tent(:,:,2);
flip_chans(:, :, 2) = tent(:,:,3);
flip_chans(:, :, 3) = tent(:,:,1);
figure;
imshow(flip_chans);
imwrite(flip_chans, "output/tent_flip_ch.png");

num_questions = num_questions + 1;

%% Q10
doub_bw_tent = double(bw_tent) / 255;
[nRows, nCols] = size(doub_bw_tent);

for row = 5:5:nRows
    for col = 5:5:nCols
        doub_bw_tent(row, col) = 1;
    end
end

f5 = figure;

subplot(1,2,1);
imagesc(doub_bw_tent);
title('imagesc()');
pbaspect([1 1 1]);

subplot(1,2,2);
imshow(doub_bw_tent);
title('imshow()');

set(gca, 'LooseInset', get(gca, 'TightInset'));
exportgraphics(f5, 'output/tent_checker_bw_for_loop.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Q11

redCh = reshape(tent(:, :, 1),[],1);
grnCh = reshape(tent(:, :, 2),[],1);
bluCh = reshape(tent(:, :, 3),[],1);

figure;
f6 = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');

nexttile;
histogram(redCh, 20, 'FaceColor', 'r');
title('Red');

nexttile;
histogram(grnCh, 20, 'FaceColor', 'g');
title('Green');

nexttile;
histogram(bluCh, 20, 'FaceColor', 'b');
title('Blue');

exportgraphics(f6, 'output/tent_rgb_hist.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Q12

f7 = figure;
x_pi = linspace(-pi,pi,100);
plot(x_pi,sin(x_pi), Color='r');
hold on;
plot(x_pi,cos(x_pi), Color='g');
legend("sin","cos");
hold off;

exportgraphics(f7, 'output/trig_waves.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Q13
% Define the matrix A
A = [3 4 1; 2 -1 2; 1 1 -1];

% Define the vector B
B = [9; 8; 0];

% Solve for X using the inverse of A
X1 = inv(A) * B;
X2 = linsolve(A,B);

% Display the solution
fprintf("Question 13: Playing with linear algebra\n")
fprintf("Inverse:  x = %.4f, y = %.4f, z = %.4f\n", X1(1), X1(2), X1(3));
fprintf("Linsolve: x = %.4f, y = %.4f, z = %.4f\n", X2(1), X2(2), X2(3));

val1 = (2*X1(1)) - (1*X1(2)) + (2*X1(3));
val2 = (2*X2(1)) - (1*X2(2)) + (2*X2(3));
if val1 == 8
    fprintf("Valid inversion.\n");
end

if (val1 - val2) == 0
    fprintf("Equal results.\n\n\n");
end

num_questions = num_questions + 1;

%% Q14
% Define the matrix A
A = [3.0 4.0 1.0; 
     3.1 2.9 0.9; 
     2.0 -1.0 2.0; 
     2.1 -1.1 2.0; 
     1.0 1.0 -1.0; 
     1.1 1.0 -0.9];

% Define the vector b
b = [9; 9; 8; 8; 0; 0];

% Compute the Moore-Penrose inverse
X = inv(A' * A) * A' * b;

% Display the solution using fprintf
fprintf("Question 14: Playing with linear algebra II\n")
fprintf("Moore-Penrose Inverse: x = %.4f, y = %.4f, z = %.4f\n", X(1), X(2), X(3));

% Calculate the error vector
error_vector = A * X - b;

% Calculate the magnitude of the error vector
error_magnitude = norm(error_vector);

% Display the magnitude of the error vector
fprintf("Magnitude of the error vector = %.4f\n\n\n", error_magnitude);

num_questions = num_questions + 1;

%% Q15
% Step 1: Create a random 4x4 matrix R
R = rand(4, 4);
fprintf("Question 15: Playing with linear algebra III\n")
fprintf("The input matrix R is:\n");
disp(R);

% Step 2: Create a symmetric matrix A
A = R * R';

% Step 3: Compute eigenvectors and eigenvalues of A
[V, D] = eig(A);

% Step 4: Select one eigenvector (let's choose the first one) and its corresponding eigenvalue
v = V(:, 1); % First eigenvector
k = D(1, 1); % Corresponding eigenvalue

% Step 5: Verify the eigenvector equation Av = kv
Av = A * v;      % Compute A*v
kv = k * v;      % Compute k*v
Av_over_v = Av ./ v; % Element-wise division of Av by v

% Step 6: Display the results
fprintf("The matrix A*v./v is:\n");
disp(Av_over_v);

% Additionally, you can print out the eigenvalue to verify consistency
fprintf("The eigenvalue k is: %.4f\n\n\n", k);

num_questions = num_questions + 1;

%% Q16
doub_bw_tent = double(bw_tent) / 255;
[nRows, nCols] = size(doub_bw_tent);

doub_bw_tent(5:5:nRows, 5:5:nCols) = 0;

f8 = figure;

subplot(1,2,1);
imagesc(doub_bw_tent);
title('imagesc()');
pbaspect([1 1 1]);

subplot(1,2,2);
imshow(doub_bw_tent);
title('imshow()');

set(gca, 'LooseInset', get(gca, 'TightInset'));
exportgraphics(f8, 'output/tent_checker_bw_vectorized.png', 'Resolution', 200);

doub_bw_tent = double(bw_tent) / 255;
mask = find(doub_bw_tent>0.5);
doub_bw_tent(mask) = 0;

f9 = figure;
imshow(doub_bw_tent);
title('Mask Pixels >0.5');
set(gca, 'LooseInset', get(gca, 'TightInset'));
exportgraphics(f9, 'output/tent_bw_0.5_mask.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Q17
pcaArr = readtable("pca.txt");
pcaArr.Properties.VariableNames = ["x","y"];

f10 = figure;
scatter(pcaArr.x,pcaArr.y);
title('PCA');xlabel('X');ylabel('Y');
set(gca, 'LooseInset', get(gca, 'TightInset'));
exportgraphics(f10, 'output/pca_raw.png', 'Resolution', 200);

data = table2array(pcaArr);
cov_mat = cov(data);
fprintf("Question 17: PCA\n");
fprintf("The original covariance matrix is \n");
disp(cov_mat);
fprintf("\n\n");

% Shift origin to center and rotate matrix
xCent = pcaArr.x - pcaArr.x(1);
yCent = pcaArr.y - pcaArr.y(1);
originCent = [xCent yCent];

[eigvec, eigval] = eig(cov_mat);
[sort_eigval, sort_idx] = sort(diag(eigval), 'descend');
rotX = originCent * eigvec(:, sort_idx(1));
rotY = originCent * eigvec(:, sort_idx(2));
originRot = [rotX,rotY];

% Shorter syntax
% coeff = pca(originCent);
% originRot = originCent * coeff;

f11 = figure;
scatter(originRot(:,1),originRot(:,2));
title('PCA Origin Centered');xlabel('shift-rotation X');ylabel('shift-rotation Y');
set(gca, 'LooseInset', get(gca, 'TightInset'));
axis equal;
exportgraphics(f11, 'output/pca_origin_cent.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Q18

fprintf("Question 18: PCA II\n");
centered_data = data - mean(data);

% Calculate eigenvectors and eigenvalues
[eigvec, eigval] = eig(cov(centered_data));

fprintf("The eigenvector matrix is \n");
disp(eigvec);

% Verify orthogonality 
ortho_check = eigvec' * eigvec;
disp('Orthogonality Check (should be Identity Matrix):');
disp(ortho_check);

% Rotate data using eigenvectors
transformed_data = centered_data * eigvec;

% Plot transformed data
f12 = figure;
scatter(transformed_data(:, 1), transformed_data(:, 2));
title('PCA Mean Centered');xlabel('PC1');ylabel('PC2');
set(gca, 'LooseInset', get(gca, 'TightInset'));
axis equal;
exportgraphics(f12, 'output/pca_mean_cent.png', 'Resolution', 200);

% Recompute covariance matrix for transformed data
transformed_covariance = cov(transformed_data);
disp('Transformed Covariance Matrix:');
disp(transformed_covariance);

% Sum of variances
variance_before = sum(diag(cov(centered_data)));
variance_after = sum(diag(cov(transformed_data)));
T = table(variance_before, variance_after, ...
    RowNames={'Variance Sum'}, VariableNames={'Before Transformation','After Transformation'});

disp(T);

num_questions = num_questions + 1;


end

% num_q = hw1("tent.jpg");