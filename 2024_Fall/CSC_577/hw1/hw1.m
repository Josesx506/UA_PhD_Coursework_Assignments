function [num_questions] = hw1(infile)
% This function reads in an image, performs some preprocessing
% and outputs several images within the active directory. 
% Additional text is also printed for some internal calculations
% that were required.

close all;
clc;
num_questions = 0;



%% Q1 - Q7
tent = imread(infile);
imwrite(tent, "out.jpg");
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
fprintf("B & W image dimensions are %d rows and %d columns.\n",size(bw_tent));
f1 = figure;
imshow(bw_tent);
imwrite(bw_tent, "bw_tent.jpg");

num_questions = num_questions + 1;

%%
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
fprintf("Linsolve: x = %.4f, y = %.4f, z = %.4f\n\n\n", X2(1), X2(2), X2(3));

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
fprintf("The eigenvalue k is: %.4f\n", k);



end

% num_q = hw1("tent.jpg");