function [] = hw7()

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

imgdir = 'color_constancy_images/';
max_scale = 250;


%% Part A
fprintf("Part A\n")

% Load light directions
mcbth = prerocess_image(strcat(imgdir,'macbeth_syl-50MR16Q.tif'), 'uint8');
mcbth_solux = prerocess_image(strcat(imgdir,'macbeth_solux-4100.tif'), 'uint8');

% Plot to interactively get white pixel bounds
% figure;
% imshow(mcbth);
% datacursormode on;

% Problem 1
mcbth_illm_col = get_canonical_light_estimate(mcbth,320,386,88,156,max_scale);
fprintf('1). The illuminant light color for macbeth_syl-50MR16Q is \n');
disp(round(mcbth_illm_col));

% Problem 2
mcbth_slx_illm_col = get_canonical_light_estimate(mcbth_solux,320,386,88,156,max_scale);
fprintf('2). The illuminant light color for macbeth_solux-4100 is \n');
disp(round(mcbth_slx_illm_col));

% Problem 3
mcbth_ae = angular_error(mcbth_illm_col,mcbth_slx_illm_col);
fprintf('3). The angular error between both images is %.2f˚.\n\n', mcbth_ae);

% Problem 4
diag_mcbth = create_diagonal_matrix(mcbth_illm_col,mcbth_slx_illm_col);
fprintf('4). The diagonal matrix is \n');
disp(diag_mcbth);

% Scale the RGB array
corr_mcbth = applyDiagonalMap(diag_mcbth, mcbth_solux);

% Estimate a single scale factor for each image's RGB so that the max RGB value is 250
mcbth_sx_sf = double(max_scale / max(mcbth_solux(:)));
corr_mcbth_sf = double(max_scale / max(corr_mcbth(:)));
mcbth_sf = double(max_scale / max(mcbth(:)));

figure('Position',[1, 1, 920, 300],'visible','off');
f1 = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
nexttile;
imshow(uint8(double(mcbth_solux)*mcbth_sx_sf));
title('Original Image (bluish)','FontSize',16);
nexttile;
imshow(uint8(corr_mcbth*corr_mcbth_sf));
title('Corrected Image','FontSize',16);
nexttile;
imshow(uint8(double(mcbth)*mcbth_sf));
title('Canonical Image','FontSize',16);
exportgraphics(f1, 'output/f1_macbeth_img_results.png', 'Resolution', 200);

% Problem 5
mcbth_orig_rms = rg_rms_error(mcbth_solux,mcbth);
mcbth_corr_rms = rg_rms_error(corr_mcbth,mcbth);

fprintf('5). The RG RMS error between the original and canonical image is %.2f.\n', mcbth_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical image is %.2f.\n\n', mcbth_corr_rms);

% Problem 6
% Load the canonical images
appl_can = prerocess_image(strcat(imgdir,'apples2_syl-50MR16Q.tif'), 'uint8');
ball_can = prerocess_image(strcat(imgdir,'ball_syl-50MR16Q.tif'), 'uint8');
blck_can = prerocess_image(strcat(imgdir,'blocks1_syl-50MR16Q.tif'), 'uint8');

% Load the solux-4100 images
appl_slx = prerocess_image(strcat(imgdir,'apples2_solux-4100.tif'), 'uint8');
ball_slx = prerocess_image(strcat(imgdir,'ball_solux-4100.tif'), 'uint8');
blck_slx = prerocess_image(strcat(imgdir,'blocks1_solux-4100.tif'), 'uint8');

% Get the max-rgb white light equivalents of the standard/canonical images
appl_cn_illm = estimate_rgb_illum(appl_can,'max');
ball_cn_illm = estimate_rgb_illum(ball_can,'max');
blck_cn_illm = estimate_rgb_illum(blck_can,'max');

% Get the max-rgb white light equivalents of the unknown/solux images
appl_mx_illm = estimate_rgb_illum(appl_slx,'max');
ball_mx_illm = estimate_rgb_illum(ball_slx,'max');
blck_mx_illm = estimate_rgb_illum(blck_slx,'max');

appl_ae = angular_error(mcbth_illm_col,appl_mx_illm);
ball_ae = angular_error(mcbth_illm_col,ball_mx_illm);
blck_ae = angular_error(mcbth_illm_col,blck_mx_illm);
fprintf('6). The angular error for the apple image is %.2f˚.\n', appl_ae);
fprintf('    The angular error for the ball image is %.2f˚.\n', ball_ae);
fprintf('    The angular error for the block image is %.2f˚.\n\n', blck_ae);

% Problem 7
% Get the multiplier factor
appl_scl = appl_cn_illm ./ appl_mx_illm;
ball_scl = ball_cn_illm ./ ball_mx_illm;
blck_scl = blck_cn_illm ./ blck_mx_illm;

% Estimate the corrected arrays
corr_appl = applyDiagonalMap(diag(appl_scl), appl_slx);
corr_ball = applyDiagonalMap(diag(ball_scl), ball_slx);
corr_blck = applyDiagonalMap(diag(blck_scl), blck_slx);

appl_sx_sf = double(max_scale / max(appl_slx(:)));
corr_appl_sf = double(max_scale / max(corr_appl(:)));
appl_sf = double(max_scale / max(appl_can(:)));

ball_sx_sf = double(max_scale / max(ball_slx(:)));
corr_ball_sf = double(max_scale / max(corr_ball(:)));
ball_sf = double(max_scale / max(corr_ball(:)));

blck_sx_sf = double(max_scale / max(blck_slx(:)));
corr_blck_sf = double(max_scale / max(corr_blck(:)));
blck_sf = double(max_scale / max(blck_can(:)));

figure('Position',[1, 1, 920, 670],'visible','off');
f2 = tiledlayout(3,3,'TileSpacing','Compact','Padding','Compact');
nexttile;
imshow(uint8(double(appl_slx)*appl_sx_sf));
title('Original Image (bluish)','FontSize',16);
nexttile;
imshow(uint8(corr_appl*corr_appl_sf));
title('Corrected Image','FontSize',16);
nexttile;
imshow(uint8(double(appl_can)*appl_sf));
title('Canonical Image','FontSize',16);

nexttile;
imshow(uint8(double(ball_slx)*ball_sx_sf));
nexttile;
imshow(uint8(corr_ball*corr_ball_sf));
nexttile;
imshow(uint8(double(ball_can)*ball_sf));

nexttile;
imshow(uint8(double(blck_slx)*blck_sx_sf));
nexttile;
imshow(uint8(corr_blck*corr_blck_sf));
nexttile;
imshow(uint8(double(blck_can)*blck_sf));
exportgraphics(f2, 'output/f2_maxRGB_img_results.png', 'Resolution', 200);

% Calculate the RMS between the canonical and solux images
appl_orig_rms = rg_rms_error(appl_slx,appl_can);
ball_orig_rms = rg_rms_error(ball_slx,ball_can);
blck_orig_rms = rg_rms_error(blck_slx,blck_can);

% Calculate the RMS between the canonical and corrected images
appl_corr_rms = rg_rms_error(corr_appl,appl_can);
ball_corr_rms = rg_rms_error(corr_ball,ball_can);
blck_corr_rms = rg_rms_error(corr_blck,blck_can);

fprintf('7). The RG RMS error between the original and canonical apples image is %.2f.\n', appl_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical apples image is %.2f.\n\n', appl_corr_rms);
fprintf('    The RG RMS error between the original and canonical balls image is %.2f.\n', ball_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical balls image is %.2f.\n\n', ball_corr_rms);
fprintf('    The RG RMS error between the original and canonical blocks image is %.2f.\n', blck_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical blocks image is %.2f.\n\n', blck_corr_rms);


% Problem 8 (angular error and RMS error using gray-world algorithm)
% Get the max-rgb white light equivalents of the standard/canonical images
appl_cn_illm = estimate_rgb_illum(appl_can,'avg');
ball_cn_illm = estimate_rgb_illum(ball_can,'avg');
blck_cn_illm = estimate_rgb_illum(blck_can,'avg');

% Get the max-rgb gray light equivalents of the unknown/solux images and
% multiply by 2 to get white light
appl_av_illm = estimate_rgb_illum(appl_slx,'avg');
ball_av_illm = estimate_rgb_illum(ball_slx,'avg');
blck_av_illm = estimate_rgb_illum(blck_slx,'avg');

appl_ae = angular_error(mcbth_illm_col,appl_av_illm);
ball_ae = angular_error(mcbth_illm_col,ball_av_illm);
blck_ae = angular_error(mcbth_illm_col,blck_av_illm);
fprintf('8). The angular error for the apple image is %.2f˚.\n', appl_ae);
fprintf('    The angular error for the ball image is %.2f˚.\n', ball_ae);
fprintf('    The angular error for the block image is %.2f˚.\n\n', blck_ae);

% Get the multiplier factor
appl_scl = appl_cn_illm ./ appl_av_illm;
ball_scl = ball_cn_illm ./ ball_av_illm;
blck_scl = blck_cn_illm ./ blck_av_illm;

% Estimate the corrected arrays
corr_appl = applyDiagonalMap(diag(appl_scl), appl_slx);
corr_ball = applyDiagonalMap(diag(ball_scl), ball_slx);
corr_blck = applyDiagonalMap(diag(blck_scl), blck_slx);

appl_sx_sf = double(max_scale / max(appl_slx(:)));
corr_appl_sf = double(max_scale / max(corr_appl(:)));
appl_sf = double(max_scale / max(appl_can(:)));

ball_sx_sf = double(max_scale / max(ball_slx(:)));
corr_ball_sf = double(max_scale / max(corr_ball(:)));
ball_sf = double(max_scale / max(corr_ball(:)));

blck_sx_sf = double(max_scale / max(blck_slx(:)));
corr_blck_sf = double(max_scale / max(corr_blck(:)));
blck_sf = double(max_scale / max(blck_can(:)));

figure('Position',[1, 1, 920, 670],'visible','off');
f2 = tiledlayout(3,3,'TileSpacing','Compact','Padding','Compact');
nexttile;
imshow(uint8(double(appl_slx)*appl_sx_sf));
title('Original Image (bluish)','FontSize',16);
nexttile;
imshow(uint8(corr_appl*corr_appl_sf));
title('Corrected Image','FontSize',16);
nexttile;
imshow(uint8(double(appl_can)*appl_sf));
title('Canonical Image','FontSize',16);

nexttile;
imshow(uint8(double(ball_slx)*ball_sx_sf));
nexttile;
imshow(uint8(corr_ball*corr_ball_sf));
nexttile;
imshow(uint8(double(ball_can)*ball_sf));

nexttile;
imshow(uint8(double(blck_slx)*blck_sx_sf));
nexttile;
imshow(uint8(corr_blck*corr_blck_sf));
nexttile;
imshow(uint8(double(blck_can)*blck_sf));
exportgraphics(f2, 'output/f3_gray_world_img_results.png', 'Resolution', 200);

% Calculate the RMS between the canonical and solux images
appl_orig_rms = rg_rms_error(appl_slx,appl_can);
ball_orig_rms = rg_rms_error(ball_slx,ball_can);
blck_orig_rms = rg_rms_error(blck_slx,blck_can);

% Calculate the RMS between the canonical and corrected images
appl_corr_rms = rg_rms_error(corr_appl,appl_can);
ball_corr_rms = rg_rms_error(corr_ball,ball_can);
blck_corr_rms = rg_rms_error(corr_blck,blck_can);

fprintf('    The RG RMS error between the original and canonical apples image is %.2f.\n', appl_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical apples image is %.2f.\n\n', appl_corr_rms);
fprintf('    The RG RMS error between the original and canonical balls image is %.2f.\n', ball_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical balls image is %.2f.\n\n', ball_corr_rms);
fprintf('    The RG RMS error between the original and canonical blocks image is %.2f.\n', blck_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical blocks image is %.2f.\n\n', blck_corr_rms);


num_questions = num_questions + 1;

%% Part B
fprintf("\nPart B\n")

% Problem 9 (custom sse error (incorrect) and lsqr (correct))
% Custom formula
[cust_dm_mcbth, cust_mc_rmse, corr_mcbth] = customDiagonal(mcbth,mcbth_solux);
[cust_dm_appl, cust_ap_rmse, corr_appl] = customDiagonal(appl_can,appl_slx);
[cust_dm_ball, cust_ba_rmse, corr_ball] = customDiagonal(ball_can,ball_slx);
[cust_dm_blck, cust_bl_rmse, corr_blck] = customDiagonal(blck_can,blck_slx);
% LSQR solution
[lsqr_dm_mcbth, lsqr_mc_rmse, corr_mcbth] = lsqrDiagonal(mcbth,mcbth_solux);
[lsqr_dm_appl, lsqr_ap_rmse, corr_appl] = lsqrDiagonal(appl_can,appl_slx);
[lsqr_dm_ball, lsqr_ba_rmse, corr_ball] = lsqrDiagonal(ball_can,ball_slx);
[lsqr_dm_blck, lsqr_bl_rmse, corr_blck] = lsqrDiagonal(blck_can,blck_slx);

cust_rmse = [cust_mc_rmse;cust_ap_rmse;cust_ba_rmse;cust_bl_rmse];
lsqr_rmse = [lsqr_mc_rmse;lsqr_ap_rmse;lsqr_ba_rmse;lsqr_bl_rmse];

T = table(cust_rmse, lsqr_rmse, ...
    RowNames={'Macbeth','Apples','Ball','Block'}, VariableNames={'Custom SSE','LSQR'});
fprintf("RMSE across images\n")
disp(round(T,3));

% Initial guess for the diagonal elements [d_R, d_G, d_B]
% d_init = [1,1,1];

% Oracle color constancy startpoint from QA5
orcl_mc_rmse = mcbth_corr_rms;
orcl_ap_rmse = rg_rms_error(applyDiagonalMap(diag_mcbth, appl_slx),appl_can);
orcl_ba_rmse = rg_rms_error(applyDiagonalMap(diag_mcbth, ball_slx),ball_can);
orcl_bl_rmse = rg_rms_error(applyDiagonalMap(diag_mcbth, blck_slx),blck_can);

[~, fm_mc_rmse, ~] = optimizeDiagonalMatrix(diag(diag_mcbth)',mcbth,mcbth_solux);
[~, fm_ap_rmse, ~] = optimizeDiagonalMatrix(diag(diag_mcbth)',appl_can,appl_slx);
[~, fm_ba_rmse, ~] = optimizeDiagonalMatrix(diag(diag_mcbth)',ball_can,ball_slx);
[~, fm_bl_rmse, ~] = optimizeDiagonalMatrix(diag(diag_mcbth)',blck_can,blck_slx);

orcl_origin_rms = [orcl_mc_rmse;orcl_ap_rmse;orcl_ba_rmse;orcl_bl_rmse];
fm_orcl_origin_rms = [fm_mc_rmse;fm_ap_rmse;fm_ba_rmse;fm_bl_rmse];

T = table(orcl_origin_rms, fm_orcl_origin_rms, ...
    RowNames={'Macbeth','Apples','Ball','Block'}, VariableNames={'oracle','fminsearch'});
fprintf("RMSE from oracle color constancy startpoint across images\n")
disp(round(T,3));

% LSQR startpoints from QB9
[~, fm_mc_rmse, ~] = optimizeDiagonalMatrix(lsqr_dm_mcbth,mcbth,mcbth_solux);
[~, fm_ap_rmse, ~] = optimizeDiagonalMatrix(lsqr_dm_appl,appl_can,appl_slx);
[~, fm_ba_rmse, ~] = optimizeDiagonalMatrix(lsqr_dm_ball,ball_can,ball_slx);
[~, fm_bl_rmse, ~] = optimizeDiagonalMatrix(lsqr_dm_blck,blck_can,blck_slx);

fm_lsqr_origin_rms = [fm_mc_rmse;fm_ap_rmse;fm_ba_rmse;fm_bl_rmse];

T = table(lsqr_rmse, fm_lsqr_origin_rms, ...
    RowNames={'Macbeth','Apples','Ball','Block'}, VariableNames={'LSQR','fminsearch'});
fprintf("RMSE from optimized LQSR startpoint across images\n")
disp(round(T,3));

num_questions = num_questions + 2;



end

function processed_img = prerocess_image(input_fpath,output)
    img = imread(input_fpath);
    img_double = double(img);

    if strcmp(output,'double')
        processed_img = img_double + 0.0001;
    elseif strcmp(output,'uint8')
        img_rounded = round(img_double);
        processed_img = uint8(img_rounded) + 0.0001;
    else
        processed_img = img;
    end
end

function canonical_light = get_canonical_light_estimate(img,pxx1,pxx2,pxy1,pxy2,scale)
    sub_img = img(pxx1:pxx2,pxy1:pxy2,:);   % Extract a subset of the image e.g. white pixels
    avg_wpx = squeeze(mean(sub_img,[1,2])); % get the average across rgb channels
    illm_val = scale / max(avg_wpx);        % canonical illuminant
    canonical_light = illm_val * avg_wpx;   % illuminant color
end

function angle = angular_error(illum1, illum2)
    % Ensure the input matrices are the same size
    if size(illum1) ~= size(illum2)
        error('The input matrices must be the same size.');
    end

    % Compute the dot product of the two vectors
    dot_product = dot(illum1, illum2);
    
    % Compute the magnitudes of the vectors
    mag1 = norm(illum1);
    mag2 = norm(illum2);

    % Calculate the cosine of the angle
    cos_theta = dot_product / (mag1 * mag2);
    angle = acosd(cos_theta);
end

function diag_mat = create_diagonal_matrix(can_light,bluish_light)
    div = can_light./bluish_light;
    diag_mat = diag(div);
end

function rms_error = rg_rms_error(input_image, target_image)
    % Convert input images to double format
    input_image = double(input_image);
    target_image = double(target_image);

    % Reshape the images into (n, 3) format, where n is the number of pixels
    input_flat = reshape(input_image, [], 3);   % Flatten to (n, 3)
    target_flat = reshape(target_image, [], 3); % Flatten to (n, 3)

    % Calculate R, G, B sums over channel dimensions
    sum_rgb_input = sum(input_flat, 2);
    sum_rgb_target = sum(target_flat, 2);

    % Create a mask to exclude dark pixels (where R + G + B < 10)
    valid_pixels = (sum_rgb_input >= 10) & (sum_rgb_target >= 10);

    % Apply the valid pixel mask
    input_flat = input_flat(valid_pixels, :);
    target_flat = target_flat(valid_pixels, :);
    sum_rgb_input = sum_rgb_input(valid_pixels);
    sum_rgb_target = sum_rgb_target(valid_pixels);

    % Calculate r and g for the input image
    r_input = input_flat(:, 1) ./ sum_rgb_input;
    g_input = input_flat(:, 2) ./ sum_rgb_input;

    % Calculate r and g for the target image
    r_target = target_flat(:, 1) ./ sum_rgb_target;
    g_target = target_flat(:, 2) ./ sum_rgb_target;

    % Compute the pixel-by-pixel differences in r and g
    diff_r = r_input - r_target;
    diff_g = g_input - g_target;

    % Calculate the RMS error
    rms_error = sqrt(mean(diff_r.^2) + mean(diff_g.^2));
end

function out_rgb = estimate_rgb_illum(img, output)
    % Estimate either maxRGB or grayWorld illumination colors for an image 
    img = double(img);
    img_flat = reshape(img, [], 3);
    
    if strcmp(output,'max')
        % Max across all channels and transpose to be consistent with macbeth image format
        out_rgb = max(img_flat)';
    elseif strcmp(output,'avg')
        out_rgb = mean(img_flat)';
    else
        error('Valid output arguments are max and avg.');
    end
end

function [d_opt, final_rmse, corrected_img] = customDiagonal(img1, img2)
    % Inputs:
    % img1: RGB image under first (canocical) light (dimensions: x, y, 3)
    % img2: RGB image under second (unknown) light (target image with same dimensions)
    
    % Outputs:
    % d_opt: Optimized diagonal matrix (3x3 diagonal matrix)
    % final_rmse: Final RMSE from diagonal matrix (doublle)
    % corrected_img: Corrected image after applying the diagonal map

    % Flatten the images into n x 3 matrices (where n is the number of pixels)
    R1 = reshape(img1(:,:,1), [], 1);  % Red channel of img1
    G1 = reshape(img1(:,:,2), [], 1);  % Green channel of img1
    B1 = reshape(img1(:,:,3), [], 1);  % Blue channel of img1

    R2 = reshape(img2(:,:,1), [], 1);  % Red channel of img2
    G2 = reshape(img2(:,:,2), [], 1);  % Green channel of img2
    B2 = reshape(img2(:,:,3), [], 1);  % Blue channel of img2

    % Compute the sums of products for each channel
    d_R = sum(R1 .* R2) / sum(R1);  % Optimal scaling factor for Red channel
    d_G = sum(G1 .* G2) / sum(G1);  % Optimal scaling factor for Green channel
    d_B = sum(B1 .* B2) / sum(B1);  % Optimal scaling factor for Blue channel

    % Construct the final diagonal matrix using the optimized values
    d_opt = [d_R, d_G, d_B];
    D_opt = diag(d_opt);

    % Apply the optimized diagonal map to img2
    corrected_img = applyDiagonalMap(D_opt, img2);

    % Display final RMSE
    final_rmse = computeRMSE(d_opt, img1, img2);
end

function [d_opt, final_rmse, corrected_img] = lsqrDiagonal(img1, img2)
    % Inputs:
    % img1: RGB image under first (canocical) light (dimensions: x, y, 3)
    % img2: RGB image under second (unknown) light (target image with same dimensions)
    
    % Outputs:
    % d_opt: Optimized diagonal matrix (3x3 diagonal matrix)
    % final_rmse: Final RMSE from diagonal matrix (doublle)
    % corrected_img: Corrected image after applying the diagonal map

    % Flatten the images into n x 3 matrices (where n is the number of pixels)
    X = double(reshape(img1, [], 3));  % img1 as an n x 3 matrix
    Y = double(reshape(img2, [], 3));  % img2 as an n x 3 matrix

    % Initialize the diagonal scaling factors (d_R, d_G, d_B)
    d_opt = zeros(1, 3);

    % Solve the least squares problem for each channel (R, G, B)
    for k = 1:3
        Xk = X(:,k);  % Channel k of img1
        Yk = Y(:,k);  % Channel k of img2
        
        % Compute the least squares solution for the k-th channel
        d_opt(k) = (Yk \ Xk);
    end

    % Construct the final diagonal matrix using the optimized values
    D_opt = diag(d_opt);

    % Apply the optimized diagonal map to img2
    corrected_img = applyDiagonalMap(D_opt, img1);

    % Display final RMSE
    final_rmse = computeRMSE(d_opt, img1, img2);
end


function [d_opt, final_rmse, corrected_img] = optimizeDiagonalMatrix(d_init, img1, img2)
    % Inputs:
    % img1: RGB image under first (canocical) light (dimensions: x, y, 3)
    % final_rmse: Final RMSE from diagonal matrix (doublle)
    % img2: RGB image under second (unknown) light (target image with same dimensions)
    
    % Outputs:
    % d_opt: Optimized diagonal matrix (3x3 diagonal matrix)
    % corrected_img: Corrected image after applying the diagonal map
    
    % Objective function that computes the RMSE between the corrected image and img2
    obj_fun = @(d) computeRMSE(d, img1, img2);
    
    % Optimization options: specify tolerance, display option, etc.
    options = optimset('Display', 'off', 'TolFun', 1e-10, 'TolX', 1e-4, 'MaxIter', 1000); % use `iter` to display each step
    
    % Use fminsearch to minimize RMSE by adjusting the diagonal values
    d_opt = fminsearch(obj_fun, d_init, options);
    % options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton', 'TolFun', 1e-6, 'MaxIter', 1000);
    % d_opt = fminunc(obj_fun, d_init, options);
    
    % Construct the final diagonal matrix using the optimized values
    D_opt = diag(d_opt);
    
    % Apply the optimized diagonal map to img2
    corrected_img = applyDiagonalMap(D_opt, img2);
    
    % Display final RMSE
    final_rmse = computeRMSE(d_opt, img1, img2);
end

% Helper function to apply diagonal map to the image
function corrected_img = applyDiagonalMap(D, img)
    % Apply the diagonal map (D is 3x3 diagonal matrix) to the RGB channels of img
    corrected_img = img; % Preallocate corrected image
    for c = 1:3
        corrected_img(:,:,c) = D(c,c) * img(:,:,c);  % Apply the diagonal scaling
    end
end

% Helper function to compute RMSE between the corrected and target image
function rmse = computeRMSE(d, img1, img2)
    % Construct the diagonal matrix from the vector d = [d_R, d_G, d_B]
    D = diag(d);

    % Apply diagonal map to img1 to get the corrected image
    corrected_img = applyDiagonalMap(D, img2);

    % Compute RMSE (Root Mean Square Error)
    rmse = rg_rms_error(corrected_img, img1);
end
