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

% Plot to get white pixel bounds
figure;
imshow(mcbth);
datacursormode on;

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

corr_mcbth = zeros(size(mcbth_solux)); % Preallocate for the scaled array
% Scale the RGB array
for i = 1:3
    arr_double = double(mcbth_solux);
    corr_mcbth(:, :, i) = arr_double(:, :, i) * diag_mcbth(i, i);
end

% Estimate a single scale factor for all images so that the max RGB value is 250
scl_fct = double(max_scale / max([max(mcbth_solux(:)),max(corr_mcbth(:)),max(mcbth(:))]));

figure('Position',[1, 1, 920, 300],'visible','off');
f1 = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
nexttile;
imshow(uint8(double(mcbth_solux)*scl_fct));
title('Original Image (bluish)','FontSize',16);
nexttile;
imshow(uint8(corr_mcbth*scl_fct));
title('Corrected Image','FontSize',16);
nexttile;
imshow(uint8(double(mcbth)*scl_fct));
title('Canonical Image','FontSize',16);
exportgraphics(f1, 'output/f1_macbeth_img_results.png', 'Resolution', 200);

% Problem 5
mcbth_orig_rms = rg_rms_error(mcbth_solux,mcbth);
mcbth_corr_rms = rg_rms_error(corr_mcbth,mcbth);

fprintf('5). The RG RMS error between the original and canonical image is %.2f.\n', mcbth_orig_rms);
fprintf('    The RG RMS error between the corrected and canonical image is %.2f.\n\n', mcbth_corr_rms);

% Problem 6
appl_org = prerocess_image(strcat(imgdir,'apples2_syl-50MR16Q.tif'), 'uint8');
appl_slx = prerocess_image(strcat(imgdir,'apples2_solux-4100.tif'), 'uint8');
ball_slx = prerocess_image(strcat(imgdir,'ball_solux-4100.tif'), 'uint8');
blck_slx = prerocess_image(strcat(imgdir,'blocks1_solux-4100.tif'), 'uint8');

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
appl_scl = mcbth_illm_col ./ appl_mx_illm;
ball_scl = mcbth_illm_col ./ ball_mx_illm;
blck_scl = mcbth_illm_col ./ blck_mx_illm;

% Preallocate for the corrected arrays
corr_appl = zeros(size(appl_slx));
corr_ball = zeros(size(ball_slx));
corr_blck = zeros(size(blck_slx));
% Scale the RGB array
for i = 1:3
    appl_double = double(appl_slx);
    ball_double = double(ball_slx);
    blck_double = double(blck_slx);
    corr_appl(:, :, i) = appl_double(:, :, i) * appl_scl(i);
    corr_ball(:, :, i) = ball_double(:, :, i) * ball_scl(i);
    corr_blck(:, :, i) = blck_double(:, :, i) * blck_scl(i);
end

appl_scl_fct = double(max_scale / max([max(appl_slx(:)),max(corr_appl(:)),max(appl_org(:))]));
disp(appl_scl_fct);

figure;
subplot(1,3,1);
imshow(uint8(appl_slx));
subplot(1,3,2);
imshow(uint8(corr_appl));
subplot(1,3,3);
imshow(uint8(appl_org));



num_questions = num_questions + 1;

%% Part B
fprintf("\nPart B\n")


num_questions = num_questions + 1;



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
    diag_mat = [div(1),0,0;
                0,div(2),0;
                0,0,div(3)];
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
