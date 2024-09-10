function [num_questions] = hw2(sensors, light, responses)
% This function meets all the deliverables for hw2

close all;
clc;
num_questions = 0;

% Create an output dir to save the images
if ~exist('output', 'dir')
    mkdir('output');
end

%%  Q1
fprintf('Question 1:\n');
rgb_snsr = importdata(sensors);
[r,g,b] = deal(rgb_snsr(:,1), rgb_snsr(:,2), rgb_snsr(:,3));

f1 = figure('visible','off');
plot(r, Color='r');
hold on;
plot(g, Color='g');
plot(b, Color='b');
legend("r","g","b"); xlabel('Wavelengths'); ylabel('Sensor Sensitivity');
hold off;
exportgraphics(f1, 'output/f1_rgb.png', 'Resolution', 200);

% Set the random number generator seed and scale the values to give 10^-4
rng(477);
mult_scale = 9.4e-4;
spectra = rand(1600, 101) * mult_scale;

rnd_rgb_resp = spectra * rgb_snsr;
rnd_rgb40 = reshape(rnd_rgb_resp, [40, 40, 3]);

% min max across channels
rmin = min(rnd_rgb_resp(:, 1),[],"all");
gmin = min(rnd_rgb_resp(:, 2),[],"all");
bmin = min(rnd_rgb_resp(:, 3),[],"all");
omin = min(rnd_rgb_resp(:));

rmax = max(rnd_rgb_resp(:, 1),[],"all");
gmax = max(rnd_rgb_resp(:, 2),[],"all");
bmax = max(rnd_rgb_resp(:, 3),[],"all");
omax = max(rnd_rgb_resp(:));

red_stats = [rmin; rmax];
grn_stats = [gmin; gmax];
blu_stats = [bmin; bmax];
ovr_stats = [omin; omax];

T = table(red_stats, grn_stats, blu_stats, ovr_stats, ...
    RowNames={'Min','Max'}, VariableNames={'Red','Green','Blue','Overall'});
fprintf("Min-Max across channels\n")
disp(T);


rnd_rgb400 = zeros(400, 400, 3);  % 400x400 because we want 10x10 blocks

% Loop through the 40x40 grid and assign 10x10 color blocks to the output image
for i = 1:40
    for j = 1:40
        r = rnd_rgb40(i, j, 1);
        g = rnd_rgb40(i, j, 2);
        b = rnd_rgb40(i, j, 3);
        
        % Create a 10x10 block with the same color for the current square
        color_block = cat(3, r*ones(10,10), g*ones(10,10), b*ones(10,10));
        rnd_rgb400((i-1)*10+1:i*10, (j-1)*10+1:j*10, :) = color_block;
    end
end

f2 = figure('visible','off');
imshow(uint8(rnd_rgb400));
exportgraphics(f2, 'output/f2_rgb400by400.png', 'Resolution', 200);

num_questions = num_questions + 1;


%% Q2
fprintf('\n\nQuestion 2:\n'); 
inv_spct = linsolve(spectra, rnd_rgb_resp); % Size (101 x 3)
x = 1:size(rgb_snsr, 1); 

figure('Position', [1, 1, 820, 400],'visible','off');
f3 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; hold on; % True Spectra
plot(x, rgb_snsr(:,1), 'r', 'LineWidth', 2);
plot(x, rgb_snsr(:,2), 'g', 'LineWidth', 2);
plot(x, rgb_snsr(:,3), 'b', 'LineWidth', 2);
hold off; title('True Spectra'); legend("r","g","b");
xlabel('Wavelengths'); ylabel('Sensor Sensitivity');

nexttile; hold on; % Estimated Spectra
plot(x, inv_spct(:,1), 'r--', 'LineWidth', 2);
plot(x, inv_spct(:,2), 'g--', 'LineWidth', 2);
plot(x, inv_spct(:,3), 'b--', 'LineWidth', 2);
hold off; title('Estimated Spectra'); 
legend("r","g","b"); xlabel('Wavelengths'); ylim([0 2.5e4]);

title(f3, 'Inverted sensitivity for 0 noise');

exportgraphics(f3, 'output/f3_inverted_spectra.png', 'Resolution', 200);

% Compute the rms for each spectra band
rms_red = sqrt(mean((rgb_snsr(:,1) - inv_spct(:,1)).^2, 'all'));
rms_grn = sqrt(mean((rgb_snsr(:,2) - inv_spct(:,2)).^2, 'all'));
rms_blu = sqrt(mean((rgb_snsr(:,3) - inv_spct(:,3)).^2, 'all'));

% Compute the RGB responses using the estimated sensors
rnd_rgb_inv = spectra * inv_spct;
% Compute the RMS error between the true and estimated RGB values
rms_rgb = sqrt(mean((rnd_rgb_resp - rnd_rgb_inv).^2, 'all'));


T = table(rms_red, rms_grn, rms_blu, rms_rgb, ...
    RowNames={'RMS Errors'}, VariableNames={'Red','Green','Blue','RGB'});
fprintf('RMS Error between true and estimated spectra\n');
disp(T);

num_questions = num_questions + 1;


%% Q3
fprintf('\n\nQuestion 3:\n');

    function [out_spct,snsr_rms,rgb_rms] = simulate_sensitivity(inp_spct,noise_std,clip)
        % Args
        % inp_sens -> input spectra for camera sensitivity 
        % noise_std -> noise std dev
        % clip -> boolean for whether or not to clip between 0,255
        
        rng(477);
        rnd_spectra = rand(1600, 101) * 9.4e-4;

        rgb_data = rnd_spectra * inp_spct;
        
        if ~isempty(noise_std)
            noise = noise_std * randn(size(rgb_data));
            rgb_data = rgb_data + noise;
        end
        
        if clip
            rgb_data = max(0, min(255, rgb_data));
        end
        
        % Invert the image sensitivity spectrum
        out_spct = linsolve(rnd_spectra, rgb_data);

        % Generate an rgb spectrum from the inverted light responses
        rgb_inv_resp = rnd_spectra * out_spct;
        
        % Calculate the rms error for sensors and rgb 
        snsr_rms = sqrt(mean((inp_spct - out_spct).^2, 'all'));
        rgb_rms = sqrt(mean((rgb_data - rgb_inv_resp).^2, 'all'));
    end

    function [r_err,g_err,b_err] = channel_sensitivity_rms_error(true_sens,esti_sens)
        r_err = sqrt(mean((true_sens(:,1) - esti_sens(:,1)).^2, 'all'));
        g_err = sqrt(mean((true_sens(:,2) - esti_sens(:,2)).^2, 'all'));
        b_err = sqrt(mean((true_sens(:,3) - esti_sens(:,3)).^2, 'all'));
    end

% Invert for the camera sensitivities and estimate the rms errors
[spec_noise10,srms10,rrms10] = simulate_sensitivity(rgb_snsr,10,false);
[rms_red, rms_grn, rms_blu] = channel_sensitivity_rms_error(rgb_snsr,spec_noise10);

f4 = figure('Position', [1, 1, 820, 400], 'visible','off');
tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; hold on; % True Spectra
plot(x, rgb_snsr(:,1), 'r', 'LineWidth', 2);
plot(x, rgb_snsr(:,2), 'g', 'LineWidth', 2);
plot(x, rgb_snsr(:,3), 'b', 'LineWidth', 2);
plot(x, spec_noise10(:,1), 'r--', 'LineWidth', 2);
plot(x, spec_noise10(:,2), 'g--', 'LineWidth', 2);
plot(x, spec_noise10(:,3), 'b--', 'LineWidth', 2);
hold off; 
legend("r","g","b","est. r","est. g","est. b");
xlabel('Wavelengths'); ylabel('Sensor Sensitivity');
title('Inverted sensitivity for 10 noise');

exportgraphics(f4, 'output/f4_inverted_spectra_noise10.png', 'Resolution', 200);

T = table(rms_red, rms_grn, rms_blu, srms10, rrms10, ...
    RowNames={'RMS Errors'}, VariableNames={'Red','Green','Blue','Sensor','RGB'});
fprintf('Unclipped RMS Error between true and estimated spectra\n');
disp(T);

[spec_noise10,srms10,rrms10] = simulate_sensitivity(rgb_snsr,10,true);
[rms_red, rms_grn, rms_blu] = channel_sensitivity_rms_error(rgb_snsr,spec_noise10);

T = table(rms_red, rms_grn, rms_blu, srms10, rrms10, ...
    RowNames={'RMS Errors'}, VariableNames={'Red','Green','Blue','Sensor','RGB'});
fprintf('Clipped RMS Error between true and estimated spectra\n');
disp(T);

num_questions = num_questions + 1;

%% Q4

fprintf('\n\nQuestion 4:\n');
noise_stds = [0:10:100, 120:20:200, 250:50:400];

results = table('Size', [length(noise_stds) 5], ...
                'VariableTypes', {'double', 'double', 'double', 'double', 'double'}, ...
                'VariableNames', {'Noise Std', 'Unclipped RGB RMS', 'Clipped RGB RMS', 'Unclipped Sensor RMS', 'Clipped Sensor RMS'});


for i = 1:length(noise_stds)
    n_std = noise_stds(i);
    [clp_est_snst,unc_snst_rms,unc_rgb_rms] = simulate_sensitivity(rgb_snsr,n_std,false);
    [uclp_est_snst,clp_snst_rms,clp_rgb_rms] = simulate_sensitivity(rgb_snsr,n_std,true);

    if i == 6 || i == 11
        figure('Position', [1, 1, 820, 400], 'visible','off');
        f56 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
        nexttile; hold on; % True Spectra
        plot(x, rgb_snsr(:, 1), 'r', 'DisplayName', 'True Red');
        plot(x, rgb_snsr(:, 2), 'g', 'DisplayName', 'True Green');
        plot(x, rgb_snsr(:, 3), 'b', 'DisplayName', 'True Blue');
        plot(x, clp_est_snst(:, 1), 'r--', 'DisplayName', 'Est. Red');
        plot(x, clp_est_snst(:, 2), 'g--', 'DisplayName', 'Est. Green');
        plot(x, clp_est_snst(:, 3), 'b--', 'DisplayName', 'Est. Blue');
        hold off; title('Clipped Sensitivity Spectra'); 
        xlabel('Wavelengths'); ylabel('Sensor Sensitivity');
        
        nexttile; hold on; % Estimated Spectra
        plot(x, rgb_snsr(:, 1), 'r', 'DisplayName', 'True Red');
        plot(x, rgb_snsr(:, 2), 'g', 'DisplayName', 'True Green');
        plot(x, rgb_snsr(:, 3), 'b', 'DisplayName', 'True Blue');
        plot(x, uclp_est_snst(:, 1), 'r--', 'DisplayName', 'Est. Red');
        plot(x, uclp_est_snst(:, 2), 'g--', 'DisplayName', 'Est. Green');
        plot(x, uclp_est_snst(:, 3), 'b--', 'DisplayName', 'Est. Blue');
        hold off; xlabel('Wavelengths'); 
        title('Unclipped Sensitivity Spectra');
        
        title(f56, sprintf('Inverted sensitivity for Noise Std. %d',n_std));

        if i == 6
            exportgraphics(f56, sprintf('output/f5_inverted_spectra_noise%d.png',n_std), 'Resolution', 200);
        end

        if i == 11
            exportgraphics(f56, sprintf('output/f6_inverted_spectra_noise%d.png',n_std), 'Resolution', 200);
        end
       
    end
    
    results{i, 'Noise Std'} = n_std;
    results{i, 'Unclipped RGB RMS'} = unc_rgb_rms;
    results{i, 'Clipped RGB RMS'} = clp_rgb_rms;
    results{i, 'Unclipped Sensor RMS'} = unc_snst_rms;
    results{i, 'Clipped Sensor RMS'} = clp_snst_rms;
end

disp(results);

figure('Position', [1, 1, 820, 400], 'visible','off');
f7 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');

nexttile; hold on;
plot(results.('Noise Std'),results.('Unclipped RGB RMS'),'black-');
plot(results.('Noise Std'),results.('Clipped RGB RMS'),'b--');
legend("unclipped","clipped"); xlabel('Noise Std.'); % yscale log;
ylabel('Error magnitude'); title('RGB RMS errors');

nexttile; hold on;
plot(results.('Noise Std'),results.('Unclipped Sensor RMS'),'black-');
plot(results.('Noise Std'),results.('Clipped Sensor RMS'),'b--');
legend("unclipped","clipped"); xlabel('Noise Std.'); 
title('Sensor RMS errors'); % yscale log;
hold off;

exportgraphics(f7, 'output/f7_rms_err_vs_noise.png', 'Resolution', 200);

num_questions = num_questions + 1;

%% Q5

fprintf('\n\nQuestion 5:\n');
% Known values
raw_intensity = 80;
perceived_intensity = 0.5;

% Normalize the raw intensity
I_raw = raw_intensity / 255;

% Compute the gamma value
gamma_value = log(perceived_intensity) / log(I_raw);

% Display the computed gamma value
fprintf('The computed gamma value is: %.2f\n', gamma_value);

num_questions = num_questions + 1;

%%  Q6
fprintf('\n\nQuestion 6:\n');

light_spec = importdata(light);
orig_rgb_resp = importdata(responses);

est_rgb_resp = light_spec * rgb_snsr;
est_spct = linsolve(light_spec, orig_rgb_resp);

light_snsr_rms = sqrt(mean((rgb_snsr - est_spct).^2, 'all'));
light_rgb_rms = sqrt(mean((orig_rgb_resp - est_rgb_resp).^2, 'all'));

figure('Position', [1, 1, 820, 400],'visible','off');
f8 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; hold on; % True Spectra
plot(x, rgb_snsr(:,1), 'r', 'LineWidth', 2);
plot(x, rgb_snsr(:,2), 'g', 'LineWidth', 2);
plot(x, rgb_snsr(:,3), 'b', 'LineWidth', 2);
hold off; title('True Spectra'); legend("r","g","b");
xlabel('Wavelengths'); ylabel('Sensor Sensitivity');

nexttile; hold on; % Estimated Spectra
plot(x, est_spct(:,1), 'r--', 'LineWidth', 2);
plot(x, est_spct(:,2), 'g--', 'LineWidth', 2);
plot(x, est_spct(:,3), 'b--', 'LineWidth', 2);
hold off; title(sprintf('Estimated Spectra for light spectra data')); 
legend("r","g","b"); xlabel('Wavelengths'); % ylim([0 2.5e4]);

exportgraphics(f8, 'output/f8_inverted_light_spectra.png', 'Resolution', 200);

fprintf('The overall sensor RMS value is: %f\n', light_snsr_rms);
fprintf('The overall RGB RMS value is: %f\n', light_rgb_rms);

num_questions = num_questions + 1;

%%  Q7
fprintf('\n\nQuestion 7:\n');

[n,n_wvln] = size(light_spec);
[n,n_ch] = size(orig_rgb_resp);

% Initialize estimated sensor matrix
est_spct_quadprog = zeros(n_wvln, n_ch);

 % Set up the quadratic programming problem for each sensor (Red, Green, 
 % Blue) to apply constrained least squares
for i = 1:n_ch
    % Minimize 0.5*x'Hx + f'x, where 
    % H = light_spec' * light_spec, and 
    % f = -light_spec' * orig_rgb_resp(:,i)
    
    H = light_spec' * light_spec;
    f = -light_spec' * orig_rgb_resp(:,i);
    
    % No inequality constraints (Ax ≤ b), but we need non-negativity constraints (x ≥ 0)
    A = [];  % No inequality constraints
    b = [];  % No inequality constraints
    lb = zeros(n_wvln, 1);  % Non-negativity constraint: solution ≥ 0
    
    % Use quadprog to solve the constrained least squares problem
    est_spct_quadprog(:, i) = quadprog(H, f, A, b, [], [], lb, []); 
end

% Compute the estimated RGB responses using the new sensor estimates
est_rgb_resp_quadprog = light_spec * est_spct_quadprog;

% Calculate RMS error between actual and estimated sensors
light_snsr_rms_quadprog = sqrt(mean((rgb_snsr - est_spct_quadprog).^2, 'all'));

% Calculate RMS error between original RGB responses and estimated RGB responses
light_rgb_rms_quadprog = sqrt(mean((orig_rgb_resp - est_rgb_resp_quadprog).^2, 'all'));

% Plot the real sensors and the estimated sensors from constrained least squares
f9 = figure('visible','off');
hold on;
plot(1:n_wvln, rgb_snsr(:, 1), 'r', 'DisplayName', 'Real Red');
plot(1:n_wvln, rgb_snsr(:, 2), 'g', 'DisplayName', 'Real Green');
plot(1:n_wvln, rgb_snsr(:, 3), 'b', 'DisplayName', 'Real Blue');
plot(1:n_wvln, est_spct_quadprog(:, 1), 'r--', 'DisplayName', 'Est. Red');
plot(1:n_wvln, est_spct_quadprog(:, 2), 'g--', 'DisplayName', 'Est. Green');
plot(1:n_wvln, est_spct_quadprog(:, 3), 'b--', 'DisplayName', 'Est. Blue');
legend;
xlabel('Wavelengths'); ylabel('Sensor Sensitivity');
title('Sensors comparison using Constrained Least Squares');
hold off;
exportgraphics(f9, 'output/f9_constrained_lst_sqr.png', 'Resolution', 200);

% Display the RMS errors
fprintf('RMS error between real and estimated sensors (Constrained): %.4f\n', light_snsr_rms_quadprog);
fprintf('RMS error between real and estimated RGB responses (Constrained): %.4f\n', light_rgb_rms_quadprog);

%% Q8
fprintf('\n\nQuestion 8:\n');

% Create the differencing matrix M (size 100x101)
M = diag(ones(n_wvln, 1), 0) - diag(ones(n_wvln-1, 1), 1);
M = M(1:end-1, :);  % 100x101 matrix
disp(size(M));

% Lambda values to test
lambdas = [0, 0.01, 0.0225, 0.1, 1];


results = table('Size', [length(lambdas) 3], ...
                'VariableTypes', {'double', 'double', 'double'}, ...
                'VariableNames', {'Lambda', 'RGB RMS', 'Sensor RMS'});

% Initialize the figure for plotting
figure('Position', [100, 300, 620, 900],'visible','off');
f10 = tiledlayout(3, 2,'TileSpacing','Compact','Padding','Compact');

% Loop over lambda values and perform constrained least squares with smoothness
for idx = 1:length(lambdas)
    lambda = lambdas(idx);
    
    % Augment the light spectra matrix and the response vector
    light_spec_aug = [light_spec; lambda * M];
    zero_vec = zeros(size(M, 1), 1);  % 100 zeros for smoothness
    est_spct_quadprog = zeros(n_wvln, 3);  % 101x3 matrix for RGB sensors
    
    for i = 1:3  % Solve for each sensor (Red, Green, Blue)
        % Augment the original RGB response with zeros for smoothness
        rgb_resp_aug = [orig_rgb_resp(:, i); zero_vec];
        
        % Set up the quadratic programming problem
        H = light_spec_aug' * light_spec_aug;
        f = -light_spec_aug' * rgb_resp_aug;
        
        % No inequality constraints, but non-negativity constraint
        lb = zeros(n_wvln, 1);  % Non-negative constraint
        
        % Solve using quadprog
        est_spct_quadprog(:, i) = quadprog(H, f, [], [], [], [], lb, []);
    end

    % Compute the estimated RGB responses
    smth_rgb_resp = light_spec * est_spct_quadprog;
    
    % Calculate RMS error for sensor and rgb response
    smth_snsr_rms = sqrt(mean((rgb_snsr - est_spct_quadprog).^2, 'all'));
    smth_rgb_rms = sqrt(mean((orig_rgb_resp - smth_rgb_resp).^2, 'all'));

    results{idx, 'Lambda'} = lambda;
    results{idx, 'RGB RMS'} = smth_rgb_rms;
    results{idx, 'Sensor RMS'} = smth_snsr_rms;
    
    % Plot the estimated sensors for this lambda
    nexttile;
    hold on;
    plot(1:n_wvln, rgb_snsr(:, 1), 'r', 'DisplayName', 'True Red');
    plot(1:n_wvln, rgb_snsr(:, 2), 'g', 'DisplayName', 'True Green');
    plot(1:n_wvln, rgb_snsr(:, 3), 'b', 'DisplayName', 'True Blue');
    plot(1:n_wvln, est_spct_quadprog(:, 1), 'r--', 'DisplayName', 'Est. Red');
    plot(1:n_wvln, est_spct_quadprog(:, 2), 'g--', 'DisplayName', 'Est. Green');
    plot(1:n_wvln, est_spct_quadprog(:, 3), 'b--', 'DisplayName', 'Est. Blue');
    title(['Lambda = ', num2str(lambda)]);
    xlabel('Wavelength');
    ylabel('Sensitivity');
    legend;
    hold off;
end

title(f10, 'Smoothed quadprog() solutions');
exportgraphics(f10, 'output/f10_lambda_smooth_constr_lst_sqr.png', 'Resolution', 200);

disp(results);

% close all;

end