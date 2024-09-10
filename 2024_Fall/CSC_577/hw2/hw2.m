function [num_questions] = hw2(infile)
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
rgb_snsr = importdata(infile);
[r,g,b] = deal(rgb_snsr(:,1), rgb_snsr(:,2), rgb_snsr(:,3));

f1 = figure;
plot(r, Color='r');
hold on;
plot(g, Color='g');
plot(b, Color='b');
legend("r","g","b");xlabel('row index');ylabel('amplitude');
hold off;
exportgraphics(f1, 'output/f1_rgb.png', 'Resolution', 200);

% Set the random number generator seed and scale the values to give 10^-4
rng(477);
mult_scale = 9.4e-4;
spectra = rand(1600, 101) * mult_scale;

rnd_rgb = spectra * rgb_snsr;
rnd_rgb40 = reshape(rnd_rgb, [40, 40, 3]);

% min max across channels
rmin = min(rnd_rgb(:, 1),[],"all");
gmin = min(rnd_rgb(:, 2),[],"all");
bmin = min(rnd_rgb(:, 3),[],"all");
omin = min(rnd_rgb(:));

rmax = max(rnd_rgb(:, 1),[],"all");
gmax = max(rnd_rgb(:, 2),[],"all");
bmax = max(rnd_rgb(:, 3),[],"all");
omax = max(rnd_rgb(:));

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

f2 = figure;
imshow(uint8(rnd_rgb400));
exportgraphics(f2, 'output/f2_rgb400by400.png', 'Resolution', 200);

num_questions = num_questions + 1;


%% Q2
fprintf('\n\nQuestion 2:\n');
% inv_spct = (spectra' * spectra) \ (spectra' * rnd_rgb);  % Size (101 x 3)
inv_spct = linsolve(spectra' * spectra, spectra' * rnd_rgb);
x = 1:size(rgb_snsr, 1); 

figure('Position', [1, 1, 820, 400]);
f3 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; hold on; % True Spectra
plot(x, rgb_snsr(:,1), 'r', 'LineWidth', 2);
plot(x, rgb_snsr(:,2), 'g', 'LineWidth', 2);
plot(x, rgb_snsr(:,3), 'b', 'LineWidth', 2);
hold off; title('True Spectra'); legend("r","g","b");
xlabel('row index');ylabel('spectra amplitude');

nexttile; hold on; % Estimated Spectra
plot(x, inv_spct(:,1), 'r--', 'LineWidth', 2);
plot(x, inv_spct(:,2), 'g--', 'LineWidth', 2);
plot(x, inv_spct(:,3), 'b--', 'LineWidth', 2);
hold off; title('Estimated Spectra'); 
legend("r","g","b"); xlabel('row index'); ylim([0 2.5e4]);

exportgraphics(f3, 'output/f3_inverted_spectra.png', 'Resolution', 200);

% Compute the rms for each spectra band
rms_red = sqrt(mean((rgb_snsr(:,1) - inv_spct(:,1)).^2, 'all'));
rms_grn = sqrt(mean((rgb_snsr(:,2) - inv_spct(:,2)).^2, 'all'));
rms_blu = sqrt(mean((rgb_snsr(:,3) - inv_spct(:,3)).^2, 'all'));

% Compute the RGB responses using the estimated sensors
rnd_rgb_inv = spectra * inv_spct;
% Compute the RMS error between the true and estimated RGB values
rms_rgb = sqrt(mean((rnd_rgb - rnd_rgb_inv).^2, 'all'));


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
        rnd_spectra = rand(1600, 101);
        
        if ~isempty(noise_std)
            noise = noise_std * randn(size(rnd_spectra));
            rnd_spectra = rnd_spectra + noise;
        end
        
        if clip
            rnd_spectra = max(0, min(255, rnd_spectra));
        end

        rgb_data = rnd_spectra * inp_spct;
        
        % Invert the image sensitivity spectrum
        out_spct = linsolve(rnd_spectra' * rnd_spectra, rnd_spectra' * rgb_data);

        % Generate an rgb spectrum from the inverted light responses
        rgb_inv = rnd_spectra * out_spct;
        
        % Calculate the rms error for sensors and rgb 
        snsr_rms = sqrt(mean((inp_spct - out_spct).^2, 'all'));
        rgb_rms = sqrt(mean((rgb_data - rgb_inv).^2, 'all'));
    end

    function [r_err,g_err,b_err] = channel_sensitivity_rms_error(true_sens,esti_sens)
        r_err = sqrt(mean((true_sens(:,1) - esti_sens(:,1)).^2, 'all'));
        g_err = sqrt(mean((true_sens(:,2) - esti_sens(:,2)).^2, 'all'));
        b_err = sqrt(mean((true_sens(:,3) - esti_sens(:,3)).^2, 'all'));
    end

% Invert for the camera sensitivities and estimate the rms errors
[spec_noise10,srms10,rrms10] = simulate_sensitivity(rgb_snsr,10,false);
[rms_red, rms_grn, rms_blu] = channel_sensitivity_rms_error(rgb_snsr,spec_noise10);

figure('Position', [1, 1, 820, 400]);
f4 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
nexttile; hold on; % True Spectra
plot(x, rgb_snsr(:,1), 'r', 'LineWidth', 2);
plot(x, rgb_snsr(:,2), 'g', 'LineWidth', 2);
plot(x, rgb_snsr(:,3), 'b', 'LineWidth', 2);
hold off; title('True Spectra'); legend("r","g","b");
xlabel('row index');ylabel('spectra amplitude');

nexttile; hold on; % Estimated Spectra
plot(x, spec_noise10(:,1), 'r--', 'LineWidth', 2);
plot(x, spec_noise10(:,2), 'g--', 'LineWidth', 2);
plot(x, spec_noise10(:,3), 'b--', 'LineWidth', 2);
hold off; title(sprintf('Estimated Spectra for noise std %d', 400)); 
legend("r","g","b"); xlabel('row index'); ylim([0 2.5e4]);

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
    [est_snst,unc_snst_rms,unc_rgb_rms] = simulate_sensitivity(rgb_snsr,n_std,false);
    [est_snst,clp_snst_rms,clp_rgb_rms] = simulate_sensitivity(rgb_snsr,n_std,true);
    
    results{i, 'Noise Std'} = n_std;
    results{i, 'Unclipped RGB RMS'} = unc_rgb_rms;
    results{i, 'Clipped RGB RMS'} = clp_rgb_rms;
    results{i, 'Unclipped Sensor RMS'} = unc_snst_rms;
    results{i, 'Clipped Sensor RMS'} = clp_snst_rms;
end

disp(results);

figure('Position', [1, 1, 820, 400]);
f5 = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');

nexttile; hold on;
plot(results.('Noise Std'),results.('Unclipped RGB RMS'),'black-');
plot(results.('Noise Std'),results.('Clipped RGB RMS'),'b--');
legend("unclipped","clipped"); xlabel('Noise Std.'); 
ylabel('Error magnitude'); title('RGB RMS errors');

nexttile; hold on;
plot(results.('Noise Std'),results.('Unclipped Sensor RMS'),'black-');
plot(results.('Noise Std'),results.('Clipped Sensor RMS'),'b--');
legend("unclipped","clipped"); xlabel('Noise Std.'); 
title('Sensor RMS errors'); yscale log;
hold off;

exportgraphics(f5, 'output/f5_rms_err_vs_noise.png', 'Resolution', 200);

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




close all;

end