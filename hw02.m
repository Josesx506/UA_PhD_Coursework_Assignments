
% The script file hw1 produces some of the results for CS 477 home work
% assignment two. 
%
% Use of this file is restricted to students in the CS 477/577 class offered in
% the term that this file was posted (and only if it was posted!).  In
% particular, please do not share this file with anyone who might take this
% course at a future time, which might use similar questions on an assignment. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function hw2()

    close all 
    format compact 

    rng(477);

    do_01 = 1;
    do_02_03_04 = 1;
    do_05 = 1;
    do_06 = 1;
    do_07 = 1;
    do_08 = 1;
  
    % This tells Matlab the color of successive lines when plotting lines in a
    % matrix to be R,G, B.
    %
    set(groot,'DefaultAxesColorOrder',[1,0,0; 0,1,0; 0,0,1]); 

    num_samples = 1600;

    global x_values;
    x_values = [380:4:780]';

    % Needed for multiple questions. 
    %
    sensors=load('rgb_sensors.txt');
    l = rand(num_samples, 101);
    k = 255 / max(max(l*sensors));
    l = l * k;
    rgb = l*sensors;
    num_channels = size(rgb, 2);

    if (do_01) 
        fprintf(1, '---------------------------------------------------------\n');
        fprintf(1, 'HW two, question one\n\n');

        kjb_figure();
        plot(sensors);
        title('Camera sensitivities');
        xlabel('Wavelength in nanometers');
        ylabel('Sensitivity (arbitrary scale)');
        write_figure('sensors'); 

        % Note the cast. Matlab data types cause no end of headaches!
        image = uint8(zeros(400,400,3));

        bs = 10;

        % Notice that we fill each block with implicit loops
        count = 1;
        for i=1:40
            for j=1:40
                image(1+(i-1)*bs:i*bs, 1+(j-1)*bs:j*bs, 1)=uint8(rgb(count,1));
                image(1+(i-1)*bs:i*bs, 1+(j-1)*bs:j*bs, 2)=uint8(rgb(count,2));
                image(1+(i-1)*bs:i*bs, 1+(j-1)*bs:j*bs, 3)=uint8(rgb(count,3));
                count = count + 1;
            end
        end

        kjb_figure;
        imshow(image);
        write_figure('visualization_collage'); 
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (do_02_03_04) 
        fprintf(1, '---------------------------------------------------------\n');
        fprintf(1, 'HW two, questions two through four\n\n');

        % When i is 0 we are doing question 2, when it is 1 we are doing
        % question 3.
        %
        result_count = 1;

        for (i=0:50) 
            noise_stdev = 10 * i; 
            base_noisy_rgb = rgb + noise_stdev * randn(num_samples, num_channels);

            for (clip = 0:1) 
                if (clip) 
                    clip_str = 'with clipping';
                    filename_clip_str = '_clipped';
                    noisy_rgb = max(base_noisy_rgb, 0);
                    noisy_rgb = min(base_noisy_rgb, 255);
                else 
                    clip_str = 'without clipping';
                    filename_clip_str = '_without_clipping';
                    noisy_rgb = base_noisy_rgb;
                end

                if (i == 0) 
                    % Amazingly this is not precisely zero. I am not sure why. 
                    norm(noisy_rgb - base_noisy_rgb)
                end 

                % Fancy way to do solve for the sensors. 
                estimated_sensors = l \ noisy_rgb;

                [overall_sensor_rms(clip+1) sensor_rms] = compute_rms_error(estimated_sensors, sensors);

                estimated_rgb = l*estimated_sensors;
                [overall_rgb_rms(clip+1) rgb_rms] = compute_rms_error(estimated_rgb, rgb);


                if ((i==0) || (i==1) || (i==5) || (i==10)) 
                    title_str_one = sprintf('Estimated and actual sensors for noise level %.0f %s', i*10., clip_str);
                    title_str_two = sprintf('Overall RMS error for estimated sensors is %.0f and for reconstructed RGB is %.1f', overall_sensor_rms(clip+1), overall_rgb_rms(clip+1));
                    filename_str = sprintf('synthetic_data_noise_%.0f%s', i*10., filename_clip_str);
                    
                    plot_sensors_and_estimates(sensors, estimated_sensors, {title_str_one, title_str_two, ' '});
                    write_figure(filename_str); 
                end
            end

            error_table( result_count, :) = [ noise_stdev overall_rgb_rms(1) overall_rgb_rms(2) overall_sensor_rms(1) overall_sensor_rms(2) ];

            result_count = result_count + 1;
        end

        format shortg
        round(error_table, 3, 'significant')
        clear error_table;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if (do_05) 
        fprintf(1, '---------------------------------------------------------\n');
        fprintf(1, 'HW two, question five\n\n');
        fprintf(1, 'This does not need Matlab but we include a solution here to keep things organized. \n\n');

        fprintf(1, 'We are told that the gray that matched equal black and white was g1 = 80.\n');
        fprintf(1, 'If it were linear, g2 = 255/2.\n');
        fprintf(1, 'Assuming that single gamma map can fix this, we have g2=255*(g1/255)^(1/gamma).\n');
        fprintf(1, 'Taking logs says: log(g2/255) = (1/gamma)*log(g1/255)\n');
        fprintf(1, 'So, gamma = log(g1/255) / log(g2/255)\n');

        g1=80
        g2=255/2
        gamma = log(g1/255) / log(g2/255)

        fprintf(1, 'Check\n');

        255*(g1/255)^(1/gamma)
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Needed for multiple questions. 
    light_spectra = load('light_spectra.txt');
    responses = load('responses.txt');
    num_channels = size(responses, 2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    result_count = 1;

    if (do_06) 
        fprintf(1, '---------------------------------------------------------\n');
        fprintf(1, 'HW two, question five\n\n');

        estimated_sensors = light_spectra \ responses;

        [overall_sensor_rms sensor_rms] = compute_rms_error(estimated_sensors, sensors);
        estimated_responses = light_spectra * estimated_sensors;
        [overall_response_rms response_rms ] = compute_rms_error(estimated_responses, responses)

        error_table( result_count, :) = [ sensor_rms response_rms ];
        result_count = result_count + 1;

        title_str_one = sprintf('Estimated and actual sensors for real data');
        title_str_two = sprintf('Overall RMS error for estimated sensors is %.0f and for reconstructed RGB is %.1f', overall_sensor_rms, overall_response_rms)
                    
        plot_sensors_and_estimates(sensors, estimated_sensors, {title_str_one, title_str_two, ' '});
        write_figure('least_squares_for_real_data'); 
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (do_07) 
        fprintf(1, '---------------------------------------------------------\n');
        fprintf(1, 'HW two, question six\n\n');

        for (i = 1:num_channels)
            X(:,i) = positive_least_squares(light_spectra, responses(:,i));
        end 

        [overall_sensor_rms sensor_rms ] = compute_rms_error(X, sensors)
        estimated_responses = light_spectra * X;
        [overall_response_rms response_rms ] = compute_rms_error(estimated_responses, responses)

        error_table( result_count, :) = [ sensor_rms response_rms ];
        result_count = result_count + 1;

        title_str_one = 'Constrained least squares to enforce positive recovered sensors';
        title_str_two = sprintf('Overall RMS error for estimated sensors is %.0f and for reconstructed RGB is %.1f',...
                                overall_sensor_rms, overall_response_rms);
        plot_sensors_and_estimates(sensors, X, {title_str_one, title_str_two, ' '});
        write_figure('non_negative_least_squares_for_real_data'); 
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (do_08) 
        fprintf(1, '---------------------------------------------------------\n');
        fprintf(1, 'HW two, question seven\n\n');

        diff_M_p1 = [eye(100) zeros(100, 1)];
        diff_M_p2 = [zeros(100, 1) eye(100)];
        diff_M = diff_M_p1 - diff_M_p2; 

        % We do not need do do lambda=0 as the previous question already did it. 
        lambda_vals = [0.001 0.01 0.03 0.1 1];

        for (lambda_index = 1:size(lambda_vals,2))
            lambda = lambda_vals(lambda_index);

            smooth_U = [light_spectra; lambda * diff_M]; 

            for (i = 1:num_channels)
                smooth_y = [responses(:,i); zeros(100,1)];
                X(:,i) = positive_least_squares(smooth_U, smooth_y);
            end 

            [ overall_sensor_rms_vs_lambda(lambda_index) sensor_rms ] = compute_rms_error(X, sensors);

            estimated_responses = light_spectra * X;
            [ overall_response_rms_vs_lambda(lambda_index) response_rms ] = compute_rms_error(estimated_responses, responses);

            error_table( result_count, :) = [ sensor_rms response_rms ];
            result_count = result_count + 1;

            title_str = sprintf('Constrained least squares with smoothing (lambda=%.3f)', lambda);
            plot_sensors_and_estimates(sensors, X, title_str);

            % Putting decimals in the file name confuses LaTeX who thinks that
            % it must be a suffix! So multiply by 1000. 
            %
            filename_str = sprintf('non_negative_least_squares_lambda_%d', int32(round(1000*lambda)));
            write_figure(filename_str); 
        end

        overall_sensor_rms_vs_lambda
        overall_response_rms_vs_lambda

        fprintf(1, '---------------------------------------------------------\n');
    end

    if (result_count > 1)
        format shortg
        round(error_table, 3, 'significant')
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ x ] = positive_least_squares(U, y)

    H = U'*U;

    num_samples  = size(y, 1);
    [num_samples_2 num_unknowns] = size(U);

    assert(num_samples == num_samples_2);

    A = - eye(num_unknowns);
    b = zeros(num_unknowns, 1);

    X  = zeros(num_unknowns);

    options = optimset('MaxIter', 1000, 'Display', 'none');

    f = -y'*U;
    [x fval fexitflag] = quadprog(H, f, A, b, [], [], [], [], [], options);
end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_sensors_and_estimates(sensors, estimated_sensors, title_str)
    global x_values

    kjb_figure;

    hold on;
    plot(x_values, sensors(:,1), 'r', 'LineWidth', 1);
    plot(x_values, sensors(:,2), 'g', 'LineWidth', 1);
    plot(x_values, sensors(:,3), 'b', 'LineWidth', 1);
    plot(x_values, estimated_sensors(:,1), 'r-.', 'LineWidth', 2);
    plot(x_values, estimated_sensors(:,2), 'g-.', 'LineWidth', 2);
    plot(x_values, estimated_sensors(:,3), 'b-.', 'LineWidth', 2);
    xlabel('Wavelength in nanometers');
    ylabel('Sensitivity (obscure units)');
    title(title_str); 

    legend('Actual sensor (red)', 'Actual sensor (green)','Actual sensor (blue)', ...
           'Estimated sensor (red)', 'Estimated sensor (green)','Estimated sensor (blue)'); 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function write_figure(base_fname)
    figure_dir = 'figures/';
    have_epstopdf = 1;

    if (strlength(figure_dir) > 0) 
        cmd = sprintf('mkdir -p %s', figure_dir');
        system(cmd);
    end 

    eps_fname = sprintf('%s%s.eps', figure_dir, base_fname);
    print('-depsc2', eps_fname)

    if (have_epstopdf)
        pdf_fname = sprintf('%s%s.pdf', figure_dir, base_fname);
        cmd = sprintf('epstopdf %s --outfile="%s"', eps_fname, pdf_fname);
        system(cmd);
    end
end

