
function write_figure(base_fname)
    % Perhaps these two variables should be globals, and/or passed in as an
    % optional argument with values like 'ps' or 'eps'. 
    want_eps = 1;
    want_pdf = 1;
    figure_dir = 'figures/';

    if (strlength(figure_dir) > 0) 
        cmd = sprintf('mkdir -p %s', figure_dir');
        system(cmd);
    end 

    if (want_eps) 
        eps_fname = sprintf('%s%s.eps', figure_dir, base_fname);
        print('-depsc2', eps_fname)
    end 

    if (want_pdf) 
        pdf_fname = sprintf('%s%s.pdf', figure_dir, base_fname);

        % This does not really work because it thinks it is sending to a
        % printer.
        %
        % print('-dpdf', pdf_fname)

        cmd = sprintf('epstopdf %s --outfile="%s"', eps_fname, pdf_fname);
        system(cmd);
    end 
end

