
function [overall, column_wise] = compute_rms_error(varargin)
    if (nargin == 1)
        diff = varargin{1};
    elseif (nargin == 2)
        diff = varargin{1} - varargin{2};
    else 
        error('Function compute_rms_error needs either one or two arguments.\n'); 
    end
    
    

    [m n] = size(diff);
    diff_sqrd = diff .^ 2;
    sum_diff_sqrd_cols = sum(diff_sqrd);

    column_wise = sqrt(sum_diff_sqrd_cols / m);

    overall = sqrt(sum(sum_diff_sqrd_cols) / (m * n));
end

