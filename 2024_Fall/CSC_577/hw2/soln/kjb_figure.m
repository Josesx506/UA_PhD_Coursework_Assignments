
function [ h ] = kjb_figure()
    h = figure;

    ss = get(0,'screensize'); %The screen size
    screen_width = ss(3);
    screen_height = ss(4);

    % We will try to offset figures by 500 in the horizontal direction and 200
    % in the vertical. 
    %

    fig_num = get(h, 'Number');
    cur_pos = get(h, 'Position');

    num_horizontal = round(screen_width / 500) ;
    num_vertical = round((screen_height - 500) / 200) ;

    horizontal_tile_index = mod(fig_num-1, num_horizontal);

    vertical_tile_index =  floor((fig_num - 1) / (num_horizontal)) ;

    vertical_tile_index = mod(vertical_tile_index, num_vertical) ;

    new_j = screen_height - 600 - 200*vertical_tile_index;
    new_i = 40 + 500*horizontal_tile_index;

    new_pos = [ new_i new_j cur_pos(3) cur_pos(4) ];

    set(h, 'Position', new_pos);
end




