# gnuplot script to generate plots for VNN-COMP
# made for gnuplot 5.2 patchlevel 8
# Stanley Bak, July 2020

unset label
set xtic auto
set ytic auto
set xlabel "Number of Instances Verified" offset 0,0.3

set style line 1 lc rgb "#ff4040" lw 5.0 dt 2
set style line 2 lc rgb "#008000" lw 3.0 dt 4

set style line 3 lc rgb "#00E0E0" lw 3.0 dt 6
set style line 4 lc rgb "#ff8080" lw 5.5 dt 8
set style line 5 lc rgb "#ef8d00" lw 3.0 dt 5

set style line 6 lc rgb "#9400D3" lw 0.5 dt 6
set style line 7 lc rgb "#00FF00" lw 3.5 dt 4
    
set style line 8 lc rgb "#8080ff" lw 2.5 dt 2

set style line 9 lc rgb "#ff00ff" lw 6.5 dt 1

set style line 10 lc rgb "#ff4040" lw 2.5 dt 1
set style line 11 lc rgb "#0000FF" lw 2.0 dt 1

# 20 = timeout line
set style line 20 lc rgb "#808080" lw 2.0 pt 13 ps 0.8 dt 2

#set logscale x
set logscale y
#set format x "10^{%L}"

    
##########
load 'generated.gnuplot'
##########

#all_instances_y_list = '3 90'

do for [index=1:words(outputs)] {
	
input_str = word(input_list, index)
pretty_input_str = word(pretty_input_list, index)
num_inputs = words(input_str)
title_str = word(title_list, index)

ymin_str = word(ymin_list, index)
ymax_str = word(ymax_list, index)
xmax_plot_str = word(xmax_plot_list, index)

#all_instances_y_str = word(all_instances_y_list, index)
ratio_str = '0.5'
size_xstr = '7.0
size_ystr = '3'
key_str = word(key_list, index)
time_label_offset_str = '1.7'
timeout_y = word(timeout_y_list, index)
timeout_x = word(timeout_x_list, index)
timeout_str = word(timeout_str_list, index)
key_font_str = ',20'
lmargin_str = '0.08'
tool_index_str = word(tool_index_list, index)

set terminal pdfcairo enhanced dashed font 'Times,20' fontscale 0.45 dl 0.4 size size_xstr, size_ystr
set lmargin at screen lmargin_str
set bmargin at screen 0.15
set tmargin at screen 0.9
set rmargin at screen 0.98
set size ratio ratio_str

keyx_str = word(key_str, 1)
keyy_str = word(key_str, 2)


set key on at keyx_str, keyy_str left top font key_font_str nobox


set ylabel "Time (sec)" offset time_label_offset_str,0

xmax_plot = int(xmax_plot_str)
set xrange [1:xmax_plot]

set yrange [ymin_str:ymax_str]

# outfile = sprintf('animation/bessel%03.0f.png',t)
set output word(outputs, index)

set title title_str font 'Times,28' offset 0,-1

# time label: "One Minute"
set arrow 51 from 1, timeout_y to xmax_plot, timeout_y nohead ls 20
set label 22 timeout_str at timeout_x,timeout_y font 'Times,20' tc rgb "#808080" offset 0, -0.5

#set arrow 52 from xmax, ymin_str to xmax, ymax_str nohead ls 20 
#all_instances_y = int(all_instances_y_str)
#set label 24 "All Instances" at xmax,all_instances_y font 'Times,18' tc rgb "#000000" offset -1, 0 rotate right

set grid ytics lc '#606060' lw 0.25 lt 1 dt 3
set grid xtics lc '#606060' lw 0.25 lt 1 dt 3

plot for [i=1:num_inputs] "./accumulated-".word(input_str, i).".txt" using 2:1 title word(pretty_input_str, i) with lines ls (1 + word(tool_index_str, i))
    
unset arrow 50
unset arrow 51
unset arrow 52
    
}

