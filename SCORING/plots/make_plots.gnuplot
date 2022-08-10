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

#outputs = 'all.pdf rl_benchmarks.pdf'
outputs = 'all.pdf'

    
##########
load 'generated.gnuplot'
#input_list = "'acasxu-all-NNV acasxu-all-MIPVerify acasxu-all-PeregriNN acasxu-all-venus acasxu-all-eran acasxu-all-nnenum'"

#pretty_input_list = "\"'NNV' 'MIPVerify' 'PeregriNN' 'Venus' 'ERAN' 'nnenum'\" \
                     \"'NNV' 'Venus' 'PeregriNN' 'MIPVerify' 'VeriNet' 'nnenum' 'ERAN'\" \
                     \"'NNV' 'MIPVerify' 'Oval' 'VeriNet' 'nnenum' 'ERAN'\" \
                     \"'ERAN' 'Oval'\" \
                     \"'NNV' 'Verinet' 'ERAN'\""
############

title_list = "'All Instances' 'RL Benchmarks'"

xmax_list = '950 297'
xmax_plot_list = '1000 310'

ymin_list = '0.9 0.9'
ymax_list = '2000 100'

timeout_y_list = '300 60'
timeout_x_list = '7 7'
timeout_str_list = "'Five Minutes' 'One Minute'"

key_list = "'1400 1000' '420 90'"
key_font_str_list = ",20 ,20"
key_box_str_list = 'box box'
all_instances_y_list = '3 90'

ratio_list = '0.5 0.5'
sizex_list = '7.0 7.0'
sizey_list = '3 3'

time_label_offset_list = '1.7 1.7'
lmargin_str_list = '0.08 0.08'

do for [index=1:words(outputs)] {
	
input_str = word(input_list, index)
pretty_input_str = word(pretty_input_list, index)
num_inputs = words(input_str)
title_str = word(title_list, index)
xmax_str = word(xmax_list, index)
ymax_str = word(ymax_list, index)
xmax_plot_str = word(xmax_plot_list, index)
ymin_str = word(ymin_list, index)
all_instances_y_str = word(all_instances_y_list, index)
ratio_str = word(ratio_list, index)
size_xstr = word(sizex_list, index)
size_ystr = word(sizey_list, index)
key_str = word(key_list, index)
time_label_offset_str = word(time_label_offset_list, index)
timeout_y = word(timeout_y_list, index)
timeout_x = word(timeout_x_list, index)
timeout_str = word(timeout_str_list, index)
key_font_str = word(key_font_str_list, index)
lmargin_str = word(lmargin_str_list, index)
key_box_str = word(key_box_str_list, index)

set terminal pdfcairo enhanced dashed font 'Times,20' fontscale 0.45 dl 0.4 size size_xstr, size_ystr
set lmargin at screen lmargin_str
set bmargin at screen 0.15
set tmargin at screen 0.9
set rmargin at screen 0.98
set size ratio ratio_str

keyx_str = word(key_str, 1)
keyy_str = word(key_str, 2)


set key on at keyx_str, keyy_str font key_font_str

if (key_box_str eq 'box') {set key box} 
else {set key nobox}

set ylabel "Time (sec)" offset time_label_offset_str,0

xmax = int(xmax_str)
xmax_plot = int(xmax_plot_str)
set xrange [0:xmax_plot]

set yrange [ymin_str:ymax_str]

# outfile = sprintf('animation/bessel%03.0f.png',t)
set output word(outputs, index)

set title title_str font 'Times,28' offset 0,-1

#set arrow 50 from 0, 60 to xmax, 60 nohead ls 9
#set label 21 "One Minute" at 3,60 font 'Times,14' tc rgb "#808080" offset 0, -0.4

set arrow 51 from 0, timeout_y to xmax, timeout_y nohead ls 20
set label 22 timeout_str at timeout_x,timeout_y font 'Times,20' tc rgb "#808080" offset 0, -0.5

set arrow 52 from xmax, ymin_str to xmax, ymax_str nohead ls 20 
all_instances_y = int(all_instances_y_str)
set label 24 "All Instances" at xmax,all_instances_y font 'Times,18' tc rgb "#000000" offset -1, 0 rotate right

set grid ytics lc '#606060' lw 0.25 lt 1 dt 3
set grid xtics lc '#606060' lw 0.25 lt 1 dt 3

plot for [i=1:num_inputs] "./accumulated-".word(input_str, i).".txt" using 2:1 title word(pretty_input_str, i) with lines ls (i)
    
unset arrow 50
unset arrow 51
unset arrow 52
    
}

