input_list = "'all-alpha_beta_crown all-mn_bab all-verinet all-nnenum all-cgdtest all-peregrinn all-marabou all-debona all-fastbatllnn all-verapak all-averinn ' 'carvana_unet_2022-alpha_beta_crown carvana_unet_2022-mn_bab carvana_unet_2022-verinet ' 'cifar100_tinyimagenet_resnet-alpha_beta_crown cifar100_tinyimagenet_resnet-mn_bab cifar100_tinyimagenet_resnet-verinet cifar100_tinyimagenet_resnet-cgdtest ' 'cifar_biasfield-alpha_beta_crown cifar_biasfield-mn_bab cifar_biasfield-verinet cifar_biasfield-nnenum cifar_biasfield-cgdtest cifar_biasfield-marabou cifar_biasfield-verapak ' 'collins_rul_cnn-alpha_beta_crown collins_rul_cnn-mn_bab collins_rul_cnn-verinet collins_rul_cnn-nnenum collins_rul_cnn-cgdtest collins_rul_cnn-peregrinn ' 'mnist_fc-alpha_beta_crown mnist_fc-mn_bab mnist_fc-verinet mnist_fc-nnenum mnist_fc-cgdtest mnist_fc-peregrinn mnist_fc-marabou mnist_fc-debona mnist_fc-verapak ' 'nn4sys-alpha_beta_crown nn4sys-mn_bab nn4sys-verinet nn4sys-nnenum nn4sys-cgdtest nn4sys-peregrinn nn4sys-debona ' 'oval21-alpha_beta_crown oval21-mn_bab oval21-verinet oval21-nnenum oval21-cgdtest oval21-peregrinn oval21-marabou ' 'reach_prob_density-alpha_beta_crown reach_prob_density-mn_bab reach_prob_density-verinet reach_prob_density-nnenum reach_prob_density-cgdtest reach_prob_density-peregrinn reach_prob_density-marabou reach_prob_density-debona ' 'rl_benchmarks-alpha_beta_crown rl_benchmarks-mn_bab rl_benchmarks-verinet rl_benchmarks-nnenum rl_benchmarks-cgdtest rl_benchmarks-peregrinn rl_benchmarks-marabou rl_benchmarks-debona rl_benchmarks-verapak rl_benchmarks-averinn ' 'sri_resnet_a-alpha_beta_crown sri_resnet_a-mn_bab sri_resnet_a-verinet sri_resnet_a-cgdtest ' 'sri_resnet_b-alpha_beta_crown sri_resnet_b-mn_bab sri_resnet_b-verinet sri_resnet_b-cgdtest ' 'tllverifybench-alpha_beta_crown tllverifybench-mn_bab tllverifybench-verinet tllverifybench-nnenum tllverifybench-cgdtest tllverifybench-peregrinn tllverifybench-marabou tllverifybench-debona tllverifybench-fastbatllnn ' 'vggnet16_2022-alpha_beta_crown vggnet16_2022-mn_bab vggnet16_2022-verinet vggnet16_2022-nnenum vggnet16_2022-cgdtest ' "

pretty_input_list = "\"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' 'Marabou' 'Debona' 'Fastbatllnn' 'Verapak' 'Averinn' \" \"'AB-CROWN' 'MN BaB' 'Verinet' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Cgdtest' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Marabou' 'Verapak' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' 'Marabou' 'Debona' 'Verapak' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' 'Debona' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' 'Marabou' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' 'Marabou' 'Debona' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' 'Marabou' 'Debona' 'Verapak' 'Averinn' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Cgdtest' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Cgdtest' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' 'Peregrinn' 'Marabou' 'Debona' 'Fastbatllnn' \" \"'AB-CROWN' 'MN BaB' 'Verinet' 'Nnenum' 'Cgdtest' \" "

tool_index_list = "'0 1 2 3 4 5 6 7 8 9 10 ' '0 1 2 ' '0 1 2 4 ' '0 1 2 3 4 6 9 ' '0 1 2 3 4 5 ' '0 1 2 3 4 5 6 7 9 ' '0 1 2 3 4 5 7 ' '0 1 2 3 4 5 6 ' '0 1 2 3 4 5 6 7 ' '0 1 2 3 4 5 6 7 9 10 ' '0 1 2 4 ' '0 1 2 4 ' '0 1 2 3 4 5 6 7 8 ' '0 1 2 3 4 ' "

title_list = "'All Instances' 'Carvana Unet 2022' 'CIFAR100 TinyImageNet' 'CIFAR Biasfield' 'Collins Rul CNN' 'MNIST FC' 'NN4SYS' 'OVAL 21' 'Reach Prob Density' 'RL Benchmarks' 'SRI Resnet A' 'SRI Resnet B' 'Two-Level Lattice Verify Benchmark' 'VGGNet16 2022' "

outputs = 'all.pdf carvana_unet_2022.pdf cifar100_tinyimagenet_resnet.pdf cifar_biasfield.pdf collins_rul_cnn.pdf mnist_fc.pdf nn4sys.pdf oval21.pdf reach_prob_density.pdf rl_benchmarks.pdf sri_resnet_a.pdf sri_resnet_b.pdf tllverifybench.pdf vggnet16_2022.pdf '

xmax_plot_list = '997.5 40.95 99.75 74.55 64.05 88.2 159.6 27.3 37.800000000000004 310.8 33.6 40.95 33.6 15.75 '

ymin_str = '0.9'

timeout_y_list = '300 60 60 60 60 60 60 300 300 60 60 60 300 300 '

timeout_str_list = "'Five Minutes' 'One Minute' 'One Minute' 'One Minute' 'One Minute' 'One Minute' 'One Minute' 'Five Minutes' 'Five Minutes' 'One Minute' 'One Minute' 'One Minute' 'Five Minutes' 'Five Minutes' "

timeout_x_list = '10.975 1.4095 1.9975 1.7454999999999998 1.6404999999999998 1.8820000000000001 2.596 1.2730000000000001 1.3780000000000001 4.1080000000000005 1.336 1.4095 1.336 1.1575 '

ymax_list = '1504.8966933615 408.4165105395 287.254787184 385.2140720415 304.333060704 427.48821768150003 384.4932007725 1037.507272794 590.1033440835 143.2913627955 356.31095452349996 283.595479764 891.8467143314999 1504.8966933615 '

key_list = "'1047.375 1504.8966933615' '42.9975 408.4165105395' '104.73750000000001 287.254787184' '78.2775 385.2140720415' '67.2525 304.333060704' '92.61000000000001 427.48821768150003' '167.58 384.4932007725' '28.665000000000003 1037.507272794' '39.690000000000005 590.1033440835' '326.34000000000003 143.2913627955' '35.28 356.31095452349996' '42.9975 283.595479764' '35.28 891.8467143314999' '16.5375 1504.8966933615' "

