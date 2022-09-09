'''
vnn comp global settings
'''

from pathlib import Path

class GnuplotSettings:
    """settings for gnuplot"""

    def __init__(self, prefix, title):
        self.prefix = prefix
        self.title = title

class Settings:
    '''static container for settings'''

    CSV_GLOB = "../*/results.csv"
    TOOL_LIST_GLOB_INDEX = 1

    SCORING_MIN_TIME = 1.0
    PLOT_MIN_TIME = 0 #0.01

    UNSCORED_CATEGORIES = ['acasxu', 'cifar2020']

    BENCHMARK_REPO = "/home/stan/repositories/vnncomp2022_benchmarks"
    COUNTEREXAMPLE_TOL = 1e-4

    TOOL_NAME_SUBS_LATEX = [
            ('alpha_beta_crown', '$\\alpha$,$\\beta$ Crown'),
            ('mn_bab', 'MN BaB')
            ]

    TOOL_NAME_SUBS_LONGTABLE = [
            ('alpha_beta_crown', '$\\alpha$,$\\beta$-C'),
            ('mn_bab', 'MnB'),
            ('peregrinn', 'Pereg'),
            ('fastbatllnn', 'FastBaT'),
            ('verapak', 'Verap'),
            ('nnenum', 'nnen'),
            ('verinet', 'Verin'),
            ('averinn', 'Averi'),
            ('marabou', 'Marab'),
            ('debona', 'Debon'),
            ('cgdtest', 'CGD')
            ]

    TOOL_NAME_SUBS_GNUPLOT = [
        ('alpha_beta_crown', 'AB-CROWN'),
        ('mn_bab', 'MN BaB')
        ]

    CAT_NAME_SUBS_LATEX = [
        ('carvana_unet_2022', 'carvana 2022'),
        ('cifar100_tinyimagenet_resnet', 'cifar100 tiny'),
        ('reach_prob_density', 'reach prob den')
        ]

    SKIP_TOOLS = [] #['marabou', 'verapak', 'cgdtest']

    SKIP_BENCHMARK_TUPLES = [('marabou', 'sri_resnet_a'), ('marabou', 'sri_resnet_b')]

    PLOTS_DIR = "./plots"

    CSV_SUBSTITUTIONS = [('unsat', 'holds'),
                         ('sat', 'violated'),
                         ('no_result_in_file', 'unknown'),
                         ('prepare_instance_error_', 'unknown'),
                         ('run_instance_timeout', 'timeout'),
                         ('prepare_instance_timeout', 'timeout'),
                         ('error_exit_code_', 'error'),
                         ('error_nonmaximal', 'unknown'),
                         ]

    # latex output files
    TOTAL_SCORE_LATEX = "latex/total.tex"
    SCORED_LATEX = "latex/scored.tex"
    UNSCORED_LATEX = "latex/unscored.tex"
    STATS_LATEX = "latex/stats.tex"
    LONGTABLE_LATEX = "latex/longtable.tex"

    # gnuplot information
    PLOT_FOLDER = "cactus" # folder containing the .pdfs
    
    gnuplot_data = (
        GnuplotSettings('all', 'All Instances'),
        #
        GnuplotSettings('acasxu', 'ACAS Xu (Unscored)'),
        GnuplotSettings('cifar2020', 'CIFAR 2020 (Unscored)'),
        #
        GnuplotSettings('carvana_unet_2022', 'Carvana Unet 2022'),
        GnuplotSettings('cifar100_tinyimagenet_resnet', 'CIFAR100 Tiny ImageNet ResNet'),
        GnuplotSettings('cifar_biasfield', 'CIFAR Biasfield'),
        GnuplotSettings('collins_rul_cnn', 'Collins Rul CNN'),
        GnuplotSettings('mnist_fc', 'MNIST FC'),
        GnuplotSettings('nn4sys', 'NN4SYS'),
        GnuplotSettings('oval21', 'OVAL 21'),
        GnuplotSettings('reach_prob_density', 'Reachability Probability Density'),
        GnuplotSettings('rl_benchmarks', 'Reinforcement Learning Benchmarks'),
        GnuplotSettings('sri_resnet_a', 'SRI Resnet A'),
        GnuplotSettings('sri_resnet_b', 'SRI Resnet B'),
        GnuplotSettings('tllverifybench', 'Two-Level Lattice Verify Benchmark'),
        GnuplotSettings('vggnet16_2022', 'VGGNet16 2022'),
        )
    
assert Path(Settings.BENCHMARK_REPO).is_dir(), f"directory in Settings.BENCHMARK_REPO ('{Settings.BENCHMARK_REPO}') " + \
    "doesn't exist. Please clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks and edit " + \
    "path in Settings.BENCHMARK_REPO in settings.py"
