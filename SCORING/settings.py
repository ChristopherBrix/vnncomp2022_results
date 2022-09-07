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

    UNSCORED_CATEGORIES = ['acasxu', 'cifar2020']

    BENCHMARK_REPO = "/home/stan/repositories/vnncomp2022_benchmarks"
    COUNTEREXAMPLE_TOL = 1e-4

    TOOL_NAME_SUBS_LATEX = [
            ('alpha_beta_crown', '$\\alpha$,$\\beta$-CROWN'),
            ('mn_bab', 'MN BaB')
            ]

    TOOL_NAME_SUBS_GNUPLOT = [
        ('alpha_beta_crown', 'AB-CROWN'),
        ('mn_bab', 'MN BaB')
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

    # gnuplot information
    gnuplot_data = (
        GnuplotSettings('all', 'All Instances'),
        #
        GnuplotSettings('carvana_unet_2022', 'Carvana Unet 2022'),
        GnuplotSettings('cifar100_tinyimagenet_resnet', 'CIFAR100 TinyImageNet'),
        GnuplotSettings('cifar_biasfield', 'CIFAR Biasfield'),
        GnuplotSettings('collins_rul_cnn', 'Collins Rul CNN'),
        GnuplotSettings('mnist_fc', 'MNIST FC'),
        GnuplotSettings('nn4sys', 'NN4SYS'),
        GnuplotSettings('oval21', 'OVAL 21'),
        GnuplotSettings('reach_prob_density', 'Reach Prob Density'),
        GnuplotSettings('rl_benchmarks', 'RL Benchmarks'),
        GnuplotSettings('sri_resnet_a', 'SRI Resnet A'),
        GnuplotSettings('sri_resnet_b', 'SRI Resnet B'),
        GnuplotSettings('tllverifybench', 'Two-Level Lattice Verify Benchmark'),
        GnuplotSettings('vggnet16_2022', 'VGGNet16 2022'),
        )
    
assert Path(Settings.BENCHMARK_REPO).is_dir(), f"directory in Settings.BENCHMARK_REPO ('{Settings.BENCHMARK_REPO}') " + \
    "doesn't exist. Please clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks and edit " + \
    "path in Settings.BENCHMARK_REPO in settings.py"
