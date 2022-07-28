'''
vnn comp global settings
'''

class Settings:
    '''static container for settings'''

    CSV_GLOB = "../*/results.csv"
    TOOL_LIST_GLOB_INDEX = 1

    UNSCORED_CATEGORIES = ['acasxu', 'cifar2020']

    BENCHMARK_REPO = "/home/stan/repositories/vnncomp2022_benchmarks"
    COUNTEREXAMPLE_TOL = 1e-4

    TOOL_NAME_SUBS = [
            ('alpha_beta_crown', '$\\alpha$,$\\beta$-CROWN'),
            ('mn_bab', 'MN BaB')
            ]

    SKIP_TOOLS = [] #['marabou', 'verapak', 'cgdtest']

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
    
