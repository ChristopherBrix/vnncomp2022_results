"""
Process vnncomp results

Stanley Bak
"""

from typing import Dict, List

import glob
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

from counterexamples import is_correct_counterexample, CounterexampleResult
from settings import Settings

class ToolResult:
    """Tool's result"""

    # columns
    CATEGORY = 0
    NETWORK = 1
    PROP = 2
    PREPARE_TIME = 3
    RESULT = 4
    RUN_TIME = 5

    all_categories = set()

    # stats
    num_verified = defaultdict(int) # number of benchmarks verified
    num_violated = defaultdict(int) 
    num_holds = defaultdict(int)
    incorrect_results = defaultdict(int)

    num_categories = defaultdict(int)
    toolerror_counts = defaultdict(int)

    def __init__(self, scored, tool_name, csv_path, cpu_benchmarks, skip_benchmarks):
        assert "csv" in csv_path

        self.tool_name = tool_name
        self.category_to_list = defaultdict(list) # maps category -> list of results

        self.skip_benchmarks = skip_benchmarks
        self.cpu_benchmarks = cpu_benchmarks
        self.gpu_overhead = np.inf # default overhead
        self.cpu_overhead = np.inf # if using separate overhead for cpu
        
        self.max_prepare = 0.0

        self.load(scored, csv_path)

    @staticmethod
    def reset():
        """reset static variables"""

        ToolResult.all_categories = set()

        # stats
        ToolResult.num_verified = defaultdict(int) # number of benchmarks verified
        ToolResult.num_violated = defaultdict(int) 
        ToolResult.num_holds = defaultdict(int)
        ToolResult.incorrect_results = defaultdict(int)

        ToolResult.num_categories = defaultdict(int)

        ToolResult.toolerror_counts = defaultdict(int)

    def result_instance_str(self, cat, index):
        """get a string representation of the instance for the given category and index"""

        row = self.category_to_list[cat][index]

        net = row[ToolResult.NETWORK]
        prop = row[ToolResult.PROP]

        return Path(net).stem + "-" + Path(prop).stem

    def single_result(self, cat, index):
        """get result_str, runtime of tool, after subtracting overhead"""

        row = self.category_to_list[cat][index]

        res = row[ToolResult.RESULT]
        t = float(row[ToolResult.RUN_TIME])

        t -= self.cpu_overhead if cat in self.cpu_benchmarks else self.gpu_overhead

        # all results less than 1.0 second are treated the same
        t = max(1.0, t)

        return res, t

    def load(self, scored, csv_path):
        """load data from file"""

        unexpected_results = set()
                
        with open(csv_path, newline='') as csvfile:
            for row in csv.reader(csvfile):
                # rename results
                
                #print(f"{csv_path}: {row}")
                
                row[ToolResult.RESULT] = row[ToolResult.RESULT].lower()

                substitutions = Settings.CSV_SUBSTITUTIONS

                for from_prefix, to_str in substitutions:
                    if row[ToolResult.RESULT] == '': # don't use '' as prefix
                        row[ToolResult.RESULT] = 'unknown'
                    elif row[ToolResult.RESULT].startswith(from_prefix):
                        row[ToolResult.RESULT] = to_str

                network = row[ToolResult.NETWORK]
                result = row[ToolResult.RESULT]
                cat = row[ToolResult.CATEGORY]
                prepare_time = float(row[ToolResult.PREPARE_TIME])
                run_time = float(row[ToolResult.RUN_TIME])

                # workaround to drop convBigRELU from cifar2020
                if cat == 'cifar2020':
                    if 'convBigRELU' in network:
                        result = row[ToolResult.RESULT] = "unknown"

                if cat in self.skip_benchmarks or \
                        (scored and cat in Settings.UNSCORED_CATEGORIES) or \
                        (not scored and cat not in Settings.UNSCORED_CATEGORIES):
                    result = row[ToolResult.RESULT] = "unknown"

                if result.startswith('timeout'):
                    result = 'timeout' # fix for verapak "timeout(X_00 ..."

                if not ("test_nano" in network or "test_tiny" in network):
                    self.category_to_list[cat].append(row)

                if result not in ["holds", "violated", "timeout", "error", "unknown"]:
                    unexpected_results.add(result)
                    print(f"Unexpected results: {unexpected_results}")
                    exit(1)

                if result in ["holds", "violated"]:
                    if cat in self.cpu_benchmarks:
                        self.cpu_overhead = min(self.cpu_overhead, run_time)
                    else:
                        self.gpu_overhead = min(self.gpu_overhead, run_time)
                        
                    self.max_prepare = max(self.max_prepare, prepare_time)

        assert not unexpected_results, f"Unexpected results: {unexpected_results}"

        print(f"Loaded {self.tool_name}, default-overhead (gpu): {round(self.gpu_overhead, 1)}s," + \
              f"cpu-overhead: {round(self.cpu_overhead, 1)}s, " + \
              f"prepare time: {round(self.max_prepare, 1)}s")

        for skip_benchmark in self.skip_benchmarks:
            assert skip_benchmark in self.category_to_list, f"skip benchmark '{skip_benchmark}' not found in cat " + \
                f"list: {list(self.category_to_list.keys())}"

        self.delete_empty_categories()

    def delete_empty_categories(self):
        """delete categories without successful measurements"""

        to_remove = [] #['acasxu', 'cifar2020'] # benchmarks to skip

        for key in self.category_to_list.keys():
            rows = self.category_to_list[key]

            should_remove = True

            for row in rows:
                result = row[ToolResult.RESULT]

                if result in ('holds', 'violated'):
                    
                    should_remove = False
                    break

            if should_remove:
                to_remove.append(key)
            elif key != "test":
                ToolResult.all_categories.add(key)

        for key in to_remove:
            if key in self.category_to_list:
                #print(f"empty category {key} in tool {self.tool_name}")
                del self.category_to_list[key]

        ToolResult.num_categories[self.tool_name] = len(self.category_to_list)

def compare_results(all_tool_names, result_list, single_overhead):
    """compare results across tools"""

    min_percent = 0 # minimum percent for total score

    total_score = defaultdict(int)
    all_cats = {}

    tool_times = {}

    ### TODO: if you have time later, allow for multiple gnuplot categories
    # you probably still will need to manually provide plot settings
    # also: the sorting of tools in gnuplot is based on total score, not category score
    gnuplot_cat = 'all'

    for tool in all_tool_names:
        tool_times[tool] = []

    for cat in sorted(ToolResult.all_categories):
        print(f"\nCategory {cat}:")

        # maps tool_name -> [score, num_verified, num_falsified, num_fastest, num_errors]
        cat_score: Dict[str, List[int, int, int, int, int]] = {}
        all_cats[cat] = cat_score

        num_rows = 0

        participating_tools = []

        for tool_result in result_list:
            cat_dict = tool_result.category_to_list

            if not cat in cat_dict:
                continue
            
            rows = cat_dict[cat]
            assert num_rows == 0 or len(rows) == num_rows, f"tool {tool_result.tool_name}, cat {cat}, " + \
                f"got {len(rows)} rows expected {num_rows}"

            if num_rows == 0:
                num_rows = len(rows)
                print(f"Category {cat} has {num_rows} (from {tool_result.tool_name})")

            participating_tools.append(tool_result)

        # work with participating tools only
        tool_names = [t.tool_name for t in participating_tools]
        print(f"{len(participating_tools)} participating tools: {tool_names}")
        table_rows = []
        all_times = []
        all_results = []

        for index in range(num_rows):
            rand_gen_succeeded = False
            times_holds = []
            tools_holds = []
            times_violated = []
            tools_violated = []
            counterexamples_violated = []
            correct_violations = {}
            
            table_row = []
            table_rows.append(table_row)
            instance_str = participating_tools[0].result_instance_str(cat, index)
            table_row.append(instance_str)

            for t in participating_tools:
                res, secs = t.single_result(cat, index)

                if res == "unknown":
                    table_row.append("-")
                    continue

                if not res in ["holds", "violated"]:
                    table_row.append(res)
                    continue

                if res == "holds":
                    times_holds.append(secs)
                    tools_holds.append(t.tool_name)
                else:
                    assert res == "violated"
                    times_violated.append(secs)
                    tools_violated.append(t.tool_name)

                    # construct counterexample path
                    row = t.category_to_list[cat][index]
                    net = Path(row[ToolResult.NETWORK]).stem
                    prop = Path(row[ToolResult.PROP]).stem

                    ce_path = f"../{t.tool_name}/{cat}/{net}_{prop}.counterexample.gz"

                    assert Path(ce_path).is_file(), f"CE path not found: {ce_path}"
                    tup = ce_path, cat, net, prop
                    counterexamples_violated.append(tup)

                table_row.append(f"{round(secs, 1)} ({res[0]})")

                if t.tool_name == "randgen":
                    assert res == "violated"
                    rand_gen_succeeded = True

            print()

            if times_holds and times_violated:
                print(f"WARNING: multiple results for index {index}. Violated: {len(times_violated)} " +
                      f"({tools_violated}), Holds: {len(times_holds)} ({tools_holds})")
                table_row.append('*multiple results*')

                for tup, tool in zip(counterexamples_violated, tools_violated):
                    print(f"\nchecking counterexample for {tool}")
                    res = is_correct_counterexample(*tup)

                    correct_violations[tool] = res

                print(f"were violated counterexamples valid?: {correct_violations}")
                
            print(f"Row: {table_row}")

            row_times = []
            all_times.append(row_times)
            all_results.append(None)
            
            for t in participating_tools:
                res, secs = t.single_result(cat, index)
                
                score, is_verified, is_falsified, is_fastest, is_error = get_score(t.tool_name, res, secs, rand_gen_succeeded,
                                                                times_holds, times_violated,
                                                                correct_violations)
                print(f"{index}: {t.tool_name} score: {score}, is_ver: {is_verified}, is_fals: {is_falsified}, " + \
                      f"is_fastest: {is_fastest}")

                if is_verified or is_falsified:
                    all_results[-1] = 'H' if is_verified else 'V'
                    row_times.append(secs)
                else:
                    row_times.append(None)

                if t.tool_name in cat_score:
                    tool_score_tup = cat_score[t.tool_name]
                else:
                    tool_score_tup = [0, 0, 0, 0, 0]
                    cat_score[t.tool_name] = tool_score_tup

                # [score, num_verified, num_falsified, num_fastest]
                tool_score_tup[0] += score
                tool_score_tup[1] += 1 if is_verified else 0
                tool_score_tup[2] += 1 if is_falsified else 0
                tool_score_tup[3] += 1 if is_fastest else 0
                tool_score_tup[4] += 1 if is_error else 0
                tool_score_tup = None

        print("--------------------")
        num_holds = 0
        num_violated = 0
        num_unknown = 0

        for i, (row_times, result) in enumerate(zip(all_times, all_results)):
            assert len(row_times) == len(tool_names)

            if result is None:
                num_unknown += 1
            else:
                if gnuplot_cat == 'all' or gnuplot_cat == cat:
                    for t, tool in zip(row_times, tool_names):
                        if t is not None:
                            tool_times[tool].append(t)
                
                if result == 'V':
                    num_violated += 1
                elif result == 'H':
                    num_holds += 1
        
        print(f"Total Violated: {num_violated}")
        print(f"Total Holds: {num_holds}")
        print(f"Total Unknown: {num_unknown}")

        print("--------------------")
        print(", ".join(tool_names))

        for table_row in table_rows:
            print(", ".join(table_row))

        print(f"---------\nCategory {cat}:")

        if cat_score:
            max_score = max([t[0] for t in cat_score.values()])

            for tool, score_tup in cat_score.items():
                score = score_tup[0]
                percent = max(min_percent, 100 * score / max_score)
                print(f"{tool}: {score} ({round(percent, 2)}%)")

                total_score[tool] += percent

    print("\n###############")
    print("### Summary ###")
    print("###############")

    sorted_tools = []

    with open(Settings.TOTAL_SCORE_LATEX, 'w', encoding='utf-8') as f:
        tee(f, "\n%Total Score:")
        res_list = []

        print_table_header(f, "Overall Score", "tab:score", ["\\# ~", "Tool", "Score"])

        for tool, score in total_score.items():
            tool_latex = latex_tool_name(tool)
            desc = f"{tool_latex} & {round(score, 1)} \\\\"

            res_list.append((score, desc, tool))

        for i, s in enumerate(reversed(sorted(res_list))):
            sorted_tools.append(s[2])
            
            tee(f, f"{i+1} & {s[1]}")

        print_table_footer(f)

    #######

    print("--------------------")

    for tool in all_tool_names:
        times_list = tool_times[tool]
        times_list.sort()

        with open(Settings.PLOTS_DIR + f"/accumulated-all-{tool}.txt", 'w', encoding='utf-8') as f:
            for i, t in enumerate(times_list):
                f.write(f"{t}\t{i+1}\n")

    with open(Settings.PLOTS_DIR + "/generated.gnuplot", 'w', encoding='utf-8') as f:
        f.write("input_list = \"'")

        for tool in sorted_tools:
            f.write(f"all-{tool} ")

        f.write("'\"\n\n")

        f.write("pretty_input_list = \"\\\"")
        for tool in sorted_tools:
            f.write(f"'{gnuplot_tool_name(tool)}' ")

        f.write("\\\"\"\n")


    #######

    for cat in sorted(all_cats.keys()):
        cat_score = all_cats[cat]

        if not cat_score:
            continue

        filename = Settings.UNSCORED_LATEX if cat in Settings.UNSCORED_CATEGORIES else Settings.SCORED_LATEX

        with open(filename, 'a', encoding='utf-8') as f:
        
            tee(f, f"\n% Category {cat} (single_overhead={single_overhead}):")
            res_list = []
            max_score = max([t[0] for t in cat_score.values()])

            cat_str = cat.replace('_', '-')

            print_table_header(f, f"Benchmark \\texttt{{{cat_str}}}", "tab:cat_{cat}",
                               ("\\# ~", "Tool", "Verified", "Falsified", "Fastest", "Penalty", "Score", "Percent"),
                               align='llllllrr')

            for tool, score_tup in cat_score.items():
                score, num_verified, num_falsified, num_fastest, num_error = score_tup

                percent = max(min_percent, 100 * score / max_score)
                tool_latex = latex_tool_name(tool)

                #desc = f"{tool}: {score} ({round(percent, 2)}%)"
                desc = f"{tool_latex} & {num_verified} & {num_falsified} & {num_fastest} & {num_error} & {score} & {round(percent, 1)}\\% \\\\"

                res_list.append((percent, desc))

            for i, s in enumerate(reversed(sorted(res_list))):
                tee(f, f"{i+1} & {s[1]}")

            print_table_footer(f)

def tee(fout, line):
    """print to temrinal and a file"""
    
    print(line)
    fout.write(line + "\n")

def print_table_header(f, title, label, columns, align=None):
    """print latex table header"""

    bold_columns = ["\\textbf{" + c + "}" for c in columns]

    if align is None:
        align = 'l' * len(columns)
    else:
        assert len(columns) == len(align)

    tee(f, '\n\\begin{table}[h]')
    tee(f, '\\begin{center}')
    tee(f, '\\caption{' + title + '} \\label{' + label + '}')
    tee(f, '{\\setlength{\\tabcolsep}{2pt}')
    tee(f, '\\begin{tabular}[h]{@{}' + align + '@{}}')
    tee(f, '\\toprule')
    tee(f, ' & '.join(bold_columns) + "\\\\")
    #\textbf{\# ~} & \textbf{Tool} & \textbf{Score}  \\
    tee(f, '\\midrule')

def print_table_footer(f):
    """print latex table footer"""

    tee(f, '''\\bottomrule
\\end{tabular}
}
\\end{center}
\\end{table}\n\n''')

def get_score(tool_name, res, secs, rand_gen_succeded, times_holds, times_violated, ce_results):
    """Get the score for the given result
    Actually returns a 4-tuple: score, is_verified, is_falsified, is_fastest

    Correct hold: 10 points
    Correct violated (where random tests did not succeed): 10 points
    Correct violated (where random test succeeded): 1 point
    Incorrect result: -100 points

    Time bonus: 
        The fastest tool for each solved instance will receive +2 points. 
        The second fastest tool will receive +1 point.
        If two tools have runtimes within 0.2 seconds, we will consider them the same runtime.
    """

    penalize_no_ce = False

    is_verified = False
    is_falsified = False
    is_fastest = False
    is_error = False

    num_holds = len(times_holds)
    num_violated = len(times_violated)

    #print(f"tool: {tool_name} {res}")

    valid_ce = False

    for ce_valid_res in ce_results.values():
        if ce_valid_res == CounterexampleResult.CORRECT:
            valid_ce = True
            break

    assert not rand_gen_succeded, "VNNCOMP 2022 didn't use randgen"

    if res not in ["holds", "violated"]:
        score = 0
    elif rand_gen_succeded:
        assert res == "violated"
        score = 1

        ToolResult.num_verified[tool_name] += 1
        ToolResult.num_violated[tool_name] += 1

        is_falsified = True
    elif penalize_no_ce and num_holds > 0 and res == "violated" and not ce_results[tool_name]:
        # Rule: If a witness is not provided, for the purposes of scoring if there are
        # mismatches between tools we will count the tool without the witness as incorrect.
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
        print(f"tool {tool_name} did not produce a valid counterexample and there are mismatching results")

        ToolResult.toolerror_counts[f'{tool_name}_no-ce-but-required'] += 1
        is_error = True
    elif res == "violated" and num_holds > 0 and not valid_ce:
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
        is_error = True

        ToolResult.toolerror_counts[f'{tool_name}_{ce_results[tool_name]}'] += 1
    elif res == "holds" and valid_ce:
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
        is_error = True

        ToolResult.toolerror_counts[f'{tool_name}_incorrect_unsat'] += 1
    else:
        # correct result!

        ToolResult.num_verified[tool_name] += 1

        if res == "holds":
            is_verified = True
            times = times_holds.copy()
            ToolResult.num_holds[tool_name] += 1
        else:
            assert res == "violated"
            times = times_violated.copy()
            ToolResult.num_violated[tool_name] += 1

            is_falsified = True
            
        score = 10

        min_time = min(times)

        if secs < min_time + 0.2:
            score += 2
            is_fastest = True
        else:
            times.remove(min_time)
            second_time = min(times)

            if secs < second_time + 0.2:
                score += 1

    return score, is_verified, is_falsified, is_fastest, is_error

def print_stats(result_list):
    """print stats about measurements"""

    with open(Settings.STATS_LATEX, 'w', encoding='utf-8') as f:
        tee(f, '\n%%%%%%%%%% Stats %%%%%%%%%%%')

        tee(f, "\n% Overhead:")
        olist = []

        for r in result_list:
            olist.append((r.gpu_overhead, r.cpu_overhead, r.tool_name))

        #print_table_header("Overhead", "tab:overhead", ["\\# ~", "Tool", "Seconds", "~~CPU Mode"], align='llrr')
        print_table_header(f, "Overhead", "tab:overhead", ["\\# ~", "Tool", "Seconds"], align='llr')

        for i, n in enumerate(sorted(olist)):
            #cpu_overhead = "-" if n[1] == np.inf else round(n[1], 1)

            #print(f"{i+1} & {n[2]} & {round(n[0], 1)} & {cpu_overhead} \\\\")
            tee(f, f"{i+1} & {latex_tool_name(n[2])} & {round(n[0], 1)} \\\\")

        print_table_footer(f)

        items = [("Num Benchmarks Participated", ToolResult.num_categories),
                 ("Num Instances Verified", ToolResult.num_verified),
                 ("Num SAT", ToolResult.num_violated),
                 ("Num UNSAT", ToolResult.num_holds),
                 ("Incorrect Results (or Missing CE)", ToolResult.incorrect_results),
                 ]

        for index, (label, d) in enumerate(items):
            tee(f, f"\n% {label}:")

            tab_label = f"tab:stats{index}"
            print_table_header(f, label, tab_label, ["\\# ~", "Tool", "Count"], align='llr')

            l = []

            for tool, count in d.items():
                tool_latex = latex_tool_name(tool)

                l.append((count, tool_latex))

            for i, s in enumerate(reversed(sorted(l))):
                tee(f, f"{i+1} & {s[1]} & {s[0]} \\\\")

            print_table_footer(f)

    print(ToolResult.toolerror_counts)            

def latex_tool_name(tool):
    """get latex version of tool name"""

    subs = Settings.TOOL_NAME_SUBS_LATEX

    found = False

    for old, new in subs:
        if tool == old:
            tool = new
            found = True
            break

    if not found:
        tool = tool.capitalize()

    return tool

def gnuplot_tool_name(tool):
    """get fnuplot version of tool name"""

    subs = Settings.TOOL_NAME_SUBS_GNUPLOT

    found = False

    for old, new in subs:
        if tool == old:
            tool = new
            found = True
            break

    if not found:
        tool = tool.capitalize()

    return tool

def main():
    """main entry point"""

    # use single overhead for all tools. False will have two different overheads for some tools depending
    # on if GPU needed to be initialized (manually entered)
    single_overhead = True
    print(f"using single_overhead={single_overhead}")

    #####################################3
    #csv_list = glob.glob("results_csv/*.csv")
    csv_list = glob.glob(Settings.CSV_GLOB)
    csv_list.sort()

    # clear files so we can append to them
    with open(Settings.SCORED_LATEX, 'w', encoding='utf-8') as f:
        pass

    with open(Settings.UNSCORED_LATEX, 'w', encoding='utf-8') as f:
        pass

    if Settings.SKIP_TOOLS:
        new_csv_list = []

        for csv_file in csv_list:
            skip_tool = False
            
            for skip in Settings.SKIP_TOOLS:
                if skip in csv_file:
                    skip_tool = True
                    break

            if not skip_tool:
                new_csv_list.append(csv_file)

            csv_list = new_csv_list

    tool_list = [c.split('/')[Settings.TOOL_LIST_GLOB_INDEX] for c in csv_list]

    cpu_benchmarks = {x: [] for x in tool_list}
    skip_benchmarks = {x: [] for x in tool_list}
    #skip_benchmarks['RPM'] = ['mnistfc']

    for tool, benchmark in Settings.SKIP_BENCHMARK_TUPLES:
        assert tool in tool_list, f"{tool} not in tool list: {tool_list}"
        skip_benchmarks[tool].append(benchmark)

    if not single_overhead: # Define a dict with the cpu_only benchmarks for each tool
        #pass
        cpu_benchmarks["ERAN"] = ["acasxu", "eran"]

    for scored in [False, True]:
        result_list = []
        ToolResult.reset()

        for csv_path, tool_name in zip(csv_list, tool_list):
            tr = ToolResult(scored, tool_name, csv_path, cpu_benchmarks[tool_name], skip_benchmarks[tool_name])
            result_list.append(tr)

        # compare results across tools
        compare_results(tool_list, result_list, single_overhead)

        if scored:
            print_stats(result_list)

    if Settings.SKIP_TOOLS:
        print(f"Note: tools were skipped: {Settings.SKIP_TOOLS}")

if __name__ == "__main__":
    #from counterexamples import get_ce_diff
    #get_ce_diff.clear_cache()
    #exit(1)
    main()
