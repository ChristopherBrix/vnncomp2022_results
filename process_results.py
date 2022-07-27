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

from counterexamples import is_correct_counterexample

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

    def __init__(self, tool_name, csv_path, cpu_benchmarks, skip_benchmarks):
        assert "csv" in csv_path

        self.tool_name = tool_name
        self.category_to_list = defaultdict(list) # maps category -> list of results

        self.skip_benchmarks = skip_benchmarks
        self.cpu_benchmarks = cpu_benchmarks
        self.gpu_overhead = np.inf # default overhead
        self.cpu_overhead = np.inf # if using separate overhead for cpu
        
        self.max_prepare = 0.0
        self.had_error = False

        self.load(csv_path)

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

    def load(self, csv_path):
        """load data from file"""

        unexpected_results = set()
                
        with open(csv_path, newline='') as csvfile:
            for row in csv.reader(csvfile):
                # rename results

                #if row[0] == "errored":
                #    had_error = True
                #    continue
                
                #print(f"{csv_path}: {row}")
                
                row[ToolResult.RESULT] = row[ToolResult.RESULT].lower()

                substitutions = [('unsat', 'holds'),
                                 ('sat', 'violated'),
                                 ('no_result_in_file', 'unknown'),
                                 ('prepare_instance_error_', 'unknown'),
                                 ('run_instance_timeout', 'timeout'),
                                 ('prepare_instance_timeout', 'timeout'),
                                 ('error_exit_code_', 'error'),
                                 ('error_nonmaximal', 'unknown'),
                                 ]

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

                if cat in self.skip_benchmarks:
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

        self.delete_empty_categories()

    def delete_empty_categories(self):
        """delete categories without successful measurements"""

        to_remove = ['acasxu', 'cifar2020'] # benchmarks to skip

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
                print(f"deleting {key} in tool {self.tool_name}")
                del self.category_to_list[key]

        ToolResult.num_categories[self.tool_name] = len(self.category_to_list)

def compare_results(result_list, single_overhead):
    """compare results across tools"""

    min_percent = 0 # minimum percent for total score

    total_score = defaultdict(int)
    all_cats = {}

    for cat in sorted(ToolResult.all_categories):
        print(f"\nCategory {cat}:")

        # maps tool_name -> [score, num_verified, num_falsified, num_fastest]
        cat_score: Dict[str, List[int, int, int, int]] = {}
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

                    ce_path = f"./{t.tool_name}/{cat}/{net}_{prop}.counterexample.gz"

                    assert Path(ce_path).is_file()
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
                    is_correct = is_correct_counterexample(*tup)

                    correct_violations[tool] = is_correct

                print(f"were violated counterexamples valid?: {correct_violations}")
                
            print(f"Row: {table_row}")
            
            for t in participating_tools:
                res, secs = t.single_result(cat, index)
                
                score, is_verified, is_falsified, is_fastest = get_score(t.tool_name, res, secs, rand_gen_succeeded,
                                                                times_holds, times_violated,
                                                                correct_violations)
                print(f"{index}: {t.tool_name} score: {score}, is_ver: {is_verified}, is_fals: {is_falsified}, " + \
                      f"is_fastest: {is_fastest}")

                if t.tool_name in cat_score:
                    tool_score_tup = cat_score[t.tool_name]
                else:
                    tool_score_tup = [0, 0, 0, 0]
                    cat_score[t.tool_name] = tool_score_tup

                # [score, num_verified, num_falsified, num_fastest]
                tool_score_tup[0] += score
                tool_score_tup[1] += 1 if is_verified else 0
                tool_score_tup[2] += 1 if is_falsified else 0
                tool_score_tup[3] += 1 if is_fastest else 0
                tool_score_tup = None
                

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

    for cat in sorted(all_cats.keys()):
        cat_score = all_cats[cat]

        if not cat_score:
            continue
        
        print(f"\n% Category {cat} (single_overhead={single_overhead}):")
        res_list = []
        max_score = max([t[0] for t in cat_score.values()])

        cat_str = cat.replace('_', '-')
        
        print_table_header(f"Benchmark \\texttt{{{cat_str}}}", "tab:cat_{cat}",
                           ("\\# ~", "Tool", "Verified", "Falsified", "Fastest", "Score", "Percent"),
                           align='lllllrr')
                
        for tool, score_tup in cat_score.items():
            score, num_verified, num_falsified, num_fastest = score_tup
            
            percent = max(min_percent, 100 * score / max_score)
            tool_latex = latex_tool_name(tool)
            
            #desc = f"{tool}: {score} ({round(percent, 2)}%)"
            desc = f"{tool_latex} & {num_verified} & {num_falsified} & {num_fastest} & {score} & {round(percent, 1)}\\% \\\\"

            res_list.append((percent, desc))

        for i, s in enumerate(reversed(sorted(res_list))):
            #print(f"{i+1}. {s[1]}")
            print(f"{i+1} & {s[1]}")

        print_table_footer()

    res_list = []

    print(f"\nTotal Score (single_overhead={single_overhead}):")

    print_table_header("Overall Score", "tab:score", ["\\# ~", "Tool", "Score"])
    
    for tool, score in total_score.items():
        tool_latex = latex_tool_name(tool)
        desc = f"{tool_latex} & {round(score, 1)} \\\\"

        res_list.append((score, desc))

    for i, s in enumerate(reversed(sorted(res_list))):
        print(f"{i+1} & {s[1]}")

    print_table_footer()

def print_table_header(title, label, columns, align=None):
    """print latex table header"""

    bold_columns = ["\\textbf{" + c + "}" for c in columns]

    if align is None:
        align = 'l' * len(columns)
    else:
        assert len(columns) == len(align)

    print('\n\\begin{table}[h]')
    print('\\begin{center}')
    print('\\caption{' + title + '} \\label{' + label + '}')
    print('{\\setlength{\\tabcolsep}{2pt}')
    print('\\begin{tabular}[h]{@{}' + align + '@{}}')
    print('\\toprule')
    print(' & '.join(bold_columns) + "\\\\")
    #\textbf{\# ~} & \textbf{Tool} & \textbf{Score}  \\
    print('\\midrule')

def print_table_footer():
    """print latex table footer"""

    print('''\\bottomrule
\\end{tabular}
}
\\end{center}
\\end{table}\n\n''')

def get_score(tool_name, res, secs, rand_gen_succeded, times_holds, times_violated, correct_violations):
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

    is_verified = False
    is_falsified = False
    is_fastest = False

    num_holds = len(times_holds)
    num_violated = len(times_violated)

    #print(f"tool: {tool_name} {res}")

    valid_ce = False

    for is_correct in correct_violations.values():
        if is_correct:
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
    elif num_holds > 0 and res == "violated" and not correct_violations[tool_name]:
        # Rule: If a witness is not provided, for the purposes of scoring if there are
        # mismatches between tools we will count the tool without the witness as incorrect.
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
        print(f"tool {tool_name} did not produce a valid counterexample and there are mismatching results")
    elif res == "violated" and not valid_ce:
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
    elif res == "holds" and valid_ce:
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
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

    return score, is_verified, is_falsified, is_fastest

def print_stats(result_list):
    """print stats about measurements"""

    print('\n------- Stats ----------')

    print("\nOverhead:")
    olist = []

    for r in result_list:
        olist.append((r.gpu_overhead, r.cpu_overhead, r.tool_name))

    #print_table_header("Overhead", "tab:overhead", ["\\# ~", "Tool", "Seconds", "~~CPU Mode"], align='llrr')
    print_table_header("Overhead", "tab:overhead", ["\\# ~", "Tool", "Seconds"], align='llr')
        
    for i, n in enumerate(sorted(olist)):
        cpu_overhead = "-" if n[1] == np.inf else round(n[1], 1)
        
        #print(f"{i+1} & {n[2]} & {round(n[0], 1)} & {cpu_overhead} \\\\")
        print(f"{i+1} & {n[2]} & {round(n[0], 1)} \\\\")

    print_table_footer()

    items = [("Num Benchmarks Participated", ToolResult.num_categories),
             ("Num Instances Verified", ToolResult.num_verified),
             ("Num Violated", ToolResult.num_violated),
             ("Num Holds", ToolResult.num_holds),
             ("Mismatched (Incorrect) Results", ToolResult.incorrect_results),
             ]

    for index, (label, d) in enumerate(items):
        print(f"\n% {label}:")

        tab_label = f"tab:stats{index}"
        print_table_header(label, tab_label, ["\\# ~", "Tool", "Count"], align='llr')

        l = []

        for tool, count in d.items():
            tool_latex = latex_tool_name(tool)
            
            l.append((count, tool_latex))
        
        for i, s in enumerate(reversed(sorted(l))):
            print(f"{i+1} & {s[1]} & {s[0]} \\\\")

        print_table_footer()

def latex_tool_name(tool):
    """get latex version of tool name"""

    if tool == 'a-b-CROWN':
        tool = '$\\alpha$,$\\beta$-CROWN'

    return tool

def main():
    """main entry point"""

    # use single overhead for all tools. False will have two different overheads for some tools depending
    # on if GPU needed to be initialized (manually entered)
    single_overhead = True
    print(f"using single_overhead={single_overhead}")

    #####################################3
    #csv_list = glob.glob("results_csv/*.csv")
    csv_list = glob.glob("*/results.csv")
    csv_list.sort()
    
    tool_list = [c.split('/')[0] for c in csv_list]
    result_list = []

    cpu_benchmarks = {x: [] for x in tool_list}
    skip_benchmarks = {x: [] for x in tool_list}
    #skip_benchmarks['RPM'] = ['mnistfc']
    
    if not single_overhead: # Define a dict with the cpu_only benchmarks for each tool
        pass
        #cpu_benchmarks["ERAN"] = ["acasxu", "eran"]

    had_error = False

    for csv_path, tool_name in zip(csv_list, tool_list):
        tr = ToolResult(tool_name, csv_path, cpu_benchmarks[tool_name], skip_benchmarks[tool_name])
        result_list.append(tr)

        if tr.had_error:
            had_error = True

    # compare results across tools
    compare_results(result_list, single_overhead)

    print_stats(result_list)

    print("\nTODO: Delete acasxu and cifar2020")
    print("TODO: cgdtest had prepare_instance_timeout, set as timeout on line 100")

    if had_error:
        print("WARNING: some tools had errors that were ignored")
    else:
        print("Success. No tools had errors.")

if __name__ == "__main__":
    main()
