#!/bin/bash -e

python3 process_results.py

# run again, capturing output to file
python3 process_results.py > results.txt && pushd plots && gnuplot make_plots.gnuplot && cp *.pdf ../latex/ && popd && pushd latex && make ; popd


