#!/bin/bash -e

python3 process_results.py

# run again, capturing output to file
python3 process_results.py > results.txt && pushd latex && make ; popd
