#!/bin/bash
module load python/3.7
cmd="python bin/concat_tables_sum_rows.py "
files=$(find *.zihit.csv.gz | perl -ne 'chomp; $in=$_; if ($in =~ m/(.+)_IDX(\d+).zihit.csv.gz/ && $2 < 91) {print $in." ";}')
$cmd $files> LMW_NHITS.SAMPLES.csv
gzip LMW_NHITS.SAMPLES.csv
