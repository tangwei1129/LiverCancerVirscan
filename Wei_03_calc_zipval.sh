#!/bin/bash
#INPUT.count.csv.gz may vary in each run
mkdir zipval_count_done
mkdir log
find raw_count_virus_reads/*.gz | perl -ne 'chomp; $file=$_; $out=$file; $out=~ s/raw_count_virus_reads\///g;$out=~ s/csv.gz/zipval.csv/g; $cmd = "/data/tangw3/miniconda/envs/virscan/bin/python src/calc_zipval.py $file input.COUNT.csv.gz log > zipval_count_done/$out; gzip zipval_count_done/$out"; print $cmd."\n";' > calc_zipval.swarm
[tangw3@cn0849 code]$ less calc_zipval.swarm

swarm -t 4 -g 32 --time 04:00:00 -f calc_zipval.swarm
