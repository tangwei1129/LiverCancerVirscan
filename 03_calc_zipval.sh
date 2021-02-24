#!/bin/bash
#INPUT.count.csv.gz may vary in each run
mkdir zipval_count_done
mkdir log
find raw_count_virus_reads/*.gz | perl -ne 'chomp; $file=$_; $out=$file; $out=~ s/raw_count_virus_reads\///g;$out=~ s/csv.gz/zipval.csv/g; $cmd = "module load python/3.7 && python bin/calc_zipval.py $file INPUT.count.csv.gz log > zipval_count_done/$out; gzip zipval_count_done/$out"; print $cmd."\n";' > calc_zipval.swarm
swarm -t 4 -g 32 --time 04:00:00 -f calc_zipval.swarm
