#!/bin/bash
mkdir hits
mkdir hits/REP1
mkdir hits/REP2
find zipval_count_done/*.zipval.csv.gz | perl -ne 'chomp; $in=$_; if ($in =~ m/(.+)_IDX(\d+).zipval.csv.gz/ && $2 < 97) {system("mv $in hits/REP1");} else {system("mv $in hits/REP2");}'

find hits/REP1/* | perl -ne 'chomp; $in=$_; $rep2=$in; $rep2=~s/REP1/REP2/g; if ($rep2 =~ m/hits\/REP2\/(.+)_IDX(\d+).zipval.csv.gz/ && $2+96 < 100) {$id=$1."_IDX".$2; $index=$2+96; $indexString="IDX0".$index; $rep2 =~ s/IDX(\d+)/$indexString/;} else {$id=$1."_IDX".$2; $index=$2+96; $indexString="IDX".$index; $rep2 =~ s/IDX(\d+)/$indexString/;} $cmd = "module load python/3.7 && mkdir hit_plots\_$id; python bin/call_hits.py $in $rep2 hit_plots\_$id 2.3 >$id\.zihit.csv 2>log\_$id.err; gzip $id\.zihit.csv"; print $cmd."\n"' > call_hits.swarm
swarm -t 4 -g 32 --time 04:00:00 -f call_hits.swarm
