#!/bin/bash
mkdir all_species_ziscore
find *.zihit.csv.gz | perl -ne 'chomp; $in=$_; if ($in =~ m/(.+)_IDX(\d+).zihit.csv.gz/ && $2 < 91) {$cmd  = "module load python/3.7 && python bin/calc_scores.py $in reference/VIR2.csv.gz LMW_NHITS.BEADS.csv.gz LMW_NHITS.SAMPLES.csv.gz Species 7 > all_species_ziscore/$1\_IDX$2.ziscore.spp.csv; gzip all_species_ziscore/$1\_IDX$2.ziscore.spp.csv"; print $cmd."\n";}' > calc_score_species.swarm
swarm -t 4 -g 32 --time 04:00:00 -f calc_score_species.swarm
