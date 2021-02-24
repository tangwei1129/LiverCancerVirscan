#!/bin/bash
module load python/3.7
cmd="python bin/concat_tables.py "

files=$(find all_species_ziscore/*.ziscore.spp.csv.gz | perl -ne 'chomp; $in=$_; print $in." ";')
$cmd $files> combined_species_ziscore.csv

files=$(find all_organism_ziscore/*.ziscore.org.csv.gz | perl -ne 'chomp; $in=$_; print $in." ";')
$cmd $files> combined_organism_ziscore.csv
