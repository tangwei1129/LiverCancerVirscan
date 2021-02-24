#!/bin/bash
module load samtools
mkdir raw_count_virus_reads
#find *sorted.bam | perl -ne 'chomp; $bam =$_; $out=$bam; $out=~ s/\.sorted\.bam/\.csv/g; $out=~ s/\_IDX\d*//g; $cmd = "./02_raw_count.sh $bam raw_count_virus_reads/$out && cd raw_count_virus_reads && gzip $out "; system ( $cmd ) '
find *sorted.bam | perl -ne 'chomp; $bam =$_; $out=$bam; $out=~ s/\.sorted\.bam/\.csv/g; $cmd = "./02_raw_count.sh $bam raw_count_virus_reads/$out && cd raw_count_virus_reads && gzip $out "; system ( $cmd ) '
