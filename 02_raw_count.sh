#!/bin/bash
if [ $# -ne 2 ]; then
    echo $0: usage: raw_count $input_bam_file $out_put_count_csv
    exit 1
fi
#bam_file=$1
#out_file=$2
#module load samtools && samtools idxstats $bam_file | cut -f 1,3 | sed -e '/^\*\t/d' -e '1 i id\tSAMPLE_ID' | tr "\\t" "," >$out_file
samtools idxstats $1 | cut -f 1,3 | sed -e '/^\*\t/d' -e '1 i id\tSAMPLE_ID' | tr "\\t" "," >$2
