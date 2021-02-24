#!/bin/bash
find fastq/*.fastq.gz | perl -ne 'chomp; $in=$_; $out=$in; $out=~s/fastq\///; $out=~ s/\_S(\d+)_R1_001.fastq.gz//g; $bowtie_cmd = "module load bowtie/1.2.3 && bowtie -n 3 -l 30 -e 1000 --tryhard --nomaqround --norc --best --sam --quiet reference/ref_vir2 -q $in > $out\.sam 2>run_bowtie\_$out\.err "; $bam_cmd="module load samtools && samtools view -bS $out\.sam >$out\.bam 2>run_samtool_view.err && samtools sort $out\.bam -o $out\.sorted.bam >run\_$out\.sorted.log 2>run_samtools_sort.err && samtools index $out\.sorted.bam 2>run_samtools_index.err "; $bam_stats = "module load bamtools && bamtools stats -in $out\.sorted.bam 1>run_bam_stats\_$out.log 2>run_bam_stats\_$out.err "; print "$bowtie_cmd && $bam_cmd && $bam_stats && rm $out\.sam\n"' > bowtie.swarm
swarm -t 8 -g 32 --time 08:00:00 -f bowtie.swarm




module load bowtie/1.2.2
module load samtools/1.2


for i in *.fastq.gz; do 
gzip -dc $i |  bowtie -n 0 -5 5 -3 5 -l 30 -e 1000 --tryhard --nomaqround --norc --best --sam --quiet /data/tangw3/viruscan/code/vir.92/vir92 - | samtools view -u - |   samtools sort -T BAM_OUTPUT_NAME.bam - -o $i.bam; samtools index $i.bam; samtools idxstats ${i}.bam | cut -f 1,3 | sed -e '/^\*\t/d' -e '1 i id\t'${i}'' | tr "\\t" "," > ${i}_COUNT.csv;done &>> alignment.log

# gzip all the _COUNT.csv files ##
gzip *_COUNT.csv

