#! /bin/bash
count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

echo 'start'
for (( c=count; c>=1; c-- ))
do
      python benchmark_models.py -g $c -e $1
done
echo 'end'