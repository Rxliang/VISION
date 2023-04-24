#!/bin/bash
#$ -l ram_free=40G,mem_free=40G,hostname=c*
#$ -e /home/xzhan233/MIND/logs/result.log
#$ -o /home/xzhan233/MIND/logs/result.txt

python3 /home/xzhan233/MIND/main.py --config /home/xzhan233/MIND/config.yaml