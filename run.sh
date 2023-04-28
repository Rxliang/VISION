#!/bin/bash
#$ -l gpu=1,ram_free=40G,mem_free=40G,hostname=b18
#$ -e /home/xzhan233/MIND/logs/result.log
#$ -o /home/xzhan233/MIND/logs/result.txt
#$ -q g.q

yaml_location=/home/xzhan233/MIND/config.yaml

python3 /home/xzhan233/MIND/main.py --config ${yaml_location}