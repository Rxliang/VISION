#!/bin/bash
#$ -l gpu=1,ram_free=30G,mem_free=30G,hostname=b1[123456789]|c0*|c1[123456789]
#$ -e /home/xzhan233/MIND/logs/result.log
#$ -o /home/xzhan233/MIND/logs/result.txt
#$ -q g.q

yaml_location=/home/xzhan233/MIND/config.yaml

python3 /home/xzhan233/MIND/main.py --config ${yaml_location}