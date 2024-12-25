#!/bin/bash
while true
do
~/miniconda3/envs/py310/bin/python compare-bert-base.py hyena 512
~/miniconda3/envs/py310/bin/python compare-bert-base.py dnab 512
~/miniconda3/envs/py310/bin/python compare-bert-base.py gt 512
~/miniconda3/envs/py310/bin/python compare-bert-base.py hyena 1024
~/miniconda3/envs/py310/bin/python compare-bert-base.py dnab 1024
~/miniconda3/envs/py310/bin/python compare-bert-base.py gt 1024
~/miniconda3/envs/py310/bin/python compare-bert-base.py hyena 1536
~/miniconda3/envs/py310/bin/python compare-bert-base.py dnab 1536
~/miniconda3/envs/py310/bin/python compare-bert-base.py gt 1536
~/miniconda3/envs/py310/bin/python compare-bert-base.py hyena 2048
~/miniconda3/envs/py310/bin/python compare-bert-base.py dnab 2048
~/miniconda3/envs/py310/bin/python compare-bert-base.py gt 2048
~/miniconda3/envs/py310/bin/python compare-bert-base.py hyena 4096
~/miniconda3/envs/py310/bin/python compare-bert-base.py dnab 4096
~/miniconda3/envs/py310/bin/python compare-bert-base.py gt 4096
done