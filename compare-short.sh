#!/bin/bash
python compare-save-datasets.py hyena 512
python compare-save-datasets.py dnab 512
python compare-save-datasets.py gt 512
python compare-save-datasets.py hyena 1024
python compare-save-datasets.py dnab 1024
python compare-save-datasets.py gt 1024
# python compare-save-datasets.py hyena 1536
# python compare-save-datasets.py dnab 1536
# python compare-save-datasets.py gt 1536
python compare-save-datasets.py hyena 2048
python compare-save-datasets.py dnab 2048
python compare-save-datasets.py gt 2048
python compare-save-datasets.py hyena 4096
python compare-save-datasets.py dnab 4096
python compare-save-datasets.py gt 4096
