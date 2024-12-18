from genomic_benchmarks.data_check import list_datasets, info
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn
print(list_datasets())
print(info("human_enhancers_cohn", version=0))
dset = HumanEnhancersCohn(split='train', version=0)

print(dset[0])