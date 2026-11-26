from src.dataloader.data_wrapper import (
    ClinVarDataWrapper,
)

NUM_RECORDS = 1000
ALL_RECORDS = False
SEQ_LEN = 20



# Load ClinVar data
data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)
