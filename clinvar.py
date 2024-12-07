"""
Real ClinVar" essentially refers to the most up-to-date, reliable information within the ClinVar database,
often implying a focus on clinically validated and well-established variant interpretations,
while "ClinVar" encompasses the entire database, which may include submissions with varying levels of evidence
and potential for conflicting interpretations depending on the source.


GV Record: Following this construction process, the minimum unit of GV-Rep dataset is a record,
which is an (x, y) pair. Here, x = (ref, alt, annotation), and y is the corresponding label indicating
the class of GV or a real value quantifying the effects of the GV.

"""

from src.dataloader.data_wrapper import (
    RealClinVar
)

NUM_RECORDS = 10
ALL_RECORDS = False
SEQ_LEN = 512

# Load RealClinVar data

data_loader = RealClinVar(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)

for row in data:
    for section in row:
        #if section is not an array
        if not isinstance(section, list):
            label = section
        else:
            reference = section[0]
            alternate = section[1]
            annotation = section[2]
    print(f"Label: {label}, Reference: {reference}, Alternate: {alternate}, Annotation: {annotation}")

