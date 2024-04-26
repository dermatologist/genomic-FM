import os
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import glob
from src.sequence_extractor import RandomSequenceExtractor, FastaStringExtractor
from src.blast_search import run_blast_query
from src.datasets.ncbi_reference_genome.get_accession import search_species
import random

species_to_epd = {
    "Apis mellifera": "A_mellifera",
    "Arabidopsis thaliana": "A_thaliana",
    "Caenorhabditis elegans": "C_elegans",
    "Canis familiaris": "C_familiaris",
    "Drosophila melanogaster": "D_melanogaster",
    "Danio rerio": "D_rerio",
    "Gallus gallus": "G_gallus",
    "Homo sapiens": "H_sapiens",
    "Homo sapiens (non-coding)": "H_sapiens_nc",
    "Macaca mulatta": "M_mulatta",
    "Mus musculus": "M_musculus",
    "Mus musculus (non-coding)": "M_musculus_nc",
    "Plasmodium falciparum": "P_falciparum",
    "Rattus Norvegicus": "R_norvegicus",
    "Saccharomyces cerevisiae": "S_cerevisiae",
    "Schizosaccharomyces pombe": "S_pombe",
    "Zea mays": "Z_mays"
}

def download_epd(out_dir='./root/data/epd'):
    """
    Downloads all .dat files for a all species.

    Parameters:
    - out_dir (str): The base directory where the species directory will be created and files downloaded.
    """
    for eukaryote_species, species in species_to_epd.items():
        print(f"Downloading {eukaryote_species}")
        base_url = f"http://epd.expasy.org/ftp/epdnew/{species}/current"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        local_directory = os.path.join(out_dir, species)

        response = requests.get(base_url)
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Ensure the local directory exists
            os.makedirs(local_directory, exist_ok=True)

            # Find all the .dat file links
            for link in soup.find_all('a'):
                href = link.get('href')
                if href.endswith('.dat'):
                    dat_url = os.path.join(base_url, href)
                    print(f"Downloading {href}...")
                    dat_response = requests.get(dat_url)
                    if dat_response.status_code == 200:
                        file_path = os.path.join(local_directory, href)
                        with open(file_path, 'wb') as f:
                            f.write(dat_response.content)
                        print(f"Downloaded {href} to {local_directory}")
                    else:
                        print(f"Failed to download {href}")
        else:
            print(f"Failed to access {base_url}") 

def parse_epd(file_path):
    """
    Parses an EPDnew .dat file and extracts key information about promoters,
    changing cross-references into individual keys with clean values.

    Args:
        file_path (str): Path to the .dat file.

    Returns:
        list of dicts: A list containing dictionaries with the extracted information.
    """
    promoters = []
    with open(file_path, 'r') as file:
        entry = {}
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if line.startswith('ID'):
                if entry:
                    promoters.append(entry)
                    entry = {}
                entry['ID'] = line.split()[1]
            elif line.startswith('GN'):
                entry['Gene Name'] = line.split('=')[1].split(';')[0].strip()
            elif line.startswith('DE'):
                entry['Description'] = line[5:]
            elif line.startswith('OS'):
                entry['Species'] = line[5:].rstrip('.')
            elif line.startswith('ME'):
                entry['Methods'] = line[5:].rstrip('.')
            elif line.startswith('DT'):
                entry['Date'] = line[5:]
            elif line.startswith('DR'):
                dr_parts = line[5:].split(';')
                key = dr_parts[0].strip()
                values = [part.strip() for part in dr_parts[1:] if part.strip()]
                # Handling multiple values differently if needed
                if len(values) == 1:
                    entry[key] = values[0]
                else:
                    entry[key] = values
            elif line.startswith('SE'):
                entry['Sequence'] = line[5:]
            elif line.startswith('FP'):
                entry['Functional Position'] = ' '.join(line[5:].split())

        if entry:  # Include the last promoter if file doesn't end with //
            promoters.append(entry)

    return promoters

def get_eukaryote_promoters(sequence_length=1024, limit=None):
    combined_tuples = []

    for eukaryote_species, species_id in species_to_epd.items():
        print(f"Processing {eukaryote_species}")
        species_dir = os.path.join('./root/data/epd', species_id)
        file_path = next(iter(glob.glob(f"{species_dir}/*.dat")), None)
        fasta_path, gtf_path = (glob.glob(os.path.join("./root/data", search_species(eukaryote_species)[0], f"ncbi_dataset/data/GCF*/{p}"))[0] for p in ["GCF*fna", "genomic.gtf"])

        if not file_path or not fasta_path or not gtf_path:
            print(f"Required files missing for {eukaryote_species}. Skipping...")
            continue

        fasta_extractor = FastaStringExtractor(fasta_path)
        eukaryote_promoters = parse_epd(file_path)
        promoter_data = []

        for index, promoter in enumerate(tqdm(eukaryote_promoters, desc="Extracting promoters")):
            if limit and index >= limit:
                break

            species, gene_name, sequence = promoter['Species'], promoter['Gene Name'], promoter['Sequence'].upper()

            if sequence: 
                interval = run_blast_query(sequence, fasta_path)
                if interval is not None:
                    interval = interval.resize(sequence_length)
                    extended_sequence = fasta_extractor.extract(interval)
                    promoter_data.append((species, gene_name, extended_sequence, interval))

        if promoter_data:
            intervals = [data[-1] for data in promoter_data] 
            random_sequences = RandomSequenceExtractor(fasta_path, gtf_path).extract_random_sequence(
                length_range=(sequence_length, sequence_length),
                num_sequences=len(promoter_data),
                known_regions=intervals
            )
            combined_tuples.extend([(data[0], data[1], data[2], 1) for data in promoter_data])
            combined_tuples.extend([(data[0], data[1], seq, 0) for data, seq in zip(promoter_data, random_sequences)])
        random.shuffle(combined_tuples)

    return combined_tuples
