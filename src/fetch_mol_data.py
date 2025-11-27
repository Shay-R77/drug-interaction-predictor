import pandas as pd
import pubchempy as pcp
import json
import time
from tqdm import tqdm
from pathlib import Path
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

def fetch_drug_info(drug_name, retry_count=3):

    for attempt in range(retry_count):
        try:
            time.sleep(0.2)
            
            compounds = pcp.get_compounds(drug_name, 'name')
            
            if compounds:
                compound = compounds[0]
                
                return {
                    'pubchem_cid': compound.cid,
                    'smiles': compound.canonical_smiles,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight,
                }
            else:
                return None
                
        except Exception as e:
            if attempt == retry_count - 1:
                print(f"\nError with {drug_name} after {retry_count} attempts: {str(e)}")
                return None
            time.sleep(1) 
    
    return None

def main():

    data = pd.read_csv('data/processed/data_binary.csv')

    all_drugs = pd.concat([data['drug_1'], data['drug_2']]).unique()

    print('fetching drug structures from PubChem')
    print('may take several minutes')

    drug_info = {}
    failed_drugs = []

    for i, drug_name in enumerate(tqdm(all_drugs)):
        
        if drug_name in drug_info:
            continue

        result = fetch_drug_info(drug_name)

        if result:
            drug_info[drug_name] = result
        else:
            failed_drugs.append(drug_name)

    with open('data/processed/drug_info.json', 'w') as f:
        json.dump(drug_info, f, indent=2)

if __name__ == "__main__":
    main()