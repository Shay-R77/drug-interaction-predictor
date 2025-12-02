import pandas as pd
import requests
import json
import time
from tqdm import tqdm # command line progress bar
from pathlib import Path
import urllib3
from rdkit import Chem # mol object and conversion to smiles

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def compound_to_smiles(pc_compound):

    # extract pubchem ID 
    cid = pc_compound['id']['id']['cid'] 
    
    # build url and download
    sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF" 
    sdf_response = requests.get(sdf_url, verify=False, timeout=10)
    
    # 200 = request succeeded
    if sdf_response.status_code == 200:
        mol = Chem.MolFromMolBlock(sdf_response.text) # parse into molecule object from RDkit
        
        # convert the molecule to SMILES and return it
        if mol:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return smiles
    
        return None

def fetch_drug_info(drug_name):

    try:
        time.sleep(0.3)
        
        drug_encoded = requests.utils.quote(drug_name)
        
        compound_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_encoded}/JSON"
        response = requests.get(compound_url, verify=False, timeout=10)
        
        # if request succeeded, fetch full compound data and convert to SMILES
        if response.status_code == 200:
            data = response.json() # parse
            # get compound and extract CID if one exists
            if 'PC_Compounds' in data and len(data['PC_Compounds']) > 0:
                pc_compound = data['PC_Compounds'][0]
                cid = pc_compound['id']['id']['cid']
                
                # convert
                smiles = compound_to_smiles(pc_compound)
                
                # get the other properties needed is smiles conversion succeeded
                if smiles:
                    props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight/JSON"
                    props_response = requests.get(props_url, verify=False, timeout=10)
                    
                    if props_response.status_code == 200:
                        props_data = props_response.json()
                        props = props_data['PropertyTable']['Properties'][0]
                        
                        return {
                            'pubchem_cid': cid,
                            'smiles': smiles,
                            'molecular_formula': props.get('MolecularFormula'),
                            'molecular_weight': props.get('MolecularWeight'),
                        }
        
        return None
    except:
        return None

def main():
    df = pd.read_csv('data/processed/data_binary.csv')
    
    # put all drug names into a list
    all_drugs = list(pd.concat([df['drug_1'], df['drug_2']]).unique())
    
    # check for backup files. helpful for large sets, because of the rate limit
    backup_path = Path('data/processed/drug_info_backup.json')
    if backup_path.exists():
        print("Found backup")
        with open(backup_path, 'r') as f:
            drug_info = json.load(f)
    else:
        drug_info = {}
    
    failed_drugs = []
    success_count = 0
    
    # iterate through all drugs
    for i, drug_name in enumerate(tqdm(all_drugs, desc="Progress")):
        if drug_name in drug_info and drug_info[drug_name].get('smiles'):
            success_count += 1
            continue
        # save info for the specifc drug in result
        result = fetch_drug_info(drug_name)
        
        # if theres a smiles value in result, save result into drug_info
        if result and result.get('smiles'):
            drug_info[drug_name] = result
            success_count += 1
        # no smiles value, failed drug
        else:
            failed_drugs.append(drug_name)
        
        # write to back up every 50 drugs
        if (i + 1) % 50 == 0:
            with open('data/processed/drug_info_backup.json', 'w') as f:
                json.dump(drug_info, f, indent=2)
    
    # write to final json file with all drug info for all drugs processed
    with open('data/processed/drug_info.json', 'w') as f:
        json.dump(drug_info, f, indent=2)
        print('saved drug info to "drug_info.json"')
    
    # write failed drugs to a file
    if failed_drugs:
        with open('data/processed/failed_drugs.txt', 'w') as f:
            f.write('\n'.join(failed_drugs))
            print('saved failed drugs to "failed_drugs.txt"')

if __name__ == "__main__":
    main()