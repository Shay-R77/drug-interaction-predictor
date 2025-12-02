from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

def convert_to_features(smiles):
    '''
    smiles to molecular features
    '''

    # no smiles value
    if pd.isna(smiles) or smiles is None:
        return None, None
    
    # convert from smiles to Mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # descriptors
    features = {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_h_donors': Descriptors.NumHDonors(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'tpsa': Descriptors.TPSA(mol) 
    }

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp = np.array(fingerprint)

    return features, fp

def interaction_features(drug1_features, drug1_fp, drug2_features, drug2_fp):
    '''
    combine features from both drugs to represent the interaction
    '''

    # concatenate descriptors
    combined_features = {}
    for key in drug1_features.keys():
        combined_features[f'drug1_{key}'] = drug1_features[key]
        combined_features[f'drug2_{key}'] = drug2_features[key]
        # difference/ratio
        combined_features[f'diff_{key}'] = abs(drug1_features[key] - drug2_features[key])

        if drug2_features[key] != 0:
            combined_features[f'ratio_{key}'] = (drug1_features[key] / drug2_features[key])

    combined_fp = np.concatenate([drug1_fp, drug2_fp])

    return combined_features, combined_fp

def main():
    with open('data/processed/drug_info.json', 'r') as file:
        drug_info = json.load(file)

    data = pd.read_csv('data/processed/data_binary.csv')

    feature_rows = []
    fingerprint_rows = []
    skipped_pairs = 0

    print(f'{len(data)} pairs')

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        drug1_info = drug_info.get(row['drug_1'])
        drug2_info = drug_info.get(row['drug_2'])

        # skip if either drug is missing info
        if not drug1_info or not drug2_info:
            skipped_pairs += 1
            continue

        drug1_smiles = drug1_info.get('smiles')
        drug2_smiles = drug2_info.get('smiles')

        if drug1_smiles and drug2_smiles:
            drug1_feature, drug1_fingerprint = convert_to_features(drug1_smiles)
            drug2_feature, drug2_fingerprint = convert_to_features(drug2_smiles)

            if drug1_feature is not None and drug2_feature is not None:
                combined_feature, combined_fingerprint = interaction_features(drug1_feature, drug1_fingerprint, drug2_feature, drug2_fingerprint)

                # metadata
                combined_feature['drug_1'] = row['drug_1']
                combined_feature['drug_2'] = row['drug_2']
                combined_feature['target'] = row['interaction']

                feature_rows.append(combined_feature)
                fingerprint_rows.append(combined_fingerprint)
            else:
                skipped_pairs += 1
        else:
            skipped_pairs += 1


    print(f"\nSuccessfully processed: {len(feature_rows)} pairs")
    print(f"Skipped: {skipped_pairs} pairs (missing data)")

    # Create final datasets
    feature_df = pd.DataFrame(feature_rows)
    fingerprint_array = np.array(fingerprint_rows)

    # Save processed data
    feature_df.to_csv('data/processed/features.csv', index=False)
    np.save('data/processed/fingerprints.npy', fingerprint_array)

    print(f"\nFinal dataset shape: {feature_df.shape}")
    print(f"Fingerprint array shape: {fingerprint_array.shape}")
    print(f"\nFeatures created: {[col for col in feature_df.columns if col not in ['drug_1', 'drug_2', 'target']]}")
    print(f"\nTarget distribution:\n{feature_df['target'].value_counts()}")

if __name__ == "__main__":
    main()