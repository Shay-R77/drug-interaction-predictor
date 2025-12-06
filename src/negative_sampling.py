import pandas as pd
import random
import json
from pathlib import Path

def create_negative_samples(df, drug_info, negative_ratio=1.0):

    drugs_with_smiles = [
        drug for drug, info in drug_info.items()
        if info.get('smiles')
    ]

    print(f'unique drugs in binary data: {len(set(df['drug_1']) | set(df['drug_2']))}')
    print(f' valid SMILES drugs: {len(drugs_with_smiles)}')

    # create set of existing pairs
    existing_pairs = set()
    for _, row in df.iterrows():
        pair = tuple(sorted([row['drug_1'], row['drug_2']]))
        existing_pairs.add(pair)

    # gen the negative samples
    num_negatives = int(len(df) * negative_ratio)
    negative_samples = []

    # randomly select 2 different drugs
    while len(negative_samples) < num_negatives:
        drug_1, drug_2 = random.sample(drugs_with_smiles, 2)
        pair = tuple(sorted([drug_1, drug_2]))

        # check if pair doesnt exist
        if pair not in existing_pairs:
            negative_samples.append({
                'drug_1': pair[0],
                'drug_2': pair[1],
                'interaction': 0
            })
            existing_pairs.add(pair)
        
        if len(negative_samples) % 1000 == 0 and len(negative_samples) > 0:
            print(f'generated {len(negative_samples)}/{num_negatives}')

    # combined dataset creation
    df_positive = df.copy()
    df_positive['interaction'] = 1

    df_negative = pd.DataFrame(negative_samples)

    df_combined = pd.concat([df_positive, df_negative], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f'positive samples: {(df_combined['interaction'] == 1).sum()}')
    print(f'negative samples: {(df_combined['interaction'] == 0).sum()}')
    print(f'total: {len(df_combined)}')

    return df_combined

def main():

    # load drug info
    with open('data/processed/drug_info.json') as file:
        drug_info = json.load(file)

    df = pd.read_csv('data/processed/data_binary.csv')

    # generate negatives
    df_with_negatives = create_negative_samples(df, drug_info, negative_ratio=1.0)

    # save
    df_with_negatives.to_csv('data/processed/interactions_with_negatives.csv', index=False)

if __name__ == '__main__':
    main()
