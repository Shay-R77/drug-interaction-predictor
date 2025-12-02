import random
import pandas as pd

def create_negative_samples(df, negative_ratio=1.0):

    all_drugs = list(set(df['drug_1']) | set(df['drug_2']))
    
    existing_pairs = set()
    for _, row in df.iterrows():
        existing_pairs.add((row['drug_1'], row['drug_2']))
        existing_pairs.add((row['drug_2'], row['drug_1']))
    
    num_negatives = int(len(df) * negative_ratio)
    negative_samples = []
    
    attempts = 0
    max_attempts = num_negatives * 10 
    
    while len(negative_samples) < num_negatives and attempts < max_attempts:
        attempts += 1
        
        drug_1, drug_2 = random.sample(all_drugs, 2)
        pair = tuple(sorted([drug_1, drug_2]))
        
        if pair not in existing_pairs and (pair[1], pair[0]) not in existing_pairs:
            negative_samples.append({
                'drug_1': pair[0],
                'drug_2': pair[1],
                'target': 0,  
                'drug_pair': pair
            })
            existing_pairs.add(pair) 
        
        if len(negative_samples) % 1000 == 0 and len(negative_samples) > 0:
            print(f"Generated {len(negative_samples)}/{num_negatives} negative samples")
    
    df_positive = df.copy()
    df_positive['target'] = 1 
    
    df_negative = pd.DataFrame(negative_samples)
    
    df_combined = pd.concat([df_positive, df_negative], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal dataset:")
    print(f"Positive samples: {(df_combined['target'] == 1).sum()}")
    print(f"Negative samples: {(df_combined['target'] == 0).sum()}")
    print(f"Total: {len(df_combined)}")
    
    return df_combined

df = pd.read_csv('data/processed/data_binary.csv')
df_with_negatives = create_negative_samples(df, negative_ratio=1.0)
df_with_negatives.to_csv('data/processed/interactions_with_negatives.csv', index=False)