"""
Disease-Drug Association Rule Mining - Data Processing Script
Run this first to generate the association rules model
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DISEASE-DRUG ASSOCIATION RULE MINING")
print("=" * 60)

# =============================================================================
# 1. LOAD AND CLEAN DATA
# =============================================================================
print("\n📁 Loading data...")
df = pd.read_csv('disease_drug.csv')

print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Get column names
disease_col = df.columns[0]
drug_col = df.columns[1]

print(f"Disease column: '{disease_col}'")
print(f"Drug column: '{drug_col}'")

# Clean data
df_clean = df.copy()
df_clean = df_clean.dropna(subset=[disease_col, drug_col])
df_clean = df_clean[(df_clean[disease_col] != '') & (df_clean[drug_col] != '')]
df_clean[disease_col] = df_clean[disease_col].astype(str).str.strip().str.title()
df_clean[drug_col] = df_clean[drug_col].astype(str).str.strip().str.title()

print(f"After cleaning: {len(df_clean)} rows")
print(f"Unique diseases: {df_clean[disease_col].nunique()}")
print(f"Unique drugs: {df_clean[drug_col].nunique()}")

# =============================================================================
# 2. DATA SAMPLING AND FILTERING
# =============================================================================
print("\n⚡ Optimizing dataset...")

# Sample if too large
max_sample_size = 50000
if len(df_clean) > max_sample_size:
    df_sample = df_clean.sample(n=max_sample_size, random_state=42)
else:
    df_sample = df_clean.copy()

# Filter to top items
top_n_diseases = 50
top_n_drugs = 100

popular_diseases = df_sample[disease_col].value_counts().head(top_n_diseases).index
popular_drugs = df_sample[drug_col].value_counts().head(top_n_drugs).index

df_filtered = df_sample[
    (df_sample[disease_col].isin(popular_diseases)) & 
    (df_sample[drug_col].isin(popular_drugs))
]

print(f"Filtered dataset: {len(df_filtered)} rows")
print(f"Using top {top_n_diseases} diseases and {top_n_drugs} drugs")

# =============================================================================
# 3. CREATE TRANSACTION MATRIX
# =============================================================================
print("\n🔄 Creating transaction matrix...")

transactions = []
for _, row in df_filtered.iterrows():
    transaction = [f"Disease_{row[disease_col]}", f"Drug_{row[drug_col]}"]
    transactions.append(transaction)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Transaction matrix shape: {df_encoded.shape}")

# =============================================================================
# 4. MINE FREQUENT ITEMSETS
# =============================================================================
print("\n⛏️ Mining frequent itemsets...")

min_support = 0.01  # Adjust this based on your data
frequent_itemsets = fpgrowth(
    df_encoded, 
    min_support=min_support, 
    use_colnames=True,
    max_len=3
)

print(f"Found {len(frequent_itemsets)} frequent itemsets")

# =============================================================================
# 5. GENERATE ASSOCIATION RULES
# =============================================================================
print("\n📋 Generating association rules...")

if len(frequent_itemsets) > 0:
    min_confidence = 0.2
    
    rules = association_rules(
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets)
    )
    
    # Filter for drug recommendation rules
    drug_rules = rules[
        rules['antecedents'].astype(str).str.contains('Disease_') &
        rules['consequents'].astype(str).str.contains('Drug_')
    ].copy()
    
    # Clean up rule format
    drug_rules['disease'] = drug_rules['antecedents'].apply(
        lambda x: ', '.join([item.replace('Disease_', '') for item in x])
    )
    drug_rules['drug'] = drug_rules['consequents'].apply(
        lambda x: ', '.join([item.replace('Drug_', '') for item in x])
    )
    
    # Sort by confidence and support
    drug_rules = drug_rules.sort_values(['confidence', 'support'], ascending=False)
    
    print(f"Generated {len(drug_rules)} drug recommendation rules")
    
    # =============================================================================
    # 6. SAVE MODEL AND DATA
    # =============================================================================
    print("\n💾 Saving model artifacts...")
    
    # Save rules
    drug_rules.to_csv('association_rules.csv', index=False)
    
    # Save disease list
    disease_list = sorted(list(set(drug_rules['disease'].values)))
    
    # Create model dictionary
    model_data = {
        'rules': drug_rules,
        'diseases': disease_list,
        'stats': {
            'total_rules': len(drug_rules),
            'avg_confidence': drug_rules['confidence'].mean(),
            'avg_support': drug_rules['support'].mean(),
            'unique_diseases': len(disease_list)
        }
    }
    
    # Save as pickle
    with open('drug_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model saved successfully!")
    print("\nModel Statistics:")
    print(f"  - Total rules: {model_data['stats']['total_rules']}")
    print(f"  - Unique diseases: {model_data['stats']['unique_diseases']}")
    print(f"  - Average confidence: {model_data['stats']['avg_confidence']:.3f}")
    print(f"  - Average support: {model_data['stats']['avg_support']:.3f}")
    
    print("\n📄 Files created:")
    print("  1. association_rules.csv")
    print("  2. drug_recommendation_model.pkl")
    print("\n✨ Ready for Streamlit deployment!")
    
else:
    print("❌ No frequent itemsets found. Try reducing min_support.")

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
