"""
Vérification empirique de la couverture textuelle des avis clients
sur la population de travail (commandes livrées, 96 470 cas).
"""
import pandas as pd
import numpy as np

# Charger les données brutes
orders = pd.read_csv('data/raw/olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date'])
reviews = pd.read_csv('data/raw/olist_order_reviews_dataset.csv', parse_dates=['review_creation_date'])

print("=" * 80)
print("EXPLORATION DES DONNÉES D'AVIS - POPULATION DE TRAVAIL")
print("=" * 80)

# Filtrer sur la population de travail
orders_delivered = orders[
    (orders['order_status'] == 'delivered') &
    (orders['order_delivered_customer_date'].notna())
]

print(f"\nPopulation de travail : {len(orders_delivered)} commandes livrées")

# Fusion avec les avis
merged = orders_delivered[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']].merge(
    reviews[['order_id', 'review_score', 'review_comment_title', 'review_comment_message', 'review_creation_date']],
    on='order_id',
    how='inner'
)

print(f"Commandes livrées avec avis : {len(merged)}")
print(f"Couverture : {100*len(merged)/len(orders_delivered):.1f}%")

# 1. COUVERTURE TEXTUELLE
print(f"\n{'='*80}")
print("1. COUVERTURE TEXTUELLE")
print(f"{'='*80}")

has_title = merged['review_comment_title'].notna() & (merged['review_comment_title'].str.strip() != '')
has_message = merged['review_comment_message'].notna() & (merged['review_comment_message'].str.strip() != '')

print(f"   - Avec titre rempli : {has_title.sum():,} ({100*has_title.sum()/len(merged):.1f}%)")
print(f"   - Avec message rempli : {has_message.sum():,} ({100*has_message.sum()/len(merged):.1f}%)")
print(f"   - Avec titre OU message : {(has_title | has_message).sum():,} ({100*(has_title | has_message).sum()/len(merged):.1f}%)")
print(f"   - Avec titre ET message : {(has_title & has_message).sum():,} ({100*(has_title & has_message).sum()/len(merged):.1f}%)")

# 2. STABILITÉ TEMPORELLE
print(f"\n{'='*80}")
print("2. STABILITÉ TEMPORELLE (par mois de commande)")
print(f"{'='*80}")

merged['year_month'] = merged['order_purchase_timestamp'].dt.to_period('M')
coverage_by_month = merged.groupby('year_month').apply(
    lambda x: pd.Series({
        'total': len(x),
        'avec_texte': ((x['review_comment_title'].notna() & (x['review_comment_title'].str.strip() != '')) | 
                       (x['review_comment_message'].notna() & (x['review_comment_message'].str.strip() != ''))).sum(),
    })
)
coverage_by_month['pct'] = 100 * coverage_by_month['avec_texte'] / coverage_by_month['total']

for period, row in coverage_by_month.iterrows():
    print(f"   {period}: {int(row['total']):4d} avis, {int(row['avec_texte']):4d} avec texte ({row['pct']:5.1f}%)")

# 3. LONGUEUR MOYENNE
print(f"\n{'='*80}")
print("3. LONGUEUR MOYENNE DES TEXTES")
print(f"{'='*80}")

messages_filled = merged[has_message]['review_comment_message'].str.split().str.len()
titles_filled = merged[has_title]['review_comment_title'].str.split().str.len()

print(f"   Messages remplis : {messages_filled.mean():.1f} mots (min={messages_filled.min()}, max={messages_filled.max()})")
print(f"   Titres remplis : {titles_filled.mean():.1f} mots (min={titles_filled.min()}, max={titles_filled.max()})")

# 4. DISTRIBUTION PAR NOTE
print(f"\n{'='*80}")
print("4. DISTRIBUTION PAR NOTE")
print(f"{'='*80}")

for score in sorted(merged['review_score'].unique()):
    score_data = merged[merged['review_score'] == score]
    count = len(score_data)
    with_message = (score_data['review_comment_message'].notna() & 
                    (score_data['review_comment_message'].str.strip() != '')).sum()
    print(f"   Note {int(score)}: {count:,} avis ({with_message:,} avec message, {100*with_message/count:.1f}%)")

# 5. AVIS NÉGATIFS (note <= 2)
print(f"\n{'='*80}")
print("5. AVIS NÉGATIFS (note ≤ 2)")
print(f"{'='*80}")

negative_reviews = merged[merged['review_score'] <= 2].copy()
negative_reviews = negative_reviews[
    (negative_reviews['review_comment_message'].notna()) & 
    (negative_reviews['review_comment_message'].str.strip() != '')
]

print(f"Avis négatifs avec message : {len(negative_reviews)}")
print(f"\nÉchantillon de 15-20 avis négatifs réels :\n")

# Limiter à 20
sample_negative = negative_reviews.sample(min(20, len(negative_reviews)), random_state=42)

for idx, (i, row) in enumerate(sample_negative.iterrows(), 1):
    title = row['review_comment_title'] if pd.notna(row['review_comment_title']) else "[SANS TITRE]"
    message = row['review_comment_message']
    score = int(row['review_score'])
    
    print(f"\n--- Avis #{idx} (note={score}) ---")
    print(f"Titre: {title}")
    print(f"Message: {message}")
