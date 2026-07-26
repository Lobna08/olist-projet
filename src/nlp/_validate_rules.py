"""
Définition et validation des règles de classification pour les motifs d'insatisfaction.
Test sur le corpus complet des avis négatifs (note ≤ 2, 9 283 messages).
"""
import pandas as pd
import re
from collections import defaultdict

# Charger les données
orders = pd.read_csv('data/raw/olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date'])
reviews = pd.read_csv('data/raw/olist_order_reviews_dataset.csv', parse_dates=['review_creation_date'])

# Filtrer population de travail
orders_delivered = orders[
    (orders['order_status'] == 'delivered') &
    (orders['order_delivered_customer_date'].notna())
]

merged = orders_delivered[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']].merge(
    reviews[['order_id', 'review_score', 'review_comment_title', 'review_comment_message', 'review_creation_date']],
    on='order_id',
    how='inner'
)

# Avis négatifs (note ≤ 2)
negative_reviews = merged[merged['review_score'] <= 2].copy()
negative_reviews = negative_reviews[
    (negative_reviews['review_comment_message'].notna()) & 
    (negative_reviews['review_comment_message'].str.strip() != '')
]

print("=" * 80)
print(f"CORPUS : {len(negative_reviews)} avis négatifs avec message (note ≤ 2)")
print("=" * 80)

# Définition des règles de mots-clés
# Convertir tout en minuscules et nettoyer
def normalize_text(text):
    if pd.isna(text):
        return ""
    return text.lower().strip()

# Règles pour chaque catégorie
rules = {
    'retard_livraison': [
        r'\batraso',
        r'\bnão.*recebi.*prazo',
        r'\bvenceu.*prazo',
        r'prazo.*venc',
        r'\bfora.*prazo',
        r'\bacima.*prazo',
        r'\bprazo.*não.*cum',
        r'\bdemora',
        r'\blento',
        r'\bperdeu',
        r'\bcorreios.*perderam',
    ],
    'livraison_incomplete': [
        r'\bfaltou',
        r'\bmissing',
        r'\breceber.*menos',
        r'\brecebemos.*menos',
        r'\brecebida?.*incompleta',
        r'\bfalta.*item',
        r'\bfaltam.*item',
        r'\bapenass?.*uma?',
        r'\bapenas.*dois',
        r'\bapenas.*duas',
        r'\bsó.*recebi',
        r'\bsomente.*recebi',
        r'\brecebido?.*só',
        r'\brecebido?.*apenas',
        r'\bchegaram.*\d+',
        r'\bfalta.*uma',
        r'\bfalta.*um',
        r'\bincompleto',
        r'\bitem.*falta',
        r'\bquantidade.*err',
        r'\bquantidade.*meno',
        r'\bchegou.*(apenas|só)',
        r'\brecebi.*(apenas|só|menos)',
        r'\bpedi\s+(um|uma|dois|duas|tr[êe]s|quatro|cinco|seis|sete|oito|nove|dez|\d+)\b',
        r'\bs[óo]\s+veio\b',
        r'\bn[ãa]o\s+receb\w*.*complet',
        r'\bpedi\b.*\breceb(i|emos)\b',
    ],
    'produit_incorrect': [
        r'\bdiferente do anúncio',
        r'\bdiferente do anuncio',
        r'\bproduto diferente',
        r'\bmodelo errado',
        r'\bcor errada',
        r'\bcor diferente',
        r'\bfragrancia diferente',
        r'\bperfume diferente',
        r'\brecebi outro',
        r'\bmandaram uma fragrancia diferente',
        r'\brecebi um produto que não tem nada haver',
        r'\bfoi enviado produto diferente',
        r'\bnão foi o que pedi',
        r'\bnao foi o que pedi',
        r'\brecebi.*(outro|outra|diferente)',
        r'\bproduto.*(errado|diferente)',
    ],
    'produit_endommage': [
        r'\bquebr',
        r'\bromp',
        r'\bfragi',
        r'\bdano',
        r'\bdanado',
        r'\bdefeit',
        r'\bmanutenção',
        r'\bdefeituoso',
        r'\bem.*ruim',
        r'\bproduto.*ruim',
        r'\bqualidade.*péss',
        r'\bpéssima',
        r'\bmal.*embal',
        r'\bestoura',
        r'\bfuro',
        r'\b(estou|tá).*solta',
        r'\bsuja',
        r'\bsujo',
    ],
    'aspect_mismatch': [
        r'\bcor.*escur',
        r'\bcor.*diferent',
        r'\bimagem.*diferent',
        r'\bnão.*igual',
        r'\btamanho.*pequen',
        r'\btamanho.*grand',
        r'\btamanho.*diferen',
        r'\bmaterial.*diferent',
        r'\btextura.*diferent',
        r'\bfoto.*fals',
        r'\bdurabilidade',
    ],
}

# Compter les matches par catégorie
category_independent_counts = defaultdict(int)
category_priority_counts = defaultdict(int)
uncategorized = 0
text_matches = defaultdict(list)

for idx, row in negative_reviews.iterrows():
    text = normalize_text(row['review_comment_message'])
    title = normalize_text(row['review_comment_title'])
    full_text = title + " " + text
    
    # Parcourir les catégories
    matched_categories = []
    
    for category, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, full_text):
                matched_categories.append(category)
                text_matches[category].append({
                    'text': text[:100],
                    'score': row['review_score']
                })
                break  # Arrêter à la première match pour cette catégorie
    
    # Compter indépendamment toutes les catégories qui matchent
    if matched_categories:
        for cat in matched_categories:
            category_independent_counts[cat] += 1

        # Assigner la meilleure catégorie trouvée selon la priorité
        # Priorité : retard_livraison > livraison_incomplete > produit_incorrect > produit_endommage > aspect_mismatch
        priority = ['retard_livraison', 'livraison_incomplete', 'produit_incorrect', 'produit_endommage', 'aspect_mismatch']
        for cat in priority:
            if cat in matched_categories:
                category_priority_counts[cat] += 1
                break
    else:
        uncategorized += 1

print("\n" + "=" * 80)
print("RÉSULTATS : Comptage indépendant vs après priorité")
print("=" * 80)
print(f"{'catégorie':25s} {'indépendant':>12s} {'après priorité':>15s} {'%indép.':>8s} {'%priorité':>10s}")
for cat in ['retard_livraison', 'livraison_incomplete', 'produit_incorrect', 'produit_endommage', 'aspect_mismatch']:
    indep = category_independent_counts[cat]
    prio = category_priority_counts[cat]
    pct_indep = 100 * indep / len(negative_reviews)
    pct_prio = 100 * prio / len(negative_reviews)
    print(f"{cat:25s} {indep:12d} {prio:15d} {pct_indep:8.1f}% {pct_prio:10.1f}%")

print(f"{'autre / non-catégorisé':25s} {uncategorized:12d} {0:15d} {100*uncategorized/len(negative_reviews):8.1f}% {'0.0%':>10s}")

print("\n" + "=" * 80)
print("ANALYSE : Fréquence d'Aspect ≠ Description (indépendant et après priorité)")
print("=" * 80)
indep_aspect = category_independent_counts['aspect_mismatch']
prio_aspect = category_priority_counts['aspect_mismatch']
pct_indep_aspect = 100 * indep_aspect / len(negative_reviews)
pct_prio_aspect = 100 * prio_aspect / len(negative_reviews)
print(f"Aspect ≠ Description - indépendant: {indep_aspect} avis ({pct_indep_aspect:.1f}%)")
print(f"Aspect ≠ Description - après priorité: {prio_aspect} avis ({pct_prio_aspect:.1f}%)")
if pct_indep_aspect < 3:
    print(f"⚠️  MARGINAL (< 3%) → À absorber dans 'Autre' ou fusionner avec une autre catégorie")
else:
    print(f"✓ Acceptable (> 3%) → À conserver comme catégorie distincte")

# Échantillon de non-catégorisés pour inspection manuelle
uncategorized_samples = []
for idx, row in negative_reviews.iterrows():
    text = normalize_text(row['review_comment_message'])
    title = normalize_text(row['review_comment_title'])
    full_text = title + " " + text
    matched_categories = []
    for category, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, full_text):
                matched_categories.append(category)
                break
    if not matched_categories:
        uncategorized_samples.append({
            'review_comment_title': row['review_comment_title'],
            'review_comment_message': row['review_comment_message'],
            'review_score': int(row['review_score'])
        })

print("\n" + "=" * 80)
print("ÉCHANTILLON MANUEL : 25 avis non catégorisés")
print("=" * 80)

for sample in pd.DataFrame(uncategorized_samples).sample(n=25, random_state=42).itertuples(index=False):
    title = sample.review_comment_title if pd.notna(sample.review_comment_title) else "[SANS TITRE]"
    print(f"\n--- Note={sample.review_score} ---")
    print(f"Titre : {title}")
    print(f"Message : {sample.review_comment_message}")

print("\n" + "=" * 80)
print("ÉCHANTILLON ALÉATOIRE : 20 avis non catégorisés")
print("random_state=7")
print("=" * 80)

for sample in pd.DataFrame(uncategorized_samples).sample(n=20, random_state=7).itertuples(index=False):
    title = sample.review_comment_title if pd.notna(sample.review_comment_title) else "[SANS TITRE]"
    print(f"\n--- Note={sample.review_score} ---")
    print(f"Titre : {title}")
    print(f"Message : {sample.review_comment_message}")

print("\n" + "=" * 80)
print("EXEMPLES par catégorie (premiers 5)")
print("=" * 80)

for category in ['retard_livraison', 'livraison_incomplete', 'produit_incorrect', 'produit_endommage', 'aspect_mismatch']:
    print(f"\n{category}:")
    for example in text_matches[category][:5]:
        print(f"  - \"{example['text']}...\" (note={int(example['score'])})")

