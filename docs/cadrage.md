# Cadrage — Définition de la cible et frontière de fuite

## 1. Variable cible : `is_late`

- **Population** : on n'utilise que les commandes effectivement livrées, donc celles avec `order_status = 'delivered'`. Les autres (annulées, en cours, indisponibles) n'ont pas de date de livraison réelle, donc pas de cible calculable.
- **Règle de calcul** : pour ces commandes, on compare la date de livraison réelle à la date de livraison estimée.
  - Si `order_delivered_customer_date > order_estimated_delivery_date` → `is_late = 1`
  - Sinon → `is_late = 0`

## 2. Moment de prédiction

- Le modèle prédit à **l'instant de l'achat** (`order_purchase_timestamp`).
- Justification métier : c'est le seul moment où la prédiction a une valeur d'action. En prédisant le risque de retard dès l'achat, on peut alerter la société/le vendeur pour qu'ils interviennent à temps et évitent le retard. Une prédiction faite après la livraison serait inutile.

## 3. Frontière de fuite (data leakage)

- Principe : à l'instant de l'achat, le modèle ne connaît que ce qui existe déjà à ce moment. Toute information postérieure à l'achat est interdite en feature.
- **Interdit en feature** : `order_delivered_customer_date` (date de livraison réelle) — elle n'existe pas encore au moment de l'achat. L'utiliser reviendrait à donner la réponse au modèle (fuite).
- **Autorisé en feature** : `order_estimated_delivery_date` (date estimée) — elle est connue dès l'achat, donc utilisable.
- Cas particulier à retenir : la date de livraison réelle est **interdite en feature** mais **nécessaire pour fabriquer la cible** (le calcul de `is_late` au point 1 en a besoin). Même colonne, deux usages : bannie de l'entrée du modèle, indispensable pour construire le label.
- Note : les données de review (note, commentaire) sont aussi postérieures à la livraison → également interdites en feature, autorisées seulement pour les KPIs BI.

## 4. Limite assumée

Le modèle prédit le retard *conditionnellement au fait que la commande sera livrée*. Il ne traite pas les commandes annulées ou jamais livrées. À mentionner dans la section « limites » du projet.
