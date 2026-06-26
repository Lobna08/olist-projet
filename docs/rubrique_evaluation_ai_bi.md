# Rubrique d'évaluation — Projet AI & BI (Olist Delivery-Risk Command Center)

> Document de référence. À charger dans les *connaissances* du Projet Claude.
> La skill « auditeur-projet-ai-bi » s'appuie sur cette rubrique pour juger le travail.

## 1. Critères de validation académique (issus du brief de cours)

| Critère | Ce qui est attendu | Échec typique à signaler |
|---|---|---|
| **Idée originale** | Problématique métier concrète, formulée clairement | Sujet générique recopié d'un tutoriel, sans cadrage |
| **Pertinence métier** | Le problème a un coût/impact réel et identifié | « C'est intéressant » sans enjeu business chiffrable |
| **Faisabilité** | Périmètre réaliste, données disponibles, plan crédible | Ambition non dimensionnée pour le temps/les moyens |
| **Approche** | Méthode et feuille de route explicites et défendables | Liste d'outils sans logique de pipeline |

## 2. Exigence centrale — complémentarité réelle AI ↔ BI

Le brief impose une *réelle complémentarité*. Critère de réussite, non négociable :

- La **prédiction du modèle vit DANS la couche BI** (dimension/colonne filtrable de l'entrepôt, exploitable par un utilisateur métier).
- **Idéalement une boucle** : les segments BI nourrissent aussi la modélisation.
- **Échec à signaler** : un modèle dans un notebook + un dashboard à côté = adjacence, pas intégration. C'est le piège que les examinateurs traquent.

## 3. Barre « publiable / CV »

Un dépôt est CV-grade seulement si :

1. **README narratif** : problème → données → schéma → modèle → résultat → comment lancer.
2. **Reproductible** : clone → install → une commande → résultats. Pas de « works on my machine ».
3. **Structure propre** : `data/ src/ models/ notebooks/` séparés, pas un notebook unique de 2000 lignes.
4. **Modèle correctement évalué** : une baseline à battre, la *bonne* métrique justifiée (classes déséquilibrées → pas l'accuracy), une section « limites » honnête.
5. **Intégration AI↔BI visible** (cf. §2).
6. **Références réelles, citées.**
7. **Hygiène git** : commits réels et atomiques, pas un seul commit « final ».

## 4. Pièges éliminatoires — fuite de données (data leakage)

La fuite se cache à **trois endroits simultanés**. À vérifier systématiquement :

- **Cible** : définie sans utiliser d'information postérieure à l'achat.
- **Features** : uniquement des signaux disponibles **au moment de la commande** (point-in-time). Pas de date de livraison réelle, pas de note de review, etc.
- **Split** : **temporel**, jamais aléatoire.

> Signal d'alerte : un score suspectement élevé = fuite quasi certaine. Le dire franchement.

## 5. Posture d'évaluation

Objectivité stricte, aucune flatterie. Nommer les faiblesses directement ; une faiblesse cachée se lit comme « junior », une faiblesse assumée comme « senior ». Le verdict prime sur l'encouragement.
