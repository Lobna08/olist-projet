# Modélisation : résultats et justification du split

> Instantané versionné pour la soutenance, généré par `python src/models/train.py`
> et copié depuis `artifacts/modeling_report.md` (non versionné, régénéré à chaque
> run). Ce fichier n'est PAS auto-mis-à-jour : recopier manuellement la version fraîche ici après un changement du script.

Cutoff retenu : **2018-06-01** (train = achats avant cette date, test = à partir de cette date).

## Comparaison des 3 modèles (test agrégé, juin-août 2018)

| modele | n_test | no_skill_pr_auc | pr_auc | precision | recall | f1 |
|---|---|---|---|---|---|---|
| Dummy (plancher) | 18603 | 0.0548 | 0.0548 | 0.0 | 0.0 | 0.0 |
| Régression logistique | 18603 | 0.0548 | 0.1128 | 0.0766 | 0.9235 | 0.1414 |
| LightGBM | 18603 | 0.0548 | 0.0871 | 0.0817 | 0.4347 | 0.1376 |

## Diagnostic sur-apprentissage : PR-AUC train vs test

| modele | pr_auc_train | pr_auc_test |
|---|---|---|
| Dummy (plancher) | 0.0874 | 0.0548 |
| Régression logistique | 0.1599 | 0.1128 |
| LightGBM | 0.2809 | 0.0871 |

## Matrices de confusion (test agrégé)

**Dummy (plancher)**
```
[[17584     0]
 [ 1019     0]]
```

**Régression logistique**
```
[[ 6236 11348]
 [   78   941]]
```

**LightGBM**
```
[[12608  4976]
 [  576   443]]
```

## Ventilation mensuelle du PR-AUC (robustesse — le score agrégé n'est-il pas un artefact d'un mois facile ?)

| mois | modele | n | n_retards | no_skill_pr_auc | pr_auc |
|---|---|---|---|---|---|
| 2018-06 | Dummy (plancher) | 6096 | 83 | 0.0136 | 0.0136 |
| 2018-06 | Régression logistique | 6096 | 83 | 0.0136 | 0.041 |
| 2018-06 | LightGBM | 6096 | 83 | 0.0136 | 0.1241 |
| 2018-07 | Dummy (plancher) | 6156 | 276 | 0.0448 | 0.0448 |
| 2018-07 | Régression logistique | 6156 | 276 | 0.0448 | 0.114 |
| 2018-07 | LightGBM | 6156 | 276 | 0.0448 | 0.0937 |
| 2018-08 | Dummy (plancher) | 6351 | 660 | 0.1039 | 0.1039 |
| 2018-08 | Régression logistique | 6351 | 660 | 0.1039 | 0.1419 |
| 2018-08 | LightGBM | 6351 | 660 | 0.1039 | 0.1003 |

## Justification du cutoff — taux de retard mensuel historique complet

| mois | n | n_retards | pct_retard |
|---|---|---|---|
| 2016-09 | 1 | 1.0 | 100.0 |
| 2016-10 | 265 | 3.0 | 1.13 |
| 2016-12 | 1 | 0.0 | 0.0 |
| 2017-01 | 750 | 23.0 | 3.07 |
| 2017-02 | 1653 | 53.0 | 3.21 |
| 2017-03 | 2546 | 142.0 | 5.58 |
| 2017-04 | 2303 | 181.0 | 7.86 |
| 2017-05 | 3545 | 128.0 | 3.61 |
| 2017-06 | 3135 | 121.0 | 3.86 |
| 2017-07 | 3872 | 133.0 | 3.43 |
| 2017-08 | 4193 | 139.0 | 3.32 |
| 2017-09 | 4150 | 216.0 | 5.2 |
| 2017-10 | 4478 | 237.0 | 5.29 |
| 2017-11 | 7288 | 1043.0 | 14.31 |
| 2017-12 | 5513 | 462.0 | 8.38 |
| 2018-01 | 7069 | 464.0 | 6.56 |
| 2018-02 | 6555 | 1048.0 | 15.99 |
| 2018-03 | 7003 | 1496.0 | 21.36 |
| 2018-04 | 6798 | 361.0 | 5.31 |
| 2018-05 | 6749 | 556.0 | 8.24 |
| 2018-06 | 6096 | 83.0 | 1.36 |
| 2018-07 | 6156 | 276.0 | 4.48 |
| 2018-08 | 6351 | 660.0 | 10.39 |

## Justification du cutoff — comparaison de cutoffs candidats

| cutoff | n_test | pct_test | n_retards_test | pct_retard_test | pct_retard_train |
|---|---|---|---|---|---|
| 2018-04-01 | 32150 | 33.3 | 1936 | 6.02 | 9.16 |
| 2018-05-01 | 25352 | 26.3 | 1575 | 6.21 | 8.79 |
| 2018-06-01 | 18603 | 19.3 | 1019 | 5.48 | 8.74 |
| 2018-07-01 | 12507 | 13.0 | 936 | 7.48 | 8.21 |

## Analyse de seuil de décision (régression logistique, modèle final)

Seuil par défaut de scikit-learn (0.5) : ~66% des commandes de test flaguées, précision 7.7%. C'est un choix de seuil non examiné, pas un défaut du modèle — d'où cette analyse. Le seuil ci-dessous n'est **pas** optimisé en boucle sur le test : ce tableau expose le compromis précision/recall pour un arbitrage métier humain, fait une seule fois.

| seuil | pct_flague | n_flague | precision | recall | retards_captes | fausses_alertes |
|---|---|---|---|---|---|---|
| 0.1 | 99.52 | 18514 | 0.055 | 1.0 | 1019 | 17495 |
| 0.15 | 98.88 | 18394 | 0.0554 | 1.0 | 1019 | 17375 |
| 0.2 | 97.8 | 18193 | 0.056 | 1.0 | 1019 | 17174 |
| 0.25 | 95.9 | 17841 | 0.057 | 0.998 | 1017 | 16824 |
| 0.3 | 93.06 | 17312 | 0.0586 | 0.9961 | 1015 | 16297 |
| 0.35 | 88.97 | 16552 | 0.0611 | 0.9931 | 1012 | 15540 |
| 0.4 | 83.53 | 15540 | 0.0645 | 0.9833 | 1002 | 14538 |
| 0.45 | 76.44 | 14220 | 0.0695 | 0.9696 | 988 | 13232 |
| 0.5 | 66.06 | 12289 | 0.0766 | 0.9235 | 941 | 11348 |
| 0.55 | 52.73 | 9809 | 0.0903 | 0.8695 | 886 | 8923 |
| 0.6 | 36.81 | 6848 | 0.1095 | 0.736 | 750 | 6098 |
| 0.65 | 21.02 | 3910 | 0.1274 | 0.4887 | 498 | 3412 |
| 0.7 | 11.08 | 2062 | 0.1193 | 0.2414 | 246 | 1816 |
| 0.75 | 6.08 | 1131 | 0.1026 | 0.1138 | 116 | 1015 |
| 0.8 | 3.23 | 600 | 0.1033 | 0.0608 | 62 | 538 |
| 0.85 | 1.33 | 247 | 0.1255 | 0.0304 | 31 | 216 |
| 0.9 | 0.43 | 80 | 0.225 | 0.0177 | 18 | 62 |
| 0.95 | 0.03 | 5 | 0.0 | 0.0 | 0 | 5 |

### Seuils défendables proposés

| regle | seuil | pct_flague | n_flague | precision | recall | retards_captes | fausses_alertes |
|---|---|---|---|---|---|---|---|
| Flaguer les 10% les plus à risque | 0.7082 | 10.0 | 1861 | 0.1123 | 0.2051 | 209 | 1652 |
| Précision >= 20% (au plus 1 fausse alerte pour 4 vraies) | 0.9 | 0.43 | 80 | 0.225 | 0.0177 | 18 | 62 |

### Seuil retenu (décision métier, pas un tuning)

**Seuil = 0.6**, tranché par l'utilisateur après revue du tableau ci-dessus.

Raisonnement : coût de l'action déclenchée par une alerte → tolérance aux fausses alertes → seuil. Ici, l'alerte déclenche une action automatique à faible coût (pas d'intervention humaine par commande) : une fausse alerte ne coûte quasiment rien à traiter, alors qu'un retard non détecté est une occasion manquée d'agir. Le recall est donc priorisé sur la précision.

À ce seuil :

| seuil | pct_flague | n_flague | precision | recall | retards_captes | fausses_alertes |
|---|---|---|---|---|---|---|
| 0.6 | 36.81 | 6848 | 0.1095 | 0.736 | 750 | 6098 |

**Limite assumée** : la précision (~11%) reste basse à ce seuil comme à tout autre seuil raisonnable de la grille — ce n'est pas une conséquence du choix de seuil, c'est le plafond du signal disponible dans les features (les causes de retard sont en grande partie post-achat : aléas transporteur, dernier kilomètre — non observables au moment de la commande). Changer le seuil déplace le curseur précision/recall, il ne fait pas monter ce plafond.
