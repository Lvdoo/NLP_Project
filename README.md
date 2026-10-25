# Traducteur lexical expérimental

Application locale qui prend un texte français, choisit elle-même une langue cible (anglais, allemand ou espagnol), puis affiche deux résultats distincts : la prédiction du Transformer PyTorch et, lorsqu’elle existe, une correspondance du dictionnaire MUSE. L’utilisateur ne choisit pas la langue.

## Ce que fait le programme

- `main.py` démarre le serveur local et ouvre l’interface dans le navigateur. Le bouton **Traduire** lance la traduction ; **Nouvelle traduction** tire de nouveau une langue au sort (elle peut être la même que précédemment).
- La saisie accepte jusqu’à 500 caractères. Le programme découpe le texte en mots : il traduit chaque mot isolément et conserve la ponctuation. Il ne réordonne pas les mots et ne produit donc pas une traduction grammaticale de phrase.
- Le modèle est un Transformer encodeur-décodeur écrit à la main dans `src/model.py` : encodage positionnel, attention multi-têtes, masques causal et padding, blocs encodeur-décodeur et projection de sortie. Il n’utilise pas `torch.nn.Transformer`.
- L’interface sépare la sortie expérimentale du modèle et la correspondance MUSE. La correspondance dictionnaire n’est jamais présentée comme une prédiction du Transformer. Si un mot est absent du vocabulaire du modèle, celui-ci ne fournit pas de prédiction ; s’il est absent du dictionnaire, l’interface le signale.

## Installation, préparation, entraînement et lancement

Depuis PowerShell, à la racine du projet :

```powershell
uv sync
.\.venv\Scripts\python.exe -m src.prepare_muse_words
.\.venv\Scripts\python.exe -m src.train --epochs 15 --max-vocab 100000 --batch-size 64 --device auto
.\.venv\Scripts\python.exe .\main.py
```

La préparation télécharge au besoin les dictionnaires complets français-anglais, français-allemand et français-espagnol dans `data/muse/raw/`, puis génère les splits JSONL dans `data/muse/processed/`. Les fichiers bruts font environ 4,5 Mo au total. Ils contiennent respectivement 96 775, 61 048 et 86 864 paires lexicales retenues après nettoyage. MUSE fournit jusqu’à environ 100 000 paires par langue ; sa licence est CC BY-NC 4.0 (usage non commercial, attribution requise) : [données](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries), [licence](https://github.com/facebookresearch/MUSE/blob/main/LICENSE).

Le split train/validation/test est fait par lemme français, à 90/5/5, pour éviter qu’un même mot source apparaisse dans plusieurs splits. Les mots de validation/test sont donc inconnus du vocabulaire d’entraînement. Le Transformer a un vocabulaire fermé : ce protocole teste justement sa limite sur les mots non vus et ne garantit pas qu’il saura les traduire. Avec `--max-vocab 100000`, une partie des mots rares ou des formes fléchies reste hors vocabulaire. Les mots isolés sont aussi ambigus selon leur contexte ; une entrée de dictionnaire n’est pas une traduction validée dans une phrase.

`--device auto` utilise CUDA si cette installation de PyTorch détecte un GPU ; sinon il choisit le CPU. Vérification sous PowerShell :

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

L’entraînement n’affiche ses métriques qu’à la fin de chaque époque. Les poids et métadonnées sont enregistrés dans `artifacts/word_muse_full_transformer.pt`. Le checkpoint antérieur `artifacts/word_multilingual_transformer.pt` est conservé et n’est pas chargé par défaut. Un avertissement PyTorch indiquant que NumPy est absent n’empêche pas le prétraitement ou l’entraînement dans l’environnement vérifié.

## Organisation

- `src/model.py` : Transformer manuel.
- `src/prepare_muse_words.py` : téléchargement et préparation des données MUSE.
- `src/train.py` et `src/train_multilingual.py` : entraînement, validation, test et sauvegarde.
- `src/translation.py` : chargement des poids, choix aléatoire de la langue, prédiction et recherche dictionnaire séparées.
- `src/webapp.py`, `main.py`, `web/` : serveur local et interface.
- `data/muse/` : fichiers MUSE bruts, splits et manifeste ; `artifacts/` : checkpoints.

Ce prototype est une expérience de traduction lexicale mot-à-mot, pas un traducteur général ni un traducteur LSF. Aucun modèle préentraîné ni service externe de traduction n’est utilisé.
