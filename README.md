# Traducteur de phrases avec Transformer manuel

Le modèle encodeur-décodeur de `src/model.py` est écrit à la main en PyTorch (pas de `torch.nn.Transformer`). Il est entraîné de zéro avec teacher forcing sur des paires de phrases OPUS-100 ; `tiktoken` sert uniquement de tokenizer BPE, pas de modèle de traduction.

## Préparer, entraîner et lancer (PowerShell)

Depuis la racine du projet :

```powershell
uv sync
.\.venv\Scripts\python.exe -m src.prepare_opus100
.\.venv\Scripts\python.exe -m src.train --epochs 10 --max-vocab 16000 --batch-size 32 --device auto
.\.venv\Scripts\python.exe .\main.py
```

La préparation télécharge les trois archives nécessaires dans `data/opus100/raw/`, puis crée les splits JSONL dans `data/opus100/processed/`. Par défaut, elle garde au plus 50 000 phrases d'entraînement par paire source pour limiter le volume ; `--max-train-per-pair 0` utilise toutes les données. OPUS-100 est centré sur l'anglais, donc les traductions française→allemand/espagnol passent par français→anglais puis anglais→langue cible. Les archives OPUS agrègent plusieurs corpus : vérifie leurs licences avant toute redistribution. Voir [OPUS-100](https://opus.nlpl.eu/OPUS-100) et sa page de téléchargement.

Le Transformer est entraîné sans poids préentraînés. `--device auto` utilise CUDA uniquement si PyTorch le détecte ; le batch de 32 et le vocabulaire de 16 000 peuvent demander un GPU et plusieurs heures sur CPU. Réduis `--batch-size` si la mémoire GPU manque. Le checkpoint est `artifacts/sentence_transformer.pt`; l'ancien checkpoint lexical reste intact.

L'interface prend des phrases, tire la langue cible au sort et affiche la génération du Transformer. Pour de/es, elle affiche également l'étape pivot anglaise. La qualité dépend du volume et du domaine : un entraînement de démonstration depuis zéro n'égale pas un système de production et ne garantit pas la traduction de mots/phrases hors distribution.
