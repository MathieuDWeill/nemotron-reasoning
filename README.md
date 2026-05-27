# NVIDIA Nemotron Model Reasoning Challenge — clean starter repo

Base de travail propre pour entraîner, valider et packager un adaptateur LoRA compatible avec la compétition Kaggle **NVIDIA Nemotron Model Reasoning Challenge**.

Objectif: produire un `submission.zip` contenant un adaptateur LoRA Nemotron-3-Nano-30B avec `adapter_config.json` à la racine du zip.

## Structure

```text
.
├── configs/
│   └── sft_default.yaml              # hyperparamètres modifiables
├── data/
│   └── raw/                          # train.csv / test.csv si disponibles localement
├── docs/
│   └── competition_notes.md          # résumé règles / idées d'expériences
├── notebooks/
│   ├── 01_kaggle_sfttrainer_training_reference.ipynb
│   └── 02_kaggle_submission_reference.ipynb
├── outputs/
│   ├── adapters/                     # adaptateurs LoRA entraînés
│   └── submissions/                  # submission.zip
├── scripts/
│   ├── kaggle_train.py               # entraînement SFT sur Kaggle
│   ├── package_adapter.py            # zip propre de l'adaptateur
│   └── smoke_test_data.py            # sanity checks rapides
├── src/nemotron_reasoning/
│   ├── data.py
│   ├── metric.py
│   ├── prompts.py
│   └── utils.py
├── requirements.txt
└── .gitignore
```

## Démarrage rapide Kaggle

1. Créer un notebook Kaggle avec GPU adapté.
2. Ajouter comme inputs:
   - la compétition `nvidia-nemotron-model-reasoning-challenge`
   - le modèle Kaggle `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
   - éventuellement le dataset offline packages si nécessaire.
3. Copier ce repo dans `/kaggle/working`, ou uploader le zip du repo comme dataset.
4. Lancer:

```bash
python scripts/kaggle_train.py --config configs/sft_default.yaml
python scripts/package_adapter.py --adapter-dir outputs/adapters/nemotron_lora --out outputs/submissions/submission.zip
```

## Points importants compétition

- Rang LoRA maximum: **32**.
- Le zip doit contenir au minimum `adapter_config.json` et les poids de l'adaptateur.
- Le score extrait prioritairement la réponse dans `\boxed{...}`.
- Les prix demandent un notebook public Kaggle + write-up reproductible.

## Stratégie recommandée

Commencer simple:

1. SFT baseline: `prompt -> \boxed{answer}`.
2. Ajouter des traces de raisonnement synthétiques pour les familles faciles.
3. Valider localement par famille de tâche, pas seulement en split aléatoire.
4. Soumettre uniquement un zip dont le contenu est vérifié.

