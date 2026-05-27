# Competition notes

## Evaluation

L'évaluateur charge `Nemotron-3-Nano-30B` + l'adaptateur LoRA via vLLM. La génération est déterministe (`temperature=0.0`, `top_p=1.0`) et la réponse finale doit idéalement être dans `\boxed{}`.

## Families observées dans `train.csv`

- bit manipulation 8-bit
- transformations numériques / unités / équations
- systèmes de numération type Wonderland / Roman-like
- ciphers texte
- transformations symboliques

## Idées d'amélioration

- Construire un validateur local stratifié par famille.
- Générer des targets SFT plus riches: courte analyse puis `\boxed{answer}`.
- Générer du synthétique par famille, surtout pour les règles déterministes.
- Tester `target_modules="all-linear"` versus `['in_proj', 'x_proj', 'dt_proj', 'out_proj']`.
- Ne pas choisir par loss seulement: scorer sur split local.

## Checklist submission

```bash
python scripts/package_adapter.py --adapter-dir outputs/adapters/nemotron_lora --out outputs/submissions/submission.zip
unzip -l outputs/submissions/submission.zip | head
```

Le zip doit montrer `adapter_config.json` à la racine.
