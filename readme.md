project/
├─ README.md
├─ .gitignore
├─ config/
│  ├─ config.env               # shared paths + knobs for bash
│  ├─ eval.yaml                # optional: thresholds, metrics toggles
├─ envs/                       # env definitions (not the envs themselves)
│  ├─ clip-requirements.txt
│  ├─ embed-requirements.txt
│  ├─ eval-requirements.txt
├─ scripts/                    # entrypoints you actually run
│  ├─ run_all.sh
│  ├─ run_extract.sh
│  ├─ run_embed.sh
│  ├─ run_eval.sh
├─ src/                        # python code you’re iterating on
│  ├─ clip_extractor.py
│  ├─ batch_embed.py
│  ├─ embedding_eval_pipeline.py
│  └─ embedding_eval_diagnostics.py
├─ data/                       # big files; keep out of git
│  ├─ audio/
│  │  ├─ source/               # long .flac
│  │  └─ clips/                # generated: tp/, fp/ subdirs
│  └─ annotations/
│     ├─ train_tp.csv
│     └─ train_fp.csv
├─ artifacts/                  # generated, versioned by run
│  ├─ embeddings/              # <run_id>/<model>/<split>/{embeddings.joblib,...}
│  └─ results/                 # <run_id>/{metrics.csv, logs/}
└─ logs/
