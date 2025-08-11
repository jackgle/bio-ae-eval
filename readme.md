
# Embeddings Pipeline
This repository contains a small audio-processing pipeline for:
1. Extracting fixed-length audio clips from long recordings based on annotation CSVs
2. Generating embeddings for each clip using models from the `bacpipe` repo
3. Evaluating how well embeddings separate classes (per-class, micro, macro AUROC; optional AP)
4. Currently this code is prepared to use the training data from the RFCx Species Audio Detection Challenge: https://www.kaggle.com/c/rfcx-species-audio-detection/data. This data contains TP and FP examples for each class, where FP's detections are derived from a classical DSP detection algorithm, and essentially represent "hard negatives". This makes the evaluation more robust than standard positive-classification. 
5. To do: evaluate on long-form PAM data (https://github.com/jackgle/open-bioacoustic-benchmarks).

---
## Setup 

1. `./setup.sh`
2. Copy `config.env.example` to `config.env` and edit as necessary

## Run
1. `source venv/bin/activate`
2. `./scripts/run_all.sh`
	- See the Python scripts called in this function for more modularity

This will:
3. Extract TP and FP clips into `data/audio/clips/tp/` and `.../fp/`
4. Run each model listed in `MODELS` in `config.env` to produce embeddings
5. Evaluate the embeddings and save results to `RESULTS_CSV`

## Notes
- `BACPIPE_ROOT` in the config path must point to the bacpipe repo root. The repo is downloaded during setup.
- To run some models, weights will need to be added to the `bacpipe/model_checkpoints` in the bacpipe repo. See https://github.com/bioacoustic-ai/bacpipe/tree/main 
- Some `.flac` files require `tensorflow-io` — ensure `tensorflow` and `tensorflow-io` versions match
