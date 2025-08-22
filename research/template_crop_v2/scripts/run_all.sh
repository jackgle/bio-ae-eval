#!/usr/bin/env bash
set -euo pipefail

source config.env

# # step 1: extract TP clips
# python src/clip_extractor_bandpass.py \
#   --input-root "$AUDIO_SOURCE_ROOT" \
#   --out-root "$CLIPS_OUT" \
#   --csv "${TP_CSV}:tp" \
#   --seconds "$CLIP_SECONDS" \
#   --overwrite 

# # step 2: extract FP clips
# python src/clip_extractor_bandpass.py \
#   --input-root "$AUDIO_SOURCE_ROOT" \
#   --out-root "$CLIPS_OUT" \
#   --csv "${FP_CSV}:fp" \
#   --seconds "$CLIP_SECONDS" \
#   --overwrite

# # step 3: generate embeddings inside bacpipe root
# for MODEL in "${MODELS[@]}"; do
#   (
#     cd "$BACPIPE_ROOT" || { echo "bad BACPIPE_ROOT: $BACPIPE_ROOT"; exit 1; }
#     python "$(pwd)/../src/batch_embed.py" \
#       --audio-root "$CLIPS_OUT" \
#       --model "$MODEL" \
#       --out-root "$EMBEDDINGS_OUT" \
#   )
# done

# step 4: evaluate (cropped -> full)
python src/embedding_eval_pipeline.py \
  $EMBEDDINGS_CROPPED \
  $EMBEDDINGS_FULL \
  --metric cosine \
  --ap \
  --n-proto 20 \
  --csv "$RESULTS_CSV"


