# RoboCasa MimicGen Pipeline

Pipeline minima per:
- estrarre un task RoboCasa da un dataset LeRobot
- generare nuove demo con MimicGen
- convertire `demo.hdf5` in LeRobot con stato canonico `53` e azione `12`
- fare fine-tuning RLDX
- valutare un checkpoint nello stesso ambiente della demo di training

## Script tenuti

- `export_training_task.py`
- `generate_mimicgen_dataset.py`
- `mimicgen_to_lerobot.py`
- `train_rldx_robocasa.sh`
- `eval_rldx_robocasa.sh`
- `eval_rldx_same_demo.sh`
- `eval_same_demo.py`

## Prerequisiti

Serve configurare questi path quando MimicGen non e installato nei default locali:

```bash
export RLDX_MIMICGEN_ROOT=/path/to/mimicgen
export RLDX_ROBOCASA_ROOT=/path/to/robocasa
export RLDX_ROBOMIMIC_ROOT=$RLDX_MIMICGEN_ROOT/external/robomimic
export RLDX_MIMICGEN_PYTHON=/path/to/envs/mimicgen310/bin/python
```

Per il setup simulatore RoboCasa usato in eval:

```bash
bash rldx/eval/sim/robocasa/setup_RoboCasa.sh
```

## Runbook

### 1. Stage del task di training

```bash
python analysis/robocasa_pipeline/export_training_task.py \
  --input /path/to/source/lerobot \
  --task PnPStoveToCounter \
  --output ~/datasets/robocasa_pnp_stovetocounter_train \
  --max-episodes 10
```

### 2. Generazione MimicGen

```bash
python analysis/robocasa_pipeline/generate_mimicgen_dataset.py \
  --lerobot-root ~/datasets/robocasa_pnp_stovetocounter_train \
  --task PnPStoveToCounter \
  --layout 53 \
  --style 43 \
  --num-episodes 100 \
  --max-num-failures 500 \
  --run
```

Output principale:

```text
~/datasets/robocasa_mimicgen_runs/PnPStoveToCounter_layout53_style43/demo/demo.hdf5
```

### 3. Conversione MimicGen -> LeRobot

```bash
python analysis/robocasa_pipeline/mimicgen_to_lerobot.py \
  --hdf5 ~/datasets/robocasa_mimicgen_runs/PnPStoveToCounter_layout53_style43/demo/demo.hdf5 \
  --output ~/datasets/lerobot_pnp_stovetocounter_mg_robocasa
```

### 4. Fine-tuning RLDX

```bash
export DATA_DIR=~/datasets/lerobot_pnp_stovetocounter_mg_robocasa
export CKPT_NAME=rldx_pnp_stovetocounter
export CUDA_VISIBLE_DEVICES=0
export MAX_STEPS=1000
export GLOBAL_BATCH_SIZE=8

bash analysis/robocasa_pipeline/train_rldx_robocasa.sh
```

Il train script accetta anche:
- `ACTION_HORIZON`
- `MODALITY_CONFIG_PATH`
- `MAX_STATE_DIM`
- `MAX_ACTION_DIM`

### 5. Same-demo evaluation

Usa la stessa identica istanza della demo di training: stesso `model_file` e stesso `states[0]`.

```bash
export CUDA_VISIBLE_DEVICES=0
export PORT=20140
export LOG_DIR=./output_final/same_demo/rldx_pnp_stovetocounter

bash analysis/robocasa_pipeline/eval_rldx_same_demo.sh \
  /path/to/checkpoint \
  ~/datasets/robocasa_mimicgen_runs/PnPStoveToCounter_layout53_style43/demo/demo.hdf5
```

Artifact principali:
- `same_demo_ground_truth.mp4`
- `same_demo_policy.mp4`
- `same_demo_summary.json`
- `eval.log`
- `server.log`

## Note

- I video LeRobot vengono scritti direttamente dai frame `obs/*_image` di MimicGen.
- Il dataset convertito usa layout nativo RoboCasa: `53` dim di stato e `12` di azione.
- Il same-demo test e un rollout closed-loop nello stesso setting iniziale, non un replay state-by-state della demo.
