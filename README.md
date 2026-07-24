# QRouter:  Question-Conditioned Visual Evidence Routing for Fine-Grained Visual Question Answering
# [Note] More provenance and authenticity-related information is available in the [`metadata/`](metadata/) directory.

Official PyTorch implementation of QRouter, a vision-language model that uses
question-conditioned grounding to construct a compact visual prefix. QRouter-B
uses 32 region tokens, 128 routed context tokens, and one background token.

## Installation

The experiments use Python 3.10, PyTorch 2.1.2, CUDA 11.8, and eight NVIDIA
A800 80GB GPUs.

Create a clean environment and install the official CUDA 11.8 PyTorch wheels:

```bash
conda create -n qrouter python=3.10 -y
conda activate qrouter

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies after PyTorch. `--no-build-isolation` is
required so that Mamba and causal-conv1d are built against the CUDA-enabled
PyTorch already present in the environment.

```bash
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -e . --no-deps
```

Verify that both PyTorch and the local CUDA toolkit report CUDA 11.8:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

Place the [SAM 2](https://github.com/facebookresearch/sam2) source revision
used for the experiment at `third_party/sam2`. Install it without allowing its
dependency resolver to replace the pinned PyTorch installation:

```bash
python -m pip install \
  --no-deps --no-build-isolation -e third_party/sam2
```

Download the SAM 2 Hiera-L checkpoint and place it under `weights/`; the
remaining model weights are downloaded by Hugging Face on first use. If the
optional SAM 2 post-processing CUDA extension cannot be compiled, install with
the following command while keeping the same source revision and checkpoint:

```bash
SAM2_BUILD_CUDA=0 python -m pip install \
  --no-deps --no-build-isolation -e third_party/sam2
```

## Repository layout

```text
qrouter/
  conf/                 configuration loading and validation
  models/
    backbones/          DINOv2-SigLIP and Mamba wrappers
    grounding/          Qwen2.5-VL-to-SAM2 grounding
    vlms/               QRouter and the region tokenizer
  preprocessing/        JSONL datasets and collation
  training/             checkpointing and distributed strategies
  util/                 PyTorch utilities
scripts/
  pretrain.py           Stage-I and Stage-II training
  evaluate.py           benchmark evaluation
```

The default configuration is `configs/qrouter_b_paper.yaml`. Dataset formats
and expected manifest names are documented in
[`data/manifests/README.md`](data/manifests/README.md).

## Training

Stage I runs 20,000 QA optimization steps. Stage II starts from the best
Stage-I checkpoint and runs 10,000 steps with a 2:1 mixture of QA and
conversational segmentation samples. The global batch size is
`16 per device x 8 devices x 1 accumulation step = 128`.
The SAM2 image encoder remains frozen. In Stage II, the SAM2 mask decoder,
Qwen LoRA parameters, and learned cross-model prompt projections are adapted
with the joint QA and CIS objectives.

```bash
torchrun --standalone --nproc-per-node 8 scripts/pretrain.py \
  --config configs/qrouter_b_paper.yaml \
  --stage stage1

torchrun --standalone --nproc-per-node 8 scripts/pretrain.py \
  --config configs/qrouter_b_paper.yaml \
  --stage stage2 \
  --init-checkpoint outputs/qrouter_b/stage1/checkpoints/step-020000.pt
```

The grounding adapter issues 32 independent question-conditioned prompt
queries, and SAM2 decodes one mask for each query (`multimask_output=False`).
After confidence filtering, the number of valid masks is sample-dependent;
`K=32` is the maximum number of region-token slots, and invalid slots are
masked out. It is therefore unrelated to SAM2's three-mask ambiguity output.
`L=128` is the context-patch routing budget.

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/qrouter_b_paper.yaml \
  --checkpoint weights/qrouter_b_stage2.pt \
  --manifest data/manifests/gqa_test.jsonl \
  --benchmark gqa \
  --output results/gqa.jsonl
```

## Checkpoint

The checkpoint stores the complete QRouter model state, including the
grounding module, together with optimizer, scheduler, scaler, step, epoch,
resolved arguments, and training metrics.

The QRouter-B Stage-II checkpoint used for the reported GQA result has:

```text
SHA-256: 1f008f067425fd42db94fb23616664feb37d20be1234822dae2ecd4dbea37613
size:    18,227,596,894 bytes
```

The GQA evaluation contains 12,578 samples and 8,572 normalized exact
matches, corresponding to 68.2 accuracy. Checkpoint metadata and the evaluation
summary are stored in `metadata/`.

## Acknowledgements

QRouter is built on the model organization and Mamba VLM components of
[Cobra](https://github.com/OpenHelix-Team/cobra). It also uses DINOv2, SigLIP,
Qwen2.5-VL, and SAM 2.


