# Interpreting visual information processing in LFM2.5-VL

A replication of [Neo et al., *Towards Interpreting Visual Information Processing in
Vision-Language Models*](https://arxiv.org/abs/2410.07149) (ICLR 2025) on
[`LiquidAI/LFM2.5-VL-3B`](https://huggingface.co/LiquidAI/LFM2.5-VL-3B).

The paper studies LLaVA-1.5-7B (plus LLaVA-Phi-3 and Qwen2VL-2B) with three experiments:
token ablation, logit lens, and attention knockout. Their reference implementation is in
`../llava-interp-main`, kept here untouched. This package re-implements the same three
experiments for a model whose architecture breaks nearly every assumption that code makes.

---

## Why this is a port and not a config change

| | LLaVA-1.5-7B (paper) | LFM2.5-VL-3B |
|---|---|---|
| Vision tower | CLIP ViT-L/14 @ 336 | SigLIP2 `so400m-patch16-naflex`, 27 layers, **variable patch count** |
| Preprocessing | center-crop to square, resize 336×336 | aspect-preserving `smart_resize`, **no crop**; tiling only for large images |
| Projector | MLP, 1 patch → 1 token | **2×2 pixel-unshuffle** + MLP, 4 patches → 1 token |
| Visual tokens | fixed **576**, grid 24×24 | **variable 64–256**, grid `(H/32)×(W/32)` |
| LM | Vicuna-7B, **32 attention layers** | LFM2-2.6B, 30 layers = **8 attention + 22 short-conv** |
| Attention layers | all | **`[2, 5, 9, 13, 17, 21, 24, 27]`** |
| Image token | id 32000, one run | id **124907**, wrapped in `<\|image_start\|>` / `<\|image_end\|>` |
| Final norm | `language_model.model.norm` | `model.model.language_model.embedding_norm` |
| `lm_head` | separate | **tied** to the input embeddings |

A typical COCO image (640×480) is resized to 416×576, giving 26×36 patches and a
**13×18 = 234** visual-token grid — not 576 tokens on a square.

Three consequences drove the design:

1. **Geometry is computed, not assumed.** [`src/geometry.py`](src/geometry.py) mirrors the
   processor's resize arithmetic, so any COCO annotation maps to token indices for any
   image shape. The processor's `spatial_shapes` is treated as authoritative at runtime
   and cross-checked against the offline computation on every image.
2. **Attention knockout is a partial cut.** Only 8 of 30 layers have attention; the 22
   `Lfm2ShortConv` layers mix ~2 positions backwards each and ignore token-pair masking.
   [`scripts/attention_knockout.py`](scripts/attention_knockout.py) reports this and runs
   a **distance control** to separate real attention routing from conv leakage.
3. **The final hidden state is already normed.** `lm_head(hidden_states[-1]) == logits`
   on this model, so the logit lens must *not* re-apply the norm to the last layer —
   which the reference implementation does uniformly.

---

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`transformers==5.8.1` is a hard requirement, not a suggestion: the checkpoint uses v5
conventions (`TokenizersBackend`, `rope_parameters`, no `special_tokens_map.json`) and
transformers 4.x cannot load its processor.

COCO (19 GB, needed only on the machine that runs the experiments):

```bash
wget -P data/coco/ http://images.cocodataset.org/zips/train2017.zip
wget -P data/coco/ http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip data/coco/train2017.zip -d data/coco/
unzip data/coco/annotations_trainval2017.zip -d data/coco/
```

---

## Developing without a GPU

Every script takes `--tiny`, which swaps in a random-weight miniature built from a
shrunken `Lfm2VlConfig` — same `Lfm2VlForConditionalGeneration` class, same hybrid
conv/attention layer pattern, same real processor and image-token id, 6 layers at
hidden size 64 on CPU. Results are meaningless; every code path executes.

```bash
pytest                       # 231 tests: geometry, spans, hooks, lens, script smoke tests
pytest tests/test_geometry.py   # pure Python, no downloads at all
```

The test suite downloads only the processor (~10 MB). It never downloads model weights.

---

## Running the experiments

All commands read [`configs/lfm2_5_vl_3b.yaml`](configs/lfm2_5_vl_3b.yaml); every value
can be overridden on the command line.

```bash
# 0. Sanity check: does the real checkpoint load and caption?
python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
m = AutoModelForImageTextToText.from_pretrained('LiquidAI/LFM2.5-VL-3B', dtype='bfloat16', device_map='cuda:0')
print(m.config.text_config.layer_types)"

# 1. Filter COCO and apply the hallucination control.
python scripts/build_dataset.py --split train2017

# 2. Mean visual token, the ablation replacement vector.
python scripts/compute_mean_visual_token.py --split train2017 --num-images 10000

# 3. Ablation: generative + polling (Table 1).
python scripts/ablation_experiment.py --split train2017 --attn-implementation sdpa

# 4. Ablation: VQA, reusing the paper's 100 curated questions (Table 1, VQA column).
python scripts/ablation_experiment_vqa.py --split train2017 --attn-implementation sdpa

# 5. Attention knockout (Table 2). Needs eager attention.
python scripts/attention_knockout.py --split train2017 --attn-implementation eager

# 6. Logit lens: the 23.7% measurement, plus interactive pages.
python scripts/logit_lens/quantify_object_tokens.py --split val2017
python scripts/logit_lens/create_logit_lens.py --image-folder data/coco/val2017 --limit 20
python scripts/logit_lens/generate_overview.py

# 7. Render both tables.
python scripts/analyze_results.py --split train2017
```

`--attn-implementation eager` is **required** for step 5: LFM2's attention adds a float
mask, and under sdpa the mask can be `None` (the `is_causal` fast path), leaving nothing
for the knockout hook to patch. Steps 3–4 run faster under `sdpa`.

Everything is resumable — results are keyed by image id and written atomically after each
image, so re-running picks up where it stopped.

---

## What the paper found, and what to compare against

| Experiment | Paper's LLaVA-1.5 result |
|---|---|
| Ablation | Object+1 buffer (33.4 of 576 tokens) retains 71.8 / 51.3 / 86.7 % across generative / polling / VQA, well below random-40 (2.4 / 0.0 / 0.0 → i.e. far *less* damaging) and int-gradients-40 |
| Logit lens | 23.7 % of object-patch positions decode to the class token at the best layer, on average layer 25.7 of 33 (Qwen2VL-2B: 6.5 % at 25.1 of 29) |
| Knockout | Object+1 → last token falls to 0.82 at mid-late layers, 0.67 across all layers; visual → last visual row stays at 1.00 everywhere |

**Absolute numbers will not match**, and should not be expected to: this model has ~234
visual tokens instead of 576, a 4:1 projector, and a mostly-convolutional LM. The
replication targets are the *qualitative* claims:

- object-token ablation hurts more than random or integrated-gradients ablation **at
  matched token counts** (the analysis script prints realised counts alongside every row);
- visual tokens become vocabulary-interpretable, peaking in mid-to-late layers;
- object → last-token attention matters most in mid-to-late layers, while visual → last
  visual row does not matter at all.

---

## Experiment 2: perception probing on synthetic scenes

A second, self-contained suite that asks a different question: **where in the network are
individual perceptual abilities computed** — counting, colour, shape, orientation, spatial
relation — and are they computed in the same place?

COCO cannot answer this: there is no way to change exactly one property of an image and
watch what moves. Synthetic scenes can, which unlocks **activation patching over minimal
pairs** — the method the replication could not use.

```bash
# 1. Generate scenes + the inspection viewer. No GPU, no download, seconds.
python scripts/synthetic/build_dataset.py --config configs/synthetic.yaml
#    -> open results/synthetic/viewer.html and CHECK IT before spending GPU time

# 2. The gate: can the model do these tasks at all?
python scripts/synthetic/baseline.py --config configs/synthetic.yaml

# 3. Localize, on the cells the model actually gets right.
python scripts/synthetic/patching.py --config configs/synthetic.yaml
```

### Why the design looks the way it does

* **512×512 canvas.** The one size that survives `smart_resize` untouched, giving a clean
  16×16 = 256 token grid at exactly 32 px per cell — shape coordinates map to token cells
  with zero interpolation error.
* **Bare-digit answers.** `' 3'` is *two* tokens (`[229, 27]`); `'3'` is one. So counting
  prefills end with a space ("The number of red circles is ") and the answer is a bare
  digit. Digits are contiguous ids 24–33, so a softmax over them gives a full
  `P(count=k)` distribution rather than a correct/incorrect bit. Word answers use the
  opposite convention. `tasks.verify_single_token` asserts this before a run starts.
* **Patching, not knockout.** Only 8 of 30 LM layers have attention, so knockout is blind
  to 73% of the network. Patching the residual stream is architecture-agnostic and reaches
  all 30 — it is the right instrument for this model.
* **Minimal pairs change exactly one shape.** Which attribute changes depends on what the
  question filters on: recolouring one circle does nothing to "how many circles", so
  unfiltered counting swaps the shape's *kind* instead.
* **Group size confounds recovery.** Patching 240 background tokens moves the answer more
  than patching 6 object tokens almost by construction, so the analysis reports recovery
  **per token** alongside the raw value.

### The data viewer

`results/synthetic/viewer.html` is a single self-contained file showing every sample:
the minimal pair side by side with the changed shape outlined, a toggleable token-grid
overlay, per-shape token cells, the patching groups, the exact prompt and the tokenized
answer id — then enriched with predictions and per-layer curves once stages run. It works
with **no model at all**, which is the point: catch a broken generator for free.

```
src/synthetic.py    scenes, shapes, COCO-style annotations, minimal pairs
src/tasks.py        5 tasks: prompts, single-token answers, pair rules
src/patching.py     residual-stream patch hook + recovery metric
src/dataviewer.py   the per-sample viewer
configs/synthetic.yaml
```

---

## Layout

```
src/
  geometry.py         COCO annotation <-> visual-token index mapping; resize arithmetic
  hooked_lfm2vl.py    model wrapper: ablation, attention knockout, activation capture
  logit_lens.py       per-layer unembedding, the 23.7% metric, interactive HTML
  prompts.py          chat-template prompt construction
  coco.py             COCO reader + the paper's image filter (pycocotools optional)
  attribution.py      integrated gradients over visual tokens
  experiment.py       config, CLI, resumable results
  tiny_model.py       random-weight miniature for offline testing
scripts/              one per experiment, plus build_dataset / analyze_results
configs/              lfm2_5_vl_3b.yaml
tests/                231 tests, no GPU and no model weights required
data/clean_questions.json   the paper's 100 curated VQA questions, verbatim
```

## Deviations from the paper, and why

- **Mean visual token from COCO, not ImageNet.** The paper averages over 50,000 ImageNet
  validation images. `compute_mean_visual_token.py` takes any image folder and defaults to
  the COCO split already on disk, avoiding a 6.7 GB download for a vector that serves the
  same purpose. Point `--image-dir` at ImageNet to match the paper exactly.
- **Single-tile processing.** `do_image_splitting: false` forces one visual-token grid per
  image so patch↔token mapping is unambiguous. Typical COCO images fall below the splitting
  threshold anyway, so this changes nothing for them.
- **Knockout windows over attention layers.** The paper's five fixed windows are mapped to
  the attention layers they contain, and — since 8 layers is cheap to sweep exhaustively —
  each layer is also blocked alone, plus sliding windows of 3.
- **Baseline counts are capped.** The paper's `n=250` conditions exceed this model's ~234
  token budget; they are capped at the grid size and the realised count is recorded.
- **Two logit-lens criteria.** The paper does not state whether its 23.7 % counts exact
  class tokens or partial word-pieces (its Figure 3 labels credit "swe"(ater)), so both a
  strict and a lenient criterion are reported.
