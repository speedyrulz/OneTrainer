# Dataset Curation

Two toggles on the *training* tab turn a run into a dataset critic: it watches how hard each image is
as training goes, tells you which ones are fighting you, and optionally trains them less.

Ported from [Fizgig](https://github.com/shootthesound/Fizgig) by Peter Neill (Apache 2.0). The
detection logic is Fizgig's; see [What is different here](#what-is-different-here) for the changes.

## Why per-image loss needs normalising first

A diffusion step's loss is dominated by the *timestep it happened to draw*. Loss at t≈0.95 is
structurally enormous next to t≈0.2, so ranking images by raw loss mostly ranks the dice roll. The
watch buckets every observation by noise level and compares each image against the average for
**that** bucket. What is left is the part that belongs to the image.

On top of that it needs a *trajectory*, not a snapshot. An image can be hard because it is rich and
still being learned, or hard because its caption does not describe it. Those look identical in one
epoch and completely different over eight.

## The settings

| Setting | Default | What it does |
| --- | --- | --- |
| **Detect Problem Images** | off | Watch and report. Purely observational — nothing about training changes. |
| **Per-Image Adaptive LR** | off | Act on the verdicts by scaling each image's share of the loss. |
| **Warmup Epochs** | 2 | Epochs to watch before acting. A trend needs data. |
| **Plateau Patience** | 2 | Epoch boundaries with nothing improving before the run is called finished. |
| **Write Loss Log** | off | Also write every image-step to `loss_log/per_image_loss.jsonl` for offline analysis. |
| **Auto-Recaption Stuck** | off | Rewrite the caption of a confirmed-stuck image and give it a fresh start. |
| **Recaption Model** | BLIP2 | Which captioner does the rewriting. `QWEN3_VL_4B` is what Fizgig uses. |
| **Recaption Trigger Word** | *(empty)* | Appended to the end of every rewritten caption. |
| **Warm Up Look Outliers** | off | Ease images that look unlike the rest of the set in over the first epochs. |
| **Outlier Start LR** | 0.4 | The multiplier an outlier starts at. |
| **Outlier Ramp Epochs** | 4 | How long it takes to reach full strength. |
| **Outlier Sensitivity** | 1.5 | How far below the middle of the pack counts as an outlier, in robust deviations. |

Turning either of the first two on enables the watch. Detection alone never touches training.

## Verdicts

At each epoch boundary every image is reclassified:

| Verdict | Meaning | With Per-Image Adaptive LR |
| --- | --- | --- |
| `stuck` | High residual, not descending. Usually a caption that does not match the image. | throttled ×0.5, escalating toward ×0.1 the longer it stays confirmed |
| `suspect` | Extreme residual early on, before a trend exists to confirm it. | ×0.7, provisional |
| `exhausted` | Had a good run, then plateaued while still above average. Mined out, not broken. | ×0.6, to avoid overbake |
| `watch` | Suspicious this epoch, not yet confirmed. | unchanged |
| `learning` | Hard but descending. This is what a good difficult image looks like. | unchanged |
| `excluded` | Benefit of the doubt spent. | weight 0 — contributes no gradient |

A verdict needs consecutive agreement to stick, so one noisy epoch cannot flip an image, and
consistently healthy images get a gentle ×1.1.

Everything lands in the console, in tensorboard under `curation/*`, and in
`<workspace>/loss_log/problem_images.json`.

## How a per-image learning rate works

A single optimizer step cannot use a different learning rate per image. But scaling a sample's loss
scales exactly its contribution to the gradient, which comes to the same thing — so the multiplier is
applied to each sample's loss before the batch is reduced.

Two details that matter:

* The reduction stays a mean over the **batch**, not over the weights. Throttling an image reduces
  the step; it does not get renormalised back up by its neighbours.
* The losses fed back to the watch are the **raw** ones. The weighting is the action, not the
  measurement — recording weighted losses would let a throttled image look like it had improved.

## Exclusions travel with your images

An image whose benefit of the doubt is spent is excluded for the rest of the run, so the loss average
stops carrying its permanent error term. The exclusion is written to `excluded_images.json` **in that
image's own folder**, so it travels with the dataset across runs.

Fix the caption and the image is re-admitted automatically: each entry snapshots the caption at
exclusion time, and a later run that finds it changed prunes the entry. Entries for images no longer
in the dataset are pruned too, and a state that would exclude *every* image is refused — a run that
trains on nothing is never what anyone wanted, whatever the file says.

## The plateau banner

Every image has a knowable finish epoch: the last boundary at which it was still improving. When
nothing has improved for **Plateau Patience** boundaries, the run has stopped extracting signal and a
banner reports it with a suggested checkpoint to compare around.

It distinguishes a *provisional* plateau (images still being adjudicated, so the run may yet have a
second wind) from a *confirmed* one. It detects "learning finished", not "quality peaked" — treat it
as a window worth comparing checkpoints in, not a verdict.

Nothing stops automatically. To stop on a plateau, use **End Early** on the
[Multi Config](MultiConfigTraining.md) tab.

## Rewriting the caption of a stuck image

A `stuck` verdict is nearly always a caption that does not describe the picture. **Auto-Recaption
Stuck** acts on that directly: the image is shown to a captioning model, the caption is replaced with
what is actually visible, and the image's history is cleared so it is judged fresh from there.

Each image gets **two** attempts. The second asks for exhaustive detail, on the grounds that the first
caption demonstrably was not enough. An image still confirmed stuck after both has had its benefit of
the doubt spent, and the watch excludes it rather than throttling it further down the ladder forever.

### Which captioner

`QWEN3_VL_4B` is the one Fizgig uses and the one to pick if you can afford it — it is a real
vision-language model, so it is *told what to write* rather than primed with the first few words, and
the instruction it is given asks for exactly what captions usually miss:

> the camera viewpoint (e.g. 'viewed from behind', 'side profile', 'close-up'), whether the face is
> visible, and the setting. State only what is visible — no speculation, no names, no style commentary.

The second attempt swaps in a longer instruction asking for 2–4 sentences covering every visible
element — the short caption demonstrably was not enough, so the miss is probably something it skipped.

Costs: `Qwen/Qwen3-VL-4B-Instruct` is an ~8GB download on first use, but it loads **4-bit quantized
by default** (bitsandbytes NF4), needing about 4.5GB of VRAM free rather than the ~10.5GB the full
bf16 weights would. That matters because it loads at an epoch boundary with the training model still
resident — on Windows an allocation past the card does not error, it pages into shared system memory
and the whole run crawls, which reads as a freeze. So two protections are in place:

* **Recaption Precision** on the training tab picks the footprint: `NF4` (~4.5GB free needed, the
  default), `INT8` (~6.5GB), or `BF16` (~10.5GB, byte-exact weights). Captioning is robust to NF4.
* Before loading, free VRAM is measured. If even the chosen precision will not fit, that boundary's
  recaptioning is **skipped with a console message** naming the setting to change — the stuck images
  are simply retried at the next boundary. The run never stalls.

`BLIP2` is the light fallback; it produces noticeably blander captions.

### Training Krea 2: no captioner loads at all

Krea 2's text encoder *is* Qwen3-VL, and when you train Krea 2 the recaptioner borrows it — the same
trick Fizgig uses, which is why Fizgig never pays for a captioner. OneTrainer loads that encoder
without its language head, but Qwen3-VL-4B *ties* the head to the input embeddings (the checkpoint
does not even ship `lm_head.weight`), so a weightless generation shell grafted around the resident
encoder is the complete model. Nothing downloads, and when the encoder is already on the GPU nothing
extra loads into VRAM.

The console says so when it happens: `reusing the training run's own Qwen3-VL text encoder`.

One thing worth knowing: with latent caching on (the default), OneTrainer moves the text encoder to
the CPU for the whole training phase — the run's VRAM budget fits *because* it is off the card. So at
a recaption boundary the encoder usually is not resident, and bringing back ~9GB of bf16 weights
needs room that a full card does not have.

That is handled by **the denoiser stepping aside**. At an epoch boundary the denoiser is idle — no
step in flight, no gradients — so when neither the resident encoder nor even the quantized captioner
fits beside it, it moves to system RAM for the duration, the text encoder comes in, the captions are
written, and both go back where the training loop keeps them. Two PCIe copies (a few seconds) buys
captioning with the run's own full-precision encoder on cards where nothing else would fit at all.
The console narrates each move.

The borrow falls back to the quantized separate load whenever the resident encoder is not safe to
use: it is being trained (captions written through a half-trained LoRA would be shaped by weights
still moving), or it is stored in fp8 (the vision tower cannot run in it).

### Everything else

For every other model family there is no Qwen anywhere in the run, so a separate captioner has to
load, and quantization is what makes that affordable.

Decoding is **sampled** (temperature 0.5), not greedy, so a second look at the same image produces a
fresh phrasing rather than the identical caption. Because sampling draws from the global torch RNG,
the RNG state is saved and restored around every call — a run that recaptions follows the same noise
path as one that does not, so your seed still reproduces your model.

### The rest of it

* The caption that was replaced is kept as `<image>.txt.orig` — from the *first* rewrite, so a hand
  written caption is never lost to a second machine attempt.
* **Recaption Trigger Word**, when set, is appended at the **end**. A trailing token is a much weaker
  identity claim than a leading one, which is what you want for an image the model is already
  struggling to reconcile.
* The captioner is loaded at the epoch boundary and freed immediately — the training model is still
  resident and the VRAM is not ours to keep.
* Whenever any caption changes, `<cache>/text` is deleted so the new captions are re-encoded next
  epoch. The image latents — the expensive half of the cache — are left alone.

## Warming up the images that look unlike the rest

Some images are genuine and worth keeping but simply look different: a tight angle, a profile, heavy
occlusion. They pull hardest exactly while the identity is still forming. **Warm Up Look Outliers**
eases them in at **Outlier Start LR**, ramping to full over **Outlier Ramp Epochs**, and releases
them early the moment they start improving on their own.

This is the *prior* — it covers the epochs before the loss watch has a trend to act on, and the watch
takes over from there.

Every image is embedded with CLIP and compared against the **median** embedding of the set rather than
the mean, so a handful of oddities cannot drag the reference towards themselves and make the normal
images look like the outliers. The cutoff comes from the spread of the scores themselves, using the
median absolute deviation for the same reason.

Two refusals are built in:

* If the cutoff would take **more than half** the set, the dataset is simply varied — there is no core
  for the odd ones to be odd against — and the warm-up is switched off with a note.
* Fewer than three images says nothing about any of them.

Scores are cached to a `look_scores.json` beside the images, so this runs once per dataset. A
`fizgig_look_scores.json` written by Fizgig's Look Filter is read too, so a dataset already scored
there needs no rescoring.

## The Problem Images window

**Tools → Problem Images** on the training tab opens a live view of what curation has decided:
thumbnail, verdict, what the verdict means, the residual and trend, and the multiplier the image is
currently training at — worst first.

Each row's caption is editable **while the run continues**. Saving writes the caption and tells the
trainer to give that image a fresh start at the next epoch boundary; nothing needs restarting, and a
hand edit also hands the image its recaption attempts back, because your caption outranks the
machine's verdict.

The window and the training loop never share an object. The trainer writes `problem_images.json`
atomically at each boundary and the window polls it; the window leaves edited keys in
`caption_edits.json` and the trainer consumes it at the next one. A slow disk or a closed window can
never block training, and a caption you are halfway through typing survives the report being rewritten
underneath you.

## What is different here

Adaptations from Fizgig's original:

* **Any batch size.** Fizgig trains one image per step, so it reads the batch's mean loss and turns
  per-image learning rates off when the batch is larger. OneTrainer hands the watch the *unreduced*
  per-sample losses, so every image in a batch is measured and weighted individually whatever the
  batch size.
* **Per-directory exclusions.** A OneTrainer run can train several concepts at once, so each
  directory keeps its own `excluded_images.json` rather than there being one per run.
* **Captions beside the image.** Found as `<image>.txt`, matching how OneTrainer's data pipeline
  resolves them.
* **CLIP for the look score.** Fizgig scores look consistency with a face embedding, which restricts
  it to datasets of people. CLIP is already an OneTrainer dependency (the aesthetic scorer uses it) and
  says something useful about a style or an object dataset too.
* **OneTrainer's own captioners.** Recaptioning goes through the same models the captioning tool
  offers rather than bundling a second stack.

## Cost

The watch itself is negligible: a few thousand `(key, epoch, bucket, loss)` tuples in memory and a
pass at each epoch boundary. The per-step cost is reading a tensor that has already been computed.

The two acting features cost something, but each only once:

* **Look scoring** is one CLIP forward pass per image, at the first epoch boundary, cached to disk
  afterwards.
* **Recaptioning** loads a captioning model at a boundary where stuck images were confirmed and frees
  it again, then clears the text cache so those captions are re-encoded next epoch. The image latents
  are untouched.
