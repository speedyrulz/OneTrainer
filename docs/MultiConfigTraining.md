# Multi-Config Training

Everything for this lives on the **Multi Config** tab.

Multi-config training races several sets of training settings against each other and keeps whichever
one is winning, re-deciding at every validation interval.

Instead of guessing a learning rate, optimizer or loss weighting up front and finding out 8 hours
later that it was wrong, you hand OneTrainer a handful of candidates. It trains all of them over the
same stretch of training, compares their validation loss, and continues from the best one. Then it
does it again for the next stretch. The settings that work best early are not always the ones that
work best later, and this follows that instead of fighting it.

There are three modes:

* **Full configs** — race up to 5 saved config files, which may differ in many settings at once.
  Good for comparing whole recipes.
* **Single setting** — race up to 10 values of one setting, taking everything else from the training
  page. Good for answering one question at a time: which learning rate, which optimizer, which
  timestep distribution.
* **Adaptive learning rate** — race one learning rate against its two neighbours every round and
  follow the winner, so the learning rate moves up and down during the run instead of being decided
  once.

Whichever mode you use, the model written to your output destination at the end is the one with the
lowest validation loss of the whole run — see [The model you get](#the-model-you-get).

## How a round works

One **round** covers exactly one validation interval.

1. Every config starts from the same training state — the same weights, the same optimizer state, the
   same step count.
2. Config #1 trains for one validation interval, then validates. Its state is set aside.
3. The state is rewound and config #2 trains the same interval on the same batches, then validates.
   Its state is set aside too.
4. Repeat for every config.
5. The validation losses are compared. The best config wins: its state becomes the shared state, and
   the model is saved.
6. The next round starts from there, again with every config.

So over a run the winner can change from round to round. A config that wins the first three rounds and
then falls behind simply stops being picked; nothing is locked in.

Every config in a round trains the **same number of steps on the same batches** and validates on the
**same validation samples**. That is what makes the comparison mean something, and it is why the data
pipeline is rebuilt between configs.

## Setting it up

Both modes need this first:

1. **Enable validation** on the *general* tab and set **Validate after**. This interval is also the
   round length, so it controls how often the tournament re-decides. One epoch is a reasonable start.
2. **Give it something to validate against.** Either mark a concept as a validation concept on the
   *concepts* tab, or set **Auto Validation Split %** on the *general* tab to hold part of every
   concept back — see [Getting a validation set](#getting-a-validation-set).
3. Set up everything else on the *training* page as you normally would. Whatever the tournament does
   not vary comes from there.

### Single setting

4. On the **Multi Config** tab, switch **Multi-Config Training** on and set **Mode** to *Single
   setting*.
5. Pick the **Setting** to vary and how many values to compare under **Number of Values** (1–10; two
   or more are needed to have anything to compare).
6. Fill in the boxes that appear — dropdowns for everything except Learning Rate, which is typed in.
   Boxes are pre-filled with sensible starting values, and are re-filled when you change setting.
7. Start training as usual.

### Adaptive learning rate

4. On the **Multi Config** tab, switch **Multi-Config Training** on and set **Mode** to *Adaptive
   learning rate*.
5. Type a **Starting Learning Rate** and pick **Number of Learning Rates** (3–10). The preview
   underneath shows what the first round will compare.
6. Start training as usual. The learning rate on the training page is ignored in this mode.

### Full configs

4. **Create your config sets.** Set up the settings you want to try on the training page, then use
   **Save config** in the top bar. Repeat for each variant. Only vary the settings listed under
   [What a config set may change](#what-a-config-set-may-change) — everything else is ignored.
5. On the **Multi Config** tab, switch **Multi-Config Training** on, leave **Mode** on *Full configs*,
   pick **Number of Configs** (1–10), and point the boxes at your saved files. Fill at least two;
   empty boxes are skipped, and slots past the selected number are ignored.
6. Start training as usual.

The config currently open in the UI is the **base config**. It supplies the model, the dataset, the
workspace, the epoch count, the validation interval and every other structural setting. The config
sets only contribute the settings they are allowed to vary. If you want your current settings to be
one of the contenders, save them to a file and select that file.

## Settings

| Setting | What it does |
| --- | --- |
| **Multi-Config Training** | Turns the tournament on. |
| **Mode** | *Full configs* races saved config files; *Single setting* races values of one setting. |
| **Number of Configs** | *Full configs* only. How many config boxes to show, 1–10. |
| **Config Set 1…10** | *Full configs* only. Saved config files to race. At least 2 must be filled. |
| **Setting** | *Single setting* only. Which training setting to vary. |
| **Number of Values** | *Single setting* only. How many values to compare, 1–10. |
| **Starting Learning Rate** | *Adaptive learning rate* only. Where the ladder starts. |
| **Number of Learning Rates** | *Adaptive learning rate* only. How wide the window is, 3–10. |
| **End Early** / **Rounds Without Improvement** | Stop once that many rounds in a row fail to beat the best validation loss so far. Off by default, 2 rounds. |
| **Drop Losing Configs** / **Check Every** / **Drop Rule** | Every so many rounds, narrow the field. Off by default. See below. |
| **Selection Metric** | How validation losses are reduced to one number. See below. |
| **State Storage** | `DISK` (default) writes candidate states under `<workspace>/multi_config/state`. `RAM` is faster but needs enough free system memory for one copy of the trainable weights and optimizer state per config. |
| **Save Each Round** | Save the winning model after every round, on top of the final save. On by default. |
| **Deterministic Data** | Rebuild the data pipeline per config so all configs see identical batches. On by default; see [Cost](#cost). |

### Settings a single-setting tournament can vary

| Setting | Values | Notes |
| --- | --- | --- |
| **Optimizer** | Every optimizer OneTrainer supports | Each value also brings that optimizer's own parameters, including any you saved for it in the optimizer settings window. Optimizer state cannot carry across a change of optimizer, so a config that switches optimizer starts that round with fresh optimizer state — prefer a longer validation interval so the warm-up is a smaller share of the segment. |
| **Learning Rate Scheduler** | Every schedule except `CUSTOM` | `CUSTOM` needs a class name and parameters, which cannot come from a dropdown. |
| **Learning Rate** | Typed, e.g. `1e-4` | |
| **Timestep Distribution** | Every distribution | The distribution's own weight and bias values come from the training page and stay the same for every candidate. |
| **Layer Filter** | The presets your model type offers | Behaves differently — see below. |

Adding another setting to this list is a small change: one member in `MultiConfigSweepSetting` and
one entry in `SWEEP_SETTINGS` in `modules/util/multi_config_sweep.py`. The UI builds its inputs from
that registry, so nothing in the interface needs touching.

### Layer Filter is a special case

Every other setting leaves the trained weights untouched, so candidates can hand their progress to
one another. A layer filter decides *which* weights exist, so they cannot: an attention-only candidate
has nowhere to put the layers a full-model winner trained, and OneTrainer's placeholder modules hold
such weights without applying them to the model.

Layer filter tournaments therefore run as **independent lineages**. Every value trains its own model
from start to finish, and the rounds only report which one is ahead and save it. You still get the
answer you were after — which part of the model is worth training — but the run does not converge on
one lineage the way the other settings do, and it costs one full training run per value.

Because of that, layer filter tournaments require LoRA training with **EMA off** and **no embedding
training**: both keep trainable state outside the adapter that carries a lineage forward. Training
refuses to start with a clear message otherwise.

## The adaptive learning rate ladder

The ladder walks the learning rate in single steps of its leading digit, rolling over at each decade:

```
... 0.00008  0.00009  0.0001  0.0002  ...  0.0008  0.0009  0.001  0.002 ...
```

Every round trains a window of consecutive rungs — 3 to 10 of them, set by **Number of Learning
Rates** — centred on the rate the last round settled on.

**How far the window moves** depends on where in it the winner sat. The shift is the winner's
distance from the middle, so a narrow win nudges the window by one rung and a win at the edge moves
it far enough that the winner is no longer at the edge next round:

| Window | Round compares | Winner | Next round |
| --- | --- | --- | --- |
| 3 | 0.0002 / **0.0003** / 0.0004 | 0.0004 (one above the middle) | 0.0003 – 0.0005 |
| 5 | 0.0001 – 0.0005, middle 0.0003 | 0.0004 (one above) | 0.0002 – 0.0006 |
| 5 | 0.0001 – 0.0005, middle 0.0003 | 0.0001 (two below) | 0.00008 – 0.0003 |

With an **odd** window there is a true middle rung, and a win there leaves the ladder exactly where
it is. With an **even** window there is no middle: a win anywhere in the lower half moves the window
down and anywhere in the upper half moves it up, again scaled by distance, so it always moves by at
least one rung.

A wider window finds a far-off learning rate in fewer rounds, at the cost of a full segment of
training per extra rate. There is nothing stopping the window walking back and forth as training
progresses — that is the point, since the best learning rate early is rarely the best learning rate
late.

The ladder runs from `1e-12` up to `9.0`. The ceiling is that high because Prodigy and the
D-Adaptation optimizers are used at a learning rate of `1.0` by convention. At either end the window
slides inwards rather than shrinking, so a round always compares the number of rates you asked for.

A starting rate that is not a single leading digit is snapped to the nearest rung — `0.00034` becomes
`0.0003` — and both the preview and the console say so.

## Getting a validation set

The tournament needs validation data to compare anything. There are two ways to give it some.

**Mark concepts as validation concepts** on the *concepts* tab. Those concepts are never trained on;
they exist only to be validated against. Use this when you have images set aside for the purpose.

**Or set Auto Validation Split %** on the *general* tab. OneTrainer then holds back that share of
every concept and validates on it, leaving the rest to train on. So a 10% split over a 200-image
concept and a 20-image one holds back 20 and 2 — the percentage is taken from **each concept
separately**, not from the dataset as a whole, so a small concept is still represented.

Details worth knowing:

* Which images are held back is random but **fixed**: it is worked out from a hash of the concept, so
  the same images are held back on every run and on every restart. Nothing leaks between the two
  sides and nothing goes missing from both.
* Every concept with at least 2 images contributes **at least one** validation image, and never all
  of them. A concept with a single image is left in training and contributes nothing.
* The validation copies have **random augmentation turned off** — flips, jitter, rotation, colour and
  tag shuffling — so the same validation image looks the same on every pass and the loss curve
  measures the model rather than the draw.
* Concepts already marked as validation concepts are **not split**; they keep all of their images.
* **Clear the cache after changing the percentage.** Cached latents are grouped per concept, and
  changing which images are on which side invalidates them.
* This applies to the image/text data loaders. Fine Tune VAE has its own path and is unaffected.

The split works whether or not multi-config training is on — it is just a way of producing a
validation set, and the ordinary validation loss curve uses it too.

## Ending early

**End Early** stops the run once **Rounds Without Improvement** rounds in a row fail to reach a lower
validation loss than the best one seen so far. The default is 2. Since the winner of a round is the
best of its candidates, that means no setting in any of those rounds improved on anything seen up to
that point.

It works in all three modes. It is off by default, because a run that has plateaued sometimes escapes
later — especially with the adaptive ladder, which may need a round or two of no progress before it
finds a better rate. Turn it on when you would rather stop than pay for that.

Ending early costs nothing in model quality: the model you get is the best one either way.

## Dropping configs

**Drop Losing Configs** watches the last **Check Every** rounds and narrows the field at the end of
each such window. A config that keeps losing still costs a full segment of training per round for a
result the run will never use.

**Drop Rule** decides what goes:

* **Never won** — drop every config that won none of the rounds in the window. With 6 configs and a
  check every 5 rounds: after 5 rounds, if only 3 of them won anything, the other 3 go and the run
  continues with those 3. Clears the field quickly, but it cannot tell the difference between a
  config that was consistently second by a hair and one that was nowhere near.
* **Worst average loss** — drop the single config with the highest average validation loss **across
  every round it has run so far**, not just the rounds since the last check. Ranks on how far behind
  a config actually is rather than on whether it happened to come first, so a close second survives
  and a genuine no-hoper goes, and one good round late on cannot rescue a config that has been far
  behind all run. Only one is dropped per check: once the worst is gone the remaining averages say
  nothing new until they have been measured against each other again. A config that produced no
  usable validation loss at all counts as worse than any that did, and goes first.

  The console names the config, its average and how many rounds that average covers:

  ```
  [multi-config] dropping divergent with the worst average loss 4.318702 over 6 round(s) | continuing with slow, good
  ```

Either way the check repeats over the survivors, and **the last config standing is never dropped**.
Once one is left the tournament is effectively ordinary training, and it carries on that way until
the epochs run out or End Early stops it.

Both rules apply to *Full configs* and *Single setting*. Neither does anything in *Adaptive learning
rate*, where every round compares a fresh window of rates and past performance says nothing about any
of them.


## The model you get

Whenever a round beats the best validation loss so far, its state is set aside. At the end of the run
— whether it finished its epochs, ended early, or you pressed Stop — that state is put back before the
final model is written to your output destination.

So the model in the output folder is always the one with the lowest validation loss the tournament
saw, even if that was hours before the run stopped. The console says which round it came from:

```
[multi-config] restoring the best model: 0.0004 from round 7, validation loss 0.041273
```

If the last round *was* the best one, nothing is restored and the run's final state is saved as-is.

This is independent of **Save Each Round** — that setting controls the extra per-round files in
`<workspace>/save`, not the final output.

### Selection Metric

* `TOTAL_AVERAGE` — the average loss over every validation sample. A concept with more samples counts
  for more. This is the default and usually what you want.
* `MEAN_PER_CONCEPT` — the unweighted mean of the per-concept averages, so a 5-image concept counts as
  much as a 500-image one.
* `WORST_CONCEPT` — the highest per-concept average. Picks the settings that leave no concept behind.

Lower is better in all three. An exact tie goes to the earlier config, so the run does not switch
settings on noise.

## What a config set may change

This section applies to *Full configs* mode. Single-setting mode varies exactly the one setting you
picked, so there is nothing here to worry about.

A config set may differ from the base on:

* **Learning rate and schedule** — learning rate, scheduler and its parameters, warmup steps, cycles,
  min factor, learning rate scaler, gradient clipping
* **Per-model-part learning rates** — the unet / transformer / text encoder / VAE learning rate
  overrides
* **Optimizer** — the optimizer itself and all of its parameters
* **Loss** — MSE / MAE / log-cosh / Huber strengths, Huber delta, VB loss, loss weight function and
  strength, loss scaler
* **Noise and timestep sampling** — offset noise, perturbation noise, timestep distribution, noising
  strengths, noising weight and bias, timestep shift
* **Masked training weights** — unmasked probability and weight, masked area normalisation, masked
  prior preservation weight
* **Embedding learning rate** and norm preservation
* **EMA decay** and update interval

Everything else — model type, training method, base model, LoRA rank, quantisation, dtypes, batch
size, accumulation steps, resolution, dataset, epoch count, validation interval, workspace — comes
from the base config. All configs share one loaded model and one latent cache, so those cannot vary
without reloading the model between every config, which would cost more than the tournament saves.

A difference on a structural setting is **not an error**. It is reported once at startup, in the
console and in `<workspace>/multi_config/candidates.json`, and the base config's value is used. That
way a config saved on another machine or for another project can still be used for its learning rate
and optimizer.

### Changing the optimizer between configs

Allowed, with one caveat: optimizer state cannot carry across a change of optimizer type. When a
config uses a different optimizer than the state it inherits, it starts with fresh optimizer state and
a line is printed saying so. That is a real disadvantage in the round where it happens — a config
switching from Adam to Prodigy is judged partly on its warm-up. If you are comparing optimizers,
prefer a longer validation interval so the warm-up is a smaller share of the segment.

## Output

Under `<workspace>/multi_config/`:

* `candidates.json` — the configs that were loaded, their key settings, and any structural fields that
  were overruled by the base config.
* `results.json` — one entry per completed round: every config's score, per-concept validation losses,
  steps trained, and who won. Written after each round.
* `summary.json` — the same history plus a win count per config, the best round and its score, and
  (in adaptive mode) the learning rate the run finished on. Written when training ends.
* `state/` — the training states, if **State Storage** is `DISK`. Overwritten every round, so the size
  is bounded by (configs + 2) states — one per config, one for the round winner, one for the best
  round of the whole run.

In tensorboard:

* `loss/multi_config/<config name>/<concept>` — each config's validation loss per concept, so you can
  see the whole race.
* `loss/multi_config/winner` — the winning score per round.
* `loss/validation_step/...` — the winner's losses, logged under the usual series so the normal
  validation curve stays continuous.

Models saved per round are named `<prefix>round003-<config name>-<timestamp>-save-<progress>`, so it is
clear which settings produced each file.

## Cost

The honest trade-offs:

* **Training time scales with the number of configs.** Three configs means roughly 3× the training
  time for the same number of promoted steps. You are buying a settings search with that time. This
  is the main reason to think before picking 10 values.
* **Validation runs once per config per round** instead of once per round.
* **Sampling and scheduled saves/backups run only on the winner**, once per round, not once per
  config. Manual *sample now*, *save now* and *backup now* still work at any time.
* **Deterministic Data rebuilds the data pipeline** once per config per round. On a large dataset that
  rebuild is noticeable. Turning it off skips the rebuild when a round starts on an epoch boundary, at
  the cost of comparing configs on different shuffles of the same data. It is ignored when a round
  starts mid-epoch, where reusing the pipeline would put the data out of step with the recorded
  progress.
* **State snapshots** are cheap for LoRA (tens of MB per config) and expensive for full fine-tunes,
  where each snapshot holds a copy of the trainable weights and optimizer state. Use `DISK` storage
  for fine-tunes.

## Stopping and resuming

Pressing **Stop** finishes cleanly:

* If some configs in the current round already completed their segment, the best of those is promoted
  and saved.
* If the stop lands during the first config of a round, that partial segment is discarded and the
  model is rewound to the last promoted state — the last one that actually won a comparison.

Backups and *continue from last backup* work as usual; a resumed run restarts the tournament from the
restored state.

## Not supported

* **Multi-GPU** training
* **Cloud** training
* **Only Cache**

Training refuses to start with a clear message if any of these, or a missing/duplicate config file, or
disabled validation, is detected.

## Command line

Multi-config training is a config setting, so `scripts/train.py` picks it up from an exported config
with no extra flags:

```bash
python scripts/train.py --config-path my_config.json
```

Make sure the `multi_config_path_*` entries in that config point at files that exist on the machine
running the training. A single-setting tournament needs no external files at all — the values live in
`multi_config_sweep_value_*` inside the config itself.
