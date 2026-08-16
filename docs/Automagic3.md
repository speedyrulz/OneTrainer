# Automagic3

Automagic3 sets the learning rate for you. Pick it from the **Optimizer** dropdown on the *training*
tab, give it a starting learning rate, and it adjusts from there for the rest of the run.

Ported from [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) (MIT licence). It is
experimental upstream and under active development — expect it to change.

## How it decides

Each element of each trained tensor remembers the sign of its last few updates (**Polarity History**
of them, default 8, stored 1 bit per step). Every step, each element casts one of three votes:

| Its window | What that means | Vote |
| --- | --- | --- |
| every sign the same | the step is too small, it is still travelling | **up** |
| the signs perfectly alternate | the step is too large, it is hopping over the minimum | **down** |
| anything else | noise | abstain |

The two decisive states are exact mirrors and equally likely under pure noise, so they balance
without any correction factor. Votes are weighted by update size, so the elements actually moving the
weights dominate.

Those votes are pooled across **the whole parameter group** and the group's single learning rate is
nudged once per step by the result. One rate for the group, not one per tensor — that is the point of
v3. Coupled tensors fight per-tensor rates: with Q and K, scaling Q up while scaling K down leaves
the attention logits unchanged, so each votes to move in opposite directions and both run away. One
shared rate makes those votes cancel, and only agreement about the *group's* step size moves it.

A tensor abstains until its window has filled, so the first few steps are a warmup.

## Settings

Everything lives in the optimizer settings window (the **…** next to the Optimizer dropdown).

| Setting | Default | What it does |
| --- | --- | --- |
| **Learning Rate** (training tab) | — | Where it starts. The controller takes over from there. |
| **Polarity History** | 8 | How many past update signs each element remembers, 2–64. Longer is steadier but slower to react, and lengthens the warmup. |
| **Minimum LR** | 1e-8 | Hard floor for the adapted rate. |
| **Maximum LR** | 1e3 | Hard ceiling for the adapted rate. |
| **Beta2** | 0.999 | Second-moment decay, as in Adafactor. |
| **Eps** | 1e-30 | Second-moment epsilon. |
| **Clip Threshold** | 1.0 | RMS clipping on the update. |
| **Weight Decay** | 0.0 | Decoupled weight decay. |

At their defaults **Minimum LR** and **Maximum LR** are decades outside the usable range and act only
as overflow guards. Tighten them if you want hard rails on how far the controller may go.

Setting a starting rate above 1e-3 prints a note: it is not clamped, but the controller will spend
its first steps walking it back down.

## Using it

* **Start low.** 1e-6 is a reasonable starting point; the controller climbs quickly when the
  direction is consistent. Starting too high wastes steps coming back down.
* **Learning rate schedulers are pointless with it.** The controller sets the rate every step, so a
  schedule just fights it. Use `CONSTANT`.
* **Per-parameter-group learning rate overrides still apply**, since each group carries its own
  adapted rate.
* **Fused Back Pass is supported.** It uses OneTrainer's own fused path rather than upstream's
  internal hooks, and the pooled vote is applied on the trainer's `zero_grad` call.
* **Tensorboard shows the adapted rate**, not the configured one — the reported learning rate comes
  from the optimizer.
* **Resuming keeps the adapted rate** and the sign histories. Changing **Polarity History** on resume
  keeps the rate but restarts the windows, so expect another warmup.

## With multi-config training

Automagic3 shows up in the **Optimizer** sweep on the [Multi Config](MultiConfigTraining.md) tab like
any other optimizer, so you can race it against AdamW or Prodigy.

Two things are worth knowing when you do:

* Optimizer state cannot carry across a change of optimizer, so a round where a config switches to
  Automagic3 starts with fresh state and pays the warmup. Use a longer validation interval so that is
  a small share of the segment.
* Racing Automagic3 against itself in the **Adaptive learning rate** mode is mostly redundant — both
  adapt the rate, so the ladder would be steering something that already steers itself. The starting
  rate still matters, so it is not meaningless, just largely duplicated effort.
