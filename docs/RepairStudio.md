# Repair Studio

A LoRA is not one thing. It is a few hundred small weight changes spread across a model's blocks, and
they do not all do the same job — early blocks tend to carry composition and pose, late ones carry
texture and detail, and a LoRA that is *nearly* right is usually right in most blocks and wrong in a
few. The **Repair Studio** tab gives you a slider per block so you can turn the wrong ones down, and
saves the result as an ordinary `.safetensors`.

Ported from [Fizgig](https://github.com/shootthesound/Fizgig) by Peter Neill (Apache 2.0).

## Opening a LoRA

Browse to any `.safetensors` LoRA. It does not have to be one OneTrainer trained — the block layout is
read out of the key names, so kohya-ss, ai-toolkit and ComfyUI files open the same way, as do all four
of OneTrainer's own LoRA formats.

The sliders are built from whatever the file turns out to contain. A Flux LoRA gets 19 double and 38
single blocks; an SDXL one gets input/middle/output; a flat DiT stack gets one slider per block. Any
key that does not resolve to a block still gets a slider, grouped under **Other** — a key with no
slider would be one that kept contributing after you thought you had removed it.

Browsing to a different LoRA **auto-swaps**: settings for blocks the new file also has are kept, so
flipping between two checkpoints of the same run does not cost you your edits.

## The sliders

Each block has a slider from -3 to +3 and four quick-sets:

| Button | Does |
| --- | --- |
| **0** | Takes the block out. Its keys are *dropped* from the saved file, not written as zeros. |
| **1** | Back to the strength it was trained at. |
| **±** | Flips the sign, so the block subtracts what it used to add. |
| **⚖** | Balance — see below. |

The same four buttons at the top apply to every block at once.

Negative is meaningful, not a mistake: inverting a block that is overcooking a feature is a real
repair. Past about ±3 a block stops being a contribution and starts being the whole image, which is
where the range ends.

## Blending in a donor

Load a second LoRA as a **donor** and every block gets a second slider. The saved file contains both
contributions, mixed per block — so you can take composition from one LoRA and detail from another.

**⚖ Balance** holds primary + donor at 1.0 for that block: move one side and the other takes up the
slack. The block's total contribution stays put and only its *source* shifts, which is what you want
when cross-fading two LoRAs rather than stacking them.

The donor has to have been trained on the same base model. If it shares no layers with the primary, or
a shared layer is a different shape, the tab says so and refuses it rather than producing a file that
is quietly two unrelated LoRAs stapled together.

## What "baked" means

The **Save baked LoRA** button writes a normal `.safetensors`. Nothing downstream needs to know it was
edited: load it at strength 1.0 in ComfyUI, or anywhere else, and you get exactly what the sliders
said.

That works because a LoRA module contributes `scale · up @ down`, where `scale = alpha / rank`. Setting
a slider to *m* means contributing `m · scale · up @ down` instead, which is baked by folding
`m · scale` into `up` and writing `alpha = rank` — making the new file's own scale exactly 1.0.

A donor blend uses **rank concatenation**:

```
up   = cat([up_p · m_p · scale_p,  up_d · m_d · scale_d], dim=-1)
down = cat([down_p,                down_d],               dim=0)
```

Multiply those out and you get `m_p·scale_p·up_p@down_p + m_d·scale_d·up_d@down_d` — the sum of the two
contributions, with no SVD and nothing approximated. The cost is a file whose rank for that block is
the sum of the two, which is why a donor sitting at zero is dropped rather than concatenated as a
block of zeros.

None of this is lossy, and the tests check it by measurement rather than by argument: they multiply the
baked matrices back out and compare against the weighted sum they replaced.

## Presets

**Save preset** writes the slider positions to a JSON file; **Load preset** reads one back. A preset
from a different model is filtered down to the blocks the open LoRA actually has, so it cannot populate
the tab with sliders that control nothing.

The settings are also written into the saved LoRA's own metadata under `ot_repair_studio`, so a file
you made six months ago can tell you what you did to it.

## What is carried across, and what is not

* **Kept:** bundled embeddings and anything else that was never part of the adapter, plus the source's
  metadata.
* **Dropped:** `sshs_model_hash`, `sshs_legacy_hash` and `modelspec.hash_sha256` — those describe the
  file the LoRA came *from*, and inheriting them would leave every repaired LoRA claiming a content
  hash that is no longer its own.
* **Rewritten:** `ss_network_dim` and `ss_network_alpha`, because a donor blend changes both.

## LyCORIS: LoKR and LoHa edit exactly, in native format

Both LyCORIS forms are *linear in their first factor*:

```
LoKR:  kron(m·w1, w2)      = m · kron(w1, w2)
LoHa:  (m·W1) ∘ W2         = m · (W1 ∘ W2)
```

So a per-block multiplier folds into `lokr_w1` (or `lokr_w1_a` / `hada_w1_a`) the same way the
standard bake folds into `lora_up` — exactly, with no SVD and no format conversion. A module you
did not touch comes out **byte-identical**, alpha included, and OneTrainer's own LoKr exports open
like any other LoRA.

One deliberate difference from Fizgig: alpha is left untouched rather than replaced with a
"scale already baked" sentinel value. The sentinel needs the loader to know the convention; with
alpha as it was, anything that read the original correctly reads the edit correctly.

Three things stay refused, each with a message saying why:

* **Donor-blending a LoKR/LoHa block** — rank concatenation needs the two-matrix form, so an exact
  blend does not exist. Set one side of the block to zero, or blend standard-LoRA exports.
* **Tucker-decomposed modules** (`lokr_t2`) — the core tensor breaks the linearity the exact bake
  relies on.
* **Partial strengths on DoRA blocks** — a DoRA delta is renormalised at load time, so scaling its
  matrices does not scale its effect; the slider would lie. `[0]` and `[1]` still work.

## Limits

* **No preview yet.** Fizgig shows a live side-by-side render as you drag. That needs a base model
  resident in VRAM and OneTrainer supports a lot more architectures than Fizgig does, so it is a
  separate piece of work. For now the workflow is: edit, bake, look at it in your sampler of choice.
