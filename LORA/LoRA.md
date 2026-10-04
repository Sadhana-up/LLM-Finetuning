# LoRA — Low Rank Adaptation

## 1. The problem LoRA solves

Fine-tuning a big model means changing **all** its weights.

If a layer has a weight matrix `W` of size `d x k`, full fine-tuning stores and updates every one of those `d * k` numbers.

For a 7B model (~7 billion parameters) that means:

- 7B trainable weights
- A separate optimizer state (Adam keeps 2 more numbers per weight) → 3x the memory
- Activations for every layer → the activations dominate memory too

Most of those 7B numbers barely change during a fine-tune. LoRA's observation:

> The **update** `ΔW` that fine-tuning produces is almost always close to a low-rank matrix.

So we throw away 99% of the trainable weights on purpose.

---

## 2. What "rank" means

Rank = how many linearly independent columns (or rows) a matrix has.

- A 4096 x 4096 matrix could have rank 4096 (full rank) — all columns are independent.
- If it has rank 8, those 4096 columns can be written as **linear combinations of just 8 columns**.

So a low-rank matrix is not a matrix with few entries. It is a matrix whose content is made from a few "ingredients".

That is why the parameter count drops dramatically even though the *output shape* stays the same.

---

## 3. The formula

Frozen original weight: `W0` of size `d x k`.

Normally fine-tuning computes `W = W0 + ΔW`, where `ΔW` is also `d x k` and every entry of it is trainable.

LoRA replaces the big trainable `ΔW` with two thin matrices:

```
ΔW = B · A          where  A is d x r   and   B is r x k
```

`r` is the **rank**. It is a small number (1, 2, 4, 8, 16, 32, 64), always much smaller than `d` and `k`.

```
W  =  W0 + B·A
     d x k   r x k · d x r
```

The shapes line up: `(r x k) · (d x r)` gives a `d x k` matrix. Only `A` and `B` are trained. `W0` is frozen.

### Parameter count

Weights in the LoRA matrices: `d * r + r * k = r(d + k)`.
Weights in the full matrix: `d * k`.

Example with `d = k = 4096`:

| what | r = 8 | full fine-tune |
|---|---|---|
| trainable weights | 8 × (4096 + 4096) = 65,536 | 4096 × 4096 = 16,777,216 |
| ratio | **256x fewer** | 1x |

The saving factor is `d*k / (r*(d+k))`. For `d = k = 4096` this becomes `4096 / (2r)`, so `r = 8` → 256x and `r = 16` → 128x.

For an 8x smaller layer the saving is smaller. LoRA is therefore always applied to the **large** matrices (attention projections), never to tiny ones.

### Which matrix is "rank r"?

Both of them. `A` has `r` columns, `B` has `r` rows. The product `B·A` therefore has rank at most `r` no matter how large `d` and `k` are. That is the whole trick — we hard-force the update to be low rank.

---

## 4. The forward pass

Standard linear layer: `y = W · x`

LoRA linear layer:

```
y = W0 · x  +  B · A · x
```

Read it as: `x -> A -> B` is the small trainable path, `x -> W0` is the big frozen path. Both are added. Nothing else about the layer changes, so LoRA can be dropped into any `nn.Linear`.

`h = A·x` (r numbers), then `y = W0·x + B·h`.

### Merging for inference

At deployment you do not need to keep the two extra matrices. Multiply them once:

```
W_merged = W0 + B·A
```

`W_merged` is the same size as `W0`, so the served model has **zero extra latency**. This is a big practical advantage of LoRA over adapters.

### Initialization

- `A` ~ random (Gaussian), `B` = 0.
- So at step 0, `B·A = 0` and the model behaves exactly like the frozen base model. Training starts from the original behaviour and then moves away from it.

### The `alpha` scaling

`W = W0 + (alpha / r) · B·A`

Scaling by `alpha / r` keeps the effective magnitude of the update comparable when you change `r`. Common: `r = 8` with `alpha = 16` (ratio 2), or `r = 16` with `alpha = 32`. When comparing across ranks, people often keep `alpha/r` constant. Too large a scale makes training unstable.

---

## 5. Which modules get LoRA

The original paper applied it to the **attention query and value** projection matrices only — the cheapest choice that already worked well:

```
q_proj, v_proj   (also k_proj, o_proj in practice)
```

Most implementations today wrap **every** large linear layer:

- in attention: `q_proj, k_proj, v_proj, o_proj`
- in the MLP/FFN: `gate_proj, up_proj, down_proj`

Typical config: `r = 16`, `alpha = 32`, `dropout = 0.05`, only these modules trainable (~0.1–1% of total params).

---

## 6. Why it works so well

- **Fewer params to overfit.** Small datasets + millions of trainable weights = memorisation. LoRA's implicit regularisation (the rank cap) is a feature, not a compromise.
- **Speed.** Backprop does not need gradients for `W0`, only for `A` and `B`. Less memory traffic, larger batch.
- **One adapter per task.** Each LoRA adapter is ~1–10 MB. You keep the frozen 7B base on disk once and swap a tiny adapter file per task. (Adapters, in the classic sense, are 10–100x bigger.)
- **Multiple adapters at once.** Separate `B·A` per task, sum them at inference to build a merged model.
- **No extra inference cost** because of merging.

Weakness: with very large rank settings on small tasks, LoRA can underperform a full fine-tune, because the update genuinely is high-rank there. And adapter quality depends on picking good target modules.

---

## 7. QLoRA — LoRA on a quantized base

LoRA already says "I only train a small part of the weights". QLoRA goes further: **freeze the base weights in 4-bit** (NF4, blocksize 64, double quantisation of the constants) and train the LoRA matrices in bf16.

- Base model memory drops ~4x (a 7B model goes from ~14 GB to ~4 GB), so it fits on one consumer GPU.
- The frozen 4-bit base is never backpropagated through, so quantisation noise barely hurts.
- Accuracy is very close to bf16 LoRA.
- The trick that makes it work: `bnb.matmul_4bit` computes in bf16/fp16 by dequantizing on the fly in the kernel, so the `y = W0·x + B·A·x` math is unchanged numerically.

Practical note: use `prepare_model_for_kbit_training` and a `bf16` compute dtype; do not quantize the LoRA layers themselves.

---

## 8. Phase two — adapters

"Adapter" in the original literature (Houlsby et al., 2019) means: insert small trainable **bottleneck modules** *between* the frozen layers of a transformer block.

```
x -> LN -> MultiHeadAttention -> +
  -> LN -> FFN -> +
        |
     [down-project d->r -> ReLU -> up-project r->d]
```

- Two inserted linear layers per sub-layer, with a nonlinearity between them.
- Trained together with a few layer norms and biases.
- Cannot be merged into the surrounding matmuls (the ReLU sits in the middle), so they add real latency at inference.

### LoRA vs adapters

| | LoRA | Adapters |
|---|---|---|
| where | reparameterizes existing weight matrices | inserts new modules between layers |
| rank idea | yes, `ΔW = B·A` | bottleneck width `r`, but with nonlinearity |
| per task | ~0.1–1% of params | ~3–10% of params |
| merging | possible → free inference | not possible → slower inference |
| parallel to base | yes, weight-level | yes, architecture-level |

### Other low-parameter fine-tuning families

- **Prefix / prompt tuning** — prepend trainable virtual tokens to the input or every attention key/value. No new matrices; just embeddings. Works on frozen models.
- **(IA)³** — train only rescaling vectors for keys, values, and FFN.
- **BitFit** — train only the biases (~0.1% of params).
- **Adapters + parallel training** (LST) — train the adapter while the frozen backbone runs ahead of it in a separate stream, which removes the activation-memory cost.

The common thread: freeze the big weights, train a tiny number of extra parameters, and spend the saved memory on larger batches or longer context.

---

## 9. Minimal implementation

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        h = self.drop(x) @ self.A.T
        return out + (h @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self):
        self.base.weight += (self.B @ self.A) * self.scale
```

Use it:

```python
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear) and name.split(".")[-1] in target:
        path = name.split(".")
        parent = model.get_submodule(".".join(path[:-1]))
        setattr(parent, path[-1], LoRALinear(module))

for m in model.modules():
    if isinstance(m, LoRALinear):
        m.A.requires_grad = True
        m.B.requires_grad = True

model.print_trainable_parameters()
```

`model.print_trainable_parameters()` prints something like
`trainable params: 3,932,160 || all params: 3,303,196,416 || 0.12%` — usually well under 1%.

---

## 10. Rules of thumb

- Start with `r = 16`, `alpha = 32`, `dropout = 0.05`, lr `1e-4` to `2e-4`.
- Target `q_proj, v_proj` first (paper-faithful); add the rest if underfitting.
- Larger `r` only helps when the task genuinely needs more rank.
- Always merge before deploying.
- Validate on a held-out set — low train loss with a high `r` is usually memorisation.
- Save only the adapter weights (`d*r + r*k` per layer), never the base.

---

## TL;DR

Freeze `W0 (d x k)`. Learn a rank-`r` update as `ΔW = B·A`, where `A` is `d x r` and `B` is `r x k`. Because `r` is tiny, you train `r(d+k)` parameters instead of `d·k` — often 100x+ fewer. Forward pass is `W0·x + B·(A·x)`, and at inference you collapse it into `W0 + B·A` so there is no speed cost. The earlier "adapter" approach instead inserted small bottleneck modules between layers, which needs more parameters and cannot be merged. QLoRA combines LoRA with a 4-bit frozen base so the whole thing fits on a single GPU.