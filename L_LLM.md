# LLM Learning Notes

## Positional Embeddings

### Q: Why do positional embeddings need the same dimension as token embeddings (e.g., 768 in GPT-2)?
Positional embeddings are **added directly** to token embeddings:
```
output = token_embedding + positional_embedding
```
You can only add vectors of the same dimensionality. Additionally, each position needs to contribute information across **all 768 dimensions** of the representation space.

### Q: What is the shape of positional embeddings in GPT-2?
`(1024, 768)` — a learnable embedding table where:
- **1024** = maximum sequence length
- **768** = embedding dimension
- Each row is a unique learned 768-dimensional vector for that position

### Q: What are the two main approaches to positional embeddings?

**1. Learned Embeddings (GPT-2, GPT-3, GPT-4):**
- Shape: `(max_seq_len, embedding_dim)`
- Each position has a unique learned vector
- Better empirical performance
- Cannot extrapolate beyond training sequence length
- Requires storing `max_seq_len × embedding_dim` parameters

**2. Sinusoidal Encodings (Original Transformer, BERT, T5):**
- Mathematically derived using sine and cosine functions at different frequencies
- No learnable parameters
- Can extrapolate beyond training sequence length
- Orthogonal to semantic information (different from word embeddings)
- Theoretically cleaner but empirically underperforms

### Q: What are sinusoidal positional encodings?
Mathematical formulas that create position "fingerprints" using different frequency oscillations:

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where `pos` is position and `i` is the dimension index. Each dimension oscillates at a different frequency:
- **Early dimensions**: Very slow oscillation (changes every ~10,000 positions)
- **Middle dimensions**: Medium oscillation (changes every ~100 positions)
- **Late dimensions**: Fast oscillation (changes every ~1 position)

This creates a unique fingerprint for each position.

### Q: Why did SOTA models abandon sinusoidal encodings for learned embeddings?
Empirically, learned embeddings **perform better** despite being theoretically less elegant. Models can learn which positions matter most for their task, and the flexibility is useful during fine-tuning and adaptation to new contexts.

### Q: What is Rotary Position Embedding (RoPE)?
A modern **hybrid approach** used by LLaMA and other recent models:
- Mathematically derived like sinusoidal (better extrapolation)
- Integrates smoothly with learned patterns
- Better empirical performance than pure sinusoidal
- More efficient than learned embeddings
- Can extrapolate better than pure learned approaches

---

## Layer Normalization (LayerNorm)

### Q: What does LayerNorm do?
Normalizes activations across features for each sample independently:
- Computes mean and variance per sample across all dimensions
- Normalizes: `(x - mean) / sqrt(var + eps)`
- Helps stabilize training and enables higher learning rates

### Q: What is `elementwise_affine` in LayerNorm?

**`elementwise_affine=True` (default in GPT-2):**
```
output = gamma * (x - mean) / sqrt(var + eps) + beta
```
- `gamma`: **Learned scale** (one value per dimension, e.g., 768 values)
- `beta`: **Learned bias** (one value per dimension, e.g., 768 values)
- Total: `2 × embedding_dim` learnable parameters per LayerNorm layer

**`elementwise_affine=False`:**
```
output = (x - mean) / sqrt(var + eps)
```
- Pure normalization only
- **No learnable parameters**
- Fixed behavior

### Q: Why does LayerNorm need learnable gamma and beta (elementwise_affine)?
After normalization, values are constrained to mean 0 and standard deviation 1. This is **too restrictive** — the model needs to:
- **Re-scale** values using `gamma` for expressivity
- **Re-shift** values using `beta` to recover information
Without these, the layer can only apply fixed normalization, limiting the model's ability to learn the function it needs.

### Q: How many parameters does LayerNorm add in GPT-2?
For embedding dimension 768:
- `gamma`: 768 parameters
- `beta`: 768 parameters
- **Total: 1,536 parameters per LayerNorm layer**

(GPT-2 has LayerNorm in every transformer block, so this multiplies by the number of layers — 12 or 24 depending on model size.)

---

## Transformer Architecture Overview

### Q: Why do transformers need positional information?
Without positional embeddings, transformers can't distinguish between different word orders because attention operates on all positions equally:
- "The cat sat on the dog" would be identical to "The dog sat on the cat" (same words just reordered)
- Position information must be **explicitly encoded** into embeddings

### Q: What is the relationship between token embeddings and positional embeddings in GPT-2?
```
transformer_input = token_embedding + positional_embedding
```
- **Token embeddings**: 768-dimensional vector representing the word's semantic meaning
- **Positional embeddings**: 768-dimensional vector representing where the word is in the sequence
- The two are added together, combining both semantic and positional information

---

## Attention Mechanisms

### Q: What is the relationship between masked attention and auto-regressive training?
Masked attention **enforces auto-regressive properties during training**:

- **Without masking**: Token at position 3 could attend to position 4's embeddings → model cheats during training by looking ahead
- **With masking** (causal mask): Token at position 3 only attends to positions 0-3 → mirrors auto-regressive generation at inference

This ensures **training matches inference behavior**. If you train with masking and generate auto-regressively, the model sees consistent attention patterns in both settings.

### Q: Is masked attention always required for auto-regressive generation?
**No, but it's required for proper auto-regressive training.**

- **Auto-regressive generation**: Just sampling one token at a time (no masking needed)
- **Auto-regressive training**: Requires causal masking to prevent information leakage

Without masking during training, the model learns to rely on future tokens, which breaks at inference when you can only feed past tokens.

### Q: What are the different masking strategies in Transformers?

**1. Causal Masking (GPT-style decoders):**
```
[1 0 0 0]  ← position 0 sees only itself
[1 1 0 0]  ← position 1 sees 0, 1
[1 1 1 0]  ← position 2 sees 0, 1, 2
[1 1 1 1]  ← position 3 sees all past
```
Enforces auto-regressive generation (left-to-right)

**2. No Masking (BERT-style encoders):**
```
[1 1 1 1]  ← all positions attend to all
[1 1 1 1]
[1 1 1 1]
[1 1 1 1]
```
Bidirectional context, not auto-regressive

**3. Prefix Masking (Hybrid):**
```
[1 0 0 0 0]  ← prefix can only look at itself
[1 1 0 0 0]  ← prefix can look at earlier prefix
[1 1 0 0 0]  ← generation starts, sees whole prefix
[1 1 0 1 0]  ← can see prefix + current, not future
[1 1 0 1 1]
```
Bidirectional prefix + auto-regressive generation

### Q: Can you have auto-regressive generation without masked attention?
**Technically yes, but it's inefficient.**

You could train without masking and the model would eventually learn auto-regressive behavior (generating one token at a time). However:
- It's wasteful—the model wastes capacity learning NOT to use future tokens
- **Training-inference mismatch**: Model trained to see all tokens but generates with only past tokens
- Worse performance compared to masked training

**Best practice**: Use causal masking during training for auto-regressive models (GPT-style).
