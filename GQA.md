
# GQA Testing in OLMo-Core Model

This script is designed to verify that Grouped Query Attention (GQA) is correctly implemented and active in the custom OLMo-core Transformer model configuration.

## Description

The script:
1. Loads a user-defined configuration (`config.yaml`) using a helper from `utils.load_config`.
2. Builds a Transformer model using the OLMo-core architecture, specifically the 190M parameter configuration.
3. Enables activation checkpointing to reduce memory usage.
4. Applies a custom optimizer configuration using `AdamW`.
5. Verifies that GQA is active by checking that the number of key/value heads (`n_kv_heads`) is less than the number of total attention heads (`n_heads`) (Further explanation in next section).

6. Runs a dummy forward pass with random input to validate that the model is functional.

##  GQA Validation Logic

The script inspects the first Transformer block's attention weights:

```python
attention_layer = model.blocks["0"].attention
```

## How to check GQA

In standard attention Q, K, V are all split into the same number of heads

Shapes of Q, K, V are:
- Q: [batch, seq_len, n_heads, head_dim]
- K: [batch, seq_len, n_heads, head_dim]
- V: [batch, seq_len, n_heads, head_dim]


For GQA:
- Keep n_heads for the Queries Q
- Reduce the number of key/value heads to n_kv_heads and share them across query heads

GQA -> query heads are grouped such that multiple query heads attend to the same key/value head
Thus:
- If n_kv_heads == n_heads: standard MHA
- If n_kv_heads < n_heads: GQA is active




