"""
Optimizer configuration module for OLMo training.
"""

from olmo_core.optim import AdamWConfig, OptimGroupOverride


def get_optimizer_config():
    """
    Configure the optimizer for OLMo training.
    
    Returns:
        AdamWConfig: Optimizer configuration
    """
    # Configure optimizer
    optim_config = AdamWConfig(
        lr=4e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )
    
    return optim_config 