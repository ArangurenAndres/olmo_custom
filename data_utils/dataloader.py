"""
Data loader module for OLMo training.
"""

from olmo_core.data import NumpyDataLoaderConfig


def build_dataloader(dataset, batch_size, sequence_length):
    """
    Build data loader for training.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        sequence_length: Sequence length
        
    Returns:
        object: Data loader
    """
    # Calculate global batch size
    global_batch_size = batch_size * sequence_length
    
    # Configure data loader
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=42,
        num_workers=4,
    )

    # Build data loader
    data_loader = data_loader_config.build(dataset)
    print("Data loader built successfully")
    
    return data_loader 