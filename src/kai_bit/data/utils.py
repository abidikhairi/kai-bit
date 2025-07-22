


def collate_chatml(batch):
    """
    Collate function for ChatML format.
    
    Args:
        batch (list): List of dictionaries containing 'messages' and 'metadata'.
    
    Returns:
        dict: Collated dictionary with 'messages' and 'metadata'.
    """
    messages = [item['messages'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    return {
        'messages': messages,
        'metadata': metadata
    }
    