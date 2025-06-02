import tensorflow as tf

def softmax(logits):
    """
    Compute softmax probabilities from logits.
    
    Args:
        logits: Tensor of shape (batch_size, num_actions)
    
    Returns:
        probs: Tensor of shape (batch_size, num_actions) with probabilities
    """
    exp_logits = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
    return exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)