import tensorflow as tf
import numpy as np

class Utilities:    
    @staticmethod
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
    
    @staticmethod
    def normalize_expand_transpose_state(state):
        state = np.array(state).astype(np.float32) / 255.0
        state = np.expand_dims(state, axis=0) # shape : (1, 4, 84, 84)
        state = np.transpose(state, (0, 2, 3, 1))  # shape: (1, 84, 84, 4)
        return state


