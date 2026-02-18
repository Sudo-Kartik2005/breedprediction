"""
Compatibility layer for keras imports
Handles different TensorFlow/keras version combinations
"""

import tensorflow as tf

# Try tf.keras first (works with TensorFlow 2.16.1+)
_keras = None

if hasattr(tf, 'keras'):
    _keras = tf.keras
else:
    # Fallback to standalone keras
    try:
        import keras
        _keras = keras
    except ImportError:
        raise ImportError(
            "Could not import keras. Please ensure TensorFlow is properly installed.\n"
            "Try: pip install tensorflow"
        )

# Export keras
keras = _keras

