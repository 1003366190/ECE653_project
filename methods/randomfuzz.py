import numpy as np
import tensorflow as tf

# the fuzz step of random fuzz
def random_fuzz(x, epsilon, clip_min=0, clip_max=1):
    # randomly change the rgb values in the range between -epsilon to epsilon, and return clipped result
    return np.clip(x + epsilon * tf.random.uniform(x.shape, minval=-1, maxval=1), clip_min, clip_max)
