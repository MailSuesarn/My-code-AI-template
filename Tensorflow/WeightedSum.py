"""
Author: Suesarn Wilainuch
"""
import tensorflow as tf

class WeightedSum(tf.keras.layers.Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        self.a = self.add_weight(
            name='alpha',
            shape=[1],
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            constraint=constraint,    
        )
        super(WeightedSum, self).build(input_shape)

    def call(self, inputs):
        return self.a * inputs[0] + (1 - self.a) * inputs[1]  

    def compute_output_shape(self, input_shape):
        return input_shape