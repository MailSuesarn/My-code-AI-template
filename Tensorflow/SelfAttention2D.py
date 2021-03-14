
"""
Author: Suesarn Wilainuch
"""
import tensorflow as tf


class SelfAttention2D(tf.keras.layers.Layer):

    def __init__(self, k, **kwargs):
        super(SelfAttention2D, self).__init__(**kwargs)
        """
         follow the paper for memory efficiency and did not notice any significant performance decrease
         when reducing the channel number C/k, where k = 1, 2, 4, 8 and they choose k for all experiment
        """
        self.k = k

    def build(self, input_shape):
        if self.k > input_shape[-1] or self.k < 0:
            raise Exception(
                f'Error at "SelfAttention2D" k value should be in range (0, C] where C is the number channel of the previous layer.')
        if len(input_shape) != 4:
            raise Exception(
                f'Error at "SelfAttention2D" Input of layer SelfAttention2D is incompatible with the layer: : expected min_ndim=4, found ndim=2. Full shape received: {input_shape}')

        kernel_shape_f_g = (1, 1) + (input_shape[-1], int(input_shape[-1] // self.k))
        kernel_shape_h = (1, 1) + (input_shape[-1], input_shape[-1])

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f',
                                        trainable=True)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h',
                                        trainable=True)

        super(SelfAttention2D, self).build(input_shape)
        # Set input spec.
        self.input_spec = tf.keras.layers.InputSpec(ndim=4,
                                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[3]])  # N = h x w

        f = tf.nn.conv2d(x,
                         filters=self.kernel_f,
                         strides=(1, 1), padding='SAME')  # [bs, h, w, c']
        g = tf.nn.conv2d(x,
                         filters=self.kernel_g,
                         strides=(1, 1), padding='SAME')  # [bs, h, w, c']
        h = tf.nn.conv2d(x,
                         filters=self.kernel_h,
                         strides=(1, 1), padding='SAME')  # [bs, h, w, c]

        # Transpose and dot product
        s = tf.linalg.matmul(hw_flatten(g), hw_flatten(f), transpose_a=False, transpose_b=True)  # [bs, N, N]

        beta = tf.nn.softmax(s, axis=1)  # attention map

        v = tf.linalg.matmul(beta, hw_flatten(h))  # [bs, N, C]

        v = tf.reshape(v, shape=tf.shape(x))  # [bs, h, w, c]

        x = self.gamma * v + x  # a scale parameter and add back the input feature map

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'k': self.k,
        })
        return config
