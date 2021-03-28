"""Tensorflow-Keras Implementation of Swish"""

## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


class Swish(Activation):
    '''
    Swish Activation Function.
    .. math::
        swish(x) = x * sigmoid(x) = x * (1 / (1 + e^{-x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Swish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'


def swish(inputs):
    return inputs * tf.math.sigmoid(inputs)

get_custom_objects().update({'Swish': Swish(swish)})