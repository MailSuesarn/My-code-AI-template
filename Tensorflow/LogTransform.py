class LogTransform(Layer):
  """Normalization inputs by the following log transform
  x = log(x + 1 - min(x))
  
  Input shape:
    Arbitrary.
  Output shape:
    Same as input.
  """

  def __init__(self, name=None, **kwargs):

    super(LogTransform, self).__init__(name=name, **kwargs)

  def call(self, inputs):
    dtype = self._compute_dtype
    inputs = math_ops.cast(inputs, dtype)
    min = tf.math.reduce_min(inputs)
    log = tf.math.log(inputs + 1. - min)
    return log

  def compute_output_shape(self, input_shape):
    return input_shape


logTrans = LogTransform()
numeric_logTrans = logTrans(x)