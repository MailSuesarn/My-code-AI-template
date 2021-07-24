"""
Author: Suesarn Wilainuch
Attend-and-Compare Module (ACM) tensorflow 2 version
Paper reference: https://arxiv.org/abs/2007.07506
"""

import tensorflow as tf

class AttendModule(tf.keras.layers.Layer):

    def __init__(self, num_features, num_heads=4, **kwargs):
        super(AttendModule, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.num_features = num_features
        self.num_c_per_head = self.num_features // self.num_heads
        assert self.num_features % self.num_heads == 0

        self.map_gen = tf.keras.layers.Conv2D(self.num_heads, kernel_size=1, strides=(1, 1), padding='same', groups=self.num_heads)
    
    
    def batch_weighted_avg(self, xhats, weights):

        # xhat reshape
        xhats_reshape = tf.reshape(xhats, shape=[tf.shape(xhats)[0] * self.num_heads, self.num_c_per_head, tf.shape(xhats)[1] * tf.shape(xhats)[2]])

        # weight reshape
        weights_reshape = tf.reshape(weights, shape=[tf.shape(weights)[0] * self.num_heads, 1, tf.shape(xhats)[1] * tf.shape(xhats)[2]])

        weights_normalized = tf.nn.softmax(weights_reshape, axis=2)
        
        mus = tf.linalg.matmul(xhats_reshape, weights_normalized, transpose_a=False, transpose_b=True)
        mus = tf.reshape(mus, shape=[tf.shape(weights)[0], 1, 1, self.num_heads * self.num_c_per_head])

        return mus

    def call(self, x):

        weights = self.map_gen(x)

        mus = self.batch_weighted_avg(x, weights)

        return mus


class ModulateModule(tf.keras.layers.Layer):

    def __init__(self, channel, num_groups=32, compressions=2, **kwargs):
        super(ModulateModule, self).__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(channel // compressions, kernel_size=1, strides=(1, 1), padding='same', groups=num_groups)
        self.conv_2 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=(1, 1), padding='same', groups=num_groups)

    def call(self, x):
        y = self.conv_1(x)
        y = tf.nn.relu(y)
        y = self.conv_2(y)
        y = tf.math.sigmoid(y)

        return y


class ACM(tf.keras.layers.Layer):

  def __init__(self, num_heads, num_features, orthogonal_loss=False, **kwargs):
      super(ACM, self).__init__(**kwargs)

      if num_features % num_heads != 0:
        raise Exception(f'Error at "ACM" num_features must be divisible by num_heads.')

      self.num_features = num_features
      self.num_heads = num_heads

      self.add_mod = AttendModule(self.num_features, num_heads=self.num_heads)
      self.sub_mod = AttendModule(self.num_features, num_heads=self.num_heads)
      self.mul_mod = ModulateModule(channel=self.num_features, num_groups=self.num_heads, compressions=2)

      self.orthogonal_loss = orthogonal_loss

  def call(self, x):

      mu = tf.math.reduce_mean(x, axis=[1, 2], keepdims=True)
      x_mu = x - mu

      # creates multipying feature
      mul_feature = self.mul_mod(mu)  # P

      # creates add or sub feature
      add_feature = self.add_mod(x_mu)  # K

      # creates add or sub feature
      sub_feature = self.sub_mod(x_mu)  # Q

      y = (x + (add_feature - sub_feature)) * mul_feature

      if self.orthogonal_loss:
          dp = tf.math.reduce_mean(add_feature * sub_feature, axis=-1, keepdims=True)
          return y, dp
      else:
          return y



# Test: ACM layer
input_shape = (2, 2, 2, 2)
x = tf.random.normal(input_shape)

acm = ACM(num_heads=1, num_features=2, orthogonal_loss=True)

y, dp = acm(x)

print(x.shape)
print(y.shape)
print(dp.shape)


# Test: create model by using ACM layer 
def model():
  image_shape = (128, 128, 3)
  in_img = tf.keras.Input(image_shape)
      
  x = tf.keras.layers.Conv2D(64, 3, 2, padding='same')(in_img)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = ACM(num_heads=32, num_features=64, orthogonal_loss=False)(x)

  x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = ACM(num_heads=32, num_features=128, orthogonal_loss=False)(x)

  x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = ACM(num_heads=32, num_features=256, orthogonal_loss=False)(x)

  x = tf.keras.layers.Conv2D(512, 3, 2, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = ACM(num_heads=32, num_features=512, orthogonal_loss=False)(x)

  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  output = tf.keras.layers.Dense(10, activation='softmax')(x) 
  
  model = tf.keras.Model(in_img, output)
  model.summary()
  return model

model = model()
