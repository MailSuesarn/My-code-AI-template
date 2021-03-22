"""
Author: Suesarn Wilainuch

Example of Tensorflow implementation of multi-conditional element-wise and control flow 
"""

import tensorflow as tf

class MultiConditionLayer(tf.keras.layers.Layer):
    def __init__(self, S=None, name=None, *args, **kwargs):
        
        super(MultiConditionLayer, self).__init__(*args, **kwargs)
        self.s0 = tf.cast(S[0], tf.float32)
        self.s1 = tf.cast(S[1], tf.float32)
        self.pos_INF = tf.constant(np.inf)
        self.neg_INF = tf.constant(-np.inf) 

    def fn_s0(self, x):
        return self.s0 * x
    
    def fn_s1(self, x):
        return self.s1 * x

    def fn_s2(self, x):
        return self.s0 * self.s1 * x

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        
        # lower bound of condition 1
        greater_con_1 = tf.math.greater(inputs, self.neg_INF)
        # upper bound of condition 1
        less_con_1 = tf.math.less(inputs, 0.0)
        # condition 1
        condition_1 = tf.logical_and(greater_con_1, less_con_1)

        # lower bound of condition 2
        greater_con_2 = tf.math.greater_equal(inputs, 0.0)
        # upper bound of condition 2
        less_con_2 = tf.math.less(inputs, 10.0)
        # condition 2
        condition_2 = tf.logical_and(greater_con_2, less_con_2)

        # lower bound of condition 3
        greater_con_3 = tf.math.greater_equal(inputs, 10.0)
        # upper bound of condition 3
        less_con_3 = tf.math.less(inputs, self.pos_INF)
        # condition 3
        condition_3 = tf.logical_and(greater_con_3, less_con_3)
        
        # element-wise conditional 
        output = tf.where(condition_1, self.fn_s0(inputs),
                          tf.where(condition_2, self.fn_s1(inputs),
                                   tf.where(condition_3, self.fn_s2(inputs), self.fn_s2(inputs)
                                   )
                          )
                 )
               
        return output


S = np.asarray([2, 2])

x = np.asarray([-1, 2, 3, 9, 10, 12])

f = MultiConditionLayer(S=S)(x)

print(f)
print(f.numpy())