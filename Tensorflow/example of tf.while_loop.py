"""
Author: Suesarn Wilainuch
"""

# example of tf.while_loop()

# Python with normal numerical
def loop_sum(n):
  i = 1
  sum = 0
  while i <= n:
    sum = sum + i
    i += 1
  return sum

n = 5
print(loop_sum(n))


# Tensorflow implement while loop by using tf.while_loop() Which is equivalent to the python code above

def loop_sum_tf(n):
    # Track both the loop index and summation in a tuple in the form (index, summation)
    index_summation = (tf.constant(1), tf.constant(0.0))

    # The loop condition, note the loop condition is 'i <= n'
    def condition(index, summation):
        return tf.less_equal(index, n)

    # The loop body, this will return a result tuple in the same form (index, summation)
    def body(index, summation):
        summation = tf.add(tf.cast(summation, tf.float32), tf.cast(index, tf.float32))
        index = tf.add(index, 1)
        return index, summation
    
    # In this case interested only summation value
    result = tf.while_loop(condition, body, index_summation)[1]
    return result
    


n = 5
result = loop_sum_tf(n)
print(result)
print(result.numpy())