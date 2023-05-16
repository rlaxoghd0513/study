import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()     

print(tf.__version__)         #1.14.0
print(tf.executing_eagerly()) #True

gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
   try:
       
       tf.config.experimental.set_visible_devices(gpu[0],'GPU')
       print(gpu[0])
   except RuntimeError as e:
       print(e)
else :
    print('none')