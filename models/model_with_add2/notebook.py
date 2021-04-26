# Please run the following script with tensorflow>=2.3.0

# IMPORTS
import os
import sys
import tensorflow as tf
import numpy as np

# LOAD CUSTOM OP LIBRARY

sys.path.append(os.path.join(os.getcwd(), "../../tensorflow/"))
from tensorflow_add2.python.ops.add2_ops import add2

# CREATE TEST MODEL

class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  @tf.function (input_signature=[
      tf.TensorSpec(shape=[1, 2], dtype=tf.float32),
      tf.TensorSpec(shape=[1, 2], dtype=tf.float32)
  ])
  def add2(self, x, y):
    '''
    TODO
    '''
    # Name the output 'result' for convenience.
    return {'result' : add2(x, y)}


SAVED_MODEL_PATH = 'out/saved_models/test_variable'
TFLITE_FILE_PATH = 'out/model_with_add2.tflite'

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
tf.saved_model.save(
    module, SAVED_MODEL_PATH)
    #signatures={'my_signature':module.add.get_concrete_function()})

# RUN TESTS FOR CUSTOM OPERATOR
from tensorflow_add2.python.ops.add2_ops_test import Add2Test

Add2Test().testAdd2()


# CONVERT TEST MODEL

# Convert the model using TFLiteConverter
# converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_concrete_functions([module.add2.get_concrete_function()])
converter.allow_custom_ops = True
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)

# LOAD AND TEST TEST MODEL

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.int32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(input_data, output_data)



