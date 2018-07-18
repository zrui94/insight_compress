import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

checkpoint_path='./pretrained/model.ckpt-1450000'
# checkpoint_path = os.path.join(model_dir, "model.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
i = 0
for key in var_to_shape_map:
    if key == 'MobilenetV2/InvertedResidual_16_0/depthwise/BatchNorm/beta':
        print('############################')
    print("tensor_name: ", key)
    i+=1
print(i)
    # print(tf.shape(reader.get_tensor(key)))