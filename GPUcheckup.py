import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("Number of GPUs available:", len(gpus))
