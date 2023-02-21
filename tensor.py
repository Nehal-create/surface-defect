import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Restrict TensorFlow to only use the first GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')
    # Allow TensorFlow to allocate memory on an as-needed basis
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU found and set up successfully")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
else:
  print("No GPU found on this system")