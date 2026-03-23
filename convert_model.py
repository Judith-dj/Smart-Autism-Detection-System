import tensorflow as tf
from app import Any, NotEqual  # your exact custom layers used during training

# Load your existing HDF5 model
model = tf.keras.models.load_model(
    "multimodal_autism_model.h5",
    compile=False,
    custom_objects={"Any": Any, "NotEqual": NotEqual}
)

print("✅ HDF5 model loaded successfully!")

