import tensorflow as tf
from data_preprocessing import load_data

MODEL_PATH = "saved_models/emotion_cnn.h5"
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

model = tf.keras.models.load_model(MODEL_PATH)

_, test_data = load_data(TRAIN_DIR, TEST_DIR)

loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc:.2f}")
