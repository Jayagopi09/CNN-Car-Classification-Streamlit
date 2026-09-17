# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from tensorflow.keras.models import load_model
import joblib

model = load_model("model.keras")
label = joblib.load("label_encoder.pkl")

print("Model loaded successfully!")
print("Classes:", label.classes_)

# %%
