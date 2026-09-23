import base64
import io
import os
import requests
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Animal Predictor")
st.title("Animal Predictor")
endpoint = os.environ.get("ENDPOINT_URL")
key = os.environ.get("ENDPOINT_KEY")
upload = st.file_uploader("Upload an animal photo", type=["jpg", "jpeg", "png"])
if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded photo")
    image.thumbnail((512, 512)); buffer = io.BytesIO(); image.save(buffer, "JPEG", quality=85)
    if not endpoint or not key:
        st.error("Set ENDPOINT_URL and ENDPOINT_KEY before making a prediction.")
    else:
        response = requests.post(endpoint, json={"image": base64.b64encode(buffer.getvalue()).decode("ascii")},
                                 headers={"Authorization": f"Bearer {key}"}, timeout=60)
        result = response.json()
        if "error" in result: st.error(result["error"])
        else:
            st.subheader(result["animal"]); st.write(f"Confidence: {result['confidence']:.1%}")
            st.bar_chart(pd.DataFrame.from_dict(result["all_scores"], orient="index", columns=["score"]))
            if result["confidence"] < 0.60: st.warning("The model is not very confident in this prediction.")