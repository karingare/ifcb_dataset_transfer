import streamlit as st
from PIL import Image
import io
from web_inference import predict_all  # Uses your existing inference code

st.set_page_config(
    page_title="IFCB Plankton Classifier",
    page_icon="🪸",
    layout="wide",   # important for column layout
)

# -----------------------------
# Title
# -----------------------------
st.title("🪸 IFCB Plankton Classifier")
st.write("""
Upload an IFCB image to see predictions from **two different models**:
- One trained on **Baltic Sea** data  
- One trained on **Skagerrak/Kattegat** data  
""")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an IFCB image (PNG/JPG/TIF)", 
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    # Read bytes
    image_bytes = uploaded_file.read()

    # Display the image
    st.subheader("Uploaded Image")
    st.image(image_bytes, use_column_width=True)

    st.subheader("Predictions")
    with st.spinner("Running models..."):
        results = predict_all(image_bytes, topk=3)

    # -----------------------------
    # Show predictions in two columns
    # -----------------------------
    model_names = list(results.keys())
    col1, col2 = st.columns(2, gap="large")

    # Left column for model 1
    with col1:
        name = model_names[0]
        st.markdown(f"### **Model: `{name}`**")
        for cls, prob in results[name]:
            st.write(f"**{cls}** — {prob:.3f}")
            st.progress(min(max(prob, 0), 1))

    # Right column for model 2
    with col2:
        name = model_names[1]
        st.markdown(f"### **Model: `{name}`**")
        for cls, prob in results[name]:
            st.write(f"**{cls}** — {prob:.3f}")
            st.progress(min(max(prob, 0), 1))
