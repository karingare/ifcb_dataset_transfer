import streamlit as st
from PIL import Image
import io
from web_inference import predict_all  # Uses your existing inference code

st.set_page_config(
    page_title="IFCB Plankton Classifier",
    page_icon="🪸",
    layout="wide",
)

# -----------------------------
# Session state for history
# -----------------------------
if "history" not in st.session_state:
    # Each entry: {"filename": str, "image_bytes": bytes, "results": dict}
    st.session_state["history"] = []

# -----------------------------
# Title
# -----------------------------
st.title("🪸 IFCB Plankton Classifier")

st.markdown("""
Upload an Imaging FlowCytobot image to see predictions from **two different models**:

- **Baltic Sea model**  
  [Dataset DOI: 10.23728/b2share.7c273b6f409c47e98a868d6517be3ae3](https://doi.org/10.23728/b2share.7c273b6f409c47e98a868d6517be3ae3)

- **Skagerrak/Kattegat model**  
  [Dataset DOI: 10.17044/scilifelab.25883455.v4](https://doi.org/10.17044/scilifelab.25883455.v4)
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

    # Display the big image
    st.subheader("Uploaded Image")
    st.image(image_bytes, use_column_width=True)

    st.subheader("Predictions")
    with st.spinner("Running models..."):
        results = predict_all(image_bytes, topk=3)

    # Save to history
    st.session_state["history"].append({
        "filename": uploaded_file.name,
        "image_bytes": image_bytes,
        "results": results,
    })

    # -----------------------------
    # Show current predictions in two columns
    # -----------------------------
    model_names = list(results.keys())
    if len(model_names) >= 2:
        col1, col2 = st.columns(2, gap="large")
        cols = [col1, col2]
    else:
        cols = [st]

    for col, model_name in zip(cols, model_names):
        with col:
            col.markdown(f"### **Model: `{model_name}`**")
            for cls, prob in results[model_name]:
                col.write(f"**{cls}** — {prob:.3f}")
                col.progress(min(max(prob, 0), 1))

# -----------------------------
# History section
# -----------------------------
st.markdown("---")
st.subheader("Prediction history (this session)")

if not st.session_state["history"]:
    st.write("No predictions yet. Upload an image above to get started.")
else:
    # Show most recent first
    for i, entry in enumerate(reversed(st.session_state["history"]), start=1):
        filename = entry["filename"]
        img_bytes = entry["image_bytes"]
        results = entry["results"]

        with st.expander(f"{i}. {filename}"):
            # Small thumbnail + summary
            thumb_col, info_col = st.columns([1, 2])

            with thumb_col:
                st.image(img_bytes, width=120)

            with info_col:
                st.markdown(f"**File:** `{filename}`")

                for model_name, preds in results.items():
                    top_cls, top_prob = preds[0]
                    st.write(
                        f"**{model_name}** → {top_cls} ({top_prob:.3f})"
                    )
