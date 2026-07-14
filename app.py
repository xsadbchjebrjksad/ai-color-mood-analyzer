import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

st.set_page_config(page_title="AI Color & Mood Analyzer", page_icon="🎨", layout="centered")

st.title("🎨 AI Color & Mood Analyzer")
st.write(
    "Upload any photo and this app uses **K-Means clustering** (unsupervised machine "
    "learning) to find its dominant colors, then analyzes the palette to guess the "
    "overall **mood** of the image."
)

st.divider()


def get_dominant_colors(image, k=5):
    img = image.copy()
    img.thumbnail((150, 150))  # resize for speed
    img_array = np.array(img.convert("RGB"))
    pixels = img_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / counts.sum() * 100

    # sort by percentage, descending
    order = np.argsort(-percentages)
    return colors[order], percentages[order]


def analyze_mood(colors, percentages):
    # weighted average brightness and saturation
    total_r, total_g, total_b = 0, 0, 0
    for color, pct in zip(colors, percentages):
        r, g, b = color
        total_r += r * pct / 100
        total_g += g * pct / 100
        total_b += b * pct / 100

    brightness = (total_r + total_g + total_b) / 3
    max_c = max(total_r, total_g, total_b)
    min_c = min(total_r, total_g, total_b)
    saturation = (max_c - min_c)

    warm_score = (total_r - total_b)  # positive = warm (red/orange), negative = cool (blue)

    mood_tags = []

    if brightness > 170:
        mood_tags.append("Bright")
    elif brightness < 80:
        mood_tags.append("Dark")
    else:
        mood_tags.append("Balanced")

    if saturation > 60:
        mood_tags.append("Vibrant")
    else:
        mood_tags.append("Muted")

    if warm_score > 20:
        mood_tags.append("Warm")
    elif warm_score < -20:
        mood_tags.append("Cool")
    else:
        mood_tags.append("Neutral-toned")

    mood_descriptions = {
        ("Bright", "Vibrant", "Warm"): "Energetic and cheerful — feels lively and inviting.",
        ("Bright", "Vibrant", "Cool"): "Fresh and crisp — feels clean and modern.",
        ("Bright", "Muted", "Warm"): "Soft and cozy — feels calm and comforting.",
        ("Bright", "Muted", "Cool"): "Airy and serene — feels peaceful and light.",
        ("Dark", "Vibrant", "Warm"): "Bold and intense — feels dramatic and powerful.",
        ("Dark", "Vibrant", "Cool"): "Moody and striking — feels mysterious.",
        ("Dark", "Muted", "Warm"): "Rich and moody — feels intimate and warm.",
        ("Dark", "Muted", "Cool"): "Somber and quiet — feels melancholic or dramatic.",
        ("Balanced", "Vibrant", "Warm"): "Warm and dynamic — feels welcoming and rich.",
        ("Balanced", "Vibrant", "Cool"): "Cool and refreshing — feels calm yet energetic.",
        ("Balanced", "Muted", "Warm"): "Earthy and grounded — feels natural and relaxed.",
        ("Balanced", "Muted", "Cool"): "Tranquil and neutral — feels understated and calm.",
    }

    key = (mood_tags[0], mood_tags[1], mood_tags[2])
    description = mood_descriptions.get(key, "A balanced mix of tones.")

    return mood_tags, description


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

num_colors = st.slider("Number of dominant colors to extract:", 3, 8, 5)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing colors with K-Means clustering..."):
        colors, percentages = get_dominant_colors(image, k=num_colors)
        mood_tags, mood_description = analyze_mood(colors, percentages)

    st.subheader("🎨 Dominant Color Palette")
    cols = st.columns(len(colors))
    for i, (color, pct) in enumerate(zip(colors, percentages)):
        hex_color = "#%02x%02x%02x" % tuple(color)
        with cols[i]:
            st.markdown(
                f"<div style='background-color:{hex_color}; height:80px; "
                f"border-radius:8px; border:1px solid #ccc;'></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{hex_color}\n{pct:.1f}%")

    st.divider()
    st.subheader("🧠 Detected Mood")
    st.write(" • ".join([f"**{tag}**" for tag in mood_tags]))
    st.info(mood_description)

else:
    st.info("👆 Upload a photo to get started (landscapes, portraits, artwork all work well).")

st.divider()
st.caption("Built with Streamlit + scikit-learn (K-Means clustering for color extraction)")
