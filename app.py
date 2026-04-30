import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.transform import resize
import os
from io import BytesIO
from datetime import datetime
import cv2

# PAGE CONFIGURATION
st.markdown("""
<style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #f5f7fa 100%);
        color: #000000;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    /* Title Styling */
    .stTitle h1 {
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #3498db, #2c3e50);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        -webkit-text-fill-color: transparent;
    }

    /* Custom box styling */
    .custom-box {
        background: white !important;
        color: black !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.25rem 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border-left: 5px solid #3498db;
    }
    
    .custom-title {
        color: #3498db;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }

    /* Selectbox label & File uploader label */
    .stSelectbox label,
    .stFileUploader label {
        color: #3498db !important;
        font-weight: bold !important;
        font-size: 20px !important;
    }

    /* Inputs & buttons */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stFileUploader > div > div {
        background-color: white;
        color: #000000;
        border: 1px solid #B6B09F;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #3498db, #2980b9);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(to right, #2980b9, #3498db);
    }

    /* Download Button */
    .stDownloadButton button {
        background: #3498db !important;
        color: #f5f7fa !important;
        font-weight: bold !important;
    }
    .stDownloadButton button:hover {
        background: #CC0000 !important;
        color: white !important;
    }

    /* Force specific divs text to black */
    div.st-ak.st-al.st-bd.st-be.st-bf.st-as.st-bg.st-bh.st-ar.st-bi.st-bj.st-bk.st-bl {
        color: black !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    div.st-an.st-ao.st-ap.st-aq.st-ak.st-ar.st-am.st-as.st-at.st-au.st-av.st-aw.st-ax.st-ay.st-az.st-b0.st-b1.st-b2.st-b3.st-b4.st-b5.st-b6.st-b7.st-b8.st-b9.st-ba.st-bb.st-bc {
        background-color: white !important;
    }
    svg[data-baseweb="icon"] {
    fill: #2980b9; 
    }
            
    section.st-emotion-cache-1erivf3.e16xj5sw0 {
        background-color: white !important;
        color: black !important;
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        padding: 1rem !important;
    }

    section.st-emotion-cache-1erivf3.e16xj5sw0 * {
        color: black !important;
    }

    div.st-emotion-cache-j7qwjs {
        background-color: white !important;
        color: black !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
    }

    div.st-emotion-cache-j7qwjs * {
        color: black !important;
    }

    small.st-emotion-cache-c8ta4l.ejh2rmr0 {
        background-color: white !important;
        color: black !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 4px !important;
    }
    button.st-emotion-cache-ktz07o.eacrzsi2 {
        background-color: #3498db !important;
        color: #f5f7fa !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        cursor: pointer !important;
    }

    button.st-emotion-cache-ktz07o.eacrzsi2:hover {
        background-color: #3498db !important;
    }      

    /* Custom styling for metrics */
    [data-testid="stMetricLabel"] p {
        color: black !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricValue"] div {
        color: black !important;
    }
 
</style>

<script>
document.addEventListener("DOMContentLoaded", function() {
    function changeLabelColor() {
        document.querySelectorAll("*").forEach(function(el) {
            if (el.textContent.trim().toLowerCase() === "select task" ||
                el.textContent.trim().toLowerCase() === "upload mri image") {
                el.style.color = "#3498db";
                el.style.fontWeight = "bold";
                el.style.fontSize = "20px";
            }
        });
    }
    changeLabelColor();
    const observer = new MutationObserver(changeLabelColor);
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""", unsafe_allow_html=True)

# CUSTOM COMPONENTS
def show_box(message, title_type="Info", icon="ℹ️"):
    icons = {
        "Info": "ℹ️",
        "Success": "✅",
        "Error": "❌",
        "Welcome": "📌",
        "Result": "🔍",
        "Confidence": "📊",
        "Saved": "💾",
        "Upload Required": "📤"
    }
    icon = icons.get(title_type, icon)
    title = f"{icon} {title_type}"
    st.markdown(
        f"""
        <div class='custom-box'>
            <div class='custom-title'>
                {title}
            </div>
            <div style='line-height: 1.6;'>
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# MODEL FUNCTIONS
def apply_filters(image):
    """Enhance quality and remove noise using Gaussian and Median filtering."""
    # Ensure image is in uint8 for OpenCV
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 1. Apply Gaussian Blur (removes high-frequency noise)
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Apply Median Filter (removes salt-and-pepper noise)
    median = cv2.medianBlur(gaussian, 5)
    
    return median / 255.0

# --- TRADITIONAL SEGMENTATION METHODS ---

def seg_thresholding(image):
    """Otsu's Thresholding Algorithm."""
    img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def seg_kmeans(image):
    """K-Means Segmentation Algorithm."""
    img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    pixel_values = gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()].reshape((gray.shape))
    _, mask = cv2.threshold(res, np.max(centers) - 1, 255, cv2.THRESH_BINARY)
    return mask

def seg_contours(image):
    """Extract tumor ROI by stripping the skull ring and finding internal bright spots."""
    img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # 1. Create a mask of the entire head
    _, thresh_head = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours_head, _ = cv2.findContours(thresh_head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_head:
        return np.zeros_like(gray)
        
    # 2. Get the largest contour (the whole head/skull)
    largest_cnt = max(contours_head, key=cv2.contourArea)
    head_mask = np.zeros_like(gray)
    cv2.drawContours(head_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
    
    # 3. Erode the mask to remove the skull "ring"
    kernel = np.ones((11,11), np.uint8)
    brain_mask_no_skull = cv2.erode(head_mask, kernel, iterations=1)
    
    # 4. Strip the image to show ONLY the internal brain tissue
    stripped_brain = cv2.bitwise_and(gray, gray, mask=brain_mask_no_skull)
    
    # 5. Now find the brightest remaining spot (the tumor)
    mean, std = cv2.meanStdDev(stripped_brain, mask=brain_mask_no_skull)
    _, thresh_tumor = cv2.threshold(stripped_brain, int(mean[0][0] + 1.2 * std[0][0]), 255, cv2.THRESH_BINARY)
    
    contours_tumor, _ = cv2.findContours(thresh_tumor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    if contours_tumor:
        # Pick the largest remaining internal contour
        best_cnt = max(contours_tumor, key=cv2.contourArea)
        cv2.drawContours(mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
            
    return mask

def seg_grabcut(image):
    """Accurate GrabCut using intensity-based mask initialization."""
    img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if len(img.shape) == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # 1. Background/Foreground Estimation
    mean, std = cv2.meanStdDev(gray)
    thresh_val = int(mean[0][0] + std[0][0])
    
    mask[:] = cv2.GC_PR_BGD  # Default: Probable Background
    mask[gray > thresh_val] = cv2.GC_PR_FGD  # Bright spots: Probable Foreground
    mask[gray < 20] = cv2.GC_BGD  # Very dark: Certain Background
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        # 2. Run GrabCut using the MASK initialization
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        # 3. Filter the result
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        return mask2 * 255
    except:
        return np.zeros(img.shape[:2], np.uint8)


def preprocess_segmentation_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = resize(image, (256, 256), mode='constant', preserve_range=True)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    
    # Apply noise removal filters
    image = apply_filters(image)
    return image

def preprocess_classification_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = resize(image, (224, 224), mode='constant', preserve_range=True)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    
    # Apply noise removal filters
    image = apply_filters(image)
    return image

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

@st.cache_resource
def load_segmentation_model():
    try:
        model = tf.keras.models.load_model(
            "BrainTumor_Segmentation_Unet.h5",
            custom_objects={'combined_loss': combined_loss, 'iou_metric': iou_metric}
        )
        show_box("Segmentation model loaded successfully.", "Success")
        return model
    except Exception as e:
        show_box(f"Failed to load segmentation model: {e}", "Error")
        return None

@st.cache_resource
def load_classification_model():
    try:
        model = tf.keras.models.load_model(
            "BrainTumor_classification_model.h5"
        )
        show_box("Classification model loaded successfully.", "Success")
        return model
    except Exception as e:
        show_box(f"Failed to load classification model: {e}", "Error")
        return None

# MAIN APP INTERFACE
st.markdown("""
<div class='stTitle'>
    <h1>🧠 Advanced Brain Tumor Analysis</h1>
</div>
""", unsafe_allow_html=True)

show_box("Welcome to our advanced MRI analysis tool. Upload an image and select a task to proceed.", "Welcome")

# Sidebar
with st.sidebar:
    st.title("🔬 Project Information")
    st.markdown("---")
    st.markdown("""
    **Model Architecture**  
    🧠 Classification: CNN  
    🎯 Segmentation: U-Net  
    """)
    st.markdown("---")
    st.markdown(f"**Last Run**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Developer**: Team CV")
    st.markdown("---")
    st.markdown("""
    <div class='sidebar-footer'>
        Medical Imaging AI • Research Project
    </div>
    """, unsafe_allow_html=True)

task = st.selectbox("Select Task", ["Segmentation", "Classification"], key="task_select")

uploaded_file = st.file_uploader("Upload MRI Image", type=["tif", "png", "jpg", "jpeg"], key="file_uploader")

output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    except Exception as e:
        show_box(f"Error loading image: {e}", "Error")
        st.stop()

    if task == "Segmentation":
        processed_image = preprocess_segmentation_image(image)
        processed_image_batch = np.expand_dims(processed_image, axis=0)

        segmentation_model = load_segmentation_model()
        if segmentation_model is None:
            st.stop()

        try:
            with st.spinner("Performing segmentation..."):
                pred_mask = segmentation_model.predict(processed_image_batch, verbose=0)[0]
                pred_mask_binary = (pred_mask > 0.5).astype(np.float32)

            st.subheader("Comparison of Segmentation Methods")
            
            # Row 1: Original vs AI
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.image(image, caption="Original MRI", use_container_width=True)
            with row1_col2:
                st.image(pred_mask_binary[..., 0], caption="AI (U-Net) Prediction", use_container_width=True, clamp=True)

            # Row 2: Traditional Methods
            st.markdown("### Traditional Computer Vision Methods")
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.image(seg_thresholding(processed_image), caption="Thresholding", use_container_width=True)
            with c2:
                st.image(seg_kmeans(processed_image), caption="K-Means", use_container_width=True)
            with c3:
                st.image(seg_contours(processed_image), caption="Contour Detection", use_container_width=True)
            with c4:
                st.image(seg_grabcut(processed_image), caption="GrabCut", use_container_width=True)

            # --- ENHANCED FEATURE EXTRACTION ---
            mask_uint8 = (pred_mask_binary[..., 0] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                # Circularity: How round the tumor is (1.0 = perfect circle)
                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
                
                # Bounding Box (Dimensions)
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Intensity Analysis
                if processed_image.max() <= 1.0:
                    temp_img = (processed_image * 255).astype(np.uint8)
                else:
                    temp_img = processed_image.astype(np.uint8)
                
                # Convert to grayscale for single-channel analysis to avoid OpenCV errors
                temp_gray = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY) if len(temp_img.shape) == 3 else temp_img
                
                mean_intensity = cv2.mean(temp_gray, mask=mask_uint8)[0]
                max_val = cv2.minMaxLoc(temp_gray, mask=mask_uint8)[1]

                # UI Layout: Metric Cards
                st.markdown("### 📊 Tumor Metrics Analysis")
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Tumor Area", f"{area:.0f} px²", help="Total pixel count of the segmented region")
                with m_col2:
                    st.metric("Perimeter", f"{perimeter:.1f} px")
                with m_col3:
                    st.metric("Circularity", f"{circularity:.2f}", help="Shape regularity (1.0 is a circle)")

                st.markdown("---")
                
                d_col1, d_col2, d_col3 = st.columns(3)
                with d_col1:
                    st.write("**Dimensions**")
                    st.info(f"Width: {w}px | Height: {h}px")
                with d_col2:
                    st.write("**Intensity (Brightness)**")
                    st.info(f"Mean: {mean_intensity:.1f} | Max: {max_val:.0f}")
                with d_col3:
                    st.write("**Location (Center)**")
                    st.info(f"X: {x + w//2} | Y: {y + h//2}")

            mask_image = Image.fromarray((pred_mask_binary[..., 0] * 255).astype(np.uint8))
            
            output_image_path = os.path.join(output_dir, f"mask_{uploaded_file.name}")
            png_buffer = BytesIO()
            mask_image.save(png_buffer, format="PNG")
            mask_filename = f"mask_{os.path.splitext(uploaded_file.name)[0]}.png"
            if st.download_button(
                "Download Mask",
                data=png_buffer.getvalue(),
                file_name=mask_filename,
                mime="image/png",
                help="Click to download and save the mask"
            ):
                mask_image.save(os.path.join(output_dir, mask_filename))
                show_box(f"Mask has been saved to {os.path.join(output_dir, mask_filename)}", "Saved")

        except Exception as e:
            show_box(f"Error during segmentation prediction: {e}", "Error")

    elif task == "Classification":
        processed_image = preprocess_classification_image(image)
        processed_image_batch = np.expand_dims(processed_image, axis=0)

        classification_model = load_classification_model()
        if classification_model is None:
            st.stop()

        try:
            with st.spinner("Performing classification..."):
                pred_prob = classification_model.predict(processed_image_batch, verbose=0)[0][0]
                pred_class = "Tumor" if pred_prob > 0.5 else "No Tumor"
                confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

            st.subheader("Classification Result")
            show_box(f"Prediction: {pred_class}", "Result")
            show_box(f"Confidence: {confidence:.2%}", "Confidence")

            result_content = f"Prediction: {pred_class}\nConfidence: {confidence:.2%}"
            output_text_path = os.path.join(output_dir, f"result_{uploaded_file.name}.txt")
            
            if st.download_button(
                "Download Result",
                data=result_content.encode("utf-8"),
                file_name=os.path.basename(output_text_path),
                mime="text/plain",
                help="Click to download and save the result"
            ):
                with open(output_text_path, "w", encoding="utf-8") as f:
                    f.write(result_content)
                show_box(f"Result has been saved to {output_text_path}", "Saved")

        except Exception as e:
            show_box(f"Error during classification prediction: {e}", "Error")
else:
    show_box("Please upload an MRI image to get started.", "Upload Required")

