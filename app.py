from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
import io
import zipfile
import base64
import concurrent.futures

app = Flask(__name__)

high_quality = []
low_quality = []

# 📌 Function to analyze image quality efficiently
def analyze_image(image):
    """Computes sharpness, colorfulness, brightness, contrast, and resolution."""
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = gray.std()
    resolution = image.shape[0] * image.shape[1]

    # Colorfulness metric
    (B, G, R) = cv2.split(image)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    colorfulness = np.sqrt((rg.var() ** 2) + (yb.var() ** 2))

    return sharpness, colorfulness, brightness, contrast, resolution

# 📌 Function to process an individual image
def process_single_image(uploaded_file):
    """Reads, analyzes, and classifies the image into high or low quality."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return None  # Skip invalid images
    
    metrics = analyze_image(image)
    if not metrics:
        return None
    
    sharpness, colorfulness, brightness, contrast, resolution = metrics

    # 🎯 Define quality thresholds
    is_high_quality = (
        sharpness > 200 and
        colorfulness > 50 and
        80 < brightness < 200 and
        contrast > 50 and
        resolution > 1000000
    )

    # Convert image to Base64 for preview
    _, buffer = cv2.imencode(".png", image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return {
        "filename": uploaded_file.filename,
        "image": image,
        "is_high_quality": is_high_quality,
        "preview": encoded_image
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    global high_quality, low_quality
    high_quality = []
    low_quality = []
    preview_high = []
    preview_low = []

    uploaded_files = request.files.getlist('file')

    # 🏎️ Parallel processing for speed optimization
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single_image, uploaded_files))

    for result in results:
        if result:
            if result["is_high_quality"]:
                high_quality.append((result["filename"], result["image"]))
                if len(preview_high) < 3:
                    preview_high.append(result["preview"])
            else:
                low_quality.append((result["filename"], result["image"]))
                if len(preview_low) < 3:
                    preview_low.append(result["preview"])

    return jsonify({
        "success": True,
        "preview_high": preview_high,
        "preview_low": preview_low,
        "high_quality": len(high_quality) > 0,
        "low_quality": len(low_quality) > 0
    })

@app.route('/download_high')
def download_high():
    return download_zip(high_quality, "high_quality_images.zip")

@app.route('/download_low')
def download_low():
    return download_zip(low_quality, "low_quality_images.zip")

def download_zip(image_list, zip_name):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, img in image_list:
            _, buffer = cv2.imencode(".png", img)
            zip_file.writestr(filename, buffer.tobytes())
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=zip_name)

if __name__ == '__main__':
    app.run(debug=True)
