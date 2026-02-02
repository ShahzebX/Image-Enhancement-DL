"""
AI Image Enhancement Studio - Unified Flask Application
Combines Image Colorization and Low-Light Enhancement
"""

from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io
from PIL import Image
import traceback

# Import MIRNet enhancer
from mirnet_model import create_enhancer

# ========================================
# FLASK APP INITIALIZATION
# ========================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# ========================================
# LOAD MODELS ON STARTUP
# ========================================
print("=" * 60)
print("🚀 Initializing AI Image Enhancement Studio")
print("=" * 60)

# Load colorization model
print("\n📦 Loading Colorization Model...")
try:
    PROTOTXT = "models/colorization_deploy_v2.prototxt"
    MODEL = "models/colorization_release_v2.caffemodel"
    POINTS = "models/pts_in_hull.npy"
    
    colorization_net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    
    class8 = colorization_net.getLayerId("class8_ab")
    conv8 = colorization_net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    colorization_net.getLayer(class8).blobs = [pts.astype("float32")]
    colorization_net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    print("✅ Colorization model loaded successfully")
    COLORIZATION_AVAILABLE = True
except Exception as e:
    print(f"❌ Failed to load colorization model: {e}")
    COLORIZATION_AVAILABLE = False

# Load MIRNet enhancer
print("\n📦 Loading Light Enhancement Model...")
try:
    mirnet_enhancer = create_enhancer(use_deep_learning=False)
    print("✅ MIRNet enhancer loaded successfully")
    MIRNET_AVAILABLE = True
except Exception as e:
    print(f"❌ Failed to load MIRNet enhancer: {e}")
    MIRNET_AVAILABLE = False

print("\n" + "=" * 60)
print("✅ Server ready to process images!")
print("=" * 60 + "\n")

# ========================================
# COLORIZATION FUNCTION
# ========================================
def colorize_image(image):
    """
    Colorize a grayscale or B&W image
    
    Args:
        image: Input image (numpy array, BGR format)
        
    Returns:
        Colorized image (numpy array, BGR format)
    """
    # Normalize image
    scaled = image.astype("float32") / 255.0
    
    # Convert to LAB color space
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Resize L channel to 224x224 for model input
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # Mean subtraction
    
    # Predict ab channels
    colorization_net.setInput(cv2.dnn.blobFromImage(L))
    ab = colorization_net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize ab to match original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    
    # Get L channel from original image
    L = cv2.split(lab)[0]
    
    # Combine L and ab channels
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    # Convert back to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    
    # Convert to 8-bit
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

# ========================================
# ENHANCEMENT FUNCTION
# ========================================
def enhance_image(image):
    """
    Enhance a low-light image
    
    Args:
        image: Input image (numpy array, BGR format)
        
    Returns:
        Enhanced image (numpy array, BGR format)
    """
    return mirnet_enhancer.enhance_image(image)

# ========================================
# UTILITY FUNCTIONS
# ========================================
def read_image_from_request(file):
    """
    Read image from Flask request file
    
    Args:
        file: FileStorage object from request
        
    Returns:
        Image as numpy array in BGR format
    """
    # Read image file
    image_stream = io.BytesIO(file.read())
    pil_image = Image.open(image_stream).convert('RGB')
    
    # Convert to numpy array and BGR format
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

def encode_image_to_bytes(image):
    """
    Encode numpy image array to JPEG bytes
    
    Args:
        image: Numpy array in BGR format
        
    Returns:
        BytesIO object containing JPEG image
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes
    img_io = io.BytesIO()
    pil_image.save(img_io, 'JPEG', quality=95)
    img_io.seek(0)
    
    return img_io

# ========================================
# ROUTES
# ========================================
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    """
    Process uploaded image based on selected mode
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'mode' not in request.form:
            return jsonify({'error': 'No mode specified'}), 400
        
        file = request.files['image']
        mode = request.form['mode']
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate mode
        if mode not in ['colorize', 'enhance']:
            return jsonify({'error': 'Invalid mode'}), 400
        
        # Read image
        image = read_image_from_request(file)
        
        # Process based on mode
        if mode == 'colorize':
            if not COLORIZATION_AVAILABLE:
                return jsonify({'error': 'Colorization model not available'}), 503
            
            print(f"🎨 Colorizing image: {file.filename}")
            processed_image = colorize_image(image)
            
        elif mode == 'enhance':
            if not MIRNET_AVAILABLE:
                return jsonify({'error': 'Enhancement model not available'}), 503
            
            print(f"💡 Enhancing image: {file.filename}")
            processed_image = enhance_image(image)
        
        # Encode result
        result_bytes = encode_image_to_bytes(processed_image)
        
        print(f"✅ Processing complete: {file.filename}")
        
        # Return processed image
        return send_file(
            result_bytes,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=f'processed_{file.filename}'
        )
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'colorization_available': COLORIZATION_AVAILABLE,
        'enhancement_available': MIRNET_AVAILABLE
    })

# ========================================
# ERROR HANDLERS
# ========================================
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File size exceeds 10MB limit'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

# ========================================
# MAIN
# ========================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Starting Flask server...")
    print("📍 Access the application at: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )