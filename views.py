from flask import request, render_template, Blueprint
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
import numpy as np
from PIL import Image

views = Blueprint('views', __name__)

views.config = {
    'INITIAL_FILE_UPLOADS': 'static/uploads',
    'EXISTING_FILES': 'static/original',
    'GENERATED_FILE': 'static/generated'
}

# Define your routes within the blueprint
@views.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    
    if request.method == 'POST':
        file_upload = request.files['file_upload']
        file_name = file_upload.filename

        original_image = Image.open(os.path.join(views.config['EXISTING_FILES'], 'original.png')).resize((340, 200))
        original_image.save(os.path.join(views.config['EXISTING_FILES'], 'image.jpg'))

        uploaded_image = Image.open(file_upload).resize((340, 200))
        uploaded_image = uploaded_image.convert('RGB')
        uploaded_image.save(os.path.join(views.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

        # Convert PIL.Image to numpy array
        original_np = np.array(original_image)
        uploaded_np = np.array(uploaded_image)

        # Convert color mode if necessary
        original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        uploaded_np = cv2.cvtColor(uploaded_np, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        original_gray = cv2.cvtColor(original_np, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_np, cv2.COLOR_BGR2GRAY)

        # Calculate structural similarity
        (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
        diff = (diff * 255).astype('uint8')

        # Find contours only on the differences
        thres = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw rectangles only on the tampered image based on differences
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(uploaded_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save images with rectangles drawn only on differences
        # uploaded_image_path = 'generated/image_uploaded_with_rectangles.jpg'
        uploaded_image_path='uploads/image.jpg'
        original_image_path= 'original/original.png'
        diff_image_path='generated/image_diff.jpg'
        cv2.imwrite(os.path.join(views.config['GENERATED_FILE'], 'image_uploaded_with_rectangles.jpg'), uploaded_np)
        cv2.imwrite(os.path.join(views.config['GENERATED_FILE'], 'image_diff.jpg'), diff)
        cv2.imwrite(os.path.join(views.config['GENERATED_FILE'], 'image_thresh.jpg'), thres)

        return render_template('index.html', pred=str(round(score * 100, 2)) + '%' + ' correct',original_image=original_image_path,
                               uploaded_image=uploaded_image_path,diff_image=diff_image_path)
