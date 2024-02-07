import sys 
import os 
import torch
import datetime
import numpy as np
from PIL import Image
import albumentations as A
from flask import Flask, render_template, request,jsonify, flash, request, redirect, url_for, send_from_directory
from dncnn.components.model import DnCNN 
from albumentations.pytorch import ToTensorV2
from dncnn.utils.common import read_config
from dncnn.utils.logger import logger
from dncnn.utils.exception import CustomException
from dncnn.utils.common import denormalize

current_time = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

config = read_config("config/config.yaml")
prediction_config = config["Prediction_config"]

UPLOAD_FOLDER = './static/uploads/'
RESTORED_FOLDER = './static/restored_images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


model = DnCNN() # change this according to
model.load_state_dict(torch.load(prediction_config['model_path'], map_location=torch.device('cpu')))
model.eval()

prediction_transform = A.Compose([
    A.Normalize(
        mean=prediction_config['transform']['normalization']['mean'],
        std=prediction_config['transform']['normalization']['std'],
        max_pixel_value=255.0,
        p=1
    ),
    A.Resize(prediction_config['transform']['image_size'], prediction_config['transform']['image_size'], p=1),
    ToTensorV2(),
])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESTORED_FOLDER'] = RESTORED_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("error.html", error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template("error.html", error="No file selected")
        if file and allowed_file(file.filename):
            try:
                preds = get_prediction(file)
                preds_reshaped = np.moveaxis(preds, 0, -1)
                # preds_reshaped = preds_reshaped.astype(np.uint8)
                converted_image = Image.fromarray(preds_reshaped, mode='RGB')
                # name the file with todays date and time and save it in the static/uploads folder
                filename = f"converted_on_{current_time}.jpg"
                converted_image.save(os.path.join(app.config['RESTORED_FOLDER'], filename))
                return render_template("success.html", filename=filename)
            except CustomException as e:
                logger.error(e)
                return render_template("error.html", error=CustomException(e, sys))
            except Exception as e:
                logger.error(e)
                return render_template("error.html", error="Something went wrong. Please try again later.")
    return render_template("index.html")

@app.route('/restored-images/<filename>')
def restored_file(filename):
    return send_from_directory(app.config['RESTORED_FOLDER'], filename)

def get_prediction(image):
    image = Image.open(image)
    image = np.array(image)
    image = prediction_transform(image=image)["image"]
    # permute the channel to the first dimension as pytorch expects the channel to be the first dimension 
    image = image.permute(0, 2, 1)
    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        logger.info(preds.squeeze(0).squeeze(0).shape)
        preds = denormalize(preds.squeeze(0).squeeze(0).numpy(), prediction_config['transform']['normalization']['mean'], prediction_config['transform']['normalization']['std'])
        # preds = preds.astype(np.uint8)
    return preds

if __name__ == "__main__":
    app.run(debug=True, port=8000) 