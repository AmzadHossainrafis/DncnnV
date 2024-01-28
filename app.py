import sys 
import os 
import torch
import numpy as np
from PIL import Image
import albumentations as A
from flask import Flask, render_template, request,jsonify  
from dncnn.components.model import DnCNN 
from albumentations.pytorch import ToTensorV2
from dncnn.utils.common import read_config
from dncnn.utils.logger import logger
from dncnn.utils.exception import CustomException
from dncnn.utils.common import denormalize


config = read_config(r"C:\Users\Amzad\Desktop\Dncnn\config\config.yaml")
prediction_config = config["Prediction_config"]


model = DnCNN() # change this according to
model.load_state_dict(torch.load(prediction_config['model_path'], map_location=torch.device('cpu')))
model.eval()

prediction_transform = A.Compose([
    A.Resize(prediction_config['transfortm']['image_size'], prediction_config['transfortm']['image_size']),
    A.Normalize(
        mean=prediction_config['transfortm']['normalization']['mean'],
        std=prediction_config['transfortm']['normalization']['std'],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])



app = Flask(__name__)

@app.route("/predict/<image>", methods=["POST"])
def predict(image):
    try:
        preds ,input = get_prediction(image)
        return render_template("index.html", input=input, preds=preds)
    except CustomException as e:
        logger.error(e)
        return render_template("index.html", error=CustomException(e, sys))
    except Exception as e:
        logger.error(e)
        return render_template("index.html", error="Something went wrong. Please try again later.")
    


def get_prediction(image):
    image = Image.open(image)
    image = np.array(image)
    input = image.copy()
    image = prediction_transform(image=image)["image"]
    # permote the channel to the first dimension as pytorch expects the channel to be the first dimension 
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        preds = denormalize(preds.squeeze(0).squeeze(0), prediction_config['normalization']['mean'], prediction_config['normalization']['std'])
        preds = preds.detach().numpy()
        preds = preds.astype(np.uint8)
    return jsonify(preds.tolist()), {"input": input.tolist()}

    


@app.route("/")
def index():
    return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True, port=8000) 
