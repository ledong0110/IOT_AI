import json
import torch
from model import FertilzerModel
from flask import Flask, jsonify, request


app = Flask(__name__)
model_area1 = FertilzerModel()
model_area2 = FertilzerModel()
model_area3 = FertilzerModel()

model_area1.load_state_dict(torch.load('ckpt/model_area1.pth'))
model_area2.load_state_dict(torch.load('ckpt/model_area2.pth'))
model_area3.load_state_dict(torch.load('ckpt/model_area3.pth'))

model_area1.eval()
model_area2.eval()
model_area3.eval()


def get_prediction(seq, area):
    if area == 1:
        model = model_area1
    elif area == 2:
        model = model_area2
    else:
        model = model_area3

    with torch.no_grad():
        seq = torch.tensor(seq).unsqueeze(0)
        output = model(seq)
        pred = output.tolist()
        return pred


@app.route('/predict', methods=['POST'])
def predict():
    seq = request.json['seq']
    area = request.json['area']
    pred = get_prediction(seq, area)
    return jsonify({'prediction': pred})


if __name__ == '__main__':
    app.run()