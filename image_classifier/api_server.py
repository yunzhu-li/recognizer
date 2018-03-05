'''
yl-recognizer
Copyright (C) 2017-2018 Yunzhu Li

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

# Train
import sys
import flask
from flask import request
import data_utils
import model

# Config
trained_model_path = 'trained_models/180301_tiny_imagenet_weights.h5'

# Global model var
predict_model = None

# Create Flask app
app = flask.Flask(__name__)


# Health route
@app.route('/health')
def health():
    return 'ok'


# Annotate route
@app.route('/images/annotate', methods=['POST'])
def annotate():
    status = 200
    results = None
    error = None

    # Accept POST requests
    if request.method == 'POST':
        try:
            # Read data
            img_bytes = request.files['image'].read()

            # Annotate
            results = annotate_image(img_bytes)
        except:
            status = 400
            error = 'Failed to receive and process image'
    else:
        status = 400
        error = 'Method not supported'

    return flask.jsonify({'results': results, 'error': error}), status


# Annotate
def annotate_image(image_bytes):
    # Decode and pre-process
    img_array = data_utils.image_array_from_bytes(image_bytes)

    # Predict and decode result
    preds = predict_model.predict(img_array, verbose=0)
    results = data_utils.decode_predictions(preds)
    if len(results) > 0:
        return results[0]

    return None


def main():
    print('Starting API server...')
    print('Loading model...')
    global predict_model
    predict_model, _ = model.load(trained_model_path)

    # Serve
    app.run(host='0.0.0.0', port=8000)

if __name__ == "__main__":
    main()
