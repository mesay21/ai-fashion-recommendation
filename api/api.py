
# curl -X POST http://0.0.0.0:5000/generate -H "Content-Type: application/json" -d @query.json
from flask import Flask, request, jsonify, url_for
import numpy as np
from configs import set_args
import numpy as np
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import Image, ImageFont, ImageDraw
from generate import Outfits


app = Flask(__name__)

import generate
@app.route('/generate', methods=['POST'])
def predict_as_post():

	query = request.get_json()
	outfit = Outfits(model_path, feats_path, img_path, cuda, query, vocab)
	result_outfits = outfit.generate(model_path, feats_path, img_path, query, vocab, cuda)

	for i in range(len(result_outfits)):
	    if type(result_outfits[i]) == str:
	        result_outfits[i] = result_outfits[i].encode()

	outfit_names = ['../data/images/' + str(i.decode()).replace('_', '/') + '.jpg' for i in result_outfits]
	return jsonify(Recommended_Outfits=outfit_names)

if __name__ == '__main__':
    #args = set_args()
    model_path = '../models/model_85000.pth'
    feats_path = '../saved_features'
    cuda = False
    vocab = '../data/vocab.json'
    img_path = './generated_images'
    #print(model_path, feats_path, cuda, vocab)
    app.run(host='0.0.0.0', port=5000, threaded=False)