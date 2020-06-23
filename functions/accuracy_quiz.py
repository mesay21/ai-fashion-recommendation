
import os
import sys
import json

import h5py
import time
import numpy as np
import torch
from model import Bi_lstm 

from utils import predict_single_direction, predict_multi_direction

import argparse 


def main(model_name, feats_name, img_savepath, cuda):
    # 1) open json multiple choice json file 2) load features name 3) create data dictionary
    outfit_file = '../data/label/fill_in_blank_test.json'
    fitb = json.load(open(outfit_file))
    print(feats_name)
    data = h5py.File(feats_name, 'r')
    data_dict = dict()
    for fname, feat in zip(data['filenames'], data['features']):
        data_dict[fname] = feat

    # 1) load model 2) load outfit items with one item missing 3) load answers 
    # 4) if the blank is in first, last, and middle
    model = Bi_lstm(512, 512, 2758, batch_first=True, dropout=0.0, batch_size = 20)

    if cuda:
        model = model.cuda()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    scores = []

    tic = time.time()
    #fitb = fitb[:100]
    for i, outfit in enumerate(fitb):
        sys.stdout.write('Outfit %d/%d - %2.f secs remaining\r' % (i, len(fitb), (time.time() - tic)/
                                                                   (i + 1)*(len(fitb) - i)))
        sys.stdout.flush()

        for i in range(len(outfit['question'])):
            if type(outfit['question'][i]) != str:
                outfit['question'][i] = outfit['question'][i].decode('UTF-8')

        

        for i in range(len(outfit['answers'])):
            if type(outfit['answers'][i]) != str:
                outfit['answers'][i] = outfit['answers'][i].decode('UTF-8')






        question_feats = torch.from_numpy(np.array([data_dict[str.encode(q)] for q in outfit['question']]))
        question_feats = torch.nn.functional.normalize(question_feats, p=2, dim=1)

        answers_feats = torch.from_numpy(np.array([data_dict[str.encode(a)] for a in outfit['answers']]))
        answers_feats = torch.nn.functional.normalize(answers_feats, p=2, dim=1)

        if cuda:
            question_feats = question_feats.cuda()
            answers_feats = answers_feats.cuda()

        position = outfit['blank_position'] - 1
        position_type = None

        if position == 0:
            out, _ = model.lstm(torch.autograd.Variable(question_feats).unsqueeze(0))
            out = out.data
            bw_hidden = out[0, :question_feats.size(0), out.size(2) // 2:][0].view(1, -1)
            pred = predict_single_direction(torch.autograd.Variable(bw_hidden),
                                            torch.autograd.Variable(answers_feats))
            position_type = 'START'

        elif position == len(question_feats):
            out, _ = model.lstm(torch.autograd.Variable(question_feats).unsqueeze(0))
            out = out.data
            fw_hidden = out[0, :question_feats.size(0), :out.size(2) // 2][-1].view(1, -1)
            pred = predict_single_direction(torch.autograd.Variable(fw_hidden),
                                            torch.autograd.Variable(answers_feats))
            position_type = 'END'

        else:
            prev = question_feats[:position]
            prev_out, _ = model.lstm(torch.autograd.Variable(prev).unsqueeze(0))
            prev_out = prev_out.data
            fw_hidden = prev_out[0, :prev.size(0), :prev_out.size(2) // 2][-1].view(1, -1)

            post = question_feats[position:]
            post_out, _ = model.lstm(torch.autograd.Variable(post).unsqueeze(0))
            post_out = post_out.data
            bw_hidden = post_out[0, :post.size(0), post_out.size(2) // 2:][0].view(1, -1)

            pred = predict_multi_direction(torch.autograd.Variable(fw_hidden),
                                           torch.autograd.Variable(bw_hidden),
                                           torch.autograd.Variable(answers_feats))

            position_type = 'MIDDLE'

        printing = False
        if printing:
            print('\n',"### NEXT ###",'\n')
            print('QUESTIONS: ', outfit['question'])
            print('ANSWERS: ', outfit['answers']) 
            print('BLANK POS: ', outfit['blank_position']-1)       
            print('LOCATION: ', position_type)
            print('PREDICTION: ', pred)

        scores.append(pred[0].data)

    count = 0
    for i in range(len(scores)):
        if scores[i].cpu().numpy() == 0:
            count = count + 1

    print(count/len(scores),'ACC')



PARSER = argparse.ArgumentParser()
PARSER.add_argument('--load_model', '-m', type=str, help='load the model')
PARSER.add_argument('--load_features', '-f', type=str, help='save the features', default='../saved_features')
PARSER.add_argument('--save_quiz_images', '-i', type=str, help='save the images', default='./saved_quiz_images')
PARSER.add_argument('--cuda', '-c', type=str, help='cuda is available?', default='True')
ARGS = PARSER.parse_args()


main(ARGS.load_model, ARGS.load_features, ARGS.save_quiz_images, ARGS.cuda)
