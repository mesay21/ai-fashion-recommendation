"""Evaluate results with a trained model."""
import os
import sys
import json
import argparse
import h5py
import time
import numpy as np
import torch

#newdir = os.path.join('../','functions/models')

import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('../functions/')

# Now you can import your module
from model import Bi_lstm
#from create_mosaic import create_img_outfit
#from utils import TextTransforms, get_one_hot


# Disable superfluous-parens warning for python 3.
# pylint: disable=C0325
class Outfits:

    def __init__(self, model_path, feats_path, img_path, cuda, query, vocab):
        self.model_path = model_path#'../models/model_80000.pth'
        self.feats_path = feats_path#'../saved_features'
        self.img_path = img_path#'./saved_outfit_images'
        self.cuda = cuda#True
        #img_savepath = './save_generated_outfits'
        self.query = query#'query.json'
        self.vocab = vocab#'../data/vocab.json'


    def get_img_path(self, img_name):
        return 'data/images/' + img_name.replace('_', '/') + '.jgp'


    def run_one_lstm(self, model, feats, direction, hidden=None):
        if not hidden:
            out, hidden = model.lstm(torch.autograd.Variable(feats).unsqueeze(0))
        else:
            out, hidden = model.lstm(torch.autograd.Variable(feats).unsqueeze(0), hidden)
        out = out.data
        if direction == 'f':
            return out[0, :feats.size(0), :out.size(2) // 2][-1].view(1, -1), hidden
        elif direction == 'b':
            return out[0, :feats.size(0), out.size(2) // 2:][0].view(1, -1), hidden
        else:
            print("Specifiy a direction for lstm inference")
            return None


    def run_forward_lstm(self, model, prev_prod, answers_feats, data_dict, zero_idx, cuda):
        forward_seq = []
        while len(prev_prod) < 10:
            fw_hidden, _ = self.run_one_lstm(model, prev_prod, 'f')
            pred = self.predict_single_direction(torch.autograd.Variable(fw_hidden),
                                            torch.autograd.Variable(answers_feats),
                                            zero_idx)
            max_prob_img = list(data_dict.keys())[pred[0].data[0]]
            zero_prob = pred[2].data
            if max_prob_img == 'zeros' or zero_prob > 0.00005:
                break

            forward_seq.append(max_prob_img)
            new_prod = answers_feats[pred[0].data[0]].unsqueeze(0)
            # new_prod = torch.nn.functional.normalize(new_prod, p=2, dim=1)
            if cuda:
                new_prod = new_prod.cuda()
            prev_prod = torch.cat((prev_prod, new_prod))
        return forward_seq


    def run_backward_lstm(self,model, next_prod, answers_feats, data_dict, zero_idx, cuda):
        backward_seq = []
        while len(next_prod) < 10:
            bw_hidden, _ = self.run_one_lstm(model, next_prod, 'b')
            pred = self.predict_single_direction(torch.autograd.Variable(bw_hidden),
                                            torch.autograd.Variable(answers_feats),
                                            zero_idx)
            max_prob_img = list(data_dict.keys())[pred[0].data[0]]
            zero_prob = pred[2].data
            if max_prob_img == 'zeros' or zero_prob > 0.00005:
                break

            backward_seq.append(max_prob_img)
            new_prod = answers_feats[pred[0].data[0]].unsqueeze(0)
            if cuda:
                new_prod = new_prod.cuda()
            next_prod = torch.cat((new_prod, next_prod))
        return backward_seq[::-1]


    def run_fill_lstm(self,model, start_feats, end_feats, num_blank, answers_feats, data_dict, zero_idx, cuda):
        if num_blank == 0:
            return []

        # TODO: merge this two for loops
        forward_seq = []
        forward_hiddens = []
        for i in range(num_blank):
            fw_hidden, _ = self.run_one_lstm(model, start_feats, 'f')
            forward_hiddens.append(fw_hidden)
            pred = self.predict_single_direction(torch.autograd.Variable(fw_hidden),
                                            torch.autograd.Variable(answers_feats),
                                            zero_idx)
            max_prob_img = list(data_dict.keys())[pred[0].data]
            forward_seq.append(max_prob_img)
            new_prod = answers_feats[pred[0].data[0]].unsqueeze(0)
            if cuda:
                new_prod = new_prod.cuda()
            start_feats = torch.cat((start_feats, new_prod))
        forward_hiddens = torch.stack(forward_hiddens)

        backward_seq = []
        backward_hiddens = []
        for i in range(num_blank):
            bw_hidden, _ = self.run_one_lstm(model, end_feats, 'b')
            backward_hiddens.append(bw_hidden)
            pred = self.predict_single_direction(torch.autograd.Variable(bw_hidden),
                                            torch.autograd.Variable(answers_feats),
                                            zero_idx)
            max_prob_img = list(data_dict.keys())[pred[0].data[0]]
            print(pred,'pre')
            zero_prob = pred[2].data
            if max_prob_img == 'zeros' or zero_prob > 0.00005:
                break

            backward_seq.append(max_prob_img)
            new_prod = answers_feats[pred[0].data[0]].unsqueeze(0)
            if cuda:
                new_prod = new_prod.cuda()

            print(new_prod, end_feats)
            end_feats = torch.cat((new_prod, end_feats))
        backward_hiddens = torch.stack(backward_hiddens)

        hiddens = forward_hiddens + backward_hiddens
        hiddens = hiddens.view(len(hiddens), -1)
        blank_scores = torch.nn.functional.log_softmax(torch.autograd.Variable(torch.mm(
                                                 hiddens, answers_feats.permute(1, 0))), dim=1)
        _, blank_idxs = torch.max(blank_scores, 1)
        print(blank_idxs)
        for idx in blank_idxs:
            print(idx,'idx')
            #print(idx.data[0])
            ####################
        blank_imgs = [list(data_dict.keys())[idx.data] for idx in blank_idxs]
        return blank_imgs

    def predict_single_direction(self,ht, feats, zero_pos):
        scores = torch.nn.functional.log_softmax(torch.mm(ht, feats.permute(1, 0)), dim=1)
        maxv, idx = torch.max(scores, 1)
        return idx, torch.exp(maxv), torch.exp(scores[0, zero_pos])


    def predict_multi_direction(self,hf, hb, feats, zero_pos):
        scores = torch.nn.functional.log_softmax(torch.mm(hf, feats.permute(1, 0)), dim=1) + \
                 torch.nn.functional.log_softmax(torch.mm(hb, feats.permute(1, 0)), dim=1)
        maxv, idx = torch.max(scores, 1)
        return idx, torch.exp(maxv), torch.exp(scores[0, zero_pos])


    def nn_search(self,img, text_feat, data_dict, answers_feats, cuda, balance_factor = 2):
        img_feat = torch.from_numpy(data_dict[img]).unsqueeze(0)
        img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=1)
        if cuda:
            img_feat = img_feat.cuda()
        scores = torch.nn.functional.log_softmax(torch.autograd.Variable(
                                                 torch.mm((img_feat + balance_factor * text_feat.data).view(1, 512),
                                                          answers_feats.permute(1, 0))), dim=1)
        _, idx = torch.max(scores, 1)
        return list(data_dict.keys())[idx.data[0]]


    # Disable too-many-locals. No clear way to reduce them
    # pylint: disable= R0914
    def generate(self, model_name, feats_name, img_savepath, query_file, vocab_file, cuda):
        """Main function."""
        queries = json.load(open(query_file))
        vocab = json.load(open(vocab_file))

        data = h5py.File(feats_name, 'r')
        data_dict = dict()
        for fname, feat in zip(data['filenames'], data['features']):
            data_dict[fname] = feat


        model = Bi_lstm(512, 512, 2758, batch_first=True, dropout=0.7, batch_size=20)


        """Load the model weights."""
        if cuda:
            model = model.cuda()
            model.load_state_dict(torch.load(model_name))
        else:
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        model.eval()

        #txt_trf = TextTransforms()
        # pylint: disable=W0108
        #txt_norm = lambda x: txt_trf.normalize(x)

        # First of all, add a zero vector to the features data for start/stop.
        data_dict['zeros'] = np.zeros_like(data['features'][0])
        zero_idx = list(data_dict.keys()).index('zeros')

        answers_feats = torch.from_numpy(np.array(list(data_dict.values())))
        answers_feats = torch.nn.functional.normalize(answers_feats, p=2, dim=1)
        if cuda:
            answers_feats = answers_feats.cuda()

        for nq, query in enumerate(queries):
            # Now, generate outfit for one image (forward and backward prediction until start/stop):
            #query_feats = torch.from_numpy(np.array([data_dict[q] for q in query['image_query']]))
            query_feats = torch.from_numpy(np.array([data_dict[str.encode(q)] for q in query['image_query']]))
            query_feats = torch.nn.functional.normalize(query_feats, p=2, dim=1)

            if cuda:
                query_feats = query_feats.cuda()

            first_prod = query_feats[0].unsqueeze(0)  # Start with the first image
            # Forward prediction
            forward_seq = self.run_forward_lstm(model, first_prod, answers_feats, data_dict, zero_idx, cuda)
            # Backward prediction
            backward_seq = self.run_backward_lstm(model, first_prod, answers_feats, data_dict, zero_idx, cuda)

            # Concatenate full sequence (forward + backward) generated by first product
            first_sequence = backward_seq + [query['image_query'][0]] + forward_seq

            for i in range(len(first_sequence)):
                if type(first_sequence[i]) != str:
                    #print(i)
                    first_sequence[i] = first_sequence[i].decode('UTF-8')

            seq_feats = torch.from_numpy(np.array([data_dict[str.encode(im)] for im in first_sequence]))
            seq_feats = torch.nn.functional.normalize(seq_feats, p=2, dim=1)
            if cuda:
                seq_feats = seq_feats.cuda()

            # If there are more images, substitute the nearest one by the query and recompute:
            if len(query['image_query']) >= 2:
                positions = [len(backward_seq)]  # Position of the first query in the sequence
                for i, img in enumerate(query['image_query'][1:]):
                    # Find NN of the next item
                    dists = torch.mm(query_feats[i + 1].unsqueeze(0), seq_feats.permute(1, 0))
                    _, idx = torch.max(dists, 1)
                    positions.append(idx)

                start_pos = np.min(positions)
                end_pos = np.max(positions)
                if start_pos == positions[0]:
                    start_feats = query_feats[0].unsqueeze(0)
                    end_feats = query_feats[i + 1].unsqueeze(0)
                    start_item = query['image_query'][0]
                    end_item = query['image_query'][i + 1]
                elif end_pos == positions[0]:
                    start_feats = query_feats[i + 1].unsqueeze(0)
                    end_feats = query_feats[0].unsqueeze(0)
                    start_item = query['image_query'][i + 1]
                    end_item = query['image_query'][0]

                blanks = self.run_fill_lstm(model, start_feats, end_feats, end_pos - start_pos - 1,
                                     answers_feats, data_dict, zero_idx, cuda)
                sets = [start_item] + blanks + [end_item]

                for i in range(len(sets)):
                    if type(sets[i]) != str:
                        #print(i)
                        sets[i] = sets[i].decode('UTF-8')

                sets_feats = torch.from_numpy(np.array([data_dict[str.encode(im)] for im in sets]))
                sets_feats = torch.nn.functional.normalize(sets_feats, p=2, dim=1)
                if cuda:
                    sets_feats = sets_feats.cuda()

                # run bi LSTM again
                forward_seq = self.run_forward_lstm(model, sets_feats, answers_feats, data_dict, zero_idx, cuda)
                backward_seq = self.run_backward_lstm(model, sets_feats, answers_feats, data_dict, zero_idx, cuda)
                sets = backward_seq + sets + forward_seq
                positions = [len(backward_seq), len(sets) - len(forward_seq) - 1]

            else:
                sets = backward_seq + query['image_query'] + forward_seq

            if len(query['text_query']):
                text_query = txt_norm(query['text_query'])
                texts = torch.stack([get_one_hot(word, vocab) for word in text_query.split()])
                texts = torch.autograd.Variable(texts)
                if cuda:
                    texts = texts.cuda()
                text_query_feat = model.textn(texts)
                text_query_feat = torch.mean(text_query_feat.view(len(text_query_feat), -1), 0)
                text_query_feat = torch.nn.functional.normalize(text_query_feat.unsqueeze(0), p=2, dim=1)

                sets_text = sets[:]
                for i, j in enumerate(sets):
                    if j not in query['image_query']:
                        sets_text[i] = nn_search(j, text_query_feat, data_dict, answers_feats, cuda)

            print(sets)
            #create_img_outfit(sets, positions, os.path.join(img_savepath, "%d.jpg" % nq))
            #create_img_outfit(sets_text, positions, os.path.join(img_savepath, "%d_%s.jpg" % (nq, text_query)))
            return sets

    print("BEFORE X")
    # main_single_prev(ARGS.model_path, ARGS.model_type, ARGS.feats_path, ARGS.img_path, ARGS.cuda)
    #x = main(model_path, feats_path, img_path, query, vocab, cuda)
    print("AFTER X")
