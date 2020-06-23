import os
import sys
import json
import time
import argparse
import h5py
import torch
from model import Bi_lstm
from data import Preprocess, collate_seq
from eval import Evaluation
from torchvision import transforms

from PIL import Image


def generate_img_features(model_name, feats_filename):
    print("WHY-4",feats_filename) 
    if not os.path.exists(feats_filename):
        print("WHY-3")  
        batch_size = 1

        model = Bi_lstm(512, 512, 2758, batch_first=True, dropout=0.7, batch_size=20)
        print("WHY-2") 
        evaluator = Evaluation(model, model_name, 'data/images',batch_first=True, cuda=True)
        print("WHY-1")    
        json_filenames = {'train': 'train_no_dup.json',
                          'test': 'test_no_dup.json',
                          'val': 'valid_no_dup.json'}


        img_transforms = {
            'train': transforms.Compose([
                transforms.Resize((299,299)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                #transforms.Resize(256),
                transforms.Resize((299,299)),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((299,299)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        img_dir, json_dir = '../data/images', '../data/label'
        dataloaders = {x: torch.utils.data.DataLoader(
            Preprocess(os.path.join(json_dir, json_filenames[x]), img_dir,
                            img_transform=img_transforms[x]),
            batch_size=batch_size,
            shuffle=False, num_workers=4,
            collate_fn=collate_seq)
                       for x in ['test']}
        print("WHY0")                
        test_files = json.load(open(os.path.join(json_dir, json_filenames['test'])))

        filenames = []
        features = torch.Tensor().cuda()



        for i, (test_file, batch) in enumerate(zip(test_files, dataloaders['test'])):
            print("WHY1")
            if i == 0:
                tic = time.time()
            sys.stdout.write("%d/%d sets - %.2f secs remaining\r" % (i, len(test_files),
                                                                     (time.time() - tic)/
                                                                     (i + 1)*(len(test_files) - i)))
            sys.stdout.flush()
            set_id = test_file['set_id']
            im_idxs = [x['index'] for x in test_file['items']]


            #print(batch[0]['images'])
            #im_feats = evaluator.get_img_feats(Image.fromarray((batch[0]['images'])))
            im_feats = evaluator.get_img_feats((batch[0]['images']))
            for idx in im_idxs:
                filenames.append(set_id + '_' + str(idx))
            features = torch.cat((features, im_feats.data))
            for ignored in batch[0]['ignored']:
                filenames.remove(ignored)
        if not os.path.exists(os.path.dirname(feats_filename)):
            os.makedirs(os.path.dirname(feats_filename))
        filenames = [n.encode("ascii", "ignore") for n in filenames]
        savefile = h5py.File(feats_filename, 'w')
        savefile.create_dataset('filenames', data=filenames)
        savefile.create_dataset('features', data=features.cpu().numpy())
        savefile.close()


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--load_model', '-m', type=str, help='load the model')
PARSER.add_argument('--save_features', '-f', type=str, help='save the features', default='../saved_features')
ARGS = PARSER.parse_args()

import os
os.system("rm ../saved_features")

generate_img_features(ARGS.load_model, ARGS.save_features)
