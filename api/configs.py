import argparse


def set_args():
    parser = argparse.ArgumentParser(description='DressUp')
    parser.add_argument('--mp', default='../models/model_85000.pth', help='path to saved model weights')
    parser.add_argument('--fp', default='../saved_features', help='path to saved feature representations')
    parser.add_argument('--cuda', default=False, help='with GPU or not')
    parser.add_argument('--v', default='../data/vocab.json', help='path to vocabulary')
    return parser.parse_args()