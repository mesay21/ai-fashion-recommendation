import streamlit as st
import pandas as pd
import numpy as np
import cv2
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.imagenet_utils import preprocess_input
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#from PIL import image
MONTAGE_COL = 5
from PIL import Image, ImageFont, ImageDraw


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def build_montages(image_paths, image_shape, montage_shape):
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    # start with black canvas to draw images onto

    montage_image = 255 * np.ones(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3), dtype=np.uint8)
    #print(montage_image)
    cursor_pos = [0, 0]
    start_new_img = False

    count = 0
    

    for path in image_paths:
        # if type(img).__module__ != np.__name__:
        #     raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        #img = image.load_img(path)    
        #img = img.resize(image_shape)
        #img = os.path.join(path)
        img = Image.open(path)
        #
        img = img.convert('RGB')
        #
        width = img.size[0]
        height = img.size[1]

        print("BEFORE SIZES", img.size)

        if width == height:
            img = img.resize(image_shape)
        else:
            img = make_square(img, fill_color=(255, 255, 255, 255)).resize(image_shape)


            
        #font = ImageFont.truetype("BebasNeue.ttf", 45)
        #img = img.resize(image_shape)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('../data/Font.ttf', 30)
        draw.text((0,0), str(count), (0,0,0), font=font)
        print("AFTER SIZES", img.size)

        img = np.array(img)
        img = img[:,:,:3]

        

        # img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = 255 * np.ones(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3), 
                    dtype=np.uint8)
                start_new_img = True
        count +=1 

    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    image_montages = np.asarray(image_montages)
    return image_montages



def load_image(path, target_size=None):
    # TODO: compare to vgg19.preprocess input

    x = Image.open(path)
    #
    img = x.convert('RGB')
    #
    width = img.size[0]
    height = img.size[1]

    print("BEFORE SIZES", img.size)

    if width == height:
        img = img.resize((200,200))
    else:
        img = make_square(img, fill_color=(255, 255, 255, 255)).resize((200,200))

    img = np.array(img)
    img = img[:,:,:3]
    #img = image.load_img(path, target_size=target_size)
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    return img,  img


st.title('FASHION RECOMMENDER')
st.markdown(
"""
Input your items, we will find you the most compatible items based on decision behavior of fashion stylists.
""")



arr = ['tops','bottoms','shoes','bags','accessories']
dic = {}

data_dir = '../data/images/'


dic['tops']=['question.jpg','192637078/1.jpg','208454316/1.jpg','207613236/1.jpg','212427289/1.jpg','150649189/4.jpg',              '192668803/1.jpg','215989339/2.jpg','213451868/5.jpg','214878634/1.jpg']
dic['bottoms'] = ['question.jpg','163560873/2.jpg','150649189/1.jpg','202412432/2.jpg','209653609/3.jpg','212427289/3.jpg',         '190818551/1.jpg','192668803/3.jpg','190902727/2.jpg','210963952/2.jpg']
dic['shoes'] = ['question.jpg','119704139/4.jpg','119314458/3.jpg','147361785/3.jpg','192637078/4.jpg','163560873/3.jpg',           '215989339/3.jpg','192668803/4.jpg','198660399/6.jpg','170475795/3.jpg']
dic['bags'] = ['question.jpg','147361785/4.jpg','163560873/4.jpg','216061834/5.jpg','195200115/5.jpg','148787971/5.jpg',            '199071284/4.jpg','216191160/5.jpg','185402098/4.jpg','199353590/5.jpg']
dic['accessories'] = ['question.jpg','196280291/6.jpg','195200115/6.jpg','216061834/6.jpg','192637078/6.jpg','212427289/7.jpg',     '128829024/5.jpg','215483702/4.jpg','215483702/7.jpg','213451868/3.jpg']


# pull these images from data/images and save them in the website



for key in dic.keys():
	for i in range(len(dic[key])):
		dic[key][i] = data_dir + dic[key][i]
		print(dic[key][i])



### TOP ### 
st.sidebar.subheader('pick a top')

REF_IMG_PATHS = dic['tops']
#print(np.array(cv2.imread(REF_IMG_PATHS[0])).shape)
st.header('Tops')
montage = build_montages(REF_IMG_PATHS, (200, 200), (MONTAGE_COL, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, width=None)# use_column_width=True)


ref_id0 = st.sidebar.text_input('Enter preferred furniture ID', '0', key='a')
assert ref_id0.isnumeric(), 'Please enter a number'

ref_img, x = load_image(REF_IMG_PATHS[int(ref_id0)])
st.sidebar.image(ref_img, use_column_width=True)



### BOTTOM ### 
st.sidebar.subheader('pick a bottom')

REF_IMG_PATHS = dic['bottoms']
#print(np.array(cv2.imread(REF_IMG_PATHS[0])).shape)
st.header('Bottoms')
montage = build_montages(REF_IMG_PATHS, (200, 200), (MONTAGE_COL, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, width=None)


ref_id1 = st.sidebar.text_input('Enter preferred furniture ID', '0', key='b')
assert ref_id1.isnumeric(), 'Please enter a number'

ref_img, x = load_image(REF_IMG_PATHS[int(ref_id1)])
st.sidebar.image(ref_img, use_column_width=True)



### SHOES ### 
st.sidebar.subheader('pick shoes')

REF_IMG_PATHS = dic['shoes']
#print(np.array(cv2.imread(REF_IMG_PATHS[0])).shape)
st.header('Shoes')
montage = build_montages(REF_IMG_PATHS, (200, 200), (MONTAGE_COL, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, width=None)


ref_id2 = st.sidebar.text_input('Enter preferred furniture ID', '0', key='c')
assert ref_id2.isnumeric(), 'Please enter a number'

ref_img, x = load_image(REF_IMG_PATHS[int(ref_id2)])
st.sidebar.image(ref_img, use_column_width=True)

### BAGS ### 
st.sidebar.subheader('pick a bag')

REF_IMG_PATHS = dic['bags']
#print(np.array(cv2.imread(REF_IMG_PATHS[0])).shape)
st.header('Bags')
montage = build_montages(REF_IMG_PATHS, (200, 200), (MONTAGE_COL, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, width=None)


ref_id3 = st.sidebar.text_input('Enter preferred furniture ID', '0', key='d')
assert ref_id3.isnumeric(), 'Please enter a number'

ref_img, x = load_image(REF_IMG_PATHS[int(ref_id3)])
st.sidebar.image(ref_img, use_column_width=True)



### ACCESSORIES ### 
st.sidebar.subheader('pick an accessory')

REF_IMG_PATHS = dic['accessories']
#print(np.array(cv2.imread(REF_IMG_PATHS[0])).shape)
st.header('Accessories')
montage = build_montages(REF_IMG_PATHS, (200, 200), (MONTAGE_COL, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, width=None)


ref_id4 = st.sidebar.text_input('Enter preferred furniture ID', '0', key='e')
assert ref_id4.isnumeric(), 'Please enter a number'


ref_img, x = load_image(REF_IMG_PATHS[int(ref_id4)])
st.sidebar.image(ref_img, use_column_width=True)


##### COLLECT ALL THE INPUTS FROM THE USER and PROVIDE THE RECOMMENDATION ##### 

import generate

model_path = '../models/model_85000.pth'
feats_path = '../saved_features'
img_path = './saved_outfit_images'
cuda = False
#img_savepath = './save_generated_outfits'
#query = 'query.json'
vocab = '../data/vocab.json'

query_dic = {}
query_dic["image_query"] = []
arr = [ref_id0, ref_id1, ref_id2, ref_id3, ref_id4]
arr_types=['tops','bottoms','shoes','bags','accessories']

for i in range(len(arr)):
	#print(ref_id)
    ref_id = arr[i]
    if ref_id != '0':
        path_to_img = dic[arr_types[i]][int(ref_id)]
        splitted = path_to_img.split("/")
        query_dic["image_query"].append(splitted[-2] +'_' + splitted[-1].split('.')[0])

#import json
#with open('query.json','w') as output:
#	json.dump([query_dic],output)

print(query_dic)

#query = 'query.json'
query = [query_dic]

from generate import Outfits

outfit = Outfits(model_path, feats_path, img_path, cuda, query, vocab)

result_outfits = outfit.generate(model_path, feats_path, img_path, query, vocab, cuda)

for i in range(len(result_outfits)):
    if type(result_outfits[i]) == str:
        result_outfits[i] = result_outfits[i].encode()

outfit_names = ['../data/images/' + str(i.decode()).replace('_', '/') + '.jpg' for i in result_outfits]
print(outfit_names)
print("OUTFIT_NAMES")


### SHOW RESULTS ON WEBSITE

#st.sidebar.header('GENERATED OUTFITS')

REF_IMG_PATHS = outfit_names
#print(np.array(cv2.imread(REF_IMG_PATHS[0])).shape)
st.header('GENERATED OUTFITS')
montage = build_montages(REF_IMG_PATHS, (250, 250), (6, math.ceil(len(REF_IMG_PATHS)/MONTAGE_COL)))
st.image(montage, width=None)#use_column_width=True)


#ref_id4 = st.sidebar.text_input('Enter preferred furniture ID', '0', key='e')
#assert ref_id4.isnumeric(), 'Please enter a number'


#ref_img, x = load_image(REF_IMG_PATHS[int(ref_id4)])
#st.sidebar.image(ref_img, use_column_width=True)


