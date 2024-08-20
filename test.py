import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image, ImageDraw, ImageFont
import svgwrite
from models.models import Generator
# from models.models_improved import Generator

from misc.utils import _init_input, ID_COLOR, draw_masks, draw_graph, estimate_graph
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import glob
import cv2
import webcolors
import time


from image_utils import postprocessor



#Web:
from flask import Flask, send_from_directory, jsonify, request
import os
import base64
import shutil
import re
import io

postprocessor = postprocessor.PostProcessor()
parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--checkpoint", type=str, default='./checkpoints/pretrained.pth', help="checkpoint path")
parser.add_argument("--data_path", type=str, default='./data/sample_list.txt', help="path to dataset list file")
parser.add_argument("--out", type=str, default='./dump', help="output folder")
opt = parser.parse_args()
print(opt)

# Create output dir
os.makedirs(opt.out, exist_ok=True)

# Initialize generator and discriminator
model = Generator()
model.load_state_dict(torch.load(opt.checkpoint), strict=True) 
model = model.eval()

# Initialize variables
if torch.cuda.is_available():
    model.cuda()

# initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(opt.data_path, transforms.Normalize(mean=[0.5], std=[0.5]), split='test')
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False, collate_fn=floorplan_collate_fn)
# optimizers
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# run inference
def _infer(graph, model, prev_state=None):
    
    # configure input to the network
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
    # run inference model
    with torch.no_grad():
        masks = model(z.to('cuda'), given_masks_in.to('cuda'), given_nds.to('cuda'), given_eds.to('cuda'))
        masks = masks.detach().cpu().numpy()
    return masks


def image_to_data_uri(image_path):
    with open(image_path, 'rb') as image_file:
        data = image_file.read()
        data_uri = 'data:image/png;base64,' + base64.b64encode(data).decode('utf-8')
    return data_uri


def main_minimized(data_folder_path):
    globalIndex = 0
    session_folder_path = os.path.join(PUBLIC_DIR, "test_hybrid")
    opt.out =session_folder_path

    for i in range(0,4):
        graph = np.load(data_folder_path+"/{0}_graph.npy".format(i), allow_pickle=True)
        os.makedirs('./{}/'.format(opt.out), exist_ok=True)
        _round = 0

        real_nodes = graph[0]

        _types = sorted(list(set(real_nodes)))
        selected_types = [_types[:k+1] for k in range(10)]

        # initialize layout
        state = {'masks': None, 'fixed_nodes': []}
        masks = _infer(graph, model, state)
        im0 = draw_masks(masks.copy(), real_nodes)
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0 
        # save_image(im0, './{}/fp_init_{}.png'.format(opt.out, i), nrow=1, normalize=False) # visualize init image

        # generate per room type
        for _iter, _types in enumerate(selected_types):
            _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
                if len(_types) > 0 else np.array([]) 
            state = {'masks': masks, 'fixed_nodes': _fixed_nds}
            masks = _infer(graph, model, state)
            
        # save final floorplans
        imk = draw_masks(masks.copy(), real_nodes)
        imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
        imk = postprocessor.remove_white_background(imk)
        imk_after = postprocessor.remove_white_background_after(imk)
        print("OPT = ",opt.out," i = ",i," data path ",opt.data_path)
        save_image(imk, './{}/V{}.png'.format(opt.out, i+1), nrow=1, normalize=False)
        save_image(imk_after, './{}/V{}_after.png'.format(opt.out, i+1), nrow=1, normalize=False)
        # Display the loaded array
               
        # with open(data_folder_path+"/{0}_graph.dat".format(i), "r") as f:
        #     # exec("import numpy as np; data = " + f.read())
        #     data = f.read()
        #     array_1, array_2 = tuple(data)
        #     array_1 = np.array(array_1)
        #     array_2 = np.array(array_2)
        #     print(array_1)
        #     print(array_2)

def main(session_folder_path):
    globalIndex = 0
    opt.out =session_folder_path
    for i, sample in enumerate(fp_loader):

        # draw real graph and groundtruth
        mks, nds, eds, _, _ = sample
        real_nodes = np.where(nds.detach().cpu()==1)[-1]
        graph = [nds, eds]
        true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
        graph_im.save('./{}/graph_{}.png'.format(opt.out, i)) # save graph

        # add room types incrementally
        _types = sorted(list(set(real_nodes)))
        selected_types = [_types[:k+1] for k in range(10)]
        os.makedirs('./{}/'.format(opt.out), exist_ok=True)
        _round = 0
        
        # initialize layout
        state = {'masks': None, 'fixed_nodes': []}
        masks = _infer(graph, model, state)
        im0 = draw_masks(masks.copy(), real_nodes)
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0 
        # save_image(im0, './{}/fp_init_{}.png'.format(opt.out, i), nrow=1, normalize=False) # visualize init image

        # generate per room type
        for _iter, _types in enumerate(selected_types):
            _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
                if len(_types) > 0 else np.array([]) 
            state = {'masks': masks, 'fixed_nodes': _fixed_nds}
            masks = _infer(graph, model, state)
            
        # save final floorplans
        imk = draw_masks(masks.copy(), real_nodes)
        imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
        imk = postprocessor.remove_white_background(imk)
        imk_after = postprocessor.remove_white_background_after(imk)
        print("OPT = ",opt.out," i = ",i," data path ",opt.data_path)
        save_image(imk, './{}/V{}.png'.format(opt.out, i+1), nrow=1, normalize=False)
        save_image(imk_after, './{}/V{}_after.png'.format(opt.out, i+1), nrow=1, normalize=False)


app = Flask(__name__)


PUBLIC_DIR = 'public'

def datauri_to_image(datauri):
    # Extract the base64 part of the data URI
    image_data = re.sub("^data:image/.+;base64,", "", datauri)
    # Decode the base64 image data
    image_data = base64.b64decode(image_data)
    # Create an image from the decoded data
    image = Image.open(io.BytesIO(image_data))
    return image

def superimpose_image_to_datauri(image1_datauri, image2_path, x, y, scale_width, displayed_width, displayed_height):

    # Decode image1 from data URI
    x= int(x)
    y = int(y)
    image1 = datauri_to_image(image1_datauri)
    # Open image2 from file path
    image2 = Image.open(image2_path)
    #display scale:
    display_ratio = image1.width/displayed_width
    display_ratio_height = image1.height/displayed_height
    x = int(x*display_ratio)
    y= int(y*display_ratio)
    # Calculate the scaling factor to maintain aspect ratio
    aspect_ratio = image2.height / image2.width
    new_width = scale_width*display_ratio*0.9
    new_height = int(new_width * aspect_ratio)
    
    # Resize image2
    image2 = image2.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
    
    # Paste image2 on image1 at the specified coordinate
    print(x," y:",y)

    image1.paste(image2, (x, y), image2.convert("RGBA"))
    
    # Save image to a bytes buffer
    buffered = io.BytesIO()
    image1.save(buffered, format="PNG")
    buffered.seek(0)
    
    # Encode as base64 and convert to data URI
    img_str = base64.b64encode(buffered.read()).decode('utf-8')
    data_uri = f"data:image/png;base64,{img_str}"
    with open("current.txt","w") as f:
        f.write(data_uri)
    return data_uri

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(PUBLIC_DIR, filename)

@app.route('/generate_floorplans/<session_id>', methods=['GET'])
def create_session_folder(session_id):
    
    session_folder_path = os.path.join(PUBLIC_DIR, session_id)
    image_files = ['V1.png', 'V2.png', 'V3.png', 'V4.png']
    
    if not os.path.exists(session_folder_path):
        os.makedirs(session_folder_path)
    main(session_folder_path)
    
    images_data_uri = {}
    for image_file in image_files:
        image_path = os.path.join(session_folder_path, image_file)
        if os.path.exists(image_path):
            images_data_uri[image_file.replace(".png","")] = {"dataUri":image_to_data_uri(image_path), "text": "1 BR | 1 BA"}
            # images_data_uri[image_file.replace(".png","")] = image_to_data_uri(image_path)
        else:
            images_data_uri[image_file.replace(".png","")] = None

    
    # if os.path.exists(session_folder_path):
    #     shutil.rmtree(session_folder_path)

    return jsonify(images_data_uri)

@app.route('/use_floorplan', methods=['POST'])
def use_session_folder():

    data = request.get_json()
    session_id = data.get('session_id')
    selected_version = data.get('version')

    x = data.get("x")
    y = data.get("y")

    scale_width = data.get("scale_width")
    image1 = data.get("map_image")

    displayed_width = data.get("displayed_width")
    displayed_height = data.get("displayed_height")

    session_folder_path = os.path.join(PUBLIC_DIR, session_id)
    image2_path = os.path.join(session_folder_path, selected_version+"_after.png")
    imposed_darauri = superimpose_image_to_datauri(image1, image2_path, x, y, scale_width, displayed_width, displayed_height)

    return jsonify(imposed_darauri)

@app.route('/test', methods=['GET'])
def test_server():
    return("Test running")

if __name__ == '__main__':
    # main_minimized("D:/Floor generator/houseganpp_demo")  
    # main((os.getcwd(), "public","test2"))  
     
    if not os.path.exists(PUBLIC_DIR):
        os.makedirs(PUBLIC_DIR)
    
    app.run(debug=True,port=3200)
# if __name__ == '__main__':
#     main()
        
# main("/public/test2") 