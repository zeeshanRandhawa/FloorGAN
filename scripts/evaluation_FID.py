import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn
# from floorplan_dataset_maps_functional import FloorplanGraphDataset, floorplan_collate_fn

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image, ImageDraw, ImageFont
import svgwrite
from misc.utils import ID_COLOR, ROOM_CLASS
from models.models_exp_high_res import Generator
# from models_exp_3 import Generator

from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import glob
import cv2
import webcolors
from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil
from pathlib import Path
from misc.utils import fill_regions

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num_variations", type=int, default=1, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exps', help="destination folder")

opt = parser.parse_args()
print(opt)

target_set = 8
phase='eval'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/gen_housegan_E_1000000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/functional_graph_fixed_A_300000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_functional_graph_with_l1_loss_attempt_3_A_550000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_128_A_750000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_with_doors_64x64_per_room_type_A_230000.pth'
# checkpoint = './checkpoints/exp_random_node_type_A_350000.pth'
# checkpoint = './checkpoints/exp_sample_from_all_types_A_250000.pth'

checkpoint = './checkpoints/exp_random_types_attempt_3_A_500000_G.pth' #.format(target_set)
# checkpoint = './checkpoints/exp_finetune_fid_seq_2_A_200000.pth'
# checkpoint = './checkpoints/exp_finetune_from_scratch_8_500000.pth'

# checkpoint = './checkpoints/exp_high_res_with_doors_64x64_per_room_type_attempt_2_A_360000.pth'
# checkpoint = './checkpoints/exp_random_350000_A_200000.pth'

# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_per_room_type_enc_dec_plus_local_A_260000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_autoencoder_A_72900.pth'

print(checkpoint)

PREFIX = "./"
IM_SIZE = 64


def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.BICUBIC)
    return padded_im

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    for k, label in enumerate(g_true[0]):
        _type = label+1 
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            if _type == 15 or _type == 17:
                node_size.append(500)
            else:
                node_size.append(1000)

    for k, m, l in g_true[1]:
        if m > 0:
            G_true.add_edges_from([(k, l)], color='b',weight=4)    
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')

    edges = G_true.edges()
    colors = ['black' for u,v in edges]
    weights = [4 for u, v in edges]

    nx.draw(G_true, pos, node_size=node_size, node_color=colors_H, font_size=14, font_color='white', font_weight='bold', edge_color=colors, width=weights, with_labels=True)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    plt.close('all')
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    return rgb_arr, G_true


def estimate_graph(masks, nodes):
    G_estimated = nx.Graph()
    colors_H = []
    for k, label in enumerate(nodes):
        _type = label+1 
        if _type >= 0:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])

    for k in range(len(nodes)):
        for l in range(len(nodes)):
            if k > l:   
                m1, m2 = masks[k], masks[l]
                m1[m1>0] = 1.0
                m1[m1<=0] = 0.0
                m2[m2>0] = 1.0
                m2[m2<=0] = 0.0
                iou = np.logical_and(m1, m2).sum()/float(np.logical_or(m1, m2).sum())
                if iou > 0 and iou < 0.1:
                    G_estimated.add_edges_from([(k, l)], color='b',weight=4)    

    # plt.figure()
    # pos = nx.nx_agraph.graphviz_layout(G_estimated, prog='neato')
    # edges = G_estimated.edges()
    # colors = ['black' for u,v in edges]
    # weights = [4 for u,v in edges]
    # nx.draw(G_estimated, pos, node_size=1000, node_color=colors_H, font_size=14, font_weight='bold', edges=edges, edge_color=colors, width=weights, with_labels=True)
    # plt.tight_layout()
    # plt.savefig('./dump/_fake_graph.jpg', format="jpg")

    return G_estimated

def detailed_viz(masks, nodes, G_gt):
    G_estimated = nx.Graph()
    colors_H = []
    node_size = []
    for k, label in enumerate(nodes):
        _type = label+1 
        if _type >= 0:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            if _type == 15 or _type == 17:
                node_size.append(500)
            else:
                node_size.append(1000)
    
    # add node-to-door connections
    doors_inds = np.where(nodes > 10)[0]
    rooms_inds = np.where(nodes <= 10)[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:   
                m1, m2 = masks[k], masks[l]
                m1[m1>0] = 1.0
                m1[m1<=0] = 0.0
                m2[m2>0] = 1.0
                m2[m2<=0] = 0.0
                iou = np.logical_and(m1, m2).sum()/float(np.logical_or(m1, m2).sum())
                if iou > 0:
                    doors_rooms_map[k].append((l, iou))    

    # draw connections            
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)

        _conn_top2 = _conn[:2]
        for l, _ in _conn_top2:
            G_estimated.add_edges_from([(k, l)], color='green', weight=4)

        if len(_conn_top2) > 1:
            l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
            G_estimated.add_edges_from([(l1, l2)])
    
    # add missed edges 
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])

    # add edges colors
    colors = []
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')

    # add node-to-node connections
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
    weights = [4 for u, v in G_estimated_complete.edges()]
    nx.draw(G_estimated_complete, pos, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', \
            edge_color=colors, width=weights, with_labels=True)
    plt.tight_layout()
    plt.savefig('./dump/_fake_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_fake_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    plt.close('all')
    return rgb_arr, G_estimated_complete


# def draw_masks(masks, real_nodes):
#     bg_img = np.zeros((256, 256, 3)) + 255
#     for m, nd in zip(masks, real_nodes):
#         m[m>0] = 255
#         m[m<0] = 0
#         m = m.detach().cpu().numpy()
#         m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_AREA) 
#         color = ID_COLOR[nd+1]
#         r, g, b = webcolors.name_to_rgb(color)
#         inds = np.array(np.where(m>0))
#         bg_img[inds[0, :], inds[1, :], :] = np.array([[r, g, b]])
#     bg_img = Image.fromarray(bg_img.astype('uint8'))
#     return bg_img

def draw_masks(masks, real_nodes, im_size=256):
    bg_img = np.zeros((256, 256, 3)) + 255
    for m, nd in zip(masks, real_nodes):
        
        # resize map
        m_lg = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 

        # grab color
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)

        # draw region
        reg = np.zeros_like(bg_img) + 255
        m_lg = np.repeat(m_lg[:, :, np.newaxis], 3, axis=2)
        m_lg[m_lg>0] = 255
        m_lg[m_lg<0] = 0
        inds = np.where(m_lg > 0)
        reg[inds[0], inds[1], :] = [r, g, b]

        # draw contour
        m_cv = m_lg[:, :, 0].astype('uint8')
        ret,thresh = cv2.threshold(m_cv, 127, 255 , 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:  
            contours = [c for c in contours]
        cv2.drawContours(reg, contours, -1, (0, 0, 0), 1)

        # paste content to background
        inds = np.where(np.prod(reg/255.0, -1) < 1.0)
        bg_img[inds[0], inds[1], :] = reg[inds[0], inds[1], :]
    
    # convert to PIL
    bg_img = Image.fromarray(bg_img.astype('uint8'))
    return bg_img

def draw_floorplan(dwg, junctions, juncs_on, lines_on):
    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])
        x2, y2 = np.array(junctions[l])
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=1.0))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])
        dwg.add(dwg.circle(center=(float(x), float(y)), r=3, stroke='red', fill='white', stroke_width=2, opacity=1.0))
    return 


def filter_masks(masks):
    new_masks = []
    for m in masks:
        b_m = np.zeros_like(m)
        n_m = np.zeros_like(m)-1.0
        b_m[m >= 0] = 0
        b_m[m < 0] = 1
        filled_m = fill_regions(b_m)
        tags = set(list(filled_m.ravel()))
        tag_lg = -1
        area_lg = -1
        for t in list(tags):
            pxs = np.array(np.where(filled_m == t))
            if (t > 1) and (pxs.shape[-1] > area_lg):
                area_lg = pxs.shape[-1]
                tag_lg = t
        if tag_lg != -1:
            n_m[filled_m == tag_lg] = 1.0
        else:
            n_m = m
        new_masks.append(n_m)
    
    new_masks = np.stack(new_masks)
    return new_masks

# Create folder
os.makedirs(opt.exp_folder, exist_ok=True)

# Initialize generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load(checkpoint), strict=False)
generator = generator.eval()

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = './'

# Initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set, split=phase)
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False, collate_fn=floorplan_collate_fn)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Generate state
def gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample, initial_state, remove_multiple_components=False):

    # unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = sample

    # configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds

    # set up fixed nodes
    ind_fixed_nodes = torch.tensor(curr_fixed_nodes_state)
    ind_not_fixed_nodes = torch.tensor([k for k in range(real_mks.shape[0]) if k not in ind_fixed_nodes])

    # initialize given masks
    given_masks = torch.zeros_like(real_mks)
    given_masks = given_masks.unsqueeze(1)
    given_masks[ind_not_fixed_nodes.long()] = -1.0
    inds_masks = torch.zeros_like(given_masks)
    given_masks_in = torch.cat([given_masks, inds_masks], 1)    
    real_nodes = np.where(given_nds.detach().cpu()==1)[-1]

    # generate layout
    z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
    with torch.no_grad():

        # look for given feats
        if not initial_state:
            print('running state: {}, {}'.format(str(curr_fixed_nodes_state), str(prev_fixed_nodes_state)))
            prev_mks = np.load('{}/feats/feat_debug_{}.npy'.format(PREFIX, '_'.join(map(str, prev_fixed_nodes_state))), allow_pickle=True)
            prev_mks = torch.tensor(prev_mks).cuda().float()   
            given_masks_in[ind_fixed_nodes.long(), 0, :, :] = prev_mks[ind_fixed_nodes.long()]
            given_masks_in[ind_fixed_nodes.long(), 1, :, :] = 1.0
            curr_gen_mks = generator(z, None, given_masks_in, given_nds, given_eds)
        else:
            print('running initial state')
            curr_gen_mks = generator(z, None, given_masks_in, given_nds, given_eds)

        # reconstruct
        curr_gen_mks = curr_gen_mks.detach().cpu().numpy()
        if remove_multiple_components:
            curr_gen_mks = filter_masks(curr_gen_mks)
            
        # save current features
        np.save('{}/feats/feat_debug_{}.npy'.format(PREFIX, '_'.join(map(str, curr_fixed_nodes_state))), curr_gen_mks)
        
    return curr_gen_mks

#  Vectorize
globalIndex = 0
final_images = []
page_count = 0
n_rows = 0
feats_tensor = []
all_images = []
all_dists = []


# # save groundtruth
# os.makedirs('./FID/gt_small_{}'.format(target_set), exist_ok=True)
# for i, sample in enumerate(fp_loader):
#     if i == 1000:
#         break
#     # draw real graph and groundtruth
#     mks, nds, eds, _, _ = sample
#     real_nodes = np.where(nds.detach().cpu()==1)[-1]
#     real_im = draw_masks(mks.detach().cpu().numpy(), real_nodes)
#     real_im.save('./FID/gt_small_{}/{}.png'.format(target_set, i))

FIDS = []
for run in range(5):
    
# #     # CUSTOM sequence
#     all_states = [[6], [5, 6, 14], [2, 5, 6, 9, 14, 16], [2, 4, 5, 6, 9, 14, 16], [2, 4, 5, 6, 9, 14, 16], [2, 4, 5, 6, 9, 14, 16],\
#                   [2, 4, 5, 6, 9, 14, 16], [2, 4, 5, 6, 7, 9, 14, 16], [2, 4, 5, 6, 7, 9, 14, 16], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]] # best for exp_random_types_attempt_2_A_500000 - FID
#     all_states = [[0, 14], [0, 2, 14, 16], [0, 2, 3, 14, 16], [0, 2, 3, 14, 16], [0, 2, 3, 14, 16], [0, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16],\
#                  [0, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16], [0, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16],\
#                  [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]] # best for exp_high_res_with_doors_64x64_per_room_type_attempt_2_A_360000 - FID
#     all_states = [[16], [16], [7, 16], [2, 7, 14, 15, 16], [2, 4, 5, 7, 14, 15, 16], [0, 2, 4, 5, 6, 7, 14, 15, 16], [0, 2, 4, 5, 6, 7, 9, 14, 15, 16],\
#                  [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16],\
#                  [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]] # # best for exp_random_types_attempt_2_A_500000 - Valid houses

#     all_states = [[9, 15], [9, 15], [9, 15], [3, 9, 15], [1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 5, 9, 15], [0, 1, 2, 3, 5, 6, 9, 15], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]] # best for exp_random_types_attempt_3_A_500000_G - FID
    print('** CUSTOM **')
#     all_states = [[14, 15], [5, 14, 15], [3, 5, 14, 15], [3, 5, 9, 14, 15], [1, 3, 5, 9, 14, 15], [1, 2, 3, 4, 5, 9, 14, 15], [0, 1, 2, 3, 4, 5, 9, 14, 15], [0, 1, 2, 3, 4, 5, 9, 14, 15], [0, 1, 2, 3, 4, 5, 9, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]]
    all_states = [[9, 15], [9, 15], [9, 15], [3, 9, 15], [1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 5, 9, 15], [0, 1, 2, 3, 5, 6, 9, 15], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]]
    
#     # ALL FIXED
#     all_types = [ROOM_CLASS[k]-1 for k in ROOM_CLASS]
#     all_states = [[t for t in all_types] for k in range(10)] 
    
#     all_states = [[16], [16], [16], [16], [0, 2, 4, 6, 7, 14, 16], [0, 2, 3, 4, 6, 7, 14, 16], [0, 2, 3, 4, 6, 7, 9, 14, 16],\
#                   [0, 1, 2, 3, 4, 6, 7, 9, 14, 15, 16], [0, 1, 2, 3, 4, 6, 7, 9, 14, 15, 16], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]] #exp_random_types_attempt_2_A_500000 - GED

#     # pick a sequence
#     print('** RANDOM **')
#     all_types = [ROOM_CLASS[k]-1 for k in ROOM_CLASS]
#     all_states = [[t for t in all_types if random.uniform(0, 1) > 0.5] for _ in range(10)] # RANDOM SCHEME
    
    # create dirs
    dirpath = Path('./FID/test/debug')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.makedirs('./FID/test/debug', exist_ok=True)
    
    # compute FID for a given sequence
    for i, sample in enumerate(fp_loader):

        if i == 1000:
            break
        mks, nds, eds, _, _ = sample
        real_nodes = np.where(nds.detach().cpu()==1)[-1]
#         print(real_nodes)
#         room_types = list(set(real_nodes))
#         all_states = [[t for t in room_types if t <= room_types[min(k, len(room_types)-1)]] for k in range(len(room_types)+1)] # CLASSES ORDER SCHEME

        #### FIX PER ROOM TYPE #####
        # generate final layout initialization
        for j in range(1):
            prev_fixed_nodes_state = []
            curr_fixed_nodes_state = []
            curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample, initial_state=True)

            # generate per room type
            for _types in all_states:
                if len(_types) > 0:
                    curr_fixed_nodes_state = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types])
                else:
                    curr_fixed_nodes_state = np.array([])
                    
                curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample, initial_state=False)
                prev_fixed_nodes_state = list(curr_fixed_nodes_state)

            # save final floorplans
            imk = draw_masks(curr_gen_mks, real_nodes)
            imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0
            save_image(imk, './FID/test/debug/{}_{}.png'.format(i, j), nrow=1, normalize=False)

    # write current results
    fid_value = calculate_fid_given_paths(['./FID/gt_small_{}/'.format(target_set), './FID/test/debug/'], 2, 'cpu', 2048)
    FIDS.append(fid_value)
    with open('./FID/results_{}.txt'.format(run), 'w') as f:
        f.write("\n".join([str(f) for f in FIDS]))
print('FID: ', FIDS)


