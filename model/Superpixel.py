import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, download_url
from typing import Optional, Callable, List, Union, Tuple, Dict, Iterable
from torch_geometric.nn import GCNConv
from scipy.spatial import distance
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from matplotlib import pyplot as plt
from matplotlib import image as img
from skimage.segmentation import mark_boundaries
from torch_geometric.transforms import ToSLIC
from torch_geometric.utils import to_networkx
import os
import networkx as nx
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter
from sklearn.decomposition import PCA
from Graph_Construction import adjacency_matrix, get_dataset, MyOwnDataset



def visualize_SLIC_superpixels(image, slic_graph, slic_images_and_graphs_dir):
    image_segments = torch.squeeze(slic_graph.seg).detach().numpy()
    rgb_image=np.array([image[:,:,29],image[:,:,20],image[:,:,11]])
    rgb_image =rgb_image.transpose(1,2,0)
    image_with_slic_boundaries = mark_boundaries(rgb_image, image_segments)#,mode='subpixel')
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.imshow(image_with_slic_boundaries)
    axs.axis("off")
    plt.savefig(os.path.join(slic_images_and_graphs_dir, "SLIC_graph.png"))
    plt.imshow(image_with_slic_boundaries)
    plt.show()
    plt.close()

    #fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    #graph_viz = to_networkx(slic_graph)
    #nx.draw(graph_viz, ax=axs)
    #plt.savefig(os.path.join(slic_images_and_graphs_dir, "/SLIC_graph.png"))
    #plt.close()

def creat_SLIC_graphs(image, labels, n_segments,project_root) :
    print("Creating SLIC graphs")
    img_slic_input = torch.from_numpy(image).permute(2, 0, 1)
    print(img_slic_input.size())
    #img_slic_input=img_slic_input[:3,:,:]
    Dim=np.array([np.size(image,0),np.size(image,1),np.size(image,2)])
    bands_xy_num=Dim[2]
    bands_num=bands_xy_num-2
    pca = PCA(n_components=30)
    data=image.reshape((Dim[0]*Dim[1],bands_xy_num))[:,0:bands_num]
    pca.fit(data)
    data_pca=pca.transform(data)
    pca_3d=data_pca.reshape((Dim[0],Dim[1],30))
    pca_data=pca_3d[:,:,0:3]
    rgb_image=pca_data#np.array([image[:,:,29],image[:,:,20],image[:,:,11]])
    #rgb_image =rgb_image.transpose(1,2,0)
    img_slic_input = torch.from_numpy(rgb_image).permute(2, 0, 1)
    slic_transform = ToSLIC(n_segments=n_segments, add_seg=True, add_img=True)
    slic_graph_position_similarity = slic_transform(img_slic_input)
    print(slic_graph_position_similarity)
    slic_graph_position_similarity.y = labels
    visualize_SLIC_superpixels(image,  slic_graph_position_similarity, project_root)
    return slic_graph_position_similarity


# Python3 program to find the most
# frequent element in an array.
def mostFrequent(arr, n):
  maxcount = 0
  element_having_max_freq = 0
  for i in range(0, n):
    count = 0
    for j in range(0, n):
      if(arr[i] == arr[j]):
        count += 1
    if(count > maxcount):
      maxcount = count
      element_having_max_freq = arr[i]
    
  return element_having_max_freq

def find_center(sp):
    sp=np.array(sp)
    y=int(np.median(sp[1,:]))
    x_list=np.where(sp[1,:]==y)
    x=np.median(sp[0,x_list])
    return int(x),int(y)


def build_superpixel_graph(slic_graph, image, labels, train_m, n_segments,node_extract_mode):
    """Build a graph from a superpixel graph and an image.
    
    Parameters
    ----------
    slic_graph : networkx.Graph
        The superpixel graph.
    image : numpy.ndarray
        The image.
    labels : numpy.ndarray
        The superpixel labels.
    n_segments : int
        The number of superpixels.
    compactness : float
        The compactness parameter for SLIC.
        
    Returns
    -------
    networkx.Graph
        The superpixel graph.
    """
    if node_extract_mode == 'all':
      sp_graph=[]
      for i in range(n_segments):
        # Get the superpixel
        sp = np.where(slic_graph.seg[0,:,:]==i)
        # Get the superpixel features
        x=torch.from_numpy(np.array(image[sp]))
        
        y=torch.from_numpy(np.array(labels[sp])).unsqueeze(1)
        train_mask = torch.from_numpy(np.array(train_m[sp]))
        def options(mode,k,sigma,miu):
            options = {
                "NeighborMode": 'KNN',
                "k": k,
                "WeightMode": mode,
                "t":sigma**2,
                "xy_scaler":miu,
                }
            return options
        options=options("HeatKernel",10,1,1)
        L=adjacency_matrix(x,options)
        data_=get_dataset(x,y,L,train_mask,None,options)
        sp_graph.append(data_)
    else:
    # Build a graph from the superpixel graph
        x_zeros=torch.zeros((slic_graph.x.size(0),image.shape[2]))
        sp_graph = Data(x=x_zeros,pos=slic_graph.pos,y=slic_graph.x[:,0])
    #sp_graph.add_nodes_from(slic_graph.x(data=True))

    
    # Add the superpixel features
        for i in range(n_segments):
        # Get the superpixel
            sp = np.where(slic_graph.seg[0,:,:]==i)
        # Get the superpixel features
            if node_extract_mode == 'mean':
               x=torch.from_numpy(np.array(np.mean(image[sp], axis=0)))
               labels_list=labels[sp]
               labels_list=labels_list[np.where(labels_list>0)]
               n=len(labels_list)
               y=torch.tensor(mostFrequent(labels_list,n),dtype=torch.int)
            elif node_extract_mode == 'middle':
               r,c=find_center(sp)
           #data=image[sp]
               x=torch.from_numpy(np.array(image[r,c,:]))
               labels_list=labels[sp]
               labels_list=labels_list[np.where(labels_list>0)]
               n=len(labels_list)
               y=torch.tensor(mostFrequent(labels_list,n),dtype=torch.int)
        # Add the superpixel features
            sp_graph.x[i] = x
            sp_graph.y[i] = y
        
        
    return sp_graph

def build_train_mask(sp_graph,num_labels):
    train_mask = torch.full([sp_graph.x.size(0)],False)
    num_classes=int(torch.max(sp_graph.y))
    for i in range(1,num_classes):
        list=np.array(np.where(sp_graph.y==i))
        if list.size>num_labels:
            train_mask[np.random.choice(list[0], size=num_labels,replace=False)] = True
        elif list.size>int(num_labels/2):
            train_mask[np.random.choice(list[0], size=int(num_labels/2),replace=False)] = True
        else:
            train_mask[np.random.choice(list[0], size=1,replace=False)] = True
    return train_mask

def return_from_sp(slic_graph,result,Dim):

    index=np.array(slic_graph.seg[0,:,:]).reshape((Dim[0]*Dim[1]))
    o_result=np.zeros((Dim[0]*Dim[1]))
    for i in range (0, result.shape[0]):
        o_result[np.where(index==i)]=result[i]
    return o_result

def return_from_sp_bool(slic_graph,train_m,Dim):

    index=np.array(slic_graph.seg[0,:,:]).reshape((Dim[0]*Dim[1]))
    o_result=np.full((Dim[0]*Dim[1]),False)
    for i in range (0, train_m.shape[0]):
        o_result[np.where(index==i)]=train_m[i]
    return o_result


def process_superpixel(IP_dataset,IP_gt,train_m,num_seg,num_labels,Dim,project_root,node_extract_mode):
  slic_graph = creat_SLIC_graphs(IP_dataset[:,:,:], IP_gt, num_seg,project_root)
  train_map=np.reshape(train_m,(Dim[0],Dim[1]))
  sp_graph=build_superpixel_graph(slic_graph, IP_dataset, IP_gt,train_map,  slic_graph.x.size(0),node_extract_mode)
    #train_m=build_train_mask(sp_graph,num_lables)
  if node_extract_mode == 'all':
         data=np.reshape(IP_dataset,(Dim[0]*Dim[1],Dim[2]+2))
         gt_flat=np.reshape(IP_gt,(Dim[0]*Dim[1],1))
         sp_trian=sp_graph
  else:
    data=np.array(sp_graph.x)
    gt_flat=np.array(sp_graph.y)
    gt_flat=np.array([gt_flat]).transpose()
    index=np.array(slic_graph.seg[0,:,:]).reshape((Dim[0]*Dim[1]))
    n_segments=np.max(index)
    
    if train_m is None:
        sp_trian=build_train_mask(sp_graph,num_labels)
    else:
        sp_trian=np.full((n_segments+1),False)
        for i in range(0,n_segments):
        # Get the superpixel
            sp = np.where(index==i)
            true_list=np.array(np.where(train_m[sp]==True))
            if true_list.size>0:
                sp_trian[i]=True
  return data,gt_flat,sp_trian,slic_graph
