import torch
import math
import random
import scipy.io
import numpy as np
import torch
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
import copy
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from tqdm import trange
from sklearn.model_selection import StratifiedKFold
#from dig.sslgraph.method import InfoGraph, MVGRL, GRACE, GraphCL
#from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask
#from dig.sslgraph.method import Contrastive
from torch_geometric.utils import subgraph, k_hop_subgraph

def cosine_similarity(x,y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def KNN(data,options):
  bands_xy_num=data.shape[1]
  bands_num=bands_xy_num-2
  newDim=data.shape[0]
  col=[]
  row=[]
  val=[]

  if options["WeightMode"]=="Eudist":
    spectral_distance=pairwise_distances(data[:,0:bands_num], metric='euclidean')/np.sqrt(bands_num)
    spatial_distance=pairwise_distances(data[:,bands_num:bands_xy_num], metric='euclidean')/np.sqrt(2)
    for i in range(0,newDim):
        distances=np.zeros(newDim)
        distances=((1-options["xy_scaler"])*spectral_distance[i]+options["xy_scaler"]*spatial_distance[i])
        nearest_neighbor_ids = distances.argsort()[:options["k"]]
        #max=np.max (distances)
        #print(nearest_neighbor_ids,distances)
        for id in nearest_neighbor_ids:
          row=np.append(row,[i])
          col=np.append(col,[id])
        
          val=np.append(val,[1-distances[id]])
  elif options["WeightMode"]=="HeatKernel":
    spectral_distance=pairwise_distances(data[:,0:bands_num], metric='euclidean')/np.sqrt(bands_num)
    spatial_distance=pairwise_distances(data[:,bands_num:bands_xy_num], metric='euclidean')/np.sqrt(2)
    for i in range(0,newDim):
        distances=np.zeros(newDim)
        distances=((1-options["xy_scaler"])*spectral_distance[i]+options["xy_scaler"]*spatial_distance[i])
        nearest_neighbor_ids = distances.argsort()[:options["k"]]
        #max=np.max (distances)
        #print(nearest_neighbor_ids,distances)
        for id in nearest_neighbor_ids:
          row=np.append(row,[i])
          col=np.append(col,[id])
        
          val=np.append(val,[np.exp(-distances[id]/options["t"])])
  elif options["WeightMode"]=="Binary":
    spectral_distance=pairwise_distances(data[:,0:bands_num], metric='euclidean')/np.sqrt(bands_num)
    spatial_distance=pairwise_distances(data[:,bands_num:bands_xy_num], metric='euclidean')/np.sqrt(2)
    for i in range(0,newDim):
        distances=np.zeros(newDim)
        distances=((1-options["xy_scaler"])*spectral_distance[i]+options["xy_scaler"]*spatial_distance[i])
        nearest_neighbor_ids = distances.argsort()[:options["k"]]
        #max=np.max (distances)
        #print(nearest_neighbor_ids,distances)
        for id in nearest_neighbor_ids:
          row=np.append(row,[i])
          col=np.append(col,[id])
        
          val=np.append(val,[1])
  elif options["WeightMode"]=="Cosine":
    spectral_distance=pairwise_distances(data[:,0:bands_num], metric='cosine')#/np.sqrt(bands_num)#histogram 
    spatial_distance=pairwise_distances(data[:,bands_num:bands_xy_num], metric='euclidean')/np.sqrt(2)#cosine_similarity
    for i in range(0,newDim):
        distances=np.zeros(newDim)
        distances=((1-options["xy_scaler"])*spectral_distance[i]+options["xy_scaler"]*spatial_distance[i])
        nearest_neighbor_ids = distances.argsort()[:options["k"]]
        #max=np.max (distances)
        #print(nearest_neighbor_ids,distances)
        for id in nearest_neighbor_ids:
          row=np.append(row,[i])
          col=np.append(col,[id])
          val=np.append(val,[1-distances[id]])
          
  row=np.array(row,dtype=int)
  col=np.array(col,dtype=int)
  val=np.array(val,dtype=float)

  W=sparse.coo_matrix((val, (row,col)),shape=(newDim,newDim))
  W=sparse.coo_matrix(W - sparse.eye(W.shape[0]))
  return W


def adjacency_matrix(data,options):
    W=KNN(data,options)
    D =sparse.coo_matrix.sum(W, 1)
    D=sparse.coo_matrix(D)
    D =sparse.coo_matrix.power(D,-0.5)
    D=sparse.coo_matrix((D.data, (D.row,D.row)))#convert D n*1 colomn matrix to diagonal matrix (n,n)
    L_temp = W*D

    L = D * L_temp

    L = sparse.coo_matrix(L + sparse.eye(L.shape[0]))
   
    return L#convert all computation to sparse

def get_dataset(data,gt_flat,L,train_m,ratio_labels,options):
    bands_xy_num=data.shape[1]
    bands_num=bands_xy_num-2
    edge_index = torch.tensor(np.array([L.row, L.col]),dtype=torch.int64)

    edge_attr = torch.tensor(L.data,dtype=torch.float32)


    x=torch.tensor(data[:,0:bands_num], dtype=torch.float32)#
       
    y=torch.tensor(gt_flat[:,0], dtype=torch.int64)

    train_mask=np.full((y.size(0)),False)
    train_mask=torch.tensor(train_mask, dtype=torch.bool)
    test_mask=np.full((y.size(0)),False)
    test_mask=torch.tensor(test_mask, dtype=torch.bool)

    
    num_classes=int(y.max())
    if train_m is None:
      for i in range(1,int(y.max())+1):
        if np.array(np.where(y==i)).size>0:
          list=np.array(np.where(y==i))
          k=int(ratio_labels*(np.array(np.where(y==i)).size))
          train_mask[np.random.choice(list[0], size=k,replace=False)] = True

    else:
        train_mask[np.where(train_m == True)] = True#define mask    
    test_mask[np.where(train_mask == False)] = True
    test_mask[np.where(y==0)]=False
    dataset = Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y, train_mask=train_mask,test_mask=test_mask,num_classes=num_classes)
    return dataset

def build_subgraph(G, id, second_gen=True):
    num_classes=G.num_classes
    node_num, _ = G.x.size()
    edge_index = G.edge_index
    #edge_index = edge_index[:,np.array(np.where(edge_index[0]!=edge_index[1]))][:,0,:]
    edge_attr = G.edge_attr
    #edge_attr = edge_attr[np.array(np.where(edge_index[0]!=edge_index[1]))]
    node_num=edge_attr.size()
    if second_gen==True:
        if np.array(np.where(G.edge_index[0]==id)).size>0:
            i_list=np.array(np.where(G.edge_index[0]==id))
            row=G.edge_index[0,i_list]
            col=G.edge_index[1,i_list]
            row=row[torch.where(col!=id)]
            col=col[torch.where(col!=id)]
            for m in col:
              if m!=id:#col2!=i
               if np.array(np.where(G.edge_index[0]==m)).size>0:
                i_list2=np.array(np.where(G.edge_index[0]==m))
                row2=G.edge_index[0,i_list2][0]
                col2=G.edge_index[1,i_list2][0]
                col2=col2[torch.where(col2!=id)]
                row2=row2[torch.where(col2!=id)]
                rowall=np.append(row,row2)
                colall=np.append(col,col2)  
    elif second_gen==False:
        if np.array(np.where(G.edge_index[0]==id)).size>0:
            i_list=np.array(np.where(G.edge_index[0]==id))
            row=G.edge_index[0,i_list]
            col=G.edge_index[1,i_list]
            row=row[torch.where(col!=id)]
            col=col[torch.where(col!=id)]
            rowall=row
            colall=col
    idx_list=np.concatenate((rowall,colall),axis=0)
    idx_list=np.unique(idx_list)
    idx_sub=torch.from_numpy(idx_list)
        
    #idx_sub = torch.LongTensor(idx_sub, device=device)
    mask_nondrop = torch.zeros_like(G.x[:,0]).scatter_(0, idx_sub, 1.0).bool()
    edge_index, edge_attr = subgraph(mask_nondrop, edge_index, edge_attr, relabel_nodes=True, num_nodes=node_num)

    return Data(x=G.x[mask_nondrop], edge_index=edge_index,edge_attr=edge_attr,y=G.y[mask_nondrop],train_mask=G.train_mask[mask_nondrop],test_mask=G.test_mask[mask_nondrop],num_classes=num_classes+1)

def build_Khopsubgraph(G, id, second_gen):
    num_classes=G.num_classes
    node_num, _ = G.x.size()
    edge_index = G.edge_index
    #edge_index = edge_index[:,np.array(np.where(edge_index[0]!=edge_index[1]))][:,0,:]
    edge_attr = G.edge_attr
    #edge_attr = edge_attr[np.array(np.where(edge_index[0]!=edge_index[1]))]
    #node_num=edge_attr.size()

    subset, edge_index, mapping, edge_mask  = k_hop_subgraph(id, second_gen, edge_index,  relabel_nodes=True, num_nodes=node_num,flow='source_to_target')
    
    return Data(x=G.x[subset], edge_index=edge_index,edge_attr=edge_attr[edge_mask], y=G.y[id], train_mask=G.train_mask[id],test_mask=G.test_mask[id],num_classes=num_classes+1)


class MyOwnDataset(InMemoryDataset):
    def __init__(self,root,dataset,Second_gen,transform=None, pre_transform=None, pre_filter=None):
    #def __init__(self, dataset):
        
        self.dataset=dataset
        self.Second_gen=Second_gen
        super().__init__(root,transform, pre_transform, pre_filter)
        #super().__init__(dataset)
        #self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        # Read data into huge `Data` list.
      d_list=[]
      for i in range(0,self.dataset.x.shape[0]):
            sub_gidx = build_Khopsubgraph(self.dataset,i,self.Second_gen)
            self.dataset[i] = sub_gidx#=Data(x=G.x[subset], edge_index=edge_index,edge_attr=edge_attr[edge_mask], y=G.y[subset], train_mask=G.train_mask[subset],test_mask=G.test_mask[subset],num_classes=num_classes)
            d_list.append(sub_gidx)
    
            
      data_list=d_list#self.dataset    
        #self.data, self.slices =  self.collate(data_list)
      self.num_nodes=self.dataset.x.shape[0]
        #self[0]=self.data
        
      self.data, self.slices =  self.collate(data_list)
