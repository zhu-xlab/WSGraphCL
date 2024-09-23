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
from torch_geometric.utils import  subgraph,dropout_adj
from torch_geometric.data import Batch
#from dig.sslgraph.method import InfoGraph, MVGRL, GRACE, GraphCL
#from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask
#from dig.sslgraph.method import Contrastive
from torch_geometric.utils import subgraph

# +
class UniformSample():
    r"""Uniformly node dropping on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optinal): Ratio of nodes to be dropped. (default: :obj:`0.1`)
    """
    def __init__(self, ratio=0.1):
        self.ratio = ratio
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):

        device = data.x.device
        subgraph_num_node=data.x.size()[0]
        keep_num = int(subgraph_num_node * (1-self.ratio))
        idx_nondrop = torch.randperm(subgraph_num_node)[:keep_num].cpu()
        idx_list=data.edge_index[1,idx_nondrop.cpu()]
        edge_index, edge_attr = subgraph(idx_nondrop, edge_index=data.edge_index.cpu(), edge_attr=data.edge_attr.cpu(), relabel_nodes=True, num_nodes=subgraph_num_node)
        edge_index=edge_index.long()
        data=Data(x=data.x[idx_nondrop.cpu()], edge_index=edge_index.to(device),edge_attr=edge_attr.to(device))

        
        return data
    
    def views_fn(self, data):
        r"""Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
        
class EdgePerturbation():
    '''Edge perturbation on the given graph or batched graphs. Class objects callable via 
    method :meth:`views_fn`.
    
    Args:
        add (bool, optional): Set :obj:`True` if randomly add edges in a given graph.
            (default: :obj:`True`)
        drop (bool, optional): Set :obj:`True` if randomly drop edges in a given graph.
            (default: :obj:`False`)
        ratio (float, optional): Percentage of edges to add or drop. (default: :obj:`0.1`)
    '''
    def __init__(self, add=False, drop=True, ratio=0.1):
        self.add = add
        self.drop = drop
        self.ratio = ratio
        
    def __call__(self, data):
        return self.views_fn(data)
        
    def do_trans(self, data):
        device = data.x.device
        sub_node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * self.ratio)
        idx_add = torch.tensor([], device=device).reshape(2, -1).long()

        if self.drop:
            new_edge_index,new_edge_attr = dropout_adj(data.edge_index,data.edge_attr, p=self.ratio)
            new_edge_index=new_edge_index.long()
            
        if self.add:
            idx_add = torch.randint(sub_node_num, (2, perturb_num), device=device)
            new_edge_index = torch.cat((new_edge_index, idx_add), dim=1)
            new_edge_index = torch.unique(new_edge_index, dim=1)
        data=Data(x=data.x, edge_index=new_edge_index,edge_attr=new_edge_attr)

        return data

    def views_fn(self, data):
        r"""Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
        
class NodeAttrMask():
    '''Node attribute masking on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    '''
    def __init__(self, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5, return_mask=False):
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.return_mask = return_mask
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        _, feat_dim = data.x.size()
        x = data.x.detach().clone()
        sub_node_num, _ = x.size()
        if self.mode == 'whole':
            mask = torch.zeros(sub_node_num)
            mask_num = int(sub_node_num * self.mask_ratio)
            idx_mask = torch.randperm(x.size(0), device=x.device)[:mask_num]
            if self.mask_std > 0:
                x[idx_mask] = torch.empty((mask_num, feat_dim), dtype=torch.float32, 
                    device=x.device).normal_(mean=self.mask_mean,std=self.mask_std)
            else:
                    x[idx_mask] = self.mask_mean

            mask[idx_mask] = 1

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(self.mode))

        if self.return_mask:
            data= Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, mask=mask)
        else:
            data= Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        return data
    def views_fn(self, data):
        r"""Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)


# -

class RWSample():
    """Subgraph sampling based on random walk on the given graph or batched graphs.
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optional): Percentage of nodes to sample from the graph.
            (default: :obj:`0.1`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`False`)
    """
    def __init__(self, ratio=0.1, add_self_loop=False):
        self.ratio = ratio
        self.add_self_loop = add_self_loop
    
    def __call__(self, data):
        return self.views_fn(data)
    
    def do_trans(self, data):
        device = data.x.device

        sun_node_num, _ = data.x.size()
        sub_num = int(sun_node_num * self.ratio)

        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(sun_node_num)], device=device).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index

        idx_sub = [torch.randint(sun_node_num, size=(1,), device=device)[0]]
        idx_neigh = set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > sun_node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = list(idx_neigh)[torch.randperm(len(idx_neigh), device=device)[0]]
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_sub = torch.LongTensor(idx_sub, device=device)
        edge_index, edge_attr = subgraph(idx_sub, data.edge_index.cpu(), data.edge_attr.cpu(), relabel_nodes=True, num_nodes=sun_node_num)
        data=Data(x=data.x[idx_sub], edge_index=edge_index, edge_attr=edge_attr)
        return data

    def views_fn(self, data):
        r"""Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)


class AddGussianNoise():
    def __init__(self, std, mean=0,  aug_ratio=0.1):
        self.mean = mean
        self.std = std
        self.aug_ratio = aug_ratio

    def __call__(self, data):
        return self.views_fn(data)

    def do_trans(self, data):
        num_bands = int(data.x.size()[1]*self.aug_ratio)
        randomband_list = np.random.choice(data.x.size()[1], size=num_bands, replace=False)
        data.x[:,randomband_list] = data.x[:,randomband_list] + torch.randn(data.x[:,randomband_list].size()) * self.std + self.mean
        data.x = torch.clamp(data.x, 0, 1)
        data=Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        return data
    
    def views_fn(self, data):
        r"""Method to be called when :class:`AddGussianNoise` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)


class random():
    def __init__(self, aug_ratio, add_self_loop=False):
        self.aug_ratio = aug_ratio
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        
        return self.views_fn(data)
        
    def do_trans(self, data):
        self.data=data
        device = data.x.device
        canditates =   [UniformSample(ratio=self.aug_ratio),
                        RWSample(ratio=self.aug_ratio),
                        EdgePerturbation(ratio=self.aug_ratio),
                        UniformSample(ratio=self.aug_ratio)]#AddGussianNoise(std=0.05,aug_ratio=self.aug_ratio)
        id_aug = int(np.random.randint(0,3+1))
        data = canditates[id_aug](data)
        return data
    def views_fn(self, data):
        r"""Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
            
        elif isinstance(data, Data):
            return self.do_trans(data)


class strong():
    def __init__(self, aug_ratio, add_self_loop=False):
        self.aug_ratio = aug_ratio
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        
        return self.views_fn(data)
        
    def do_trans(self, data):
        self.data=data
        device = data.x.device
        canditates =   [UniformSample(ratio=self.aug_ratio),
                        RWSample(ratio=self.aug_ratio),
                        EdgePerturbation(ratio=self.aug_ratio),
                        AddGussianNoise(std=0.05,aug_ratio=self.aug_ratio)]
        id_aug = int(np.random.randint(0,4))
        data = canditates[id_aug](data)
        return data
    def views_fn(self, data):
        r"""Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
            
        elif isinstance(data, Data):
            return self.do_trans(data)


class weak():
    def __init__(self, aug_ratio, add_self_loop=False):
        self.aug_ratio = aug_ratio
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        
        return self.views_fn(data)
        
    def do_trans(self, data):
        self.data=data
        device = data.x.device
        canditates =   [EdgePerturbation(ratio=self.aug_ratio/8),
                        AddGussianNoise(std=0.05,aug_ratio=self.aug_ratio/8)]
        id_aug = int(np.random.randint(0,2))
        data = canditates[id_aug](data)
        return data
    def views_fn(self, data):
        r"""Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
            
        elif isinstance(data, Data):
            return self.do_trans(data)


class weak2():
    def __init__(self, aug_ratio, add_self_loop=False):
        self.aug_ratio = aug_ratio
        self.add_self_loop = add_self_loop
        
    def __call__(self, data):
        
        return self.views_fn(data)
        
    def do_trans(self, data):
        self.data=data
        device = data.x.device
        canditates =   [EdgePerturbation(ratio=self.aug_ratio/8),
                        UniformSample(ratio=self.aug_ratio/8)]
        id_aug = int(np.random.randint(0,2))
        data = canditates[id_aug](data)
        return data
    def views_fn(self, data):
        r"""Method to be called when :class:`RWSample` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
            
        elif isinstance(data, Data):
            return self.do_trans(data)
