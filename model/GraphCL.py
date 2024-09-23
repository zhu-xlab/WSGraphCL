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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from tqdm import trange
from sklearn.model_selection import StratifiedKFold


from contrasive import Contrastive
from Graph_Construction import adjacency_matrix, get_dataset, MyOwnDataset
from Augmentations import UniformSample, EdgePerturbation, NodeAttrMask, RWSample, random, AddGussianNoise, strong, weak

from torch.nn import Parameter
from torch.nn import Sequential, Linear, BatchNorm1d
from functools import partial
from torch_scatter import scatter_add
from torch_geometric.nn import  GCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from dig.sslgraph.method.contrastive.views_fn import  RandomView

class GraphUnsupervised(object):
    r"""
    The evaluation interface for unsupervised graph representation learning evaluated with 
    linear classification. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph>`_ 
    for examples of usage.
    
    Args:
        dataset (torch_geometric.data.Dataset): The graph classification dataset.
        classifier (string, optional): Linear classifier for evaluation, :obj:`"SVC"` or 
            :obj:`"LogReg"`. (default: :obj:`"SVC"`)
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        epoch_select (string, optional): :obj:`"test_max"` or :obj:`"val_max"`.
            (default: :obj:`"test_max"`)
        n_folds (int, optional): Number of folds for evaluation. (default: :obj:`10`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    #>>> encoder = Encoder(...)
    #>>> model = Contrastive(...)
    #>>> evaluator = GraphUnsupervised(dataset, log_interval=10, device=0, p_lr = 0.001)
    #>>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, dataset, classifier, log_interval=1, epoch_select='test_max', 
                 metric='acc', n_folds=10, device=None, **kwargs):
        
        self.dataset = dataset
        self.epoch_select = epoch_select
        self.metric = metric
        self.classifier = classifier
        self.log_interval = log_interval
        self.n_folds = n_folds
        self.out_dim = dataset.num_classes
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device
        #print(self.device)
        # Use default config if not further specified
        self.setup_train_config(**kwargs)

    def setup_train_config(self, batch_size = 256, p_optim = 'Adam', p_lr = 0.01, 
                           p_weight_decay = 0, p_epoch = 20, svc_search = True):
        r"""Method to setup training config.
        
        Args:
            batch_size (int, optional): Batch size for pretraining and inference. 
                (default: :obj:`256`)
            p_optim (string, or torch.optim.Optimizer class): Optimizer for pretraining.
                (default: :obj:`"Adam"`)
            p_lr (float, optional): Pretraining learning rate. (default: :obj:`0.01`)
            p_weight_decay (float, optional): Pretraining weight decay rate. 
                (default: :obj:`0`)
            p_epoch (int, optional): Pretraining epochs number. (default: :obj:`20`)
            svc_search (string, optional): If :obj:`True`, search for hyper-parameter 
                :obj:`C` in SVC. (default: :obj:`True`)
        """
        
        self.batch_size = batch_size

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.search = svc_search
    
    def evaluate(self, learning_model, encoder, fold_seed=None):
        r"""Run evaluation with given learning model and encoder(s).
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive) 
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            fold_seed (int, optional): Seed for fold split. (default: :obj:`None`)

        :rtype: (float, float)
        """
        
        pretrain_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        if isinstance(encoder, list):
            params = [{'params': enc.parameters()} for enc in encoder]
        else:
            params = encoder.parameters()
        p_optimizer = torch.optim.SGD(params, lr=self.p_lr, momentum=0.9, weight_decay=self.p_weight_decay)
        #p_optimizer = torch.optim.AdamW(params, lr=self.p_lr,betas=(0.9, 0.999), eps=1e-2, weight_decay=self.p_weight_decay)
        test_scores_m,preds= [],[]
        for i, enc in enumerate(learning_model.train(encoder, pretrain_loader, 
                                                     p_optimizer, self.p_epoch, True)):
                
            if (i+1)%self.log_interval==0:
                test_acc = []
                #preds = []
                loader = DataLoader(self.dataset, self.batch_size, shuffle=False)
                embed, lbls = self.get_embed(enc.to(self.device), loader)
                train_embs=embed[self.dataset._data.train_mask]
                train_lbls=lbls[self.dataset._data.train_mask]
                test_embs=embed[self.dataset._data.test_mask]
                test_lbls=lbls[self.dataset._data.test_mask]
                if self.classifier == 'LogReg':
                    if self.search:
                        params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
                        classifier = GridSearchCV(LogisticRegression(), params, cv=5, scoring='accuracy', verbose=0,return_train_score=True)
                    else:
                        classifier = LogisticRegression(C=10)
                    classifier.fit(train_embs, train_lbls)
                    train_scores=classifier.score(train_embs, train_lbls)
                    pred=classifier.predict(test_embs)
                    result=classifier.predict(embed)
                    test_acc = accuracy_score(test_lbls, pred)
                    preds.append(np.array(result))
                    test_scores_m.append(test_acc)
                    sd=np.array(test_scores_m).std().item()
                elif self.classifier == 'SVC':

                    if self.search:
                      params = {'C':[1,10,100,1000,10000]}#,'kernel': ('linear','poly' 'rbf'), 'gamma': [1e-3, 1e-4, 'auto', 'scale']}
                      classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0,return_train_score=True)
                    else:
                      classifier = SVC(C=10)
                    classifier.fit(train_embs, train_lbls)
                    train_scores=classifier.score(train_embs, train_lbls)
                    pred=classifier.predict(test_embs)
                    result=classifier.predict(embed)
                    test_acc = accuracy_score(test_lbls, pred)

                    #loss = nn.CrossEntropyLoss(out, train_lbls)
                    #print(result)
                    #result = np.argmax(result)#, dim=1)
                    preds.append(np.array(result))
                    test_scores_m.append(test_acc)
                    sd=np.array(test_scores_m).std().item()
                elif self.classifier == 'RF':
                    rf_clf = RandomForestClassifier(bootstrap=True,
                                                    criterion='entropy', max_depth=15, max_features=0.5,max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                    min_samples_leaf=3, min_samples_split=8, min_weight_fraction_leaf=0.0, n_estimators=185, 
                                                    n_jobs=1, oob_score=False, random_state=42, verbose=0, warm_start=False)
            
                    rf_clf.fit(train_embs, train_lbls)
                    train_scores=rf_clf.score(train_embs, train_lbls)
                    pred=rf_clf.predict(test_embs)
                    result=rf_clf.predict(embed)
                    test_acc = accuracy_score(test_lbls, pred)
                    preds.append(np.array(result))
                    test_scores_m.append(test_acc)
                    sd=np.array(test_scores_m).std().item()
                elif self.classifier == 'XGBoost':
                    train_lbls=train_lbls-1
                    test_lbls=test_lbls-1
                    params = {
                             "colsample_bytree": uniform(0.7, 0.3),
                             "gamma": uniform(0, 0.5),
                             "learning_rate": uniform(0.03, 0.3), # default 0.1 
                             "max_depth": randint(2, 6), # default 3
                             "n_estimators": randint(100, 180), # default 100
                             "subsample": uniform(0.6, 0.4)}
                    #xg_clf = RandomizedSearchCV(xgb.XGBClassifier(objective ='reg:logistic', n_jobs=1, random_state=42),
                    #                             param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=0, n_jobs=1, return_train_score=True)
                    
                    xg_clf = xgb.XGBClassifier(objective ='multi:softprob', colsample_bytree = 0.7, learning_rate = 0.3,
                                                max_depth = 6, alpha = 0.1, n_estimators = 100, n_jobs=30, random_state=12)
                    xg_clf.fit(train_embs, train_lbls)
                    train_scores=xg_clf.score(train_embs, train_lbls)
                    pred=xg_clf.predict(test_embs)
                    result=xg_clf.predict(embed)
                    test_acc = accuracy_score(test_lbls, pred)
                    preds.append(np.array(result+1))
                    test_scores_m.append(test_acc)
                    sd=np.array(test_scores_m).std().item()
        
        idx = np.argmax(test_scores_m)
        acc = test_scores_m[idx]
        result = preds[idx]
        print('Best epoch %d: acc %.4f +/-(%.4f)'%((idx+1)*self.log_interval, acc, sd))
        
        return acc, sd ,idx,result, test_scores_m,train_scores


    def get_embed(self, model, loader):
    
        model.eval()
        ret, y = [], []
        with torch.no_grad():
            for data in loader:
                y.append(data.y.numpy())
                data.to(self.device)
                embed = model(data)
                ret.append(embed.cpu().numpy())#cpu.to(self.device)

        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

# +

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class NodeUnsupervised(object):
    r"""
    The evaluation interface for unsupervised graph representation learning evaluated with 
    linear classification. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph>`_ 
    for examples of usage.
    
    Args:
        full_dataset (torch_geometric.data.Dataset): The graph classification dataset.
        train_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for training. Set to :obj:`None` if included in dataset.
            (default: :obj:`None`)
        val_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for validation. Set to :obj:`None` if included in dataset.
            (default: :obj:`None`)
        test_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for test. Set to :obj:`None` if included in dataset. (default: :obj:`None`)
        classifier (string, optional): Linear classifier for evaluation, :obj:`"SVC"` or 
            :obj:`"LogReg"`. (default: :obj:`"LogReg"`)
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> node_dataset = get_node_dataset("Cora") # using default train/test split
    >>> evaluator = NodeUnsupervised(node_dataset, log_interval=10, device=0)
    >>> evaluator.evaluate(model, encoder)
    
    >>> node_dataset = SomeDataset()
    >>> # Using your own dataset or with different train/test split
    >>> train_mask, val_mask, test_mask = torch.Tensor([...]), torch.Tensor([...]), torch.Tensor([...])
    >>> evaluator = NodeUnsupervised(node_dataset, train_mask, val_mask, test_mask, log_interval=10, device=0)
    >>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, full_dataset, train_mask=None, val_mask=None, test_mask=None, 
                 classifier='LogReg', metric='acc', device=None, log_interval=1, **kwargs):

        self.full_dataset = full_dataset
        self.train_mask = full_dataset[0].train_mask if train_mask is None else train_mask
        self.val_mask = full_dataset[0].val_mask if val_mask is None else val_mask
        self.test_mask = full_dataset[0].test_mask if test_mask is None else test_mask
        self.metric = metric
        self.device = device
        self.search = True
        self.classifier = classifier
        self.log_interval = log_interval
        self.num_classes = full_dataset.num_classes
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        # Use default config if not further specified
        self.setup_train_config(**kwargs)

    def setup_train_config(self, p_optim = 'Adam', p_lr = 0.01, p_weight_decay = 0, 
                           p_epoch = 2000, logreg_wd = 0, comp_embed_on='cpu'):

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.comp_embed_on = comp_embed_on
        self.logreg_wd = logreg_wd

    def evaluate(self, learning_model, encoder):
        r"""Run evaluation with given learning model and encoder(s).
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive)
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.

        :rtype: (float, float)
        """
        
        full_loader = DataLoader(self.full_dataset, 1)
        if isinstance(encoder, list):
            params = [{'params': enc.parameters()} for enc in encoder]
        else:
            params = encoder.parameters()
        
        p_optimizer = self.get_optim(self.p_optim)(params, lr=self.p_lr, 
                                                   weight_decay=self.p_weight_decay)

       
        test_scores_m,preds= [],[]
        per_epoch_out = (self.log_interval<self.p_epoch)
        for i, enc in enumerate(learning_model.train(encoder, full_loader, 
                                                     p_optimizer, self.p_epoch, per_epoch_out)):
            if not per_epoch_out or (i+1)%self.log_interval==0:
                embed, lbls = self.get_embed(enc.to(self.device), full_loader)
                lbs = np.array(preprocessing.LabelEncoder().fit_transform(lbls))
                
                test_scores = []
                for _ in range(10):
                  train_embs=embed[self.train_mask]
                  train_lbls=lbls[self.train_mask]
                  test_embs=embed[self.test_mask]
                  test_lbls=lbls[self.test_mask]
                  if self.classifier == 'SVC':

                    if self.search:
                      params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
                      classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0,return_train_score=True)
                    else:
                      classifier = SVC(C=10)
                    classifier.fit(train_embs, train_lbls)
                    train_scores=classifier.score(train_embs, train_lbls)
                    pred=classifier.predict(test_embs)
                    result=classifier.predict(embed)
                    test_acc = accuracy_score(test_lbls, pred)

                    #loss = nn.CrossEntropyLoss(out, train_lbls)
                    #print(result)
                    #result = np.argmax(result)#, dim=1)
                    preds.append(np.array(result))
                    test_scores_m.append(test_acc)
                    sd=np.array(test_scores_m).std().item()
                  elif self.classifier == 'LogReg':
                      train_embs = torch.Tensor(train_embs).to(self.device)
                      train_lbls = torch.Tensor(train_lbls).to(self.device)
                      test_embs = torch.Tensor(test_embs).to(self.device)
                      test_lbls = torch.Tensor(test_lbls).to(self.device)
                      embed = torch.Tensor(embed).to(self.device)
                      hid_units = train_embs.shape[1]
                      xent = nn.CrossEntropyLoss()
                      log = LogReg(hid_units, self.num_classes)
                      log.to(self.device)
                      opt = torch.optim.Adam(log.parameters(), lr=0.01, 
                               weight_decay=self.logreg_wd)

                      
                      
                      for it in range(300):
                          log.train()
                          opt.zero_grad()

                          logits = log(train_embs).float()
                          train_scores =  torch.sum(torch.argmax(logits, dim=1) == train_lbls).float() / train_lbls.shape[0]
                          loss = xent(logits, train_lbls.long())
                           
                          loss.backward()
                          opt.step()
                          pred=log(test_embs)
                          pred = torch.argmax(pred, dim=1)
                          result=log(embed)
                          result = torch.argmax(result, dim=1)
                          test_acc = torch.sum(pred == test_lbls).float() / test_lbls.shape[0]
                          test_acc=np.array(test_acc.cpu())
                      preds.append(result.cpu().numpy())
                      test_scores_m.append(test_acc)
                      sd=np.array(test_scores_m).std().item()
                  elif self.classifier == 'RF':
                    rf_clf = RandomForestClassifier(bootstrap=True,
                                                    criterion='entropy', max_depth=15, max_features=0.5,max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                    min_samples_leaf=3, min_samples_split=8, min_weight_fraction_leaf=0.0, n_estimators=185, 
                                                    n_jobs=1, oob_score=False, random_state=42, verbose=0, warm_start=False)
            
                    rf_clf.fit(train_embs, train_lbls)
                    train_scores=rf_clf.score(train_embs, train_lbls)
                    pred=rf_clf.predict(test_embs)
                    result=rf_clf.predict(embed)
                    test_acc = accuracy_score(test_lbls, pred)
                    preds.append(np.array(result))
                    test_scores_m.append(test_acc)
                    sd=np.array(test_scores_m).std().item()
        idx = np.argmax(test_scores_m)
        acc = test_scores_m[idx]
        result = preds[idx]
        print('Best epoch %d: acc %.4f +/-(%.4f)'%((idx+1)*self.log_interval, acc, sd))
        
        return acc, sd ,idx,result, test_scores_m,train_scores
     
    
    def get_embed(self, model, loader):
    
        model.eval()
        model.to(self.comp_embed_on)
        ret, y = [], []
        with torch.no_grad():
            for data in loader:
                y.append(data.y.numpy())
                data.to(self.comp_embed_on)
                embed = model(data)
                ret.append(embed.cpu().numpy())
                
        model.to(self.device)
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
        
        
    
    def get_optim(self, optim):
        
        optims = {'Adam': torch.optim.Adam}
        
        return optims[optim]


# -

class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=True, act='prelu', bias=True, xavier=True, edge_weight=True):
        super(GCN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool
        self.edge_weight = edge_weight
        self.normalize = not edge_weight
        self.add_self_loops = not edge_weight

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias,
                           add_self_loops=self.add_self_loops,
                           normalize=self.normalize)
            if xavier:
                self.weights_init(conv.lin)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_weight:
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        xs = []
        for i in range(self.n_layers):
            
            x = self.convs[i](x, edge_index, edge_attr)
            #x = F.dropout(x, p=0.2)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)
        
        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
            x_stack=global_add_pool(x, batch)
        elif self.pool == 'mean':
            xpool = [global_mean_pool(x, batch) for x in xs]
            x_stack=global_mean_pool(x, batch)
        elif self.pool == 'max':
            xpool = [global_max_pool(x, batch) for x in xs]
            x_stack=global_max_pool(x, batch)
        global_rep = torch.cat(xpool, 1)

        return  global_rep,x_stack

class Encoder(torch.nn.Module):
    r"""A wrapped :class:`torch.nn.Module` class for the convinient instantiation of 
    pre-implemented graph encoders.
    
    Args:
        feat_dim (int): The dimension of input node features.
        hidden_dim (int): The dimension of node-level (local) embeddings. 
        n_layer (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
        pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
            (default: :obj:`sum`)
        gnn (string, optional): The type of GNN layer, :obj:`gcn`, :obj:`gin` or 
            :obj:`resgcn`. (default: :obj:`gin`)
        bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
        act (string, optional): The activation function, :obj:`relu` or :obj:`prelu`.
            (default: :obj:`relu`)
        bias (bool, optional): Whether to include bias term in Linear. (default: :obj:`True`)
        xavier (bool, optional): Whether to apply xavier initialization. (default: :obj:`True`)
        node_level (bool, optional): If :obj:`True`, the encoder will output node level
            embedding (local representations). (default: :obj:`False`)
        graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
            embeddings (global representations). (default: :obj:`True`)
        edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
            compute the aggregation. (default: :obj:`False`)
    """
    def __init__(self, feat_dim, hidden_dim, n_layers=5, pool='sum', 
                  node_level=True, graph_level=False, **kwargs):
        super(Encoder, self).__init__()
        self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, **kwargs)
        self.linear = Linear(hidden_dim*n_layers, hidden_dim)
        self.node_level = node_level
        self.graph_level = graph_level
        self.sigm = nn.Sigmoid()
        self.hidden_dim=hidden_dim
        self.n_layers=n_layers
    def forward(self, data):
        z_g, z_n = self.encoder(data)
        #z_g = self.sigm(z_g)
        z_g = self.linear(z_g)
        #z_g = self.read(z_g, data.batch)
        #z_n = self.sigm(z_n)
        #z_n = self.read(z_n, None)
        if self.node_level and self.graph_level:
            return z_g, z_n
        elif self.graph_level:
            return z_g
        else:
            return z_n

class GraphCL(Contrastive):
    r"""    
    Contrastive learning method proposed in the paper `Graph Contrastive Learning with 
    Augmentations <https://arxiv.org/abs/2010.13902>`_. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/blob/dig/benchmarks/sslgraph/example_graphcl.ipynb>`_ for
    an example of usage.

    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`GraphCL`.
    
    Args:
        dim (int): The embedding dimension.
        aug1 (sting, optinal): Types of augmentation for the first view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug2 (sting, optinal): Types of augmentation for the second view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug_ratio (float, optional): The ratio of augmentations. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """
    
    def __init__(self, dim, aug_1, aug_2, aug_ratio=0.2, lamda=0.87,**kwargs):
        
        views_fn = []
        
        for aug in [aug_1, aug_2]:
            if aug is None:
                views_fn.append(lambda x: x)
            elif aug == 'dropN':
                views_fn.append(UniformSample(ratio=aug_ratio))
            elif aug == 'permE':
                views_fn.append(EdgePerturbation(drop=True,add=False,ratio=aug_ratio))
            elif aug == 'subgraph':
                views_fn.append(RWSample(ratio=aug_ratio))
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=aug_ratio))
            elif aug == 'noise':
                views_fn.append(AddGussianNoise(std=0.05,aug_ratio=aug_ratio))
            elif aug == 'weak':
                views_fn.append(weak(aug_ratio=aug_ratio))
            elif aug == 'strong':
                views_fn.append(strong(aug_ratio=aug_ratio))
            elif aug == 'weak2':
                canditates = [
                              EdgePerturbation(drop=True,add=False,ratio=0.05)]
                views_fn.append(RandomView(canditates))
            elif aug == 'strong2':
                canditates = [EdgePerturbation(drop=True,add=False,ratio=aug_ratio),
                              AddGussianNoise(std=0.05,aug_ratio=aug_ratio),
                              UniformSample(ratio=aug_ratio)
                              ]
                views_fn.append(RandomView(canditates))
            elif aug == 'random4':
                canditates = [UniformSample(ratio=aug_ratio),
                              RWSample(ratio=aug_ratio),
                              EdgePerturbation(ratio=aug_ratio),
                              NodeAttrMask(mask_ratio=aug_ratio)]
                views_fn.append(RandomView(canditates))
            elif aug == 'random':
                views_fn.append(random(aug_ratio=aug_ratio))
            else:
                raise Exception("Aug must be from [dropN', 'permE', 'subgraph', \
                                'maskN', 'random2', 'random3', 'random4'] or None.")
        #views_fn.append(lambda x: x)
        super(GraphCL, self).__init__(objective='NCE',
                                      views_fn=views_fn,
                                      z_dim=dim,
                                      proj='MLP',
                                      lamda=lamda,
                                      node_level=False,
                                      graph_level=True,
                                      choice_model='last',
                                      **kwargs)
        
    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, proj in super(GraphCL, self).train(encoders, data_loader, 
                                                    optimizer, epochs, per_epoch_out):
            
            yield enc

