import os
import torch
import numpy as np
from tqdm import trange
import torch.nn as nn
from torch_geometric.data import Batch, Data
from dig.sslgraph.method.contrastive.objectives import JSE_loss#,NCE_loss
from matplotlib import pyplot as plt

class Contrastive(nn.Module):
    r"""
    Base class for creating contrastive learning models for either graph-level or 
    node-level tasks.
    
    *Alias*: :obj:`dig.sslgraph.method.contrastive.model.`:obj:`Contrastive`.

    Args:
        objective (string, or callable): The learning objective of contrastive model.
            If string, should be one of 'NCE' and 'JSE'. If callable, should take lists
            of representations as inputs and returns loss Tensor 
            (see `dig.sslgraph.method.contrastive.objectives` for examples).
        views_fn (list of callable): List of functions to generate views from given graphs.
        graph_level (bool, optional): Whether to include graph-level representation 
            for contrast. (default: :obj:`True`)
        node_level (bool, optional): Whether to include node-level representation 
            for contrast. (default: :obj:`False`)
        z_dim (int, optional): The dimension of graph-level representations. 
            Required if :obj:`graph_level` = :obj:`True`. (default: :obj:`None`)
        z_dim (int, optional): The dimension of node-level representations. 
            Required if :obj:`node_level` = :obj:`True`. (default: :obj:`None`)
        proj (string, or Module, optional): Projection head for graph-level representation. 
            If string, should be one of :obj:`"linear"` or :obj:`"MLP"`. Required if
            :obj:`graph_level` = :obj:`True`. (default: :obj:`None`)
        proj_n (string, or Module, optional): Projection head for node-level representations. 
            If string, should be one of :obj:`"linear"` or :obj:`"MLP"`. Required if
            :obj:`node_level` = :obj:`True`. (default: :obj:`None`)
        neg_by_crpt (bool, optional): The mode to obtain negative samples in JSE. If True, 
            obtain negative samples by performing corruption. Otherwise, consider pairs of
            different graph samples as negative pairs. Only used when 
            :obj:`objective` = :obj:`"JSE"`. (default: :obj:`False`)
        tau (int): The tempurature parameter in InfoNCE (NT-XENT) loss. Only used when 
            :obj:`objective` = :obj:`"NCE"`. (default: :obj:`0.5`)
        device (int, or `torch.device`, optional): The device to perform computation.
        choice_model (string, optional): Whether to yield model with :obj:`best` training loss or
            at the :obj:`last` epoch. (default: :obj:`last`)
        model_path (string, optinal): The directory to restore the saved model. 
            (default: :obj:`models`)
    """
    
    def __init__(self, objective, views_fn,
                 graph_level=True,
                 node_level=False,
                 z_dim=None,
                 z_n_dim=None,
                 proj=None,
                 proj_n=None,
                 neg_by_crpt=False,
                 tau=0.5,
                 lamda=0.87,
                 device=None,
                 choice_model='last',
                 model_path='models'):

        assert node_level is not None or graph_level is not None
        assert not (objective=='NCE' and neg_by_crpt)

        super(Contrastive, self).__init__()
        self.loss_fn = self._get_loss(objective)
        self.views_fn = views_fn # fn: (batched) graph -> graph
        self.node_level = node_level
        self.graph_level = graph_level
        self.z_dim = z_dim
        self.z_n_dim = z_n_dim
        self.proj = proj
        self.proj_n = proj_n
        self.neg_by_crpt = neg_by_crpt
        self.tau = tau
        self.lamda = lamda
        self.choice_model = choice_model
        self.model_path = model_path
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device
        
        
        
    def train(self, encoder, data_loader, optimizer, epochs, per_epoch_out=False):
        r"""Perform contrastive training and yield trained encoders per epoch or after
        the last epoch.
        
        Args:
            encoder (Module, or list of Module): A graph encoder shared by all views or a list 
                of graph encoders dedicated for each view. If :obj:`node_level` = :obj:`False`, 
                the encoder should return tensor of shape [:obj:`n_graphs`, :obj:`z_dim`].
                Otherwise, return tuple of shape ([:obj:`n_graphs`, :obj:`z_dim`], 
                [:obj:`n_nodes`, :obj:`z_n_dim`]) representing graph-level and node-level embeddings.
            dataloader (Dataloader): Dataloader for unsupervised learning or pretraining.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in encoder(s).
            epochs (int): Number of total training epochs.
            per_epoch_out (bool): If True, yield trained encoders per epoch. Otherwise, only yield
                the final encoder at the last epoch. (default: :obj:`False`)
                
        :rtype: :class:`generator`.
        """
        self.per_epoch_out = per_epoch_out
        
        if self.z_n_dim is None:
            self.proj_out_dim = self.z_dim
        else:
            self.proj_out_dim = self.z_n_dim
        
        if self.graph_level and self.proj is not None:
            self.proj_head_g = self._get_proj(self.proj, self.z_dim).to(self.device)
            optimizer.add_param_group({"params": self.proj_head_g.parameters()})
        elif self.graph_level:
            self.proj_head_g = lambda x: x
        else:
            self.proj_head_g = None
            
        if self.node_level and self.proj_n is not None:
            self.proj_head_n = self._get_proj(self.proj_n, self.z_n_dim).to(self.device)
            optimizer.add_param_group({"params": self.proj_head_n.parameters()})
        elif self.node_level:
            self.proj_head_n = lambda x: x
        else:
            self.proj_head_n = None
        
        if isinstance(encoder, list):
            encoder = [enc.to(self.device) for enc in encoder]
        else:
            encoder = encoder.to(self.device)
            
        if self.node_level and self.graph_level:
            train_fn = self.train_encoder_node_graph
        elif self.graph_level:
            train_fn = self.train_encoder_graph
        else:
            train_fn = self.train_encoder_node
            
        for enc in train_fn(encoder, data_loader, optimizer, epochs):
            yield enc

        
    def train_encoder_graph(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be Tensor for graph-level embedding
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
            [enc.train() for enc in encoders]
        else:
            encoder.train()
            encoders = [encoder]*len(self.views_fn)

        try:
            self.proj_head_g.train()
        except:
            pass
        
        min_loss = 1e9
        out_loss = []
        out_loss_pos = []
        out_loss_neg = []
        with trange(epochs) as t:
            for epoch in t:
                epoch_loss = 0.0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                iteration = 0
                for data in data_loader:
                    iteration += 1
                    optimizer.zero_grad()
                    if None in self.views_fn: 
                        # For view fn that returns multiple views
                        views = []
                        for v_fn in self.views_fn:
                            if v_fn is not None:
                                views += [*v_fn(data)]
                    else:
                        views = [v_fn(data) for v_fn in self.views_fn]
                    
                    zs = []
                    for view, enc in zip(views, encoders):
                        z = self._get_embed(enc, view.to(self.device))
                        zs.append(self.proj_head_g(z))
                    #print(zs[0].shape,zs[1].shape,'NCE',self.views_fn)
                    loss,loss_pos,loss_neg = self.loss_fn(zs, neg_by_crpt=self.neg_by_crpt, tau=self.tau,lamda=self.lamda,iteration=iteration)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    
                loss_pos = loss_pos.cpu().detach().numpy()
                loss_neg = loss_neg.cpu().detach().numpy()
                if self.per_epoch_out:
                    yield encoder, self.proj_head_g
                        
                t.set_postfix(loss='{:.6f}'.format(float(loss)))
                out_loss.append(loss.cpu().detach().numpy() )
                out_loss_pos.append(loss_pos)
                out_loss_neg.append(loss_neg)
                if self.choice_model == 'best' and epoch_loss < min_loss:
                    min_loss = epoch_loss

                    if not os.path.exists(self.model_path):
                        try:
                            os.mkdir(self.model_path)
                        except:
                            raise RuntimeError('cannot create model path')

                    if isinstance(encoder, list):
                        for i, enc in enumerate(encoder):
                            torch.save(enc.state_dict(), self.model_path+'/enc%d_best.pkl'%i)
                    else:
                        torch.save(encoder.state_dict(), self.model_path+'/enc_best.pkl')
            
            if self.choice_model == 'best':
                
                if not os.path.exists(self.model_path):
                    try:
                        os.mkdir(self.model_path)
                    except:
                        raise RuntimeError('cannot create model path')

                if isinstance(encoder, list):
                    for i, enc in enumerate(encoder):
                        enc.load_state_dict(torch.load(self.model_path+'/enc%d_best.pkl'%i))
                else:
                    encoder.load_state_dict(torch.load(self.model_path+'/enc_best.pkl'))

        if not self.per_epoch_out:
            yield encoder, self.proj_head_g
        fig= plt.figure(figsize=(5, 4)) 
        plt.plot(out_loss, color='green')
        plt.plot(out_loss_pos,color='red')
        plt.plot(out_loss_neg,color='blue')
        plt.title('Pretrain loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Pretrain','Loss_pos','Loss_neg'], loc='upper right')
        plt.show()
        np.savetxt("pretrain_loss.csv", out_loss, delimiter=",")


    
    def train_encoder_node(self, encoder, data_loader, optimizer, epochs,objective='NCE'):
        
        # output of each encoder should be Tensor for node-level embedding
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
            [encoder.train() for encoder in encoders]
        else:
            encoder.train()
            encoders = [encoder]*len(self.views_fn)
        
        try:
            self.proj_head_n.train()
        except:
            pass
        
        min_loss = 1e9
        with trange(epochs) as t:
            for epoch in t:
                epoch_loss = 0.0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                for data in data_loader:
                    optimizer.zero_grad()
                    if None in self.views_fn:
                        # For view fn that returns multiple views
                        views = []
                        for v_fn in self.views_fn:
                            if v_fn is not None:
                                views += [*v_fn(data)]
                    else:
                        views = [v_fn(data) for v_fn in self.views_fn]

                    zs_n = []
                    for view, enc in zip(views, encoders):
                        z_n = self._get_embed(enc, view.to(self.device))###
                        zs_n.append(self.proj_head_n(z_n))
                    #print(zs_n[0].shape,zs_n[1].shape,objective,self.views_fn)
                    if objective=='NCE':
                        print('NCE')
                        loss = self.loss_fn(zs=None, zs_n=zs_n, batch=data.batch, 
                                        neg_by_crpt=self.neg_by_crpt, tau=self.tau,lamda=self.lamda)
                    elif objective=='JSE':
                        print('JSE')
                        loss = self.loss_fn(zs=zs_n, zs_n=None, batch=data.batch,
                                        neg_by_crpt=self.neg_by_crpt)

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                print(epoch_loss,loss)
                if self.per_epoch_out:
                    yield encoder, self.proj_head_n
                    
                t.set_postfix(loss='{:.6f}'.format(float(loss)))
                
                if self.choice_model == 'best' and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    if isinstance(encoder, list):
                        for i, enc in enumerate(encoder):
                            torch.save(enc.state_dict(), self.model_path+'/enc%d_best.pkl'%i)
                    else:
                        torch.save(encoder.state_dict(), self.model_path+'/enc_best.pkl')
            
            if self.choice_model == 'best':
                if isinstance(encoder, list):
                    for i, enc in enumerate(encoder):
                        enc.load_state_dict(torch.load(self.model_path+'/enc%d_best.pkl'%i))
                else:
                    encoder.load_state_dict(torch.load(self.model_path+'/enc_best.pkl'))

        if not self.per_epoch_out:
            yield encoder, self.proj_head_n
    
    
    def train_encoder_node_graph(self, encoder, data_loader, optimizer, epochs):
        
        # output of each encoder should be tuple of (node_embed, graph_embed)
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
            [encoder.train() for encoder in encoders]
        else:
            encoder.train()
            encoders = [encoder]*len(self.views_fn)
        
        try:
            self.proj_head_n.train()
            self.proj_head_g.train()
        except:
            pass
        min_loss = 1e9
        with trange(epochs) as t:
            for epoch in t:
                epoch_loss = 0.0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                for data in data_loader:
                    optimizer.zero_grad()
                    if None in self.views_fn:
                        views = []
                        for v_fn in self.views_fn:
                            # For view fn that returns multiple views
                            if v_fn is not None:
                                views += [*v_fn(data)]
                        assert len(views)==len(encoders)
                    else:
                        views = [v_fn(data) for v_fn in self.views_fn]

                    zs_n, zs_g = [], []
                    for view, enc in zip(views, encoders):
                        z_g, z_n = self._get_embed(enc, view.to(self.device))
                        zs_n.append(self.proj_head_n(z_n))
                        zs_g.append(self.proj_head_g(z_g))

                    loss = self.loss_fn(zs_g, zs_n=zs_n, batch=data.batch, 
                                        neg_by_crpt=self.neg_by_crpt, tau=self.tau,lamda=self.lamda)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    
                if self.per_epoch_out:
                    yield encoder, (self.proj_head_g, self.proj_head_n)
                        

                t.set_postfix(loss='{:.6f}'.format(float(loss)))
                
                if self.choice_model == 'best' and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    if isinstance(encoder, list):
                        for i, enc in enumerate(encoder):
                            torch.save(enc.state_dict(), self.model_path+'/enc%d_best.pkl'%i)
                    else:
                        torch.save(encoder.state_dict(), self.model_path+'/enc_best.pkl')
            
            if self.choice_model == 'best' and not self.per_epoch_out:
                if isinstance(encoder, list):
                    for i, enc in enumerate(encoder):
                        enc.load_state_dict(torch.load(self.model_path+'/enc%d_best.pkl'%i))
                else:
                    encoder.load_state_dict(torch.load(self.model_path+'/enc_best.pkl'))

        if not self.per_epoch_out:
            yield encoder, (self.proj_head_g, self.proj_head_n)
    
    def _get_embed(self, enc, view):
        
        if self.neg_by_crpt:
            view_crpt = self._corrupt_graph(view)
            if self.node_level and self.graph_level:
                z_g, z_n = enc(view)
                z_g_crpt, z_n_crpt = enc(view_crpt)
                z = (torch.cat([z_g, z_g_crpt], 0),
                     torch.cat([z_n, z_n_crpt], 0))
            else:
                z = enc(view)
                z_crpt = enc(view_crpt)
                z = torch.cat([z, z_crpt], 0)
        else:
            z = enc(view)
        
        return z
                
    
    def _corrupt_graph(self, view):
        
        data_list = view.to_data_list()
        crpt_list = []
        for data in data_list:
            n_nodes = data.x.shape[0]
            perm = torch.randperm(n_nodes).long()
            crpt_x = data.x[perm]
            crpt_list.append(Data(x=crpt_x, edge_index=data.edge_index))
        view_crpt = Batch.from_data_list(crpt_list)

        return view_crpt
        
    
    def _get_proj(self, proj_head, in_dim):
        
        if callable(proj_head):
            return proj_head
        
        assert proj_head in ['linear', 'MLP']
        
        out_dim = self.proj_out_dim
        
        if proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
        elif proj_head == 'MLP':
            proj_nn = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(out_dim, out_dim))
            for m in proj_nn.modules():
                self._weights_init(m)
            
        return proj_nn
        
    def _weights_init(self, m):        
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def _get_loss(self, objective):
        
        if callable(objective):
            return objective
        
        assert objective in ['JSE', 'NCE']
        
        return {'JSE':JSE_loss, 'NCE':NCE_loss}[objective]

import itertools
import torch
import torch.nn.functional as F


def NCE_loss(zs=None, zs_n=None, batch=None, sigma=None,lamda=0.87, **kwargs):
    '''The InfoNCE (NT-XENT) loss in contrastive learning.
    
    Args:
        zs (list, optipnal): List of tensors of shape [batch_size, z_dim].
        zs_n (list, optional): List of tensors of shape [nodes, z_dim].
        batch (Tensor, optional): Required when both :obj:`zs` and :obj:`zs_n` are given.
        sigma (ndarray, optional): A 2D-array of shape [:obj:`n_views`, :obj:`n_views`] with boolean 
            values, indicating contrast between which two views are computed. Only required 
            when number of views is greater than 2. If :obj:`sigma[i][j]` = :obj:`True`, 
            infoNCE between :math:`view_i` and :math:`view_j` will be computed.
        tau (int, optional): The temperature used in NT-XENT.

    :rtype: :class:`Tensor`
    '''
    assert zs is not None or zs_n is not None
    
    if 'tau' in kwargs:
        tau = kwargs['tau']
    else:
        tau = 0.5
    
    if 'norm' in kwargs:
        norm = kwargs['norm']
    else:
        norm = True
    if 'iteration' in kwargs:
        iteration = kwargs['iteration']
    
    mean = kwargs['mean'] if 'mean' in kwargs else True
        
    if zs_n is not None:
        if zs is None:
            # InfoNCE in GRACE
            assert len(zs_n)==2
            return (infoNCE_local_intra_node(zs_n[0], zs_n[1], tau, norm, batch)+
                    infoNCE_local_intra_node(zs_n[1], zs_n[0], tau, norm, batch))*0.5
        else:
            assert len(zs_n)==len(zs)
            assert batch is not None
            
            if len(zs)==1:
                return infoNCE_local_global(zs[0], zs_n[0], batch, tau, norm)
            elif len(zs)==2:
                return (infoNCE_local_global(zs[0], zs_n[1], batch, tau, norm)+
                        infoNCE_local_global(zs[1], zs_n[0], batch, tau, norm))
            else:
                assert len(zs)==len(sigma)
                loss = 0
                for (i, j) in itertools.combinations(range(len(zs)), 2):
                    if sigma[i][j]:
                        loss += (infoNCE_local_global(zs[i], zs_n[j], batch, tau, norm)+
                                 infoNCE_local_global(zs[j], zs_n[i], batch, tau, norm))
                loss_pos = 0
                loss_neg = 0
                return loss
    
    if len(zs)==2:
        loss,loss_pos,loss_neg = NT_Xent(zs[0], zs[1], tau,lamda, norm,iteration)
        return loss,loss_pos,loss_neg
    elif len(zs)>2:
        assert len(zs)==len(sigma)
        loss = 0
        loss_pos = 0
        loss_neg = 0
        for (i, j) in itertools.combinations(range(len(zs)), 2):
            if sigma[i][j]:
                loss_,loss_pos_,loss_neg_ = NT_Xent(zs[i], zs[j], tau,lamda, norm,iteration)
                loss += loss_
                loss_pos += loss_pos_
                loss_neg += loss_neg_
        return loss,loss_pos,loss_neg


def infoNCE_local_intra_node(z1_n, z2_n, tau=0.5, norm=True, batch=None):
    '''
    Args:
        z1_n: Tensor of shape [n_nodes, z_dim].
        z2_n: Tensor of shape [n_nodes, z_dim].
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
        batch: Tensor of shape [batch_size]
    '''
    def sim(z1:torch.Tensor, z2:torch.Tensor):
            if norm:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())
    
    exp = lambda x: torch.exp(x / tau)
    if batch is not None:
        batch_size = batch.size(0)
        num_nodes = z1_n.size(0)
        indices = torch.arange(0, num_nodes).to(z1_n.device)
        losses = []
        for i in range(0, num_nodes, batch_size):
            mask = indices[i:i+batch_size]
            refl_sim = exp(sim(z1_n[mask], z1_n))
            between_sim = exp(sim(z1_n[mask], z2_n))
            losses.append(-torch.log(between_sim[:, i:i+batch_size].diag()
                            / (refl_sim.sum(1) + between_sim.sum(1)
                            - refl_sim[:, i:i+batch_size].diag())))
        losses = torch.cat(losses)
        return losses.mean()

    refl_sim = exp(sim(z1_n, z1_n))
    between_sim = exp(sim(z1_n, z2_n))
    
    pos_sim = between_sim.diag()
    intra_sim = refl_sim.sum(1) - refl_sim.diag()
    inter_pos_sim = between_sim.sum(1)
    
    loss = pos_sim / (intra_sim + inter_pos_sim)
    loss = -torch.log(loss).mean()
    #print('intra_node')
    return loss

    

def infoNCE_local_global(z_n, z_g, batch, tau=0.5, norm=True):
    '''
    Args:
        z_n: Tensor of shape [n_nodes, z_dim].
        z_g: Tensor of shape [n_graphs, z_dim].
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    # Not yet used in existing methods, to be implemented.
    loss = 0

    return loss

def normalize(x):
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x_min = torch.min(x, dim=1, keepdim=True)[0]
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def NT_Xent2(z1, z2, tau=0.5, norm=True):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    
    batch_size, _ = z1.size()
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
    
    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
    print(sim_matrix)
    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) + pos_sim)
    loss = - torch.log(loss).mean()
    return loss

def NT_Xent(z1, z2, tau=0.5,lamda=0.87,  norm=True,iteration=0):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size 16, z_dim 1024]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    z2 = z2.detach()
    batch_size, _ = z1.size()
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
    #if iteration==1:
        #fig= plt.figure(figsize=(10, 10))
        #plt.imshow(sim_matrix.cpu().detach().numpy(),cmap='coolwarm',vmin=-0.2,vmax=1)
        #plt.show()
    #print('before: ',sim_matrix)
    sorted,indices = sim_matrix.sort(stable=True)
    #print(sorted,indices)
    num_drop=int(batch_size*(1-lamda))
    idx_drop=indices[:,:(batch_size-num_drop)]
    max_row = torch.max(sim_matrix, dim=1, keepdim=True)[0]
    min_row = torch.min(sim_matrix, dim=1, keepdim=True)[0]
    threshold = (max_row - min_row) * 0.7 + min_row
    #print('threshold: ',threshold)
    
    
    #print('idx_drop: ',idx_drop)
    neg_sim = sim_matrix.clone()
    neg_sim[range(batch_size), range(batch_size)]=0
    #droped_sim_matrix = neg_sim.clone()[torch.arange(batch_size).unsqueeze(1),idx_drop]
    #droped_sim_matrix_ = neg_sim.clone()[:,idx_drop]
    droped_sim_matrix = neg_sim.clone()
    droped_sim_matrix[droped_sim_matrix>threshold]=0
    sim_matrix = torch.exp(sim_matrix / tau)
    neg_sim = torch.exp(droped_sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    up=- torch.log(pos_sim)
    down=torch.log(neg_sim.sum(dim=1) + pos_sim)
    #print('up: ',up,'dwon: ',down)
    #print(neg_sim.sum(dim=1))
    loss = pos_sim / (neg_sim.sum(dim=1) + pos_sim)
    loss = - torch.log(loss).mean()
 
    return loss,up.mean(),down.mean()


def NT_Xent1(z1, z2, tau=0.5,norm=True,iteration=0):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size 16, z_dim 1024]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    #z2 = z2.detach()
    batch_size,_=z1.size()
    
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
    if iteration==1:
        fig= plt.figure(figsize=(10, 10))
        plt.imshow(sim_matrix.cpu().detach().numpy(),cmap='coolwarm',vmin=-0.2,vmax=1)
        plt.show()
    #print('before: ',sim_matrix)
    sorted,indices = sim_matrix.sort(stable=True)
    #print(sorted,indices)
    num_drop=int(batch_size/5)
    idx_drop=indices[:,:(batch_size-num_drop)]
    max_row = torch.max(sim_matrix, dim=1, keepdim=True)[0]
    min_row = torch.min(sim_matrix, dim=1, keepdim=True)[0]
    threshold = 1#(max_row - min_row) * 0.7 + min_row
    #print('threshold: ',threshold)
    
    
    #print('idx_drop: ',idx_drop)
    neg_sim = sim_matrix.clone()
    neg_sim[range(batch_size), range(batch_size)]=0
    droped_sim_matrix = neg_sim.clone()#[torch.arange(batch_size).unsqueeze(1),idx_drop]
    #droped_sim_matrix_ = neg_sim.clone()[:,idx_drop]
    #droped_sim_matrix = neg_sim.clone()
    #droped_sim_matrix[droped_sim_matrix>threshold]=0
    sim_matrix = torch.exp(sim_matrix / tau)
    neg_sim = torch.exp(droped_sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    up=- torch.log(pos_sim)
    down=torch.log(neg_sim.sum(dim=1) + pos_sim)
    #print('up: ',up,'dwon: ',down)
    #print(neg_sim.sum(dim=1))
    loss = pos_sim / (neg_sim.sum(dim=1) + pos_sim)
    loss = - torch.log(loss).mean()
 
    return loss,up.mean(),down.mean()
