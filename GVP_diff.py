import torch
import torch.nn as nn
from .utils import register_model


@register_model(name='GVPTransCond')
class GVPTransCond(torch.nn.Module):
    '''
    Using time embedding as the only condition!

    GVP-GNN for **single** structure-conditioned autoregressive RNA design.

    Takes in RNA structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 4 bases at each position in a `torch.Tensor` of shape [n_nodes, 4].

    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    :param node_in_dim: node dimensions in input graph
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph
    :param edge_h_dim: edge dimensions to embed in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in encoder/decoder
    :param drop_rate: rate to use in all dropout layers
    :param out_dim: output dimension (4 bases)
    '''

    def __init__(self, config):
        super().__init__()
        self.node_in_dim = tuple(config.model.node_in_dim)  # node_in_dim
        self.node_h_dim = tuple(config.model.node_h_dim)  # node_h_dim
        self.edge_in_dim = tuple(config.model.edge_in_dim)  # edge_in_dim
        self.edge_h_dim = tuple(config.model.edge_in_dim)  # edge_h_dim
        self.num_layers = config.model.num_layers
        self.out_dim = config.model.out_dim
        self.time_cond = config.model.time_cond
        self.dihedral_angle = config.model.dihedral_angle
        self.drop_struct = config.model.drop_struct
        # assert config.data.construct_simplified
        drop_rate = config.model.drop_rate
        activations = (F.relu, None)

        self.construct_simplified = config.data.construct_simplified
        if self.construct_simplified:
            self.gvp_feat_fn = functools.partial(gvp_featurize,
                                                 num_posenc=config.data.num_posenc,
                                                 num_rbf=config.data.num_rbf,
                                                 knn_num=config.data.top_k)
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(self.node_h_dim, self.edge_h_dim,
                         activations=activations, vector_gate=True,
                         drop_rate=drop_rate)
            for _ in range(self.num_layers))

        # Output
        self.W_out = GVP(self.node_h_dim, (self.node_h_dim[0], 0), activations=(None, None))

        # Transformer Layers
        self.seq_res = nn.Linear(self.node_in_dim[0], self.node_h_dim[0])
        self.mix_lin = nn.Linear(self.node_h_dim[0] * 2, self.node_h_dim[0])
        self.num_trans_layer = config.model.num_trans_layer
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.node_h_dim[0],
            -1,
        )
        self.trans_layers = nn.ModuleList(
            TransformerEncoderCondLayer(config.model.trans)
            for _ in range(self.num_trans_layer))
        self.MLP_out = nn.Sequential(
            nn.Linear(self.node_h_dim[0], self.node_h_dim[0]),
            nn.ReLU(),
            nn.Linear(self.node_h_dim[0], self.out_dim)
        )

        # Time conditioning
        if self.time_cond:
            learned_sinu_pos_emb_dim = 16
            time_cond_dim = config.model.node_h_dim[0] * 2
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
            sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
            self.to_time_hiddens = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, config.model.node_h_dim[0]),
            )

        # Dihedral angle
        if self.dihedral_angle:
            if config.data.atom7:
                self.embed_dihedral = DihedralFeatures_atom7(config.model.node_h_dim[0])
            else:
                self.embed_dihedral = DihedralFeatures(config.model.node_h_dim[0])

    def struct_forward(self, batch, init_seq, batch_size, length, **kwargs):
        h_V = (init_seq, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        if self.dihedral_angle:
            dihedral_feats = self.embed_dihedral(batch.coords).reshape_as(h_V[0])
            h_V = (h_V[0] + dihedral_feats, h_V[1])

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        gvp_output = self.W_out(h_V).reshape(batch_size, length, -1)
        return gvp_output

    def forward(self, batch, cond_drop_prob=0., **kwargs):
        # construct extra node and edge features
        if self.construct_simplified:
            batch = self.gvp_feat_fn(batch)
        batch, batch_size, length = geo_batch(batch)

        z_t = batch.z_t
        cond_x = kwargs.get('cond_x', None)
        if cond_x is None:
            cond_x = torch.zeros_like(batch.z_t)
        else:
            cond_x = cond_x.reshape_as(batch.z_t)

        init_seq = torch.cat([z_t, cond_x], -1)

        if self.training:
            if self.drop_struct > 0 and random.random() < self.drop_struct:
                gvp_output = torch.zeros(batch_size, length, self.node_h_dim[0], device=batch.z_t.device)
            else:
                gvp_output = self.struct_forward(batch, init_seq, batch_size, length, **kwargs)
        else:
            if cond_drop_prob == 0.:
                gvp_output = self.struct_forward(batch, init_seq, batch_size, length, **kwargs)
            elif cond_drop_prob == 1.:
                gvp_output = torch.zeros(batch_size, length, self.node_h_dim[0], device=batch.z_t.device)
            else:
                raise ValueError(f'Invalid cond_drop_prob: {cond_drop_prob}')

        trans_x = torch.cat([gvp_output, self.seq_res(init_seq.reshape(batch_size, length, -1))], dim=-1)
        trans_x = self.mix_lin(trans_x)

        if self.time_cond:
            noise_level = kwargs.get('noise_level')
            time_cond = self.to_time_hiddens(noise_level)  # [B, d_s]
            time_cond = time_cond.unsqueeze(1).repeat(1, length, 1)  # [B, length, d_s]
        else:
            time_cond = None

        # add position embedding
        seq_mask = torch.ones((batch_size, length), device=batch.z_t.device)
        pos_emb = self.embed_positions(seq_mask)

        trans_x = trans_x + pos_emb
        trans_x = trans_x.transpose(0, 1)

        # transformer layers
        for layer in self.trans_layers:
            trans_x = layer(trans_x, None, cond=time_cond.transpose(0, 1))

        logits = self.MLP_out(trans_x.transpose(0, 1))
        # logits = logits.reshape(batch_size, -1, self.out_dim)
        return logits