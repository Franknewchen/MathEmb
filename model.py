import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from torch_scatter import scatter_add


num_atom_type = 24569
num_bond_type = 62


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding = torch.nn.Embedding(num_bond_type, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)       

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 1)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding(edge_attr[:, 0])
        
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr="add")

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(num_bond_type, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 1)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding(edge_attr[:, 0])
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)
    

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GraphSAGEConv, self).__init__(aggr="mean")

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(num_bond_type, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 1)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = self.edge_embedding(edge_attr[:,0])
        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gcn", num_node_type=None, word2vec=None, device=None):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if word2vec == 'w2c':
            if num_node_type == 2157:
                vectors = torch.load('w2c/vectors_mc=15.pt').to(device)
            elif num_node_type == 5000:
                vectors = torch.load('w2c/vectors_mc=5.pt').to(device)
            else:
                vectors = torch.load('w2c/vectors_mc=0.pt').to(device)
            dim_r = torch.randn(1, emb_dim).to(device)
            dim_r[:, -1] = 0
            self.x_embedding = torch.cat([vectors[0:num_node_type, :], dim_r], dim=0)
            self.x_embedding = torch.cat([self.x_embedding, vectors[-1].reshape(1, emb_dim)], dim=0)
        else:
            self.x_embedding = torch.nn.Embedding(num_node_type + 2, emb_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                 self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "graphsage":
                 self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        try:
            x = self.x_embedding[x[:, 0]]
        except:
            x = self.x_embedding(x[:, 0])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                ### remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation
