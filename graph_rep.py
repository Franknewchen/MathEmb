import argparse
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from model import GNN
import os


def pool_func(x, batch, mode="max"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)


def main():
    # settings
    parser = argparse.ArgumentParser(description='Generating graph representations')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--num_node_label', type=int, default=5000, help='number of node type or node feature')
    parser.add_argument('--input_model_file', type=str, default='gcn_5000_w2c.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--initial_node', type=str, default='w2c', help='w2c or emb')
    parser.add_argument('--dataset', type=str, default='dataset/dataset_5000', help='dataset to be choosed')
    parser.add_argument('--tail', type=str, default='gcn_5000_w2c', help='filename tail')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type, num_node_type=args.num_node_label, word2vec=args.initial_node, device=device).to(device)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    model.to(device)

    if not os.path.exists('graph_rep/graph_rep_' + args.tail):
        os.makedirs('graph_rep/graph_rep_' + args.tail)

    full_graph_rep_list = []

    for k in range(1, 17):

        print(args.dataset + '/' + str(k) + ' is being loaded...')
        dataset = torch.load(args.dataset + '/' + str(k) + '.pt')
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print('graph representations is being generated...')
        graph_rep_list = []
        for _, batch in enumerate(test_loader):
            model.eval()
            batch = batch.to(device)
            with torch.no_grad():
                node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
            graph_rep = pool_func(node_rep, batch.batch, mode="mean")
            graph_rep_list.append(graph_rep)
        full_graph_rep_list = full_graph_rep_list + graph_rep_list
        torch.save(graph_rep_list, 'graph_rep/graph_rep_' + args.tail + '/' + str(k) + '.pt')
    torch.save(full_graph_rep_list, 'graph_rep/graph_rep_' + args.tail + '/' + 'full.pt')


if __name__ == "__main__":
    main()
