import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

from math_tan.math_extractor import MathExtractor
from math_tan.math_document import MathDocument
from util_for_data import GraphDict


operator = True
delta = 5
allowable_features = torch.load('w2c/words_mc=' + str(delta) + '.pt')
tail = str(len(allowable_features))


def create_formula_string(filePathForresults, filePath, dirId, missing_tags=None, problem_files=None):
    parts = filePath.split('/')
    file_name = os.path.splitext(parts[len(parts)-1])[0]
    (ext, content) = MathDocument.read_doc_file(filePath)
    formulas = MathExtractor.parse_from_xml(content, 1, operator=operator, missing_tags=missing_tags,
                                                problem_files=problem_files)
    for key in formulas:
        f = open(filePathForresults+"/"+str(dirId)+"/"+file_name+"_"+str(key)+".txt", "w+", encoding='utf-8')
        try:
            f.write(formulas[key].tostring())
        except:
            print(filePath)
        f.close()


def wiki_to_string():
    root = '../NTCIR12_MathIR_WikiCorpus_v2.1.0/MathTagArticles/wpmath00000'
    if operator:
        filePathForresults = '../FormulaString/OPT'
    else:
        filePathForresults = '../FormulaString/SLT'
    for j in range(16, 17):
        tempAddress = root
        if j < 10:
            tempAddress = tempAddress + '0' + str(j) + '/Articles'
        else:
            tempAddress = tempAddress + str(j) + '/Articles'
        for filename in os.listdir(tempAddress):
            filePath = tempAddress + '/' + filename
            try:
                create_formula_string(filePathForresults, filePath, j)
            except:
                print(filePath)


def string_to_graph_dict():
    rootPath = '../FormulaString/OPT'
    resultPath = '../FormulaGraph/OPT'
    for k in range(10, 11):
        tempAddress = rootPath + '/' + str(k)
        for file in os.listdir(tempAddress):
            filePath = tempAddress + '/' + file
            with open(filePath, 'r', encoding='utf-8') as f:
                tree_string = f.read()
                f.close()
            try:
                g = GraphDict(tree_string)
                graph_dict = g.create_graph_dict_from_string()
                node_dict = g.node_dict
                unified_node_dict = g.unified_node_dict
                tree_dict = g.number_tree_dict
            except:
                print(filePath)
            with open(resultPath + '/' + str(k) + '/' + file, 'w+', encoding = 'utf-8') as f:
                try:
                    f.write(str(node_dict) + '\r\n' + str(unified_node_dict) + '\r\n'
                            + str(tree_dict) + '\r\n' + str(graph_dict))
                except:
                    print(filePath)
                f.close()


def graph_dict_to_graph_data_obj(graph_dict, node_dict):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified node and edge features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    node_features_list = []
    edges_list = []
    edge_features_list = []
    node_list = list(graph_dict.keys())
    for k, v in graph_dict.items():
        if node_dict[k] in allowable_features:
            node_feature = [allowable_features.index(node_dict[k])]
        else:
            node_feature = [len(allowable_features)]
        node_features_list.append(node_feature)
        if len(v) > 0:
            i = node_list.index(k)
            for v_i in v:
                j = node_list.index(v_i)
                edges_list.append((i, j))
                if node_dict[k][0] == 'O':
                    edge_feature = [v.index(v_i)]
                else:
                    edge_feature = [0]
                # if node_dict[k][0] == 'U':
                #     edge_feature = [0]
                # else:
                #     edge_feature = [v.index(v_i)]
                edge_features_list.append(edge_feature)
    x = torch.tensor(np.array(node_features_list), dtype=torch.long)

    if len(graph_dict) > 1:
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def graph_data_obj_to_nx_simple(data):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified node and edge features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # nodes
    node_features = data.x.cpu().numpy()
    num_nodes = node_features.shape[0]
    for i in range(num_nodes):
        node_feature_idx = node_features[i]
        G.add_node(i, node_feature_idx=node_feature_idx)
        pass

    # edges
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_edges = edge_index.shape[1]
    for j in range(0, num_edges):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        edge_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, edge_idx=edge_idx)

    return G


def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified node and edge
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # nodes
    node_features_list = []
    for _, node in G.nodes(data=True):
        node_feature = node['node_feature_idx']
        node_features_list.append(node_feature)
    x = torch.tensor(np.array(node_features_list), dtype=torch.long)

    # edges
    if len(G.edges()) > 0:
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = edge['edge_idx']
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def create_dataset():
    rootPath = '../FormulaGraph/OPT'
    full_data_list = []
    for k in range(1, 17):
        data_list = []
        print('{0} is being processed...'.format(k))
        tempAddress = rootPath + '/' + str(k)
        for file in os.listdir(tempAddress):
            filePath = tempAddress + '/' + file
            with open(filePath, 'r', encoding='utf-8') as f:
                content = f.readlines()
                node_dict = eval(content[0])
                graph_dict = eval(content[-1])
                f.close()
            try:
                data = graph_dict_to_graph_data_obj(graph_dict, node_dict)
                data_list.append(data)
            except:
                print(filePath)
        torch.save(data_list, 'dataset_' + tail + '/' + str(k) + '.pt')
        full_data_list = full_data_list + data_list
    print(len(full_data_list), ' graphs in total')
    torch.save(full_data_list, 'dataset_' + tail + '/' + 'full.pt')


def main():
    wiki_to_string()
    string_to_graph_dict()
    create_dataset()


if __name__ == '__main__':
    main()
