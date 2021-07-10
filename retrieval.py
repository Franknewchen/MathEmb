import torch
import os

top_k = 1000
tail = 'gcn_5000_w2c'
query_list = [['Prime_zeta_function_28.txt',9],
              ['Codex_Sinaiticus_1.txt',1],
              ['Word_lists_by_frequency_1.txt',14],
              ['On_Physical_Lines_of_Force_0.txt',14],
              ["Backhouse's_constant_0.txt",12],
              ['Unbinilium_1.txt',1],
              ['Extension_of_a_topological_group_38.txt',16],
              ['Nowcast_(Air_Quality_Index)_1.txt',16],
              ['Generator_(circuit_theory)_0.txt',16],
              ['Lerch_zeta_function_0.txt',5],
              ['Monic_polynomial_1.txt',2],
              ['Hunt–McIlroy_algorithm_2.txt',13],
              ['Mathematical_morphology_24.txt',2],
              ['Hyperbolic_law_of_cosines_2.txt',12],
              ['Strong_antichain_0.txt',5],
              ['Delay_spread_2.txt',7],
              ['Goldbach–Euler_theorem_7.txt',11],
              ['Statistical_coupling_analysis_1.txt',12],
              ['Anisotropic_Network_Model_4.txt',13],
              ['Correlation_and_dependence_1.txt',2]]

rootPath = '../FormulaGraphLinux/OPT'
file_path_list = []
for i in range(1, 17):
    tempAddress = rootPath + '/' + str(i)
    for file in os.listdir(tempAddress):
        file_path_list.append(tempAddress + '/' + file)


def create_rusult_txt():
    t = open('result/result_'+tail+'.txt', 'w+', encoding='utf-8')
    full_graph_rep_list = torch.load('graph_rep/graph_rep_' + tail + '/' + 'full.pt')
    full_graph_rep_tensor = full_graph_rep_list[0]
    for tensor in full_graph_rep_list[1:]:
        full_graph_rep_tensor = torch.cat([full_graph_rep_tensor, tensor], dim=0)

    for i in range(0, 20):
        print('********query'+str(i+1)+'********')
        query_name = query_list[i][0]
        k = query_list[i][1]
        tempAddress = rootPath + '/' + str(k)
        tempAddress_list = os.listdir(tempAddress)
        query_index = tempAddress_list.index(query_name)
        graph_rep_list = torch.load('graph_rep/graph_rep_' + tail + '/' + str(k) + '.pt')
        query_rep = graph_rep_list[query_index // 10000][query_index % 10000, :]
        cos_tensor = torch.cosine_similarity(query_rep, full_graph_rep_tensor, dim=-1)
        values, indices = torch.topk(cos_tensor, top_k, largest=True, sorted=True)

        for j in range(top_k):
            file = file_path_list[indices[j]]
            file_split = file.split('/')
            name = file_split[-1][0:-4]
            rfind = name.rfind('_')
            id = name[0:rfind] + ':' + name[rfind + 1:]
            temp = 'NTCIR12-MathWiki-' + str(i+1) + ' xxx ' + id + ' ' + str(j+1) + ' ' + str(values[j].item()) + ' ' + tail
            t.write(temp + '\n')
    t.close()


def main():
    create_rusult_txt()


if __name__ == "__main__":
    main()


