from collections import Counter
from math_tan.semantic_symbol import SemanticSymbol


class TreeDict:
    def __init__(self, tree_string):
        opt = self.create_opt_from_string(tree_string)
        self.tree_dict = {}
        self.key_list = []
        self.child_list =[self.find_label(opt.tostring())]

    @staticmethod
    def find_key(any_dict, value):
        return [k for k, v in any_dict.items() if v == value]

    @staticmethod
    def find_label(tree_substring):
        pos = 1
        while not tree_substring[pos] in ["[", "]"]:
            if tree_substring[pos] == "," and pos > 1:
                break
            pos += 1
        label = tree_substring[1:pos]
        return label

    @staticmethod
    def find_matching_bracket(tree_substring, offset):
        pos = offset
        if tree_substring[pos] == "[":
            count = 1
            pos += 1
        else:
            count = 0
        while count > 0:
            if tree_substring[pos] == "[":
                count += 1
            if tree_substring[pos] == "]":
                count -= 1
            pos += 1
        return pos

    def create_node_list_from_string(self, tree_substring):
        pos = 1
        while not tree_substring[pos] in ["[", "]"]:
            if tree_substring[pos] == "," and pos > 1:
                break
            pos += 1
        node_list = tree_substring[1:pos].split(',')
        while tree_substring[pos] != "]":
            if tree_substring[pos] == ",":
                child_end = self.find_matching_bracket(tree_substring, pos + 2)
                child_text = tree_substring[(pos + 2):child_end]
                pos = child_end
                child_node = self.create_node_list_from_string(child_text)
                node_list = node_list + child_node
        return node_list

    @staticmethod
    def create_unified_node_list(node_list):
        n_dict = {}
        v_dict = {}
        i = 0
        j = 0
        for l in range(len(node_list)):
            node = node_list[l]
            if node[0] == 'N':
                if node not in n_dict:
                    i += 1
                    n_dict[node] = 'N!' + '_' + str(i)
                node_list[l] = n_dict[node]
            elif node[0] == 'V' and 'normal' not in node:
                if node not in v_dict:
                    j += 1
                    v_dict[node] = 'V!' + '_' + str(j)
                node_list[l] = v_dict[node]
            elif node[0] == 'T':
                node_list[l] = 'T!'
        return node_list

    @staticmethod
    def create_node_dict(node_list):
        node_dict = {i: node_list[i] for i in range(len(node_list))}
        return node_dict

    def create_label_dict(self, node_dict):
        label_dict = {}
        for label in set(node_dict.values()):
            label_dict[label] = self.find_key(node_dict, label)
        return label_dict

    def create_opt_from_string(self, tree_substring):
        pos = 1
        while not tree_substring[pos] in ["[", "]"]:
            if tree_substring[pos] == "," and pos > 1:
                break
            pos += 1
        label = tree_substring[1:pos]
        root = SemanticSymbol(label)

        children = []
        while tree_substring[pos] != "]":
            if tree_substring[pos] == ",":
                # get child ...
                child_end = self.find_matching_bracket(tree_substring, pos + 2)
                child_text = tree_substring[(pos + 2):child_end]
                pos = child_end
                child_node = self.create_opt_from_string(child_text)  # 递归
                child_node.parent = root
                children.append(child_node)
        root.children = children
        if tree_substring != root.tostring():
            print("Mismatch: " + tree_substring + " -> " + root.tostring(), flush=True)
            exit(1)
        return root

    def create_label_tree_dict_from_opt(self, opt):
        if not opt.is_leaf():
            root = self.find_label(opt.tostring())
            if root in self.key_list:
                root = root + '@' + str(self.key_list.count(root))
            self.key_list.append(self.find_label(opt.tostring()))
            self.tree_dict[root] = []
            for child in opt.children:
                child_label = self.find_label(child.tostring())
                if child_label in self.child_list:
                    child_label = child_label + '@' + str(self.child_list.count(child_label))
                self.child_list.append(self.find_label(child.tostring()))
                self.tree_dict[root].append(child_label)
                self.create_label_tree_dict_from_opt(child)
        else:
            self.key_list.append(self.find_label(opt.tostring()))
        return self.tree_dict

    @staticmethod
    def create_number_tree_dict(label_dict, tree_dict):
        number_tree_dict = {}
        for k, v in tree_dict.items():
            if '@' in k:
                pos = k.find('@')
                k_id = label_dict[k[0:pos]][int(k[pos + 1:])]
            else:
                k_id = label_dict[k][0]
            number_tree_dict[k_id] = []
            for i in v:
                if '@' in i:
                    pos = i.find('@')
                    i_id = label_dict[i[0:pos]][int(i[pos + 1:])]
                else:
                    i_id = label_dict[i][0]
                number_tree_dict[k_id].append(i_id)
        return number_tree_dict


class GraphDict:
    def __init__(self, tree_string):
        self.tree_string = tree_string
        self.node_list = None
        self.unified_node_list = None
        self.node_dict = None
        self.unified_node_dict = None
        self.opt = None
        self.label_dict = None
        self.label_tree_dict = None
        self.number_tree_dict = None
        self.graph_dict = None

    @staticmethod
    def find_key(any_dict, value):
        return [k for k, v in any_dict.items() if v == value]

    @staticmethod
    def find_father(tree_dict, child):
        return [k for k, v in tree_dict.items() if child in v]

    def convert_tree_to_graph(self, node_dict, tree_dict):
        # 以空列表的形式将叶子节点以及只有一个点(即空字典)的情况加入
        for k in node_dict.keys():
            if k not in tree_dict.keys():
                tree_dict[k] = []

        graph_dict = dict(tree_dict)
        df_dict = dict(tree_dict)

        for k, v in sorted(tree_dict.items(), reverse=True):
            for i in range(len(v)):
                if v[i] in tree_dict:
                    df_dict[k] = sorted(df_dict[k] + df_dict[v[i]])
                else:
                    df_dict[v[i]] = []

        len_dict = {}
        for k, v in df_dict.items():
            len_dict[k] = len(v)

        repeat_length_dict = {}
        count = dict(Counter(list(len_dict.values())))
        repeat_length = [key for key, value in count.items() if value > 1]
        if repeat_length:
            for l in repeat_length:
                repeat_length_dict[l] = sorted(self.find_key(len_dict, l))
        repeat_length_dict = dict(sorted(repeat_length_dict.items(), reverse=True))

        if repeat_length_dict:
            for k, v in repeat_length_dict.items():
                for i in range(len(v)):
                    for j in range(i + 1, len(v)):

                        if v[i] in graph_dict and v[j] in graph_dict:
                            if node_dict[v[i]] == node_dict[v[j]]:

                                if k != 0:  # 非叶子节点的情况
                                    i_father = self.find_father(graph_dict, v[i])
                                    j_father = self.find_father(graph_dict, v[j])
                                    if set(i_father).intersection(set(j_father)) == set():  # 如果两个点的父节点的交集为空
                                        labels_i = []
                                        labels_j = []
                                        for l in range(k):
                                            labels_i.append(node_dict[df_dict[v[i]][l]])
                                            labels_j.append(node_dict[df_dict[v[j]][l]])
                                        if labels_i == labels_j:
                                            for f in i_father:
                                                graph_dict[f] = [v[j] if x == v[i] else x for x in graph_dict[f]]
                                            del graph_dict[v[i]]
                                            for c in df_dict[v[i]]:
                                                try:
                                                    del graph_dict[c]
                                                except:
                                                    continue
                                else:  # 叶子节点的情况
                                    i_father = self.find_father(graph_dict, v[i])
                                    j_father = self.find_father(graph_dict, v[j])
                                    if set(i_father).intersection(set(j_father)) == set():
                                        del graph_dict[v[i]]
                                        try:
                                            for f in i_father:
                                                graph_dict[f] = [v[j] if x == v[i] else x for x in graph_dict[f]]
                                        except:
                                            continue
        return graph_dict

    def create_graph_dict_from_string(self):
        T = TreeDict(self.tree_string)
        self.node_list = T.create_node_list_from_string(self.tree_string)
        self.node_dict = T.create_node_dict(self.node_list)
        self.label_dict = T.create_label_dict(self.node_dict)
        self.unified_node_list = T.create_unified_node_list(self.node_list)
        self.unified_node_dict = T.create_node_dict(self.unified_node_list)
        self.opt = T.create_opt_from_string(self.tree_string)
        self.label_tree_dict = T.create_label_tree_dict_from_opt(self.opt)
        self.number_tree_dict = T.create_number_tree_dict(self.label_dict, self.label_tree_dict)
        self.graph_dict = self.convert_tree_to_graph(self.unified_node_dict, self.number_tree_dict)
        return self.graph_dict