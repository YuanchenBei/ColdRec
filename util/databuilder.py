import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import torch

class ColdStartDataBuilder(object):
    def __init__(self, training_data, warm_valid_data, cold_valid_data, overall_valid_data,
                 warm_test_data, cold_test_data, overall_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx,
                 user_content=None, item_content=None):
        super(ColdStartDataBuilder, self).__init__()
        self.training_data = training_data
        self.warm_valid_data = warm_valid_data
        self.warm_test_data = warm_test_data
        self.cold_valid_data = cold_valid_data
        self.cold_test_data = cold_test_data
        self.overall_valid_data = overall_valid_data
        self.overall_test_data = overall_test_data

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.warm_valid_set = defaultdict(dict)
        self.warm_valid_set_item = set()
        self.cold_valid_set = defaultdict(dict)
        self.cold_valid_set_item = set()
        self.overall_valid_set = defaultdict(dict)
        self.overall_valid_set_item = set()
        self.warm_test_set = defaultdict(dict)
        self.warm_test_set_item = set()
        self.cold_test_set = defaultdict(dict)
        self.cold_test_set_item = set()
        self.overall_test_set = defaultdict(dict)
        self.overall_test_set_item = set()
        self.source_user_content = None
        self.mapped_user_content = None
        self.source_item_content = None
        self.mapped_item_content = None
        if user_content is not None:
            self.source_user_content = user_content
            self.mapped_user_content = np.empty((user_content.shape[0], user_content.shape[1]))
            self.user_content_dim = user_content.shape[-1]
        if item_content is not None:
            self.source_item_content = item_content
            self.mapped_item_content = np.empty((item_content.shape[0], item_content.shape[1]))
            self.item_content_dim = item_content.shape[-1]

        self.generate_set()

        self.user_num = user_num
        self.item_num = item_num
        #print(self.item_num, len(self.item.keys()))
        #raise Exception("debugging...")
        # PLEASE NOTE: the original and mapped index are different!
        self.source_warm_user_idx = warm_user_idx
        self.source_warm_item_idx = warm_item_idx
        self.source_cold_user_idx = cold_user_idx
        self.source_cold_item_idx = cold_item_idx
        self.mapped_warm_user_idx = self.get_user_id_list(self.source_warm_user_idx)
        self.mapped_warm_item_idx = self.get_item_id_list(self.source_warm_item_idx)
        self.mapped_cold_user_idx = self.get_user_id_list(self.source_cold_user_idx)
        self.mapped_cold_item_idx = self.get_item_id_list(self.source_cold_item_idx)
        #print(self.source_cold_item_idx, self.mapped_cold_item_idx)
        #raise Exception("debugging...")
        self.ui_adj = self.create_sparse_complete_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.create_sparse_interaction_matrix()

    def generate_set(self):
        # training set building
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        # warm validation set building
        for entry in self.warm_valid_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (warm validation set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (warm validation set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.warm_valid_set[user][item] = rating
            self.warm_valid_set_item.add(item)

        # warm testing set building
        for entry in self.warm_test_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (warm test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (warm test set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.warm_test_set[user][item] = rating
            self.warm_test_set_item.add(item)

        for entry in self.cold_valid_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (cold validation set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (cold valid set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.cold_valid_set[user][item] = rating
            self.cold_valid_set_item.add(item)

        for entry in self.cold_test_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (cold test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (cold test set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.cold_test_set[user][item] = rating
            self.cold_test_set_item.add(item)

        for entry in self.overall_valid_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (overall valid set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.overall_valid_set[user][item] = rating
            self.overall_valid_set_item.add(item)

        for entry in self.overall_test_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (overall test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.overall_test_set[user][item] = rating
            self.overall_test_set_item.add(item)

        #raise Exception("now debugging...")

    def create_sparse_complete_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (|u| + |i|, |u| + |i|)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        '''
        :param adj_mat: the sparse adjacency matrix
        :return: normalized adjacency matrix
        '''
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]
        else:
            raise Exception(f"user {u} not in current id table")

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]
        else:
            raise Exception(f"item {i} not in current id table")

    def get_user_id_list(self, u_list):
        mapped_list = []
        for u in u_list:
            if u in self.user:
                mapped_list.append(self.user[u])
            else:
                raise Exception(f"user {u} not in current id table")
        return np.array(mapped_list)

    def get_item_id_list(self, i_list):
        mapped_list = []
        for i in i_list:
            if i in self.item:
                mapped_list.append(self.item[i])
            else:
                raise Exception(f"item {i} not in current id table")
        return np.array(mapped_list)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def warm_valid_size(self):
        return len(self.warm_valid_set), len(self.warm_valid_set_item), len(self.warm_valid_data)

    def warm_test_size(self):
        return len(self.warm_test_set), len(self.warm_test_set_item), len(self.warm_test_data)

    def cold_valid_size(self):
        return len(self.cold_valid_set), len(self.cold_valid_set_item), len(self.cold_valid_data)

    def cold_test_size(self):
        return len(self.cold_test_set), len(self.cold_test_set_item), len(self.cold_test_data)

    def overall_valid_size(self):
        return len(self.overall_valid_set), len(self.overall_valid_set_item), len(self.overall_valid_data)

    def overall_test_size(self):
        return len(self.overall_test_set), len(self.overall_test_set_item), len(self.overall_test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m


class ColdStartMetaDataBuilder(object):
    def __init__(self, training_data, warm_valid_data, cold_valid_data, overall_valid_data,
                 warm_test_data, cold_test_data, overall_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx,
                 user_content=None, item_content=None):
        #training_set, warm_valid_set, cold_valid_set, overall_valid_set,
        #warm_test_set, cold_test_set, overall_test_set, user_num, item_num,
        #warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx,
        #user_content, item_content
        super(ColdStartMetaDataBuilder, self).__init__()
        self.training_data = training_data
        self.warm_valid_data = warm_valid_data
        self.warm_test_data = warm_test_data
        self.cold_valid_data = cold_valid_data
        self.cold_test_data = cold_test_data
        self.overall_valid_data = overall_valid_data
        self.overall_test_data = overall_test_data

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.warm_valid_set = defaultdict(dict)
        self.warm_valid_set_item = set()
        self.cold_valid_set = defaultdict(dict)
        self.cold_valid_set_item = set()
        self.overall_valid_set = defaultdict(dict)
        self.overall_valid_set_item = set()
        self.warm_test_set = defaultdict(dict)
        self.warm_test_set_item = set()
        self.cold_test_set = defaultdict(dict)
        self.cold_test_set_item = set()
        self.overall_test_set = defaultdict(dict)
        self.overall_test_set_item = set()
        self.source_user_content = None
        self.source_item_content = None
        if user_content is not None:
            self.source_user_content = user_content
            self.mapped_user_content = np.empty((user_content.shape[0], user_content.shape[1]))
            self.user_content_dim = user_content.shape[-1]
        if item_content is not None:
            self.source_item_content = item_content
            self.mapped_item_content = np.empty((item_content.shape[0], item_content.shape[1]))
            self.item_content_dim = item_content.shape[-1]

        self.generate_set()

        self.user_num = user_num
        self.item_num = item_num
        #print(self.item_num, len(self.item.keys()))
        #raise Exception("debugging...")
        # PLEASE NOTE: the original and mapped index are different!
        self.source_warm_user_idx = warm_user_idx
        self.source_warm_item_idx = warm_item_idx
        self.source_cold_user_idx = cold_user_idx
        self.source_cold_item_idx = cold_item_idx
        self.mapped_warm_user_idx = self.get_user_id_list(self.source_warm_user_idx)
        self.mapped_warm_item_idx = self.get_item_id_list(self.source_warm_item_idx)
        self.mapped_cold_user_idx = self.get_user_id_list(self.source_cold_user_idx)
        self.mapped_cold_item_idx = self.get_item_id_list(self.source_cold_item_idx)
        #print(self.source_cold_item_idx, self.mapped_cold_item_idx)
        #raise Exception("debugging...")
        self.ui_adj = self.create_sparse_complete_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.create_sparse_interaction_matrix()

    def generate_set(self):
        # training set building
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        # warm validation set building
        for entry in self.warm_valid_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (warm validation set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (warm validation set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.warm_valid_set[user][item] = rating
            self.warm_valid_set_item.add(item)

        # warm testing set building
        for entry in self.warm_test_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (warm test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (warm test set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.warm_test_set[user][item] = rating
            self.warm_test_set_item.add(item)

        for entry in self.cold_valid_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (cold validation set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (cold valid set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.cold_valid_set[user][item] = rating
            self.cold_valid_set_item.add(item)

        for entry in self.cold_test_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (cold test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                #print(f"item {item} not in current id table (cold test set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.cold_test_set[user][item] = rating
            self.cold_test_set_item.add(item)

        for entry in self.overall_valid_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (overall valid set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.overall_valid_set[user][item] = rating
            self.overall_valid_set_item.add(item)

        for entry in self.overall_test_data:
            user, item, rating = entry
            if user not in self.user:
                #print(f"user {user} not in current id table (overall test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = self.source_user_content[user]
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = self.source_item_content[item]
            self.overall_test_set[user][item] = rating
            self.overall_test_set_item.add(item)

        #raise Exception("now debugging...")

    def create_sparse_complete_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (|u| + |i|, |u| + |i|)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        '''
        :param adj_mat: the sparse adjacency matrix
        :return: normalized adjacency matrix
        '''
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]
        else:
            raise Exception(f"user {u} not in current id table")

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]
        else:
            raise Exception(f"item {i} not in current id table")

    def get_user_id_list(self, u_list):
        mapped_list = []
        for u in u_list:
            if u in self.user:
                mapped_list.append(self.user[u])
            else:
                raise Exception(f"user {u} not in current id table")
        return np.array(mapped_list)

    def get_item_id_list(self, i_list):
        mapped_list = []
        for i in i_list:
            if i in self.item:
                mapped_list.append(self.item[i])
            else:
                raise Exception(f"item {i} not in current id table")
        return np.array(mapped_list)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def warm_valid_size(self):
        return len(self.warm_valid_set), len(self.warm_valid_set_item), len(self.warm_valid_data)

    def warm_test_size(self):
        return len(self.warm_test_set), len(self.warm_test_set_item), len(self.warm_test_data)

    def cold_valid_size(self):
        return len(self.cold_valid_set), len(self.cold_valid_set_item), len(self.cold_valid_data)

    def cold_test_size(self):
        return len(self.cold_test_set), len(self.cold_test_set_item), len(self.cold_test_data)

    def overall_valid_size(self):
        return len(self.overall_valid_set), len(self.overall_valid_set_item), len(self.overall_valid_data)

    def overall_test_size(self):
        return len(self.overall_test_set), len(self.overall_test_set_item), len(self.overall_test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m


class DataBuilder(object):
    def __init__(self, training_data, valid_data, test_data):
        super(DataBuilder, self).__init__()
        self.training_data = training_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.valid_set = defaultdict(dict)
        self.valid_set_item = set()
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.generate_set()

        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.create_sparse_complete_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.create_sparse_interaction_matrix()

    def generate_set(self):
        # training set building
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        # validation set building
        for entry in self.valid_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.valid_set[user][item] = rating
            self.valid_set_item.add(item)

        # testing set building
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)

    def create_sparse_complete_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (|u| + |i|, |u| + |i|)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        '''
        :param adj_mat: the sparse adjacency matrix
        :return: normalized adjacency matrix
        '''
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def valid_size(self):
        return len(self.valid_set), len(self.valid_set_item), len(self.valid_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
