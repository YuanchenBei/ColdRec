import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import os
import sys
import time
sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="citeulike", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="./", help='Director of the dataset.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--cold_object', type=str, default='item', help='item or user')
args = parser.parse_args()
pprint(vars(args))


class Timer(object):
    def __init__(self, name=''):
        self._name = name
        self.begin_time = time.time()
        self.last_time = time.time()
        self.current_time = time.time()
        self.stage_time = 0.0
        self.run_time = 0.0

    def logging(self, message):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.update()
        message = '' if message is None else message
        print("{} {} {:.0f}s {:.0f}s | {}".format(current_time, self._name, self.run_time, self.stage_time, message))

    def update(self):
        self.current_time = time.time()
        self.stage_time = self.current_time - self.last_time
        self.last_time = self.current_time
        self.run_time = self.current_time - self.begin_time
        return self


random.seed(args.seed)
np.random.seed(args.seed)

store_path = os.path.join(args.datadir, f"{args.dataset}/cold_{args.cold_object}/")
procedure_timer = Timer("Convert")


"""Read data from file"""
df_emb = pd.read_csv(os.path.join(store_path, 'warm_train.csv'))
df_cold = pd.read_csv(os.path.join(store_path, f'cold_{args.cold_object}.csv'))
df_warm_val = pd.read_csv(os.path.join(store_path, 'warm_val.csv'))
df_warm_test = pd.read_csv(os.path.join(store_path, 'warm_test.csv'))
df_cold_val = pd.read_csv(os.path.join(store_path, f'cold_{args.cold_object}_val.csv'))
df_cold_test = pd.read_csv(os.path.join(store_path, f'cold_{args.cold_object}_test.csv'))
print(df_emb.shape, df_warm_val.shape, df_warm_test.shape, df_cold_val.shape, df_cold_test.shape)

"""Build overall validation/test set"""
warm_object = 'user' if args.cold_object == 'item' else 'item'
overall_val_user_set = np.array(list(set(df_cold_val[warm_object]) & set(df_warm_val[warm_object])), dtype=np.int32)
df_overall_val = pd.concat([df_cold_val, df_warm_val])
print(df_overall_val.shape)
df_overall_val = df_overall_val[df_overall_val[warm_object].isin(overall_val_user_set)]
print(df_overall_val.shape)

overall_test_user_set = np.array(list(set(df_cold_test[warm_object]) & set(df_warm_test[warm_object])), dtype=np.int32)
df_overall_test = pd.concat([df_cold_test, df_warm_test])
print(df_overall_test.shape)
df_overall_test = df_overall_test[df_overall_test[warm_object].isin(overall_test_user_set)]
print(df_overall_test.shape)

df_overall_val.to_csv(os.path.join(store_path, 'overall_val.csv'), index=False)
df_overall_test.to_csv(os.path.join(store_path, 'overall_test.csv'), index=False)

n_user_item = pickle.load(open(os.path.join(store_path, 'n_user_item.pkl'), 'rb'))
user_num = n_user_item['user']
item_num = n_user_item['item']
procedure_timer.logging('Finish loading data.')
print("Global user_num: {}  item_num: {}".format(user_num, item_num))

"""Get testing users"""
emb_user = np.array(list(set(df_emb['user'])), dtype=np.int32)
warm_val_user = np.array(list(set(df_warm_val['user'])), dtype=np.int32)
warm_test_user = np.array(list(set(df_warm_test['user'])), dtype=np.int32)
cold_val_user = np.array(list(set(df_cold_val['user'])), dtype=np.int32)
cold_test_user = np.array(list(set(df_cold_test['user'])), dtype=np.int32)
overall_val_user = np.array(list(overall_val_user_set), dtype=np.int32)
overall_test_user = np.array(list(overall_test_user_set), dtype=np.int32)
procedure_timer.logging('Finish getting testing users.')

"""Get testing items"""
emb_item = np.array(list(set(df_emb['item'])), dtype=np.int32)
warm_val_item = np.array(list(set(df_warm_val['item'])), dtype=np.int32)
warm_test_item = np.array(list(set(df_warm_test['item'])), dtype=np.int32)
cold_val_item = np.array(list(set(df_cold_val['item'])), dtype=np.int32)
cold_test_item = np.array(list(set(df_cold_test['item'])), dtype=np.int32)

overall_val_item = np.array(list(set(df_overall_val['item'])), dtype=np.int32)
overall_test_item = np.array(list(set(df_overall_test['item'])), dtype=np.int32)
procedure_timer.logging('Finish getting testing items.')

"""Statistics"""
user_array = np.arange(user_num, dtype=np.int32)
item_array = np.arange(item_num, dtype=np.int32)
warm_user = np.array(list(set(df_emb['user'].tolist())), dtype=np.int32)
warm_item = np.array(list(set(df_emb['item'].tolist())), dtype=np.int32)
cold_user = np.array(list(set(df_cold['user'].tolist())), dtype=np.int32)
cold_item = np.array(list(set(df_cold['item'].tolist())), dtype=np.int32)

print("[warm] user: {}  item: {}".format(len(warm_user), len(warm_item)))
print("[cold] user: {}  item: {}".format(len(cold_user), len(cold_item)))

"""Save results"""
para_dict = {}
para_dict['user_num'] = user_num
para_dict['item_num'] = item_num
para_dict['user_array'] = user_array
para_dict['item_array'] = item_array
para_dict['warm_user'] = warm_user
para_dict['warm_item'] = warm_item
para_dict['cold_user'] = cold_user
para_dict['cold_item'] = cold_item

para_dict['train_user'] = emb_user
para_dict['warm_val_user'] = warm_val_user
para_dict['warm_test_user'] = warm_test_user
para_dict['cold_val_user'] = cold_val_user
para_dict['cold_test_user'] = cold_test_user
para_dict['overall_val_user'] = overall_val_user
para_dict['overall_test_user'] = overall_test_user

para_dict['train_item'] = emb_item
para_dict['warm_val_item'] = warm_val_item
para_dict['warm_test_item'] = warm_test_item
para_dict['cold_val_item'] = cold_val_item
para_dict['cold_test_item'] = cold_test_item
para_dict['overall_val_item'] = overall_val_item
para_dict['overall_test_item'] = overall_test_item

dict_path = os.path.join(store_path, 'info_dict.pkl')
pickle.dump(para_dict, open(dict_path, 'wb'), protocol=4)
procedure_timer.logging('Convert {} successfully, store the dict to {}'.format(args.dataset, dict_path))
