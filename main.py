import argparse
import torch
import numpy as np
import pickle
from util.loader import DataLoader
from util.utils import set_seed
from config.model_param import model_specific_param
from model.MF import MF
from model.LightGCN import LightGCN
from model.KNN import KNN
from model.DUIF import DUIF
from model.DeepMusic import DeepMusic
from model.MTPR import MTPR
from model.VBPR import VBPR
from model.AMR import AMR
from model.GAR import GAR
from model.ALDI import ALDI
from model.CLCRec import CLCRec
from model.LARA import LARA
from model.CCFCRec import CCFCRec
from model.DropoutNet import DropoutNet
from model.Heater import Heater
from model.MetaEmbedding import MetaEmbedding
from model.GoRec import GoRec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='citeulike')
    parser.add_argument('--model', default='MF')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--topN', default='10,20')
    parser.add_argument('--bs', type=int, default=2048, help='training batch size')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', type=float, default=0.0001)
    parser.add_argument('--runs', type=int, default=1, help='model runs')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--use_gpu', default=True, help='Whether to use CUDA')
    parser.add_argument('--save_emb', default=True, help='Whether to save the user/item embeddings')
    parser.add_argument('--gpu_id', type=int, default=0, help='CUDA id')
    parser.add_argument('--cold_object', default='item', type=str, choices=['user', 'item'])
    parser.add_argument('--backbone', default='MF')
    args, _ = parser.parse_known_args()
    parser = model_specific_param(args.model, parser)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:%d" % (args.gpu_id) if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    # data loader
    training_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_train.csv')
    # following the widely used setting in previous works, the 'all' set is used for validation.
    all_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/overall_val.csv')
    warm_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_val.csv')
    cold_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_val.csv')
    all_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/overall_test.csv')
    warm_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_test.csv')
    cold_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_test.csv')

    # dataset information
    data_info_dict = pickle.load(open(f'./data/{args.dataset}/cold_{args.cold_object}/info_dict.pkl', 'rb'))
    user_num = data_info_dict['user_num']
    item_num = data_info_dict['item_num']
    warm_user_idx = data_info_dict['warm_user']
    warm_item_idx = data_info_dict['warm_item']
    cold_user_idx = data_info_dict['cold_user']
    cold_item_idx = data_info_dict['cold_item']
    print(f"Dataset: {args.dataset}, User num: {user_num}, Item num: {item_num}.")

    # content obtaining
    user_content, item_content = None, None
    if args.cold_object == 'user':
        user_content = np.load(f'./data/{args.dataset}/{args.dataset}_{args.cold_object}_content.npy')
        print(f'user content shape: {user_content.shape}')
    if args.cold_object == 'item':
        item_content = np.load(f'./data/{args.dataset}/{args.dataset}_{args.cold_object}_content.npy')
        print(f'item content shape: {item_content.shape}')

    top_Ns = args.topN.split(',')
    all_hit_results = [[] for _ in top_Ns]
    all_precision_results = [[] for _ in top_Ns]
    all_recall_results = [[] for _ in top_Ns]
    all_ndcg_results = [[] for _ in top_Ns]
    cold_hit_results = [[] for _ in top_Ns]
    cold_precision_results = [[] for _ in top_Ns]
    cold_recall_results = [[] for _ in top_Ns]
    cold_ndcg_results = [[] for _ in top_Ns]
    warm_hit_results = [[] for _ in top_Ns]
    warm_precision_results = [[] for _ in top_Ns]
    warm_recall_results = [[] for _ in top_Ns]
    warm_ndcg_results = [[] for _ in top_Ns]

    for round in range(args.runs):
        print(f"Start round {round} running!")
        if args.runs == 1:
            set_seed(args.seed, args.use_gpu)
        else:
            set_seed(round, args.use_gpu)
        # model register
        available_models = ['MF', 'LightGCN', 'KNN', 'DUIF', 'DeepMusic', 'MTPR', 'VBPR', 'AMR', 'GAR', 'ALDI',
                            'CLCRec', 'LARA', 'CCFCRec', 'DropoutNet', 'Heater', 'MetaEmbedding', 'GoRec']
        if args.model in available_models:
            if args.model == 'MF' or args.model == 'LightGCN':
                # recommender backbone training
                model = eval(args.model)(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                         warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                         warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device)
            else:
                # cold-start model training
                model = eval(args.model)(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                         warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                         warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                         user_content=user_content, item_content=item_content)
            print(f"Registered model: {args.model}.")
        else:
            raise ValueError(f"Invalid model name: {args.model}!")

        # model running
        model.run()

        # results recording
        for i in range(len(top_Ns)):
            all_hit_results[i].append(model.overall_test_results[i][0])
            all_precision_results[i].append(model.overall_test_results[i][1])
            all_recall_results[i].append(model.overall_test_results[i][2])
            all_ndcg_results[i].append(model.overall_test_results[i][3])

            cold_hit_results[i].append(model.cold_test_results[i][0])
            cold_precision_results[i].append(model.cold_test_results[i][1])
            cold_recall_results[i].append(model.cold_test_results[i][2])
            cold_ndcg_results[i].append(model.cold_test_results[i][3])

            warm_hit_results[i].append(model.warm_test_results[i][0])
            warm_precision_results[i].append(model.warm_test_results[i][1])
            warm_recall_results[i].append(model.warm_test_results[i][2])
            warm_ndcg_results[i].append(model.warm_test_results[i][3])

    for i in range(len(top_Ns)):
        print("*" * 80)
        print(f"Top-{top_Ns[i]} Overall Test Performance:")
        mean_all_hit, std_all_hit = np.mean(all_hit_results[i]), np.std(all_hit_results[i])
        mean_all_precision, std_all_precision = np.mean(all_precision_results[i]), np.std(all_precision_results[i])
        mean_all_recall, std_all_recall = np.mean(all_recall_results[i]), np.std(all_recall_results[i])
        mean_all_ndcg, std_all_ndcg = np.mean(all_ndcg_results[i]), np.std(all_ndcg_results[i])
        print(f"Hit@{top_Ns[i]}: {mean_all_hit:.4f}±{std_all_hit:.4f}, "
              f"Precision@{top_Ns[i]}: {mean_all_precision:.4f}±{std_all_precision:.4f}, "
              f"Recall@{top_Ns[i]}: {mean_all_recall:.4f}±{std_all_recall:.4f}, "
              f"NDCG@{top_Ns[i]}: {mean_all_ndcg:.4f}±{std_all_ndcg:.4f}")

        print(f"Top-{top_Ns[i]} Cold-Start Test Performance:")
        mean_cold_hit, std_cold_hit = np.mean(cold_hit_results[i]), np.std(cold_hit_results[i])
        mean_cold_precision, std_cold_precision = np.mean(cold_precision_results[i]), np.std(cold_precision_results[i])
        mean_cold_recall, std_cold_recall = np.mean(cold_recall_results[i]), np.std(cold_recall_results[i])
        mean_cold_ndcg, std_cold_ndcg = np.mean(cold_ndcg_results[i]), np.std(cold_ndcg_results[i])
        print(f"Hit@{top_Ns[i]}: {mean_cold_hit:.4f}±{std_cold_hit:.4f}, "
              f"Precision@{top_Ns[i]}: {mean_cold_precision:.4f}±{std_cold_precision:.4f}, "
              f"Recall@{top_Ns[i]}: {mean_cold_recall:.4f}±{std_cold_recall:.4f}, "
              f"NDCG@{top_Ns[i]}: {mean_cold_ndcg:.4f}±{std_cold_ndcg:.4f}")

        print(f"Top-{top_Ns[i]} Warm-Start Test Performance:")
        mean_warm_hit, std_warm_hit = np.mean(warm_hit_results[i]), np.std(warm_hit_results[i])
        mean_warm_precision, std_warm_precision = np.mean(warm_precision_results[i]), np.std(warm_precision_results[i])
        mean_warm_recall, std_warm_recall = np.mean(warm_recall_results[i]), np.std(warm_recall_results[i])
        mean_warm_ndcg, std_warm_ndcg = np.mean(warm_ndcg_results[i]), np.std(warm_ndcg_results[i])
        print(f"Hit@{top_Ns[i]}: {mean_warm_hit:.4f}±{std_warm_hit:.4f}, "
              f"Precision@{top_Ns[i]}: {mean_warm_precision:.4f}±{std_warm_precision:.4f}, "
              f"Recall@{top_Ns[i]}: {mean_warm_recall:.4f}±{std_warm_recall:.4f}, "
              f"NDCG@{top_Ns[i]}: {mean_warm_ndcg:.4f}±{std_warm_ndcg:.4f}")
