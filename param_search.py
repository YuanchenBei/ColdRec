import argparse
import pickle

import numpy as np
import optuna
import torch

from config.model_param import model_specific_param
from util.loader import DataLoader


def objective(trial):
    args.lr = trial.suggest_categorical("lr", [1e-4, 5e-3, 1e-3, 5e-2, 1e-2])
    args.reg = trial.suggest_categorical("reg", [1e-5, 5e-4, 1e-4, 5e-3, 1e-3])
    if args.model == 'KNN':
        args.knn_num = trial.suggest_int("knn_num", 3, 12)
    elif args.model == 'SimGCL':
        args.cl_rate = trial.suggest_categorical("cl_rate", [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
        args.tau = trial.suggest_categorical("tau", [0.2, 0.4, 0.6, 0.8, 1.0])
        args.eps = trial.suggest_categorical("eps", [0.1, 0.2, 0.3, 0.4, 0.5])
    elif args.model == 'XSimGCL':
        args.cl_rate = trial.suggest_categorical("cl_rate", [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
        args.tau = trial.suggest_categorical("tau", [0.2, 0.4, 0.6, 0.8, 1.0])
        args.eps = trial.suggest_categorical("eps", [0.1, 0.2, 0.3, 0.4, 0.5])
    elif args.model == 'NCL':
        args.alpha = trial.suggest_categorical("alpha", [0.5, 1.0, 1.5, 2.0])
        args.ssl_reg = trial.suggest_loguniform("ssl_reg", 1e-8, 1e-4)
        args.proto_reg = trial.suggest_loguniform("proto_reg", 1e-8, 1e-4)
        args.tau = trial.suggest_categorical("tau", [0.01, 0.05, 0.1, 0.2, 0.5])
        args.hyper_layers = trial.suggest_int("hyper_layers", 1, 3)
        args.num_clusters = trial.categorical("num_clusters", [20, 50, 80, 100, 200, 500])
    elif args.model == 'ALDI':
        args.alpha = trial.suggest_categorical("alpha", [0.5, 1.0, 1.5, 2.0])
        args.beta = trial.suggest_categorical("beta", [0.05, 0.1, 0.2, 0.5])
        args.gamma = trial.suggest_categorical("gamma", [0.1, 0.2, 0.5, 1.0])
        args.tws = trial.suggest_categorical("tws", [0, 1])
        args.freq_coef_M = trial.suggest_categorical("freq_coef_M", [2.0, 4.0, 6.0, 8.0])
    elif args.model == 'GAR':
        args.alpha = trial.suggest_categorical("alpha", [0.05, 0.1, 0.2, 0.5])
        args.beta = trial.suggest_categorical("beta", [0.05, 0.1, 0.2, 0.5])
    elif args.model == 'CLCRec':
        args.num_neg = trial.suggest_categorical("alpha", [32, 64, 128, 256])
        args.temp_value = trial.suggest_categorical("temp_value", [0.1, 0.2, 0.5, 1.0, 2.0])
        args.lr_lambda = trial.suggest_categorical("lr_lambda", [0.1, 0.2, 0.5, 1.0])
        args.num_sample = trial.suggest_categorical("num_sample", [0.1, 0.2, 0.5, 1.0])
    elif args.model == 'CCFCRec':
        args.positive_number = trial.suggest_categorical("positive_number", [2, 5, 10])
        args.negative_number = trial.suggest_categorical("negative_number", [10, 20, 40, 80])
        args.self_neg_number = trial.suggest_categorical("self_neg_number", [10, 20, 40, 80])
        args.tau = trial.suggest_categorical("tau", [0.1, 0.2, 0.5, 1.0])
        args.lambda1 = trial.suggest_categorical("lambda1", [0.2, 0.4, 0.6, 0.8, 1.0])
    elif args.model == 'DropoutNet':
        args.n_dropout = trial.suggest_categorical("n_dropout", [0.2, 0.5, 0.8])
    elif args.model == 'Heater':
        args.n_expert = trial.suggest_categorical("n_expert", [3, 5, 7])
        args.n_dropout = trial.suggest_categorical("n_dropout", [0.2, 0.5, 0.8])
        args.alpha = trial.suggest_categorical("alpha", [0.2, 0.5, 0.8])
    elif args.model == 'MetaEmbedding':
        args.alpha = trial.suggest_categorical("alpha", [0.2, 0.5, 0.8])
    elif args.model == 'GoRec':
        args.pre_cluster_num = trial.suggest_categorical("pre_cluster_num", [20, 50, 80, 100, 200, 500])
        args.uni_coeff = trial.suggest_categorical("uni_coeff", [1, 5, 10, 15])
        args.kl_coeff = trial.suggest_categorical("kl_coeff", [10, 50, 100, 500, 1000])
        args.dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.5])

    # model register
    available_models = ['MF', 'LightGCN', 'SimGCL', 'XSimGCL', 'NCL', 'KNN', 'DUIF', 'DeepMusic', 'MTPR',
                        'VBPR', 'AMR', 'GAR', 'ALDI', 'CLCRec', 'LARA', 'CCFCRec', 'DropoutNet', 'Heater',
                        'MetaEmbedding', 'GoRec']
    if args.model in available_models:
        if args.model == 'MF' or args.model == 'LightGCN' or args.model == 'SimGCL' \
                or args.model == 'XSimGCL' or args.model == 'NCL':
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
    all_ndcg_results = model.overall_test_results[0][3]
    return all_ndcg_results


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

    device = torch.device(f"cuda:{args.gpu_id}") if (args.use_gpu and torch.cuda.is_available()) else torch.device("cpu")

    # data loader
    training_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_train.csv')
    # following the widely used setting in previous works, the 'all' set is used for validation.
    all_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/overall_val.csv')
    warm_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_val.csv')
    cold_valid_data = DataLoader.load_data_set(
        f'./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_val.csv')
    all_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/overall_test.csv')
    warm_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_test.csv')
    cold_test_data = DataLoader.load_data_set(
        f'./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_test.csv')

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

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna.db",
        study_name=f"{args.model}_{args.dataset}_{args.cold_object}_cs",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    best_param = study.best_params
    pickle.dump(best_param, open(f'./param/{args.model}_{args.dataset}_{args.cold_object}_cs.pkl', 'wb'))
    print(best_param)
