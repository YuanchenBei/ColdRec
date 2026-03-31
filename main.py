import argparse
import json
import os
from datetime import datetime

import torch
import numpy as np
import pickle
from util.loader import DataLoader
from util.utils import set_seed
from config.model_param import model_specific_param, _str2bool
from model import AVAILABLE_MODELS
from util.databuilder import ColdStartDataBuilder

class Config:
    """
    Configuration class that encapsulates all model and training parameters.
    
    This class centralizes all configuration data, making it easier to pass
    to model constructors and maintain consistency across the codebase.
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda:%d" % (args.gpu_id) if (torch.cuda.is_available() and args.use_gpu) else "cpu")
        
        # Load data
        training_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_train.csv')
        all_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/overall_val.csv')
        warm_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_val.csv')
        cold_valid_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_val.csv')
        all_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/overall_test.csv')
        warm_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/warm_test.csv')
        cold_test_data = DataLoader.load_data_set(f'./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_test.csv')

        # Dataset information
        data_info_dict = pickle.load(open(f'./data/{args.dataset}/cold_{args.cold_object}/info_dict.pkl', 'rb'))
        user_num = data_info_dict['user_num']
        item_num = data_info_dict['item_num']
        warm_user_idx = data_info_dict['warm_user']
        warm_item_idx = data_info_dict['warm_item']
        cold_user_idx = data_info_dict['cold_user']
        cold_item_idx = data_info_dict['cold_item']
        print(f"Dataset: {args.dataset}, User num: {user_num}, Item num: {item_num}.")

        # Content obtaining
        user_content, item_content = None, None
        if args.cold_object == 'user':
            user_content = np.load(f'./data/{args.dataset}/{args.dataset}_{args.cold_object}_content.npy')
            print(f'user content shape: {user_content.shape}')
        if args.cold_object == 'item':
            item_content = np.load(f'./data/{args.dataset}/{args.dataset}_{args.cold_object}_content.npy')
            print(f'item content shape: {item_content.shape}')
            
        self.data = ColdStartDataBuilder(training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                     warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                     warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx,
                                     user_content, item_content)

def model_factory(config: Config):
    """
    Factory function to create model instances based on configuration.
    
    Args:
        config: Configuration object containing all necessary parameters
        
    Returns:
        Model instance implementing BaseColdStartTrainer
        
    Raises:
        ValueError: If the model name is not in the available models list
    """
    model_name = config.args.model
    model_class = AVAILABLE_MODELS.get(model_name)

    if model_class is None:
        raise ValueError(f"Invalid model name: {model_name}. "
                         f"Available models: {list(AVAILABLE_MODELS.keys())}")

    return model_class(config)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments and return a namespace object.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='citeulike')
    parser.add_argument('--model', default='MF')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--topN', default='10,20')
    parser.add_argument('--bs', type=int, default=4096, help='training batch size')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', type=float, default=0.0001)
    parser.add_argument('--runs', type=int, default=1, help='model runs')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument(
        '--use_gpu',
        type=_str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to use CUDA (true/false; default true)',
    )
    parser.add_argument(
        '--save_emb',
        type=_str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to save user/item embeddings (true/false; default true)',
    )
    parser.add_argument('--gpu_id', type=int, default=0, help='CUDA id')
    parser.add_argument('--cold_object', default='item', type=str, choices=['user', 'item'])
    parser.add_argument(
        '--backbone',
        default='MF',
        help='Name tag for ./emb/..._{backbone}_*.pt (pretrain / KNN). Train that checkpoint with the same --emb_size (default 64).',
    )
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience. If set to 0, early stopping is disabled.')
    parser.add_argument(
        '--eval_every',
        type=int,
        default=1,
        help='Run validation (fast_evaluation) every N training epochs (>=1). Default 1 = every epoch.',
    )
    parser.add_argument(
        '--result_dir', type=str, default='./result',
        help='Base directory when --result_file is not set (uses <result_dir>/<model>/).')
    parser.add_argument(
        '--result_log', type=str, default='history.txt',
        help='Log filename under <result_dir>/<model>/ when --result_file is not set.')
    parser.add_argument(
        '--result_file', type=str, default='',
        help='If set, write to this path instead of result_dir/<model>/result_log.')
    parser.add_argument(
        '--result_overwrite', action='store_true',
        help='Overwrite the result file instead of appending (default: append).')
    args, _ = parser.parse_known_args()
    parser = model_specific_param(args.model, parser, AVAILABLE_MODELS)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    config = Config(args)

    top_Ns = args.topN.split(',')
    results = {setting: {metric: [[] for _ in top_Ns] for metric in ['hit', 'precision', 'recall', 'ndcg']} for setting in ['all', 'cold', 'warm']}
    
    time_results = []

    for round_num in range(args.runs):
        print(f"Start round {round_num} running!")

        seed = args.seed if args.runs == 1 else round_num
        set_seed(seed, args.use_gpu)
            
        model = model_factory(config)
        print(f"Registered model: {args.model}.")

        model.run()

        # Results recording
        for i in range(len(top_Ns)):
            for setting, test_results in [('all', model.overall_test_results), ('cold', model.cold_test_results), ('warm', model.warm_test_results)]:
                results[setting]['hit'][i].append(test_results[i][0])
                results[setting]['precision'][i].append(test_results[i][1])
                results[setting]['recall'][i].append(test_results[i][2])
                results[setting]['ndcg'][i].append(test_results[i][3])

        elapsed = model.train_end_time - model.train_start_time
        er = int(getattr(model, 'epochs_ran', 0) or 0)
        if er > 0:
            epochs_done = er
        elif args.epochs > 0:
            epochs_done = args.epochs
        else:
            epochs_done = 1
        time_results.append(elapsed / epochs_done)

    for i, top_n in enumerate(top_Ns):
        print("*" * 80)
        for setting_name, setting_key in [('Overall', 'all'), ('Cold-Start', 'cold'), ('Warm-Start', 'warm')]:
            print(f"Top-{top_n} {setting_name} Test Performance:")
            
            metrics = {
                'Hit': (np.mean(results[setting_key]['hit'][i]), np.std(results[setting_key]['hit'][i])),
                'Precision': (np.mean(results[setting_key]['precision'][i]), np.std(results[setting_key]['precision'][i])),
                'Recall': (np.mean(results[setting_key]['recall'][i]), np.std(results[setting_key]['recall'][i])),
                'NDCG': (np.mean(results[setting_key]['ndcg'][i]), np.std(results[setting_key]['ndcg'][i]))
            }
            
            print(', '.join([f"{name}@{top_n}: {mean:.4f}±{std:.4f}" for name, (mean, std) in metrics.items()]))

    print(f"Efficiency Performance:")
    mean_time, std_time = np.mean(time_results), np.std(time_results)
    print(f"Time: {mean_time:.4f}±{std_time:.4f} seconds per completed training epoch.")

    # --- Write method, hyperparameters, and metrics to result file ---
    def _args_to_serializable(ns):
        out = {}
        for k, v in sorted(vars(ns).items()):
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = list(v)
            else:
                out[k] = repr(v)
        return out

    metrics_payload = {}
    for i, top_n in enumerate(top_Ns):
        tn = str(top_n)
        metrics_payload[tn] = {}
        for setting_name, setting_key in [('Overall', 'all'), ('Cold-Start', 'cold'), ('Warm-Start', 'warm')]:
            metrics_payload[tn][setting_key] = {
                'Hit': {'mean': float(np.mean(results[setting_key]['hit'][i])),
                        'std': float(np.std(results[setting_key]['hit'][i]))},
                'Precision': {'mean': float(np.mean(results[setting_key]['precision'][i])),
                              'std': float(np.std(results[setting_key]['precision'][i]))},
                'Recall': {'mean': float(np.mean(results[setting_key]['recall'][i])),
                           'std': float(np.std(results[setting_key]['recall'][i]))},
                'NDCG': {'mean': float(np.mean(results[setting_key]['ndcg'][i])),
                         'std': float(np.std(results[setting_key]['ndcg'][i]))},
            }

    if args.result_file and str(args.result_file).strip():
        result_path = os.path.abspath(args.result_file)
    else:
        result_path = os.path.join(
            os.path.abspath(args.result_dir), args.model, args.result_log
        )
    result_dir_for_file = os.path.dirname(result_path)
    if result_dir_for_file:
        os.makedirs(result_dir_for_file, exist_ok=True)

    lines = [
        '=== ColdRec Run Result ===',
        f'timestamp: {datetime.now().isoformat(timespec="seconds")}',
        f'method: {args.model}',
        f'dataset: {args.dataset}',
        f'cold_object: {args.cold_object}',
        f'backbone: {args.backbone}',
        f'runs: {args.runs}',
        '',
        '--- Hyperparameters ---',
    ]
    for k, v in sorted(_args_to_serializable(args).items()):
        lines.append(f'{k}: {v}')
    lines.extend([
        '',
        '--- Test Metrics (mean ± std) ---',
    ])
    for i, top_n in enumerate(top_Ns):
        for setting_name, setting_key in [('Overall', 'all'), ('Cold-Start', 'cold'), ('Warm-Start', 'warm')]:
            m = metrics_payload[str(top_n)][setting_key]
            lines.append(
                f'Top-{top_n} {setting_name}: '
                f"Hit={m['Hit']['mean']:.4f}±{m['Hit']['std']:.4f}, "
                f"Precision={m['Precision']['mean']:.4f}±{m['Precision']['std']:.4f}, "
                f"Recall={m['Recall']['mean']:.4f}±{m['Recall']['std']:.4f}, "
                f"NDCG={m['NDCG']['mean']:.4f}±{m['NDCG']['std']:.4f}"
            )
    lines.extend([
        '',
        '--- Efficiency ---',
        f'seconds_per_completed_epoch_mean: {float(mean_time):.6f}',
        f'seconds_per_completed_epoch_std: {float(std_time):.6f}',
        '',
        '--- JSON (machine-readable) ---',
        json.dumps(
            {
                'method': args.model,
                'hyperparameters': _args_to_serializable(args),
                'metrics': metrics_payload,
                'efficiency': {
                    'seconds_per_completed_epoch_mean': float(mean_time),
                    'seconds_per_completed_epoch_std': float(std_time),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
    ])

    block = '\n'.join(lines) + '\n'
    run_sep = '\n' + '=' * 80 + '\n'
    mode = 'w' if args.result_overwrite else 'a'
    with open(result_path, mode, encoding='utf-8') as f:
        if mode == 'a' and os.path.isfile(result_path) and os.path.getsize(result_path) > 0:
            f.write(run_sep)
        f.write(block)
    print(f"Results written ({'overwrite' if args.result_overwrite else 'append'}) to: {result_path}")
