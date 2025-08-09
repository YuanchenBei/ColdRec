import argparse
import pickle

import numpy as np
import torch

from config.model_param import model_specific_param
from model import AVAILABLE_MODELS
from util.databuilder import ColdStartDataBuilder
from util.loader import DataLoader
from util.utils import set_seed


class Config:
    """
    Configuration class that encapsulates all model and training parameters.

    This class centralizes all configuration data, making it easier to pass
    to model constructors and maintain consistency across the codebase.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}") if (args.use_gpu and torch.cuda.is_available()) else torch.device("cpu")

        # Load data
        training_data = DataLoader.load_data_set(f"./data/{args.dataset}/cold_{args.cold_object}/warm_train.csv")
        all_valid_data = DataLoader.load_data_set(f"./data/{args.dataset}/cold_{args.cold_object}/overall_val.csv")
        warm_valid_data = DataLoader.load_data_set(f"./data/{args.dataset}/cold_{args.cold_object}/warm_val.csv")
        cold_valid_data = DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_val.csv"
        )
        all_test_data = DataLoader.load_data_set(f"./data/{args.dataset}/cold_{args.cold_object}/overall_test.csv")
        warm_test_data = DataLoader.load_data_set(f"./data/{args.dataset}/cold_{args.cold_object}/warm_test.csv")
        cold_test_data = DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_test.csv"
        )

        # Dataset information
        data_info_dict = pickle.load(open(f"./data/{args.dataset}/cold_{args.cold_object}/info_dict.pkl", "rb"))
        user_num = data_info_dict["user_num"]
        item_num = data_info_dict["item_num"]
        warm_user_idx = data_info_dict["warm_user"]
        warm_item_idx = data_info_dict["warm_item"]
        cold_user_idx = data_info_dict["cold_user"]
        cold_item_idx = data_info_dict["cold_item"]
        print(f"Dataset: {args.dataset}, User num: {user_num}, Item num: {item_num}.")

        # Content obtaining
        user_content, item_content = None, None
        if args.cold_object == "user":
            user_content = np.load(f"./data/{args.dataset}/{args.dataset}_{args.cold_object}_content.npy")
            print(f"user content shape: {user_content.shape}")
        if args.cold_object == "item":
            item_content = np.load(f"./data/{args.dataset}/{args.dataset}_{args.cold_object}_content.npy")
            print(f"item content shape: {item_content.shape}")

        self.data = ColdStartDataBuilder(
            training_data,
            warm_valid_data,
            cold_valid_data,
            all_valid_data,
            warm_test_data,
            cold_test_data,
            all_test_data,
            user_num,
            item_num,
            warm_user_idx,
            warm_item_idx,
            cold_user_idx,
            cold_item_idx,
            user_content,
            item_content,
        )


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
        raise ValueError(f"Invalid model name: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")

    return model_class(config)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments and return a namespace object.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeulike")
    parser.add_argument("--model", default="MF")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--topN", default="10,20")
    parser.add_argument("--bs", type=int, default=2048, help="training batch size")
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--runs", type=int, default=1, help="model runs")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--use_gpu", default=True, help="Whether to use CUDA")
    parser.add_argument("--save_emb", default=True, help="Whether to save the user/item embeddings")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA id")
    parser.add_argument("--cold_object", default="item", type=str, choices=["user", "item"])
    parser.add_argument("--backbone", default="MF")
    parser.add_argument(
        "--early_stop", type=int, default=10, help="Early stopping patience. If set to 0, early stopping is disabled."
    )
    args, _ = parser.parse_known_args()
    parser = model_specific_param(args.model, parser, AVAILABLE_MODELS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    config = Config(args)

    top_Ns = args.topN.split(",")
    results = {
        setting: {metric: [[] for _ in top_Ns] for metric in ["hit", "precision", "recall", "ndcg"]}
        for setting in ["all", "cold", "warm"]
    }

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
            for setting, test_results in [
                ("all", model.overall_test_results),
                ("cold", model.cold_test_results),
                ("warm", model.warm_test_results),
            ]:
                results[setting]["hit"][i].append(test_results[i][0])
                results[setting]["precision"][i].append(test_results[i][1])
                results[setting]["recall"][i].append(test_results[i][2])
                results[setting]["ndcg"][i].append(test_results[i][3])

        time_results.append((model.train_end_time - model.train_start_time) / args.epochs)

    for i, top_n in enumerate(top_Ns):
        print("*" * 80)
        for setting_name, setting_key in [("Overall", "all"), ("Cold-Start", "cold"), ("Warm-Start", "warm")]:
            print(f"Top-{top_n} {setting_name} Test Performance:")

            metrics = {
                "Hit": (np.mean(results[setting_key]["hit"][i]), np.std(results[setting_key]["hit"][i])),
                "Precision": (
                    np.mean(results[setting_key]["precision"][i]),
                    np.std(results[setting_key]["precision"][i]),
                ),
                "Recall": (np.mean(results[setting_key]["recall"][i]), np.std(results[setting_key]["recall"][i])),
                "NDCG": (np.mean(results[setting_key]["ndcg"][i]), np.std(results[setting_key]["ndcg"][i])),
            }

            print(", ".join([f"{name}@{top_n}: {mean:.4f}±{std:.4f}" for name, (mean, std) in metrics.items()]))

    print("Efficiency Performance:")
    mean_time, std_time = np.mean(time_results), np.std(time_results)
    print(f"Time: {mean_time:.4f}±{std_time:.4f} seconds per epoch.")
