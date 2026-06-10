import pickle
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='citeulike')
    parser.add_argument('--model', default='MF')
    parser.add_argument('--cold_object', default='item', type=str, choices=['user', 'item'])
    parser.add_argument('--backbone', default='MF')
    args = parser.parse_args()
    candidates = [
        os.path.join(
            './param',
            f'{args.model}_{args.dataset}_{args.cold_object}_bb_{args.backbone}_cs.pkl',
        ),
        f'./{args.model}_{args.dataset}_{args.cold_object}_bb_{args.backbone}_cs.pkl',
        f'./{args.model}_{args.dataset}_{args.cold_object}_cs.pkl',
    ]
    best_path = next((p for p in candidates if os.path.isfile(p)), None)
    if best_path is None:
        raise FileNotFoundError(
            'No best-parameter file found. Tried: ' + ', '.join(candidates)
        )
    with open(best_path, 'rb') as f:
        best_param_dict = pickle.load(f)
    print(f"Best parameters of {args.model} on {args.dataset} ({args.cold_object}, backbone={args.backbone}):")
    print(f"Loaded from: {best_path}")
    print(best_param_dict)
