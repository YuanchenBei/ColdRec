import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='citeulike')
    parser.add_argument('--model', default='MF')
    parser.add_argument('--cold_object', default='item', type=str, choices=['user', 'item'])
    args = parser.parse_args()
    with open(f'./{args.model}_{args.dataset}_{args.cold_object}_cs.pkl', 'rb') as f:
        best_param_dict = pickle.load(f)
    print(f"Best parameters of {args.model} on {args.dataset}:")
    print(best_param_dict)
