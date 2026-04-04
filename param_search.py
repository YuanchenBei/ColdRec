"""
Hyper-parameter search with Optuna using the same training pipeline as main.py.
Run from the ColdRec repo root (same working directory as main.py).

Suggestion spaces mirror config/model_param.py registered arguments per model.
"""
import argparse
import os
import pickle

import optuna
from optuna.trial import TrialState

from config.model_param import model_specific_param, _str2bool
from main import Config, model_factory
from model import AVAILABLE_MODELS
from util.utils import set_seed

args = None

# MLP / mapper hidden widths tuned in Optuna (emb_size stays 64 below).
HIDDEN_DIM_CHOICES = [64, 128, 256]


def _search_result_slug(args: argparse.Namespace) -> str:
    """Unique Optuna study / pickle basename; distinguishes --backbone for warm-emb baselines."""
    return f'{args.model}_{args.dataset}_{args.cold_object}_bb_{args.backbone}_cs'


def _persist_study_best_params(study: optuna.Study, out_path: str) -> bool:
    """Pickle ``study.best_params`` (best over **completed** trials only). Returns False if none finished."""
    try:
        best_param = study.best_params
    except (RuntimeError, ValueError):
        return False
    with open(out_path, 'wb') as f:
        pickle.dump(best_param, f)
    return True


def _make_after_trial_save_callback(out_path: str):
    """After each COMPLETE trial, refresh pkl with current best among all completed trials."""

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        _persist_study_best_params(study, out_path)

    return _cb


def _suggest_float_log(trial, name: str, low: float, high: float) -> float:
    """Optuna 2.x+ compatible log-uniform suggestion."""
    try:
        return trial.suggest_float(name, low, high, log=True)
    except TypeError:
        return trial.suggest_loguniform(name, low, high)


def _apply_optuna_suggestions(trial, args: argparse.Namespace) -> None:
    """
    Fill args with trial suggestions. Matches model_specific_param definitions.
    VBPR / AMR / MTPR use per-parameter-group lrs (p_emb, p_ctx, p_proj), not args.lr.
    """
    m = args.model

    # --- Optimizers that do NOT use args.lr in train() ---
    if m == 'VBPR':
        args.p_emb = [
            trial.suggest_categorical('p_emb_lr', [1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 0.05]),
            trial.suggest_categorical('p_emb_wd', [0.0, 1e-6, 1e-5, 1e-4]),
        ]
        args.p_ctx = [
            trial.suggest_categorical('p_ctx_lr', [1e-4, 5e-3, 1e-3, 1e-2, 0.05]),
            trial.suggest_categorical('p_ctx_wd', [0.0, 1e-4, 1e-3, 1e-2, 0.05]),
        ]
        args.lr = args.p_emb[0]
    elif m == 'AMR':
        args.p_emb = [
            trial.suggest_categorical('p_emb_lr', [1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 0.05]),
            trial.suggest_categorical('p_emb_wd', [0.0, 1e-6, 1e-5, 1e-4]),
        ]
        args.p_ctx = [
            trial.suggest_categorical('p_ctx_lr', [1e-4, 5e-3, 1e-3, 1e-2, 0.05]),
            trial.suggest_categorical('p_ctx_wd', [0.0, 1e-4, 1e-3, 1e-2, 0.05]),
        ]
        args.eps = trial.suggest_categorical('amr_eps', [0.05, 0.1, 0.2, 0.3])
        args.lmd = trial.suggest_categorical('amr_lmd', [0.5, 1.0, 1.5, 2.0])
        args.lr = args.p_emb[0]
    elif m == 'MTPR':
        args.p_emb = [
            trial.suggest_categorical('p_emb_lr', [1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 0.05]),
            trial.suggest_categorical('p_emb_wd', [0.0, 1e-6, 1e-5, 1e-4]),
        ]
        args.p_ctx = [
            trial.suggest_categorical('p_ctx_lr', [1e-4, 5e-3, 1e-3, 1e-2, 0.05]),
            trial.suggest_categorical('p_ctx_wd', [0.0, 1e-4, 1e-3, 1e-2, 0.05]),
        ]
        args.p_proj = [
            trial.suggest_categorical('p_proj_lr', [1e-4, 5e-3, 1e-3, 1e-2, 0.05]),
            trial.suggest_categorical('p_proj_wd', [0.0, 1e-4, 1e-3, 1e-2, 0.05]),
        ]
        args.lr = args.p_emb[0]
    else:
        args.lr = trial.suggest_categorical('lr', [1e-4, 5e-3, 1e-3, 5e-2, 1e-2])

    args.reg = trial.suggest_categorical('reg', [1e-5, 5e-4, 1e-4, 5e-3, 1e-3])

    # Fixed latent dim; must match ./emb/..._{backbone}_*.pt when loading pretrain / KNN backbone.
    args.emb_size = 64

    # --- Model-specific ---
    if m == 'KNN':
        args.knn_num = trial.suggest_int('knn_num', 3, 12)
    elif m in ('LightGCN', 'NGCF'):
        args.layers = trial.suggest_int('layers', 1, 3)
    elif m == 'SimGCL':
        args.layers = trial.suggest_int('layers', 1, 3)
        args.cl_rate = trial.suggest_categorical('cl_rate', [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
        args.tau = trial.suggest_categorical('tau', [0.2, 0.4, 0.6, 0.8, 1.0])
        args.eps = trial.suggest_categorical('eps', [0.1, 0.2, 0.3, 0.4, 0.5])
    elif m == 'XSimGCL':
        args.layers = trial.suggest_int('layers', 1, 3)
        args.l_cl = trial.suggest_int('l_cl', 1, args.layers)
        args.cl_rate = trial.suggest_categorical('cl_rate', [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
        args.tau = trial.suggest_categorical('tau', [0.2, 0.4, 0.6, 0.8, 1.0])
        args.eps = trial.suggest_categorical('eps', [0.1, 0.2, 0.3, 0.4, 0.5])
    elif m == 'NCL':
        # LGCN_Encoder builds emb_list of length (layers+1); train() uses emb_list[hyper_layers*2].
        # Valid indices are 0..layers, so require hyper_layers*2 <= layers.
        args.layers = trial.suggest_int('layers', 1, 3)
        hl_max = args.layers // 2
        if hl_max < 1:
            args.hyper_layers = 0
        else:
            args.hyper_layers = trial.suggest_int('hyper_layers', 1, hl_max)
        args.alpha = trial.suggest_categorical('alpha', [0.5, 1.0, 1.5, 2.0])
        args.ssl_reg = _suggest_float_log(trial, 'ssl_reg', 1e-8, 1e-4)
        args.proto_reg = _suggest_float_log(trial, 'proto_reg', 1e-8, 1e-4)
        args.tau = trial.suggest_categorical('tau', [0.01, 0.05, 0.1, 0.2, 0.5])
        args.num_clusters = trial.suggest_categorical(
            'num_clusters', [20, 50, 80, 100, 200, 500]
        )
    elif m == 'ALDI':
        args.alpha = trial.suggest_categorical('alpha', [0.5, 1.0, 1.5, 2.0])
        args.beta = trial.suggest_categorical('beta', [0.05, 0.1, 0.2, 0.5])
        args.gamma = trial.suggest_categorical('gamma', [0.1, 0.2, 0.5, 1.0])
        args.tws = trial.suggest_categorical('tws', [0, 1])
        args.freq_coef_M = trial.suggest_categorical('freq_coef_M', [2.0, 4.0, 6.0, 8.0])
        args.aldi_hidden = trial.suggest_categorical('aldi_hidden', HIDDEN_DIM_CHOICES)
    elif m == 'GAR':
        args.alpha = trial.suggest_categorical('alpha', [0.05, 0.1, 0.2, 0.5])
        args.beta = trial.suggest_categorical('beta', [0.05, 0.1, 0.2, 0.5])
    elif m == 'CLCRec':
        args.num_neg = trial.suggest_categorical('num_neg', [32, 64, 128, 256])
        args.temp_value = trial.suggest_categorical('temp_value', [0.1, 0.2, 0.5, 1.0, 2.0])
        args.lr_lambda = trial.suggest_categorical('lr_lambda', [0.1, 0.2, 0.5, 1.0])
        args.num_sample = trial.suggest_categorical('num_sample', [0.1, 0.2, 0.5, 1.0])
    elif m == 'CCFCRec':
        args.positive_number = trial.suggest_categorical('positive_number', [2, 5, 10])
        args.negative_number = trial.suggest_categorical('negative_number', [10, 20, 40, 80])
        args.self_neg_number = trial.suggest_categorical('self_neg_number', [10, 20, 40, 80])
        args.tau = trial.suggest_categorical('ccfc_tau', [0.1, 0.2, 0.5, 1.0])
        args.lambda1 = trial.suggest_categorical('lambda1', [0.2, 0.4, 0.6, 0.8, 1.0])
        args.attr_present_dim = 64
        args.implicit_dim = 64
        args.cat_implicit_dim = 64
        args.pretrain = trial.suggest_categorical('pretrain', [False, True])
        args.pretrain_update = trial.suggest_categorical('pretrain_update', [False, True])
    elif m == 'DropoutNet':
        args.n_dropout = trial.suggest_categorical('n_dropout', [0.2, 0.5, 0.8])
        args.dropoutnet_hidden1 = trial.suggest_categorical(
            'dropoutnet_hidden1', HIDDEN_DIM_CHOICES
        )
        args.dropoutnet_hidden2 = trial.suggest_categorical(
            'dropoutnet_hidden2', HIDDEN_DIM_CHOICES
        )
    elif m == 'Heater':
        args.n_expert = trial.suggest_categorical('n_expert', [3, 5, 7])
        args.n_dropout = trial.suggest_categorical('heater_n_dropout', [0.2, 0.5, 0.8])
        args.alpha = trial.suggest_categorical('heater_alpha', [0.2, 0.5, 0.8])
        args.heater_mlp_hidden = trial.suggest_categorical(
            'heater_mlp_hidden', HIDDEN_DIM_CHOICES
        )
    elif m == 'MetaEmbedding':
        args.alpha = trial.suggest_categorical('meta_alpha', [0.2, 0.5, 0.8])
    elif m == 'GoRec':
        args.pre_cluster_num = trial.suggest_categorical(
            'pre_cluster_num', [20, 50, 80, 100, 200, 500]
        )
        args.uni_coeff = trial.suggest_categorical('uni_coeff', [1, 5, 10, 15])
        args.kl_coeff = trial.suggest_categorical('kl_coeff', [10, 50, 100, 500, 1000])
        args.dropout = trial.suggest_categorical('gorec_dropout', [0.0, 0.2, 0.5])
    elif m == 'AGNN':
        args.agnn_knn_k = trial.suggest_int('agnn_knn_k', 5, 20)
        args.agnn_dropout = trial.suggest_categorical('agnn_dropout', [0.3, 0.4, 0.5, 0.6])
        args.agnn_rank_weight = trial.suggest_categorical(
            'agnn_rank_weight', [0.5, 1.0, 1.5, 2.0]
        )
        args.agnn_align_weight = trial.suggest_categorical(
            'agnn_align_weight', [0.5, 1.0, 1.5, 2.0]
        )
        args.agnn_vae_lambda = trial.suggest_categorical(
            'agnn_vae_lambda', [0.5, 1.0, 1.5, 2.0]
        )
        args.agnn_eval_chunk = trial.suggest_categorical(
            'agnn_eval_chunk', [256, 512, 1024]
        )
        args.agnn_finetune = trial.suggest_categorical('agnn_finetune', [False, True])
        args.agnn_no_backbone = trial.suggest_categorical('agnn_no_backbone', [False, True])
    elif m == 'M2VAE':
        args.positive_number = trial.suggest_categorical('positive_number', [5, 10, 20])
        args.negative_number = trial.suggest_categorical('negative_number', [20, 40, 80])
        args.self_neg_number = trial.suggest_categorical('self_neg_number', [20, 40, 80])
        args.attr_present_dim = 64
        args.implicit_dim = 64
        args.cat_implicit_dim = 64
        args.tau = trial.suggest_categorical('m2vae_tau', [0.05, 0.1, 0.2, 0.5])
        args.m2vae_weight_decay = trial.suggest_categorical(
            'm2vae_weight_decay', [0.01, 0.05, 0.1, 0.2]
        )
        args.m2vae_kld_weight = trial.suggest_categorical(
            'm2vae_kld_weight', [0.5, 1.0, 2.0]
        )
        args.m2vae_recon_weight = trial.suggest_categorical(
            'm2vae_recon_weight', [0.5, 1.0, 2.0]
        )
        args.m2vae_decouple_weight = trial.suggest_categorical(
            'm2vae_decouple_weight', [50.0, 100.0, 200.0]
        )
        args.m2vae_pretrain = trial.suggest_categorical('m2vae_pretrain', [False, True])
        args.m2vae_pretrain_update = trial.suggest_categorical(
            'm2vae_pretrain_update', [False, True]
        )
        args.m2vae_attr_mask_neg1 = trial.suggest_categorical(
            'm2vae_attr_mask_neg1', [False, True]
        )
    elif m == 'CGRC':
        args.cgrc_mask_rho = trial.suggest_categorical(
            'cgrc_mask_rho', [0.1, 0.2, 0.3, 0.4, 0.5]
        )
        args.cgrc_recon_topk = trial.suggest_categorical(
            'cgrc_recon_topk', [10, 20, 30, 40]
        )
        args.cgrc_layers_gprime = trial.suggest_int('cgrc_layers_gprime', 1, 3)
        args.cgrc_layers_full = trial.suggest_int('cgrc_layers_full', 1, 3)
        args.cgrc_layers_ghat = trial.suggest_int('cgrc_layers_ghat', 1, 3)
        args.cgrc_lambda_e = trial.suggest_categorical(
            'cgrc_lambda_e', [0.5, 1.0, 1.5, 2.0]
        )
        args.cgrc_tau = trial.suggest_categorical('cgrc_tau', [0.2, 0.5, 1.0])
        args.cgrc_mlp_hidden = trial.suggest_categorical('cgrc_mlp_hidden', HIDDEN_DIM_CHOICES)
        args.cgrc_le_max_edges = trial.suggest_categorical(
            'cgrc_le_max_edges', [2048, 4096, 8192]
        )
        args.cgrc_ranking_neg_per_user = trial.suggest_categorical(
            'cgrc_ranking_neg_per_user', [16, 32, 64]
        )
    elif m == 'FSGNN':
        args.fsgnn_lambda_fc = trial.suggest_categorical(
            'fsgnn_lambda_fc', [0.25, 0.5, 0.75, 1.0]
        )
        args.fsgnn_mu_sc = trial.suggest_categorical(
            'fsgnn_mu_sc', [0.25, 0.5, 0.75, 1.0]
        )
        args.fsgnn_p_drop = trial.suggest_categorical(
            'fsgnn_p_drop', [0.1, 0.2, 0.3, 0.4]
        )
        args.fsgnn_gat_hidden = trial.suggest_categorical('fsgnn_gat_hidden', HIDDEN_DIM_CHOICES)
        args.fsgnn_ppr_delta = trial.suggest_categorical(
            'fsgnn_ppr_delta', [0.1, 0.15, 0.2, 0.25]
        )
        args.fsgnn_ppr_iter = trial.suggest_categorical(
            'fsgnn_ppr_iter', [15, 30, 45]
        )
        args.fsgnn_ppr_topk = trial.suggest_categorical('fsgnn_ppr_topk', [10, 15, 20])
        args.fsgnn_knn_k = trial.suggest_categorical('fsgnn_knn_k', [10, 15, 20, 25])
        args.fsgnn_light_layers = trial.suggest_int('fsgnn_light_layers', 1, 3)
        args.fsgnn_weight_decay = trial.suggest_categorical(
            'fsgnn_weight_decay', [1e-4, 5e-4, 1e-3]
        )
        args.fsgnn_sc_max_edges = trial.suggest_categorical(
            'fsgnn_sc_max_edges', [2048, 4096, 8192]
        )
        args.fsgnn_fallback_dim = trial.suggest_categorical(
            'fsgnn_fallback_dim', HIDDEN_DIM_CHOICES
        )
        args.fsgnn_dropout_cold_side_only = trial.suggest_categorical(
            'fsgnn_dropout_cold_side_only', [True, False]
        )
        args.fsgnn_id_residual = trial.suggest_categorical('fsgnn_id_residual', [False, True])
        args.fsgnn_sc_layers = trial.suggest_int('fsgnn_sc_layers', 1, 3)
        args.fsgnn_knn_weighted = trial.suggest_categorical(
            'fsgnn_knn_weighted', [True, False]
        )
        args.fsgnn_ppr_weighted = trial.suggest_categorical(
            'fsgnn_ppr_weighted', [True, False]
        )
        args.fsgnn_fc_decoder_layers = trial.suggest_int('fsgnn_fc_decoder_layers', 1, 3)
        args.fsgnn_lfc_cold_side_only = trial.suggest_categorical(
            'fsgnn_lfc_cold_side_only', [True, False]
        )
    elif m == 'USIM':
        args.actor_lr = trial.suggest_categorical(
            'actor_lr', [1e-4, 2.5e-4, 5e-4, 1e-3]
        )
        args.critic_lr = trial.suggest_categorical(
            'critic_lr', [1e-4, 5e-4, 1e-3, 2e-3]
        )
        args.gamma = trial.suggest_categorical('usim_gamma', [0.95, 0.99, 0.995])
        args.usim_max_time = trial.suggest_int('usim_max_time', 5, 10)
        args.usim_transition_rate = trial.suggest_categorical(
            'usim_transition_rate', [0.02, 0.05, 0.1]
        )
        args.usim_k = trial.suggest_categorical('usim_k', [5, 10, 15, 20])
        args.usim_reward_weight = trial.suggest_categorical(
            'usim_reward_weight', [0.25, 0.5, 0.75, 1.0]
        )
        args.usim_reward_cost = trial.suggest_categorical(
            'usim_reward_cost', [0.1, 0.2, 0.3]
        )
        args.usim_optimize_batch = trial.suggest_categorical(
            'usim_optimize_batch', [10240, 20480, 40960]
        )
        args.usim_buffer_max_len = trial.suggest_categorical(
            'usim_buffer_max_len', [512, 1024, 2048]
        )
        args.usim_content_hidden = trial.suggest_categorical(
            'usim_content_hidden', HIDDEN_DIM_CHOICES
        )
        args.usim_actor_weight_decay = trial.suggest_categorical(
            'usim_actor_weight_decay', [0.0, 1e-6, 1e-5]
        )
        args.usim_critic_weight_decay = trial.suggest_categorical(
            'usim_critic_weight_decay', [0.0, 1e-6, 1e-5]
        )


def objective(trial):
    global args

    _apply_optuna_suggestions(trial, args)

    if args.model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model name: {args.model}. "
            f"Available: {list(AVAILABLE_MODELS.keys())}"
        )

    set_seed(args.seed, args.use_gpu)
    config = Config(args)
    model = model_factory(config)
    print(f"Registered model: {args.model}.")
    model.run()

    return model.overall_test_results[0][3]


if __name__ == '__main__':
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
        help='Whether to save embeddings (true/false; default true)',
    )
    parser.add_argument('--gpu_id', type=int, default=0, help='CUDA id')
    parser.add_argument('--cold_object', default='item', type=str, choices=['user', 'item'])
    parser.add_argument(
        '--backbone',
        default='MF',
        help='Name tag for ./emb/..._{backbone}_*.pt; train backbone with same --emb_size as this run (search fixes emb_size=64).',
    )
    parser.add_argument(
        '--early_stop',
        type=int,
        default=10,
        help='Early stopping patience (0 = disabled).',
    )
    parser.add_argument(
        '--eval_every',
        type=int,
        default=1,
        help='Validation every N epochs (>=1). Default 1 = every epoch.',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=30,
        help='Number of Optuna trials.',
    )
    ns, _ = parser.parse_known_args()
    parser = model_specific_param(ns.model, parser, AVAILABLE_MODELS)
    args = parser.parse_args()
    print(args)

    os.makedirs('./param', exist_ok=True)

    slug = _search_result_slug(args)
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///optuna.db',
        study_name=slug,
        load_if_exists=True,
    )
    out_path = f'./param/{slug}.pkl'
    save_after_trial = _make_after_trial_save_callback(out_path)
    try:
        study.optimize(
            objective,
            n_trials=max(1, int(args.n_trials)),
            callbacks=[save_after_trial],
        )
    finally:
        if _persist_study_best_params(study, out_path):
            print(study.best_params)
        else:
            print('No completed trials; ./param/*.pkl not updated.')
