import argparse


def _str2bool(v):
    """Robust bool for argparse (avoids type=bool where str('False') becomes True)."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean value, got {v!r}')


def model_specific_param(model_name, parser, available_models):
    if model_name == 'KNN':
        parser.add_argument('--knn_num', type=int, default=5)
    elif model_name == 'AGNN':
        # Align AGNN-related defaults with the official implementation when possible.
        parser.set_defaults(lr=5e-4, bs=128, emb_size=30, epochs=20)
        parser.add_argument('--agnn_knn_k', type=int, default=10, help='Top-k warm neighbors for AGNN aggregation')
        parser.add_argument('--agnn_dropout', type=float, default=0.5, help='Dropout rate in AGNN content encoder')
        parser.add_argument('--agnn_rank_weight', type=float, default=1.0, help='Weight of generated-embedding ranking loss')
        parser.add_argument('--agnn_align_weight', type=float, default=1.0, help='Weight of embedding alignment loss')
        parser.add_argument('--agnn_vae_lambda', type=float, default=1.0, help='Weight of VAE regularization loss')
        parser.add_argument('--agnn_attention_size', type=int, default=50, help='Reserved AGNN attention size setting')
        parser.add_argument('--agnn_clip_norm', type=float, default=5.0, help='Reserved AGNN gradient clip norm setting')
        parser.add_argument('--agnn_finetune', action='store_true', help='Whether to fine-tune loaded backbone embeddings')
        parser.add_argument(
            '--agnn_no_backbone',
            action='store_true',
            help='If set, Xavier-init user/item embeddings instead of loading ./emb/..._{backbone}_*.pt',
        )
        parser.add_argument(
            '--agnn_eval_chunk',
            type=int,
            default=512,
            help='Rows per chunk when materializing full user/item GCN embeddings (lower = less GPU peak memory)',
        )
    elif model_name == 'M2VAE':
        # Align defaults with the official ML-20M implementation when possible.
        parser.set_defaults(lr=5e-5, bs=2048, emb_size=128, epochs=10)
        parser.add_argument('--positive_number', type=int, default=10, help='contrast positive number')
        parser.add_argument('--negative_number', type=int, default=40, help='contrast negative number')
        parser.add_argument('--self_neg_number', type=int, default=40, help='self contrast negative number')
        parser.add_argument('--attr_present_dim', type=int, default=128, help='attribute representation dimension')
        parser.add_argument('--implicit_dim', type=int, default=128, help='u/i latent dimension')
        parser.add_argument('--cat_implicit_dim', type=int, default=128, help='decoder hidden dimension')
        parser.add_argument('--tau', type=float, default=0.1, help='contrastive temperature')
        parser.add_argument('--m2vae_weight_decay', type=float, default=0.1, help='optimizer weight decay')
        parser.add_argument('--m2vae_kld_weight', type=float, default=1.0, help='weight of KL divergence term')
        parser.add_argument('--m2vae_recon_weight', type=float, default=1.0, help='weight of reconstruction term')
        parser.add_argument('--m2vae_decouple_weight', type=float, default=100.0, help='weight of decoupled contrastive term')
        parser.add_argument('--m2vae_pretrain', action='store_true', help='load backbone user/item embeddings')
        parser.add_argument('--m2vae_pretrain_update', action='store_true', help='update loaded backbone embeddings')
        parser.add_argument(
            '--m2vae_attr_mask_neg1',
            action='store_true',
            help='Attention mask: keep dims where attribute != -1 (official); requires absent dims encoded as -1 in item content',
        )
    elif model_name == 'ALDI':
        parser.add_argument('--alpha', type=float, default=0.9)
        parser.add_argument('--beta', type=float, default=0.05)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--tws', type=int, default=0, choices=[0, 1])
        parser.add_argument('--freq_coef_M', type=float, default=4)
    elif model_name == 'GAR':
        parser.add_argument('--alpha', type=float, default=0.05, help='GAR parameter alpha')
        parser.add_argument('--beta', type=float, default=0.1, help='GAR parameter beta')
    elif model_name == 'CGRC':
        # SIGIR'24 Content-based Graph Reconstruction for Cold-start Item Recommendation (single-vector content; L_M off).
        parser.set_defaults(lr=1e-3, bs=2048, emb_size=64, epochs=500)
        parser.add_argument(
            '--cgrc_mask_rho',
            type=float,
            default=0.3,
            help='Per eligible training item, Bernoulli prob. to include in simulated I_cold (mask edges)',
        )
        parser.add_argument('--cgrc_recon_topk', type=int, default=20, help='Top-K users per cold item for graph Ĝ at inference')
        parser.add_argument('--cgrc_layers_gprime', type=int, default=2, help='LightGCN layers on masked graph G′')
        parser.add_argument('--cgrc_layers_full', type=int, default=2, help='LightGCN layers on full graph G for L_R (teacher forcing)')
        parser.add_argument('--cgrc_layers_ghat', type=int, default=2, help='LightGCN layers on reconstructed graph Ĝ at test')
        parser.add_argument('--cgrc_lambda_e', type=float, default=1.0, help='Weight λ_E for reconstruction loss L_E')
        parser.add_argument('--cgrc_tau', type=float, default=0.5, help='Temperature τ for rating ranking loss L_R (paper default)')
        parser.add_argument('--cgrc_mlp_hidden', type=int, default=64, help='Edge predictor MLP hidden dim')
        parser.add_argument(
            '--cgrc_le_max_edges',
            type=int,
            default=4096,
            help='Max masked edges per batch for L_E (subsample if larger)',
        )
        parser.add_argument(
            '--cgrc_ranking_neg_per_user',
            type=int,
            default=32,
            help='Extra random non-interacted items per user to enlarge set B for L_R',
        )
    elif model_name == 'CLCRec':
        parser.add_argument('--num_neg', type=int, default=128, help='number of negative samples')
        parser.add_argument('--temp_value', type=float, default=2.0, help='Contrastive temp_value.')
        parser.add_argument('--lr_lambda', type=float, default=0.5, help='Weight loss one.')
        parser.add_argument('--num_sample', type=float, default=0.5, help='Sample ratio.')
    elif model_name == 'CCFCRec':
        parser.add_argument('--positive_number', type=int, default=5, help='contrast positive number')
        parser.add_argument('--negative_number', type=int, default=40, help='contrast negative number')
        parser.add_argument('--self_neg_number', type=int, default=40, help='contrast negative number')
        parser.add_argument('--tau', type=float, default=0.1, help='contrast loss temperature')
        parser.add_argument('--lambda1', type=float, default=0.6, help='collaborative contrast loss weight')
        parser.add_argument('--attr_present_dim', type=int, default=64, help='the dimension of present')
        parser.add_argument('--implicit_dim', type=int, default=64, help='the dimension of u/i present')
        parser.add_argument('--cat_implicit_dim', type=int, default=64, help='the q_v_c dimension')
        parser.add_argument(
            '--pretrain', type=_str2bool, default=False, nargs='?', const=True,
            help='Load user/item embeddings from ./emb/..._{backbone}_*.pt (true/false; default false)')
        parser.add_argument(
            '--pretrain_update', type=_str2bool, default=False, nargs='?', const=True,
            help='If pretrain: train loaded embeddings (true) or freeze (false); default false')
    elif model_name == 'DropoutNet':
        parser.add_argument('--n_dropout', type=float, default=0.5, help='Dropout rate of the network training')
    elif model_name == 'Heater':
        parser.add_argument('--n_expert', type=int, default=5, help='Number of experts')
        parser.add_argument('--n_dropout', type=float, default=0.5, help='Dropout rate of the network training')
        parser.add_argument('--alpha', type=float, default=0.5)
    elif model_name == 'MetaEmbedding':
        parser.add_argument('--alpha', type=float, default=0.5)
    elif model_name == 'GoRec':
        parser.add_argument('--pre_cluster_num', type=int, default=50, help='Number of clusters')
        parser.add_argument('--uni_coeff', type=float, default=5) # c 1    b 5    s 15
        parser.add_argument('--kl_coeff', type=float, default=10) # c 5000 b 10   s 5000
        parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    elif model_name == 'USIM':
        # Official USIM (Ruochen1003/USIM): PPO-style RL + discrete actor; implementation in model/USIM.py
        parser.add_argument('--actor_lr', type=float, default=0.0005, help='Actor Adam lr (official default)')
        parser.add_argument('--critic_lr', type=float, default=0.001, help='Critic Adam lr (official default)')
        parser.add_argument('--gamma', type=float, default=0.99, help='RL discount (optimize target Q)')
        parser.add_argument('--usim_max_time', type=int, default=7, help='Max RL steps per item (official max_time)')
        parser.add_argument('--usim_transition_rate', type=float, default=0.05, help='State transition scale')
        parser.add_argument('--usim_k', type=int, default=10, help='Top-k for action masking / negatives')
        parser.add_argument('--usim_reward_weight', type=float, default=0.5, help='Weight in get_reward blend')
        parser.add_argument('--usim_reward_cost', type=float, default=0.2, help='Reward baseline cost')
        parser.add_argument('--usim_optimize_batch', type=int, default=20480, help='Transitions sampled per optimize()')
        parser.add_argument('--usim_buffer_max_len', type=int, default=1024, help='Replay buffer max length')
        parser.add_argument('--usim_content_hidden', type=int, default=150, help='content_mapper hidden dim')
        parser.add_argument('--usim_actor_weight_decay', type=float, default=1e-6, help='Actor Adam weight decay')
        parser.add_argument('--usim_critic_weight_decay', type=float, default=1e-6, help='Critic Adam weight decay')
        parser.add_argument(
            '--usim_mapper_ckpt', type=str, default='',
            help='Optional path to pretrained content_mapper state_dict (.pt), like official MLP_*.pt',
        )
    elif model_name == 'VBPR':
        parser.add_argument('--p_emb', type=list, default=[0.05, 0], help='lr and reg for id embeddings')
        parser.add_argument('--p_ctx', type=list, default=[0.05, 0.01], help='lr and reg for context features')
    elif model_name == 'AMR':
        parser.add_argument('--p_emb', type=list, default=[0.05, 0], help='lr and reg for id embeddings')
        parser.add_argument('--p_ctx', type=list, default=[0.05, 0.01], help='lr and reg for context features')
        parser.add_argument('--eps', type=float, default=0.1, help='epsilong for noises')
        parser.add_argument('--lmd', type=float, default=1, help='balance the adv')
    elif model_name == 'MTPR':
        parser.add_argument('--p_emb', type=list, default=[0.05, 0], help='lr and reg for id embeddings')
        parser.add_argument('--p_ctx', type=list, default=[0.05, 0.01], help='lr and reg for context features')
        parser.add_argument('--p_proj', type=list, default=[0.05, 0.01], help='lr and reg for wei only')
    elif model_name in available_models:
        pass
    else:
        raise ValueError(f"Invalid model name: {model_name}. "
                         f"Available models: {list(available_models.keys())}")

    return parser

