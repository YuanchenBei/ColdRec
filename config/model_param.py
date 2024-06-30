def model_specific_param(model_name, parser):
    if model_name == 'KNN':
        parser.add_argument('--knn_num', type=int, default=5)
    elif model_name == 'ALDI':
        parser.add_argument('--alpha', type=float, default=0.9)
        parser.add_argument('--beta', type=float, default=0.05)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--tws', type=int, default=0, choices=[0, 1])
        parser.add_argument('--freq_coef_M', type=float, default=4)
    elif model_name == 'GAR':
        parser.add_argument('--alpha', type=float, default=0.05, help='GAR parameter alpha')
        parser.add_argument('--beta', type=float, default=0.1, help='GAR parameter beta')
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
        parser.add_argument('--pretrain', type=bool, default=False, help='user/item embedding pre-training')
        parser.add_argument('--pretrain_update', type=bool, default=False, help='u/i pretrain embedding update')
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
    else:
        raise Exception("The model name not found.")

    return parser

