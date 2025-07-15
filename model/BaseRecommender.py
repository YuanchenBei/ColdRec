import sys
sys.path.append('..')
from util.databuilder import DataBuilder, ColdStartDataBuilder
from util.operator import find_k_largest, batch_find_k_largest
from util.evaluator import ranking_evaluation
import time


class BaseColdStartTrainer(object):
    def __init__(self, args, training_set, warm_valid_set, cold_valid_set, overall_valid_set,
                 warm_test_set, cold_test_set, overall_test_set, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx,
                 device, user_content=None, item_content=None, **kwargs):
        super(BaseColdStartTrainer, self).__init__()
        self.args = args
        self.data = ColdStartDataBuilder(training_set, warm_valid_set, cold_valid_set, overall_valid_set,
                                         warm_test_set, cold_test_set, overall_test_set, user_num, item_num,
                                         warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx,
                                         user_content, item_content)
        self.bestPerformance = []
        top = args.topN.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.emb_size = args.emb_size
        self.maxEpoch = args.epochs
        self.batch_size = args.bs
        self.lr = args.lr
        self.reg = args.reg
        self.device = device
        self.result = []
        self.early_stop_flag = False if args.early_stop == 0 else True
        if self.early_stop_flag:
            self.early_stop_patience = args.early_stop
            self.max_early_stop_patience = args.early_stop

    def print_basic_info(self):
        print('*' * 80)
        print('Model: ', self.model_name)
        print('Dataset: ', self.dataset_name)
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lr)
        print('Batch Size:', self.batch_size)
        print('*' * 80)

    def timer(self, start=True):
        if start:
            self.train_start_time = time.time()
        else:
            self.train_end_time = time.time()

    def train(self):
        pass

    def predict(self, u):
        pass

    def batch_predict(self, users):
        raise NotImplementedError("batch_predict is not implemented")

    def save(self):
        pass

    def batch_valid(self, valid_type='all'):
        if valid_type == 'warm':
            valid_set = self.data.warm_valid_set
        elif valid_type == 'cold':
            valid_set = self.data.cold_valid_set
        elif valid_type == 'all':
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError('Invalid valid type!')

        rec_list = {}
        batch_size = 10000
        for i in range(0, len(valid_set), batch_size):
            batch_users = list(valid_set.keys())[i:i + batch_size]
            batch_candidates = self.batch_predict(batch_users)
            for j, user in enumerate(batch_users):
                candidates = batch_candidates[j]
                rated_list = list(self.data.training_set_u[user].keys())
                if len(rated_list) != 0:
                    candidates[self.data.get_item_id_list(rated_list)] = -10e8

            if valid_type == 'warm' and self.args.cold_object == 'item':
                batch_candidates[:, self.data.mapped_cold_item_idx] = -10e8
            elif valid_type == 'cold' and self.args.cold_object == 'item':
                batch_candidates[:, self.data.mapped_warm_item_idx] = -10e8

            batch_ids, batch_scores = batch_find_k_largest(self.max_N, batch_candidates)

            for user, ids, scores in zip(batch_users, batch_ids, batch_scores, strict=True):
                item_names = [self.data.id2item[iid] for iid in ids]
                rec_list[user] = list(zip(item_names, scores, strict=True))
        return rec_list

    def valid(self, valid_type='all'):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        if valid_type == 'warm':
            valid_set = self.data.warm_valid_set
        elif valid_type == 'cold':
            valid_set = self.data.cold_valid_set
        elif valid_type == 'all':
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError('Invalid valid type!')

        rec_list = {}
        user_count = len(valid_set)
        for i, user in enumerate(valid_set):
            candidates = self.predict(user)
            rated_list, li = self.data.user_rated(user)
            if len(rated_list) != 0:
                candidates[self.data.get_item_id_list(rated_list)] = -10e8
            if valid_type == 'warm' and self.args.cold_object == 'item':
                candidates[self.data.mapped_cold_item_idx] = -10e8
            if valid_type == 'cold' and self.args.cold_object == 'item':
                candidates[self.data.mapped_warm_item_idx] = -10e8

            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def test(self, test_type='warm'):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        if test_type == 'warm':
            test_set = self.data.warm_test_set
        elif test_type == 'cold':
            test_set = self.data.cold_test_set
        elif test_type == 'all':
            test_set = self.data.overall_test_set
        else:
            raise ValueError('Invalid test type!')

        rec_list = {}
        user_count = len(test_set)
        for i, user in enumerate(test_set):
            candidates = self.predict(user)
            rated_list, li = self.data.user_rated(user)
            if len(rated_list) != 0:
                candidates[self.data.get_item_id_list(rated_list)] = -10e8
            if test_type == 'warm' and self.args.cold_object == 'item':
                candidates[self.data.mapped_cold_item_idx] = -10e8
            if test_type == 'cold' and self.args.cold_object == 'item':
                candidates[self.data.mapped_warm_item_idx] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def full_evaluation(self, rec_list, test_type='warm'):
        if test_type == 'warm':
            test_set = self.data.warm_test_set
        elif test_type == 'cold':
            test_set = self.data.cold_test_set
        elif test_type == 'all':
            test_set = self.data.overall_test_set
        else:
            raise ValueError('Invalid evaluation type!')
        self.result, test_performance = ranking_evaluation(test_set, rec_list, self.topN)
        if test_type == 'warm':
            self.warm_test_results = test_performance
        elif test_type == 'cold':
            self.cold_test_results = test_performance
        elif test_type == 'all':
            self.overall_test_results = test_performance
        print('*' * 80)
        print(f'[{test_type} setting] The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def fast_evaluation(self, epoch, valid_type='all'):
        if valid_type == 'warm':
            valid_set = self.data.warm_valid_set
        elif valid_type == 'cold':
            valid_set = self.data.cold_valid_set
        elif valid_type == 'all':
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError('Invalid evaluation type!')
        print(f'Evaluating the model under the {valid_type} setting...')
        try:
            rec_list = self.batch_valid(valid_type)
        except NotImplementedError:
            rec_list = self.valid(valid_type)
        measure, _ = ranking_evaluation(valid_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
                if self.early_stop_flag:
                    self.early_stop_patience = self.max_early_stop_patience
            else:
                if self.early_stop_flag:
                    self.early_stop_patience -= 1
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Performance ' + ' (Top-' + str(self.max_N) + ' Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print(f'*Best {valid_type} Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        if self.early_stop_flag:
            if self.early_stop_patience <= 0:
                print(f"Stopping early at epoch {epoch + 1}.")
            else:
                print(f"Early stopping patience left: {self.early_stop_patience}.")
        print('-' * 120)
        return measure

    def run(self):
        self.print_basic_info()
        print('Training Model...')
        self.train()
        print('*' * 80)
        print('Testing under [all] setting...')
        all_rec_list = self.test(test_type='all')
        print('Evaluating under [all] setting...')
        self.full_evaluation(all_rec_list, test_type='all')
        print('*' * 80)
        print('Testing under [cold] setting...')
        cold_rec_list = self.test(test_type='cold')
        print('Evaluating under [cold] setting...')
        self.full_evaluation(cold_rec_list, test_type='cold')
        print('*' * 80)
        print('Testing under [warm] setting...')
        warm_rec_list = self.test(test_type='warm')
        print('Evaluating under [warm] setting...')
        self.full_evaluation(warm_rec_list, test_type='warm')
