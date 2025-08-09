import sys
sys.path.append('..')
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import numpy as np
from util.evaluator import ranking_evaluation
import time
from util.utils import process_bar
import logging

logger = logging.getLogger(__name__)

class BaseColdStartTrainer(ABC):
    """
    Abstract base class for all cold-start recommendation model trainers.
    
    This class defines the contract that all recommendation model implementations
    must follow, ensuring consistent interface and behavior across different models.
    """
    
    def __init__(self, config):
        super(BaseColdStartTrainer, self).__init__()
        self.config = config
        self.args = config.args
        self.data = config.data
        self.bestPerformance = []
        top = self.args.topN.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.model_name = self.args.model
        self.dataset_name = self.args.dataset
        self.emb_size = self.args.emb_size
        self.maxEpoch = self.args.epochs
        self.batch_size = self.args.bs
        self.lr = self.args.lr
        self.reg = self.args.reg
        self.device = self.config.device
        self.result = []
        self.early_stop_flag = False if self.args.early_stop == 0 else True
        if self.early_stop_flag:
            self.early_stop_patience = self.args.early_stop
            self.max_early_stop_patience = self.args.early_stop

    def print_basic_info(self):
        logger.info('*' * 80)
        logger.info(f'Model: {self.model_name}')
        logger.info(f'Dataset: {self.dataset_name}')
        logger.info(f'Embedding Dimension: {self.emb_size}')
        logger.info(f'Maximum Epoch: {self.maxEpoch}')
        logger.info(f'Learning Rate: {self.lr}')
        logger.info(f'Batch Size: {self.batch_size}')
        logger.info('*' * 80)

    def timer(self, start=True):
        if start:
            self.train_start_time = time.time()
        else:
            self.train_end_time = time.time()

    @abstractmethod
    def train(self) -> None:
        """
        Train the recommendation model.
        
        This method should implement the training loop for the specific model.
        It should handle model training, validation, and early stopping.
        """
        pass

    @abstractmethod
    def predict(self, u: Union[str, int]) -> np.ndarray:
        """
        Predict scores for a single user.
        
        Args:
            u: User identifier (string or integer)
            
        Returns:
            numpy.ndarray: Prediction scores for all items, shape (item_num,)
        """
        pass

    @abstractmethod
    def batch_predict(self, users: List[Union[str, int]]) -> Union[np.ndarray, torch.Tensor]:
        """
        Predict scores for multiple users in batch.
        
        Args:
            users: List of user identifiers
            
        Returns:
            Union[np.ndarray, torch.Tensor]: Prediction scores for all users and items,
                                           shape (num_users, item_num)
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Save the current best model state.
        
        This method should save the best performing model embeddings or parameters.
        """
        pass

    def _evaluate(self, data_set: Dict, data_type: str = 'all') -> Dict[str, List[Tuple[str, float]]]:
        """
        Evaluate the model on a given dataset using batch prediction.
        
        Args:
            data_set: Dictionary mapping users to their test items
            data_type: Type of evaluation ('warm', 'cold', or 'all')
            
        Returns:
            Dictionary mapping users to their recommended items with scores
        """
        rec_list = {}
        batch_size = self.batch_size
        for i in range(0, len(data_set), batch_size):
            batch_users = list(data_set.keys())[i:i + batch_size]
            batch_candidates = self.batch_predict(batch_users)
            for j, user in enumerate(batch_users):
                candidates = batch_candidates[j]
                rated_list = list(self.data.training_set_u[user].keys())
                if len(rated_list) != 0:
                    candidates[self.data.get_item_id_list(rated_list)] = -10e8

            if data_type == 'warm' and self.args.cold_object == 'item':
                batch_candidates[:, self.data.mapped_cold_item_idx] = -10e8
            elif data_type == 'cold' and self.args.cold_object == 'item':
                batch_candidates[:, self.data.mapped_warm_item_idx] = -10e8

            scores, indices = torch.topk(batch_candidates, self.max_N, dim=1, largest=True, sorted=True)
            batch_ids, batch_scores = indices.cpu().numpy(), scores.cpu().numpy()

            for user, ids, scores in zip(batch_users, batch_ids, batch_scores, strict=True):
                item_names = [self.data.id2item[iid] for iid in ids]
                rec_list[user] = list(zip(item_names, scores, strict=True))
        return rec_list

    def valid(self, valid_type: str = 'all') -> Dict[str, List[Tuple[str, float]]]:
        """
        Perform validation using the specified validation set.
        
        Args:
            valid_type: Type of validation ('warm', 'cold', or 'all')
            
        Returns:
            Dictionary mapping users to their recommended items with scores
        """
        if valid_type == 'warm':
            valid_set = self.data.warm_valid_set
        elif valid_type == 'cold':
            valid_set = self.data.cold_valid_set
        elif valid_type == 'all':
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError('Invalid valid type!')
        return self._evaluate(valid_set, valid_type)

    def test(self, test_type: str = 'all') -> Dict[str, List[Tuple[str, float]]]:
        """
        Perform testing using the specified test set.
        
        Args:
            test_type: Type of testing ('warm', 'cold', or 'all')
            
        Returns:
            Dictionary mapping users to their recommended items with scores
        """
        if test_type == 'warm':
            test_set = self.data.warm_test_set
        elif test_type == 'cold':
            test_set = self.data.cold_test_set
        elif test_type == 'all':
            test_set = self.data.overall_test_set
        else:
            raise ValueError('Invalid test type!')
        return self._evaluate(test_set, test_type)

    def full_evaluation(self, rec_list: Dict[str, List[Tuple[str, float]]], test_type: str = 'warm') -> None:
        """
        Perform full evaluation and print results.
        
        Args:
            rec_list: Dictionary mapping users to their recommended items with scores
            test_type: Type of evaluation ('warm', 'cold', or 'all')
        """
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
        logger.info('*' * 80)
        logger.info(f"[{test_type} setting] The result of {self.model_name}:\n{''.join(self.result)}")

    def fast_evaluation(self, epoch: int, valid_type: str = 'all') -> List[str]:
        """
        Perform fast evaluation during training.
        
        Args:
            epoch: Current training epoch
            valid_type: Type of validation ('warm', 'cold', or 'all')
            
        Returns:
            List of performance metrics
        """
        if valid_type == 'warm':
            valid_set = self.data.warm_valid_set
        elif valid_type == 'cold':
            valid_set = self.data.cold_valid_set
        elif valid_type == 'all':
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError('Invalid evaluation type!')
        logger.info(f'Evaluating the model under the {valid_type} setting...')
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
        logger.info('-' * 120)
        logger.info('Performance ' + ' (Top-' + str(self.max_N) + ' Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        logger.info('*Current Performance*')
        logger.info(f"Epoch:{epoch + 1}, {'  |  '.join(measure)}")
        bp = ''
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        logger.info(f'*Best {valid_type} Performance* ')
        logger.info(f'Epoch:{str(self.bestPerformance[0])}, {bp}')
        if self.early_stop_flag:
            if self.early_stop_patience <= 0:
                logger.info(f"Stopping early at epoch {epoch + 1}.")
            else:
                logger.info(f"Early stopping patience left: {self.early_stop_patience}.")
        logger.info('-' * 120)
        return measure

    def run(self) -> None:
        """
        Run the complete training and evaluation pipeline.
        
        This method orchestrates the entire training process including
        training, testing on different settings, and evaluation.
        """
        self.print_basic_info()
        logger.info('Training Model...')
        self.train()
        for test_type in ['all', 'cold', 'warm']:
            logger.info('*' * 80)
            logger.info(f'Testing under [{test_type}] setting...')
            rec_list = self.test(test_type=test_type)
            logger.info(f'Evaluating under [{test_type}] setting...')
            self.full_evaluation(rec_list, test_type=test_type)
