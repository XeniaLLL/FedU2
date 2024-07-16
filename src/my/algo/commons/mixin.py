from abc import abstractmethod
from typing import Any, Iterable

import torch
from sklearn.metrics import rand_score, normalized_mutual_info_score

from .np import cluster_accuracy_score


class EvalMetric:
    """Provide `eval_metric`."""
    def eval_metric(self, pred: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        y_pred = pred.cpu().numpy()
        y_true = labels.cpu().numpy()
        acc = cluster_accuracy_score(y_true, y_pred)
        ri = rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        return {'acc': acc, 'ri': ri, 'nmi': nmi}


class ServerTest:
    """Provide `test`."""
    clients: list

    @abstractmethod
    def eval_metric(self, pred: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
        raise NotImplementedError

    def test(self) -> dict[str, Any]:
        pred_list = []
        labels_list = []
        for client in self.clients:
            result = self.evaluate(client.test_loader)
            pred_list.append(result['pred'])
            labels_list.append(result['labels'])
        pred = torch.concat(pred_list)
        labels = torch.concat(labels_list)
        return self.eval_metric(pred, labels)


class ServerClientAvgTest:
    """Provide `average_test`."""
    clients: list

    def average_test(self, metrics=('acc', 'ri', 'nmi')) -> dict[str, Any]:
        clients_metric = [client.test() for client in self.clients]
        weights = [client.train_sample_num for client in self.clients]
        total_weights = sum(weights)

        def average_metric(name: str) -> float:
            total_metric = sum(x * w for x, w in zip([result[name] for result in clients_metric], weights))
            return total_metric / total_weights

        return {key: average_metric(key) for key in metrics}
