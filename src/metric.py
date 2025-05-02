                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         src/metric.py                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("TP", default=torch.zeros(num_classes), dist_reduce_fx="sum")  # True Positive
        self.add_state("FP", default=torch.zeros(num_classes), dist_reduce_fx="sum")  # False Positive
        self.add_state("FN", default=torch.zeros(num_classes), dist_reduce_fx="sum")  # False Negative

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        예측값(preds)과 실제값(target)을 받아서 True Positive (TP),
        False Positive (FP), False Negative (FN)를 업데이트
        """
        pred_labels = torch.argmax(preds, dim=1)  # 예측값을 클래스 레이블로 변환
        for i in range(self.num_classes):
            self.TP[i] += torch.sum((target == i) & (pred_labels == i))  # TP 계산
            self.FP[i] += torch.sum((target != i) & (pred_labels == i))  # FP 계산
            self.FN[i] += torch.sum((target == i) & (pred_labels != i))  # FN 계산

    def compute(self):
        """
        F1 Score 계산: F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        precision = self.TP / (self.TP + self.FP + 1e-8)  # 1e-8 to avoid division by zero
        recall = self.TP / (self.TP + self.FN + 1e-8)  # 1e-8 to avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # F1 Score 계산
        return f1  # 클래스별 F1 Score 반환

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        predicted = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert predicted.shape == target.shape, f"Shape mismatch: {predicted.shape} vs {target.shape}"

        # [TODO] Cound the number of correct prediction
        correct = (predicted == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()









