"""
Author: Chi-An Chen
Date: 2025-12-02
Version: 2.1
Description:
"""
import torch
import torch.nn as nn

# ==================== 蒸餾損失函數 ====================
class DistillationLoss(nn.Module):
    """
    結合 Soft Loss (KL Divergence) 和 Hard Loss (CrossEntropy)
    """
    def __init__(self, temperature=4.0, alpha=0.8):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft Loss: KL Divergence between softened outputs
        soft_student = nn.functional.log_softmax(
            student_logits / self.temperature, dim=1
        )
        soft_teacher = nn.functional.softmax(
            teacher_logits / self.temperature, dim=1
        )
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard Loss: CrossEntropy with ground truth
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 結合兩種損失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss.item(), hard_loss.item()