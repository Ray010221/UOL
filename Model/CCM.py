import torch
import torch.nn.functional as F


def ccm_calculation(label, num_classes, sar_classification, opt_classification, tau):
    label_one_hot = F.one_hot(label.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2)
    
    score_sar = torch.sum(F.softmax(sar_classification / tau, dim=1) * label_one_hot, dim=1)
    score_opt = torch.sum(F.softmax(opt_classification / tau, dim=1) * label_one_hot, dim=1)

    ratio_sar = score_sar - score_opt
    ratio_opt = -ratio_sar

    mask_sar = 1 - torch.where(ratio_sar > 0, ratio_sar, torch.tensor(0., device=ratio_sar.device))
    mask_opt = 1 - torch.where(ratio_opt > 0, ratio_opt, torch.tensor(0., device=ratio_opt.device))

    return mask_sar, mask_opt


def ccm_entropy_loss(outputs, label, num_classes, ccm):
    log_probs = F.log_softmax(outputs, dim=1)
    label_one_hot = F.one_hot(label.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2)
    cross_loss = label_one_hot * log_probs
    cross_loss = torch.sum(cross_loss, dim=1) * ccm
    loss = -torch.mean(cross_loss)

    return loss
