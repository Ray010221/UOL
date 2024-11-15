import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from PohangDataSet import PohangDataset
from Model.CCM import ccm_calculation, ccm_entropy_loss
from Model.UOL import UOL

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


class Args:
    def __init__(self) -> None:
        self.batch_size = 8
        self.lr = 0.0001
        self.epochs = 100
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = 6
        self.tau = 3
        self.dataset_root = "./dataset/korea/train"


def train():
    train_dataset = PohangDataset(root=args.dataset_root, isTrain=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    model = UOL(num_classes=args.num_classes).to(args.device)
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        with tqdm(train_dataloader) as t:
            for idx, (sar, opt, label) in enumerate(t):
                sar = sar.to(args.device)
                opt = opt.to(args.device)
                label = label.to(args.device)

                optimizer.zero_grad()

                outputs, sar_classification, opt_classification = model(sar, opt)

                mask_sar, mask_opt = ccm_calculation(label, args.num_classes, sar_classification, opt_classification,
                                                     args.tau)

                loss1 = ce_loss(outputs, label)
                loss2 = ccm_entropy_loss(sar_classification, label, args.num_classes, mask_sar)
                loss3 = ccm_entropy_loss(opt_classification, label, args.num_classes, mask_opt)
                loss = loss1 + loss2 + loss3
                loss.backward()
                optimizer.step()

    torch.save(model.state_dict(), 'final_UOL_Pohang.pth')


if __name__ == '__main__':
    args = Args()
    train()

