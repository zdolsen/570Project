import argparse
# from datetime import time
import time as time_

import os
import os.path as osp

import torch
from torch import optim
from torchvision import transforms

import trainer
from datasets import get_dataloader
from models.loss import DetectionCriterion
from models.model import DetectionModel


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("traindata")
    parser.add_argument("valdata")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset", default="WIDERFace")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save-every", default=10, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

def arguments2(i):
    parser = argparse.ArgumentParser()

    parser.add_argument("traindata")
    parser.add_argument("valdata")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset", default="WIDERFace")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--workers", default=i, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save-every", default=10, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

def main():
    args = arguments()

    num_templates = 25  # aka the number of clusters

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_loader, _ = get_dataloader(args.traindata, args, num_templates,
                                     img_transforms=img_transforms)

    model = DetectionModel(num_objects=1, num_templates=num_templates)
    loss_fn = DetectionCriterion(num_templates)

    # directory where we'll store model weights
    weights_dir = "weights"
    if not osp.exists(weights_dir):
        os.mkdir(weights_dir)

    # check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    print(f'Can I can use GPU now? -- {torch.cuda.is_available()}')

    optimizer = optim.SGD(model.learnable_parameters(args.lr), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=20,
                                          last_epoch=args.start_epoch-1)

    # train and evalute for `epochs`
    for num_workers in range(0, 10, 1): 
        args = arguments2(num_workers)
        train_loader, _ = get_dataloader(args.traindata, args, num_templates,
                                     img_transforms=img_transforms)
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        # sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
        start = int(time_.time())
        # for epoch in range(1, 1):
        for i, data in enumerate(train_loader):
            # print(i)
            if i == 20:
                break
            pass
            
        # end = time.time()
        end = int(time_.time())
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
        print()


if __name__ == '__main__':
    main()
