import argparse
import os
import os.path as osp

import torch
from torch import optim
from torchvision import transforms

from datasets import get_dataloader
from models.loss import DetectionCriterion
from models.model import DetectionModel
from pathlib import Path


def save_model(state, epoch):
    save_path = "weights"
    # check if the save directory exists
    if not Path(save_path).exists():
        Path(save_path).mkdir()

    # save with .pth extenstion
    save_path1 = Path(save_path, "checkpoint_{0}.pth".format(epoch+1))
    torch.save(state, str(save_path1))

    # save with .pkl extension
    save_path2 = Path(save_path, "checkpoint_{0}.pkl".format(epoch+1))
    torch.save(state, str(save_path2))


def train(model, loss_fn, optimizer, train_loader, epoch, device):

    model = model.to(device)
    model.train()

    for idx, (img, class_map, regression_map) in enumerate(train_loader):
        x = img.float().to(device)

        class_map_var = class_map.float().to(device)
        regression_map_var = regression_map.float().to(device)

        output = model(x)
        loss = loss_fn(output,
                       class_map_var, regression_map_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx*len(img) % (len(img)*10) == 0:
            print(f"Epoch {epoch}: [{idx*len(img)}/{len(train_loader.dataset)}]" +
                  '\tloss_cls: {loss_cls:.6f}'
                  '\tloss_reg: {loss_reg:.6f}'.format(loss_cls=loss_fn.class_average.average, loss_reg=loss_fn.reg_average.average))


    save_model({
        'epoch': epoch + 1,
        'batch_size': train_loader.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, epoch)


def main():
    # parse the arguments passed in
    parser = argparse.ArgumentParser()

    parser.add_argument("traindata")
    parser.add_argument("valdata")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset", default="WIDERFace")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    num_templates = 25  # aka the number of clusters

    # get the normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # load the data
    train_loader, _ = get_dataloader(args.traindata, args, num_templates,
                                     img_transforms=img_transforms)

    # create the model and the loss function
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
    for epoch in range(args.start_epoch, args.epochs):
        train(model, loss_fn, optimizer, train_loader, epoch, device=device)
        scheduler.step()


if __name__ == '__main__':
    main()
