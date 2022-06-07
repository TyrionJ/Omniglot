import os
import torch
import time
import argparse
from torch import optim
from torch.utils.data import DataLoader

from config import last_model, best_model
from src.network import Resnet50


def set_env(device):
    torch.set_default_dtype(torch.float32)
    if device != 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        return torch.device('cuda')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return torch.device('cpu')


# 获取torch dataloader
def get_data_loader(args, train_set, test_set, shuffle=True):
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=shuffle,
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=args.batch_size
    )

    return train_loader, test_loader


# 项目运行参数
def proj_param(k=5):
    parser = argparse.ArgumentParser(description='Omniglot')
    parser.add_argument('-d', '--device', default='0', type=str)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-k', '--kernel_size', default=k, type=int, choices=[3, 5, 7, 9])

    return parser.parse_args()


# 保存模型参数
def save_model(epoch, model, opt, best_loss, test_loss, ks):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'loss': min(best_loss, test_loss),
    }
    torch.save(state, last_model(ks))

    if test_loss <= best_loss:
        torch.save(state, best_model(ks))


# 加载模型参数
def load_model(device, ks):
    epoch, best_loss = 0, 9999
    # 定义损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    model = Resnet50(ks)
    # 定义优化器
    opt = optim.SGD(model.parameters(), lr=5e-3)

    if os.path.exists(last_model(ks)):
        print('Loading model ...')
        state = torch.load(last_model(ks), map_location=device)
        epoch = state['epoch'] + 1
        best_loss = state['loss']
        model.load_state_dict(state['model_state'])
        opt.load_state_dict(state['optimizer_state'])

        print('  ' + last_model(ks) + ' loaded.')
        print(f'  epoch: {epoch}')

    if os.path.exists(best_model(ks)):
        state = torch.load(best_model(ks), map_location=device)
        best_loss = state['loss']
        print(f'  best loss: {best_loss:.5f}')
    
    model = model.to(device)
    time.sleep(0.5)
    return epoch, model, loss_fn, opt, best_loss
