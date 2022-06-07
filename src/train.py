"""
模型训练脚本
"""
import time
import torch
from tqdm import tqdm

from config import record_file
from dataset import OmniglotSet
from utils import load_model, save_model, proj_param, set_env, get_data_loader


# 训练每一个epoch
def train(epoch, data_loader, model, loss_fn, opt, device, epoch_num):
    avg_loss = 0

    model.train()
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epoch_num}', unit='it') as pbar:
        for n, item in enumerate(data_loader):
            opt.zero_grad()

            x, y = item
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)

            t_loss = loss_fn(y_hat, y)
            avg_loss = (avg_loss * n + t_loss.item()) / (n + 1)

            t_loss.backward()
            opt.step()

            pbar.set_postfix(**{'1.avg_loss': f'{avg_loss:.4f}',
                                '2.bat_loss': f'{t_loss:.4f}'})
            pbar.update()

    time.sleep(0.1)
    print('  Recalculate loss and accuracy ...', end='')
    train_loss, acc = calculate_statistic(data_loader, model, device, loss_fn)
    print(f'\r  Train loss={train_loss}, acc={acc}%')

    return train_loss, acc


def calculate_statistic(data_loader, model, device, loss_fn):
    avg_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for n, item in enumerate(data_loader):
            x, y = item
            x = x.to(device, dtype=torch.float32)
            y_hat = model(x)

            y = y.to(device, dtype=torch.long)
            loss = loss_fn(y_hat, y)
            avg_loss = (avg_loss * n + loss.item()) / (n + 1)
            correct += torch.sum(y_hat.argmax(dim=1) == y)

    acc = f'{correct / len(data_loader.dataset) * 100:.2f}'

    return round(avg_loss, 6), acc


def validation(data_loader, model, device, loss_fn):
    time.sleep(0.1)
    print(f'Validating: size={len(data_loader.dataset)} ...', end='')
    avg_loss, acc = calculate_statistic(data_loader, model, device, loss_fn)
    print(f'\rValidating: size={len(data_loader.dataset)}, loss={avg_loss:.5f}, acc={acc}%')
    time.sleep(0.1)

    return avg_loss, acc


def main(args):
    device = set_env(args.device)
    ks = args.kernel_size

    # 加载数据集
    train_set = OmniglotSet('train')
    valid_set = OmniglotSet('valid')
    train_loader, valid_loader = get_data_loader(args, train_set, valid_set)

    # 加载模型
    start_epoch, model, loss_fn, opt, best_loss = load_model(device, ks)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

    for epoch in range(start_epoch, args.epoch):
        # 训练
        train_loss, train_acc = train(epoch, train_loader, model, loss_fn, opt, device, args.epoch)
        # 验证
        test_loss, test_acc = validation(valid_loader, model, device, loss_fn)

        sched.step()

        # 保存模型
        save_model(epoch, model, opt, best_loss, test_loss, ks)
        best_loss = min(best_loss, test_loss)

        # 保存日志文件
        with open(record_file(ks), 'a') as f:
            f.write(f'[{epoch + 1}/{args.epoch}],{train_loss},{train_acc},{test_loss},{test_acc}\n')


if __name__ == '__main__':
    main(proj_param())
