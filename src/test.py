"""
模型测试脚本
"""
import torch
import numpy as np

from config import best_model
from dataset import OmniglotSet
from network import Resnet50
from utils import proj_param, set_env, get_data_loader


# 测评数据集准确率
def evaluate(model, data_loader, title, device):
    correct, cls_number = 0, np.zeros(10)

    for i, item in enumerate(data_loader):
        print(f'\r{title} [{i + 1}/{len(data_loader)}] 。。。', end='')
        x, y = item
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        y_hat = model(x)
        correct += torch.sum(y_hat.argmax(dim=1) == y)

    return f'{correct/len(data_loader.dataset) * 100:.2f}%'


def main(args):
    device = set_env(args.device)
    ks = args.kernel_size

    # 加载数据集
    train_set = OmniglotSet('train')
    test_set = OmniglotSet('test')
    train_loader, test_loader = get_data_loader(args, train_set, test_set)

    # 加载模型
    model = Resnet50(ks)
    state = torch.load(best_model(ks), map_location=device)
    model.load_state_dict(state['model_state'])
    model = model.to(device)

    # 进行测试
    model.eval()
    with torch.no_grad():
        # 测试测试集数据
        acc = evaluate(model, test_loader, f'测评测试集(ks={ks})', device)
        print(f'(ks={ks})测试集准确率为：{acc}')


if __name__ == '__main__':
    for k in [3, 5, 7, 9]:
        main(proj_param(k))
