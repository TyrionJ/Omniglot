"""
配置文件
"""

# 训练过程文件存储路径
check_folder = '../checkpoints'
# 保存训练最新模型参数
last_model = lambda ks: f'{check_folder}/ks{ks}/last_ks{ks}.pkt'
# 保存训练最优模型参数
best_model = lambda ks: f'{check_folder}/ks{ks}/best_ks{ks}.pkt'
# 日志文件
record_file = lambda ks: f'{check_folder}/ks{ks}/record_ks{ks}.csv'

# 数据文件
data_root = '../data'
train_file = f'{data_root}/train_list.txt'
valid_file = f'{data_root}/valid_list.txt'
