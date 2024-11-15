import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import random
from synthetic import data_segmentation, simulate_lorenz_96
from ComputeROC import compute_roc

def regularize(network, lam, penalty):
    x = network.penalty_x
    t, p, lag = x.shape
    if penalty == 'GL':
        total_norm = torch.sum(torch.norm(x, dim=(0, 2)))
        return lam * total_norm
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(x, dim=(0, 2))) + torch.sum(torch.norm(network.penalty_x, dim=0)))
    elif penalty == 'H':
        total_norm = 0
        for i in range(lag):
            lag_x = x[:, :, :(i + 1)]
            lag_norm = torch.norm(lag_x, dim=(0, 2))
            total_norm += torch.sum(lag_norm)
        return lam * total_norm


class CNNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, pool_size=2, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.batch_norm = nn.BatchNorm1d(output_size)
        self.batch_norm2 = nn.BatchNorm1d(input_size)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        self.conv2 = nn.Conv1d(in_channels=output_size, out_channels=output_size, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=output_size, out_channels=input_size, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, seq_len)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.elu(x)

        x = self.conv3(x)
        x = self.batch_norm2(x)

        x = x.permute(0, 2, 1)  # Change back to (batch_size, seq_len, channels)
        x = x + residual
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_size, M_hidden_size, L_hidden_size, output_size=1):
        super(BiLSTM, self).__init__()
        self.forward_network = CNNBlock(input_size, M_hidden_size)
        self.lstm = nn.LSTM(input_size, L_hidden_size, bidirectional=True)
        self.fc = nn.Linear(L_hidden_size * 2, output_size)  # 双向LSTM的输出维度是L_hidden_size的两倍
        self.penalty_x = torch.Tensor()

    def forward(self, input):
        x = self.forward_network(input)
        self.penalty_x = x.view(input.shape[1], input.shape[2], input.shape[0])
        x = x * input
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output

    def GC(self, threshold=0, ignore_lag=True):
        GC = []
        weight_norm = torch.norm(self.penalty_x, dim=(0, 2))
        return weight_norm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_loz(F, P, num_epochs, learning_rate, lam):
    # 模拟洛伦兹系统数据
    X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 准备数据集
    X = torch.tensor(X, dtype=torch.float32, device=torch.device('cuda'))
    train_x, train_y, val_x, val_y = data_segmentation(data=X, lag=1, seg=1, val_rate=0.2)
    train_x = train_x.view(train_x.shape[2], train_x.shape[0], train_x.shape[1])  # 从(999 10 1)换成(1,999,10 )
    input_seq = train_x
    target_seq = train_y

    # 定义p个BiLSTM网络
    networks = []
    for _ in range(P):
        network = BiLSTM(input_size=P, M_hidden_size=600, L_hidden_size=1200, output_size=1).cuda()
        networks.append(network)
    lstm_networks = nn.ModuleList(networks)
    models = lstm_networks

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
    best_score = 0
    total_score = 0
    for epoch in range(num_epochs):
        losses = []
        # 前向传播,循环每个序列数据专用的网络
        for j in range(0, len(models)):
            network_output = models[j](input_seq).view(-1)
            loss_i = loss_fn(network_output, target_seq[:, j])
            losses.append(loss_i)
        predict_loss = sum(losses)
        # 再此加上惩罚损失
        regularize_loss = sum([regularize(model, lam, "GL") for model in models])

        sum_loss = predict_loss + regularize_loss
        # 反向传播和优化
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()

        # 计算因果分数
        GCs = []
        for i in range(P):
            GCs.append(models[i].GC().detach().cpu().numpy())
        GCs = np.array(GCs)
        score = compute_roc(GC, GCs, False)

        if best_score < score and score > 0.985:
            best_score = score
            np.savetxt("P=" + str(P) + "F=" + str(F) + ",true.txt", GC, fmt='%.5f')
            np.savetxt(
                f"P={P}F={F},score={score},learning_rate = {learning_rate}, lam = {lam},epoch={epoch}.txt",
                GCs, fmt=f'%.5f')
        # 打印训练信息
        if (epoch + 1) % 1 == 0:
            total_score += score
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], predict_loss: {predict_loss.item():.4f}, regularize_loss: {regularize_loss.item():.4f}, AUROC: {score:.4f}, avg_AUROC: {total_score / (epoch + 1):.4f}')

    return total_score / num_epochs


def grid_search(param_grid):
    results = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        avg_score = train_loz(F=10, P=10, num_epochs=8000, learning_rate=params['learning_rate'], lam=params['lam'])
        results.append((params, avg_score))

    best_params = max(results, key=lambda x: x[1])
    print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    return best_params


if __name__ == '__main__':
    set_seed(1)

    param_grid = {
        'learning_rate': [0.001],
        'lam': [0.00001]
        # 'learning_rate': [0.009],
        # 'lam': [0.0001]
    }
    best_lam = grid_search(param_grid)
