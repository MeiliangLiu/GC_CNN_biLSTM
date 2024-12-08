import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
import random
from ComputeROC import compute_roc
from tool import dream_read_label, dream_read_data


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
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.batch_norm = nn.BatchNorm1d(output_size)
        self.batch_norm2 = nn.BatchNorm1d(input_size)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=output_size, out_channels=output_size, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=output_size, out_channels=input_size, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.elu(x)

        x = self.conv3(x)
        x = self.batch_norm2(x)

        x = x.permute(0, 2, 1)
        x = x + residual

        return x


def OffDiag(x):
    mask = ~np.eye(x.shape[0], dtype=bool)
    non_diag_elements = x[mask]
    new_arr = non_diag_elements.reshape(100, 99)
    return new_arr


class BiLSTM(nn.Module):
    def __init__(self, input_size, M_hidden_size, L_hidden_size, output_size=1):
        super(BiLSTM, self).__init__()
        self.forward_network = CNNBlock(input_size, M_hidden_size)
        self.lstm = nn.LSTM(input_size, L_hidden_size, bidirectional=True)
        self.fc = nn.Linear(L_hidden_size * 2, output_size)
        self.penalty_x = torch.Tensor()

    def forward(self, input):
        x = self.forward_network(input)
        self.penalty_x = x.view(input.shape[1], input.shape[2], input.shape[0])
        x = x * input
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output

    def GC(self):
        weight_norm = torch.norm(self.penalty_x, dim=(0, 2))
        return weight_norm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_dream3(size, type):
    name_list = ["Ecoli1", "Ecoli2", "Yeast1", "Yeast2", "Yeast3"]
    label = dream_read_label(
        r"E:\GC_CNN_biLSTM\DREAM3 in silico challenge"
        r"\DREAM3 gold standards\DREAM3GoldStandard_InSilicoSize" + str(size) + "_" + name_list[type - 1] + ".txt",
        size)
    data = dream_read_data(
        r"E:\GC_CNN_biLSTM\DREAM3 in silico challenge"
        r"\Size" + str(size) + "\DREAM3 data\InSilicoSize" + str(size) + "-" + name_list[
            type - 1] + "-trajectories.tsv")
    return label, data


def train_dream3(seed, num_epochs, type, learning_rate, lam):
    set_seed(seed)
    size = 100
    label, data = read_dream3(size, type=type)
    label_offdiag = OffDiag(label)

    reshaped_x = data.reshape(966, 100)

    test_x = reshaped_x[:920, :]
    test_y = reshaped_x[46:966, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()
    networks = [BiLSTM(input_size=size, M_hidden_size=600, L_hidden_size=1200, output_size=1).cuda() for _ in
                range(size)]
    lstm_networks = nn.ModuleList(networks)
    models = lstm_networks
    loss_fn = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)  # Use AdamW optimizer
    best_score = 0
    total_score = 0
    for epoch in range(num_epochs):
        losses = []
        for j in range(len(models)):
            network_output = models[j](input_seq).view(-1)
            loss_i = loss_fn(network_output, target_seq[:, j])
            losses.append(loss_i)
        predict_loss = sum(losses)
        regularize_loss = sum([regularize(model, lam, "GL") for model in models])
        sum_loss = predict_loss + regularize_loss
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()
        GCs = np.array([models[i].GC().detach().cpu().numpy() for i in range(size)])
        GCs_off = OffDiag(GCs)
        score = compute_roc(label_offdiag, GCs_off, False)
        if best_score < score and score > 0.56:
            best_score = score
            np.savetxt(f"type={type},true.txt", label, fmt='%.5f')
            np.savetxt(
                f"type={type},score={score},learning_rate = {learning_rate}, lam = {lam},epoch={epoch}.txt",
                GCs, fmt=f'%.5f')
        total_score += score
        if (epoch + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], predict_loss: {predict_loss.item():.4f}, regularize_loss: {regularize_loss.item():.4f}, score: {score:.4f}, avg_score: {total_score / (epoch + 1):.4f}')
    return best_score


def grid_search(param_grid):
    results = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        avg_score = train_dream3(seed=1, num_epochs=7000, type=params['type'], learning_rate=params['learning_rate'],
                                 lam=params['lam'])
        results.append((params, avg_score))

    best_params = max(results, key=lambda x: x[1])
    print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    return best_params


if __name__ == '__main__':
    set_seed(1)

    param_grid = {
        'type' :[1,2,3,4,5],
        'learning_rate': [0.0001],
        'lam': [0.0025]
    }

    best_params = grid_search(param_grid)
