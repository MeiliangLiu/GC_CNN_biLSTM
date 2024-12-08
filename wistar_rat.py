import os
import torch
import torch.nn as nn
import numpy as np
import random
import scipy.io as sio

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
    def __init__(self, input_size, M_hidden_size=600, L_hidden_size=1200, output_size=1):
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


def read_rat(name):

    base_path = r"E:\GC_CNN_biLSTM\Whisker Stimulated Wistar Rats"
    file_extension = ".mat"

    path = os.path.join(base_path, name + file_extension)

    mat = sio.loadmat(path)
    struct = (mat['data'])
    return struct


def train_rat(time, name, trial, num_epochs, learning_rate, lam):
    size = 15

    rat_data = read_rat(name)

    data = rat_data[:, :, trial]
    test_x = data[time+80:time+90, :]
    test_y = data[time+90:time+100, :]


    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()
    networks = [BiLSTM(input_size=size, M_hidden_size=600, L_hidden_size=600, output_size=1).cuda() for _ in
                range(size)]
    lstm_networks = nn.ModuleList(networks)
    models = lstm_networks
    loss_fn = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)  # Use AdamW optimizer
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

        if (epoch + 1) % 50 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], predict_loss: {predict_loss.item():.4f}, regularize_loss: {regularize_loss.item():.4f}')

        if (epoch + 1) % 3000 == 0:
            np.savetxt(
                f"time={time},name={name},trial={trial}, learning_rate={learning_rate}, lam={lam},epoch={epoch}.txt",
                GCs, fmt=f'%.5f')


if __name__ == '__main__':

    name_list = ["IC070523","RN060616A","RN060714C"]
    time_list = [200]
    for name in name_list:

        X = read_rat(name)
        size = torch.tensor(X).shape[2]

        for time in time_list:

            for trial in range(size):
                train_rat(time, name, trial, num_epochs=3000, learning_rate=0.0001, lam=0.001)
