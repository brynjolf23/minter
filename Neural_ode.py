import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

import os

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt



def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv('Monthly_passengers.csv')
rows, columns = df.shape

df.Month = pd.to_datetime(df.Month)

df = df.set_index("Month")
df.head()
df.index.freq = 'MS'


train_data = df[:len(df)-12]
test_data = df[len(df)-12:]


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, default='adams')
parser.add_argument('--data_size', type=int, default=rows)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--niters', type=int, default=rows)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=float, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([df['Frequency'][0]])
true_y0 = true_y0.type(torch.FloatTensor) 

t = torch.linspace(0., rows, rows)

true_y_list = df['Frequency'].values

tensor_arr = []
for i in true_y_list:
    tensor_arr.append([i])

true_y = torch.tensor(tensor_arr)


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_y0 = batch_y0.type(torch.FloatTensor) 
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


# the ode function
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),  # 2 changed to 1
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)
    

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()
    
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    neural_ode_loss = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        neural_ode_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))

        end = time.time()

print(pred_y)

ode_predictions = pred_y.data.numpy()
print(ode_predictions)


test_data['NODE_Predictions'] = ode_predictions
print(test_data)


#--------------ERROR
node_rmse_error = rmse(test_data['Frequency'], test_data["NODE_Predictions"])
node_mse_error = node_rmse_error**2
node_mae_error = mean_absolute_error(test_data['Frequency'],test_data["NODE_Predictions"])
node_mape_error = mean_absolute_percentage_error(test_data['Frequency'],test_data["NODE_Predictions"])


print(f'MSE Error: {node_mse_error}\nRMSE Error: {node_rmse_error}\nMAE: {node_mae_error}\nMAPE: {node_mape_error}')

