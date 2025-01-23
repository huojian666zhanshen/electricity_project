import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = '附件2-场站出力.csv'
data = pd.read_csv(file_path)
data = data.iloc[2:].reset_index(drop=True)  # 跳过前两行并重置索引
data.columns = ['_timestamp', 'Power_1']  # 重命名列

# 2. 转换时间戳为日期时间格式
data['_timestamp'] = pd.to_datetime(data['_timestamp'])

# 3. 分割数据集：前70%为训练集，后30%为测试集
split_index = int(len(data) * 0.7)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

# 4. 滑动窗口生成序列
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length, 1:].values)  # 取 Power_1 特征
        y.append(data.iloc[i+seq_length, 1])  # 假设预测目标为 Power_1
    return np.array(X), np.array(y)

seq_length = 20  # 滑动窗口长度
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# 5. 构建时间序列预测模型
class LSTMDNNQuantileModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, dnn_hidden_dim, output_dim, quantiles):
        super(LSTMDNNQuantileModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.dnn1 = nn.Linear(lstm_hidden_dim, dnn_hidden_dim)
        self.dnn2 = nn.Linear(dnn_hidden_dim, dnn_hidden_dim)
        self.fc = nn.ModuleList([nn.Linear(dnn_hidden_dim, output_dim) for _ in quantiles])
        self.quantiles = quantiles
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM 输出形状: (batch_size, seq_length, lstm_hidden_dim)
        x = x[:, -1, :]  # 取最后一个时间步的输出 (batch_size, lstm_hidden_dim)
        x = self.relu(self.dnn1(x))  # DNN 第 1 层
        x = self.relu(self.dnn2(x))  # DNN 第 2 层
        outputs = [fc(x) for fc in self.fc]  # 针对每个分位点计算输出
        return outputs

# 定义分位数回归损失函数
def quantile_loss(predictions, targets, quantile):
    errors = targets - predictions
    return torch.mean(torch.max((quantile - 1) * errors, quantile * errors))

# 6. 数据预处理
# 将 X_train 和 X_test 转换为浮点型，并确保其形状为 (batch_size, seq_length, input_size)
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)  # 输入特征数为 1
X_test = X_test.reshape(X_test.shape[0], seq_length, 1)

# 转换 y_train 和 y_test 为 float32 类型
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# 定义模型参数
input_dim = 1  # Power_1 作为唯一的输入特征
lstm_hidden_dim = 256  # LSTM 隐藏层维度
dnn_hidden_dim = 64  # DNN 隐藏层维度
output_dim = 1  # 预测一个值
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 分位点

model = LSTMDNNQuantileModel(input_dim, lstm_hidden_dim, dnn_hidden_dim, output_dim, quantiles)

# 7. 数据加载器
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 8. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. 训练模型
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = 0
        for i, quantile in enumerate(quantiles):
            loss += quantile_loss(outputs[i].squeeze(), y_batch, quantile)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
# 10. 测试模型并生成预测结果
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        mean_prediction = torch.mean(torch.stack(outputs), dim=0).squeeze()  # 计算分位点的均值
        predictions.extend(mean_prediction.tolist())
        actuals.extend(y_batch.tolist())

# 转换为 NumPy 数组
predictions = np.array(predictions)
actuals = np.array(actuals)

#傻瓜预测
def naive_forecast(train_series, test_series):
    """
    傻瓜预测：直接使用训练集前一个值作为预测值。
    """
    predictions0 = test_series[:-1]
    predictions1 = np.array(train_series[-1])
    predictions1 = predictions1.reshape(1)
    predictions = np.concatenate([predictions1, predictions0])
    return predictions

target_col = 'Power_1'
series = data[target_col]
series = np.array(series, dtype=np.float32)

# 拆分训练集和测试集（70%:30%）
split_idx = int(len(series) * 0.7)
train_series = series[:split_idx]
test_series = series[split_idx:]
naive_preds = naive_forecast(train_series, test_series)


# 11. 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='blue')
plt.plot( naive_preds, label='傻瓜预测')
plt.plot(predictions, label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted (with Quantile Regression)')
plt.xlabel('Time Steps')
plt.ylabel('Power_1 Value')
plt.legend()
plt.grid()
plt.show()

# 12. 计算误差值
differences = test_series[:len(naive_preds)] - naive_preds
sum1 = 0
for i in range(len(naive_preds)):
    if test_series[:len(naive_preds)][i] != 0:
        sum1 += abs(differences[i])
average1 = sum1 / len(naive_preds)
print(f"{average1}")

differences = actuals - predictions
sum1 = 0
for i in range(len(predictions)):
    if actuals[i] != 0:
        sum1 += abs(differences[i])
average2 = sum1 / len(predictions)
print(f"平均误差：{average2:.2f}")