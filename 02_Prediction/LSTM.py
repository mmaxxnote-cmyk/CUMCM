import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """用于单变量时间序列预测的 LSTM 网络。"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


class LSTMForecaster:
    """单变量 LSTM 数据准备、训练、预测和评估流程。

    默认使用 CPU。fit 后可用 predict_test 获取测试集结果，
    或用 forecast 递推预测未来若干期。
    """

    def __init__(
        self,
        window_size=5,
        hidden_size=64,
        num_layers=1,
        random_state=42,
    ):
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size 必须是正整数")
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.random_state = random_state
        self.device = torch.device("cpu")

        self.model = None
        self.data_min_ = None
        self.data_scale_ = None
        self.series_ = None
        self.test_x_ = None
        self.test_y_ = None
        self.history_ = []

    def fit(
        self,
        series,
        train_ratio=0.8,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        verbose=False,
    ):
        """准备数据并训练模型，返回自身。

        参数:
            series: 一维时间序列。
            train_ratio: 监督样本中训练集所占比例。
            epochs: 训练轮数。
            batch_size: 批大小。
            learning_rate: Adam 学习率。
            verbose: 是否定期打印训练损失。
        """
        values = self._validate_series(series)
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio 必须在 0 和 1 之间")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs 必须是正整数")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")

        sample_count = len(values) - self.window_size
        train_size = int(sample_count * train_ratio)
        if train_size < 1 or train_size >= sample_count:
            raise ValueError("当前序列长度和 train_ratio 无法同时形成训练集与测试集")

        # 仅使用训练区间估计归一化参数，避免测试信息泄漏。
        train_end = train_size + self.window_size
        train_values = values[:train_end]
        self.data_min_ = float(np.min(train_values))
        data_max = float(np.max(train_values))
        self.data_scale_ = data_max - self.data_min_
        if np.isclose(self.data_scale_, 0.0):
            self.data_scale_ = 1.0

        normalized = self._normalize(values)
        x, y = create_sequences(normalized, self.window_size)
        train_x, test_x = x[:train_size], x[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]

        train_dataset = TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
        )
        loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_dataset)),
            shuffle=True,
        )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history_ = []

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                prediction = self.model(batch_x)
                loss = criterion(prediction, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            epoch_loss = total_loss / len(train_dataset)
            self.history_.append(epoch_loss)
            if verbose and (
                epoch == 0
                or (epoch + 1) % max(1, epochs // 10) == 0
                or epoch + 1 == epochs
            ):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

        self.series_ = values
        self.test_x_ = torch.from_numpy(test_x)
        self.test_y_ = test_y.reshape(-1)
        return self

    def predict_test(self):
        """预测测试集，返回真实值、预测值和 MAE/RMSE/R²。"""
        self._check_fitted()
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.test_x_.to(self.device))
        y_pred = self._inverse(prediction.cpu().numpy().reshape(-1))
        y_true = self._inverse(self.test_y_)
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "metrics": regression_metrics(y_true, y_pred),
        }

    def forecast(self, steps):
        """使用最后一个窗口递推预测未来 steps 期。"""
        self._check_fitted()
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps 必须是正整数")

        window = list(self._normalize(self.series_[-self.window_size:]))
        forecasts = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                x = torch.tensor(
                    window[-self.window_size:],
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(1, self.window_size, 1)
                next_value = float(self.model(x).item())
                forecasts.append(next_value)
                window.append(next_value)
        return self._inverse(np.asarray(forecasts))

    def predict(self, steps):
        """统一预测接口：递推预测未来 steps 期。"""
        return self.forecast(steps)

    def evaluate(self, y_true, y_pred):
        """根据真实值和预测值计算回归评价指标。"""
        return regression_metrics(y_true, y_pred)

    def _validate_series(self, series):
        values = np.asarray(series, dtype=float)
        if values.ndim != 1:
            raise ValueError("series 必须是一维时间序列")
        if len(values) <= self.window_size + 1:
            raise ValueError("序列长度必须大于 window_size + 1")
        if not np.all(np.isfinite(values)):
            raise ValueError("series 不能包含 NaN 或无穷值")
        return values.copy()

    def _normalize(self, values):
        return (np.asarray(values, dtype=float) - self.data_min_) / self.data_scale_

    def _inverse(self, values):
        return np.asarray(values, dtype=float) * self.data_scale_ + self.data_min_

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("请先调用 fit() 训练模型")


def create_sequences(series, window_size):
    """将归一化序列转换为滑动窗口监督学习样本。"""
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size 必须是正整数")
    if len(values) <= window_size:
        raise ValueError("序列长度必须大于 window_size")

    x, y = [], []
    for index in range(len(values) - window_size):
        x.append(values[index:index + window_size])
        y.append(values[index + window_size])
    return (
        np.asarray(x, dtype=np.float32).reshape(-1, window_size, 1),
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
    )


def regression_metrics(y_true, y_pred):
    """计算 MAE、RMSE 和 R²。"""
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape or len(true) == 0:
        raise ValueError("y_true 与 y_pred 必须是等长非空序列")

    residual = true - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = np.sum((true - np.mean(true)) ** 2)
    r2 = float("nan") if np.isclose(denominator, 0.0) else float(
        1 - np.sum(residual ** 2) / denominator
    )
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_lstm(series, **kwargs):
    """便捷训练接口。

    模型结构参数传给 LSTMForecaster，其余参数传给 fit。
    返回模型对象以及测试集真实值、预测值和评价指标。
    """
    model_keys = {"window_size", "hidden_size", "num_layers", "random_state"}
    model_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in model_keys}
    forecaster = LSTMForecaster(**model_kwargs).fit(series, **kwargs)
    result = forecaster.predict_test()
    result["model"] = forecaster
    return result
