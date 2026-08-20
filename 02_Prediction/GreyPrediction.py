import numpy as np


class GM11:
    """GM(1,1) 灰色预测模型。

    适用于样本较少、数值为正且总体呈指数趋势的一维序列。
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.x0 = None
        self.fitted_ = None
        self.residuals_ = None
        self.relative_errors_ = None
        self.level_ratio_passed_ = None

    def fit(self, data):
        """拟合 GM(1,1)，并计算训练区间拟合值与残差。"""
        x0 = self._validate_series(data)
        self.x0 = x0
        self.level_ratio_passed_ = self.level_ratio_test(x0)

        # 1-AGO 累加生成与紧邻均值序列
        x1 = np.cumsum(x0)
        z1 = 0.5 * (x1[:-1] + x1[1:])

        # 最小二乘估计发展系数 a 和灰作用量 b
        B = np.column_stack((-z1, np.ones(len(z1))))
        self.a, self.b = np.linalg.lstsq(B, x0[1:], rcond=None)[0]

        self.fitted_ = self._restore(len(x0))
        self.residuals_ = x0 - self.fitted_
        self.relative_errors_ = np.abs(self.residuals_ / x0)
        return self

    def time_response(self, points):
        """返回从 k=0 开始的 1-AGO 时间响应序列。"""
        self._check_fitted()
        if not isinstance(points, (int, np.integer)) or points < 1:
            raise ValueError("points 必须是正整数")

        k = np.arange(points, dtype=float)
        if np.isclose(self.a, 0.0):
            return self.x0[0] + self.b * k
        return (
            (self.x0[0] - self.b / self.a) * np.exp(-self.a * k)
            + self.b / self.a
        )

    def fitted_values(self):
        """返回训练区间的拟合值。"""
        self._check_fitted()
        return self.fitted_.copy()

    def predict(self, steps):
        """预测训练序列之后的未来 steps 期原始序列值。"""
        self._check_fitted()
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps 必须是正整数")
        restored = self._restore(len(self.x0) + steps)
        return restored[len(self.x0):]

    def evaluate_fitted(self):
        """返回训练区间残差、相对误差和常用评价指标。"""
        self._check_fitted()
        mae = float(np.mean(np.abs(self.residuals_)))
        rmse = float(np.sqrt(np.mean(self.residuals_ ** 2)))
        mean_relative_error = float(np.mean(self.relative_errors_))
        return {
            "residuals": self.residuals_.copy(),
            "relative_errors": self.relative_errors_.copy(),
            "MAE": mae,
            "RMSE": rmse,
            "mean_relative_error": mean_relative_error,
            "accuracy": self._accuracy_grade(mean_relative_error),
            "level_ratio_passed": self.level_ratio_passed_,
        }

    def evaluate(self, y_true, y_pred):
        """根据真实值和预测值计算 MSE 和 R²。"""
        true = np.asarray(y_true, dtype=float).reshape(-1)
        pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if true.shape != pred.shape or len(true) == 0:
            raise ValueError("y_true 与 y_pred 必须是等长非空序列")

        residual = true - pred
        mse = float(np.mean(residual ** 2))
        denominator = np.sum((true - np.mean(true)) ** 2)
        r2 = float("nan") if np.isclose(denominator, 0.0) else float(
            1 - np.sum(residual ** 2) / denominator
        )
        return {"MSE": mse, "R2": r2}

    @staticmethod
    def level_ratio_test(data):
        """检验原始序列是否通过 GM(1,1) 级比检验。"""
        x0 = np.asarray(data, dtype=float).reshape(-1)
        if len(x0) < 2 or np.any(x0 <= 0) or not np.all(np.isfinite(x0)):
            return False
        ratios = x0[:-1] / x0[1:]
        lower = np.exp(-2 / (len(x0) + 1))
        upper = np.exp(2 / (len(x0) + 1))
        return bool(np.all((ratios > lower) & (ratios < upper)))

    def _restore(self, points):
        """对 1-AGO 时间响应累减，还原原始序列。"""
        x1_hat = self.time_response(points)
        return np.concatenate(([self.x0[0]], np.diff(x1_hat)))

    @staticmethod
    def _validate_series(data):
        x0 = np.asarray(data, dtype=float)
        if x0.ndim != 1:
            raise ValueError("data 必须是一维时间序列")
        if len(x0) < 4:
            raise ValueError("GM(1,1) 至少需要 4 个观测值")
        if not np.all(np.isfinite(x0)):
            raise ValueError("data 不能包含 NaN 或无穷值")
        if np.any(x0 <= 0):
            raise ValueError("GM(1,1) 原始序列必须全部为正数")
        return x0.copy()

    def _check_fitted(self):
        if self.x0 is None:
            raise RuntimeError("请先调用 fit() 拟合模型")

    @staticmethod
    def _accuracy_grade(error):
        if error <= 0.01:
            return "一级（好）"
        if error <= 0.05:
            return "二级（合格）"
        if error <= 0.10:
            return "三级（勉强）"
        return "四级（不合格）"
