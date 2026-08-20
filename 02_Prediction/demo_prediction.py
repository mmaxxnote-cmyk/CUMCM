"""预测模型统一调用方式示例。"""

import numpy as np

from LinearRegression import LinearRegressionModel


def main():
    # 训练数据：y = 2x + 1
    X_train = np.array([[1], [2], [3], [4]], dtype=float)
    y_train = np.array([3, 5, 7, 9], dtype=float)
    X_test = np.array([[5], [6]], dtype=float)
    y_test = np.array([11, 13], dtype=float)

    model = LinearRegressionModel()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = model.evaluate(y_test, prediction)

    print("prediction:", prediction)
    print("evaluation:", score)


if __name__ == "__main__":
    main()
