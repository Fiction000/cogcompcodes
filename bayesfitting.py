
"""ベイズ的な手法による多項式回帰を行い，PRMLの１章の図1.7を再現する．"""
import numpy as np
import matplotlib.pyplot as plt

# データと目標値
X = np.asmatrix([0.000000, 0.111111, 0.222222, 0.333333, 0.444444,
                 0.555556, 0.666667, 0.777778, 0.888889, 1.000000])
t = np.asmatrix([0.349486, 0.830839, 1.007332, 0.971507, 0.133066,
                 0.166823, -0.848307, -0.445686, -0.563567, 0.261502])

# テストデータとデータを生成したsin関数
x = np.arange(0, 1, 0.01)
sine_curve = np.sin(2 * np.pi * x)

# パラメータ
alpha = 5 * 10**(-3)
beta = 11.1
M = 9

# 予測分布の平均
Phi_X = np.asmatrix([[x**i for i in range(M + 1)] for x in X.getA1()])
phi_x = np.asmatrix([[x1**i for i in range(M + 1)] for x1 in x])
S = (alpha * np.eye(10) + (beta * Phi_X.T) @ Phi_X).I
m_x = beta * (S @ Phi_X.T * t.T)

# 予測分布の分散
phi_x_s = np.asmatrix([x**i + 1 for i, x in enumerate(X.getA1())])
s_x = 1 / beta + phi_x_s @ S @ phi_x_s.T
s_sqrt = s_x.item()**2

# フィット
y_xw = phi_x @ m_x
pred = np.array(y_xw)

#　分散を追加
pred_variance1 = y_xw + s_sqrt
pred_variance2 = y_xw - s_sqrt

# プロット
ax = plt.subplot()
ax.plot(X, t, 'bo')
ax.plot(x, sine_curve, 'g')
ax.plot(x, pred.flatten(), 'r')
ax.plot(x, pred_variance1.getA1(), 'k--')
ax.plot(x, pred_variance2.getA1(), 'k--')
plt.show()
