"""Лабораторна робота 1. Побудова багатофакторної регресійної моделі методом МНК.

Модель: y = b0 + b1*x1 + b2*x2 + b3*x1*x2 + b4*x1^2 + b5*x2^2
Виконано контрольні приклади 1, 2 та індивідуальне завдання (варіант 1).
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # графіки лише зберігаються у файли, вікна не відкриваються
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "log.txt", "w", encoding="utf-8")
RESULTS = {}

k = 6  # кількість функцій моделі регресії


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    LOG.write(text + "\n")


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Функції моделі та власна реалізація МНК
# ---------------------------------------------------------------------------
def FunFi(x):
    """Узагальнені регресори: 1, x1, x2, x1*x2, x1^2, x2^2."""
    Fi = [0] * k
    Fi[0] = 1
    Fi[1] = x[0]
    Fi[2] = x[1]
    Fi[3] = x[0] * x[1]
    Fi[4] = x[0] * x[0]
    Fi[5] = x[1] * x[1]
    return Fi


def build_F(X):
    """Матриця F розміром n x k: F[i, j] = phi_j(x_i)."""
    n = len(X)
    F = np.zeros((n, k))
    for i in range(n):
        Fi = FunFi(X[i])
        for j in range(k):
            F[i, j] = Fi[j]
    return F


def solve_gauss(A, B):
    """Розв'язання системи A*a = B методом Гаусса з вибором головного елемента."""
    A = [list(map(float, row)) for row in A]
    B = list(map(float, B))
    size = len(B)
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            raise ValueError("Матриця F^T F вироджена: rank(F) < k")
        A[col], A[pivot] = A[pivot], A[col]
        B[col], B[pivot] = B[pivot], B[col]
        for r in range(col + 1, size):
            factor = A[r][col] / A[col][col]
            for c in range(col, size):
                A[r][c] -= factor * A[col][c]
            B[r] -= factor * B[col]
    a = [0.0] * size
    for r in range(size - 1, -1, -1):
        s = sum(A[r][c] * a[c] for c in range(r + 1, size))
        a[r] = (B[r] - s) / A[r][r]
    return np.array(a)


def mnk(X, Y):
    """Оцінки параметрів за МНК: розв'язок нормальних рівнянь (F^T F) a = F^T Y."""
    F = build_F(X)
    Ft = F.transpose()
    FtF = np.dot(Ft, F)
    FtY = np.dot(Ft, Y)
    return solve_gauss(FtF, FtY), F


def correlation(Y, Y_pred):
    """Коефіцієнт кореляції між фактичними та модельними даними (1.5)."""
    n = len(Y)
    Y_mid = sum(Y) / n
    Y_pred_mid = sum(Y_pred) / n
    S1 = S2 = S3 = 0
    for i in range(n):
        S1 += (Y[i] - Y_mid) * (Y_pred[i] - Y_pred_mid)
        S2 += (Y[i] - Y_mid) ** 2
        S3 += (Y_pred[i] - Y_pred_mid) ** 2
    return S1 / math.sqrt(S2 * S3)


def durbin_watson(r):
    """Критерій Дарбіна-Уотсона (1.6)."""
    S1 = sum(ri * ri for ri in r)
    S2 = sum((r[i] - r[i - 1]) ** 2 for i in range(1, len(r)))
    return S2 / S1


def FunModel(x, a):
    Fi = FunFi(x)
    return sum(a[j] * Fi[j] for j in range(k))


def analyse(tag, X, Y, title):
    """Повний цикл: МНК, залишки, адекватність, графіки."""
    Y = np.array(Y, dtype=float)
    n = len(Y)
    log(f"\n=== {title} ===")
    b, F = mnk(X, Y)
    log("F =\n" + np.array2string(F, precision=0, suppress_small=True, max_line_width=120))
    b_np = np.dot(np.linalg.inv(np.dot(F.T, F)), np.dot(F.T, Y))  # формула (1.4) для контролю
    log("b (власна функція, метод Гаусса) =", np.array2string(b, precision=6, max_line_width=120))
    log("b (формула (F^T F)^-1 F^T Y)      =", np.array2string(b_np, precision=6, max_line_width=120))
    log("Максимальна розбіжність:", f"{np.max(np.abs(b - b_np)):.2e}")
    Y_pred = np.dot(F, b)
    r = Y - Y_pred
    log("Y_pred =", np.array2string(Y_pred, precision=4, max_line_width=120))
    log("r =", np.array2string(r, precision=5, max_line_width=120))
    kcor = correlation(Y, Y_pred)
    du = durbin_watson(r)
    r2 = 1 - np.sum(r ** 2) / np.sum((Y - Y.mean()) ** 2)
    log("Коефіцієнт кореляції =", kcor)
    log("Критерій Дарбіна-Уотсона =", du)
    log("Коефіцієнт детермінації R2 =", r2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.scatter(range(n), Y, label="Дані")
    ax1.plot(range(n), Y_pred, "r", label="Прогноз")
    ax1.set_xlabel("№ експерименту")
    ax1.set_title("Фактичні та прогнозні дані")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(range(n), r, "r", marker="o", label="Residual deviations")
    ax2.axhline(0, color="k")
    ax2.set_xlabel("№ експерименту")
    ax2.set_title("Залишки")
    ax2.legend()
    ax2.grid(alpha=0.3)
    savefig(f"{tag}_fit_resid.png")

    RESULTS[tag] = {
        "b": b.tolist(), "b_np": b_np.tolist(), "max_diff": float(np.max(np.abs(b - b_np))),
        "Y": Y.tolist(), "Y_pred": Y_pred.tolist(), "r": r.tolist(),
        "kcor": kcor, "dw": du, "r2": float(r2), "X": [list(map(float, x)) for x in X],
    }
    return b


def surface(tag, b, x1_range, x2_range, X, Y, zlabel="y", contour=False):
    N = 30
    x1 = np.linspace(*x1_range, N)
    x2 = np.linspace(*x2_range, N)
    Ygr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Ygr[j, i] = FunModel([x1[i], x2[j]], b)
    G1, G2 = np.meshgrid(x1, x2)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    if contour:
        ax.contour3D(G1, G2, Ygr, 50, cmap="plasma")
    else:
        ax.plot_surface(G1, G2, Ygr, cmap="viridis", edgecolor="none", alpha=0.85)
    ax.scatter([x[0] for x in X], [x[1] for x in X], Y, color="red", s=25, label="Дані експерименту")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(zlabel)
    ax.legend()
    savefig(f"{tag}_surface.png")


# ===========================================================================
# Приклад 1
# ===========================================================================
X1 = [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
Y1 = [4.8, 8.7, 11.2, 20.9, 7.3, 17.0, 7.4, 10.2, 3.4]
b1 = analyse("ex1", X1, Y1, "Приклад 1")

# ===========================================================================
# Приклад 2 (міцність сталі)
# ===========================================================================
X2 = [[6, 133], [10, 122], [23, 112], [24, 103], [15, 116], [24, 99], [57, 83], [54, 69], [65, 76],
      [66, 74], [68, 61], [62, 38], [83, 40], [100, 47], [116, 31]]
Y2 = [555, 499, 588, 559, 608, 507, 603, 653, 661, 678, 661, 708, 724, 703, 749]
b2 = analyse("ex2", X2, Y2, "Приклад 2")
surface("ex2", b2, (0, 120), (20, 140), X2, Y2, zlabel="y (міцність)", contour=True)

# ===========================================================================
# Індивідуальне завдання, варіант 1
# x1 - фактор у 1-му рядку таблиці (1ф), x2 - фактор у 1-му стовпці (2ф)
# ===========================================================================
f1 = [0, 1, 2]
f2 = [20, 10, 0]
table = [[14, 19.1, 21.5],
         [4, 7.9, 9.5],
         [5.8, 8.6, 8.7]]
Xv, Yv = [], []
for i, x2 in enumerate(f2):
    for j, x1 in enumerate(f1):
        Xv.append([x1, x2])
        Yv.append(table[i][j])
bv = analyse("var1", Xv, Yv, "Індивідуальне завдання, варіант 1")
surface("var1", bv, (0, 2), (0, 20), Xv, Yv)

# Застосування моделі до нових значень факторів
X_new = [[0.5, 5], [1.5, 15], [1, 5], [2, 15]]
Y_new = [FunModel(x, bv) for x in X_new]
log("\nПрогноз за моделлю варіанту 1:")
for x, y in zip(X_new, Y_new):
    log(f"x1 = {x[0]}, x2 = {x[1]} -> y = {y:.4f}")
RESULTS["var1_new"] = [{"x1": x[0], "x2": x[1], "y": y} for x, y in zip(X_new, Y_new)]

(OUT / "results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
LOG.close()
