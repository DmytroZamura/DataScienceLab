"""Лабораторна робота 2, частина 1. Розв'язання задач регресії за допомогою MLP (варіант 1).

Основне завдання: модель Y = f(1ф, 2ф) на даних з табл. 1.2 (варіант 1).
Додаткове завдання: регресія міцності бетону (UCI Concrete Compressive Strength).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # графіки лише зберігаються у файли, вікна не відкриваються
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "log.txt", "w", encoding="utf-8")
RESULTS = {}
# Приховуємо службові попередження TensorFlow (на результати не впливають)
tf.get_logger().setLevel("ERROR")


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    LOG.write(text + "\n")


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close()


def build_model(n_inputs, layers=(4, 8, 4), activation="relu", lr=0.01):
    model = tf.keras.Sequential([tf.keras.Input(shape=(n_inputs,))])
    for units in layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


# ---------------------------------------------------------------------------
# 1. Зчитування даних
# ---------------------------------------------------------------------------
tf.keras.utils.set_random_seed(42)

col_names = ["input1", "input2", "output"]
dataf = pd.read_csv(BASE / "Data.txt", names=col_names, sep="\t")
log("=== Вихідні дані ===")
log(dataf.columns)
log(dataf.shape)
log(dataf)

nrow, ncol = dataf.shape
X = dataf.iloc[:, range(0, ncol - 1)].values.astype(float)
Y = dataf.iloc[:, ncol - 1].values.astype(float)

# ---------------------------------------------------------------------------
# 2. Розбиття на тренувальну і тестову вибірки + масштабування
# ---------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
log("\n=== Розбиття вибірки ===")
log("X_train =", X_train.tolist())
log("X_test  =", X_test.tolist(), " Y_test =", Y_test.tolist())

# 2ф змінюється в діапазоні 0..20, а 1ф - в 0..2, тому масштабуємо входи в [0, 1].
# Scaler навчається лише на тренувальних даних, щоб уникнути витоку даних.
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Порівняння архітектур (Leave-One-Out на всіх 9 зразках)
# ---------------------------------------------------------------------------
configs = {
    "MLP 2-4-8-4-1 (ReLU)": dict(layers=(4, 8, 4), activation="relu"),
    "MLP 2-8-1 (tanh)": dict(layers=(8,), activation="tanh"),
    "MLP 2-6-6-1 (sigmoid)": dict(layers=(6, 6), activation="sigmoid"),
}
nEpoch = 1000
log("\n=== Порівняння архітектур (LOO-крос-валідація) ===")
cv_rows = []
for name, cfg in configs.items():
    preds = np.zeros_like(Y)
    for tr, te in LeaveOneOut().split(X):
        tf.keras.utils.set_random_seed(42)
        sc = MinMaxScaler().fit(X[tr])
        m = build_model(2, **cfg)
        m.fit(sc.transform(X[tr]), Y[tr], epochs=nEpoch, verbose=0)
        # Прямий виклик моделі замість predict(): не перебудовує tf.function для кожної нової моделі
        preds[te] = m(sc.transform(X[te]), training=False).numpy()[:, 0]
    row = {
        "model": name,
        "loo_mae": float(mean_absolute_error(Y, preds)),
        "loo_rmse": float(np.sqrt(mean_squared_error(Y, preds))),
        "loo_r2": float(r2_score(Y, preds)),
    }
    cv_rows.append(row)
    log(f"{name:25s} MAE={row['loo_mae']:.3f}  RMSE={row['loo_rmse']:.3f}  R2={row['loo_r2']:.3f}")
RESULTS["cv"] = cv_rows

# ---------------------------------------------------------------------------
# 4. Навчання основної моделі (найкраща за LOO архітектура)
# ---------------------------------------------------------------------------
best_name = max(cv_rows, key=lambda r: r["loo_r2"])["model"]
RESULTS["best_model"] = best_name
log(f"\nОбрано архітектуру: {best_name}")
tf.keras.utils.set_random_seed(42)
model = build_model(ncol - 1, **configs[best_name])
history = model.fit(X_train_s, Y_train, epochs=nEpoch, verbose=0)
summary_lines = []
model.summary(print_fn=lambda s, **kw: summary_lines.append(s))
log("\n=== Архітектура мережі ===")
log("\n".join(summary_lines))
log(f"Кінцеве значення MSE на тренувальних даних: {history.history['loss'][-1]:.4f}")

Y_pred_train = model.predict(X_train_s, verbose=0)

# Рис. 1 - фактичні vs прогнозні на тренувальних даних
plt.figure(figsize=(6, 4))
plt.scatter(range(len(Y_train)), Y_train, label="Дані")
plt.plot(range(len(Y_train)), Y_pred_train, "r", label="Прогноз")
plt.xlabel("Номер зразка")
plt.ylabel("Y")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig1_fact_vs_pred.png")

# Рис. 2 - крива навчання
with plt.style.context("seaborn-v0_8-whitegrid"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history.history["loss"], linewidth=2, label="loss")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    savefig("fig2_loss.png")

# Рис. 3 - залишки
resid = Y_pred_train[:, 0] - Y_train
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.plot(range(len(Y_train)), resid, "r", marker="o", label="Residual deviations")
ax1.axhline(0, color="k")
ax1.legend()
ax2.hist(resid, bins=5, density=True)
ax2.set_title("Гістограма залишків")
savefig("fig3_residuals.png")

# ---------------------------------------------------------------------------
# 5. Оцінка адекватності
# ---------------------------------------------------------------------------
COR_train = np.corrcoef(Y_train, Y_pred_train[:, 0])[0, 1]
Y_pred_test = model.predict(X_test_s, verbose=0)
log("\n=== Оцінка адекватності ===")
log("Коефіцієнт кореляції (train) =", round(COR_train, 6))
log("Коефіцієнт кореляції (test)  = nan (тестова вибірка містить 1 зразок)")
log(f"Тестовий зразок X={X_test.tolist()}: факт={Y_test[0]:.2f}, прогноз={Y_pred_test[0,0]:.2f}, "
    f"абсолютна помилка={abs(Y_test[0]-Y_pred_test[0,0]):.2f}")
log("Тренувальні: факт =", np.round(Y_train, 2).tolist())
log("Тренувальні: прогноз =", np.round(Y_pred_train[:, 0], 2).tolist())
log(f"MAE (train) = {mean_absolute_error(Y_train, Y_pred_train[:,0]):.4f}; "
    f"R2 (train) = {r2_score(Y_train, Y_pred_train[:,0]):.4f}")
RESULTS["main"] = {
    "final_loss": float(history.history["loss"][-1]),
    "cor_train": float(COR_train),
    "mae_train": float(mean_absolute_error(Y_train, Y_pred_train[:, 0])),
    "r2_train": float(r2_score(Y_train, Y_pred_train[:, 0])),
    "x_test": X_test.tolist(),
    "y_test": float(Y_test[0]),
    "y_pred_test": float(Y_pred_test[0, 0]),
    "train_table": [
        {"x1": float(a), "x2": float(b), "y": float(c), "pred": float(d)}
        for (a, b), c, d in zip(X_train, Y_train, Y_pred_train[:, 0])
    ],
}

# ---------------------------------------------------------------------------
# 6. Застосування до нових даних
# ---------------------------------------------------------------------------
X_new = np.array([[1.5, 15.0], [0.5, 5.0], [1.0, 20.0], [2.0, 5.0]])
Y_new = model.predict(scaler.transform(X_new), verbose=0)
log("\n=== Нові дані ===")
for x, y in zip(X_new, Y_new[:, 0]):
    log(f"X_new = {x.tolist()} -> Y_new = {y:.4f}")
RESULTS["new"] = [{"x1": float(x[0]), "x2": float(x[1]), "y": float(y)} for x, y in zip(X_new, Y_new[:, 0])]

# ---------------------------------------------------------------------------
# 7. 3D-поверхня Y(1ф, 2ф)
# ---------------------------------------------------------------------------
N = 25
x1 = np.linspace(0, 2, N)
x2 = np.linspace(0, 20, N)
G1, G2 = np.meshgrid(x1, x2)
grid = np.column_stack([G1.ravel(), G2.ravel()])
Ygr = model.predict(scaler.transform(grid), verbose=0)[:, 0].reshape(G1.shape)
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(G1, G2, Ygr, cmap="viridis", edgecolor="none", alpha=0.9)
ax.scatter(X[:, 0], X[:, 1], Y, color="red", s=30, label="Вихідні дані")
ax.set_xlabel("1ф (input1)")
ax.set_ylabel("2ф (input2)")
ax.set_zlabel("Y")
ax.legend()
savefig("fig4_surface.png")

# ===========================================================================
# ДОДАТКОВЕ ЗАВДАННЯ: UCI Concrete Compressive Strength
# ===========================================================================
from ucimlrepo import fetch_ucirepo

log("\n\n=== ДОДАТКОВЕ ЗАВДАННЯ: Concrete Compressive Strength ===")
concrete = fetch_ucirepo(id=165)
Xc = concrete.data.features
yc = concrete.data.targets.iloc[:, 0]
log("Розмір вибірки:", Xc.shape)
log(pd.concat([Xc, yc], axis=1).describe().T.round(2).to_string())
RESULTS["concrete_describe"] = pd.concat([Xc, yc], axis=1).describe().T.round(2).reset_index().to_dict("records")

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc.values, yc.values, test_size=0.2, random_state=42)
sc_c = StandardScaler().fit(Xc_train)
Xc_train_s, Xc_test_s = sc_c.transform(Xc_train), sc_c.transform(Xc_test)

tf.keras.utils.set_random_seed(42)
model_c = tf.keras.Sequential([
    tf.keras.Input(shape=(Xc.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])
model_c.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss="mse", metrics=["mae"])
early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
hist_c = model_c.fit(Xc_train_s, yc_train, validation_split=0.2, epochs=1000, batch_size=32,
                     callbacks=[early], verbose=0)
log(f"Навчання зупинено на епосі {len(hist_c.history['loss'])}")

pred_tr = model_c.predict(Xc_train_s, verbose=0)[:, 0]
pred_te = model_c.predict(Xc_test_s, verbose=0)[:, 0]
metrics_c = {}
for split, yt, yp in [("train", yc_train, pred_tr), ("test", yc_test, pred_te)]:
    metrics_c[split] = {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "cor": float(np.corrcoef(yt, yp)[0, 1]),
    }
    log(f"{split}: R2={metrics_c[split]['r2']:.4f} MAE={metrics_c[split]['mae']:.3f} "
        f"RMSE={metrics_c[split]['rmse']:.3f} r={metrics_c[split]['cor']:.4f}")
RESULTS["concrete"] = {"metrics": metrics_c, "epochs": len(hist_c.history["loss"])}

with plt.style.context("seaborn-v0_8-whitegrid"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist_c.history["loss"], linewidth=2, label="loss")
    ax.plot(hist_c.history["val_loss"], linewidth=2, label="val_loss")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    savefig("fig5_concrete_loss.png")

plt.figure(figsize=(5.5, 5.5))
plt.scatter(yc_test, pred_te, alpha=0.6)
lim = [0, max(yc_test.max(), pred_te.max()) + 5]
plt.plot(lim, lim, "r--", label="y = x")
plt.xlabel("Фактична міцність, МПа")
plt.ylabel("Прогноз, МПа")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig6_concrete_pred.png")

new_mix = pd.DataFrame([
    [380, 95, 0, 228, 0, 932, 594, 28],
    [300, 0, 120, 180, 8, 1000, 780, 90],
], columns=Xc.columns)
new_pred = model_c.predict(sc_c.transform(new_mix.values), verbose=0)[:, 0]
log("\nНові суміші бетону:")
log(new_mix.assign(predicted_strength=np.round(new_pred, 2)).to_string())
RESULTS["concrete_new"] = new_mix.assign(predicted_strength=np.round(new_pred, 2)).to_dict("records")

(OUT / "results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
LOG.close()
