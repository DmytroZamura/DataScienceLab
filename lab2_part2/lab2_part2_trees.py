"""Лабораторна робота 2, частина 2. Розв'язання задач регресії (варіант 1 - Student Performance).

Цільова ознака: G3 - підсумкова оцінка учня (0..20).
Моделі: DecisionTree, RandomForest, XGBoost, нейронна мережа (MLP, Keras).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # графіки лише зберігаються у файли, вікна не відкриваються
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from ucimlrepo import fetch_ucirepo
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "log.txt", "w", encoding="utf-8")
RESULTS = {}
SEED = 42


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    LOG.write(text + "\n")


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close()


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ===========================================================================
# Етап 1. Підготовка даних (EDA)
# ===========================================================================
student = fetch_ucirepo(id=320)
df = pd.concat([student.data.features, student.data.targets], axis=1)
log("=== Етап 1. EDA ===")
log("Розмір датасету:", df.shape)
log("Пропущені значення:", int(df.isna().sum().sum()))
log("Дублікати:", int(df.duplicated().sum()))
log("\nПерші рядки:\n" + df.head().T.to_string())
num_cols = df.select_dtypes("number").columns
log("\nОписова статистика числових ознак:\n" + df[num_cols].describe().T.round(2).to_string())
RESULTS["shape"] = list(df.shape)
RESULTS["describe"] = df[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]].describe().T.round(2).reset_index().to_dict("records")

# Викиди: G3 <= 1 - учні, які фактично не складали підсумковий іспит (при цьому G2 > 0)
zero_g3 = df[df["G3"] <= 1]
log(f"\nУчнів з G3 <= 1: {len(zero_g3)}; з них G2 > 0: {int((zero_g3['G2'] > 0).sum())}")
q1, q3 = df["absences"].quantile([0.25, 0.75])
abs_thr = q3 + 1.5 * (q3 - q1)
log(f"Пропуски (absences): межа IQR = {abs_thr:.1f}, викидів = {int((df['absences'] > abs_thr).sum())}, max = {df['absences'].max()}")
RESULTS["outliers"] = {"g3_zero": int(len(zero_g3)), "abs_thr": float(abs_thr),
                       "abs_out": int((df["absences"] > abs_thr).sum()), "abs_max": int(df["absences"].max())}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
sns.histplot(df["G3"], bins=21, ax=axes[0], color="#4C72B0")
axes[0].set_title("Розподіл G3 (підсумкова оцінка)")
sns.boxplot(y=df["absences"], ax=axes[1], color="#DD8452")
axes[1].set_title("Пропуски занять (absences)")
axes[2].scatter(df["G2"], df["G3"], alpha=0.4)
axes[2].set_xlabel("G2")
axes[2].set_ylabel("G3")
axes[2].set_title("G3 vs G2 (точки G3≤1 - викиди)")
savefig("fig1_eda.png")

corr = df[num_cols].corr()["G3"].drop("G3").sort_values()
plt.figure(figsize=(7, 5))
corr.plot.barh(color=["#C44E52" if v < 0 else "#4C72B0" for v in corr])
plt.title("Кореляція числових ознак з G3")
plt.grid(alpha=0.3, axis="x")
savefig("fig2_corr.png")
RESULTS["corr_top"] = corr.round(3).to_dict()

# Видаляємо записи з G3 <= 1 як викиди (іспит не складався, а не "нульові знання")
df = df[df["G3"] > 1].reset_index(drop=True)
# Обмежуємо екстремальні пропуски (вінзоризація за 99-м перцентилем)
abs_cap = df["absences"].quantile(0.99)
df["absences"] = df["absences"].clip(upper=abs_cap)
log(f"Після очищення: {df.shape}, absences обмежено значенням {abs_cap:.0f}")
RESULTS["clean_shape"] = list(df.shape)
RESULTS["abs_cap"] = float(abs_cap)

# ---------------------------------------------------------------------------
# Кодування категоріальних ознак
# ---------------------------------------------------------------------------
binary_cols = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid",
               "activities", "nursery", "higher", "internet", "romantic"]
nominal_cols = ["Mjob", "Fjob", "reason", "guardian"]
encoders = {c: LabelEncoder().fit(df[c]) for c in binary_cols}


def encode(raw: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Перетворює сирі дані: LabelEncoder для бінарних, OneHot для номінальних ознак."""
    out = raw.copy()
    for c in binary_cols:
        out[c] = encoders[c].transform(out[c])
    out = pd.get_dummies(out, columns=nominal_cols, dtype=int)
    if columns is not None:
        out = out.reindex(columns=columns, fill_value=0)
    return out


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Інженерія ознак: нові ознаки на базі існуючих."""
    X = X.copy()
    X["G_mean"] = (X["G1"] + X["G2"]) / 2          # середня проміжна оцінка
    X["G_trend"] = X["G2"] - X["G1"]               # динаміка успішності
    X["alc_total"] = X["Dalc"] + X["Walc"]         # сумарне вживання алкоголю
    X["parent_edu"] = X["Medu"] + X["Fedu"]        # освіта батьків
    X["study_per_fail"] = X["studytime"] / (1 + X["failures"])
    return X


y = df["G3"].astype(float)
X_raw = df.drop(columns=["G3"])
X = encode(X_raw)
FEATURES = list(X.columns)
log(f"Кількість ознак після кодування: {X.shape[1]}")
RESULTS["n_features"] = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
log(f"Train: {X_train.shape}, Test: {X_test.shape}")
RESULTS["split"] = [len(X_train), len(X_test)]
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ===========================================================================
# Етап 2. Decision Tree: MSE від max_depth
# ===========================================================================
log("\n=== Етап 2. Decision Tree ===")
depths = range(1, 21)
dt_curve = []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_train, y_train)
    cv_mse = -cross_val_score(DecisionTreeRegressor(max_depth=d, random_state=SEED), X_train, y_train,
                              cv=kf, scoring="neg_mean_squared_error").mean()
    dt_curve.append({
        "depth": d,
        "train_mse": float(mean_squared_error(y_train, m.predict(X_train))),
        "cv_mse": float(cv_mse),
        "test_mse": float(mean_squared_error(y_test, m.predict(X_test))),
    })
dt_curve_df = pd.DataFrame(dt_curve)
best_depth = int(dt_curve_df.loc[dt_curve_df["cv_mse"].idxmin(), "depth"])
log(dt_curve_df.round(3).to_string(index=False))
log(f"Оптимальна глибина (мінімум CV-MSE): {best_depth}")
RESULTS["dt_curve"] = dt_curve
RESULTS["dt_best_depth"] = best_depth

plt.figure(figsize=(8, 4.5))
plt.plot(dt_curve_df["depth"], dt_curve_df["train_mse"], "o-", label="Train MSE")
plt.plot(dt_curve_df["depth"], dt_curve_df["cv_mse"], "s-", label="CV MSE (5-fold)")
plt.plot(dt_curve_df["depth"], dt_curve_df["test_mse"], "^-", label="Test MSE")
plt.axvline(best_depth, color="gray", ls="--", label=f"Оптимум: depth={best_depth}")
plt.xlabel("max_depth")
plt.ylabel("MSE")
plt.xticks(list(depths))
plt.title("Decision Tree: залежність MSE від глибини дерева")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig3_dt_depth.png")

dt_model = DecisionTreeRegressor(max_depth=best_depth, random_state=SEED).fit(X_train, y_train)

# ===========================================================================
# Етап 3. Ансамблеві методи
# ===========================================================================
log("\n=== Етап 3. Random Forest ===")
rf_curve = []
for d in depths:
    m = RandomForestRegressor(n_estimators=200, max_depth=d, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
    rf_curve.append({"depth": d, "train_mse": float(mean_squared_error(y_train, m.predict(X_train))),
                     "test_mse": float(mean_squared_error(y_test, m.predict(X_test)))})
rf_curve_df = pd.DataFrame(rf_curve)
log(rf_curve_df.round(3).to_string(index=False))
RESULTS["rf_curve"] = rf_curve

plt.figure(figsize=(8, 4.5))
plt.plot(dt_curve_df["depth"], dt_curve_df["test_mse"], "o-", label="Decision Tree - Test MSE")
plt.plot(rf_curve_df["depth"], rf_curve_df["test_mse"], "s-", label="Random Forest - Test MSE")
plt.plot(rf_curve_df["depth"], rf_curve_df["train_mse"], "s--", alpha=0.6, label="Random Forest - Train MSE")
plt.xlabel("max_depth")
plt.ylabel("MSE")
plt.xticks(list(depths))
plt.title("Поведінка Random Forest і окремого дерева при зростанні глибини")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig4_rf_vs_dt.png")

rf_trees = []
for n in [10, 50, 100, 200, 500]:
    m = RandomForestRegressor(n_estimators=n, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
    rf_trees.append({"n_estimators": n, **metrics(y_test, m.predict(X_test))})
log(pd.DataFrame(rf_trees).round(4).to_string(index=False))
RESULTS["rf_trees"] = rf_trees

rf_model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1).fit(X_train, y_train)

log("\n=== Етап 3. XGBoost ===")
lrs = [0.01, 0.05, 0.1, 0.3, 0.9]
n_ests = [50, 100, 300, 1000]
xgb_grid = []
for lr in lrs:
    for n in n_ests:
        params = dict(n_estimators=n, learning_rate=lr, max_depth=3, subsample=0.8, random_state=SEED)
        cv_mae = -cross_val_score(XGBRegressor(**params), X_train, y_train, cv=kf,
                                  scoring="neg_mean_absolute_error").mean()
        m = XGBRegressor(**params).fit(X_train, y_train)
        p = m.predict(X_test)
        xgb_grid.append({"learning_rate": lr, "n_estimators": n, "cv_mae": float(cv_mae),
                         "train_mae": float(mean_absolute_error(y_train, m.predict(X_train))),
                         **metrics(y_test, p)})
xgb_df = pd.DataFrame(xgb_grid)
log(xgb_df.round(4).to_string(index=False))
best_xgb = xgb_df.loc[xgb_df["cv_mae"].idxmin()]
log(f"Найкраща комбінація за CV-MAE: learning_rate={best_xgb.learning_rate}, n_estimators={int(best_xgb.n_estimators)}")
RESULTS["xgb_grid"] = xgb_grid
RESULTS["xgb_best"] = {"learning_rate": float(best_xgb.learning_rate), "n_estimators": int(best_xgb.n_estimators)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, col, title in [(axes[0], "train_mae", "MAE на тренувальних даних"), (axes[1], "mae", "MAE на тестових даних")]:
    piv = xgb_df.pivot(index="learning_rate", columns="n_estimators", values=col)
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax)
    ax.set_title(f"XGBoost: {title}")
savefig("fig5_xgb_grid.png")

xgb_model = XGBRegressor(n_estimators=int(best_xgb.n_estimators), learning_rate=float(best_xgb.learning_rate),
                         max_depth=3, subsample=0.8, random_state=SEED).fit(X_train, y_train)

# ===========================================================================
# Етап 4. Нейромережна модель
# ===========================================================================
log("\n=== Етап 4. Нейронна мережа (MLP) ===")


def train_nn(Xtr, ytr):
    tf.keras.utils.set_random_seed(SEED)
    scaler = StandardScaler().fit(Xtr)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    hist = model.fit(scaler.transform(Xtr), ytr.values, validation_split=0.2, epochs=600, batch_size=16,
                     callbacks=[early], verbose=0)
    return model, scaler, hist


nn_model, nn_scaler, nn_hist = train_nn(X_train, y_train)
log(f"Навчання зупинено на епосі {len(nn_hist.history['loss'])}")
summary_lines = []
nn_model.summary(print_fn=lambda s, **kw: summary_lines.append(s))
RESULTS["nn_epochs"] = len(nn_hist.history["loss"])
RESULTS["nn_params"] = int(nn_model.count_params())

with plt.style.context("seaborn-v0_8-whitegrid"):
    plt.figure(figsize=(8, 4.5))
    plt.plot(nn_hist.history["loss"], lw=2, label="loss")
    plt.plot(nn_hist.history["val_loss"], lw=2, label="val_loss")
    plt.yscale("log")
    plt.xlabel("Number of epochs")
    plt.ylabel("MSE")
    plt.legend()
    savefig("fig6_nn_loss.png")


def nn_predict(model, scaler, Xd):
    return model.predict(scaler.transform(Xd), verbose=0)[:, 0]


# ===========================================================================
# Порівняння моделей
# ===========================================================================
log("\n=== Порівняльна таблиця (тестова вибірка) ===")
preds = {
    "Decision Tree": dt_model.predict(X_test),
    "Random Forest": rf_model.predict(X_test),
    "XGBoost": xgb_model.predict(X_test),
    "Neural Network (MLP)": nn_predict(nn_model, nn_scaler, X_test),
}
train_preds = {
    "Decision Tree": dt_model.predict(X_train),
    "Random Forest": rf_model.predict(X_train),
    "XGBoost": xgb_model.predict(X_train),
    "Neural Network (MLP)": nn_predict(nn_model, nn_scaler, X_train),
}
comparison = []
for name in preds:
    row = {"model": name, **metrics(y_test, preds[name]),
           "train_r2": float(r2_score(y_train, train_preds[name]))}
    comparison.append(row)
comp_df = pd.DataFrame(comparison)
log(comp_df.round(4).to_string(index=False))
RESULTS["comparison"] = comparison

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
for ax, (name, p) in zip(axes, preds.items()):
    ax.scatter(y_test, p, alpha=0.6)
    ax.plot([0, 20], [0, 20], "r--")
    ax.set_title(f"{name}\nR²={r2_score(y_test, p):.3f}")
    ax.set_xlabel("Факт G3")
axes[0].set_ylabel("Прогноз G3")
savefig("fig7_pred_vs_fact.png")

# Важливість ознак
fi = pd.DataFrame({"Random Forest": rf_model.feature_importances_, "XGBoost": xgb_model.feature_importances_},
                  index=FEATURES)
top = fi.sort_values("Random Forest", ascending=False).head(12)
top[::-1].plot.barh(figsize=(8, 5.5))
plt.title("Важливість ознак (Feature Importance), топ-12")
plt.grid(alpha=0.3, axis="x")
savefig("fig8_feature_importance.png")
RESULTS["feature_importance"] = top.round(4).reset_index().rename(columns={"index": "feature"}).to_dict("records")
log("\nВажливість ознак:\n" + top.round(4).to_string())

# ===========================================================================
# Етап 5. Інженерія ознак
# ===========================================================================
log("\n=== Етап 5. Feature Engineering ===")
X_train_fe, X_test_fe = add_features(X_train), add_features(X_test)
dt_fe = DecisionTreeRegressor(max_depth=best_depth, random_state=SEED).fit(X_train_fe, y_train)
rf_fe = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_train_fe, y_train)
xgb_fe = XGBRegressor(n_estimators=int(best_xgb.n_estimators), learning_rate=float(best_xgb.learning_rate),
                      max_depth=3, subsample=0.8, random_state=SEED).fit(X_train_fe, y_train)
nn_fe, nn_fe_scaler, _ = train_nn(X_train_fe, y_train)
preds_fe = {
    "Decision Tree": dt_fe.predict(X_test_fe),
    "Random Forest": rf_fe.predict(X_test_fe),
    "XGBoost": xgb_fe.predict(X_test_fe),
    "Neural Network (MLP)": nn_predict(nn_fe, nn_fe_scaler, X_test_fe),
}
fe_rows = []
for name in preds:
    before, after = metrics(y_test, preds[name]), metrics(y_test, preds_fe[name])
    fe_rows.append({"model": name, "r2_before": before["r2"], "r2_after": after["r2"],
                    "mae_before": before["mae"], "mae_after": after["mae"]})
fe_df = pd.DataFrame(fe_rows)
log(fe_df.round(4).to_string(index=False))
RESULTS["feature_engineering"] = fe_rows

# Тестова вибірка невелика (127 записів), тому перевіряємо ефект ще й 5-кратною крос-валідацією
fe_cv = []
for name, make in [
    ("Decision Tree", lambda: DecisionTreeRegressor(max_depth=best_depth, random_state=SEED)),
    ("Random Forest", lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ("XGBoost", lambda: XGBRegressor(n_estimators=int(best_xgb.n_estimators), learning_rate=float(best_xgb.learning_rate),
                                     max_depth=3, subsample=0.8, random_state=SEED)),
]:
    r2_b = cross_val_score(make(), X_train, y_train, cv=kf, scoring="r2").mean()
    r2_a = cross_val_score(make(), X_train_fe, y_train, cv=kf, scoring="r2").mean()
    fe_cv.append({"model": name, "cv_r2_before": float(r2_b), "cv_r2_after": float(r2_a)})
log("CV R2 (5-fold, train):\n" + pd.DataFrame(fe_cv).round(4).to_string(index=False))
RESULTS["feature_engineering_cv"] = fe_cv

xpos = np.arange(len(fe_df))
plt.figure(figsize=(8, 4.2))
plt.bar(xpos - 0.2, fe_df["r2_before"], 0.4, label="Базові ознаки")
plt.bar(xpos + 0.2, fe_df["r2_after"], 0.4, label="+ нові ознаки")
plt.xticks(xpos, fe_df["model"], rotation=10)
plt.ylim(min(fe_df[["r2_before", "r2_after"]].min()) - 0.05, 1)
plt.ylabel("R² (test)")
plt.title("Вплив інженерії ознак на R²")
plt.legend()
plt.grid(alpha=0.3, axis="y")
savefig("fig9_feature_engineering.png")

# ===========================================================================
# Етап 6. Застосування моделей на однакових нових даних
# ===========================================================================
log("\n=== Етап 6. Прогноз для нових учнів ===")
base_student = X_raw.iloc[0].to_dict()
new_students = pd.DataFrame([
    {**base_student, "school": "GP", "sex": "F", "age": 17, "studytime": 3, "failures": 0, "absences": 2,
     "higher": "yes", "Dalc": 1, "Walc": 1, "G1": 14, "G2": 15},
    {**base_student, "school": "MS", "sex": "M", "age": 18, "studytime": 1, "failures": 2, "absences": 14,
     "higher": "no", "Dalc": 3, "Walc": 4, "G1": 8, "G2": 7},
    {**base_student, "school": "GP", "sex": "M", "age": 16, "studytime": 2, "failures": 0, "absences": 4,
     "higher": "yes", "Dalc": 1, "Walc": 2, "G1": 11, "G2": 12},
])
X_new = encode(new_students, columns=FEATURES)
X_new_fe = add_features(X_new)
new_res = []
for i in range(len(X_new)):
    row = {"student": i + 1, "G1": int(new_students.loc[i, "G1"]), "G2": int(new_students.loc[i, "G2"]),
           "studytime": int(new_students.loc[i, "studytime"]), "failures": int(new_students.loc[i, "failures"]),
           "absences": int(new_students.loc[i, "absences"]),
           "Decision Tree": float(dt_model.predict(X_new.iloc[[i]])[0]),
           "Random Forest": float(rf_model.predict(X_new.iloc[[i]])[0]),
           "XGBoost": float(xgb_model.predict(X_new.iloc[[i]])[0]),
           "Neural Network (MLP)": float(nn_predict(nn_model, nn_scaler, X_new.iloc[[i]])[0]),
           "XGBoost + FE": float(xgb_fe.predict(X_new_fe.iloc[[i]])[0])}
    new_res.append(row)
log(pd.DataFrame(new_res).round(2).to_string(index=False))
RESULTS["new_students"] = new_res

(OUT / "results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
LOG.close()
