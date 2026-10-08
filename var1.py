import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


train_df = pd.read_csv("BT2024142_train_var1.csv")
test_df = pd.read_csv("BT2024142_test_var1.csv")

features = ["x1", "x2", "x3", "x4", "x5", "x6"]

X = train_df[features].to_numpy()
y = train_df["y"].to_numpy()
X_test = test_df[features].to_numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_var = np.array([np.var(y[test_idx]) for _, test_idx in kf.split(X)])

degrees = range(1, 5)

alphas = np.logspace(-5, 1, 25)

results = []
best_overall = None

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()

    Xp = scaler.fit_transform(poly.fit_transform(X))

    cv_model = LassoCV(
        alphas=alphas,
        cv=kf,
        max_iter=20000,
        tol=1e-3,
        n_jobs=-1,
        selection="random",
        random_state=42
    ).fit(Xp, y)

    for i, alpha in enumerate(cv_model.alphas_):
        fold_mse = cv_model.mse_path_[i]
        fold_r2 = 1 - fold_mse / fold_var

        results.append({
            "degree": degree,
            "alpha": alpha,
            "terms": poly.n_output_features_ + 1,
            "mean_mse": np.mean(fold_mse),
            "std_mse": np.std(fold_mse),
            "mean_r2": np.mean(fold_r2),
            "std_r2": np.std(fold_r2)
        })

    print(f"degree {degree} done (best alpha {cv_model.alpha_:.5g})")


results_df = pd.DataFrame(results)
results_df = results_df.sort_values("mean_mse").reset_index(drop=True)

best = results_df.iloc[0]

print("Best degree:", int(best["degree"]))
print("Best alpha:", best["alpha"])
print("CV MSE:", best["mean_mse"])
print("CV R2:", best["mean_r2"])

results_df.to_csv(
    "var1_lasso_model_selection_results.csv",
    index=False
)


poly = PolynomialFeatures(degree=int(best["degree"]), include_bias=False)
scaler = StandardScaler()

Xp = scaler.fit_transform(poly.fit_transform(X))
Xp_test = scaler.transform(poly.transform(X_test))

model = Lasso(alpha=best["alpha"], max_iter=50000, tol=1e-4)
model.fit(Xp, y)



y_pred = model.predict(Xp_test)

pd.DataFrame({"y": y_pred}).to_csv(
    "BT2024142_pred_var1.csv",
    index=False
)

print("Prediction file created: BT2024142_pred_var1.csv")