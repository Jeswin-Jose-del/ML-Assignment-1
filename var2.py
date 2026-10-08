import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error, r2_score


TRAIN_FILE = "BT2024142_train_var2.csv"
TEST_FILE = "BT2024142_test_var2.csv"
OUTPUT_FILE = "BT2024142_pred_var2.csv"

FEATURES = ["x1", "x2", "x3"]

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

X = train[FEATURES]
y = train["y"]
X_test = test[FEATURES]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scorer = make_scorer(
    mean_squared_error,
    greater_is_better=False
)

r2_scorer = make_scorer(r2_score)


degree_results = []

for degree in range(1, 21):

    model = Pipeline([
        ("poly", PolynomialFeatures(
            degree=degree,
            include_bias=False
        )),
        ("regression", LinearRegression())
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "mse": mse_scorer,
            "r2": r2_scorer
        },
        n_jobs=-1
    )

    mse = -scores["test_mse"].mean()
    r2 = scores["test_r2"].mean()

    degree_results.append([degree, mse, r2])

    print(f"Degree {degree:2d} | MSE = {mse:.6f} | R2 = {r2:.6f}")


degree_results_df = pd.DataFrame(
    degree_results,
    columns=["degree", "mse", "r2"]
)

best_degree = int(
    degree_results_df.loc[
        degree_results_df["mse"].idxmin(),
        "degree"
    ]
)

print("\nBest unregularized degree:", best_degree)


initial_alphas = [
    0,
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
]

ridge_results = []

for alpha in initial_alphas:

    model = Pipeline([
        ("poly", PolynomialFeatures(
            degree=best_degree,
            include_bias=False
        )),
        ("ridge", Ridge(alpha=alpha))
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "mse": mse_scorer,
            "r2": r2_scorer
        },
        n_jobs=-1
    )

    mse = -scores["test_mse"].mean()
    r2 = scores["test_r2"].mean()

    ridge_results.append([alpha, mse, r2])

    print(f"Alpha {alpha:<8} | MSE = {mse:.6f} | R2 = {r2:.6f}")


fine_alphas = [
    0.001,
    0.002,
    0.003,
    0.005,
    0.007,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1
]

fine_results = []

for alpha in fine_alphas:

    model = Pipeline([
        ("poly", PolynomialFeatures(
            degree=best_degree,
            include_bias=False
        )),
        ("ridge", Ridge(alpha=alpha))
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "mse": mse_scorer,
            "r2": r2_scorer
        },
        n_jobs=-1
    )

    mse = -scores["test_mse"].mean()
    r2 = scores["test_r2"].mean()

    fine_results.append([alpha, mse, r2])

    print(f"Alpha {alpha:<7} | MSE = {mse:.8f} | R2 = {r2:.8f}")


fine_results_df = pd.DataFrame(
    fine_results,
    columns=["alpha", "mse", "r2"]
)

best_alpha = fine_results_df.loc[
    fine_results_df["mse"].idxmin(),
    "alpha"
]

final_degree = best_degree

final_model = Pipeline([
    ("poly", PolynomialFeatures(
        degree=final_degree,
        include_bias=False
    )),
    ("ridge", Ridge(alpha=best_alpha))
])

scores = cross_validate(
    final_model,
    X,
    y,
    cv=kf,
    scoring={
        "mse": mse_scorer,
        "r2": r2_scorer
    },
    n_jobs=-1
)

final_mse = -scores["test_mse"].mean()
final_r2 = scores["test_r2"].mean()

final_model.fit(X, y)

predictions = final_model.predict(X_test)

pd.DataFrame({
    "y": predictions
}).to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFinal degree:", final_degree)
print("Final alpha:", best_alpha)
print("CV MSE:", final_mse)
print("CV R2:", final_r2)
print("Saved:", OUTPUT_FILE)