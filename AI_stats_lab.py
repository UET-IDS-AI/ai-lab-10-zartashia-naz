import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ============================================================
# Question 1: Model Complexity and Generalization
# ============================================================

def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """
    Generate a nonlinear regression dataset.

    True function:
        y = sin(2*pi*x) + Gaussian noise

    Parameters:
        n_samples    : number of samples
        noise        : standard deviation of Gaussian noise
        random_state : random seed

    Returns:
        X, y

    X shape must be:
        (n_samples, 1)

    y shape must be:
        (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    # Generate n_samples evenly spaced x values in [0, 1], shape (n_samples, 1)
    X = rng.uniform(0, 1, size=(n_samples, 1))

    # True function: sin(2*pi*x) + Gaussian noise
    y = np.sin(2 * np.pi * X.ravel()) + rng.normal(0, noise, size=n_samples)

    return X, y


def create_polynomial_model(degree):
    """
    Create a polynomial regression model using sklearn Pipeline.

    The pipeline must contain:
        PolynomialFeatures(degree=degree, include_bias=False)
        LinearRegression()

    Parameters:
        degree : polynomial degree

    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_regression", LinearRegression())
    ])

    return pipeline


def evaluate_polynomial_degrees(X, y, degrees, test_size=0.3, random_state=0):
    """
    Train polynomial models with different degrees and compute train/dev errors.

    Parameters:
        X            : feature matrix
        y            : target values
        degrees      : list of polynomial degrees
        test_size    : fraction of data used for dev set
        random_state : random seed

    Returns:
        {
            "degrees": list of degrees,
            "train_errors": list of training MSE values,
            "dev_errors": list of dev MSE values,
            "best_degree": degree with lowest dev error
        }
    """
    # Split data into train/dev once
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_errors = []
    dev_errors = []

    for degree in degrees:
        # 1. Create polynomial model
        model = create_polynomial_model(degree)

        # 2. Fit on train set
        model.fit(X_train, y_train)

        # 3. Predict on train set and compute train MSE
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_errors.append(train_mse)

        # 4. Predict on dev set and compute dev MSE
        y_dev_pred = model.predict(X_dev)
        dev_mse = mean_squared_error(y_dev, y_dev_pred)
        dev_errors.append(dev_mse)

    # Select best_degree using the lowest dev error
    best_degree = degrees[np.argmin(dev_errors)]

    return {
        "degrees": degrees,
        "train_errors": train_errors,
        "dev_errors": dev_errors,
        "best_degree": best_degree
    }


def diagnose_from_errors(train_error, dev_error, high_error_threshold=0.15, gap_threshold=0.05):
    """
    Diagnose model behavior using train and dev error.

    Parameters:
        train_error          : training error
        dev_error            : dev error
        high_error_threshold : threshold for high train error
        gap_threshold        : threshold for high dev-train gap

    Returns:
        {
            "train_error": train_error,
            "dev_error": dev_error,
            "generalization_gap": dev_error - train_error,
            "diagnosis": diagnosis_string
        }

    Diagnosis rules:
        If train_error > high_error_threshold and gap <= gap_threshold:
            "high_bias"

        If train_error <= high_error_threshold and gap > gap_threshold:
            "high_variance"

        If train_error > high_error_threshold and gap > gap_threshold:
            "high_bias_and_high_variance"

        Otherwise:
            "good_fit"
    """
    generalization_gap = dev_error - train_error

    # Apply diagnosis rules
    if train_error > high_error_threshold and generalization_gap <= gap_threshold:
        diagnosis = "high_bias"
    elif train_error <= high_error_threshold and generalization_gap > gap_threshold:
        diagnosis = "high_variance"
    elif train_error > high_error_threshold and generalization_gap > gap_threshold:
        diagnosis = "high_bias_and_high_variance"
    else:
        diagnosis = "good_fit"

    return {
        "train_error": train_error,
        "dev_error": dev_error,
        "generalization_gap": generalization_gap,
        "diagnosis": diagnosis
    }


# ============================================================
# Question 2: Regularization and Model Improvement
# ============================================================

def regularization_comparison(X_train, y_train, X_dev, y_dev, alphas):
    """
    Compare Ridge regression models with different regularization strengths.

    Parameters:
        X_train : training features
        y_train : training targets
        X_dev   : dev features
        y_dev   : dev targets
        alphas  : list of Ridge alpha values

    Returns:
        {
            "alphas": list of alpha values,
            "train_errors": list of training MSE values,
            "dev_errors": list of dev MSE values,
            "best_alpha": alpha with lowest dev error
        }
    """
    train_errors = []
    dev_errors = []

    for alpha in alphas:
        # Train Ridge model with this alpha
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Compute train MSE
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_errors.append(train_mse)

        # Compute dev MSE
        y_dev_pred = model.predict(X_dev)
        dev_mse = mean_squared_error(y_dev, y_dev_pred)
        dev_errors.append(dev_mse)

    # Select best_alpha using the lowest dev error
    best_alpha = alphas[np.argmin(dev_errors)]

    return {
        "alphas": alphas,
        "train_errors": train_errors,
        "dev_errors": dev_errors,
        "best_alpha": best_alpha
    }


def recommend_action(diagnosis):
    """
    Recommend an action based on bias/variance diagnosis.

    Required mapping:
        "high_bias" ->
            "increase_model_complexity"

        "high_variance" ->
            "add_regularization_or_more_data"

        "high_bias_and_high_variance" ->
            "increase_complexity_then_regularize"

        "good_fit" ->
            "keep_model_or_minor_tuning"

        anything else ->
            "unknown_diagnosis"
    """
    action_map = {
        "high_bias":                    "increase_model_complexity",
        "high_variance":                "add_regularization_or_more_data",
        "high_bias_and_high_variance":  "increase_complexity_then_regularize",
        "good_fit":                     "keep_model_or_minor_tuning",
    }

    return action_map.get(diagnosis, "unknown_diagnosis")


if __name__ == "__main__":
    # ── Q1: Data generation ───────────────────────────────────────────────
    X, y = generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42)
    print("=== generate_nonlinear_data ===")
    print(f"X shape : {X.shape}")   # (100, 1)
    print(f"y shape : {y.shape}")   # (100,)
    print()

    # ── Q1: Polynomial model ──────────────────────────────────────────────
    model = create_polynomial_model(degree=3)
    print("=== create_polynomial_model(degree=3) ===")
    print(model)
    print()

    # ── Q1: Evaluate degrees ──────────────────────────────────────────────
    degrees = [1, 2, 3, 5, 7, 9, 12, 15]
    results = evaluate_polynomial_degrees(X, y, degrees)
    print("=== evaluate_polynomial_degrees ===")
    print(f"Best degree : {results['best_degree']}")
    for d, tr, dv in zip(results["degrees"], results["train_errors"], results["dev_errors"]):
        print(f"  Degree {d:2d} | train MSE = {tr:.4f} | dev MSE = {dv:.4f}")
    print()

    # ── Q1: Diagnose ──────────────────────────────────────────────────────
    print("=== diagnose_from_errors (examples) ===")
    cases = [
        (0.20, 0.22, "→ expect high_bias"),
        (0.05, 0.30, "→ expect high_variance"),
        (0.25, 0.40, "→ expect high_bias_and_high_variance"),
        (0.05, 0.07, "→ expect good_fit"),
    ]
    for te, de, note in cases:
        diag = diagnose_from_errors(te, de)
        print(f"  train={te} dev={de} {note}")
        print(f"  → gap={diag['generalization_gap']:.2f}  diagnosis={diag['diagnosis']}")
    print()

    # ── Q2: Regularization comparison ─────────────────────────────────────
    from sklearn.model_selection import train_test_split as tts
    X_train, X_dev, y_train, y_dev = tts(X, y, test_size=0.3, random_state=0)

    # Use polynomial features first so Ridge has something to regularize
    from sklearn.preprocessing import PolynomialFeatures as PF
    poly = PF(degree=9, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_dev_poly   = poly.transform(X_dev)

    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    reg_results = regularization_comparison(X_train_poly, y_train, X_dev_poly, y_dev, alphas)
    print("=== regularization_comparison ===")
    print(f"Best alpha : {reg_results['best_alpha']}")
    for a, tr, dv in zip(reg_results["alphas"], reg_results["train_errors"], reg_results["dev_errors"]):
        print(f"  alpha={a:8.4f} | train MSE = {tr:.4f} | dev MSE = {dv:.4f}")
    print()

    # ── Q2: Recommend action ──────────────────────────────────────────────
    print("=== recommend_action ===")
    for diag in ["high_bias", "high_variance", "high_bias_and_high_variance",
                 "good_fit", "something_else"]:
        print(f"  {diag:35s} → {recommend_action(diag)}")
