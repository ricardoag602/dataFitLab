import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# DataFitLab Baseline: Supervised Learning Pipeline (Linear Regression)
#
# This file demonstrates the full beginner-friendly workflow:
# 1) Generate synthetic data from a known rule.
# 2) Fit a linear regression model to noisy observations.
# 3) Evaluate fit quality using Mean Squared Error (MSE).
# 4) Visualize true relationship vs model prediction.
# -----------------------------------------------------------------------------

# =============================
# 1) Configuration
# =============================

# Ground-truth parameters for the synthetic relationship y = 2x + 1.
TRUE_SLOPE = 2.0
TRUE_INTERCEPT = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Centralized experiment settings for quick tuning and clean demos."""
    n_samples: int = 50
    x_min: float = -3.0
    x_max: float = 3.0
    noise_std: float = 5.0
    seed: int = 42


DEFAULT_CONFIG = ExperimentConfig()


# =============================
# 2) Data Generation
# =============================


def generate_synthetic_data(
    n_samples=DEFAULT_CONFIG.n_samples,
    x_min=DEFAULT_CONFIG.x_min,
    x_max=DEFAULT_CONFIG.x_max,
    noise_std=DEFAULT_CONFIG.noise_std,
    seed=DEFAULT_CONFIG.seed,
):
    """Generate synthetic linear data y = 2x + 1 + Gaussian noise."""
    # Use a seeded random generator for reproducible experiments.
    # Reproducibility helps when comparing future models (Ridge, SVM, etc.).
    rng = np.random.default_rng(seed)

    # Sample inputs x uniformly across a range.
    x = np.linspace(x_min, x_max, n_samples)

    # Compute the noise-free (ground-truth) target.
    y_true = TRUE_SLOPE * x + TRUE_INTERCEPT

    # Add Gaussian noise to simulate real-world measurement variability.
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = y_true + noise

    # Return both noisy labels and the true line for later comparison.
    return x, y, y_true


# =============================
# 3) Model Training
# =============================


def train_linear_regression(x, y):
    """Fit a linear regression model and return model plus predictions."""
    # scikit-learn expects a 2D feature matrix: [n_samples, n_features].
    X = x.reshape(-1, 1)

    # Train the model by minimizing squared errors on the provided data.
    model = LinearRegression()
    model.fit(X, y)

    # Predict on the same x values for baseline visualization/evaluation.
    y_pred = model.predict(X)
    return model, y_pred


# =============================
# 4) Evaluation
# =============================


def evaluate_regression(y_true_values, y_pred):
    """Compute baseline regression metric(s)."""
    # MSE = average of (actual - predicted)^2.
    # Lower MSE means predictions are closer to observed values.
    mse = mean_squared_error(y_true_values, y_pred)
    return {"mse": mse}


# =============================
# 5) Visualization
# =============================


def plot_experiment(x, y_noisy, y_true_line, y_pred):
    """Visualize noisy points, true function, and learned line."""
    # Create one comparison chart so the model behavior is easy to explain.
    plt.figure(figsize=(9, 6))

    # Raw observations (what the model actually sees during training).
    plt.scatter(x, y_noisy, label="Noisy Data", alpha=0.75)

    # Underlying function used to generate the data.
    plt.plot(x, y_true_line, label="True Function (y = 2x + 1)", color="green", linewidth=2)

    # Model's learned approximation from noisy points.
    plt.plot(x, y_pred, label="Linear Regression Prediction", color="red", linewidth=2)

    # Plot formatting to make interpretation clearer in demos.
    plt.title("DataFitLab Baseline: Linear Regression on Synthetic Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def print_report(model, metrics, config):
    """Print a compact summary that is easy to present to teammates."""
    print("\n" + "=" * 52)
    print("DataFitLab Baseline Report")
    print("=" * 52)
    print(f"Config  -> samples={config.n_samples}, noise_std={config.noise_std}, seed={config.seed}")
    print(f"Truth   -> slope={TRUE_SLOPE:.2f}, intercept={TRUE_INTERCEPT:.2f}")
    print(f"Learned -> slope={model.coef_[0]:.4f}, intercept={model.intercept_:.4f}")
    print(f"Metric  -> MSE={metrics['mse']:.4f}")
    print("=" * 52)


# =============================
# 6) Pipeline Orchestration
# =============================


def run_baseline_pipeline(config=DEFAULT_CONFIG):
    """Run synthetic data generation, model training, evaluation, and plotting."""
    # Step 1: Data generation.
    x, y, y_true_line = generate_synthetic_data(
        n_samples=config.n_samples,
        x_min=config.x_min,
        x_max=config.x_max,
        noise_std=config.noise_std,
        seed=config.seed,
    )

    # Step 2: Model fitting.
    model, y_pred = train_linear_regression(x, y)

    # Step 3: Quantitative evaluation.
    metrics = evaluate_regression(y, y_pred)

    # Step 4: Visual analysis of fit quality.
    plot_experiment(x, y, y_true_line, y_pred)

    # Console summary for quick reporting to teammates.
    print_report(model, metrics, config)


if __name__ == "__main__":
    # Script entry point: runs the full baseline experiment.
    run_baseline_pipeline()