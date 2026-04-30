import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.1, iterations=700, lam=0.0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []
    for _ in range(iterations):
        y_hat = sigmoid(X @ w + b)
        eps = 1e-12
        bce = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        l2 = (lam / (2 * n_samples)) * np.sum(w ** 2)
        losses.append(bce + l2)
        error = y_hat - y
        dw = (1 / n_samples) * (X.T @ error) + (lam / n_samples) * w
        db = (1 / n_samples) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return w, b, losses


def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for row in X_test:
        distances = np.sqrt(np.sum((X_train - row) ** 2, axis=1))
        nn_idx = np.argsort(distances)[:k]
        votes = y_train[nn_idx]
        preds.append(int(np.mean(votes) >= 0.5))
    return np.array(preds)


def polynomial_ridge_predict(X_train_1d, y_train, X_eval_1d, degree=3, lam=0.01):
    # Normalize inputs before polynomial expansion to avoid exploding feature values.
    # This keeps high-degree models numerically stable in interactive sweeps.
    x_train = np.asarray(X_train_1d, dtype=np.float64).reshape(-1)
    x_eval = np.asarray(X_eval_1d, dtype=np.float64).reshape(-1)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

    scale = np.max(np.abs(x_train))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    x_train_scaled = x_train / scale
    x_eval_scaled = x_eval / scale

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi_train = poly.fit_transform(x_train_scaled.reshape(-1, 1))
    Phi_eval = poly.transform(x_eval_scaled.reshape(-1, 1))

    # Guard against rare NaN/inf propagation.
    Phi_train = np.nan_to_num(Phi_train, nan=0.0, posinf=1e6, neginf=-1e6)
    Phi_eval = np.nan_to_num(Phi_eval, nan=0.0, posinf=1e6, neginf=-1e6)

    lam = float(max(lam, 1e-12))
    reg = lam * np.eye(Phi_train.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0

    # Solve ridge system directly; if ill-conditioned, add tiny jitter fallback.
    A = Phi_train.T @ Phi_train + reg
    b = Phi_train.T @ y_train
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        jitter = 1e-8 * np.eye(A.shape[0], dtype=np.float64)
        jitter[0, 0] = 0.0
        w = np.linalg.lstsq(A + jitter, b, rcond=None)[0]

    preds = Phi_eval @ w
    return np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)


def _safe_scalar(value, fallback=0.0):
    """Return a finite float, replacing NaN/Inf with a fallback."""
    x = float(np.nan_to_num(value, nan=fallback, posinf=fallback, neginf=fallback))
    return x


def make_regression_data(n_samples=120, noise=0.25, seed=42):
    rng = np.random.default_rng(seed)
    X = np.linspace(-3.0, 3.0, n_samples)
    y_true = np.sin(X) + 0.25 * X
    y = y_true + rng.normal(0.0, noise, n_samples)
    return X, y, y_true


def _build_main_figure(model_name, n_samples, noise, degree, lam, k, seed, val_ratio, lr, iterations):
    fig = make_subplots(rows=1, cols=1)
    metrics = {}
    split_ratio = float(np.clip(val_ratio, 0.1, 0.5))

    if model_name == "Polynomial Regression":
        X, y, y_true = make_regression_data(n_samples=n_samples, noise=noise, seed=seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=seed)
        y_pred_all = polynomial_ridge_predict(X_train, y_train, X, degree=degree, lam=lam)
        y_pred_train = polynomial_ridge_predict(X_train, y_train, X_train, degree=degree, lam=lam)
        y_pred_val = polynomial_ridge_predict(X_train, y_train, X_val, degree=degree, lam=lam)

        train_mse = _safe_scalar(np.mean((y_train - y_pred_train) ** 2), fallback=1e3)
        val_mse = _safe_scalar(np.mean((y_val - y_pred_val) ** 2), fallback=1e3)
        metrics = {"train_error": train_mse, "val_error": val_mse, "label": "MSE"}

        sort_idx = np.argsort(X)
        fig.add_trace(go.Scatter(x=X, y=y, mode="markers", name="Noisy data", marker=dict(size=6, opacity=0.65)))
        fig.add_trace(go.Scatter(x=X[sort_idx], y=y_true[sort_idx], mode="lines", name="Ground truth", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=X[sort_idx], y=y_pred_all[sort_idx], mode="lines", name=f"Model (deg={degree})", line=dict(width=3)))
        fig.update_layout(
            title=f"Polynomial Ridge | Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}",
            xaxis_title="x",
            yaxis_title="y",
        )
    else:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=seed)
        x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 160), np.linspace(x1_min, x1_max, 160))
        grid = np.c_[xx.ravel(), yy.ravel()]

        if model_name == "Logistic Regression":
            w, b, losses = train_logistic_regression(X_train, y_train, lr=lr, iterations=iterations, lam=lam)
            zz = sigmoid(grid @ w + b).reshape(xx.shape)
            train_pred = (sigmoid(X_train @ w + b) >= 0.5).astype(int)
            val_pred = (sigmoid(X_val @ w + b) >= 0.5).astype(int)
            train_err = _safe_scalar(1 - np.mean(train_pred == y_train), fallback=1.0)
            val_err = _safe_scalar(1 - np.mean(val_pred == y_val), fallback=1.0)
            metrics = {"train_error": train_err, "val_error": val_err, "label": "Error Rate", "losses": losses}
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x0_min, x0_max, 160),
                    y=np.linspace(x1_min, x1_max, 160),
                    z=zz,
                    colorscale="RdBu",
                    opacity=0.55,
                    contours=dict(showlines=False),
                    showscale=False,
                    name="Decision surface",
                )
            )
            fig.update_layout(title=f"Logistic Regression | Train err={train_err:.3f}, Val err={val_err:.3f}")
        else:
            zz = knn_predict(X_train, y_train, grid, k=k).reshape(xx.shape)
            train_pred = knn_predict(X_train, y_train, X_train, k=k)
            val_pred = knn_predict(X_train, y_train, X_val, k=k)
            train_err = _safe_scalar(1 - np.mean(train_pred == y_train), fallback=1.0)
            val_err = _safe_scalar(1 - np.mean(val_pred == y_val), fallback=1.0)
            metrics = {"train_error": train_err, "val_error": val_err, "label": "Error Rate"}
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x0_min, x0_max, 160),
                    y=np.linspace(x1_min, x1_max, 160),
                    z=zz,
                    colorscale="RdBu",
                    opacity=0.45,
                    contours=dict(showlines=False),
                    showscale=False,
                    name="Decision regions",
                )
            )
            fig.update_layout(title=f"KNN (k={k}) | Train err={train_err:.3f}, Val err={val_err:.3f}")

        fig.add_trace(
            go.Scatter(
                x=X_train[:, 0],
                y=X_train[:, 1],
                mode="markers",
                marker=dict(color=y_train, colorscale="RdBu", line=dict(color="black", width=1)),
                name="Train",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=X_val[:, 0],
                y=X_val[:, 1],
                mode="markers",
                marker=dict(symbol="square", color=y_val, colorscale="RdBu", line=dict(color="black", width=1)),
                name="Validation",
            )
        )
        fig.update_layout(xaxis_title="Feature 1", yaxis_title="Feature 2")

    fig.update_layout(template="plotly_white", height=560, width=980, legend=dict(orientation="h"))
    return fig, metrics


def _build_validation_curve(model_name, n_samples, noise, degree, lam, k, seed, val_ratio, lr, iterations):
    split_ratio = float(np.clip(val_ratio, 0.1, 0.5))
    fig = go.Figure()
    if model_name == "Polynomial Regression":
        X, y, _ = make_regression_data(n_samples=n_samples, noise=noise, seed=seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=seed)
        complexity = list(range(1, 13))
        train_scores = []
        val_scores = []
        for deg in complexity:
            tr_pred = polynomial_ridge_predict(X_train, y_train, X_train, degree=deg, lam=lam)
            va_pred = polynomial_ridge_predict(X_train, y_train, X_val, degree=deg, lam=lam)
            train_scores.append(_safe_scalar(np.mean((y_train - tr_pred) ** 2), fallback=1e3))
            val_scores.append(_safe_scalar(np.mean((y_val - va_pred) ** 2), fallback=1e3))
        x_label = "Polynomial Degree"
        title = "Validation Curve: Polynomial Degree"
    elif model_name == "Logistic Regression":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=seed)
        complexity = np.logspace(-4, 1, 14)
        train_scores = []
        val_scores = []
        for lam_i in complexity:
            w, b, _ = train_logistic_regression(X_train, y_train, lr=lr, iterations=iterations, lam=lam_i)
            tr = (sigmoid(X_train @ w + b) >= 0.5).astype(int)
            va = (sigmoid(X_val @ w + b) >= 0.5).astype(int)
            train_scores.append(_safe_scalar(1 - np.mean(tr == y_train), fallback=1.0))
            val_scores.append(_safe_scalar(1 - np.mean(va == y_val), fallback=1.0))
        x_label = "Lambda (log scale)"
        title = "Validation Curve: Logistic Regularization"
    else:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=seed)
        complexity = list(range(1, 31))
        train_scores = []
        val_scores = []
        for k_i in complexity:
            tr = knn_predict(X_train, y_train, X_train, k=k_i)
            va = knn_predict(X_train, y_train, X_val, k=k_i)
            train_scores.append(_safe_scalar(1 - np.mean(tr == y_train), fallback=1.0))
            val_scores.append(_safe_scalar(1 - np.mean(va == y_val), fallback=1.0))
        x_label = "k"
        title = "Validation Curve: KNN Neighbors"

    fig.add_trace(go.Scatter(x=complexity, y=train_scores, mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x=complexity, y=val_scores, mode="lines+markers", name="Validation"))
    if model_name == "Logistic Regression":
        fig.update_xaxes(type="log")
    fig.update_layout(
        template="plotly_white",
        height=360,
        width=980,
        title=title,
        xaxis_title=x_label,
        yaxis_title="Error" if model_name != "Polynomial Regression" else "MSE",
    )
    return fig


def _build_bias_variance_figure(n_samples, noise, lam, seed, rounds):
    X_full, y_noisy_full, y_true_full = make_regression_data(n_samples=n_samples, noise=noise, seed=seed)
    rng = np.random.default_rng(seed)
    degrees = [1, 3, 5, 9, 12]
    bias_sq_vals = []
    var_vals = []
    total_vals = []
    for deg in degrees:
        preds = []
        for _ in range(rounds):
            idx = rng.choice(len(X_full), size=max(20, int(0.65 * len(X_full))), replace=True)
            x_boot = X_full[idx]
            y_boot = y_noisy_full[idx]
            preds.append(polynomial_ridge_predict(x_boot, y_boot, X_full, degree=deg, lam=lam))
        mat = np.array(preds)
        mean_pred = np.mean(mat, axis=0)
        bias_sq = _safe_scalar(np.mean((mean_pred - y_true_full) ** 2), fallback=1e3)
        var = _safe_scalar(np.mean(np.var(mat, axis=0)), fallback=1e3)
        bias_sq_vals.append(bias_sq)
        var_vals.append(var)
        total_vals.append(bias_sq + var)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=bias_sq_vals, mode="lines+markers", name="Bias^2"))
    fig.add_trace(go.Scatter(x=degrees, y=var_vals, mode="lines+markers", name="Variance"))
    fig.add_trace(go.Scatter(x=degrees, y=total_vals, mode="lines+markers", name="Bias^2 + Variance"))
    fig.update_layout(
        template="plotly_white",
        height=360,
        width=980,
        title="Bias-Variance Decomposition (Bootstrapped)",
        xaxis_title="Polynomial Degree",
        yaxis_title="Error Component",
    )
    return fig


def launch_dashboard():
    """Launch an interactive DataFitLab dashboard in Jupyter."""
    model = widgets.ToggleButtons(
        options=["Polynomial Regression", "Logistic Regression", "KNN"],
        value="Polynomial Regression",
        description="Model",
    )
    n_samples = widgets.IntSlider(value=180, min=80, max=500, step=20, description="Samples")
    noise = widgets.FloatSlider(value=0.25, min=0.0, max=0.8, step=0.05, description="Noise")
    val_ratio = widgets.FloatSlider(value=0.2, min=0.1, max=0.5, step=0.05, description="Val Split")
    degree = widgets.IntSlider(value=5, min=1, max=12, step=1, description="Degree")
    lam = widgets.FloatLogSlider(value=0.01, base=10, min=-4, max=1, step=0.1, description="Lambda")
    k = widgets.IntSlider(value=5, min=1, max=30, step=1, description="k")
    lr = widgets.FloatLogSlider(value=0.1, base=10, min=-3, max=0, step=0.1, description="LR")
    iterations = widgets.IntSlider(value=700, min=100, max=2000, step=100, description="Iters")
    seed = widgets.IntSlider(value=42, min=1, max=1000, step=1, description="Seed")
    rounds = widgets.IntSlider(value=80, min=20, max=180, step=10, description="B-V Rounds")
    auto_refresh = widgets.Checkbox(value=True, description="Auto refresh")
    refresh_btn = widgets.Button(description="Render", button_style="primary")

    metric_html = widgets.HTML()
    out_main = widgets.Output()
    out_curve = widgets.Output()
    out_bias_variance = widgets.Output()

    sweep_slider = widgets.IntSlider(value=1, min=1, max=12, step=1, description="Sweep")
    sweep_play = widgets.Play(value=1, min=1, max=12, step=1, interval=350, description="Play")
    widgets.jslink((sweep_play, "value"), (sweep_slider, "value"))
    sweep_box = widgets.HBox([sweep_play, sweep_slider])

    def _sync_model_controls():
        if model.value == "Polynomial Regression":
            degree.disabled = False
            k.disabled = True
            iterations.disabled = True
            lr.disabled = True
            sweep_slider.max = 12
            sweep_slider.description = "Degree Sweep"
        elif model.value == "KNN":
            degree.disabled = True
            k.disabled = False
            iterations.disabled = True
            lr.disabled = True
            sweep_slider.max = 30
            sweep_slider.description = "k Sweep"
        else:
            degree.disabled = True
            k.disabled = True
            iterations.disabled = False
            lr.disabled = False
            sweep_slider.max = 30
            sweep_slider.description = "Lambda Index"
        sweep_play.max = sweep_slider.max

    def _render_main():
        out_main.clear_output(wait=True)
        current_degree = degree.value
        current_k = k.value
        current_lam = lam.value
        if model.value == "Polynomial Regression":
            current_degree = sweep_slider.value
        elif model.value == "KNN":
            current_k = sweep_slider.value
        else:
            lambda_grid = np.logspace(-4, 1, 30)
            current_lam = float(lambda_grid[sweep_slider.value - 1])
        with out_main:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fig, m = _build_main_figure(
                        model.value,
                        n_samples.value,
                        noise.value,
                        current_degree,
                        current_lam,
                        current_k,
                        seed.value,
                        val_ratio.value,
                        lr.value,
                        iterations.value,
                    )
                fig.show()
            except Exception as exc:
                m = {"label": "Error", "train_error": 0.0, "val_error": 0.0}
                print(f"Render error in Model View: {exc}")
        metric_html.value = (
            f"<b>{m.get('label', 'Metric')}</b>: "
            f"Train={m.get('train_error', 0.0):.4f}, Val={m.get('val_error', 0.0):.4f} | "
            f"Generalization Gap={m.get('val_error', 0.0)-m.get('train_error', 0.0):.4f}"
        )

    def _render_curve():
        out_curve.clear_output(wait=True)
        with out_curve:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fig = _build_validation_curve(
                        model.value,
                        n_samples.value,
                        noise.value,
                        degree.value,
                        lam.value,
                        k.value,
                        seed.value,
                        val_ratio.value,
                        lr.value,
                        iterations.value,
                    )
                fig.show()
            except Exception as exc:
                print(f"Render error in Validation Curves: {exc}")

    def _render_bias_variance():
        out_bias_variance.clear_output(wait=True)
        with out_bias_variance:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fig = _build_bias_variance_figure(
                        n_samples.value,
                        noise.value,
                        lam.value,
                        seed.value,
                        rounds.value,
                    )
                fig.show()
            except Exception as exc:
                print(f"Render error in Bias-Variance tab: {exc}")

    def _render_all(_=None):
        _sync_model_controls()
        _render_main()
        _render_curve()
        _render_bias_variance()

    def _maybe_render(change):
        if auto_refresh.value:
            _render_all(change)

    refresh_btn.on_click(_render_all)
    for ctrl in [
        model, n_samples, noise, val_ratio, degree, lam, k, lr, iterations, seed, rounds, sweep_slider
    ]:
        ctrl.observe(_maybe_render, names="value")

    _render_all()

    control_col = widgets.VBox(
        [
            widgets.HTML("<h3>DataFitLab Interactive Dashboard</h3>"),
            model,
            n_samples,
            noise,
            val_ratio,
            lam,
            degree,
            k,
            lr,
            iterations,
            rounds,
            seed,
            sweep_box,
            widgets.HBox([auto_refresh, refresh_btn]),
            metric_html,
        ],
        layout=widgets.Layout(width="360px"),
    )
    tabs = widgets.Tab(children=[out_main, out_curve, out_bias_variance])
    tabs.set_title(0, "Model View")
    tabs.set_title(1, "Validation Curves")
    tabs.set_title(2, "Bias-Variance")
    display(widgets.HBox([control_col, tabs]))


if __name__ == "__main__":
    print("Run in Jupyter:")
    print("from interactive_dashboard import launch_dashboard")
    print("launch_dashboard()")
