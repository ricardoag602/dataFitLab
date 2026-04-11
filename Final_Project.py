#THIS PROJECT IS NOT FINISHED THIS IS A CURRENT OF THE TEMPLATE FILE IN JUPYTER
#DOCUMENTATION WILL BE ADDED LATER ON SO AS TO IMPROVE UNDERSTANDING
#EXPLANATION ON CODE WILL BE PRESENTED IN JUPYTER NOTEBOOK FILE

import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC

#----
#Step 1: Generate 2d "teaching tool" Dataset
#----
# Make_circles is a good starter because it is nonlinear and has a clear decision boundary
X_2d, y = make_circles(n_samples=200, factor=0.3, noise=0.05, random_state=42)

# Extract the X & Y coordinates for readability
x_coords = X_2d[:, 0]
y_coords = X_2d[:, 1]

#---
#Step 2: "Lifting" Logic (Manual Kernel Trick)
#---
# We create a new Z dimension based on the distance from the origin

#FORMULA: z =x^2 + y^2
z_coords = (x_coords ** 2) + (y_coords ** 2)

#Stack them together to create a new 3D dataset
X_3d = np.column_stack((x_coords, y_coords, z_coords))

#---
#Step 3: Train the SVM on the 3D data
#---

#We can use a linear kernel now because the data is linearly separable in 3D
# 'C' is the soft-margin parameter. 1.0 is a good default.
model = SVC(kernel='linear', C=1.0)
model.fit(X_3d, y)

#---
#Step 4: Extract the Hyperplane with the equation
#---

#note: the model gives the weights (W) and the intercept (b)
#Equation of the plane: (W0 * x) + (W1 * y) + (W2 * z) + b = 0

weights = model.coef_[0]
bias = model.intercept_[0]

w0, w1, w2 = weights

print("--- SVM Model Trained Successfully ---")
print(f"Weight Matrix (w): [{w0:.4f}, {w1:.4f}, {w2:.4f}]")
print(f"Bias / Intercept (b): {bias:.4f}")

#DAY 2 TASK: rearange the equation to solve for Z:
#z = -( (w0 * x) + (w1 * y) + b ) / w2
print("\nEquation for Day 2 3D Surface Plot:")
print(f"z = -(({w0:.4f} * x) + ({w1:.4f} * y) + {bias:.4f} ) / {w1:.4f}")

#---
#Step 5: Extracting the Support Vectors
#---

#1. Get the 3D coordinates of the support vectors
sv_coords_3d = model.support_vectors_

#2. Get the indices of these points from the original 'X_3d' dataset
sv_indices = model.support_

#3. Get the number of support vectors per class
sv_per_class = model.n_support_

print("\n--- Support Vector Informarion ---")
print(f"Total dataset size: {len(X_3d)} points")
print(f"Total Support Vectors: {len(sv_coords_3d)} points")
print(f"Support Vectors per Class: {sv_per_class}")
print(f"Indices of the first 5 Support Vectors: {sv_indices[:5]}")




import plotly.graph_objects as go

#---
# Step 1: Create the grid for the Hyperplane
#---
# Define the 'floor' space (x & y limits) so plotly knows
# how wide to draw the plane
x_min, x_max= x_coords.min() - 0.2, x_coords.max() + 0.2
y_min, y_max= y_coords.min() - 0.2, y_coords.max() + 0.2

# Create a 2d mesh-grid across that space
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                     np.linspace(y_min, y_max, 30))

# Calculate the corresponding Z height for every point on that mesh-grid
# (using the same equation from the SMV model)
zz = -(w0 * xx + w1 * yy + bias) / w2

#---
# Step 2: Build the Plotly Traces (layers)
#---
fig = go.Figure()

# Trace 1: The Base Dataset
# We map the colors 'y' (0 or 1) to distinguish the inner & outer circles
fig.add_trace(go.Scatter3d(
    x=x_coords, y=y_coords, z=z_coords,
    mode='markers',
    marker=dict(
        size=4,
        color=y,
        colorscale='Viridis',
        opacity=0.6

    ),
    name='Data Points',
    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
))

# Trace 2: Highlight the Support Vectors
# (Drawn slightly larger with a bright neon outline)
fig.add_trace(go.Scatter3d(
    x=sv_coords_3d[:, 0], y=sv_coords_3d[:, 1], z=sv_coords_3d[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color='rgba(0,0,0,0)',
        line=dict(color='red', width=3)
    ),
    name='Support Vectors',
    hoverinfo='skip'        # Skip hover to reduct clutter
    )
)

#Trace 3: The Hyperplane (Decision Boundary)
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    colorscale='Greys',
    opacity=0.5,            # Semi-transparent to points below
    showscale=False,        # Hide the colorbar for the plane
    name='Hyperplane'

))

#---
# Step 3: Polish the Layout and Render
#---
fig.update_layout(
    title='SVM Kernel Trick: 2d Circles Lifted to 3D Space',
    scene=dict(
        xaxis_title='X (Original)',
        yaxis_title='Y (Original)',
        zaxis_title='Z (Transformed: x^2 + y^2)',
    
    ),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=40) #Tight margins for cleaner look

)

fig.show()

#====================================================================
# LOGISTIC REGRESSION 
#====================================================================
# building binary logistic regression using only numpy
# no sklearn for the actual model, just for generating data

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

#---
# Step 1: Generate a different dataset for classification
#---
# using make_moons because its a classic nonlinear binary classification problem
# tried different noise values, 0.2 gives a good mix of challenge
X_lr, y_lr = make_moons(n_samples=200, noise=0.2, random_state=42)

print("\n--- Logistic Regression Dataset ---")
print(f"Shape of X: {X_lr.shape}")
print(f"Shape of y: {y_lr.shape}")

#---
# Step 2: Sigmoid function
#---
# takes any number and squashes it between 0 and 1
# FORMULA: sigma(z) = 1 / (1 + e^(-z))

def sigmoid(z):
    # had to add this clip because without it i was getting overflow warnings
    # when z is super negative, exp(-z) blows up
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

#---
# Step 3: Loss function (Binary Cross Entropy + L2)
#---
# this tells us how bad our predictions are
# FORMULA: L = -(1/n) * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ]

def compute_loss(y_true, y_pred, weights, lam=0.0):
    n = len(y_true)
    eps = 1e-12  # need this so we dont take log(0) which is -inf

    bce = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    # L2 penalty — this is the regularization part
    l2 = (lam / (2 * n)) * np.sum(weights ** 2)

    return bce + l2

#---
# Step 4: Training function (gradient descent)
#---
# gradient descent loop:
#   1. make predictions with current weights
#   2. calculate how wrong we are (gradients)
#   3. nudge the weights in the right direction
#   4. repeat

def train(X, y, lr=0.1, iterations=1000, lam=0.0):
    n_samples, n_features = X.shape

    # start with all zeros — tried random init but zeros works fine
    w = np.zeros(n_features)
    b = 0.0
    losses = []

    for i in range(iterations):
        #forward pass
        z = X @ w + b          # linear part: w1*x1 + w2*x2 + b
        y_hat = sigmoid(z)     # squash through sigmoid to get probabilities

        #compute loss (for tracking)
        loss = compute_loss(y, y_hat, w, lam)
        losses.append(loss)

        #gradients — these formulas come from the derivative of BCE
        error = y_hat - y
        dw = (1/n_samples) * (X.T @ error) + (lam/n_samples) * w   # L2 adds lam*w term
        db = (1/n_samples) * np.sum(error)

        #update weights (gradient descent step)
        w = w - lr * dw
        b = b - lr * db

        if i % 200 == 0:
            print(f"  iter {i:4d} | loss: {loss:.4f}")

    return w, b, losses

#---
# Step 5: Train the model
#---
print("\n--- Training (no regularization) ---")
w_lr, b_lr, loss_history = train(X_lr, y_lr, lr=0.1, iterations=1000, lam=0.0)

print(f"\nLearned weights: w1={w_lr[0]:.4f}, w2={w_lr[1]:.4f}")
print(f"Learned bias: {b_lr:.4f}")

# check accuracy
predictions = (sigmoid(X_lr @ w_lr + b_lr) >= 0.5).astype(int)
acc = np.mean(predictions == y_lr)
print(f"Training Accuracy: {acc:.2%}")

#---
# Step 6: Visualization — loss curve + decision boundary
#---
fig_lr, axes = plt.subplots(1, 2, figsize=(14, 5))

# plot 1: loss going down over time
axes[0].plot(loss_history, 'b-', linewidth=1.5)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss (BCE)')
axes[0].set_title('Training Loss Over Time')
axes[0].grid(True, alpha=0.3)

# plot 2: decision boundary
h = 0.02
x_min_lr, x_max_lr = X_lr[:, 0].min() - 0.5, X_lr[:, 0].max() + 0.5
y_min_lr, y_max_lr = X_lr[:, 1].min() - 0.5, X_lr[:, 1].max() + 0.5
xx_lr, yy_lr = np.meshgrid(np.arange(x_min_lr, x_max_lr, h),
                           np.arange(y_min_lr, y_max_lr, h))

grid_points = np.c_[xx_lr.ravel(), yy_lr.ravel()]
Z_lr = sigmoid(grid_points @ w_lr + b_lr).reshape(xx_lr.shape)

axes[1].contourf(xx_lr, yy_lr, Z_lr, levels=50, cmap='RdYlBu', alpha=0.6)
axes[1].contour(xx_lr, yy_lr, Z_lr, levels=[0.5], colors='black', linewidths=2)
axes[1].scatter(X_lr[:, 0], X_lr[:, 1], c=y_lr, cmap='RdYlBu', edgecolors='k', s=25)
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title(f'Decision Boundary (Accuracy: {acc:.0%})')

plt.suptitle('Logistic Regression From Scratch', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#---
# Step 7: L2 Regularization Comparison
#---
# testing what happens when we crank up the regularization parameter
print("\n--- Comparing L2 regularization ---")
lambdas = [0.0, 0.1, 1.0, 10.0]
fig_l2, axes_l2 = plt.subplots(1, 4, figsize=(18, 4))

for ax, lam in zip(axes_l2, lambdas):
    w_l, b_l, _ = train(X_lr, y_lr, lr=0.1, iterations=1000, lam=lam)

    Z_l = sigmoid(grid_points @ w_l + b_l).reshape(xx_lr.shape)

    preds = (sigmoid(X_lr @ w_l + b_l) >= 0.5).astype(int)
    a = np.mean(preds == y_lr)

    ax.contourf(xx_lr, yy_lr, Z_l, levels=50, cmap='RdYlBu', alpha=0.6)
    ax.contour(xx_lr, yy_lr, Z_l, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_lr[:, 0], X_lr[:, 1], c=y_lr, cmap='RdYlBu', edgecolors='k', s=20)
    ax.set_title(f'lambda={lam} (Acc: {a:.0%})')
    ax.set_xlabel('Feature 1')

    weight_size = np.sqrt(w_l[0]**2 + w_l[1]**2)
    print(f"  lambda={lam:5.1f} | acc={a:.2%} | weight magnitude={weight_size:.4f}")

axes_l2[0].set_ylabel('Feature 2')
plt.suptitle('Effect of L2 Regularization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


