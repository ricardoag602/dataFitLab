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


