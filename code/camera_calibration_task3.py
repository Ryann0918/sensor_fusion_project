
"""
Task 3 Camera Module Calibration

Objective: Determine camera's focal length f (pixels) and bias b
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


# Step 1: Data Loading and Preprocessing
# Read calibration data
data = pd.read_csv('D:\\Lesson\\sensor fusion\\dataset3\\dataset3\\data\\task3\\camera_module_calibration_task3.csv', 
                   header=None, names=['distance_measured', 'height_pixels'])

print("="*70)
print("Task 3: Camera Module Calibration")
print("="*70)
print("\nRaw Data:")
print(data.head(10))
print(f"\nNumber of data points: {len(data)}")


# Step 2: Distance Correction
# According to readme.txt, need to add the following distances:
# The Distance of camera pinhole to IR detector = 1.7 cm
# The Distance of wall to wooden list = 5 cm

offset = 1.7 + 5.0  # Total offset = 6.7 cm

# Calculate true distance x_3
data['x3_true'] = data['distance_measured'] + offset

print("\n" + "="*70)
print("Distance Correction")
print("="*70)
print(f"Camera offset: {offset} cm")
print("\nCorrected data:")
print(data[['distance_measured', 'height_pixels', 'x3_true']].head(10))


# Step 3a: Linear Least-Squares Regression
# Mathematical Model:
# x_3 = (h_0 * f) / h + b
# 
# Linearization:
# x_3 = k * (1/h) + b
# where k = h_0 * f

# Calculate 1/h (inverse pixel height)
data['inv_h'] = 1.0 / data['height_pixels']

# X-axis variable: 1/h
# Y-axis variable: x_3
X = data['inv_h'].values
Y = data['x3_true'].values

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

print("\n" + "="*70)
print("Step 3a: Linear Least-Squares Regression")
print("="*70)
print(f"\nRegression equation: x_3 = k * (1/h) + b")
print(f"\nGradient k: {slope:.4f} cm·pixel")
print(f"Bias b: {intercept:.4f} cm")
print(f"\nR-squared: {r_value**2:.6f}")
print(f"Standard error: {std_err:.6f}")
print(f"p-value: {p_value:.2e}")


# Step 3b: Focal Length Calculation
# Actual height of QR code
h0 = 11.5  # cm

# Calculate focal length
# From formula k = h_0 * f, we get: f = k / h_0
focal_length = slope / h0

print("\n" + "="*70)
print("Step 3b: Focal Length Calculation")
print("="*70)
print(f"\nActual QR code height h₀: {h0} cm")
print(f"Focal length f: {focal_length:.4f} pixels")
print(f"Bias b: {intercept:.4f} cm")


# Step 4: Model Validation and Visualization
# Predict distances using calibrated parameters
data['x3_predicted'] = (h0 * focal_length) / data['height_pixels'] + intercept

# Calculate residuals
data['residual'] = data['x3_true'] - data['x3_predicted']
rmse = np.sqrt(np.mean(data['residual']**2))
max_error = np.max(np.abs(data['residual']))

print("\n" + "="*70)
print("Model Validation")
print("="*70)
print(f"\nRoot Mean Square Error (RMSE): {rmse:.4f} cm")
print(f"Maximum error: {max_error:.4f} cm")
print(f"Mean residual: {np.mean(data['residual']):.4f} cm")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Camera Module Calibration Analysis', 
             fontsize=13,fontweight='bold')

# Plot 1: Linear Regression Fit
ax1 = axes[0, 0]
ax1.scatter(X, Y, alpha=0.6, s=50, label='Actual data')
X_line = np.linspace(X.min(), X.max(), 100)
Y_line = slope * X_line + intercept
ax1.plot(X_line, Y_line, 'r-', linewidth=2, 
         label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
ax1.set_xlabel('1/h (1/pixels)', fontsize=10)
ax1.set_ylabel('x₃ (cm)', fontsize=10)
ax1.set_title(f'Linear Regression\nR² = {r_value**2:.6f}', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Actual Distance vs Predicted Distance
ax2 = axes[0, 1]
ax2.scatter(data['x3_true'], data['x3_predicted'], alpha=0.6, s=50)
min_val = min(data['x3_true'].min(), data['x3_predicted'].min())
max_val = max(data['x3_true'].max(), data['x3_predicted'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
         label='Perfect match')
ax2.set_xlabel('Actual distance x₃ (cm)', fontsize=10)
ax2.set_ylabel('Predicted distance (cm)', fontsize=10)
ax2.set_title('Actual vs Predicted', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

# Plot 3: Residual Analysis
ax3 = axes[1, 0]
ax3.scatter(data['x3_true'], data['residual'], alpha=0.6, s=50)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Actual distance x₃ (cm)', fontsize=10)
ax3.set_ylabel('Residual (cm)', fontsize=10)
ax3.set_title(f'Residual Plot RMSE = {rmse:.4f} cm', fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Pixel Height vs Distance
ax4 = axes[1, 1]
ax4.scatter(data['height_pixels'], data['x3_true'], alpha=0.6, s=50, 
           label='Actual data')
h_range = np.linspace(data['height_pixels'].min(), data['height_pixels'].max(), 100)
x3_model = (h0 * focal_length) / h_range + intercept
ax4.plot(h_range, x3_model, 'r-', linewidth=2, label='Calibrated model')
ax4.set_xlabel('Detected height h (pixels)', fontsize=10)
ax4.set_ylabel('Distance x₃ (cm)', fontsize=10)
ax4.set_title('Pixel Height vs Distance', fontsize=10)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:\\Lesson\\sensor fusion\\dataset3\\dataset3\\data\\task3\\output\\camera_calibration_analysis.png', 
            dpi=300, bbox_inches='tight')
print(f"\nPlot saved: camera_calibration_analysis.png")



# Save detailed results to CSV
results_df = data[['distance_measured', 'height_pixels', 'x3_true', 
                    'x3_predicted', 'residual']]
results_df.to_csv('D:\\Lesson\\sensor fusion\\dataset3\\dataset3\\data\\task3\\output\\calibration_results.csv', index=False)
print(f"Detailed results saved: calibration_results.csv")

# Save calibration parameters
with open('D:\\Lesson\\sensor fusion\\dataset3\\dataset3\\data\\task3\\output\\calibration_parameters.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("Camera Module Calibration Results\n")
    f.write("="*70 + "\n\n")
    
    f.write("Calibration Parameters:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Focal length f: {focal_length:.4f} pixels\n")
    f.write(f"Bias b: {intercept:.4f} cm\n")
    f.write(f"QR code height h₀: {h0} cm\n")
    f.write(f"Camera offset: {offset} cm\n\n")
    
    f.write("Regression Analysis Results:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Gradient k: {slope:.4f} cm·pixel\n")
    f.write(f"R-squared: {r_value**2:.6f}\n")
    f.write(f"Standard error: {std_err:.6f}\n")
    f.write(f"p-value: {p_value:.2e}\n\n")
    
    f.write("Model Performance:\n")
    f.write("-" * 70 + "\n")
    f.write(f"RMSE: {rmse:.4f} cm\n")
    f.write(f"Maximum error: {max_error:.4f} cm\n")
    f.write(f"Mean residual: {np.mean(data['residual']):.4f} cm\n\n")
    
    f.write("Calibration Formula:\n")
    f.write("-" * 70 + "\n")
    f.write(f"x₃ = (h₀ × f) / h + b\n")
    f.write(f"x₃ = ({h0} × {focal_length:.4f}) / h + {intercept:.4f}\n")
    f.write(f"x₃ = {h0 * focal_length:.4f} / h + {intercept:.4f}\n")

print(f"Calibration parameters saved: calibration_parameters.txt")


# Final Summary
print("\n" + "="*70)
print("Final Calibration Results Summary")
print("="*70)
print(f"\n  Focal length f: {focal_length:.4f} pixels")
print(f" Bias b: {intercept:.4f} cm")
print(f" Model accuracy R²: {r_value**2:.6f}")
print(f" RMSE: {rmse:.4f} cm")
print("\nCalibration completed!")
print("="*70)

plt.show()