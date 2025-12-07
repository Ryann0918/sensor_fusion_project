import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.patches import Ellipse

FOCAL_LENGTH = 524.7240  # pixels from Task 3
QR_HEIGHT = 11.5  
TRUE_POSITION = np.array([59.5, 26.0, 90.0 * np.pi / 180])


SIGMA_HEIGHT = 2.0  # pixels
SIGMA_CX = 3.0  # pixels

print("=" * 70)
print("Task 5: Static Localization - Final Solution")
print("=" * 70)


# data load
qr_positions_file = 'C:\\Users\\huang\\OneDrive\\Desktop\\sensor_fusion_project\\data\\qr_code_position_in_global_coordinate.csv'
qr_data = np.loadtxt(qr_positions_file, delimiter=',', skiprows=1)
QR_POSITIONS = {}
for row in qr_data:
    qr_id = int(row[0])
    QR_POSITIONS[qr_id] = np.array([row[1], row[2]])

data_path = 'C:\\Users\\huang\\OneDrive\\Desktop\\sensor_fusion_project\\data\\task5\\camera_loxalization_task5.csv'
all_data = np.loadtxt(data_path, delimiter=',')
first_frame = all_data[all_data[:, 0] == all_data[0, 0], :]

qr_numbers = first_frame[:, 1].astype(int)
cx_measured = first_frame[:, 2]
height_measured = first_frame[:, 5]

qr_global_positions = np.array([QR_POSITIONS[qr_id] for qr_id in qr_numbers])
n_qr = len(qr_numbers)

print(f"\nLoad data:")
print(f"  Tested {n_qr} QR codes: {qr_numbers}")
print(f"  FOCUS f = {FOCAL_LENGTH:.4f} pixels")
print(f"  QR code height h0 = {QR_HEIGHT} cm")
print(f"  SIGMA_HEIGHT = {SIGMA_HEIGHT} px, sigma_cx = {SIGMA_CX} px")



# measurement 

def measurement_model(state, qr_positions):

    p_x, p_y, psi = state
    n = qr_positions.shape[0]
    predictions = np.zeros(2 * n)
    
    for i in range(n):
        s_x, s_y = qr_positions[i]
        
        # distance
        d_i = np.sqrt((s_x - p_x)**2 + (s_y - p_y)**2)
        
        # angle
        alpha_i = np.arctan2(s_y - p_y, s_x - p_x)
        phi_i = alpha_i - psi
        
        # test prediction
        predictions[2*i] = (QR_HEIGHT * FOCAL_LENGTH) / d_i
        predictions[2*i + 1] = FOCAL_LENGTH * np.tan(phi_i)
    
    return predictions


def jacobian_matrix(state, qr_positions):
    
    p_x, p_y, psi = state
    n = qr_positions.shape[0]
    J = np.zeros((2 * n, 3))
    
    for i in range(n):
        s_x, s_y = qr_positions[i]
        
        dx = s_x - p_x
        dy = s_y - p_y
        d = np.sqrt(dx**2 + dy**2)
        d_sq = d * d
        d_cube = d * d * d
        
        # angle
        alpha = np.arctan2(dy, dx)
        phi = alpha - psi
        
        
        J[2*i, 0] = (QR_HEIGHT * FOCAL_LENGTH) * dx / d_cube
        J[2*i, 1] = (QR_HEIGHT * FOCAL_LENGTH) * dy / d_cube
        J[2*i, 2] = 0 
        
        sec_sq = 1 / (np.cos(phi)**2)
        
        J[2*i + 1, 0] = FOCAL_LENGTH * sec_sq * (-dy / d_sq)
        J[2*i + 1, 1] = FOCAL_LENGTH * sec_sq * (dx / d_sq)
        J[2*i + 1, 2] = FOCAL_LENGTH * sec_sq * (-1)
    
    return J


y_measured = np.zeros(2 * n_qr)
for i in range(n_qr):
    y_measured[2*i] = height_measured[i]
    y_measured[2*i + 1] = cx_measured[i]

R_diag = np.zeros(2 * n_qr)
for i in range(n_qr):
    R_diag[2*i] = SIGMA_HEIGHT**2
    R_diag[2*i + 1] = SIGMA_CX**2

R = np.diag(R_diag)
W = np.diag(1.0 / R_diag)  # W = R^{-1}

print("weight matrix:")
print(f"  R = diag([SIGMA_h^2, SIGMA_cx^2, ...])  (shape: {R.shape})")
print(f"  W = R^{{-1}}  (shape: {W.shape})")


def weighted_residual(state):

    predictions = measurement_model(state, qr_global_positions)
    residual = y_measured - predictions
    
    # weighted sqrt(W) * residual
    sqrt_W = np.sqrt(np.diag(W))
    weighted_res = sqrt_W * residual
    
    return weighted_res


def weighted_jacobian(state):
   
    J = jacobian_matrix(state, qr_global_positions)
    sqrt_W = np.sqrt(np.diag(W))
    
    weighted_J = sqrt_W[:, np.newaxis] * J
    
    return weighted_J


# Initail estimate
avg_height = np.mean(height_measured)
estimated_distance = (QR_HEIGHT * FOCAL_LENGTH) / avg_height
qr_x_mean = np.mean(qr_global_positions[:, 0])

x_init = np.array([qr_x_mean, 121.5 - estimated_distance, 90.0 * np.pi / 180])

print(f"\n Initial estimate:")
print(f"  x^(0) = [{x_init[0]:.2f}, {x_init[1]:.2f}, {x_init[2]*180/np.pi:.2f}°]")

print("use Levenberg-Marquardt...")

result = least_squares(
    weighted_residual,
    x_init,
    jac=weighted_jacobian,  
    method='lm',
    verbose=2
)

x_est = result.x

print(f"\nFinish!")
print(f"  success: {result.success}")
print(f"  message: {result.message}")
print(f"  Number of function evaluations: {result.nfev}")
print(f"  Number of Jacobian evaluations: {result.njev}")
print(f"  Final cost: {result.cost:.6f}")


J_final = jacobian_matrix(x_est, qr_global_positions)


print("Jacobian Matrix J at final estimate")
print(f"Matrix dimension: {J_final.shape} = (2*{n_qr} measurements, 3 states)")
print(f"State vector: x = [p_x, p_y, psi]")
print(f"Measurement vector: y = [h_1, Cx_1, h_2, Cx_2, ..., h_n, Cx_n]")
print(f"\n{'':>12} {'d/dp_x':>12} {'d/dp_y':>12} {'d/dpsi':>12}")

for i, qr_id in enumerate(qr_numbers):
    print(f"dh_{qr_id:<2}/dx   {J_final[2*i, 0]:>12.4f} {J_final[2*i, 1]:>12.4f} {J_final[2*i, 2]:>12.4f}")
    print(f"dCx_{qr_id:<1}/dx   {J_final[2*i+1, 0]:>12.4f} {J_final[2*i+1, 1]:>12.4f} {J_final[2*i+1, 2]:>12.4f}")



H = J_final.T @ W @ J_final  # Information matrix
P = np.linalg.inv(H)  # Covariance matrix

print(f"\nCovariance matrix P:")
print(P)
print(f"\nStandard deviations:")
print(f"  SIGMA_x = {np.sqrt(P[0, 0]):.4f} cm")
print(f"  SIGMA_y = {np.sqrt(P[1, 1]):.4f} cm")
print(f"  SIGMA_ψ = {np.sqrt(P[2, 2])*180/np.pi:.4f}°")


print(f"\nFinal estimate:")
print(f"  x* = [{x_est[0]:.2f}, {x_est[1]:.2f}, {x_est[2]*180/np.pi:.2f}°]")

error = x_est - TRUE_POSITION
print(f"\nError from true position:")
print(f"  delta x = {error[0]:.2f} cm")
print(f"  delta y = {error[1]:.2f} cm")
print(f"  delta ψ = {error[2] * 180 / np.pi:.2f}°")
print(f"  location error = {np.sqrt(error[0]**2 + error[1]**2):.2f} cm")


final_residuals = weighted_residual(x_est)
print(f"\nResidual statistics:")
print(f"  Residual norm: {np.linalg.norm(final_residuals):.4f}")
print(f"  Mean residual: {np.mean(np.abs(final_residuals)):.4f}")
print(f"  Max residual: {np.max(np.abs(final_residuals)):.4f}")


fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Position plot
ax1 = axes[0, 0]
ax1.plot([0, 121.5, 121.5, 0, 0], [0, 0, 121.5, 121.5, 0], 'k-', linewidth=2)
ax1.scatter(qr_global_positions[:, 0], qr_global_positions[:, 1], 
           c='blue', s=150, marker='s', label='QR Codes', zorder=3)

for i, qr_id in enumerate(qr_numbers):
    ax1.text(qr_global_positions[i, 0], qr_global_positions[i, 1] - 3, 
            f'{qr_id}', fontsize=8, ha='center')


ax1.scatter(x_est[0], x_est[1], c='red', s=300, marker='o', label='Estimated', zorder=5)

eigenvalues, eigenvectors = np.linalg.eig(P[:2, :2])
angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2σ
ellipse = Ellipse((x_est[0], x_est[1]), width, height, angle=angle,
                 facecolor='red', alpha=0.2, edgecolor='red', linewidth=2)
ax1.add_patch(ellipse)

ax1.scatter(TRUE_POSITION[0], TRUE_POSITION[1], c='green', s=300, marker='x', 
           linewidths=3, label='True', zorder=5)


arrow_len = 15
ax1.arrow(x_est[0], x_est[1], arrow_len * np.cos(x_est[2]), arrow_len * np.sin(x_est[2]),
         head_width=3, head_length=2, fc='red', ec='red', linewidth=2, zorder=4)
ax1.arrow(TRUE_POSITION[0], TRUE_POSITION[1], arrow_len * np.cos(TRUE_POSITION[2]), 
         arrow_len * np.sin(TRUE_POSITION[2]),
         head_width=3, head_length=2, fc='green', ec='green', linewidth=2, 
         linestyle='--', alpha=0.7, zorder=4)

ax1.set_xlabel('x (cm)', fontsize=12)
ax1.set_ylabel('y (cm)', fontsize=12)
ax1.set_title(f'Robot Position (Error: {np.sqrt(error[0]**2 + error[1]**2):.2f} cm)', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 125)
ax1.set_ylim(-5, 125)

 
ax2 = axes[0, 1]
unweighted_residuals = y_measured - measurement_model(x_est, qr_global_positions)
h_residuals = unweighted_residuals[::2]
cx_residuals = unweighted_residuals[1::2]

x_pos = np.arange(n_qr)
width = 0.35

ax2.bar(x_pos - width/2, h_residuals, width, label='Height residual', alpha=0.8)
ax2.bar(x_pos + width/2, cx_residuals, width, label='Cx residual', alpha=0.8)

ax2.set_xlabel('QR Code Index', fontsize=12)
ax2.set_ylabel('Residual (pixels)', fontsize=12)
ax2.set_title('Measurement Residuals', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'QR-{qr_id}' for qr_id in qr_numbers], rotation=45)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Covariance matrix P
ax3 = axes[1, 0]
im = ax3.imshow(P, cmap='RdBu_r', aspect='auto')
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(['$p_x$', '$p_y$', '$\\psi$'], fontsize=12)
ax3.set_yticklabels(['$p_x$', '$p_y$', '$\\psi$'], fontsize=12)
ax3.set_title('Covariance Matrix P', fontsize=12)


for i in range(3):
    for j in range(3):
        text = ax3.text(j, i, f'{P[i, j]:.2e}',
                       ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax3)


ax4 = axes[1, 1]
predictions = measurement_model(x_est, qr_global_positions)
h_pred = predictions[::2]
cx_pred = predictions[1::2]

ax4.scatter(height_measured, h_pred, c='blue', s=100, alpha=0.6, label='Height')
ax4.scatter(cx_measured, cx_pred, c='red', s=100, alpha=0.6, label='Cx')


all_vals = np.concatenate([height_measured, cx_measured, h_pred, cx_pred])
min_val, max_val = all_vals.min(), all_vals.max()
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal')

ax4.set_xlabel('Measured (pixels)', fontsize=12)
ax4.set_ylabel('Predicted (pixels)', fontsize=12)
ax4.set_title('Measurement vs Prediction', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

plt.tight_layout()
output_path = 'C:\\Users\\huang\\OneDrive\\Desktop\\sensor_fusion_project\\output\\part2\\task5_final_result.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nResult image saved: {output_path}")


results_file = 'C:\\Users\\huang\\OneDrive\\Desktop\\sensor_fusion_project\\output\\part2\\task5_final_results.txt'
with open(results_file, 'w') as f:
    
    f.write("Task 5: Static Localization - Final Results\n")
    
    
    f.write("INPUTS:\n")
    f.write(f"  Focal length: {FOCAL_LENGTH:.4f} pixels\n")
    f.write(f"  QR height: {QR_HEIGHT} cm\n")
    f.write(f"  Detected QR codes: {n_qr}\n")
    f.write(f"  Measurement noise: SIGMA_h={SIGMA_HEIGHT} px, SIGMA_cx={SIGMA_CX} px\n\n")
    
    f.write("TRUE POSITION:\n")
    f.write(f"  x = {TRUE_POSITION[0]:.2f} cm\n")
    f.write(f"  y = {TRUE_POSITION[1]:.2f} cm\n")
    f.write(f"  ψ = {TRUE_POSITION[2] * 180 / np.pi:.2f}°\n\n")
    
    f.write("ESTIMATED POSITION:\n")
    f.write(f"  x = {x_est[0]:.2f} cm\n")
    f.write(f"  y = {x_est[1]:.2f} cm\n")
    f.write(f"  ψ = {x_est[2] * 180 / np.pi:.2f}°\n\n")
    
    f.write("ERRORS:\n")
    f.write(f"  delta x = {error[0]:.2f} cm\n")
    f.write(f"  delta y = {error[1]:.2f} cm\n")
    f.write(f"  delta ψ = {error[2] * 180 / np.pi:.2f}°\n")
    f.write(f"  Position error = {np.sqrt(error[0]**2 + error[1]**2):.2f} cm\n\n")
    
    f.write("COVARIANCE (P):\n")
    f.write(f"  sigma_x = {np.sqrt(P[0, 0]):.4f} cm\n")
    f.write(f"  sigma_y = {np.sqrt(P[1, 1]):.4f} cm\n")
    f.write(f"  sigma_ψ = {np.sqrt(P[2, 2])*180/np.pi:.4f}°\n\n")
    
    f.write("OPTIMIZATION:\n")
    f.write(f"  Method: Levenberg-Marquardt\n")
    f.write(f"  Success: {result.success}\n")
    f.write(f"  Function evaluations: {result.nfev}\n")
    f.write(f"  Jacobian evaluations: {result.njev}\n")
    f.write(f"  Final cost: {result.cost:.6f}\n\n")
    
    f.write("RESIDUAL STATISTICS:\n")
    f.write(f"  Residual norm: {np.linalg.norm(final_residuals):.4f}\n")
    f.write(f"  Mean residual: {np.mean(np.abs(final_residuals)):.4f}\n")
    f.write(f"  Max residual: {np.max(np.abs(final_residuals)):.4f}\n\n")

 
    f.write("JACOBIAN MATRIX J at final estimate\n")
    f.write(f"Matrix dimension: {J_final.shape} = (2*{n_qr} measurements, 3 states)\n")
    f.write(f"State vector: x = [p_x, p_y, psi]\n")
    f.write(f"Measurement vector: y = [h_1, Cx_1, h_2, Cx_2, ..., h_n, Cx_n]\n\n")
    f.write(f"{'':>12} {'d/dp_x':>12} {'d/dp_y':>12} {'d/dpsi':>12}\n")
    for i, qr_id in enumerate(qr_numbers):
        f.write(f"dh_{qr_id:<2}/dx   {J_final[2*i, 0]:>12.4f} {J_final[2*i, 1]:>12.4f} {J_final[2*i, 2]:>12.4f}\n")
        f.write(f"dCx_{qr_id:<1}/dx   {J_final[2*i+1, 0]:>12.4f} {J_final[2*i+1, 1]:>12.4f} {J_final[2*i+1, 2]:>12.4f}\n\n")
    
    
    f.write("JACOBIAN MATRIX DERIVATION:\n")
    f.write("Measurement model:\n")
    f.write("  h_i = (h0 * f) / d_i,  where d_i = sqrt[(s_x - p_x)^2 + (s_y - p_y)^2]\n")
    f.write("  Cx_i = f * tan(phi_i), where phi_i = arctan((s_y - p_y)/(s_x - p_x)) - psi\n\n")
    f.write("Jacobian elements:\n")
    f.write("  dh_i/dp_x  = (h0 * f) * (s_x - p_x) / d_i^3\n")
    f.write("  dh_i/dp_y  = (h0 * f) * (s_y - p_y) / d_i^3\n")
    f.write("  dh_i/dpsi  = 0\n\n")
    f.write("  dCx_i/dp_x = f * sec^2(phi_i) * [-(s_y - p_y) / d_i^2]\n")
    f.write("  dCx_i/dp_y = f * sec^2(phi_i) * [(s_x - p_x) / d_i^2]\n")
    f.write("  dCx_i/dpsi = -f * sec^2(phi_i)\n")


print(f"Detailed results saved: {results_file}")
print("Task 5 completed")

