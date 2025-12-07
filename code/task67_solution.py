"""
ELEC-E8740 Basic Sensor Fusion Project for Part II Task 6 and 7
Task 6: Dead Reckoning using IMU only
Task 7: EKF Tracking using IMU and Camera
Author: Haoran Cao
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# =============================================================================
# CALIBRATION PARAMETERS from Part I
# =============================================================================
FOCAL_LENGTH = 524.7240  # pixels from Task 3
QR_HEIGHT = 11.5  # cm
GYRO_BIAS_Z = 0  
ROBOT_SPEED_30PWM = 6.2307*0.75
# Initial position (from readme.txt)
X0 = 43.0  # cm
Y0 = 18.0  # cm
PHI0 = 0.0  # degrees (heading)

# Frame dimensions
FRAME_SIZE = 121.5  # cm

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("Part II: Localization and Tracking")
print("=" * 70)

# File paths - UPDATE THESE PATHS
imu_file = "D:\\Project\\sensor_fusion_project\\data\\task6-task7\\imu_tracking_task6.csv"
motor_file = "D:\\Project\\sensor_fusion_project\\data\\task6-task7\\motor_control_tracking_task6.csv"
camera_file = "D:\\Project\\sensor_fusion_project\\data\\task6-task7\\camera_tracking_task6.csv"
qr_positions_file = "D:\\Project\\sensor_fusion_project\\data\\qr_code_position_in_global_coordinate.csv"
# Load QR code positions
qr_data = np.loadtxt(qr_positions_file, delimiter=',', skiprows=1)
QR_POSITIONS = {}
for row in qr_data:
    qr_id = int(row[0])
    QR_POSITIONS[qr_id] = np.array([row[1], row[2]])

print(f"Loaded {len(QR_POSITIONS)} QR code positions")

# Load IMU data
# Columns are: timestamp, ax, ay, az, roll, pitch, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
imu_data = np.loadtxt(imu_file, delimiter=',')
imu_time = imu_data[:, 0]
gyro_z = imu_data[:, 8]  # the gyro_z in deg/s

print(f"Loaded IMU data: {len(imu_time)} samples")
print(f"  Time range: {imu_time[0]:.2f} - {imu_time[-1]:.2f} s")
print(f"  Duration: {imu_time[-1] - imu_time[0]:.2f} s")

# Load motor control data
# Columns are: timestamp, pwm_left, pwm_right
motor_data = np.loadtxt(motor_file, delimiter=',')
motor_time = motor_data[:, 0]
pwm_left = motor_data[:, 1]
pwm_right = motor_data[:, 2]

print(f"Loaded Motor data: {len(motor_time)} samples")

# Load Camera data
# Columns are: timestamp, qr_id, cx, cy, width, height, raw_distance, raw_angle
camera_data = np.loadtxt(camera_file, delimiter=',')
camera_time = camera_data[:, 0]

print(f"Loaded Camera data: {len(camera_time)} samples")

# =============================================================================
# TASK 6: DEAD RECKONING (IMU ONLY)
# =============================================================================
print("\n" + "=" * 70)
print("Task 6: Dead Reckoning using IMU and Motor Control")
print("=" * 70)

# Convert PWM to velocity

def pwm_to_velocity(pwm_left, pwm_right):
    avg_pwm = (pwm_left + pwm_right) / 2.0
    diff_pwm = abs(pwm_left - pwm_right)
    
    # 基础速度
    base_velocity = (avg_pwm / 0.3) * ROBOT_SPEED_30PWM
    
    # 转弯修正
    if diff_pwm > 0.10:  # 急转弯
        turn_factor = 0.92
    elif diff_pwm > 0.05:  # 中等转弯
        turn_factor = 0.96
    else:  # 直线
        turn_factor = 1.0
    
    return base_velocity * turn_factor * 0.84  # 0.84 = 标定修正
# Interpolate motor commands to IMU timestamps
motor_interp_left = interp1d(motor_time, pwm_left, kind='previous', 
                              bounds_error=False, fill_value=(pwm_left[0], pwm_left[-1]))
motor_interp_right = interp1d(motor_time, pwm_right, kind='previous',
                               bounds_error=False, fill_value=(pwm_right[0], pwm_right[-1]))

# Find common time range
t_start = max(imu_time[0], motor_time[0])
t_end = min(imu_time[-1], motor_time[-1])

# Filter IMU data to common time range
imu_mask = (imu_time >= t_start) & (imu_time <= t_end)
imu_time_filtered = imu_time[imu_mask]
gyro_z_filtered = gyro_z[imu_mask]

print(f"\nCommon time range: {t_start:.2f} - {t_end:.2f} s")
print(f"Filtered IMU samples: {len(imu_time_filtered)}")

# Dead reckoning using quasi-constant turn model

def dead_reckoning(time, gyro_z, motor_interp_left, motor_interp_right, x0, y0, phi0):
    n = len(time)
    trajectory = np.zeros((n, 3))
    trajectory[0] = [x0, y0, np.radians(phi0)]
    
    for i in range(1, n):
        dt = time[i] - time[i-1]
        
        # Get current state
        x, y, phi = trajectory[i-1]
        
        # Get velocity from motor PWM
        pwm_l = motor_interp_left(time[i-1])
        pwm_r = motor_interp_right(time[i-1])
        v = pwm_to_velocity(pwm_l, pwm_r)
        
        # Get angular velocity from gyroscope (corrected for bias)
        omega = np.radians(gyro_z[i-1] - GYRO_BIAS_Z)  # Convert to rad/s
        
        # Euler integration (quasi-constant turn model)
        # Update heading first
        phi_new = phi + omega * dt
        
        # Update position using average heading
        phi_avg = (phi + phi_new) / 2
        x_new = x + v * np.cos(phi_avg) * dt
        y_new = y + v * np.sin(phi_avg) * dt
        
        trajectory[i] = [x_new, y_new, phi_new]
    
    return trajectory

# Perform dead reckoning
print("\nPerforming dead reckoning...")
dr_trajectory = dead_reckoning(imu_time_filtered, gyro_z_filtered, 
                                motor_interp_left, motor_interp_right,
                                X0, Y0, PHI0)

print(f"Dead reckoning completed!")
print(f"  Start position: ({dr_trajectory[0, 0]:.2f}, {dr_trajectory[0, 1]:.2f}) cm")
print(f"  End position: ({dr_trajectory[-1, 0]:.2f}, {dr_trajectory[-1, 1]:.2f}) cm")
print(f"  Total heading change: {np.degrees(dr_trajectory[-1, 2] - dr_trajectory[0, 2]):.2f} deg")

# =============================================================================
# TASK 7: EKF TRACKING
# =============================================================================
print("\n" + "=" * 70)
print("Task 7: Extended Kalman Filter Using IMU and Camera")
print("=" * 70)

class ExtendedKalmanFilter:
    
    def __init__(self, x0, P0, Q, R_height, R_cx):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R_height = R_height
        self.R_cx = R_cx
        
    def predict(self, v, omega, dt):
        p_x, p_y, phi = self.x
        phi_new = phi + omega * dt
        phi_avg = (phi + phi_new) / 2
        p_x_new = p_x + v * np.cos(phi_avg) * dt
        p_y_new = p_y + v * np.sin(phi_avg) * dt
        
        self.x = np.array([p_x_new, p_y_new, phi_new])
        
        # Jacobian of state transition
        F = np.array([
            [1, 0, -v * np.sin(phi_avg) * dt],
            [0, 1,  v * np.cos(phi_avg) * dt],
            [0, 0,  1]
        ])
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        self.P = (self.P + self.P.T) / 2
        
    def update(self, qr_measurements):
        if len(qr_measurements) == 0:
            return
        
        p_x, p_y, phi = self.x
        
        valid_measurements = []
        for qr_id, cx_meas, h_meas in qr_measurements:
            if qr_id in QR_POSITIONS:
                valid_measurements.append((qr_id, cx_meas, h_meas))

        if len(valid_measurements) == 0:
            return

        n_meas = len(valid_measurements)

        # Build measurement vector and predicted measurements
        z = np.zeros(2 * n_meas) 
        z_pred = np.zeros(2 * n_meas)
        H = np.zeros((2 * n_meas, 3))

        R = np.zeros((2 * n_meas, 2 * n_meas))

        for i, (qr_id, cx_meas, h_meas) in enumerate(valid_measurements):
                
            s_x, s_y = QR_POSITIONS[qr_id]
            dx = s_x - p_x
            dy = s_y - p_y
            d = np.sqrt(dx**2 + dy**2)
            
            if d < 1e-6:
                continue
                
            alpha = np.arctan2(dy, dx)
            phi_rel = alpha - phi
            phi_rel = np.clip(phi_rel, -np.pi/2 + 0.1, np.pi/2 - 0.1)
            
            h_pred = (QR_HEIGHT * FOCAL_LENGTH) / d
            cx_pred = FOCAL_LENGTH * np.tan(phi_rel)
            
            z[2*i] = h_meas
            z[2*i + 1] = cx_meas
            z_pred[2*i] = h_pred
            z_pred[2*i + 1] = cx_pred
            
            # Jacobian of measurement model
            d_sq = d * d
            d_cube = d * d * d
            sec_sq = 1.0 / (np.cos(phi_rel)**2)
            
            # dh/dx
            H[2*i, 0] = (QR_HEIGHT * FOCAL_LENGTH) * dx / d_cube
            H[2*i, 1] = (QR_HEIGHT * FOCAL_LENGTH) * dy / d_cube
            H[2*i, 2] = 0
            
            # dcx/dx
            H[2*i + 1, 0] = FOCAL_LENGTH * sec_sq * (-dy / d_sq)
            H[2*i + 1, 1] = FOCAL_LENGTH * sec_sq * (dx / d_sq)
            H[2*i + 1, 2] = FOCAL_LENGTH * sec_sq * (-1)
            
            # Measurement noise
            R[2*i, 2*i] = self.R_height
            R[2*i + 1, 2*i + 1] = self.R_cx
        # Add regularization to R
        R = R + np.eye(2 * n_meas) * 1e-4
        # Innovation
        y = z - z_pred
        
        # Kalman gain
        S = H @ self.P @ H.T + R
        S += np.eye(S.shape[0]) * 1e-6
        try:
            K = self.P @ H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in Kalman gain calculation")
            return
        
        # State update
                # Limit state update
        # dx_update = K @ y
        # dx_update[0] = np.clip(dx_update[0], -2.0, 2.0)  
        # dx_update[1] = np.clip(dx_update[1], -2.0, 2.0)  
        # dx_update[2] = np.clip(dx_update[2], -np.radians(3.0), np.radians(3.0)) 
        
        # self.x = self.x + dx_update
        
        self.x = self.x + K @ y



        # Covariance update
        I = np.eye(3)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2

def run_ekf(imu_time, gyro_z, motor_interp_left, motor_interp_right,
            camera_data, x0, y0, phi0, Q, R_height, R_cx, P0):
    
    ekf = ExtendedKalmanFilter(
        x0=[x0, y0, np.radians(phi0)],
        P0=P0,
        Q=Q,
        R_height=R_height,
        R_cx=R_cx
    )

    camera_timestamps = np.unique(camera_data[:, 0])
    camera_dict = {}
    for row in camera_data:
        t = row[0]
        qr_id = int(row[1])
        cx = row[2]
        height = row[5]  
        if t not in camera_dict:
            camera_dict[t] = []
        camera_dict[t].append((qr_id, cx, height))
    n = len(imu_time)
    trajectory = np.zeros((n, 3))
    covariance = np.zeros((n, 3, 3))
    trajectory[0] = ekf.x
    covariance[0] = ekf.P
    
    camera_idx = 0
    
    for i in range(1, n):
        dt = imu_time[i] - imu_time[i-1]
        
        pwm_l = motor_interp_left(imu_time[i-1])
        pwm_r = motor_interp_right(imu_time[i-1])
        v = pwm_to_velocity(pwm_l, pwm_r)
        omega = np.radians(gyro_z[i-1] - GYRO_BIAS_Z)
        ekf.predict(v, omega, dt)
        current_time = imu_time[i]
       
        time_tolerance = 0.1  # seconds
        for cam_time in camera_timestamps:
            if abs(cam_time - current_time) < time_tolerance:
                if cam_time in camera_dict:
                    measurements = camera_dict[cam_time]
                    ekf.update(measurements)
                    del camera_dict[cam_time]
                break
        
        trajectory[i] = ekf.x
        covariance[i] = ekf.P
    
    return trajectory, covariance


q_pos = 10.0  
q_phi = np.radians(5.0)**2  
Q = np.diag([q_pos, q_pos, q_phi])


R_height = 100.0 
R_cx = 400.0 

P0 = np.diag([10.0, 10.0, np.radians(5.0)**2])

print("\nEKF Parameters:")
print(f"  Process noise Q: diag([{q_pos:.2f}, {q_pos:.2f}, {np.degrees(np.sqrt(q_phi)):.2f}° ])")
print(f"  Measurement noise R: sigma_h={np.sqrt(R_height):.1f}px, sigma_cx={np.sqrt(R_cx):.1f}px")
print("\nRunning EKF...")
ekf_trajectory, ekf_covariance = run_ekf(
    imu_time_filtered, gyro_z_filtered,
    motor_interp_left, motor_interp_right,
    camera_data, X0, Y0, PHI0, Q, R_height, R_cx, P0
)
print(f"EKF completed!")
print(f"  Start position: ({ekf_trajectory[0, 0]:.2f}, {ekf_trajectory[0, 1]:.2f}) cm")
print(f"  End position: ({ekf_trajectory[-1, 0]:.2f}, {ekf_trajectory[-1, 1]:.2f}) cm")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("Generating Plots...")
print("=" * 70)

# Create semi-elliptical reference track
def create_reference_track():
    a = 50  
    b = 40  
    cx, cy = 60, 60 
    
    theta = np.linspace(0, 2*np.pi, 100)
    x = cx + a * np.cos(theta)
    y = cy + b * np.sin(theta)
    return x, y

ref_x, ref_y = create_reference_track()


fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Task 6: Dead Reckoning trajectory
ax1 = axes[0, 0]
ax1.plot([0, FRAME_SIZE, FRAME_SIZE, 0, 0], [0, 0, FRAME_SIZE, FRAME_SIZE, 0], 'k-', linewidth=2, label='Frame')
ax1.plot(ref_x, ref_y, 'g--', linewidth=2, alpha=0.5, label='Reference Track')
ax1.plot(dr_trajectory[:, 0], dr_trajectory[:, 1], 'b-', linewidth=1.5, label='Dead Reckoning')
ax1.scatter(dr_trajectory[0, 0], dr_trajectory[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
ax1.scatter(dr_trajectory[-1, 0], dr_trajectory[-1, 1], c='red', s=100, marker='x', zorder=5, label='End')
ax1.set_xlabel('x (cm)', fontsize=12)
ax1.set_ylabel('y (cm)', fontsize=12)
ax1.set_title('Task 6: Dead Reckoning Using IMU Only)', fontsize=12)
ax1.legend(fontsize=9)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-10, FRAME_SIZE + 10)
ax1.set_ylim(-10, FRAME_SIZE + 10)

# Task 7: EKF trajectory
ax2 = axes[0, 1]
ax2.plot([0, FRAME_SIZE, FRAME_SIZE, 0, 0], [0, 0, FRAME_SIZE, FRAME_SIZE, 0], 'k-', linewidth=2, label='Frame')
ax2.plot(ref_x, ref_y, 'g--', linewidth=2, alpha=0.5, label='Reference Track')
ax2.plot(ekf_trajectory[:, 0], ekf_trajectory[:, 1], 'r-', linewidth=1.5, label='EKF')
ax2.scatter(ekf_trajectory[0, 0], ekf_trajectory[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
ax2.scatter(ekf_trajectory[-1, 0], ekf_trajectory[-1, 1], c='red', s=100, marker='x', zorder=5, label='End')

# Plot QR code positions
for qr_id, pos in QR_POSITIONS.items():
    ax2.scatter(pos[0], pos[1], c='blue', s=30, marker='s', alpha=0.5)

ax2.set_xlabel('x (cm)', fontsize=12)
ax2.set_ylabel('y (cm)', fontsize=12)
ax2.set_title('Task 7: EKF Using IMU And Camera', fontsize=12)
ax2.legend(fontsize=9)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-10, FRAME_SIZE + 10)
ax2.set_ylim(-10, FRAME_SIZE + 10)

# Comparison plot
ax3 = axes[1, 0]
ax3.plot([0, FRAME_SIZE, FRAME_SIZE, 0, 0], [0, 0, FRAME_SIZE, FRAME_SIZE, 0], 'k-', linewidth=2)
ax3.plot(ref_x, ref_y, 'g--', linewidth=2, alpha=0.5, label='Reference')
ax3.plot(dr_trajectory[:, 0], dr_trajectory[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Dead Reckoning')
ax3.plot(ekf_trajectory[:, 0], ekf_trajectory[:, 1], 'r-', linewidth=1.5, label='EKF')
ax3.set_xlabel('x (cm)', fontsize=12)
ax3.set_ylabel('y (cm)', fontsize=12)
ax3.set_title('Comparison: Dead Reckoning vs EKF', fontsize=12)
ax3.legend(fontsize=10)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-10, FRAME_SIZE + 10)
ax3.set_ylim(-10, FRAME_SIZE + 10)

# Heading over time
ax4 = axes[1, 1]
time_rel = imu_time_filtered - imu_time_filtered[0]
ax4.plot(time_rel, np.degrees(dr_trajectory[:, 2]), 'b-', linewidth=1, label='Dead Reckoning')
ax4.plot(time_rel, np.degrees(ekf_trajectory[:, 2]), 'r-', linewidth=1, label='EKF')
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylabel('Heading (degrees)', fontsize=12)
ax4.set_title('Heading Angle Over Time', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:\\Project\\sensor_fusion_project\\output\\part2\\task6_7_results.png", dpi=150, bbox_inches='tight')
print("Saved: task6_7_results.png")

# Plot 2: EKF uncertainty
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

# Position uncertainty over time
ax1 = axes2[0]
sigma_x = np.sqrt(ekf_covariance[:, 0, 0])
sigma_y = np.sqrt(ekf_covariance[:, 1, 1])
ax1.plot(time_rel, sigma_x, 'b-', label='SIGMA_X')
ax1.plot(time_rel, sigma_y, 'r-', label='SIGMA_Y')
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Standard Deviation (cm)', fontsize=12)
ax1.set_title('Position Uncertainty', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Heading uncertainty over time
ax2 = axes2[1]
sigma_phi = np.degrees(np.sqrt(ekf_covariance[:, 2, 2]))
ax2.plot(time_rel, sigma_phi, 'g-')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Standard Deviation (degrees)', fontsize=12)
ax2.set_title('Heading Uncertainty', fontsize=12)
ax2.grid(True, alpha=0.3)

# Gyroscope readings
ax3 = axes2[2]
ax3.plot(time_rel, gyro_z_filtered, 'b-', alpha=0.5, linewidth=0.5)
ax3.axhline(y=GYRO_BIAS_Z, color='r', linestyle='--', label=f'Bias = {GYRO_BIAS_Z:.4f} deg/s')
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Gyro Z (deg/s)', fontsize=12)
ax3.set_title('Gyroscope Z-axis Readings', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:\\Project\\sensor_fusion_project\\output\\part2\\task6_7_uncertainty.png", dpi=150, bbox_inches='tight')
print("Saved: task6_7_uncertainty.png")

# =============================================================================
# SAVE RESULTS TO TXT FILE
# =============================================================================
results_file = "D:\\Project\\sensor_fusion_project\\output\\part2\\task6_7_results.txt"
with open(results_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("Part II: Localization and Tracking Results\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("CALIBRATION PARAMETERS (from Part I):\n")
    f.write(f"  Focal length: {FOCAL_LENGTH:.4f} pixels\n")
    f.write(f"  QR code height: {QR_HEIGHT} cm\n")
    f.write(f"  Gyroscope Z bias: {GYRO_BIAS_Z:.6f} deg/s\n")
    f.write(f"  Robot speed at 30% PWM: {ROBOT_SPEED_30PWM:.4f} cm/s\n\n")
    
    f.write("INITIAL CONDITIONS:\n")
    f.write(f"  x0 = {X0} cm\n")
    f.write(f"  y0 = {Y0} cm\n")
    f.write(f"  phi0 = {PHI0} degrees\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("TASK 6: DEAD RECKONING USING IMU ONLY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Dynamic Model: Quasi-Constant Turn Model\n")
    f.write("  p_x_dot = v * cos(phi)\n")
    f.write("  p_y_dot = v * sin(phi)\n")
    f.write("  phi_dot = omega_gyro - bias\n\n")
    
    f.write("Discretization: Euler method\n")
    f.write("  p_x[k+1] = p_x[k] + v[k] * cos(phi_avg) * dt\n")
    f.write("  p_y[k+1] = p_y[k] + v[k] * sin(phi_avg) * dt\n")
    f.write("  phi[k+1] = phi[k] + omega[k] * dt\n\n")
    
    f.write("Results:\n")
    f.write(f"  Start: ({dr_trajectory[0, 0]:.2f}, {dr_trajectory[0, 1]:.2f}) cm, {np.degrees(dr_trajectory[0, 2]):.2f} deg\n")
    f.write(f"  End: ({dr_trajectory[-1, 0]:.2f}, {dr_trajectory[-1, 1]:.2f}) cm, {np.degrees(dr_trajectory[-1, 2]):.2f} deg\n")
    f.write(f"  Total heading change: {np.degrees(dr_trajectory[-1, 2] - dr_trajectory[0, 2]):.2f} deg\n")
    f.write(f"  Duration: {imu_time_filtered[-1] - imu_time_filtered[0]:.2f} s\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("TASK 7: EXTENDED KALMAN FILTER USING IMU AND CAMERA\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("State Vector: x = [p_x, p_y, phi]^T\n\n")
    
    f.write("Process Model FOR Prediction:\n")
    f.write("  Same as Task 6 quasi-constant turn model\n\n")
    
    f.write("Measurement Model from camera:\n")
    f.write("  h_i = (h0 * f) / d_i\n")
    f.write("  Cx_i = f * tan(alpha_i - phi)\n")
    f.write("  where d_i = sqrt[(s_x - p_x)^2 + (s_y - p_y)^2]\n")
    f.write("        alpha_i = arctan2(s_y - p_y, s_x - p_x)\n\n")
    
    f.write("EKF Parameters:\n")
    f.write(f"  Process noise Q: diag([{q_pos:.2f}, {q_pos:.2f}, {q_phi:.6f}]) (cm^2, cm^2, rad^2)\n")
    f.write(f"  Measurement noise: sigma_h = {np.sqrt(R_height):.1f} px, sigma_cx = {np.sqrt(R_cx):.1f} px\n\n")
    
    f.write("Results:\n")
    f.write(f"  Start: ({ekf_trajectory[0, 0]:.2f}, {ekf_trajectory[0, 1]:.2f}) cm, {np.degrees(ekf_trajectory[0, 2]):.2f} deg\n")
    f.write(f"  End: ({ekf_trajectory[-1, 0]:.2f}, {ekf_trajectory[-1, 1]:.2f}) cm, {np.degrees(ekf_trajectory[-1, 2]):.2f} deg\n")
    f.write(f"  Final uncertainty: sigma_x = {np.sqrt(ekf_covariance[-1, 0, 0]):.4f} cm, ")
    f.write(f"sigma_y = {np.sqrt(ekf_covariance[-1, 1, 1]):.4f} cm, ")
    f.write(f"sigma_phi = {np.degrees(np.sqrt(ekf_covariance[-1, 2, 2])):.4f} deg\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("COMPARISON\n")
    f.write("=" * 70 + "\n\n")
    
    # Calculate drift
    dr_end = dr_trajectory[-1, :2]
    ekf_end = ekf_trajectory[-1, :2]
    
    f.write("End Position Comparison:\n")
    f.write(f"  Dead Reckoning: ({dr_end[0]:.2f}, {dr_end[1]:.2f}) cm\n")
    f.write(f"  EKF: ({ekf_end[0]:.2f}, {ekf_end[1]:.2f}) cm\n")
    f.write(f"  Difference: ({ekf_end[0] - dr_end[0]:.2f}, {ekf_end[1] - dr_end[1]:.2f}) cm\n")

print(f"Saved: {results_file}")



print("\n" + "=" * 70)
print("Task 6 and 7 are Completed!")
print("=" * 70)