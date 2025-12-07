import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os
import sys

# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================
FOCAL_LENGTH = 524.7240  # pixels from Task 3
QR_HEIGHT = 11.5  
GYRO_BIAS_Z = 0.0 
IMAGE_CENTER_X = 160  #
ROBOT_SPEED_30PWM = 6.2307 * 0.75 
VELOCITY_NONLINEAR_EXP = 1.15 

# Initial position
X0, Y0, PHI0 = 43.0, 18.0, 0.0 
FRAME_SIZE = 121.5

# =============================================================================
# OPTIMIZED EKF PARAMETERS
# =============================================================================
Q_SCALE = 0.3
PROCESS_NOISE_V = 0.4  
PROCESS_NOISE_OMEGA = 0.04  
R_DISTANCE = 2.5  
R_DIRECTION = 3.5  
MAHALANOBIS_THRESHOLD = 15.0

# =============================================================================
# DATA PATHS
# =============================================================================
def get_data_paths():
   
    project_dir = "D:\\Project\\sensor_fusion_project\\data\\task6-task7"
    
    paths = {
        'imu': os.path.join(project_dir, "imu_tracking_task6.csv"),
        'motor': os.path.join(project_dir, "motor_control_tracking_task6.csv"),
        'camera': os.path.join(project_dir, "camera_tracking_task6.csv"),
        'qr_positions': os.path.join(project_dir, "qr_code_position_in_global_coordinate.csv")
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"warning: {name} file not exists in: {path}")
    
    return paths

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
  
    print("=" * 70)
    print("Task 7: Extended Kalman Filter")
    print("=" * 70)
    
    paths = get_data_paths()
    
    # Load QR code positions
    print("\nLoading QR code positions...")
    qr_data = np.loadtxt(paths['qr_positions'], delimiter=',', skiprows=1)
    QR_POSITIONS = {int(row[0]): np.array([row[1], row[2]]) for row in qr_data}
    print(f"Loaded {len(QR_POSITIONS)} QR code positions")
    
    # Load IMU data
    print("\nLoading IMU data...")
    imu_data = np.loadtxt(paths['imu'], delimiter=',')
    imu_time = imu_data[:, 0]
    gyro_z = imu_data[:, 8]
    print(f"IMU samples: {len(imu_time)}")
    
    # Load Motor data
    print("\nLoading Motor data...")
    motor_data = np.loadtxt(paths['motor'], delimiter=',')
    motor_time = motor_data[:, 0]
    pwm_left = motor_data[:, 1]
    pwm_right = motor_data[:, 2]
    print(f"Motor samples: {len(motor_time)}")
    
    # Load Camera data
    print("\nLoading Camera data...")
    camera_data = np.loadtxt(paths['camera'], delimiter=',')
    print(f"Camera detections: {len(camera_data)}")
    
    return {
        'qr_positions': QR_POSITIONS,
        'imu_time': imu_time,
        'gyro_z': gyro_z,
        'motor_time': motor_time,
        'pwm_left': pwm_left,
        'pwm_right': pwm_right,
        'camera_data': camera_data
    }

# =============================================================================
# MOTION MODEL
# =============================================================================
def pwm_to_velocity(pwm_l, pwm_r):
    avg_pwm = (pwm_l + pwm_r) / 2.0
    base_velocity = ROBOT_SPEED_30PWM
    velocity = base_velocity * (avg_pwm / 0.3) ** VELOCITY_NONLINEAR_EXP
    return velocity


def motion_model(state, v, omega, dt):
    x, y, phi = state
    
    phi_new = phi + omega * dt
    phi_avg = (phi + phi_new) / 2.0
    
    x_new = x + v * np.cos(phi_avg) * dt
    y_new = y + v * np.sin(phi_avg) * dt
    
    return np.array([x_new, y_new, phi_new])

def motion_jacobian(state, v, omega, dt):

    x, y, phi = state
    
    phi_new = phi + omega * dt
    phi_avg = (phi + phi_new) / 2.0
    
    F = np.array([
        [1, 0, -v * np.sin(phi_avg) * dt],
        [0, 1,  v * np.cos(phi_avg) * dt],
        [0, 0,  1]
    ])
    
    return F

def process_noise_covariance(v, omega, dt):

    sigma_v = PROCESS_NOISE_V * Q_SCALE
    sigma_omega = PROCESS_NOISE_OMEGA * Q_SCALE
    
    q_x = (sigma_v * dt)**2
    q_y = (sigma_v * dt)**2
    q_phi = (sigma_omega * dt)**2
    
    Q = np.diag([q_x, q_y, q_phi])
    
    return Q

# =============================================================================
# MEASUREMENT MODEL
# =============================================================================
def measurement_model(state, qr_position):
   
    x, y, phi = state
    sx, sy = qr_position
    
    # distance
    dx = sx - x
    dy = sy - y
    distance = np.sqrt(dx**2 + dy**2)
    
    # global angle
    alpha = np.arctan2(dy, dx)
    
    # relative direction (robot coordinate system)
    direction = alpha - phi
    
    # normalize to [-π, π]
    direction = np.arctan2(np.sin(direction), np.cos(direction))
    
    return np.array([distance, direction])

def measurement_jacobian(state, qr_position):
    x, y, phi = state
    sx, sy = qr_position
    
    dx = sx - x
    dy = sy - y
    d_sq = dx**2 + dy**2
    d = np.sqrt(d_sq)
    
    if d < 0.1:  
        d = 0.1
        d_sq = d**2
    
     
    dd_dx = -dx / d
    dd_dy = -dy / d
    dd_dphi = 0
    
    dalpha_dx = dy / d_sq
    dalpha_dy = -dx / d_sq
    dalpha_dphi = -1
    
    H = np.array([
        [dd_dx, dd_dy, dd_dphi],
        [dalpha_dx, dalpha_dy, dalpha_dphi]
    ])
    
    return H

def measurement_noise_covariance():

    R = np.array([
        [R_DISTANCE**2, 0],
        [0, np.radians(R_DIRECTION)**2]
    ])
    return R

def get_camera_measurements(camera_row):
    qr_id = int(camera_row[1])
    cx = camera_row[2]  # center x coordinate 
    height = camera_row[5]  # height in pixels
    
    if height < 10 or height > 250:
        return None, None, None
    
    distance = (QR_HEIGHT * FOCAL_LENGTH) / height
    
    cx_centered = cx - IMAGE_CENTER_X
    direction = np.arctan2(cx_centered, FOCAL_LENGTH)
    
    return qr_id, distance, direction

# =============================================================================
# EXTENDED KALMAN FILTER
# =============================================================================
class ExtendedKalmanFilter:
    
    def __init__(self, x0, P0):
        self.x = x0
        self.P = P0
        self.outliers_rejected = 0
        
    def predict(self, v, omega, dt):

        self.x = motion_model(self.x, v, omega, dt)
    
        F = motion_jacobian(self.x, v, omega, dt)
        Q = process_noise_covariance(v, omega, dt)
        self.P = F @ self.P @ F.T + Q
        
    def update(self, measurement, qr_position):
        
        z_pred = measurement_model(self.x, qr_position)
        y = measurement - z_pred
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))
        
        H = measurement_jacobian(self.x, qr_position)
        
        R = measurement_noise_covariance()
        
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = y.T @ S_inv @ y
            
            if mahal_dist > MAHALANOBIS_THRESHOLD:
                self.outliers_rejected += 1
                return False 
        except np.linalg.LinAlgError:
            return False
        
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        return True  

# =============================================================================
# MAIN EKF ALGORITHM
# =============================================================================
def run_ekf(data):    
    QR_POSITIONS = data['qr_positions']
    imu_time = data['imu_time']
    gyro_z = data['gyro_z']
    motor_time = data['motor_time']
    pwm_left = data['pwm_left']
    pwm_right = data['pwm_right']
    camera_data = data['camera_data']
   
    motor_interp_l = interp1d(motor_time, pwm_left, kind='previous', bounds_error=False, fill_value=(pwm_left[0], pwm_left[-1]))
    motor_interp_r = interp1d(motor_time, pwm_right, kind='previous', bounds_error=False, fill_value=(pwm_right[0], pwm_right[-1]))
    
   
    t_start = max(imu_time[0], motor_time[0])
    t_end = min(imu_time[-1], motor_time[-1])
    mask = (imu_time >= t_start) & (imu_time <= t_end)
    imu_time_f = imu_time[mask]
    gyro_z_f = gyro_z[mask]
    
    print(f"\nEKF setting:")
    print(f"  process_noise: sigma_v={PROCESS_NOISE_V}cm/s, sigma_omega={PROCESS_NOISE_OMEGA}rad/s")
    print(f"  measurement_noise: sigma_d={R_DISTANCE}cm, sigma_alpha={R_DIRECTION}°")
    print(f"  outlier_threshold: {MAHALANOBIS_THRESHOLD} (chi-square)")
    print(f"  velocity_model: nonlinear (exponent {VELOCITY_NONLINEAR_EXP})")
    
    # Initialize EKF - reduce initial uncertainty
    x0 = np.array([X0, Y0, np.radians(PHI0)])
    P0 = np.diag([2.0**2, 2.0**2, np.radians(5)**2])
    ekf = ExtendedKalmanFilter(x0, P0)

    cam_dict = {}
    for row in camera_data:
        t = row[0]
        cam_dict.setdefault(t, []).append(row)
    cam_times = sorted(cam_dict.keys())
    
    n = len(imu_time_f)
    trajectory = np.zeros((n, 3))
    covariances = np.zeros((n, 3))
    trajectory[0] = ekf.x.copy()
    covariances[0] = np.diag(ekf.P)
    
    cam_idx = 0
    update_count = 0
    accepted_count = 0
    
    print("\nStarting optimized EKF estimation...")
    for i in range(1, n):
        if i % 500 == 0:
            print(f"  Progress: {i}/{n} ({100*i/n:.1f}%)")
        
        dt = imu_time_f[i] - imu_time_f[i-1]
        
        # Control input
        v = pwm_to_velocity(motor_interp_l(imu_time_f[i-1]), 
                           motor_interp_r(imu_time_f[i-1]))
        omega = np.radians(gyro_z_f[i-1] - GYRO_BIAS_Z)
        
        # Prediction
        ekf.predict(v, omega, dt)
        
        # Check camera measurements
        t = imu_time_f[i]
        measurements = []
        
        for k in range(cam_idx, len(cam_times)):
            if abs(cam_times[k] - t) < 0.15:
                measurements.extend(cam_dict[cam_times[k]])
                cam_idx = k
                break
            if cam_times[k] > t + 0.15:
                break
        
        # Update
        for cam_row in measurements:
            qr_id, distance, direction = get_camera_measurements(cam_row)
            
            if qr_id is None or qr_id not in QR_POSITIONS:
                continue
            
            qr_pos = QR_POSITIONS[qr_id]
            z = np.array([distance, direction])
            if ekf.update(z, qr_pos):
                accepted_count += 1
            
            update_count += 1
        trajectory[i] = ekf.x.copy()
        covariances[i] = np.diag(ekf.P)
        if i > 0:
            while trajectory[i, 2] - trajectory[i-1, 2] > np.pi:
                trajectory[i, 2] -= 2*np.pi
            while trajectory[i, 2] - trajectory[i-1, 2] < -np.pi:
                trajectory[i, 2] += 2*np.pi
    
    print(f"\nEKF done!")
    print(f"  Total measurements: {update_count}")
    print(f"  Accepted measurements: {accepted_count} ({100*accepted_count/max(update_count,1):.1f}%)")
    print(f"  Rejected outliers: {ekf.outliers_rejected}")
    
    return trajectory, covariances, imu_time_f

# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_and_save(trajectory, covariances, time_data):    
    trajectory_deg = trajectory.copy()
    trajectory_deg[:, 2] = np.degrees(trajectory[:, 2])
    
    drift = np.sqrt((trajectory[-1, 0] - trajectory[0, 0])**2 + 
                   (trajectory[-1, 1] - trajectory[0, 1])**2)
    
    print(f"\nTrajectory statistics:")
    print(f"  Start: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f}) cm, {trajectory_deg[0, 2]:.2f}°")
    print(f"  End: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f}) cm, {trajectory_deg[-1, 2]:.2f}°")
    print(f"  Drift: {drift:.1f} cm")
    print(f"  Heading change: {trajectory_deg[-1, 2] - trajectory_deg[0, 2]:.1f}°")
    print(f"  Final uncertainty: σ_x={np.sqrt(covariances[-1, 0]):.3f}cm, σ_y={np.sqrt(covariances[-1, 1]):.3f}cm")
    
    # Reference trajectory
    theta = np.linspace(0, 2*np.pi, 100)
    ref_x, ref_y = 60 + 50*np.cos(theta), 60 + 40*np.sin(theta)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main trajectory plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot([0, FRAME_SIZE, FRAME_SIZE, 0, 0], 
             [0, 0, FRAME_SIZE, FRAME_SIZE, 0], 'k-', lw=2)
    ax1.plot(ref_x, ref_y, 'g--', lw=2, alpha=0.4, label='Reference')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', lw=1.5, label='Optimized EKF')

    for i in range(0, len(trajectory), 400):
        sigma_x = np.sqrt(covariances[i, 0])
        sigma_y = np.sqrt(covariances[i, 1])
        circle = plt.Circle((trajectory[i, 0], trajectory[i, 1]), 
                           max(sigma_x, sigma_y)*2, 
                           color='blue', alpha=0.1)
        ax1.add_patch(circle)
    
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=120, 
               marker='o', zorder=5, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=120, 
               marker='X', zorder=5, label='End')
    ax1.set_xlabel('X (cm)', fontsize=11)
    ax1.set_ylabel('Y (cm)', fontsize=11)
    ax1.set_title('Task 7: Optimized EKF Tracking', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, FRAME_SIZE + 5)
    ax1.set_ylim(-5, FRAME_SIZE + 5)
    
    time_rel = time_data - time_data[0]
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_rel, trajectory_deg[:, 2], 'b-', lw=1.2)
    ax2.axhline(y=1080, color='g', ls='--', alpha=0.5, label='Expected (1080°)')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Heading (°)', fontsize=11)
    ax2.set_title('Heading Over Time', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_rel, np.sqrt(covariances[:, 0]), 'b-', lw=1.2, label='σ_x')
    ax3.plot(time_rel, np.sqrt(covariances[:, 1]), 'r-', lw=1.2, label='σ_y')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Position Uncertainty (cm)', fontsize=11)
    ax3.set_title('Position Uncertainty', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_rel, trajectory[:, 0], 'b-', lw=1.2, label='X')
    ax4.plot(time_rel, trajectory[:, 1], 'r-', lw=1.2, label='Y')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Position (cm)', fontsize=11)
    ax4.set_title('X and Y Position', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_rel, np.degrees(np.sqrt(covariances[:, 2])), 'b-', lw=1.2)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Heading Uncertainty (°)', fontsize=11)
    ax5.set_title('Heading Uncertainty', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    output_dir = "D:\\Project\\sensor_fusion_project\\output\\part2"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, "task7_ekf_optimized_result.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")
    
    csv_path = os.path.join(output_dir, "task7_ekf_optimized_trajectory.csv")
    np.savetxt(csv_path, trajectory_deg, delimiter=',',
              header='x_cm,y_cm,heading_deg', comments='')
    print(f"Saved trajectory: {csv_path}")
    
    cov_path = os.path.join(output_dir, "task7_ekf_optimized_covariance.csv")
    np.savetxt(cov_path, covariances, delimiter=',',
              header='var_x,var_y,var_heading', comments='')
    print(f"Saved covariance: {cov_path}")
    
    return plot_path, csv_path

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    try:
        data = load_data()
        trajectory, covariances, time_data = run_ekf(data)
        plot_path, csv_path = visualize_and_save(trajectory, covariances, time_data)
        
        print("\n" + "=" * 70)
        print("Task 7 EKF completed!")
        print("=" * 70)
        print(f"\nResult files:")
        print(f"  Plot: {plot_path}")
        print(f"  Trajectory: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
