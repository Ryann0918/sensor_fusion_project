"""
ELEC-E8740 Basic Sensor Fusion Project - Task 7
Particle Filter Tracking (Based on Reference Implementation)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os

# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================
FOCAL_LENGTH = 524.7240  # pixels from Task 3
QR_HEIGHT = 11.5  # cm
GYRO_BIAS_Z = 0.0  # deg/s
IMAGE_CENTER_X = 160  # Principal point

# PWM to velocity - calibrated from Task 4 with friction adjustment
# Base: 6.23/0.3 = 20.77, adjusted for real-world friction
FULL_VELOCITY = 15.0  # cm/s at PWM=1.0 (optimal)

# Initial position
X0, Y0, PHI0 = 43.0, 18.0, 0.0  # cm, cm, degrees
FRAME_SIZE = 121.5

# Particle Filter Parameters
NUM_PARTICLES = 2000

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("Task 7: Particle Filter Tracking")
print("=" * 70)

# 项目路径
base_dir = r"D:\Project\sensor_fusion_project"

# 数据文件路径
imu_file = os.path.join(base_dir, "data", "task6-task7", "imu_tracking_task6.csv")
motor_file = os.path.join(base_dir, "data", "task6-task7", "motor_control_tracking_task6.csv")
camera_file = os.path.join(base_dir, "data", "task6-task7", "camera_tracking_task6.csv")
qr_file = os.path.join(base_dir, "data", "task6-task7", "qr_code_position_in_global_coordinate.csv")

# 检查文件是否存在
print("\n文件检查:")
for name, path in [("IMU", imu_file), ("Motor", motor_file), ("Camera", camera_file), ("QR", qr_file)]:
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"  {name}: {exists}")
    if not os.path.exists(path):
        print(f"    路径: {path}")
        raise FileNotFoundError(f"找不到{name}文件: {path}")

# Load data
qr_data = np.loadtxt(qr_file, delimiter=',', skiprows=1)
QR_POSITIONS = {int(row[0]): np.array([row[1], row[2]]) for row in qr_data}
print(f"\nLoaded {len(QR_POSITIONS)} QR code positions")

imu_data = np.loadtxt(imu_file, delimiter=',')
imu_time, gyro_z = imu_data[:, 0], imu_data[:, 8]
print(f"Loaded IMU data: {len(imu_time)} samples")

motor_data = np.loadtxt(motor_file, delimiter=',')
motor_time, pwm_left, pwm_right = motor_data[:, 0], motor_data[:, 1], motor_data[:, 2]
print(f"Loaded Motor data: {len(motor_time)} samples")

camera_data = np.loadtxt(camera_file, delimiter=',')
print(f"Loaded Camera data: {len(camera_data)} samples")

# Interpolators
motor_interp_l = interp1d(motor_time, pwm_left, kind='previous', bounds_error=False, fill_value=(pwm_left[0], pwm_left[-1]))
motor_interp_r = interp1d(motor_time, pwm_right, kind='previous', bounds_error=False, fill_value=(pwm_right[0], pwm_right[-1]))

# Filter time range
t_start, t_end = max(imu_time[0], motor_time[0]), min(imu_time[-1], motor_time[-1])
mask = (imu_time >= t_start) & (imu_time <= t_end)
imu_time_f, gyro_z_f = imu_time[mask], gyro_z[mask]
print(f"Time range: {t_start:.2f} - {t_end:.2f} s, {len(imu_time_f)} samples")

# =============================================================================
# PARTICLE FILTER IMPLEMENTATION
# =============================================================================
def pwm_to_velocity(pwm_l, pwm_r):
    return (pwm_l + pwm_r) / 2.0 * FULL_VELOCITY

def get_predicted_measurement(px, py, phi, sx, sy):
    """
    Predict distance and direction from robot position to QR code
    Returns: (distance, direction_in_robot_frame)
    """
    dx = sx - px
    dy = sy - py
    d = np.sqrt(dx**2 + dy**2)
    
    # Global angle to QR
    alpha = np.arctan2(dy, dx)
    # Direction in robot frame (relative to heading)
    direction = alpha - np.radians(phi)
    # Normalize to [-pi, pi]
    direction = np.arctan2(np.sin(direction), np.cos(direction))
    
    return d, np.degrees(direction)

def propagate_particle(particle, v, omega, dt, noise_scale=1.0):
    """Propagate a single particle through motion model with noise"""
    px, py, phi = particle
    
    # Add process noise
    v_noisy = v + np.random.normal(0, 0.3 * noise_scale)
    omega_noisy = omega + np.random.normal(0, 0.3 * noise_scale)
    
    # Motion model (Euler integration)
    phi_new = phi + np.degrees(omega_noisy) * dt
    phi_avg = (phi + phi_new) / 2.0
    
    px_new = px + v_noisy * np.cos(np.radians(phi_avg)) * dt
    py_new = py + v_noisy * np.sin(np.radians(phi_avg)) * dt
    
    # DON'T wrap phi - allow it to accumulate for multiple loops
    
    return np.array([px_new, py_new, phi_new])

def compute_likelihood(particle, measurements, dist_sigma=2.0, dir_sigma=3.0):
    """
    Compute likelihood of particle given QR code measurements
    measurements: list of (qr_id, cx, height)
    """
    px, py, phi = particle
    
    total_likelihood = 1.0
    
    for qid, cx, h in measurements:
        if qid not in QR_POSITIONS:
            continue
        if h < 5 or h > 200:
            continue
            
        sx, sy = QR_POSITIONS[qid]
        
        # Measured distance and direction
        d_meas = (QR_HEIGHT * FOCAL_LENGTH) / h
        dir_meas = np.degrees(np.arctan2(cx - IMAGE_CENTER_X, FOCAL_LENGTH))
        
        # Predicted distance and direction
        d_pred, dir_pred = get_predicted_measurement(px, py, phi, sx, sy)
        
        if d_pred < 1:
            continue
        
        # Compute likelihoods (Gaussian)
        likelihood_dist = np.exp(-0.5 * ((d_meas - d_pred) / dist_sigma) ** 2)
        
        # Handle angle wrapping for direction
        dir_diff = dir_meas - dir_pred
        while dir_diff > 180: dir_diff -= 360
        while dir_diff < -180: dir_diff += 360
        likelihood_dir = np.exp(-0.5 * (dir_diff / dir_sigma) ** 2)
        
        total_likelihood *= (likelihood_dist * likelihood_dir + 1e-10)
    
    return total_likelihood

def resample_particles(particles, weights):
    """Systematic resampling"""
    n = len(particles)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-10)
    
    # Cumulative sum
    cumsum = np.cumsum(weights)
    
    # Systematic resampling
    positions = (np.arange(n) + np.random.uniform()) / n
    
    new_particles = []
    idx = 0
    for pos in positions:
        while cumsum[idx] < pos:
            idx = min(idx + 1, n - 1)
        new_particles.append(particles[idx].copy())
    
    return np.array(new_particles)

def run_particle_filter():
    """Run particle filter over the full dataset"""
    
    # Initialize particles around starting position
    particles = np.zeros((NUM_PARTICLES, 3))
    particles[:, 0] = X0 + np.random.normal(0, 2.0, NUM_PARTICLES)
    particles[:, 1] = Y0 + np.random.normal(0, 2.0, NUM_PARTICLES)
    particles[:, 2] = PHI0 + np.random.normal(0, 5.0, NUM_PARTICLES)
    
    # Organize camera data
    cam_dict = {}
    for row in camera_data:
        t, qid, cx, h = row[0], int(row[1]), row[2], row[5]
        cam_dict.setdefault(t, []).append((qid, cx, h))
    cam_times = sorted(cam_dict.keys())
    
    n = len(imu_time_f)
    trajectory = np.zeros((n, 3))
    
    # Initial estimate (mean of particles)
    trajectory[0] = np.mean(particles, axis=0)
    
    cam_idx = 0
    update_count = 0
    
    for i in range(1, n):
        dt = imu_time_f[i] - imu_time_f[i-1]
        v = pwm_to_velocity(motor_interp_l(imu_time_f[i-1]), motor_interp_r(imu_time_f[i-1]))
        omega = np.radians(gyro_z_f[i-1] - GYRO_BIAS_Z)
        
        # Propagate all particles
        for j in range(NUM_PARTICLES):
            particles[j] = propagate_particle(particles[j], v, omega, dt)
        
        # Check for camera measurement
        t = imu_time_f[i]
        has_measurement = False
        measurements = []
        
        for k in range(cam_idx, len(cam_times)):
            if abs(cam_times[k] - t) < 0.15:
                measurements = cam_dict[cam_times[k]]
                cam_idx = k
                has_measurement = True
                break
            if cam_times[k] > t + 0.15:
                break
        
        # Update weights if we have measurements
        if has_measurement and measurements:
            weights = np.array([compute_likelihood(p, measurements) for p in particles])
            
            # Normalize weights
            weight_sum = np.sum(weights)
            if weight_sum > 1e-10:
                weights = weights / weight_sum
                
                # Resample if effective number of particles is low
                n_eff = 1.0 / (np.sum(weights**2) + 1e-10)
                if n_eff < NUM_PARTICLES / 2:
                    particles = resample_particles(particles, weights)
                
                update_count += 1
        
        # Estimate: weighted mean or simple mean
        trajectory[i] = np.mean(particles, axis=0)
        
        # Keep trajectory heading continuous (don't wrap at 360)
        if i > 0:
            while trajectory[i, 2] - trajectory[i-1, 2] > 180:
                trajectory[i, 2] -= 360
            while trajectory[i, 2] - trajectory[i-1, 2] < -180:
                trajectory[i, 2] += 360
    
    print(f"  Particle Filter updates: {update_count}")
    return trajectory

print(f"\nRunning Particle Filter with {NUM_PARTICLES} particles...")
trajectory = run_particle_filter()

# Light smoothing (preserve endpoints)
orig_start, orig_end = trajectory[0].copy(), trajectory[-1].copy()
trajectory[:, 0] = gaussian_filter1d(trajectory[:, 0], sigma=3)
trajectory[:, 1] = gaussian_filter1d(trajectory[:, 1], sigma=3)
trajectory[0], trajectory[-1] = orig_start, orig_end

drift = np.sqrt((trajectory[-1, 0] - trajectory[0, 0])**2 + (trajectory[-1, 1] - trajectory[0, 1])**2)

print(f"Particle Filter completed!")
print(f"  Start: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f}) cm")
print(f"  End: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f}) cm")
print(f"  Drift: {drift:.1f} cm")
print(f"  Heading change: {trajectory[-1, 2] - trajectory[0, 2]:.1f}°")

# =============================================================================
# VISUALIZATION
# =============================================================================
theta = np.linspace(0, 2*np.pi, 100)
ref_x, ref_y = 60 + 50*np.cos(theta), 60 + 40*np.sin(theta)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]
ax1.plot([0, FRAME_SIZE, FRAME_SIZE, 0, 0], [0, 0, FRAME_SIZE, FRAME_SIZE, 0], 'k-', lw=2)
ax1.plot(ref_x, ref_y, 'g--', lw=2, alpha=0.4, label='Reference')
ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=1.5, label='Particle Filter')
ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=120, marker='o', zorder=5, label='Start')
ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=120, marker='X', zorder=5, label='End')
for pos in QR_POSITIONS.values():
    ax1.scatter(pos[0], pos[1], c='blue', s=25, marker='s', alpha=0.4)
ax1.set_xlabel('X (cm)')
ax1.set_ylabel('Y (cm)')
ax1.set_title(f'Task 7: Particle Filter ({NUM_PARTICLES} particles)')
ax1.legend(fontsize=10)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, FRAME_SIZE + 5)
ax1.set_ylim(-5, FRAME_SIZE + 5)

time_rel = imu_time_f - imu_time_f[0]
ax2 = axes[0, 1]
ax2.plot(time_rel, trajectory[:, 2], 'b-', lw=1.2)
ax2.axhline(y=1080, color='g', ls='--', alpha=0.5, label='Expected (1080°)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Heading (°)')
ax2.set_title('Heading Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# X position over time
ax3 = axes[1, 0]
ax3.plot(time_rel, trajectory[:, 0], 'b-', label='X')
ax3.plot(time_rel, trajectory[:, 1], 'r-', label='Y')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Position (cm)')
ax3.set_title('X and Y Position Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Trajectory close-up (first loop)
ax4 = axes[1, 1]
first_loop = len(trajectory) // 3
ax4.plot(trajectory[:first_loop, 0], trajectory[:first_loop, 1], 'r-', lw=1.5, label='First Loop')
ax4.plot(ref_x, ref_y, 'g--', lw=2, alpha=0.4, label='Reference')
ax4.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', zorder=5)
ax4.set_xlabel('X (cm)')
ax4.set_ylabel('Y (cm)')
ax4.set_title('First Loop Detail')
ax4.legend()
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# =============================================================================
# 保存结果（增强版 - 带详细调试信息）
# =============================================================================
print("\n" + "=" * 70)
print("保存结果")
print("=" * 70)

output_dir = os.path.join(base_dir, "output", "part2")
print(f"输出目录: {output_dir}")

# 创建目录
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 目录已创建/确认存在")
except Exception as e:
    print(f"❌ 创建目录失败: {e}")
    print(f"   尝试备用目录...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"   备用目录: {output_dir}")

# 保存图片
fig_path = os.path.join(output_dir, "task7_particle_filter_result.png")
print(f"\n保存图片...")
print(f"  路径: {fig_path}")
try:
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形释放内存
    
    if os.path.exists(fig_path):
        file_size = os.path.getsize(fig_path) / 1024
        print(f"✅ 图片保存成功!")
        print(f"   文件名: task7_particle_filter_result.png")
        print(f"   文件大小: {file_size:.1f} KB")
    else:
        print(f"❌ 图片文件未创建!")
except Exception as e:
    print(f"❌ 保存图片失败: {e}")
    import traceback
    traceback.print_exc()

# 保存CSV
csv_path = os.path.join(output_dir, "task7_particle_filter_trajectory.csv")
print(f"\n保存CSV...")
print(f"  路径: {csv_path}")
try:
    np.savetxt(csv_path, trajectory, delimiter=',', header='x,y,phi', comments='')
    
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path) / 1024
        print(f"✅ CSV保存成功!")
        print(f"   文件名: task7_particle_filter_trajectory.csv")
        print(f"   文件大小: {file_size:.1f} KB")
        print(f"   数据行数: {len(trajectory)}")
    else:
        print(f"❌ CSV文件未创建!")
except Exception as e:
    print(f"❌ 保存CSV失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Task 7 Completed!")
print(f"✅ 结果已保存到: {output_dir}")
print("=" * 70)
