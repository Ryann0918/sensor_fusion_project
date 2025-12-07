import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================
FOCAL_LENGTH = 524.7240  # pixels from Task 3
QR_HEIGHT = 11.5  # cm
GYRO_BIAS_Z = 0.0  # Static bias over-corrects motion data
ROBOT_SPEED_30PWM = 6.2307*0.75  # cm/s at 30% PWM from Task 4

# Initial position (from readme.txt)
X0 = 43.0  # cm
Y0 = 18.0  # cm
PHI0 = 0.0  # degrees

FRAME_SIZE = 121.5  # cm

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("Task 6: Dead Reckoning")
print("=" * 70)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
imu_file = os.path.join(base_dir, "data", "task6-task7", "imu_tracking_task6.csv")
motor_file = os.path.join(base_dir, "data", "task6-task7", "motor_control_tracking_task6.csv")

imu_data = np.loadtxt(imu_file, delimiter=',')
imu_time = imu_data[:, 0]
gyro_z = imu_data[:, 8]
print(f"Loaded IMU data: {len(imu_time)} samples")

motor_data = np.loadtxt(motor_file, delimiter=',')
motor_time = motor_data[:, 0]
pwm_left = motor_data[:, 1]
pwm_right = motor_data[:, 2]
print(f"Loaded Motor data: {len(motor_time)} samples")

# =============================================================================
# PWM TO VELOCITY CONVERSION
# =============================================================================
def pwm_to_velocity(pwm_left, pwm_right):
    avg_pwm = (pwm_left + pwm_right) / 2.0
    diff_pwm = abs(pwm_left - pwm_right)
    
    
    base_velocity = (avg_pwm / 0.3) * ROBOT_SPEED_30PWM
    
    
    if diff_pwm > 0.10:
        turn_factor = 0.92
    elif diff_pwm > 0.05:  
        turn_factor = 0.96
    else:  
        turn_factor = 1.0
    
    return base_velocity * turn_factor * 0.84  

motor_interp_left = interp1d(motor_time, pwm_left, kind='previous', 
                              bounds_error=False, fill_value=(pwm_left[0], pwm_left[-1]))
motor_interp_right = interp1d(motor_time, pwm_right, kind='previous',
                               bounds_error=False, fill_value=(pwm_right[0], pwm_right[-1]))

# Filter to common time range
t_start = max(imu_time[0], motor_time[0])
t_end = min(imu_time[-1], motor_time[-1])
imu_mask = (imu_time >= t_start) & (imu_time <= t_end)
imu_time_filtered = imu_time[imu_mask]
gyro_z_filtered = gyro_z[imu_mask]

print(f"Time range: {t_start:.2f} - {t_end:.2f} s")
print(f"Filtered IMU samples: {len(imu_time_filtered)}")

# =============================================================================
# DEAD RECKONING
# =============================================================================
def dead_reckoning(time, gyro_z, motor_interp_left, motor_interp_right, x0, y0, phi0):
    """Quasi-constant turn model with Euler integration"""
    n = len(time)
    trajectory = np.zeros((n, 3))
    trajectory[0] = [x0, y0, np.radians(phi0)]
    
    for i in range(1, n):
        dt = time[i] - time[i-1]
        x, y, phi = trajectory[i-1]
        
        pwm_l = motor_interp_left(time[i-1])
        pwm_r = motor_interp_right(time[i-1])
        v = pwm_to_velocity(pwm_l, pwm_r)
        omega = np.radians(gyro_z[i-1] - GYRO_BIAS_Z)
        
        phi_new = phi + omega * dt
        phi_avg = (phi + phi_new) / 2
        x_new = x + v * np.cos(phi_avg) * dt
        y_new = y + v * np.sin(phi_avg) * dt
        
        trajectory[i] = [x_new, y_new, phi_new]
    
    return trajectory

print("\nPerforming dead reckoning...")
trajectory = dead_reckoning(imu_time_filtered, gyro_z_filtered, 
                            motor_interp_left, motor_interp_right,
                            X0, Y0, PHI0)

print(f"Dead reckoning completed!")
print(f"  Start: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f}) cm, {np.degrees(trajectory[0, 2]):.2f}°")
print(f"  End: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f}) cm, {np.degrees(trajectory[-1, 2]):.2f}°")
print(f"  Total heading change: {np.degrees(trajectory[-1, 2] - trajectory[0, 2]):.2f}°")

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_reference_track():
    a, b = 50, 40
    cx, cy = 60, 60
    theta = np.linspace(0, 2*np.pi, 100)
    return cx + a * np.cos(theta), cy + b * np.sin(theta)

ref_x, ref_y = create_reference_track()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trajectory plot
ax1 = axes[0]
ax1.plot([0, FRAME_SIZE, FRAME_SIZE, 0, 0], [0, 0, FRAME_SIZE, FRAME_SIZE, 0], 'k-', linewidth=2, label='Frame')
ax1.plot(ref_x, ref_y, 'g--', linewidth=2, alpha=0.5, label='Reference Track')
ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5, label='Dead Reckoning')
ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='x', zorder=5, label='End')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('y (cm)')
ax1.set_title('Task 6: Dead Reckoning Trajectory')
ax1.legend(fontsize=9)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-10, FRAME_SIZE + 10)
ax1.set_ylim(-10, FRAME_SIZE + 10)

# Heading over time
ax2 = axes[1]
time_rel = imu_time_filtered - imu_time_filtered[0]
ax2.plot(time_rel, np.degrees(trajectory[:, 2]), 'b-', linewidth=1)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Heading (degrees)')
ax2.set_title('Heading Angle Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_dir = os.path.join(base_dir, "output", "part2")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "task6_result.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved: task6_result.png")

# Save trajectory data
np.savetxt(os.path.join(output_dir, "task6_trajectory.csv"), trajectory, 
           delimiter=',', header='x,y,phi', comments='')
print("Saved: task6_trajectory.csv")

print("\n" + "=" * 70)
print("Task 6 Completed!")
print("=" * 70)
