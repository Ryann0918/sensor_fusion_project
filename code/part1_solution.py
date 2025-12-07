import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd



IMU_COLUMNS = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch',
               'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']

print("=" * 70)
print("ELEC-E8740 Sensor Fusion Project - Part I")
print("=" * 70)


# TASK 1: Static IMU Experiment - Gyroscope Bias and Variance

print("\n" + "=" * 70)
print("TASK 1: Static IMU Experiment (Gyroscope)")
print("=" * 70)

# Load static IMU data
imu_static = np.loadtxt("D:\\Project\\sensor_fusion_project\\data\\task1\\imu_reading_task1.csv", delimiter=',')
print(f"\nData shape: {imu_static.shape}")
print(f"Number of samples: {imu_static.shape[0]}")
print(f"Duration: {imu_static[-1, 0] - imu_static[0, 0]:.2f} seconds")

# Extract gyroscope data from columns 6, 7, 8
gyro_x = imu_static[:, 6]
gyro_y = imu_static[:, 7]
gyro_z = imu_static[:, 8]

# Task 1a data Visualization
print("\n--- Task 1a: Data Visualization ---")
print(f"Gyroscope X - Mean: {np.mean(gyro_x):.6f} deg/s, Std: {np.std(gyro_x):.6f} deg/s")
print(f"Gyroscope Y - Mean: {np.mean(gyro_y):.6f} deg/s, Std: {np.std(gyro_y):.6f} deg/s")
print(f"Gyroscope Z - Mean: {np.mean(gyro_z):.6f} deg/s, Std: {np.std(gyro_z):.6f} deg/s")

# Task 1b Gyroscope Bias and Variance
print("\n--- Task 1b: Gyroscope Bias and Variance ---")

# Bias (mean)
gyro_bias_x = np.mean(gyro_x)
gyro_bias_y = np.mean(gyro_y)
gyro_bias_z = np.mean(gyro_z)

# Variance
gyro_var_x = np.var(gyro_x)
gyro_var_y = np.var(gyro_y)
gyro_var_z = np.var(gyro_z)

print(f"\nGyroscope Bias:")
print(f"  b_x = {gyro_bias_x:.6f} deg/s")
print(f"  b_y = {gyro_bias_y:.6f} deg/s")
print(f"  b_z = {gyro_bias_z:.6f} deg/s")

print(f"\nGyroscope Variance:")
print(f"  SIGMA^2_x = {gyro_var_x:.6f} (deg/s)^2")
print(f"  SIGMA^2_y = {gyro_var_y:.6f} (deg/s)^2")
print(f"  SIGMA^2_z = {gyro_var_z:.6f} (deg/s)^2")

print(f"\nGyroscope Standard Deviation:")
print(f"  SIGMA_x = {np.sqrt(gyro_var_x):.6f} deg/s")
print(f"  SIGMA_y = {np.sqrt(gyro_var_y):.6f} deg/s")
print(f"  SIGMA_z = {np.sqrt(gyro_var_z):.6f} deg/s")

# Covariance matrix
gyro_data = np.column_stack([gyro_x, gyro_y, gyro_z])
gyro_cov_matrix = np.cov(gyro_data.T)
print(f"\nGyroscope Covariance Matrix:")
print(gyro_cov_matrix)

# Plotting
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Task 1: Static IMU Experiment - Gyroscope Analysis', fontsize=14, fontweight='bold')

time = imu_static[:, 0] - imu_static[0, 0]

# Time series plot
ax1 = axes1[0, 0]
ax1.plot(time, gyro_x, 'r-', alpha=0.7, label=f'Gyro X (bias={gyro_bias_x:.4f})')
ax1.plot(time, gyro_y, 'g-', alpha=0.7, label=f'Gyro Y (bias={gyro_bias_y:.4f})')
ax1.plot(time, gyro_z, 'b-', alpha=0.7, label=f'Gyro Z (bias={gyro_bias_z:.4f})')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular velocity (deg/s)')
ax1.set_title('Gyroscope Readings Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Histogram
ax2 = axes1[0, 1]
ax2.hist(gyro_x, bins=50, alpha=0.5, label='Gyro X', color='red')
ax2.hist(gyro_y, bins=50, alpha=0.5, label='Gyro Y', color='green')
ax2.hist(gyro_z, bins=50, alpha=0.5, label='Gyro Z', color='blue')
ax2.set_xlabel('Angular velocity (deg/s)')
ax2.set_ylabel('Frequency')
ax2.set_title('Gyroscope Reading Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Accelerometer data
acc_x = imu_static[:, 1]
acc_y = imu_static[:, 2]
acc_z = imu_static[:, 3]

ax3 = axes1[1, 0]
ax3.plot(time, acc_x, 'r-', alpha=0.7, label=f'Acc X (mean={np.mean(acc_x):.4f})')
ax3.plot(time, acc_y, 'g-', alpha=0.7, label=f'Acc Y (mean={np.mean(acc_y):.4f})')
ax3.plot(time, acc_z, 'b-', alpha=0.7, label=f'Acc Z (mean={np.mean(acc_z):.4f})')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (g)')
ax3.set_title('Accelerometer Readings Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Magnetometer data
mag_x = imu_static[:, 9]
mag_y = imu_static[:, 10]
mag_z = imu_static[:, 11]

ax4 = axes1[1, 1]
ax4.plot(time, mag_x, 'r-', alpha=0.7, label='Mag X')
ax4.plot(time, mag_y, 'g-', alpha=0.7, label='Mag Y')
ax4.plot(time, mag_z, 'b-', alpha=0.7, label='Mag Z')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Magnetic field (Gauss)')
ax4.set_title('Magnetometer Readings Over Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:\\Project\\sensor_fusion_project\\output\\part1\\task1_gyroscope_analysis.png", dpi=150, bbox_inches='tight')
print("\nPlot saved: task1_gyroscope_analysis.png")


# TASK 2: IMU Calibration - Accelerometer Gain and Bias
print("\n" + "=" * 70)
print("TASK 2: IMU Calibration (Accelerometer)")
print("=" * 70)

# load data
imu_calib = np.loadtxt("D:\\Project\\sensor_fusion_project\\data\\task2\\imu_calibration_task2.csv", delimiter=',')
print(f"\nData shape: {imu_calib.shape}")
print(f"Number of samples: {imu_calib.shape[0]}")



acc_x_cal = imu_calib[:, 1]
acc_y_cal = imu_calib[:, 2]
acc_z_cal = imu_calib[:, 3]

print("\n--- Analyzing accelerometer data for different orientations ---")

# Statistical analysis of the entire dataset
print(f"\nOverall statistics:")
print(f"  Acc X: min={acc_x_cal.min():.4f}, max={acc_x_cal.max():.4f}, mean={acc_x_cal.mean():.4f}")
print(f"  Acc Y: min={acc_y_cal.min():.4f}, max={acc_y_cal.max():.4f}, mean={acc_y_cal.mean():.4f}")
print(f"  Acc Z: min={acc_z_cal.min():.4f}, max={acc_z_cal.max():.4f}, mean={acc_z_cal.mean():.4f}")

# Based on accelerometer readings to determine orientation
# When an axis is facing up, its reading is approximately +1g
# When an axis is facing down, its reading is approximately -1g

# Define threshold to identify different orientations
threshold = 0.8

# Identify data where Z-axis is facing up (normal position)
z_up_mask = acc_z_cal > threshold
# Identify data where Z-axis is facing down (flipped)
z_down_mask = acc_z_cal < -threshold
# Identify data where Y-axis is facing up
y_up_mask = acc_y_cal > threshold
# Identify data where Y-axis is facing down
y_down_mask = acc_y_cal < -threshold
# Identify data where X-axis is facing up
x_up_mask = acc_x_cal > threshold
# Identify data where X-axis is facing down
x_down_mask = acc_x_cal < -threshold

print(f"\nSamples in each orientation:")
print(f"  Z-up (normal): {np.sum(z_up_mask)} samples")
print(f"  Z-down (flipped): {np.sum(z_down_mask)} samples")
print(f"  Y-up: {np.sum(y_up_mask)} samples")
print(f"  Y-down: {np.sum(y_down_mask)} samples")
print(f"  X-up: {np.sum(x_up_mask)} samples")
print(f"  X-down: {np.sum(x_down_mask)} samples")

# Calculate average readings for each orientation
g = 1.0  # Gravitational acceleration (in g units)

# Extract readings for each direction
if np.sum(z_up_mask) > 10 and np.sum(z_down_mask) > 10:
    a_z_up = np.mean(acc_z_cal[z_up_mask])
    a_z_down = np.mean(acc_z_cal[z_down_mask])
    k_z = (a_z_up - a_z_down) / (2 * g)
    b_z = (a_z_up + a_z_down) / 2
    print(f"\nZ-axis calibration:")
    print(f"  a_up = {a_z_up:.6f} g")
    print(f"  a_down = {a_z_down:.6f} g")
    print(f"  Gain k_z = {k_z:.6f}")
    print(f"  Bias b_z = {b_z:.6f} g")
else:
   
    a_z_up = np.mean(acc_z_cal[z_up_mask]) if np.sum(z_up_mask) > 0 else np.mean(acc_z_cal)
    k_z = 1.0  # 
    b_z = a_z_up - 1.0 
    print(f"\nZ-axis (estimated from normal position only):")
    print(f"  a_up = {a_z_up:.6f} g")
    print(f"  Gain k_z = {k_z:.6f} (assumed)")
    print(f"  Bias b_z = {b_z:.6f} g")

if np.sum(y_up_mask) > 10 and np.sum(y_down_mask) > 10:
    a_y_up = np.mean(acc_y_cal[y_up_mask])
    a_y_down = np.mean(acc_y_cal[y_down_mask])
    k_y = (a_y_up - a_y_down) / (2 * g)
    b_y = (a_y_up + a_y_down) / 2
    print(f"\nY-axis calibration:")
    print(f"  a_up = {a_y_up:.6f} g")
    print(f"  a_down = {a_y_down:.6f} g")
    print(f"  Gain k_y = {k_y:.6f}")
    print(f"  Bias b_y = {b_y:.6f} g")
else:
    # Estimate using static position
    a_y_static = np.mean(acc_y_cal[z_up_mask]) if np.sum(z_up_mask) > 0 else np.mean(acc_y_cal)
    k_y = 1.0
    b_y = a_y_static  # Y should be 0 in static position
    print(f"\nY-axis (estimated from static position):")
    print(f"  Static reading = {a_y_static:.6f} g")
    print(f"  Gain k_y = {k_y:.6f} (assumed)")
    print(f"  Bias b_y = {b_y:.6f} g")

if np.sum(x_up_mask) > 10 and np.sum(x_down_mask) > 10:
    a_x_up = np.mean(acc_x_cal[x_up_mask])
    a_x_down = np.mean(acc_x_cal[x_down_mask])
    k_x = (a_x_up - a_x_down) / (2 * g)
    b_x = (a_x_up + a_x_down) / 2
    print(f"\nX-axis calibration:")
    print(f"  a_up = {a_x_up:.6f} g")
    print(f"  a_down = {a_x_down:.6f} g")
    print(f"  Gain k_x = {k_x:.6f}")
    print(f"  Bias b_x = {b_x:.6f} g")
else:
    # Estimate using static position
    a_x_static = np.mean(acc_x_cal[z_up_mask]) if np.sum(z_up_mask) > 0 else np.mean(acc_x_cal)
    k_x = 1.0
    b_x = a_x_static  # X should be 0 in static position
    print(f"\nX-axis (estimated from static position):")
    print(f"  Static reading = {a_x_static:.6f} g")
    print(f"  Gain k_x = {k_x:.6f} (assumed)")
    print(f"  Bias b_x = {b_x:.6f} g")

# Plot Task 2 figures
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Task 2: IMU Calibration - Accelerometer Analysis', fontsize=14, fontweight='bold')

time_cal = imu_calib[:, 0] - imu_calib[0, 0]

ax1 = axes2[0, 0]
ax1.plot(time_cal, acc_x_cal, 'r-', alpha=0.7, label='Acc X')
ax1.plot(time_cal, acc_y_cal, 'g-', alpha=0.7, label='Acc Y')
ax1.plot(time_cal, acc_z_cal, 'b-', alpha=0.7, label='Acc Z')
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='±1g')
ax1.axhline(y=-1, color='k', linestyle='--', alpha=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (g)')
ax1.set_title('Accelerometer Readings - All Orientations')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes2[0, 1]
ax2.hist(acc_x_cal, bins=50, alpha=0.5, label='Acc X', color='red')
ax2.hist(acc_y_cal, bins=50, alpha=0.5, label='Acc Y', color='green')
ax2.hist(acc_z_cal, bins=50, alpha=0.5, label='Acc Z', color='blue')
ax2.set_xlabel('Acceleration (g)')
ax2.set_ylabel('Frequency')
ax2.set_title('Accelerometer Reading Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3D scatter plot
ax3 = axes2[1, 0]
ax3.scatter(acc_x_cal, acc_y_cal, c=acc_z_cal, cmap='viridis', alpha=0.5, s=10)
ax3.set_xlabel('Acc X (g)')
ax3.set_ylabel('Acc Y (g)')
ax3.set_title('X-Y Acceleration (colored by Z)')
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# Acceleration magnitude
acc_magnitude = np.sqrt(acc_x_cal**2 + acc_y_cal**2 + acc_z_cal**2)
ax4 = axes2[1, 1]
ax4.plot(time_cal, acc_magnitude, 'k-', alpha=0.7)
ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Expected (1g)')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Acceleration magnitude (g)')
ax4.set_title(f'Acceleration Magnitude (mean={np.mean(acc_magnitude):.4f}g)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:\\Project\\sensor_fusion_project\\output\\part1\\task2_accelerometer_calibration.png", dpi=150, bbox_inches='tight')
print("\nPlot saved: task2_accelerometer_calibration.png")


# TASK 3: Camera Calibration

print("\n" + "=" * 70)
print("TASK 3: Camera Module Calibration")
print("=" * 70)

# Load camera calibration data
camera_data = np.loadtxt("D:\\Project\\sensor_fusion_project\\data\\task3\\camera_module_calibration_task3.csv", delimiter=',')

# Distance correction
offset = 1.7 + 5.0  # Camera offset + distance from wall to wooden strip
distance_measured = camera_data[:, 0]
height_pixels = camera_data[:, 1]
x3_true = distance_measured + offset

# Linear regression: x3 = k * (1/h) + b
inv_h = 1.0 / height_pixels
slope, intercept, r_value, p_value, std_err = stats.linregress(inv_h, x3_true)


h0 = 11.5  # QR code actual height (cm)
focal_length = slope / h0

print(f"\nCamera offset: {offset} cm")
print(f"QR code actual height h₀: {h0} cm")
print(f"\nLinear regression: x₃ = k x (1/h) + b")
print(f"  Gradient k = {slope:.4f} cm·pixel")
print(f"  Bias b = {intercept:.4f} cm")
print(f"  R^2 = {r_value**2:.6f}")
print(f"\nFocal length f = k / h₀ = {focal_length:.4f} pixels")

# plot
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Task 3: Camera Module Calibration', fontsize=14, fontweight='bold')

ax1 = axes3[0]
ax1.scatter(inv_h, x3_true, alpha=0.7, s=50, label='Measured data')
x_line = np.linspace(inv_h.min(), inv_h.max(), 100)
y_line = slope * x_line + intercept
ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
ax1.set_xlabel('1/h (1/pixels)')
ax1.set_ylabel('x₃ (cm)')
ax1.set_title(f'Linear Regression (R² = {r_value**2:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes3[1]
ax2.scatter(height_pixels, x3_true, alpha=0.7, s=50, label='Measured data')
h_range = np.linspace(height_pixels.min(), height_pixels.max(), 100)
x3_model = (h0 * focal_length) / h_range + intercept
ax2.plot(h_range, x3_model, 'r-', linewidth=2, label='Calibrated model')
ax2.set_xlabel('Detected height h (pixels)')
ax2.set_ylabel('Distance x₃ (cm)')
ax2.set_title('Height vs Distance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:\\Project\\sensor_fusion_project\\output\\part1\\task3_camera_calibration.png", dpi=150, bbox_inches='tight')
print("\nPlot saved: task3_camera_calibration.png")


# TASK 4: Motor Control - Robot Speed
print("\n" + "=" * 70)
print("TASK 4: Motor Control (Robot Speed)")
print("=" * 70)


motor_data = np.loadtxt("D:\\Project\\sensor_fusion_project\\data\\task4\\robot_speed_task4.csv", delimiter=',')

distance = motor_data[:, 0]  # cm
time_elapsed = motor_data[:, 1]  # s 

print(f"\nRaw data:")
print(f"  Distance points: {distance}")
print(f"  Time intervals: {time_elapsed}")


distance_intervals = np.diff(distance)
print(f"\nDistance intervals: {distance_intervals} cm")



distance_interval = 40 
print(f"\nDistance interval: {distance_interval} cm")

# Calculate speed for each segment
speeds = distance_interval / time_elapsed
print(f"\nSpeed for each segment:")
for i, (d, t, s) in enumerate(zip(distance, time_elapsed, speeds)):
    print(f"  Segment to {d:.0f}cm: {distance_interval}cm / {t:.2f}s = {s:.4f} cm/s")

# average speed and std
avg_speed = np.mean(speeds)
std_speed = np.std(speeds)

speeds_steady = speeds[1:] 
avg_speed_steady = np.mean(speeds_steady)
std_speed_steady = np.std(speeds_steady)

print(f"\nOverall average speed: {avg_speed:.4f} ± {std_speed:.4f} cm/s")
print(f"Steady-state average (excluding start): {avg_speed_steady:.4f} ± {std_speed_steady:.4f} cm/s")

# total distance and total time
total_distance = distance[-1]
total_time = np.sum(time_elapsed)
overall_speed = total_distance / total_time

print(f"\nTotal distance: {total_distance} cm")
print(f"Total time: {total_time:.2f} s")
print(f"Overall average speed: {overall_speed:.4f} cm/s")


pwm_used = 0.30
motor_full_speed = avg_speed_steady / pwm_used 

print(f"\nAt PWM = {pwm_used*100:.0f}%:")
print(f"  Measured speed: {avg_speed_steady:.4f} cm/s")
print(f"  Estimated full speed (100% PWM): {motor_full_speed:.4f} cm/s")

# plotting
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
fig4.suptitle('Task 4: Motor Control - Robot Speed', fontsize=14, fontweight='bold')

ax1 = axes4[0]
cumulative_time = np.cumsum(time_elapsed)
cumulative_time = np.insert(cumulative_time, 0, 0)
distance_with_start = np.insert(distance, 0, 0)
ax1.plot(cumulative_time, distance_with_start, 'bo-', markersize=8, linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Distance (cm)')
ax1.set_title('Distance vs Time')
ax1.grid(True, alpha=0.3)

ax2 = axes4[1]
segment_labels = [f'{d:.0f}cm' for d in distance]
bars = ax2.bar(segment_labels, speeds, color='steelblue', alpha=0.7)
ax2.axhline(y=avg_speed_steady, color='r', linestyle='--', linewidth=2, 
            label=f'Steady avg: {avg_speed_steady:.2f} cm/s')
ax2.set_xlabel('Distance segment')
ax2.set_ylabel('Speed (cm/s)')
ax2.set_title('Speed per Segment')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:\\Project\\sensor_fusion_project\\output\\part1\\task4_motor_speed.png", dpi=150, bbox_inches='tight')
print("\nPlot saved: task4_motor_speed.png")


# conclusion

print("\n" + "=" * 70)
print("SUMMARY: Part I Calibration Results")
print("=" * 70)


print("                   TASK 1: Gyroscope                           ")
print(f"  Bias X:     {gyro_bias_x:>12.6f} deg/s                      ")
print(f"  Bias Y:     {gyro_bias_y:>12.6f} deg/s                      ")
print(f"  Bias Z:     {gyro_bias_z:>12.6f} deg/s   this will used in Part II  ")
print(f"  Variance X: {gyro_var_x:>12.6f} (deg/s)^2                    ")
print(f"  Variance Y: {gyro_var_y:>12.6f} (deg/s)^2                    ")
print(f"  Variance Z: {gyro_var_z:>12.6f} (deg/s)^2                    ")


print("\n")
print("                    TASK 2: Accelerometer                      ")
print(f"  Gain k_x:   {k_x:>12.6f}                                    ")
print(f"  Gain k_y:   {k_y:>12.6f}                                    ")
print(f"  Gain k_z:   {k_z:>12.6f}                                    ")
print(f"  Bias b_x:   {b_x:>12.6f} g                                  ")
print(f"  Bias b_y:   {b_y:>12.6f} g                                  ")
print(f"  Bias b_z:   {b_z:>12.6f} g                                  ")


print("\n")
print("                   TASK 3: Camera                               ")

print(f"  Focal length f: {focal_length:>10.4f}   ")
print(f"  Bias b:         {intercept:>10.4f} cm                        ")
print(f"  R²:             {r_value**2:>10.6f}                          ")


print("\n")
print("                   TASK 4: Motor Speed                          ")

print(f"  Speed at 30% PWM:  {avg_speed_steady:>8.4f} cm/s  this will used in Part II ")
print(f"  Est. full speed:   {motor_full_speed:>8.4f} cm/s ")


print("\n" + "=" * 70)
print("Part I Complete!")
print("=" * 70)

# Save parameters to file for Part II use
calibration_params = {
    'gyro_bias_x': gyro_bias_x,
    'gyro_bias_y': gyro_bias_y,
    'gyro_bias_z': gyro_bias_z,
    'gyro_var_x': gyro_var_x,
    'gyro_var_y': gyro_var_y,
    'gyro_var_z': gyro_var_z,
    'acc_gain_x': k_x,
    'acc_gain_y': k_y,
    'acc_gain_z': k_z,
    'acc_bias_x': b_x,
    'acc_bias_y': b_y,
    'acc_bias_z': b_z,
    'focal_length': focal_length,
    'camera_bias': intercept,
    'motor_speed_30pct': avg_speed_steady,
    'motor_full_speed': motor_full_speed
}

np.savez('D:\\Project\\sensor_fusion_project\\output\\part1\\calibration_parameters.npz', **calibration_params)
print("\nCalibration parameters saved to: calibration_parameters.npz")

# text file
with open('D:\\Project\\sensor_fusion_project\\output\\part1\\calibration_parameters.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ELEC-E8740 Part I Calibration Parameters\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("TASK 1: Gyroscope\n")
    f.write("-" * 40 + "\n")
    f.write(f"Bias X: {gyro_bias_x:.6f} deg/s\n")
    f.write(f"Bias Y: {gyro_bias_y:.6f} deg/s\n")
    f.write(f"Bias Z: {gyro_bias_z:.6f} deg/s\n")
    f.write(f"Variance X: {gyro_var_x:.6f} (deg/s)^2\n")
    f.write(f"Variance Y: {gyro_var_y:.6f} (deg/s)^2\n")
    f.write(f"Variance Z: {gyro_var_z:.6f} (deg/s)^2\n\n")
    
    f.write("TASK 2: Accelerometer\n")
    f.write("-" * 40 + "\n")
    f.write(f"Gain k_x: {k_x:.6f}\n")
    f.write(f"Gain k_y: {k_y:.6f}\n")
    f.write(f"Gain k_z: {k_z:.6f}\n")
    f.write(f"Bias b_x: {b_x:.6f} g\n")
    f.write(f"Bias b_y: {b_y:.6f} g\n")
    f.write(f"Bias b_z: {b_z:.6f} g\n\n")
    
    f.write("TASK 3: Camera\n")
    f.write("-" * 40 + "\n")
    f.write(f"Focal length: {focal_length:.4f} pixels, this will used in Part II\n")
    f.write(f"Bias: {intercept:.4f} cm\n")
    f.write(f"R^2: {r_value**2:.6f}\n\n")
    
    f.write("TASK 4: Motor Speed\n")
    f.write("-" * 40 + "\n")
    f.write(f"Speed at 30% PWM: {avg_speed_steady:.4f} cm/s\n")
    f.write(f"Estimated full speed: {motor_full_speed:.4f} cm/s\n")

print("Text parameters saved to: calibration_parameters.txt")

plt.show()
