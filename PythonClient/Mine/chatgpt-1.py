import airsim
import numpy as np


def generate_optical_flow_data():
    # Generate random data for the optical flow sensor
    time_stamp = np.random.randint(1000000000)
    velocity = np.random.uniform(-10, 10, size=3)
    integration_time_us = np.random.randint(1000, 10000)
    pixel_flow_x_integral = np.random.uniform(-100, 100)
    pixel_flow_y_integral = np.random.uniform(-100, 100)
    quality = np.random.randint(256)

    # Create an OpticalFlowSensorData object and return it
    optical_flow_data = airsim.OpticalFlowSensorData()
    optical_flow_data.time_stamp = time_stamp
    optical_flow_data.velocity.x_val = velocity[0]
    optical_flow_data.velocity.y_val = velocity[1]
    optical_flow_data.velocity.z_val = velocity[2]
    optical_flow_data.integration_time_us = integration_time_us
    optical_flow_data.pixel_flow_x_integral = pixel_flow_x_integral
    optical_flow_data.pixel_flow_y_integral = pixel_flow_y_integral
    optical_flow_data.quality = quality

    return optical_flow_data


def kalman_filter(x, P, y_gyro, y_optflow, F, B, R_gyro, R_optflow, Q):
    # Predict
    x = F.dot(x) + B.dot(thrust)
    P = F.dot(P).dot(F.T) + Q

    # Update
    K_gyro = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R_gyro))
    K_optflow = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R_optflow))

    x = x + K_gyro.dot(y_gyro) + K_optflow.dot(y_optflow)
    P = (np.eye(6) - K_gyro.dot(H)).dot(P).dot((np.eye(6) - K_gyro.dot(H)).T) + K_gyro.dot(R_gyro).dot(K_gyro.T)
    P = (np.eye(6) - K_optflow.dot(H)).dot(P).dot((np.eye(6) - K_optflow.dot(H)).T) + K_optflow.dot(R_optflow).dot(K_optflow.T)

    return x, P


# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Initialize the quadcopter
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Set the desired altitude and hover location
hover_altitude = -5  # meters
hover_location = airsim.Vector3r(0, 0, hover_altitude)

# Initialize the Kalman filter and PID controller
x = np.zeros((6, 1))
P = np.diag([1, 1, 1, 1, 1, 1])
F = np.array([[1, 0, 0.01, 0, 0, 0], [0, 1, 0, 0.01, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
R_gyro = np.diag([0.1, 0.1, 0.1])
R_optflow = np.diag([10, 10])
Q = np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
B = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
thrust = 0.5
fused_x = [x]
optflow_data = [0, 0]
pid_kp = 0.1
pid_ki = 0.001
pid_kd = 0.01

pid_integral = 0
pid_derivative = 0
pid_last_error = 0

# Define the initial PID gains and learning rate
kp = 0.5
ki = 0.001
kd = 0.1
learning_rate = 0.01

# Define the number of iterations and the convergence threshold
num_iterations = 1000
convergence_threshold = 0.01

# Tune the PID gains using gradient descent
for i in range(num_iterations):
    # Initialize the error, integral, and derivative
    error = 0
    integral = 0
    derivative = 0

    # Hover the quadcopter and record the error, integral, and derivative over a short time period
    for _ in range(10):
        # Read the IMU data and optical flow sensor readings
        imu_data = client.getImuData()
        optflow_data = generate_optical_flow_data()  # client.getOpticalFlowSensorData()
        if optflow_data:
            z_optflow = np.array([optflow_data.velocity.x_val, optflow_data.velocity.y_val]).reshape((2, 1))
            y_optflow = z_optflow - H.dot(x[2:4])
            y_gyro = np.array([imu_data.angular_velocity.x_val, imu_data.angular_velocity.y_val, imu_data.angular_velocity.z_val]).reshape((3, 1))

            # Apply the Kalman filter to fuse the sensor readings
            x, P = kalman_filter(x, P, y_gyro, y_optflow, F, B, R_gyro, R_optflow, Q)
            fused_x.append(x)

            # Use a PID controller to adjust the thrust based on the estimated position and velocity
            error = hover_location - airsim.Vector3r(x[0], x[1], hover_altitude)
            pid_integral = pid_integral + error
            pid_derivative = error - pid_last_error
            pid_last_error = error
            pid_output = pid_kp * error + pid_ki * pid_integral + pid_kd * pid_derivative
            thrust += pid_output

            # Send the updated thrust command to the quadcopter
            client.moveByAngleRatesThrottleAsync(0, 0, 0, thrust, 0.1)

    # Update the PID gains using gradient descent
    kp_gradient = 0
    ki_gradient = 0
    kd_gradient = 0
    for j in range(len(fused_x) - 1):
        # Compute the error between the current and next position estimates
        current_error = np.linalg.norm(fused_x[j][:2] - hover_location.to_numpy_array().reshape((2, 1)))
        next_error = np.linalg.norm(fused_x[j + 1][:2] - hover_location.to_numpy_array().reshape((2, 1)))
        error_change = next_error - current_error

        # Update the PID gradients
        kp_gradient += error_change * (fused_x[j][0] - hover_location.x_val)
        ki_gradient += error_change * (fused_x[j][0] - hover_location.x_val) * j
        kd_gradient += error_change * (fused_x[j][0] - hover_location.x_val - pid_last_error) / j

    # Update the PID gains using the gradients and learning rate
    kp += learning_rate * kp_gradient
    ki += learning_rate * ki_gradient
    kd += learning_rate * kd_gradient

    # Check for convergence
    if abs(kp_gradient) < convergence_threshold and abs(ki_gradient) < convergence_threshold and abs(kd_gradient) < convergence_threshold:
        break
