import matplotlib.pyplot as plt
import numpy as np

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Linear Quadratic Regulator example
#   (two-wheeled differential drive robot car)

######################## DEFINE CONSTANTS #####################################
# Supress scientific notation when printing NumPy arrays
from scipy.linalg import _solvers

np.set_printoptions(precision=3, suppress=True)

# Optional Variables
max_linear_velocity = 1.0  # meters per second
max_angular_velocity = 0.2708  # radians per second
def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.matrix(_solvers.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.matrix(np.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    return -K


def getB(yaw, deltat):
    """
    Calculates and returns the B matrix
    3x2 matix ---> number of states x number of control inputs

    Expresses how the state of the system [x,y,yaw] changes
    from t-1 to t due to the control commands (i.e. control inputs).

    :param yaw: The yaw angle (rotation angle around the z axis) in radians
    :param deltat: The change in time from timestep t-1 to t in seconds

    :return: B matrix ---> 3x2 NumPy array
    """
    B = np.array([[np.cos(yaw) * deltat, 0],
                  [np.sin(yaw) * deltat, 0],
                  [0, deltat]])
    return B


def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
    """
    Calculates the state at time t given the state at time t-1 and
    the control inputs applied at time t-1

    :param: A   The A state transition matrix
        3x3 NumPy Array
    :param: state_t_minus_1     The state at time t-1
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    :param: B   The B state transition matrix
        3x2 NumPy Array
    :param: control_input_t_minus_1     Optimal control inputs at time t-1
        2x1 NumPy Array given the control input vector is
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]

    :return: State estimate at time t
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    """
    # These next 6 lines of code which place limits on the angular and linear
    # velocities of the robot car can be removed if you desire.
    control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
                                         -max_linear_velocity,
                                         max_linear_velocity)
    control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],
                                         -max_angular_velocity,
                                         max_angular_velocity)
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1)

    return state_estimate_t


log = []

def main():
    # Let the time interval be 1.0 seconds
    dt = 0.1

    # Actual state
    # Our robot starts out at the origin (x=0 meters, y=0 meters), and
    # the yaw angle is 0 radians.
    actual_state_x = np.array([0, 0, 0])

    # Desired state [x,y,yaw angle]
    # [meters, meters, radians]
    desired_state_xf = np.array([2.000, 2.000, np.pi / 2])

    # A matrix
    # 3x3 matrix -> number of states x number of states matrix
    # Expresses how the state of the system [x,y,yaw] changes
    # from t-1 to t when no control command is executed.
    # Typically a robot on wheels only drives when the wheels are told to turn.
    # For this case, A is the identity matrix.
    # Note: A is sometimes F in the literature.
    A = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    # R matrix
    # The control input cost matrix
    # Experiment with different R matrices
    # This matrix penalizes actuator effort (i.e. rotation of the
    # motors on the wheels that drive the linear velocity and angular velocity).
    # The R matrix has the same number of rows as the number of control
    # inputs and same number of columns as the number of
    # control inputs.
    # This matrix has positive values along the diagonal and 0s elsewhere.
    # We can target control inputs where we want low actuator effort
    # by making the corresponding value of R large.
    R = np.array([[0.01, 0],  # Penalty for linear velocity effort
                  [0, 0.01]])  # Penalty for angular velocity effort

    # Q matrix
    # The state cost matrix.
    # Experiment with different Q matrices.
    # Q helps us weigh the relative importance of each state in the
    # state vector (X, Y, YAW ANGLE).
    # Q is a square matrix that has the same number of rows as
    # there are states.
    # Q penalizes bad performance.
    # Q has positive values along the diagonal and zeros elsewhere.
    # Q enables us to target states where we want low error by making the
    # corresponding value of Q large.
    Q = np.array([[1, 0, 0],  # Penalize X position error
                  [0, 1, 0],  # Penalize Y position error
                  [0, 0, 1.0]])  # Penalize YAW ANGLE heading error

    # Launch the robot, and have it move to the desired goal destination
    t = 0
    for i in range(100):
        print(f'iteration = {t} seconds')
        print(f'Current State = {actual_state_x}')
        print(f'Desired State = {desired_state_xf}')
        log.append(actual_state_x)
        state_error = actual_state_x - desired_state_xf
        state_error_magnitude = np.linalg.norm(state_error)
        print(f'State Error Magnitude = {state_error_magnitude}')

        B = getB(actual_state_x[2], dt)

        # LQR returns the optimal control input
        # optimal_control_input = lqr(actual_state_x,
        #                             desired_state_xf,
        #                             Q, R, A, B, dt)
        K,S,E = dlqr(A,B,Q,R)
        optimal_control_input = K @ state_error

        optimal_control_input[0] = np.clip(optimal_control_input[0],
                                             -max_linear_velocity,
                                             max_linear_velocity)
        optimal_control_input[1] = np.clip(optimal_control_input[1],
                                             -max_angular_velocity,
                                             max_angular_velocity)

        print(f'Control Input = {optimal_control_input}')

        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B,
                                           optimal_control_input)
        t += dt
        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 0.05:
            print("\nGoal Has Been Reached Successfully!")
            break

        print()

# Entry point for the program
main()
log=np.array(log)
x,y,t = log[:,0],log[:,1],log[:,2]
plt.figure(2)
plt.plot(x,y,label='x,y')
plt.figure(1)
plt.plot(t,label='t')
plt.show()