import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint
from numpy import sin, cos
import warnings
import matplotlib
from ExtendedKalmanFilter import ExtendedKalmanFilter

warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)

# Discretization time step (frequency of measurements)
deltaTime=0.01

# Initial true state
x0 = np.array([np.pi/2, 0, np.pi/2, 0],dtype=float)

# Simulation duration in timesteps
simulationSteps=4000
totalSimulationTimeVector=np.arange(0, simulationSteps*deltaTime, deltaTime)

l1=1
l2=2
m1=1
m2=2
g=9.81

# System dynamics (continuous, non-linear) in state-space representation (https://en.wikipedia.org/wiki/State-space_representation)
def stateSpaceModel(x,t):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """
    th1 = x[0]
    th2 = x[2]
    th1d = x[1]
    th2d = x[3]
    dxdt=np.array(
            [x[1], 
            -(l1*m2*cos(th1 - th2)*sin(th1 - th2)*th1d**2 + l2*m2*sin(th1 - th2)*th2d**2 + g*m1*sin(th1) + g*m2*sin(th1) - g*m2*cos(th1 - th2)*sin(th2))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2)),
            x[3],
            (g*m1*cos(th1 - th2)*sin(th1) - g*m2*sin(th2) - g*m1*sin(th2) + g*m2*cos(th1 - th2)*sin(th1) + l1*m1*th1d**2*sin(th1 - th2) + l1*m2*th1d**2*sin(th1 - th2) + l2*m2*th2d**2*cos(th1 - th2)*sin(th1 - th2))/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2))
            ]
    )
    return dxdt

# True solution x(t)
x_t_true = odeint(stateSpaceModel, x0, totalSimulationTimeVector)

time = np.arange(simulationSteps)*deltaTime

"""
    EKF initialization
"""
# Initial state belief distribution (EKF assumes Gaussian distributions)
x_0_mean = np.zeros(shape=(4,1), dtype=float)  # column-vector
x_0_cov = 10*np.eye(4,dtype=float)  # initial value of the covariance matrix

# Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
Q=0.00001*np.eye(4,dtype=float)

# Measurement noise covariance matrix for EKF
R = 0.05*np.eye(2,dtype=float)

# create the extended Kalman filter object
EKF = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime, l1, l2, m1, m2)

"""
    Simulate process
"""
measurement_noise_var = 0.05  # Actual measurement noise variance (uknown to the user)

for t in range(simulationSteps-1):
    # PREDICT step
    EKF.forwardDynamics()
    
    # Measurement model
    z_t = np.array((2,1), dtype=float).reshape((2,1))
    z_t[0] = x_t_true[t, 0] + np.sqrt(measurement_noise_var)*np.random.randn()
    z_t[1] = x_t_true[t, 2] + np.sqrt(measurement_noise_var)*np.random.randn()
    
    # UPDATE step
    EKF.updateEstimate(z_t)


"""
    Plot the true vs. estimated state variables
"""

x_tilde = np.hstack(EKF.posteriorMeans).T
time = np.arange(simulationSteps)*EKF.dT

fig, ax = plt.subplots(4)
ax[0].plot(time,x_tilde[:,0],'r')
ax[0].plot(time,x_t_true[:,0],'k')
ax[0].legend([r'Estimated $x_{1}$', r'True $x_{1}$'])
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$x_{1}$')
ax[0].grid()

ax[1].plot(time,x_tilde[:,1],'g')
ax[1].plot(time,x_t_true[:,1],'k')
ax[1].legend([r'Estimated $x_{2}$', r'True $x_{2}$'])
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$x_{2}$')
ax[1].grid()

ax[2].plot(time,x_tilde[:,2],'b')
ax[2].plot(time,x_t_true[:,2],'k')
ax[2].legend([r'Estimated $x_{3}$', r'True $x_{3}$'])
ax[2].set_xlabel(r'$t$')
ax[2].set_ylabel(r'$x_{3}$')
ax[2].grid()

ax[3].plot(time,x_tilde[:,3],'y')
ax[3].plot(time,x_t_true[:,3],'k')
ax[3].legend([r'Estimated $x_{4}$', r'True $x_{4}$'])
ax[3].set_xlabel(r'$t$')
ax[3].set_ylabel(r'$x_{4}$')
ax[3].grid()
plt.show()
