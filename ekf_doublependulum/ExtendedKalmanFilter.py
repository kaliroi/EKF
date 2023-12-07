"""
    Implementation of the Extended Kalman Filter
    for an unactuated pendulum system
"""
import numpy as np 
from numpy import sin, cos

class ExtendedKalmanFilter(object):
    
    
    def __init__(self, x0, P0, Q, R, dT, l1, l2, m1, m2):
        """
           Initialize EKF
            
            Parameters
            x0 - mean of initial state prior
            P0 - covariance of initial state prior
            Q  - covariance matrix of the process noise 
            R  - covariance matrix of the measurement noise
            dT - discretization step for forward dynamics
        """
        self.x0=x0
        self.P0=P0
        self.Q=Q
        self.R=R
        self.dT=dT
        
        
        self.g = 9.81  # Gravitational constant
        self.l1 = l1  # Length of the pendulum 
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        
        self.currentTimeStep = 0
        

        self.priorMeans = []
        self.priorMeans.append(None)  # no prediction step for timestep=0
        self.posteriorMeans = []
        self.posteriorMeans.append(x0)
        
        self.priorCovariances=[]
        self.priorCovariances.append(None)  # no prediction step for timestep=0
        self.posteriorCovariances=[]
        self.posteriorCovariances.append(P0)
    
    

    def stateSpaceModel(self, x,t):
        """
            Dynamics may be described as a system of first-order
            differential equations: 
            dx(t)/dt = f(t, x(t))
        """
        th1 = x[0]
        th2 = x[2]
        th1d = x[1]
        th2d = x[3]

        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        g = self.g

        dxdt=np.array(
            [x[1], 
            -(l1*m2*cos(th1 - th2)*sin(th1 - th2)*th1d**2 + l2*m2*sin(th1 - th2)*th2d**2 + g*m1*sin(th1) + g*m2*sin(th1) - g*m2*cos(th1 - th2)*sin(th2))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2)),
            x[3],
            (g*m1*cos(th1 - th2)*sin(th1) - g*m2*sin(th2) - g*m1*sin(th2) + g*m2*cos(th1 - th2)*sin(th1) + l1*m1*th1d**2*sin(th1 - th2) + l1*m2*th1d**2*sin(th1 - th2) + l2*m2*th2d**2*cos(th1 - th2)*sin(th1 - th2))/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2))
            ]
        )
        return dxdt
    
    def discreteTimeDynamics(self, x_t):
        """
            Forward Euler integration.
            
            returns next state as x_t+1 = x_t + dT * (dx/dt)|_{x_t}
        """
        x_tp1 = x_t + self.dT*self.stateSpaceModel(x_t, None)
        return x_tp1
    

    def jacobianStateEquation(self, x_t):
        """
            Jacobian of discrete dynamics w.r.t. the state variables,
            evaluated at x_t

            Parameters:
                x_t : state variables (column-vector)
        """
        th1 = float(x_t[0])
        th2 = float(x_t[2])
        th1d = float(x_t[1])
        th2d = float(x_t[3])

        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        g = self.g
        dT = self.dT

        A=np.array([
            [1,dT,0,0],
            [-dT*((l1*m2*th1d**2*cos(th1 - th2)**2 - l1*m2*th1d**2*sin(th1 - th2)**2 + l2*m2*th2d**2*cos(th1 - th2) + g*m2*sin(th2)*sin(th1 - th2) + g*m1*cos(th1) + g*m2*cos(th1))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2)) - (2*m2*cos(th1 - th2)*sin(th1 - th2)*(l1*m2*cos(th1 - th2)*sin(th1 - th2)*th1d**2 + l2*m2*sin(th1 - th2)*th2d**2 + g*m1*sin(th1) + g*m2*sin(th1) - g*m2*cos(th1 - th2)*sin(th2)))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2)**2)),1 - (2*dT*m2*th1d*cos(th1 - th2)*sin(th1 - th2))/(- m2*cos(th1 - th2)**2 + m1 + m2), dT*((l1*m2*th1d**2*cos(th1 - th2)**2 - l1*m2*th1d**2*sin(th1 - th2)**2 + l2*m2*th2d**2*cos(th1 - th2) + g*m2*cos(th2)*cos(th1 - th2) + g*m2*sin(th2)*sin(th1 - th2))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2)) - (2*m2*cos(th1 - th2)*sin(th1 - th2)*(l1*m2*cos(th1 - th2)*sin(th1 - th2)*th1d**2 + l2*m2*sin(th1 - th2)*th2d**2 + g*m1*sin(th1) + g*m2*sin(th1) - g*m2*cos(th1 - th2)*sin(th2)))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2)**2)),-(2*dT*l2*m2*th2d*sin(th1 - th2))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2))],
            [0,0,1,dT],
            [dT*((g*m1*cos(th1 - th2)*cos(th1) + g*m2*cos(th1 - th2)*cos(th1) + l1*m1*th1d**2*cos(th1 - th2) + l1*m2*th1d**2*cos(th1 - th2) - g*m1*sin(th1 - th2)*sin(th1) - g*m2*sin(th1 - th2)*sin(th1) + l2*m2*th2d**2*cos(th1 - th2)**2 - l2*m2*th2d**2*sin(th1 - th2)**2)/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2)) - (2*m2*cos(th1 - th2)*sin(th1 - th2)*(g*m1*cos(th1 - th2)*sin(th1) - g*m2*sin(th2) - g*m1*sin(th2) + g*m2*cos(th1 - th2)*sin(th1) + l1*m1*th1d**2*sin(th1 - th2) + l1*m2*th1d**2*sin(th1 - th2) + l2*m2*th2d**2*cos(th1 - th2)*sin(th1 - th2)))/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2)**2)), (dT*(2*l1*m1*th1d*sin(th1 - th2) + 2*l1*m2*th1d*sin(th1 - th2)))/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2)), -dT*((g*m1*cos(th2) + g*m2*cos(th2) + l1*m1*th1d**2*cos(th1 - th2) + l1*m2*th1d**2*cos(th1 - th2) - g*m1*sin(th1 - th2)*sin(th1) - g*m2*sin(th1 - th2)*sin(th1) + l2*m2*th2d**2*cos(th1 - th2)**2 - l2*m2*th2d**2*sin(th1 - th2)**2)/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2)) - (2*m2*cos(th1 - th2)*sin(th1 - th2)*(g*m1*cos(th1 - th2)*sin(th1) - g*m2*sin(th2) - g*m1*sin(th2) + g*m2*cos(th1 - th2)*sin(th1) + l1*m1*th1d**2*sin(th1 - th2) + l1*m2*th1d**2*sin(th1 - th2) + l2*m2*th2d**2*cos(th1 - th2)*sin(th1 - th2)))/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2)**2)), (2*dT*m2*th2d*cos(th1 - th2)*sin(th1 - th2))/(- m2*cos(th1 - th2)**2 + m1 + m2) + 1]
            ], dtype=float)

        return A
    
    
    def jacobianMeasurementEquation(self, x_t):
        """
            Jacobian of measurement model.

            Measurement model is linear, hence its Jacobian
            does not actually depend on x_t
        """
        C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        return C
    
     
    def forwardDynamics(self):
        self.currentTimeStep = self.currentTimeStep+1  # t-1 ---> t

        
        """
            Predict the new prior mean for timestep t
        """
        x_t_prior_mean = self.discreteTimeDynamics(self.posteriorMeans[self.currentTimeStep-1])
        

        """
            Predict the new prior covariance for timestep t
        """
        # Linearization: jacobian of the dynamics at the current a posteriori estimate
        A_t_minus = self.jacobianStateEquation(self.posteriorMeans[self.currentTimeStep-1])

        # Propagate the covariance matrix forward in time
        x_t_prior_cov = A_t_minus @ self.posteriorCovariances[-1] @ A_t_minus.T+self.Q
        
        # Save values
        self.priorMeans.append(x_t_prior_mean)
        self.priorCovariances.append(x_t_prior_cov)
    

    def updateEstimate(self, z_t):
        """
            Compute Posterior Gaussian distribution,
            given the new measurement z_t
        """

        # Jacobian of measurement model at x_t
        Ct = self.jacobianMeasurementEquation(self.priorMeans[self.currentTimeStep]) 
        
        # Compute the Kalman gain matrix
        K_t = self.priorCovariances[-1] @ Ct.T @ np.linalg.inv(Ct @ self.priorCovariances[-1] @ Ct.T + self.R)
        
        # Compute posterior mean
        x_t_mean = self.priorMeans[-1] + K_t @ (z_t-Ct @ self.priorMeans[-1])
        
        # Compute posterior covariance
        x_t_cov = (np.eye(4)-K_t @ Ct) @ self.priorCovariances[-1]
        
        # Save values
        self.posteriorMeans.append(x_t_mean)
        self.posteriorCovariances.append(x_t_cov)
