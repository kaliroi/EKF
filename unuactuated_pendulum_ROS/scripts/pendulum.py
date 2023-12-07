#!usr/bin/env python3

import rospy
import numpy as np
from scipy.integrate import odeint
from unuactuated_pendulum.msg import Data_state

# System dynamics (continuous, non-linear) in state-space representation (https://en.wikipedia.org/wiki/State-space_representation)
def stateSpaceModel(x,t):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """
    g=9.81
    l=1
    dxdt=np.array([x[1], -(g/l)*np.sin(x[0])])
    return dxdt


def pendulum():
    # Discretization time step (frequency of measurements)
    deltaTime=0.01

    # Initial true state
    x0 = np.array([np.pi/3, 0.5])

    # Simulation duration in timesteps
    simulationSteps=400
    totalSimulationTimeVector=np.arange(0, simulationSteps*deltaTime, deltaTime)

    # True solution x(t)
    x_t_true = odeint(stateSpaceModel, x0, totalSimulationTimeVector)

    #Publish
    pub = rospy.Publisher('true_variables',Data_state,queue_size=10)
    rospy.init_node('pendulum',anonymous=True)
    rate = rospy.Rate(10) #10 hz

    #Send messages
    for i in range(0,simulationSteps):
        data_obj=Data_state()
        data_obj.x1=x_t_true[i,0]
        data_obj.x2=x_t_true[i,1]

        pub.publish(data_obj)
        rate.sleep()

if __name__== '__main__':
    try:
        pendulum()
    except rospy.ROSInterruptException:
        pass

