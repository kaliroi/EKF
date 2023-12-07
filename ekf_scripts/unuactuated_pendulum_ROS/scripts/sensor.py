#!usr/bin/env python3

import rospy
import numpy as np
from unuactuated_pendulum.msg import Data_output
from unuactuated_pendulum.msg import Data_state

def callback(data):
    x=np.array([data.x1,data.x2])

    measurement_noise_var = 0.05  # Actual measurement noise variance (uknown to the user)

    # Measurement model
    z_t = x + np.sqrt(measurement_noise_var)*np.random.randn()

    sensor_obj=Data_output()
    sensor_obj.y=z_t[0]

    pub.publish(sensor_obj)


def sensor(): 
    #Subscribe
    rospy.init_node('sensor', anonymous=True)
    rospy.Subscriber('true_variables',Data_state,callback)

    rospy.spin()

if __name__== '__main__':
    try:
        #Publish: se lo mettiamo dentro il finto main lo vedono tutti e non serve ridefinirlo nell'EKF
        pub = rospy.Publisher('sensor_data',Data_output,queue_size=10)
        sensor()
    except rospy.ROSInterruptException:
        pass
