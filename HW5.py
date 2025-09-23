import numpy as np
from scipy.linalg import expm



def skew(vec):
    skew_matrix = np.array([[0, -vec[2], vec[1]], 
                            [vec[2], 0, -vec[0]], 
                            [-vec[1], vec[0], 0]])
    
    return skew_matrix

def twist(v, k):
    full_twist = np.zeros((4,4))

    full_twist[0:3,0:3] = skew(k)
    full_twist[0:3, 3] = v

    return(full_twist)
    


#############################################
#Problem 1
#############################################
class Three_DOF_Spatial_Manipulator:
    def __init__(self):
        self.a2 = 1
        self.a3 = 0.9
        self.d = 0
        self.k1 = np.array([0, 1, 0])
        self.k2 = np.array([0, 0, 1])
        self.k3 = np.array([0, 0, 1])
        self.q1 = np.array([0, 0, 0])
        self.q2 = np.array([0, 0, 0])
        self.q3 = np.array([self.a2, 0, 0])
        self.T_zero_config = np.array([[1, 0, 0, self.a2+self.a3], 
                                       [0, 1, 0, 0], 
                                       [0, 0, 1, self.d], 
                                       [0, 0, 0, 1]])
    def FK(self, theta1, theta2, theta3):
        v3 = np.cross(self.q3, self.k3)
        v2 = np.cross(self.q2, self.k2)
        v1 = np.cross(self.q1, self.k1)

        twist3 = twist(v3, self.k3)
        twist2 = twist(v2, self.k2)
        twist1 = twist(v1, self.k1)

        forward_kinematics = expm(twist1 * theta1) @ expm(twist2 * theta2) @ expm(twist3 * theta3) @ self.T_zero_config

        return forward_kinematics 
    

robot = Three_DOF_Spatial_Manipulator()

T_zero = robot.FK(0, 0, 0)
T_theta1_90 = robot.FK(np.pi/2, 0, 0)
T_theta2_90 = robot.FK(0, np.pi/2, 0)
T_theta3_90 = robot.FK(0, 0, np.pi/2) 

print("0 angle: \n", T_zero)
print("theta 1 = 90: \n", T_theta1_90)
print("theta 2 = 90: \n", T_theta2_90)
print("theta 3 = 90: \n", T_theta3_90)


#############################################
#Problem 2
#############################################




