import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt 


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

def prismatic_twist(v):
    full_twist = np.zeros((4,4))
    if np.linalg.norm(v) != 0:
        full_twist[0:3, 3] = v/np.linalg.norm(v)

    return full_twist

def adjoint(T):
    adjoint_matrix = np.zeros((6,6))
    s_R_b = T[0:3, 0:3]
    adjoint_matrix[0:3, 0:3] = s_R_b
    adjoint_matrix[0:3, 3:6] = skew(T[0:3, 3]) @ s_R_b
    adjoint_matrix[3:6, 3:6] = s_R_b

    return adjoint_matrix

def twist_coords(twist):
    w = np.array([twist[2,1], twist[0,2], twist[1,0]])
    v = twist[0:3, 3]
    return np.concatenate((v,w)).reshape(6,1)

def inverse_condition_number(J):
    Jv = J[0:3, :]
    lin_U, lin_s, lin_Vt = np.linalg.svd(Jv)
    lin_sig_max = np.max(lin_s)
    lin_sig_min = np.min(lin_s)
    lin_inv_condition = lin_sig_min/lin_sig_max
    # U, s, Vt = np.linalg.svd(J)
    # sig_max = np.max(s)
    # sig_min = np.min(s)
    # inv_condition = sig_min/sig_max
    return lin_inv_condition#, inv_condition

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
        self.twist1 = 0
        self.twist2 = 0
        self.twist3 = 0
        self.joint1 = 0
        self.joint2 = 0
        self.joint3 = 0
        self.T_zero_config = np.array([[1, 0, 0, self.a2+self.a3], 
                                       [0, 1, 0, 0], 
                                       [0, 0, 1, self.d], 
                                       [0, 0, 0, 1]])
    def FK(self, theta1, theta2, theta3):
        v3 = np.cross(self.q3, self.k3)
        v2 = np.cross(self.q2, self.k2)
        v1 = np.cross(self.q1, self.k1)

        self.twist3 = twist(v3, self.k3)
        self.twist2 = twist(v2, self.k2)
        self.twist1 = twist(v1, self.k1)

        self.joint1 = expm(self.twist1 * theta1)
        self.joint2 = expm(self.twist2 * theta2)
        self.joint3 = expm(self.twist3 * theta3)

        forward_kinematics = self.joint1 @ self.joint2 @ self.joint3 @ self.T_zero_config

        return forward_kinematics 
    
    def body_jacobian(self, theta1, theta2, theta3):
        fk = self.FK(theta1, theta2, theta3)

        column1 = np.linalg.inv(adjoint(fk)) @ twist_coords(self.twist1)
        column2 = np.linalg.inv(adjoint(self.joint1@self.joint2@self.T_zero_config)) @ twist_coords(self.twist2)
        column3 = np.linalg.inv(adjoint(self.joint2@self.T_zero_config)) @ twist_coords(self.twist3)

        body_jacobian = np.hstack((column1, column2, column3))
        return body_jacobian
    
    def velocity_jacobian(self, theta1, theta2, theta3):
        jacobian_rotation = np.zeros((6,6))
        fk = self.FK(theta1, theta2, theta3)
        body_jacobian = self.body_jacobian(theta1, theta2, theta3)
        R_0n = fk[0:3, 0:3]
        jacobian_rotation[0:3, 0:3] = R_0n
        jacobian_rotation[3:6, 3:6] = R_0n 
        velocity_jacobian = jacobian_rotation @ body_jacobian
        return velocity_jacobian
    
    def body_velocity(self, theta1, theta2, theta3, theta_dot1, theta_dot2, theta_dot3):
        body_jacobian = self.body_jacobian(theta1, theta2, theta3)
        body_velocity = body_jacobian @ np.array([[theta_dot1, theta_dot2, theta_dot3]])
        return body_velocity
    
##############################################
#Problem 1
##############################################
a2 = 3
a3 = 2
d = 0
robot = Three_DOF_Spatial_Manipulator()
theta1 = np.arange(-np.pi, np.pi, 0.06)
theta2 = np.arange(-np.pi/4, 5*np.pi/4, 0.05)
theta3 = np.arange(-5*np.pi/6, 5*np.pi/6, 0.05)
end_vec = []
ICN = []

for i in theta2:
    for k in theta3:
        fk = robot.FK(0, i, k)
        end_pos = fk[0:3, 3]
        J = robot.velocity_jacobian(0, i, k)
        ICN.append(inverse_condition_number(J))
        end_vec.append(end_pos)

end_vec = np.array(end_vec)
X = end_vec[:, 0]
Y = end_vec[:, 1]
# Plot workspace
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(end_vec[:, 0], end_vec[:, 1], s=5, c='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Workspace of 3-DOF Spatial Manipulator')
ax.axis('equal')
plt.show()


plt.figure(figsize=(6, 5))
plt.scatter(
    X, Y,
    c=ICN,
    cmap='gray_r',    # grayscale
    vmin=0, vmax=1
)
plt.colorbar(label='Inverse Condition Number')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Workspace (Inverse Condition Number)')
plt.show()

#######################################
#Problem 2
#######################################
J1 = np.array([[10, 0, 0, 0, 2, 0], 
               [0, 8, 0, 0, 1, 0], 
               [0, 0, 7, 0, 0, 0], 
               [0, 0, 0, 6, 0, 0], 
               [0, 0, 0, 0, 3, 1], 
               [0, 0, 0, 0, 0, 1]])
J2 = np.array([[10, 0, 0, 0, 0, 2], 
               [0, 5, 0, 2, 0, 0], 
               [0, 0, 4, 1, -3, 0], 
               [0, 0, 0, 2, 2, 0], 
               [0, 0, 0, 0, 1, 0], 
               [0, 0, 0, 0, 0, 0]])
J3 = np.array([[9, -1, 0, 0, 0, 0, 0],
               [0, 8, 0, 0, 0, 0, 0],
               [0, 0, 5, 0, 1, 3, 0],
               [0, 0, 0, 4, 0, 0, -6],
               [0, 0, 0, 0, 3, 6, 0],
               [0, 0, 0, 0, 0, 0, 0]])
J4 = np.array([[8, -2, 0, 0],
               [0, 7, 0, 0],
               [0, 0, 3, 0],
               [0, 0, 2, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])

u1, s1, v1 = np.linalg.svd(J1)
u2, s2, v2 = np.linalg.svd(J2)
u3, s3, v3 = np.linalg.svd(J3)
u4, s4, v4 = np.linalg.svd(J4)

print("u1 is: \n", u1)
print("s1 is: \n", s1)
print("v1 is: \n", v1)

print("u2 is: \n", u2)
print("s2 is: \n", s2)
print("v2 is: \n", v2)

print("u3 is: \n", u3)
print("s3 is: \n", s3)
print("v3 is: \n", v3)

print("u4 is: \n", u4)
print("s4 is: \n", s4)
print("v4 is: \n", v4)