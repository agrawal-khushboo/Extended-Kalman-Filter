import numpy as np
from utils import *

class robo(object):
	def __init__(self):
		self.mean = np.identity(4) 
		self.cov = 0.005*np.identity(6)

class Landmarks(object):
	def __init__(self,M):
		self.mean = np.zeros((M,4))
		self.cov = np.zeros((M,3,3))
		for i in range(M):
			self.cov[i,:,:] = np.identity(3)*0.005


if __name__ == '__main__':
	filename = "0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)


	car = robo()
	Landmarks = Landmarks(features.shape[1])


	position = np.zeros((4,4,t.shape[1]))

	for i,t_n in enumerate(t[0][1:],1):


		timestamp = abs(t_n - t[:,i-1])

		predict(car,timestamp,linear_velocity[:,i],rotational_velocity[:,i])


		position[:,:,i] = world_T_robo(car.mean)


		update(car,Landmarks,timestamp,features[:,:,i-1],features[:,:,i],K,b,cam_T_imu)

	visualize_trajectory_2d(position,Landmarks.mean,path_name="testset",show_ori=True)





