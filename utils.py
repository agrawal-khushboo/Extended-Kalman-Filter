import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"] # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      K = data["K"] # intrinsic calibration matrix
      b = data["b"] # baseline
      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
  return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu


def visualize_trajectory_2d(pose,landmark,path_name="Unknown",show_ori=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  ax.plot(landmark[:,0],landmark[:,1],'b.',label='landmarks')
  if show_ori:
      select_ori_index = list(range(0,n_pose,int(n_pose/50)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax



def DF(q):
    return (1/q[2])*np.array([[1,0,-q[0]/q[2],0],[0,1,-q[1]/q[2],0],[0,0,0,0],[0,0,-q[3]/q[2],1]])


def world_T_robo(X):
    R = X[0:3,0:3]
    T = np.vstack((np.hstack((np.transpose(R), -np.dot(np.transpose(R),X[0:3,3].reshape(3,1)))),np.array([0,0,0,1])))
    return T


# def skew(x):
#     return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])

def predict(robo,timestamp,lv,av):
    W = 500*np.identity(6)
    u_hat = np.vstack((np.hstack((skew(av), lv.reshape(3,1))),np.array([0,0,0,0])))
    u_cov = np.vstack((np.hstack((skew(av), skew(lv))),np.hstack((np.zeros((3,3)),skew(av)))))
    robo.cov = expm(-float(timestamp[0])*u_cov)*robo.cov*np.transpose(expm(-float(timestamp[0])*u_cov)) + W
    robo.mean = np.dot(expm(-float(timestamp[0])*u_hat),robo.mean)


def update(robo,Landmarks,timestamp,prefeature,features,K,b,cam_T_imu):
    n = features.shape[1]
    M = np.hstack((np.vstack((K[0:2,0:3],K[0:2,0:3])),np.array([0,0,-K[0,0]*b,0]).reshape(4,-1))
    M=M.reshape(4,-1)
    D = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
    V = 3000
    for i in range(n):
        if features[:,i][0] == -1:
            continue
        if prefeature[:,i][0] == -1 and features[:,i][0] != -1 and Landmarks.mean[i,:][0] == 0 and Landmarks.mean[i,:][1] == 0 and Landmarks.mean[i,:][2] == 0:
            Landmarks.mean[i,:] = np.dot(world_T_robo(robo.mean),np.dot(np.linalg.inv(cam_T_imu),np.hstack((K[0,0]*b*np.dot(np.linalg.inv(K),np.hstack((features[:,i][0:2],1)))/(features[:,i][0] - features[:,i][2]),1))))
            continue
        l = np.dot(cam_T_imu,np.dot(robo.mean,Landmarks.mean[i,:]))
        z = np.dot(M,(l/l[2]))
        H = np.dot(M,np.dot(DF(l),np.dot(cam_T_imu,np.dot(robo.mean,D))))
        KF = np.dot(Landmarks.cov[i,:,:],np.dot(np.transpose(H),np.linalg.inv(np.dot(H,np.dot(Landmarks.cov[i,:,:],np.transpose(H))) + V*np.identity(4)[0:5,0:4])))
        Landmarks.mean[i,:] = Landmarks.mean[i,:] + np.dot(D,np.dot(KF,(features[:,i] - z)))
        Landmarks.cov[i,:,:] = np.dot((np.identity(3) - np.dot(KF,H)),Landmarks.cov[i,:,:])
        





