import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def interpolate(ctrl_pts, num_sampling, show=False):

    p0 = ctrl_pts[0]
    p1 = ctrl_pts[1]
    p2 = ctrl_pts[2]
    p3 = ctrl_pts[3]

    ts = np.linspace(0, 1, num_sampling).reshape(-1,1)
    t = np.hstack((np.power(ts, 3), np.power(ts, 2), ts, np.ones((num_sampling, 1))))
    
    c = np.array([-p0 + 3*p1 - 3*p2 + p3,
                    3*p0 - 6*p1 + 3*p2,
                    -3*p0 + 3*p1,
                    p0])
    
    waypoints = c[0]*np.power(ts, 3) + c[1]*np.power(ts, 2) + c[2]*ts + c[3]  
    
    if show:
        fig = plt.figure()
        ax = fig.gca()
        # ax = fig.gca(projection="3d")
        # ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2])
        # ax.plot(ctrl_pts[:,0], ctrl_pts[:,1], ctrl_pts[:,2], 'ro:')
        ax.plot(waypoints[:,0], waypoints[:,1])
        ax.plot(ctrl_pts[:,0], ctrl_pts[:,1], 'ro:')
        plt.show()

    return waypoints

if __name__ == "__main__":
    ctrl_pts = np.array([[0,0], [0.33,0.1], [0.66,0.9], [1,1]])

    interpolate(ctrl_pts, 100, True)