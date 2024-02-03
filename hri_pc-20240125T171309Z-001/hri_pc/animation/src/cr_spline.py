#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
# from mpl_toolkits import mplot3d

def catmull_rom_one_point(x, v0, v1, v2, v3):
    """Computes interpolated y-coord for given x-coord using Catmull-Rom.
    Computes an interpolated y-coordinate for the given x-coordinate between
    the support points v1 and v2. The neighboring support points v0 and v3 are
    used by Catmull-Rom to ensure a smooth transition between the spline
    segments.
    Args:
        x: the x-coord, for which the y-coord is needed
        v0: 1st support point
        v1: 2nd support point
        v2: 3rd support point
        v3: 4th support point
    """
    # tension t, should be set between 0 to 1. 
    t=0.5
    
    c1 = v1
    c2 = -t* v0 + t * v2
    c3 = 2*t * v0 + (t-3) * v1 + (3-2*t) * v2 -t * v3
    c4 = -t * v0 + (2-t) * v1 + (t-2) * v2 + t * v3
    return (((c4 * x + c3) * x + c2) * x + c1)


# Same principles with function above, but for 2 points:
# Estimating start and end points are a problem right now: arbitrary for now
def catmull_rom_2pt(p_x, p_y, c1, c2, num_sampling, hold_out=0.0):

    # p_x = np.array([start[1], end[1]])
    # p_y = np.array([start[2], end[2]])

    x_intpol = np.empty(num_sampling*(len(p_x)-1) + 1)
    y_intpol = np.empty(num_sampling*(len(p_x)-1) + 1)

    # set the last x- and y-coord, the others will be set in the loop
    x_intpol[-1] = p_x[-1]
    y_intpol[-1] = p_y[-1]

    for i in range(len(p_x)-1):
        # set x-coords
        x_intpol[i*num_sampling:(i+1)*num_sampling] = np.linspace(
            p_x[i], p_x[i+1], num_sampling, endpoint=False)
        if i == 0:
            # need to estimate an additional support point before the first and after the last
            y_intpol[:num_sampling] = np.array([
                catmull_rom_one_point(
                    x,
                    c1, # estimated start point,
                    p_y[0],
                    p_y[1],
                    c2) # estimated end point
                for x in np.linspace(0., 1. + hold_out, num_sampling, endpoint=False)])

    return x_intpol, y_intpol

def catmull_rom_2pt_in_3d(start, end, c1, c2, num_sampling):
    p_x = np.array([start[0], end[0]])
    p_y = np.array([start[1], end[1]])
    p_z = np.array([start[2], end[2]])
    x_intpol, y_intpol = catmull_rom_2pt(p_x, p_y, c1, c2, num_sampling)
    _, z_intpol= catmull_rom_2pt(p_x, p_z, c1, c2, num_sampling)
    
    return np.stack((x_intpol, y_intpol, z_intpol))

def catmull_rom(p_x, p_y, res):
    """Computes Catmull-Rom Spline for given support points and resolution.
    Args:
        p_x: array of x-coords
        p_y: array of y-coords
        res: resolution of a segment (including the start point, but not the
            endpoint of the segment)
    """
    # create arrays for spline points
    x_intpol = np.empty(res*(len(p_x)-1) + 1)
    y_intpol = np.empty(res*(len(p_x)-1) + 1)

    # set the last x- and y-coord, the others will be set in the loop
    x_intpol[-1] = p_x[-1]
    y_intpol[-1] = p_y[-1]

    # loop over segments (we have n-1 segments for n points)
    for i in range(len(p_x)-1):
        # set x-coords
        x_intpol[i*res:(i+1)*res] = np.linspace(
            p_x[i], p_x[i+1], res, endpoint=False)
        if i == 0:
            # need to estimate an additional support point before the first
            y_intpol[:res] = np.array([
                catmull_rom_one_point(
                    x,
                    p_y[0] - (p_y[1] - p_y[0]), # estimated start point,
                    p_y[0],
                    p_y[1],
                    p_y[2])
                for x in np.linspace(0.,1.,res, endpoint=False)])
        elif i == len(p_x) - 2:
            # need to estimate an additional support point after the last
            y_intpol[i*res:-1] = np.array([
                catmull_rom_one_point(
                    x,
                    p_y[i-1],
                    p_y[i],
                    p_y[i+1],
                    p_y[i+1] + (p_y[i+1] - p_y[i]) # estimated end point
                ) for x in np.linspace(0.,1.,res, endpoint=False)])
        else:
            y_intpol[i*res:(i+1)*res] = np.array([
                catmull_rom_one_point(
                    x,
                    p_y[i-1],
                    p_y[i],
                    p_y[i+1],
                    p_y[i+2]) for x in np.linspace(0.,1.,res, endpoint=False)])

    return (x_intpol, y_intpol)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # set the resolution (number of interpolated points between each pair of
    # points, including the start point, but excluding the endpoint of each
    # interval)
    res = 50

    # generate some random support points
    
    p_x = np.arange(-7,7, dtype='float32')
    p_y = np.zeros_like(p_x)
    for i in range(len(p_x)):
        p_y[i] = np.random.rand()*3. - 1.5
    """
    p_z = np.zeros_like(p_x)
    for i in range(len(p_x)):
        p_z[i] = np.random.rand()*3. - 1.5
    """
    """
    p_x=[-10.  -9.  -8.  -7.  -6.  -5.  -4.  -3.  -2.  -1.   0.   1.   2.   3.
   4.   5.   6.   7.   8.   9.  10.]
    p_y=[-1.3053329  -0.9646554   1.4163499  -0.45993748 -1.206672   -0.59014595
  1.4450525  -1.399725    0.01969506 -0.35387275  0.17669208  0.7681426
  0.3219197  -1.039825   -0.8629476  -0.8701243   0.13252383  0.6389541
  1.133622    0.29491705 -1.4766525 ]
    """
    """
    p_x=np.array([6,2])
    p_y=np.array([4,5])
    """
    # do the catmull-rom
    x_intpol, y_intpol = catmull_rom(p_x, p_y, res)
    """
    a, z_intpol= catmull_rom(p_x,p_z,res)
    """
    
    # fancy plotting
    plt.figure()
    plt.scatter(p_x, p_y)
    plt.plot(x_intpol, y_intpol)
    plt.show()
    print(p_x)
    print(p_y)
    

    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_intpol, y_intpol, z_intpol, 'gray')
    ax.scatter3D(p_x, p_y, p_z, c=p_z, cmap='Greens');
    plt.show()
    """
# vim: set ts=4 sw=4 sts=4 expandtab:

