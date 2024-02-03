import numpy as np
import matplotlib.pyplot as plt 
# shape_data = np.array([0.0, 0., 0.005624999999999991, 0.014378906250000045, 
#         0.024486547851562568, 0.03474537368774422, 0.0443933953943253, 
#         0.0529963630275786, 0.060354717840215955, 0.06642930274659942, 
#         0.07128390374926041, 0.07504226099316191, 0.0778570788273204, 
#         0.07988866580429121, 0.08129106189085289, 0.08220379893995788, 
#         0.08274774841852639, 0.08302380913885798, 0.0790451566321223, 
#         0.07260624098380641, 0.06495160665441868, 0.05691987986583036, 
#         0.04905405662967122, 0.041685214275588134, 0.03499542709050685, 
#         0.02906453874530568, 0.02390450659228971, 0.019484261167040162, 
#         0.015747394717907204, 0.012624483451303736, 0.010041439737111357, 
#         0.007924965428885544, 0.006205920714800195, 0.004821221732756786, 
#         0.0037147237823418333, 0.002837426374215357, 0.0021472441754255556, 
#         0.0016085180924106934, 0.0011913883859058227, 0.000871112900045046, 
#         0.0006273850763798272, 0.00044368593276999935])
# shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])

# print(shape_vel)

ss = []
shape_vel2 = np.load("/home/kovan3/ur5_ws/src/breathe_data_1.npy")
shape_integral = []
print(shape_vel2.shape[0])
for i in range(shape_vel2.shape[0]):
    shape_integral.append(shape_vel2[:i].sum())
shape_integral_w = np.concatenate((shape_integral, shape_integral))
shape_integral_w = np.concatenate((shape_integral_w, shape_integral_w))

for i in range(int(shape_integral_w.shape[0]  / 4)  ):
    ss.append(shape_integral_w[4*i])
# # print(shape_vel2.shape)
# # a = shape_vel2[110,2] - shape_vel2[0,2]
# # b = max(shape_vel2[:,2]) - shape_vel2[0,2]
# # print(b/a)

plt.plot(ss, linewidth=4) 
plt.savefig('/home/kovan3/ur5_ws/src/high_freq.png', transparent=True)
plt.show()