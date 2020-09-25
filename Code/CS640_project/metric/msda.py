import torch
from scipy.spatial import distance
import numpy as np


def euclidean(x1,x2):
    return ((x1-x2)**2).sum().sqrt()


def dist(x1,x2):
	#print(x1.shape)
	v1 = x1-x2
	v1,index = torch.sort(v1)
	v2 = v1[:16]
	#print(v2)
	return (v2**2).sum().sqrt()
	
def cube_dist(x1,x2):
	temp = math.abs((x1-x2)**3).sum()
	print(temp)
	return temp

"""
def mahal(x1,x2):
	with torch.no_grad():
		s1 = x1
		s2 = x2
		cv = torch.tensor(np.cov(s1.numpy(),s2.numpy())	
		loss = distance.mahalanobis(x1,x2,cv)
		return loss
	#return x1
"""

#def cov(x,y):
#	return np.cov(x.detach(),y.detach())





def k_moment(output_s1, output_s2, output_t, k):
        output_s1 = (output_s1**k).mean(0)
        output_s2 = (output_s2**k).mean(0)
	#output_s3 = (output_s3**k).mean(0)
        output_t = (output_t**k).mean(0)
        #return  mahal(output_s1, output_t) + mahal(output_s2, output_t) + mahal(output_s1, output_s2)
        #return  cube_dist(output_s1, output_t) + cube_dist(output_s2, output_t) + cube_dist(output_s1, output_s2)
        #return  euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s1, output_s2)
        return  dist(output_s1, output_t) + dist(output_s2, output_t) + dist(output_s1, output_s2)
        

	#return  euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t)+\
                #euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) +\
                #euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s3) + \
                #euclidean(output_s4, output_t)
               


def msda_regulizer(output_s1, output_s2, output_t, belta_moment):
	# print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))
	s1_mean = output_s1.mean(0)
	s2_mean = output_s2.mean(0)
	#print(s1_mean)
	#s3_mean = output_s3.mean(0)
	t_mean = output_t.mean(0)
	output_s1 = output_s1 - s1_mean
	output_s2 = output_s2 - s2_mean
	#output_s3 = output_s3 - s3_mean
	output_t = output_t - t_mean
	#print(output_t)
	#print(output_s1)
	#moment1 = mahal(output_s1, output_t) + mahal(output_s2, output_t) + mahal(output_s1, output_s2)
	
	#moment1 = cube_dist(output_s1, output_t) + cube_dist(output_s2, output_t) + cube_dist(output_s1, output_s2)
	#moment1 = euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s1, output_s2)
	moment1 = dist(output_s1, output_t) + dist(output_s2, output_t) + dist(output_s1, output_s2)

	#moment1 = euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + ieuclidean(output_s3, outpiut_t)+\
	#euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) +\
	#euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s3) + \
	#euclidean(output_s4, output_t)
	reg_info = moment1
	#print(reg_info)
	for i in range(belta_moment-1):
		#reg_info += k_moment(output_s1,output_s2,output_s3, output_s4, output_t,i+2)
		reg_info += k_moment(output_s1,output_s2, output_t,i+2)
	
	return reg_info/6
	#return euclideain(output_s1, output_t)
