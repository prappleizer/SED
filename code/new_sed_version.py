##############################
import numpy as np 
import os
import sys
##############################
# A script for creating the empty directory structures 
# needed to make a new set of SEDs with their composite filter curves 
#############################
version_number = sys.argv[1] #specify the number of the version you are trying to make
#############################
# Make the v__ directory 
dirname = 'v' + str(version_number)
syscall = 'mkdir ../SEDs/' + dirname
os.system(syscall)
# Make the seds directory within the new v__ dir
syscall2 = 'mkdir ../SEDs/' + dirname + '/seds'
os.system(syscall2)
# Make the composite_filters directory within the new v__ dir
syscall3 = 'mkdir ../SEDs/' + dirname + '/composite_filters'
os.system(syscall3)
# Make the 32 iteration directories within the composite filters directory 
for i in range(1,33):
	sys_call = 'mkdir ' + '../SEDs/' + dirname + '/composite_filters/iteration_' + str(i)
	os.system(sys_call)



