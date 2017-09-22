import os
import sys


for i in range(5,33):
	old_loc = '/Users/ipasha/RESEARCH/CSED_2017/prospector/results/iteration_' + str(i) + '/6400i_510w_iter*.pdf'
	new_loc = '/Users/ipasha/RESEARCH/CSED_2017/plots/SEDs/iteration_' + str(i) + '/'
	sys_call = 'cp ' + old_loc + ' ' + new_loc
	os.system(sys_call)