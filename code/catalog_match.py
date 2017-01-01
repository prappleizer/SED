import numpy as np
import matplotlib.pyplot as plt
import os
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               Load data files and split into id, ra, dec etc arrays

old = np.genfromtxt('input_files/Bins_v4.7.dat', skip_header=1)
old = np.transpose(old)

old_id = old[0]
old_ra = old[1]
old_dec = old[2]
old_iter = old[3]
old_sedtype = old[4]
old_scale = old[5]

new = np.genfromtxt('input_files/cosmos-1.deblend.herschel.v1.0.cat', skip_header=31)
new = np.transpose(new)
new_id = new[0]
new_ra = new[1]
new_dec = new[2]

##########################################################################################
#                              Generate possible matches
 
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def new_matcher(old_ra,new_ra,old_dec,new_dec):
    current_num = 0
    ra_matchlist = []
    dec_matchlist = []
    for i in range(len(old_ra)):
        tot_num = len(old_ra)
        current_num +=1.
        outfrac = (current_num/tot_num)*100
        outfrac = str(outfrac)
        os.sys.stdout.write('%s Percent Complete \r' %(outfrac))
        os.sys.stdout.flush()
        id_match, ra_match = find_nearest(new_ra,old_ra[i])
        delta_RA = old_ra[i] - ra_match
        #if (delta_RA < 1E-5):
        ra_matchlist.append([old_id[i],new_id[id_match],old_ra[i],new_ra[id_match]])
    print 'Finished RA'
    current_num = 0
    for i in range(len(old_dec)):
        tot_num = len(old_dec)
        current_num +=1.
        outfrac = (current_num/tot_num)*100
        outfrac = str(outfrac)
        os.sys.stdout.write('%s Percent Complete \r' %(outfrac))
        os.sys.stdout.flush()
        id_match, dec_match = find_nearest(new_dec,old_dec[i])
        delta_dec = old_dec[i] - dec_match
        #if (delta_dec<2E-7):
        dec_matchlist.append([old_id[i],new_id[id_match],old_dec[i],new_dec[id_match]])
    print 'Finished Dec'
    return ra_matchlist, dec_matchlist


ra_matchlist, dec_matchlist = new_matcher(old_ra,new_ra,old_dec,new_dec)

def new_crosscheck(ra_matchlist,dec_matchlist):
    final_dec_match = []
    final_ra_match = []
    for i in range(len(ra_matchlist)):
        if ra_matchlist[i][1] == dec_matchlist[i][1]:
            final_ra_match.append(ra_matchlist[i])
            final_dec_match.append(dec_matchlist[i])
    return final_ra_match,final_dec_match


final_ra_match,final_dec_match = new_crosscheck(ra_matchlist,dec_matchlist)
old_idss = [f[0] for f in final_dec_match]
new_idss = [f[1] for f in final_dec_match]
out = [old_idss,new_idss]
out = np.transpose(out)
np.savetxt('ID_conversion.txt',out)

    