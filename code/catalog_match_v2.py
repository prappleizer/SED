import numpy as np
import matplotlib.pyplot as plt
import os
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               Load data files and split into id, ra, dec etc arrays

old = np.genfromtxt('../input_files/Bins_v4.7.dat', skip_header=1)
old = np.transpose(old)

old_id = old[0]
old_ra = old[1]
old_dec = old[2]
old_iter = old[3]
old_sedtype = old[4]
old_scale = old[5]

new = np.genfromtxt('../input_files/cosmos-1.deblend.herschel.v1.0.cat', skip_header=31)
new = np.transpose(new)
new_id = new[0]
new_ra = new[1]
new_dec = new[2]

##########################################################################################
#                              Generate possible matches
 
def find_nearest3(array,value):
    idx = (np.abs(array-value)).argmin()
    idx2 = np.argsort(np.abs(array-value))[1]
    idx3 = np.argsort(np.abs(array-value))[2]
    return [idx, idx2, idx3],[array[idx],array[idx2],array[idx3]]

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
        id_matches, ra_matches = find_nearest3(new_ra,old_ra[i])
        ra_matchlist.append([old_id[i],[new_id[id_matches[0]],new_id[id_matches[1]],new_id[id_matches[2]]],old_ra[i],ra_matches])
    print 'Finished RA                    '
    current_num = 0
    for i in range(len(old_dec)):
        tot_num = len(old_dec)
        current_num +=1.
        outfrac = (current_num/tot_num)*100
        outfrac = str(outfrac)
        os.sys.stdout.write('%s Percent Complete \r' %(outfrac))
        os.sys.stdout.flush()
        id_matches, dec_matches = find_nearest3(new_dec,old_dec[i])
        dec_matchlist.append([old_id[i],[new_id[id_matches[0]],new_id[id_matches[1]],new_id[id_matches[2]]],old_dec[i],dec_matches])
    print 'Finished Dec                   '
    return ra_matchlist, dec_matchlist


ra_matchlist, dec_matchlist = new_matcher(old_ra,new_ra,old_dec,new_dec)

def new_crosscheck(ra_matchlist,dec_matchlist):
    final_match = []
    for i in range(len(ra_matchlist)):
        if ra_matchlist[i][1][0] == dec_matchlist[i][1][0]:
            final_match.append([ra_matchlist[i][0],ra_matchlist[i][1][0]])
        elif ra_matchlist[i][1][1] == dec_matchlist[i][1][1]:
            final_match.append([ra_matchlist[i][0],ra_matchlist[i][1][1]])
        elif ra_matchlist[i][1][2] == dec_matchlist[i][1][2]:
            final_match.append([ra_matchlist[i][0],ra_matchlist[i][1][2]])
    return final_match


final_match = new_crosscheck(ra_matchlist,dec_matchlist)
old_idss = [f[0] for f in final_match]
new_idss = [f[1] for f in final_match]
out = [old_idss,new_idss]
out = np.transpose(out)
np.savetxt('ID_conversion_v2.txt',out)

    