import numpy as np

def load(iteration):
    #Load filters
    str2 = str(iteration)
    str4 = '/Users/ipasha/Documents/Research/Seds/filters_spec'
    str5 = str2
    str6 = '.dat'
    filter_filn = str4 + str5 + str6
    sed_filter = np.genfromtxt(filter_filn)
    sed_filter = np.transpose(sed_filter)
    
    wavelengths = sed_filter[1]
    transmission = sed_filter[2]
    
    #Split up filter files
    split_index = []
    for i in range(len(wavelengths)):
        if np.isnan(wavelengths[i]) == True:
            split_index.append(i)
    split_wl = np.split(wavelengths, split_index)
    split_wl = np.delete(split_wl,0)
    split_transmission = np.split(transmission, split_index)
    split_transmission = np.delete(split_transmission,0)
    return split_wl, split_transmission
    
def maker(iteration):
    split_wl, split_tr = load(iteration)
    better_out1 = []
    better_out2 = []
    for i in range(len(split_wl)):
        thing = split_wl[i]
        thing2 = split_tr[i]
        thing = np.delete(thing,0)
        thing2 = np.delete(thing2,0)
        better_out1.append(thing)
        better_out2.append(thing2)
    for i in range(len(better_out2)):
        num = i+1001
        filename = '/Users/ipasha/RESEARCH/new_from_scratch_research/SEDs/composite_filters/iteration_' + str(iteration) + '/filter_' + str(num) + '.par'
        out_arr = np.column_stack((better_out1[i],better_out2[i]))
        print out_arr
        np.savetxt(filename,out_arr)
    #return better_out1, better_out2
    


          
    
    
 