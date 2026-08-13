import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main(iTEL,fbeams,tag):
    
    print("\n\n\n\n\nGENERATING RESULTS FOR ",tag,"\n\n")
    if iTEL==0:
        options,labels = read_options()
    
    ####### gets original data #######
    all_station_datas = read_keane()
    all_station_data = all_station_datas[iTEL]
    
    all_stations = all_station_data[0]
    radii = all_station_data[1]
    sens = all_station_data[2]
    FOV = all_station_data[3]
    
    
    # gets previous best radius (same for both configs)
    prev_bests = ["C224","SKA041","SKA041"] # For AA4
    
    sbest = prev_bests[iTEL]
    ibest = np.where(all_stations == sbest)[0]
    rmax = radii[ibest].values
    print("Orig max radius is ",rmax," with ",ibest+1,"stations")
    
    
    plt.figure()
    
    plt.plot([rmax,rmax],[0,512],color="black",linestyle=":")
    plt.text(rmax*1.1,350,"Pre-deferral\noptimum",rotation=90,fontsize=12)
    l1,=plt.plot(radii,np.arange(radii.size)+1,label="original AA4",color="black")
    
    # this step limits the size of the FOV to the HPBW
    eff_rad = np.copy(radii.values)
    toolow = np.where(eff_rad < rmax*fbeams**0.5)[0]
    eff_rad[toolow] =  rmax*fbeams**0.5
    
    FOM = eff_rad**-2 * (np.arange(radii.size)+1)**1.5
    old_max = FOM[ibest]
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    
    if iTEL > 0:
        imax = np.argmax(FOM[1:])+1
        rmax = radii[imax]
        newmax = fbeams * FOM[imax]
        print("We find a new peak maximum at r = ",rmax," using ",imax+1," antennas", "reduction of ",newmax/old_max)
        l2,=ax2.plot(radii,FOM/FOM[ibest]*fbeams,linestyle="--",label="Relative  (deferral)")
    
    
    
    
    #ax2.plot(radii,FOM,color="black",linestyle="--")
    
    # loop over all options
    # plots graph of radius vs antenna number for each options
    if iTEL==0:
        for i,option in enumerate(options):
            matches = identify_present(option,all_stations)
            nstations = len(matches)
            stations = np.arange(nstations)+1
            plot_r = np.array(radii[matches].values)
            ax1.plot(plot_r,stations,label=labels[i])
            
            FOM = stations**1.5 / plot_r**2
            FOM_max = np.max(FOM[1:])
            Nmax = np.argmax(FOM[1:])+1
            rmax = plot_r[Nmax]
            new_max = FOM_max * fbeams
            
            ax2.plot(plot_r[1:],fbeams*FOM[1:]/old_max,color=ax1.lines[-1].get_color(),linestyle="--")
            
            Nless = np.where(plot_r <= rmax)[0][-1]
            print("Options ",i," Number of stations included is ",Nless+1)
            print("New FOM is ",new_max/old_max," of old efficiency at rmax = ",rmax, Nmax+1)
    plt.sca(ax2)
    plt.ylabel("Fraction of AA4 FOM")
    plt.sca(ax1)
    plt.ylim(0,512)
    plt.xscale('log')
    
    if iTEL==0:
        plt.ylabel("Number of stations")
        plt.legend()
    else:
        plt.ylabel("Number of antennas")
        plt.legend(handles=[l1,l2],labels=["Array density","Relative FOM (deferral)"])
    plt.xlabel("Radius [km]")
    plt.tight_layout()
    plt.savefig(tag+"stations_vs_radius.png")
    plt.close()
    

def get_optimum():
    """
    Gets optimum trade-off
    """

def identify_present(subset,full_list):
    """
    identifies which subset is present, i.e., if all antennas are actually there
    """
    
    ns = len(subset)
    matches = np.zeros([ns],dtype='int')
    
    for i,station in enumerate(subset):
        
        match = np.where(full_list == station)[0]
        if len(match)==0:
            print("could not find  station ",station)
            continue
        matches[i] = match
    
    sort_matches = np.sort(matches)
    
    return sort_matches

def read_keane():
    """
    reads Evan's info
    """
    
    files = ["../inputs/LowAA4_ID_radius_AonT_FoVdeg2","../inputs/Band1AA4_ID_radius_AonT_FoVdeg2","../inputs/Band2AA4_ID_radius_AonT_FoVdeg2"]
    #files = ["../inputs/LowAAstar_ID_radius_AonT_FoVdeg2","../inputs/Band1AAstar_ID_radius_AonT_FoVdeg2","../inputs/Band2AAstar_ID_radius_AonT_FoVdeg2"]
    
    
    #data = np.loadtxt(f,dtype='string')
    datas=[]
    for f in files:
        data = pd.read_csv(f, sep='\s+', header=None)
        datas.append(data)
    return datas

def read_options():
    """
    reads in options
    """
    
    options = ["option_5.txt","option_5.2.txt","option_7.txt"]
    option_labels = ["option 5","option 5.2","option 7"]
    stations=[]
    for i,option in enumerate(options):
        with open(option, 'r') as f:
            for line in f:
                slist = line.split(',')
            slist[-1] = slist[-1][:-1] # to avoid last \n
            stations.append(slist)
    return stations,option_labels

fbeams = [50./250,200./1125,200./1125]

labels=["Low","Mid_band1","mid_band2"]
for iTEL in np.arange(3):
    main(iTEL,fbeams[iTEL],labels[iTEL])
    
