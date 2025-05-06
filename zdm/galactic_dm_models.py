import numpy as np
import sys
from pkg_resources import resource_filename
#inputs are l, b, maximum angular separation allowed, tolerance of minimum angular separation
#in deg

def dmg_sanskriti2020(l_FRB, b_FRB, sep_th=5, sep_tol=5, verb=False):
    #Sightlines 
    l,b,DMavg,e_l,e_u,DMmax,esys = np.loadtxt(resource_filename('zdm','data/Misc/Sanskriti_DM_inputs.txt'),unpack=True)
    length = np.size(l)

    #deg to radian
    l0 = l_FRB*(np.pi/180.0)*np.ones(length)
    b0 = b_FRB*(np.pi/180.0)*np.ones(length)
    l1 = l*(np.pi/180.0)
    b1 = b*(np.pi/180.0)

    #angular separation
    sepx = np.cos(b1)*np.cos(l1) - np.cos(b0)*np.cos(l0)
    sepy = np.cos(b1)*np.sin(l1) - np.cos(b0)*np.sin(l0)
    sepz = np.sin(b1) - np.sin(b0)
    sep = np.sqrt(np.square(sepx) + np.square(sepy) + np.square(sepz))*(180.0/np.pi) #back in deg 

    # Take all sightlines within sep_th degrees
    cond = sep<=sep_th 

    dmclose = DMavg[cond]
    elclose = e_l[cond]
    euclose = e_u[cond]

    # If there are sightlines within the threshold, then use them
    if np.size(dmclose)>0:
        el_mean = np.sqrt(np.sum(np.square(elclose)))/np.size(dmclose)
        eu_mean = np.sqrt(np.sum(np.square(euclose)))/np.size(dmclose)
        mean = np.mean(dmclose)

        if verb:
            print ("separatation:",sep[cond],"deg")
            print ("l:",l[np.argwhere(cond==True)[:]].T)
            print ("b:",b[np.argwhere(cond==True)[:]].T)
            print ("Mean",np.mean(dmclose),"-",el_mean, "+",eu_mean,"cm^-3 pc")
            print ("Median:",np.median(dmclose), "-",np.median(dmclose)-np.median(dmclose-elclose),"+", np.median(dmclose+euclose)-np.median(dmclose),"cm^-3 pc")
    # If there are no sightlines within the threshold, use the closest sightline
    else:
        el_mean = e_l[np.argmin(sep)]
        eu_mean = e_u[np.argmin(sep)]
        mean = DMavg[np.argmin(sep)]

        if verb:
            print ("No sightline found within threshold of", sep_th, "degrees. Using nearest sightline.")
            print ("separatation:",np.min(sep) ,"deg","\nnearest sightline is at",l[np.argmin(sep)],",",b[np.argmin(sep)],"deg")
            print (mean, "-", el_mean, "+", eu_mean, "cm^-3 pc")

    # ####################################option I##################################################
    # print ("----------------------------------------------------------------------------------------------------")
    # print ("Option I")
    # print ("----------------------------------------------------------------------------------------------------")
    # #single sightline 
    # print ("separatation:",np.min(sep) ,"deg","\nnearest sightline is at",l[np.argmin(sep)],",",b[np.argmin(sep)],"deg")
    # print (DMavg[np.argmin(sep)],"-",e_l[np.argmin(sep)],"+",e_u[np.argmin(sep)],"cm^-3 pc")

    # ####################################option II##################################################
    # print ("----------------------------------------------------------------------------------------------------")
    # print ("Option II" )
    # print ("----------------------------------------------------------------------------------------------------")
    # #sightlines within threshold
    # cond = sep<=sep_th 

    # dmclose = DMavg[cond]
    # elclose = e_l[cond]
    # euclose = e_u[cond]

    # if np.size(dmclose)>0:
    #     el_mean = np.sqrt(np.sum(np.square(elclose)))/np.size(dmclose)
    #     eu_mean = np.sqrt(np.sum(np.square(euclose)))/np.size(dmclose)
    #     print ("separatation:",sep[cond],"deg")
    #     print ("l:",l[np.argwhere(cond==True)[:]].T)
    #     print ("b:",b[np.argwhere(cond==True)[:]].T)
    #     print ("Mean",np.mean(dmclose),"-",el_mean, "+",eu_mean,"cm^-3 pc")
    #     print ("Median:",np.median(dmclose), "-",np.median(dmclose)-np.median(dmclose-elclose),"+", np.median(dmclose+euclose)-np.median(dmclose),"cm^-3 pc")
    # else:
    #     print ("No sightline found within your threshold. Use the median of all-sky: 64 -20 +23 cm^-3 pc")
    # ####################################option III##################################################
    # print ("----------------------------------------------------------------------------------------------------")
    # print ("Option III")
    # print ("----------------------------------------------------------------------------------------------------")
    # #sightlines within tolerance of minimum 
    # cond =np.abs(sep-np.min(sep))<=sep_tol

    # dmclose = DMavg[cond]
    # elclose = e_l[cond]
    # euclose = e_u[cond]

    # if np.size(dmclose)>1:
    #         el_mean = np.sqrt(np.sum(np.square(elclose)))/np.size(dmclose)
    #         eu_mean = np.sqrt(np.sum(np.square(euclose)))/np.size(dmclose)
    #         print ("separatation:",sep[cond],"deg")
    #         print ("Mean",np.mean(dmclose),"-",el_mean, "+",eu_mean)
    #         print ("Median:",np.median(dmclose), "-",np.median(dmclose)-np.median(dmclose-elclose),"+", np.median(dmclose+euclose)-np.median(dmclose))
    # else:
    #         print ("No extra sightline found. Same result as option I")
    
    return mean, el_mean, eu_mean

