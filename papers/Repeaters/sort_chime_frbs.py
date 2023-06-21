"""
This script sorts CHIME FRBs into declination ranges,
    and also those with and without repetition

Run with:
    python3 sort_chime_frbs.py

Inputs:
    Will automatically read in CHIME FRB data contained in ./CHIME_FRBs

Outputs:
    Will write FRB data suitable for use in a survey file to ./CHIME_FRBs
    Will generate one file for each declination range

"""

import numpy as np

def main(Nbounds=6):
    
    # defines set of bounds to read in
    
    bdir = 'Nbounds'+str(Nbounds)+'/'
    
    
    ####### loads CHIME FRBs ######
    
    chimedir = 'CHIME_FRBs/'
    infile = chimedir+'chimefrbcat1.csv'
    #idec=6
    #idm=18
    #idmeg=26
    #iname=0
    #irep=2
    #iwidth=42
    #isnr=17
    
    idec=5
    idm=29
    idmeg=9
    iname=0
    irep=2
    iwidth=32
    isnr=10
    
    NFRB=536
    decs=np.zeros([NFRB])
    dms=np.zeros([NFRB])
    dmegs=np.zeros([NFRB])
    dmgs=np.zeros([NFRB])
    snrs=np.zeros([NFRB])
    widths=np.zeros([NFRB])
    names=[]
    reps=np.zeros([NFRB])
    
    # holds repeater info
    rnames=[]
    ireps=[]
    nreps=[]
    badcount=0
    with open(infile) as f:
        lines = f.readlines()
        count=-1
        for i,line in enumerate(lines):
            if count==-1:
                columns=line.split(',')
                for ic,w in enumerate(columns):
                    print(ic,w)
                count += 1
                continue
            words=line.split(',')
            # seems to indicate new bursts have been added
            #if words[5][:2]=="RA":
            #    badcount += 1
                #print("BAD : ",badcount)
                #continue
            decs[i-1]=float(words[idec])
            dms[i-1]=float(words[idm])
            dmegs[i-1]=float(words[idmeg])
            names.append(words[iname])
            snrs[i-1]=float(words[isnr])
            # guards against upper limits
            if words[iwidth] == '':
                widths[i-1]=0.
            elif words[iwidth][0]=='<':
                widths[i-1]=0.
            else:
                widths[i-1]=float(words[iwidth])*1e3 #in ms
            
            #print(i,decs[i-1],dms[i-1],dmegs[i-1],names[i-1],snrs[i-1],widths[i-1])
            dmgs[i-1] = dms[i-1]-dmegs[i-1]
            rep=words[irep]
            print(i,rep)
            
            if rep=='-9999':
                reps[i-1]=0
            elif rep=='':
                reps[i-1]=0
            else:
                reps[i-1]=1
                if rep in rnames:
                    ir = rnames.index(rep)
                    nreps[ir] += 1
                else:
                    rnames.append(rep)
                    ireps.append(i-1)
                    nreps.append(1)
            count += 1
    
    print("Total of ",len(rnames)," repeating FRBs found")
    print("Total of ",len(np.where(reps==0)[0])," once-off FRBs")
    
    #print(rnames)
    #print(nreps)
    # now breaks this up into declination bins
    #The below is hard-coded and copied from "plot
    bdir = "Nbounds"+str(Nbounds)+"/"
    bfile = bdir + "bounds.npy"
    bounds=np.load(bfile)
    
    lowers=bounds[:-1]
    uppers=bounds[1:]
    for i,lb in enumerate(lowers):
        OK1=np.where(decs > lb)[0]
        OK2=np.where(decs < uppers[i])[0]
        OK=np.intersect1d(OK1,OK2)
        OK3 = np.where(reps==0)
        nOK = np.intersect1d(OK,OK3)
        rOK = np.intersect1d(OK,ireps)
        
        print("Found ",len(rOK),len(nOK)," FRBs which do (not) repeat in dec range ",lb,uppers[i])
        
        
        print("KEY  Xname         DM     DMG     SNR     WIDTH  NREP")
        for j in nOK:
            string='FRB  {0:} {1:6.1f}  {2:5.1f}  {3:5.1f} {4:8.3f}  1 '.\
                format(names[j],dms[j],dmgs[j],snrs[j],widths[j])
            print(string)
        #### searches for repeaters ####
        
        for j in rOK:
            rindex = np.where(ireps == j)[0][0]
            string='FRB  {0:} {1:6.1f}  {2:5.1f}  {3:5.1f} {4:8.3f}  {5:} '.\
                format(names[j],dms[j],dmgs[j],snrs[j],widths[j],nreps[rindex])
            print(string)
        
main(Nbounds=30)
