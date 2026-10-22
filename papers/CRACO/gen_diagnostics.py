"""
This file prints average weighting factors for CRACO observations

It also plots a whole bunch of diagnostic plots
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from zdm import misc_functions as mf
import matplotlib
import os

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

# scaling parameter for sensitivity

def main():
    """
    main file, to iterate over different CRACO surveys
    """
    metrics=[]
    metrics2=[]
    # we do not bother with the 1.7ms sampling, it's negligible (<1 hr)
    itsamps = np.array([2,4,8,16,64])
    tsamps = itsamps*1.728
    
    names=[]
    for tsamp in tsamps:
        name = "{0:4.1f}".format(tsamp)+" ms"
        names.append(name)
    
    
    frblog = "Logs/CRACO_FRB_zdm.derived.csv"
    frbs = pd.read_csv(frblog)
    
    for i,itsamp in enumerate(itsamps):
        prefix="itsamp_"+str(itsamp)
        logfile="Logs/"+prefix+".csv"
        m,tempdf = print_metrics(logfile,prefix,frbs,alpha=1.5)
        m2,df2 = print_metrics(logfile,prefix,frbs,alpha=1.0,bkey="a1bfactors")
        metrics.append(m)
        metrics2.append(m2)
        
        if i==0:
            df = tempdf
        else:
            df = pd.concat([df,tempdf],ignore_index=True)
        
        surveyfile="Surveys/CRACO_900_itsamp_"+str(itsamp)+".ecsv"
        write_craco_survey(surveyfile,m["LOW"]["NFRB"],m["LOW"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
                    "900","itsamp_"+str(itsamp),m["LOW"]["nubar"],m["LOW"]["matches"],frbs,itsamp)
        
        
        surveyfile="Surveys/CRACO_900_itsamp_"+str(itsamp)+"_nodm.ecsv"
        write_craco_survey(surveyfile,m["LOW"]["NFRB"],m["LOW"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
                    "900","itsamp_"+str(itsamp),m["LOW"]["nubar"],m["LOW"]["matches"],frbs,itsamp,False)
                    
                    
        surveyfile="Surveys/CRACO_900_itsamp_"+str(itsamp)+"_icsw.ecsv"
        write_craco_survey(surveyfile,m["LOW"]["NFRB"],m["LOW"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
                    "900","itsamp_"+str(itsamp),m["LOW"]["nubar"],m["LOW"]["matches"],frbs,itsamp,icsw=True)
        
        
        #surveyfile="Surveys/a1CRACO_900_itsamp_"+str(itsamp)+".ecsv"
        #write_craco_survey(surveyfile,m2["LOW"]["NFRB"],m2["LOW"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
        #            "900","itsamp_"+str(itsamp),m2["LOW"]["nubar"],m2["LOW"]["matches"],frbs,itsamp)
        
        
        surveyfile="Surveys/CRACO_1300_itsamp_"+str(itsamp)+".ecsv"
        write_craco_survey(surveyfile,m["HIGH"]["NFRB"],m["HIGH"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
                    "1300","itsamp_"+str(itsamp),m["HIGH"]["nubar"],m["HIGH"]["matches"],frbs,itsamp)
        
        
        surveyfile="Surveys/CRACO_1300_itsamp_"+str(itsamp)+"_nodm.ecsv"
        write_craco_survey(surveyfile,m["HIGH"]["NFRB"],m["HIGH"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
                    "1300","itsamp_"+str(itsamp),m["HIGH"]["nubar"],m["HIGH"]["matches"],frbs,itsamp,False)
        
        
        surveyfile="Surveys/CRACO_1300_itsamp_"+str(itsamp)+"_icsw.ecsv"
        write_craco_survey(surveyfile,m["HIGH"]["NFRB"],m["HIGH"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
                    "1300","itsamp_"+str(itsamp),m["HIGH"]["nubar"],m["HIGH"]["matches"],frbs,itsamp,icsw=True)
        
        #surveyfile="Surveys/a1CRACO_1300_itsamp_"+str(itsamp)+".ecsv"
        #write_craco_survey(surveyfile,m2["HIGH"]["NFRB"],m2["HIGH"]["zdmTeff"],"{0:4.1f}".format(tsamps[i]),\
        #            "1300","itsamp_"+str(itsamp),m2["HIGH"]["nubar"],m2["HIGH"]["matches"],frbs,itsamp)
        surveyfile="Surveys/a1CRACO_1300_itsamp_"+str(itsamp)+".ecsv"
    
    
    # does cumulative plot for EVERYTHING
    logfile=None
    prefix=""
    df=df.sort_values("sbid",ignore_index=True)
    
    print_metrics(logfile,"all_",frbs,df=df)
    
    
    if os.path.exists("1300_nodm.npy"):
        nodm_1300 = np.load("1300_nodm.npy")
    else:
        nodm_1300 = np.full([itsamps.size],1.)
    if os.path.exists("900_nodm.npy"):
        nodm_900 = np.load("900_nodm.npy")
    else:
        nodm_900 = np.full([itsamps.size],1.)  
    
    if os.path.exists("1300_icsw.npy"):
        icsw_1300 = np.load("1300_icsw.npy")
    else:
        icsw_1300 = np.full([itsamps.size],1.)
    if os.path.exists("900_icsw.npy"):
        icsw_900 = np.load("900_icsw.npy")
    else:
        icsw_900 = np.full([itsamps.size],1.)  
        
    
    print_latex_table(metrics,metrics2,names,nodm_900,nodm_1300,icsw_900,icsw_1300)
    
    # plots example of sampling time effet
    plot_tsamp()



def write_craco_survey(surveyfile,NFRB, Teff, tres,sfreq, tag,nubar,ifrbs,frbs,itsamp,domask=True,icsw=False):
    """
    returns CRACO prefixes and suffixes
    
    NFRB (int):
        number of FRBs
    Teff (float):
        effective observation time [days]
    tres (float):
        time resolution (ms)
    sfreq (string):
        900 or 1300
    tag (string):
        tag for identifying survey name, beam info, mask etc
    ifrbs (array of ints):
        indices of frbs matching this survey
    frbs (pandas dataframe):
        dataframe containing frb info
    """
    
    part1="""# %ECSV 1.0
# ---
# datatype:
# - {name: TNS, datatype: string}
# - {name: DM, datatype: float64}
# - {name: DMG, datatype: float64}
# - {name: SNR, datatype: float64}
# - {name: WIDTH, datatype: float64}
# - {name: RA, datatype: string}
# - {name: DEC, datatype: string}
# - {name: Gl, unit: deg, datatype: float64}
# - {name: Gb, unit: deg, datatype: float64}
# - {name: Z, datatype: float64}
# meta: !!omap
# - {survey_data: '{"observing": {"NORM_FRB": """
    
    string = part1+str(NFRB)
    
    part2 = """, "TOBS": """
    string = string+part2+"{0:5.2f}".format(Teff/3600/24.)
    
    if icsw:
        part3=""", "MAX_IW": 12, "MAXWMETH": 1},
#                   "telescope": {"BW": 288.0, "SNRTHRESH": 9.0, "BEAM": """
    else:
        part3=""", "MAX_IW": 8, "MAXWMETH": 2},
#                   "telescope": {"BW": 288.0, "SNRTHRESH": 9.0, "BEAM": """
    string= string +part3 + '''"'''+str(itsamp)+"_CRACO_"+sfreq
    
    if domask:
        part4="""", "DMMASK": """
        string = string + part4 + '''"'''+tag+"_craco_"+sfreq+"_mask.npy"
    
    part5='''",
#                                 "DIAM": 12.0, "NBEAMS": 1, "NBINS": 5, "FBAR": '''
    string = string+part5+"{0:5.2f}".format(nubar)
    part6=''',\n#                                  "TRES": '''
    if icsw:
        string = string +part6 + "1.28"
    else:
        string = string +part6 + tres
    part7=''', "FRES": 1.0, "THRESH": 1.01}}'}\n'''
    string = string+part7
    
    string = string+"# schema: astropy-2.0\n"
    string2="TNS        DM     DMG   SNR  WIDTH RA            DEC         Gl          Gb        Z\n"
    
    string3= ""  
    for i,ifrb in enumerate(ifrbs): 
        if frbs["FRBTNS"][ifrb] == "-":
            frbs.loc[ifrb,"FRBTNS"] = "XXXXXXXXX"
        if frbs["z"][ifrb] == "-" or frbs["z"][ifrb] == "?":
            frbs.loc[ifrb,"z"] = "-1"
        string3=string3+str(frbs["FRBTNS"][ifrb])+" "+"{0:6.1f}".format(frbs["DM_craco"][ifrb])+" "\
                    + "{0:5.1f}".format(frbs["DM_NE2001"][ifrb])+" "\
                    + "{0:5.1f}".format(frbs["SNR_craco"][ifrb])+" "\
                    + "{0:5.3f}".format(frbs["width_craco"][ifrb])+" "\
                    + frbs["RA"][ifrb]+" "\
                    + frbs["Dec"][ifrb]+" "\
                    + "{0:10.6f}".format(frbs["Gl"][ifrb])+" "\
                    + "{0:10.6f}".format(frbs["Gb"][ifrb])+" "\
                    +  frbs["z"][ifrb] + "\n"
        
    
    with open(surveyfile, "w", encoding="utf-8") as s:
        s.write(string)
        s.write(string2)
        s.write(string3)
    
        

def print_latex_table(metrics,metrics2,names,nodm_900,nodm_1300,icsw_900,icsw_1300):
    """
    prints a latex-style table to copy-paste into Overleaf
    
    metrics:
        survey metrics, assuming standard value of alpha=1.5
    
    metrics2:
        survey metrics dict, assuming alpha=1.0 (or some alternative)
        
    names:
        names of surveys. For printing
    """
    #print(metrics[0]["HIGH"].keys())
    
    ############# v1: sidays table ############
    
    print("\n\n\n\n Please copy and paste into latex")
    
    nustrings="$\\overline{\\Delta \\nu}$ [MHz]  "
    bwstrings="$\\overline{\\nu}$ [MHz]  "
    gfstrings="$\\eta_{\\rm RFI}$  "
    Antstrings="$N_{\\rm ant}$  "
    bfstrings="$\\Omega_{\\rm eff}$ [deg$^2$]  "
    tstrings="$T_{\\rm obs}$ , $T_{\\rm eff}$ [hr]  "
    
    frbstrings="$N_{\\rm FRB}$"
    ratestrings="hr / FRB"
    
    for i,m in enumerate(metrics):
        
        nustring='& \\multicolumn{{2}}{{|c}}{{ {0:4.1f} }} & \\multicolumn{{2}}{{c}}{{ {1:4.1f} }}  '.format(m["LOW"]["nubar"],m["HIGH"]["nubar"])
        #print("nu string is ",nustring)
        nustrings = nustrings+nustring
        
        # RFI
        gfstring='& {0:4.2f} & {1:4.2f}  & {2:4.2f} & {3:4.2f} '.format(m["LOW"]["gf"],\
                    m["LOW"]["gf_eff"],m["HIGH"]["gf"],m["HIGH"]["gf_eff"])
        #print("gf string ",gfstring)
        gfstrings = gfstrings + gfstring
        
        
        # bw
        bwstring='& {0:4.0f} & {1:3.2f}  & {2:4.0f} & {3:3.2f} '.format(m["LOW"]["bw"],\
                    m["LOW"]["bw_eff"],m["HIGH"]["bw"],m["HIGH"]["bw_eff"])
        #print("bw string ",bwstring)
        bwstrings = bwstrings + bwstring
        
        # Nant
        Antstring='& {0:4.1f} & {1:3.2f}  & {2:4.1f} & {3:3.2f} '.format(m["LOW"]["Nant"],\
                    m["LOW"]["Nant_eff"],m["HIGH"]["Nant"],m["HIGH"]["Nant_eff"])
        #print("Nant string ",Antstring)
        Antstrings = Antstrings + Antstring
        
        bfstring='& {0:4.1f} & {1:3.2f} & {2:4.1f} & {3:3.2f}  '.format(m["LOW"]["BF"],\
                m["LOW"]["BF_eff"],m["HIGH"]["BF"],m["HIGH"]["BF_eff"])
        #print("Beam string ",bfstring)
        bfstrings = bfstrings + bfstring
        
        # total time
        tstring='& {0:5.0f} & {1:5.0f} & {2:5.0f} & {3:5.0f}  '.format(m["LOW"]["Ttot"]/3600,\
                    m["LOW"]["Teff"]/3600,m["HIGH"]["Ttot"]/3600,m["HIGH"]["Teff"]/3600)
        #print("tstring ",tstring)
        tstrings = tstrings + tstring
        
        frbstring='& \\multicolumn{{2}}{{|c}}{{ {0:3.0f} }} & \\multicolumn{{2}}{{c}}{{ {1:3.0f} }}  '.format(m["LOW"]["NFRB"],m["HIGH"]["NFRB"])
        frbstrings = frbstrings+frbstring
        
        if m["LOW"]["NFRB"] > 0:
            Lrate1 = "{0:3.0f}".format(m["LOW"]["Ttot"]/3600/m["LOW"]["NFRB"])
            Lrate2 = "{0:3.0f}".format(m["LOW"]["Teff"]/3600/m["LOW"]["NFRB"])
        else:
            Lrate1 = "$>${0:3.0f}".format(m["LOW"]["Ttot"]/3600)
            Lrate2 = "$>${0:3.0f}".format(m["LOW"]["Teff"]/3600)
        
        if m["HIGH"]["NFRB"] > 0:
            Hrate1 = "{0:3.0f}".format(m["HIGH"]["Ttot"]/3600/m["HIGH"]["NFRB"])
            Hrate2 = "{0:3.0f}".format(m["HIGH"]["Teff"]/3600/m["HIGH"]["NFRB"])
        else:
            Hrate1 = "$>${0:3.0f}".format(m["HIGH"]["Ttot"]/3600)
            Hrate2 = "$>${0:3.0f}".format(m["HIGH"]["Teff"]/3600)
        
        ratestring = "& "+Lrate1+" & " + Lrate2 + "& "+Hrate1+" & " + Hrate2
        ratestrings = ratestrings+ratestring
        #frbstring='& {0:3.0f} & '.format(m["LOW"]["NFRB"])+Lrate+ '& {0:3.0f} & '.format(m["HIGH"]["NFRB"]) + Hrate
        
    print(frbstrings," \\\\")
    print(nustrings," \\\\")
    print(bwstrings," \\\\")
    print(Antstrings," \\\\")
    print(gfstrings," \\\\")
    print(bfstrings," \\\\")
    print(tstrings," \\\\")
    print(ratestrings," \\\\")
    
    ############# v2: vertical table ############
    
    print("\n\n\n\n Please copy and paste into latex")
    
    nustrings="Band & $\\overline{\\nu}$ "
    bwstrings="$\\overline{\\Delta \\nu}$ & $\\epsilon_\\nu$"
    gfstrings="$f_g$ & $\\epsilon_g$"
    Antstrings="$N_{\\rm ant}$ & $\\epsilon_N$  "
    bfstrings="$\\Omega_{\\rm eff}$ & $\\epsilon_B$"
    
    dmstrings="$\\epsilon_{\\rm DM}$"
    wstrings="$\\epsilon_{\\rm w}$"
    
    efftotstring="$\\epsilon_{\\rm tot}$"
    tstrings="$T_{\\rm obs}$ & $T_{\\rm eff}$"
    
    frbstrings="$N_{\\rm FRB}$"
    ratestrings="hr / FRB"
    
    print("\n\n\n\n")
    # table header
    print("$t_{\\rm int}$ & " + nustrings+ " & " + Antstrings +" & " + bwstrings + " & " + gfstrings + \
            " & " + bfstrings + " & "+ dmstrings + " & "+ wstrings + " & " + efftotstring + " & " +\
            tstrings + " & $N_{\\rm FRB}$ \\\\")
    # units
    print("   & MHz &  MHz & & & MHz   &  & & & deg$^2$ & & &  & & hr & hr&  \\\\")
    
    for i,m in enumerate(metrics):
        
        # initialise for 900 MHz
        print("\\hline")
        string = names[i] + "& 900 " 
        for index in ["LOW","HIGH"]:
            # first, does 900 MHz
            
            string = string + '& {0:4.1f}  '.format(m[index]["nubar"])
            
            # Nant
            string = string +  '& {0:4.1f} & {1:3.2f}'.format(m[index]["Nant"],m[index]["Nant_eff"])
            
            
            # bw
            string = string +  '& {0:4.0f} & {1:3.2f} '.format(m[index]["bw"],m[index]["bw_eff"])
            
            # RFI
            string = string + '& {0:4.2f} & {1:4.2f}  '.format(m[index]["gf"],m[index]["gf_eff"])
            
            # beam
            string = string + '& {0:4.1f} & {1:3.2f} '.format(m[index]["BF"],m[index]["BF_eff"])
            
            if index == "LOW":
                m[index]["Teff"] *= nodm_900[i] * icsw_900[i]
                string = string + '& {0:4.2f}'.format(nodm_900[i])
                string = string + '& {0:4.2f}'.format(icsw_900[i])
            else:
                m[index]["Teff"] *= nodm_1300[i] * icsw_1300[i]
                string = string + '& {0:4.2f}'.format(nodm_1300[i])
                string = string + '& {0:4.2f}'.format(icsw_1300[i])
            
            eff = m[index]["Teff"]/m[index]["Ttot"]
            string = string + '& {0:5.2f}'.format(eff)
            
            # total time
            string = string + '& {0:5.0f} & {1:5.0f} '.format(m[index]["Ttot"]/3600,m[index]["Teff"]/3600)
            
            string = string +'& {0:3.0f} '.format(m[index]["NFRB"])
            string = string + "\\\\"
            print(string)
            string = "        & 1300 " 
        
        # don't print this currently, left in here just in case
        if m["LOW"]["NFRB"] > 0:
            Lrate1 = "{0:3.0f}".format(m["LOW"]["Ttot"]/3600/m["LOW"]["NFRB"])
            Lrate2 = "{0:3.0f}".format(m["LOW"]["Teff"]/3600/m["LOW"]["NFRB"])
        else:
            Lrate1 = "$>${0:3.0f}".format(m["LOW"]["Ttot"]/3600)
            Lrate2 = "$>${0:3.0f}".format(m["LOW"]["Teff"]/3600)
        
        if m["HIGH"]["NFRB"] > 0:
            Hrate1 = "{0:3.0f}".format(m["HIGH"]["Ttot"]/3600/m["HIGH"]["NFRB"])
            Hrate2 = "{0:3.0f}".format(m["HIGH"]["Teff"]/3600/m["HIGH"]["NFRB"])
        else:
            Hrate1 = "$>${0:3.0f}".format(m["HIGH"]["Ttot"]/3600)
            Hrate2 = "$>${0:3.0f}".format(m["HIGH"]["Teff"]/3600)
    
    
    ################ Observation time table ##############
    

    print("\n\n\n\n")
    print("Time-only table")
    TLOW=0.
    TeffLOW=0.
    THIGH=0.
    TeffHIGH=0.
    NLOW=0
    NHIGH=0
    
    for i,m in enumerate(metrics):
        # total time
        m2=metrics2[i]
        
        tstring1='& {0:5.0f} & {1:5.0f}  '.format(m["LOW"]["Ttot"]/3600,m["LOW"]["Teff"]/3600)
        #tstring1='& {0:5.0f} & {1:5.0f} & {2:5.0f} '.format(m["LOW"]["Ttot"]/3600,m["LOW"]["Teff"]/3600,m2["LOW"]["Teff"]/3600)
        print(names[i],"& 900 MHz ",tstring1," & ",m["LOW"]["NFRB"],"    &    &    \\\\")
        
        TLOW += m["LOW"]["Ttot"]/3600
        THIGH += m["HIGH"]["Ttot"]/3600
        TeffLOW += m["LOW"]["Teff"]/3600
        TeffHIGH += m["HIGH"]["Teff"]/3600
        NLOW += m["LOW"]["NFRB"]
        NHIGH += m["HIGH"]["NFRB"]
        
        # total time
        tstring2='& {0:5.0f} & {1:5.0f}  '.format(m["HIGH"]["Ttot"]/3600,m["HIGH"]["Teff"]/3600)
        #tstring2='& {0:5.0f} & {1:5.0f}  & {2:5.0f} '.format(m["HIGH"]["Ttot"]/3600,m["HIGH"]["Teff"]/3600,m2["HIGH"]["Teff"]/3600)
        print("       & 1300 MHz ",tstring2," & ",m["HIGH"]["NFRB"],"    &    &    \\\\")
        
   
    string1='{0:5.0f} & {1:5.0f} '.format(TLOW,TeffLOW)
    print("Total & 900 MHz    & "+string1+" & " +str(NLOW) + " & \\\\")
    string1='{0:5.0f} & {1:5.0f} '.format(THIGH,TeffHIGH)
    print("      & 1300 MHz    & "+string1+" & " +str(NHIGH) + " & \\\\")
    string1='{0:5.0f} & {1:5.0f} '.format(TLOW+THIGH,TeffLOW+TeffHIGH)
    print("      & Combined & "+string1+" & " +str(NHIGH+NLOW) + " & \\\\")
   
def print_metrics(logfile,prefix,frbs,alpha=1.5,bkey="bfactors",df=None):
    """
    Prints metrics for given logfile.
    
    Args:
        logfile [string]: name of observation logfile.
        prefix [string]: prefix to append to outouts
        frblog [string or None]: if present, generate a cumulative plot
                                of FRB detection rates vs observation time
    """
    
    # if data frame passed directly, do not read log file
    if df is None:
        df = pd.read_csv(logfile)
        pmv=True
    else:
        pmv=True
    
    LOW = np.where(df["fbar"] < 1000)[0]
    HIGH = np.where(df["fbar"] > 1000)[0]
    
    # prints some characteristic values
    print("##Doing mean values for ",prefix,"######")
    if pmv:
        metrics = print_mean_values(df,LOW,HIGH,savename=logfile,alpha=alpha,bkey=bkey)
    else:
        print(df.columns())
        exit()
    # produces some basic plots
    do_basic_plots(df,LOW,HIGH,prefix)
    
    if True:
        # extracts the frbs in frblog corresponding to sbids in this time integration
        sbids = np.intersect1d(frbs["SBID"],df["sbid"])
        matches=[]
        for sbid in sbids:
            imatch = np.where(sbid == frbs["SBID"])[0][0]
            matches.append(imatch)
        matches=np.array(matches)
        
        # extracts the frbs in frblog corresponding to sbids in this time integration
        sbids = np.intersect1d(frbs["SBID"],df["sbid"][LOW])
        Lmatches=[]
        for sbid in sbids:
            imatch = np.where(sbid == frbs["SBID"])[0][0]
            Lmatches.append(imatch)
        Lmatches=np.array(Lmatches)
        
        # extracts the frbs in frblog corresponding to sbids in this time integration
        sbids = np.intersect1d(frbs["SBID"],df["sbid"][HIGH])
        Hmatches=[]
        for sbid in sbids:
            imatch = np.where(sbid == frbs["SBID"])[0][0]
            Hmatches.append(imatch)
        Hmatches=np.array(Hmatches)
        
        if len(matches) > 0:
            thesefrbs=frbs.iloc[matches]
                
            # produces plot of cumulative effective and normal time vs detected FRBs
            ks_stat, ks_pval = plot_cumulative(df,LOW,HIGH,thesefrbs,ks=True,prefix=prefix)
            
            metrics["ks_stat"] = ks_stat
            metrics["ks_pval"] = ks_pval
        
        NFRB = len(matches)
        metrics["NFRB"] = NFRB
        
        LNFRB = len(Lmatches)
        metrics["LOW"]["NFRB"] = LNFRB
        
        HNFRB = len(Hmatches)
        metrics["HIGH"]["NFRB"] = HNFRB
        
        metrics["matches"] = matches
        metrics["HIGH"]["matches"] = Hmatches
        metrics["LOW"]["matches"] = Lmatches
        
        # load_frbs (but why????)
        #match_values(df,frbs)
    
    return metrics,df

def match_values(df,frbs):
    """
    For each frb, get slices corresponding to which observation they were found in
    
    Args:
        df: pandas dataframe containing logfile info
        frbs: pandas dataframe containing observed FRB info
    """
    scans = frbs["scan"]
    for i,scan in enumerate(scans):
        j = np.where(df["scan"] == scan)[0]
        fbar = df["fbar"][j].to_string(index=False, header=False)
        nchan = df["nchans"][j].to_string(index=False, header=False)
        print(i,fbar,nchan)
    
    
    
def print_mean_values(df,LOW,HIGH,savename=None,alpha=1.5,bkey="bfactors"):
    """
    Prints mean values of various quantities
    
    args:
        df: pandas dataframe containing logfile info
        LOW [list]: indices of data frame corresponding to 900 MHz sample
        HIGH [list]: indices of data frame corresponding to 1300 MHz sample
        savename: if not None, save dataframe with this name after adding efficiency
        alpha: scaling of rate with sensitivity (1.5 is Euclidean)
        bkey: key for beamfactors
        
    if LOW or HIGH is empty, we will get NANs, but that's OK - it doesn't crash!
    
    """
    metrics={}
    metrics["LOW"] = {}
    metrics["HIGH"] = {}
    
    ####### Total time ########
    
    Ttot = np.sum(df["tobs"])
    LTtot = np.sum(df["tobs"][LOW])
    HTtot = np.sum(df["tobs"][HIGH])
    
    metrics["Ttot"] = Ttot
    metrics["LOW"]["Ttot"] = LTtot
    metrics["HIGH"]["Ttot"] = HTtot
    
    ####### Beam Factors ########
    
    #units: eff deg^2
    # relative to closepack36 at 1.295 GHz at 0.9 deg (Fly's Eye Survey)
    if alpha == 1.5:
        BFNORM = 18.54742 # deg^2
    elif alpha==1.0:
        BFNORM = 17.90794
    else:
        print("no data for this alpha")
        exit()
    
    BF = np.sum(df["tobs"]*df[bkey])/Ttot
    LBF = np.sum(df["tobs"][LOW]*df[bkey][LOW])/LTtot
    HBF = np.sum(df["tobs"][HIGH]*df[bkey][HIGH])/HTtot
    
    # BFNORM already includes a weighting by Tsys and B^1.5, hence no further factor is applied
    # to the efficiency
    BF_eff = np.sum(df["tobs"]*df[bkey])/Ttot/BFNORM
    LBF_eff = np.sum(df["tobs"][LOW]*df[bkey][LOW])/LTtot/BFNORM
    HBF_eff = np.sum(df["tobs"][HIGH]*df[bkey][HIGH])/HTtot/BFNORM
    
    metrics["BF"] = BF
    metrics["LOW"]["BF"] = LBF
    metrics["HIGH"]["BF"] = HBF
    
    metrics["BF_eff"] = BF_eff
    metrics["LOW"]["BF_eff"] = LBF_eff
    metrics["HIGH"]["BF_eff"] = HBF_eff
    
    
    ####### Mean Frequency ###########
    
    # we do not calculate a frequency dependence here, because
    # the overall scaling of rate with frequency is difficult to calculate
    # FRBs are brighter at low frequencies, but scatter more, and
    # DM smearing is worse
    
    nubar=np.sum(df["tobs"]*df["fbar"])/Ttot
    nubarh=np.sum(df["tobs"][HIGH]*df["fbar"][HIGH])/HTtot
    nubarl=np.sum(df["tobs"][LOW]*df["fbar"][LOW])/LTtot
    
    #print("Total effective time is ",LTeff," at low, and ",HTeff," at high")
    #print("Mean high frequency is ",nubarh)
    #print("Mean low frequency is ",nubarl,"\n\n\n")
    
    # mean frequency, not weighted by other efficiency factors
    metrics["nubar"] = nubar
    metrics["LOW"]["nubar"] = nubarl
    metrics["HIGH"]["nubar"] = nubarh
    
    
    # on average, we have lost 0.913 due to bandwidth
    
    ################# BANDWIDTH ###############
    
    NORM_BW = 288
    
    bw = np.sum(df['nchans']*df["tobs"])/Ttot
    #print("mean bandwidth ",bw)
    #print("Mean bandwidth loss ",np.sum(df['w_bandwidth']*df["tobs"])/Ttot)
    Hbw = np.sum(df['nchans'][HIGH]*df["tobs"][HIGH])/HTtot
    Lbw = np.sum(df['nchans'][LOW]*df["tobs"][LOW])/LTtot
    #print("Mean HIGH bandwidth loss ",Hbw)
    #print("Mean LOW bandwidth loss ",Lbw,"\n\n")
    
    bw_eff = np.sum((df['nchans']/NORM_BW)**(alpha/2.)*df["tobs"])/Ttot
    Hbw_eff = np.sum((df['nchans'][HIGH]/NORM_BW)**(alpha/2.)*df["tobs"][HIGH])/HTtot
    Lbw_eff = np.sum((df['nchans'][LOW]/NORM_BW)**(alpha/2.)*df["tobs"][LOW])/LTtot
    
    metrics["bw"] = bw
    metrics["LOW"]["bw"] = Lbw
    metrics["HIGH"]["bw"] = Hbw
    
    metrics["bw_eff"] = bw_eff
    metrics["LOW"]["bw_eff"] = Lbw_eff
    metrics["HIGH"]["bw_eff"] = Hbw_eff
    
    ################# RFI ###############
    
    # assumes losing bandwidth to RFI scales sensitivity as delta_nu^0.5,
    # since we lose x% signal, but we keep threshold identical
    # hence, we don't need to calculate a weighted scaling factor here
    # then we lose 70% due to "goodfrac"
    gf = np.sum(df['goodfrac']*df["tobs"])/Ttot
    Lgf = np.sum(df['goodfrac'][LOW]*df["tobs"][LOW])/LTtot
    Hgf = np.sum(df['goodfrac'][HIGH]*df["tobs"][HIGH])/HTtot
    
    gf_eff = np.sum(df['goodfrac']**alpha *df["tobs"])/Ttot
    Lgf_eff = np.sum(df['goodfrac'][LOW]**alpha *df["tobs"][LOW])/LTtot
    Hgf_eff = np.sum(df['goodfrac'][HIGH]**alpha *df["tobs"][HIGH])/HTtot
    
    #print("Mean goodfrac ",gf)
    #print("Mean gf loss ",np.sum(df['w_goodfrac']*df["tobs"])/Ttot)
    #print("Optimal gf loss ",np.sum(df['goodfrac']**0.75*df["tobs"])/Ttot)
    #print("Mean LOW gf loss ",Lgf)
    #print("Mean HIGH gf loss ",Hgf,"\n\n")
    
    metrics["gf"] = gf
    metrics["LOW"]["gf"] = Lgf
    metrics["HIGH"]["gf"] = Hgf
    
    metrics["gf_eff"] = gf_eff
    metrics["LOW"]["gf_eff"] = Lgf_eff
    metrics["HIGH"]["gf_eff"] = Hgf_eff
    
    
    ################# Nant ###############
    
    # then we lose 91% due to less than 25 antennas
    # we scale antenna factor relative to 25
    NORM_NANT = 24
    
    #print("Max number of antennas is ",np.max(df['nant']))
    Nant = np.sum(df['nant']*df["tobs"])/Ttot
    LNant = np.sum(df['nant'][LOW]*df["tobs"][LOW])/LTtot
    HNant = np.sum(df['nant'][HIGH]*df["tobs"][HIGH])/HTtot
    
    Nant_eff = np.sum((df['nant']/NORM_NANT)**alpha *df["tobs"])/Ttot
    LNant_eff = np.sum((df['nant'][LOW]/NORM_NANT)**alpha*df["tobs"][LOW])/LTtot
    HNant_eff = np.sum((df['nant'][HIGH]/NORM_NANT)**alpha*df["tobs"][HIGH])/HTtot
    #print("Mean antennas ",Nant)
    #print("Mean nant loss ",np.sum(df['w_nant']*df["tobs"])/Ttot)
    #print("Low nant loss ",np.sum(df['w_nant'][LOW]*df["tobs"][LOW])/LTtot)
    #print("High nant loss ",np.sum(df['w_nant'][HIGH]*df["tobs"][HIGH])/HTtot,"\n\n")
    
    metrics["Nant"] = Nant
    metrics["LOW"]["Nant"] = LNant
    metrics["HIGH"]["Nant"] = HNant
    
    metrics["Nant_eff"] = Nant_eff
    metrics["LOW"]["Nant_eff"] = LNant_eff
    metrics["HIGH"]["Nant_eff"] = HNant_eff
    
    ############### Total efficiency and effective observation time ###########
    
    # beamfactor * good fraction * bandwidth * Nant
    df['eff'] = df['goodfrac']**alpha * (df['nchans']/NORM_BW)**(alpha/2.) * (df['bfactors']/BFNORM) * (df['nant']/NORM_NANT)**alpha
    df['t_eff'] = df['tobs']*df['eff']
    
    df['zdmeff'] = df['goodfrac']**alpha * (df['nchans']/NORM_BW)**(alpha/2.) * (df['nant']/NORM_NANT)**alpha
    df['zdmt_eff'] = df['tobs']*df['zdmeff']
    
    Teff = np.sum(df['t_eff'])
    HTeff = np.sum(df['t_eff'][HIGH])
    LTeff = np.sum(df['t_eff'][LOW])
    
    metrics["Teff"] = Teff
    metrics["LOW"]["Teff"] = LTeff
    metrics["HIGH"]["Teff"] = HTeff
    
    zdmTeff = np.sum(df['zdmt_eff'])
    zdmHTeff = np.sum(df['zdmt_eff'][HIGH])
    zdmLTeff = np.sum(df['zdmt_eff'][LOW])
    
    metrics["zdmTeff"] = zdmTeff
    metrics["LOW"]["zdmTeff"] = zdmLTeff
    metrics["HIGH"]["zdmTeff"] = zdmHTeff
    
    if savename is not None:
        print("saving to ",savename)
        df.to_csv(savename,index=False)
    return metrics

def plot_cumulative(df,LOW,HIGH,frbs,ks=True,prefix=""):
    """
    Generates some cumulative plots
    
    args:
        df: pandas dataframe containing logfile info
        LOW [list]: indices of data frame corresponding to 900 MHz sample
        HIGH [list]: indices of data frame corresponding to 1300 MHz sample
        frbs: pandas dataframe giving FRB info
        ks [bool]: if True, perform a ks test on the two distributions
                    for consistency
    """
    
    teff = df["t_eff"] #units of effective seconds
    
    ctraw = np.cumsum(df["tobs"]/3600)
    Lctraw = np.cumsum(df["tobs"][LOW]/3600)
    Hctraw = np.cumsum(df["tobs"][HIGH]/3600)
    
    cteff = np.cumsum(teff/3600)
    
    Lcteff = np.cumsum(teff[LOW]/3600)
    Hcteff = np.cumsum(teff[HIGH]/3600)
    
    frbxs,frbys=mf.make_cum_dist(frbs["MJD_craco"])
    frbys *= len(frbs["MJD_craco"])
    
    if ks:
        from scipy.stats import kstest
        from scipy.interpolate import interp1d
        rvs = frbs["MJD_craco"]
        cum_func = interp1d(df["tstart"],cteff/cteff.values[-1],kind="linear",assume_sorted=True)
        result = kstest(rvs,cum_func,mode="exact",alternative="two-sided")
        
        kstat = result[0]
        kpval = result[1]
        
    
    plt.figure()
    
    #plt.plot(df["tstart"],ctraw)
    plt.plot(df["tstart"],cteff,label="Total")
    plt.plot(df["tstart"][LOW],Lcteff,label="900 MHz",linestyle="--")
    plt.plot(df["tstart"][HIGH],Hcteff,label="1300 MHz",linestyle=":")
    
    # normalises
    ax = plt.gca()
    # FRBs
    ax2 = ax.twinx()
    ax2.plot(frbxs,frbys,linestyle="-.",color="black")
    ax2.set_ylabel("$N_{\\rm FRB}$")
    plt.ylim(0,frbys[-1])
    
    plt.sca(ax)
    
    # does a dummy plot
    #plt.plot([-1e9,-1e8],[-100,-100],linestyle="-.",color="black",label="CRACO FRBs")
    #plt.xlim(60280,60650)
    xmin = int(np.min(df["tstart"])/50)*50
    xmax = (int(np.max(df["tstart"])/50)+1)*50
    plt.xlim(xmin,xmax)
    plt.ylim(0,cteff.values[-1])
    plt.xlabel("mjd")
    plt.ylabel("Cumulative $T_{\\rm eff}$ [hr]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"eff_cumulative_fig.png")
    
    plt.close()
    
    plt.figure()
    
    #plt.plot(df["tstart"],ctraw)
    plt.plot(df["tstart"],ctraw,label="Total")
    plt.plot(df["tstart"][LOW],Lctraw,label="900 MHz",linestyle="--")
    plt.plot(df["tstart"][HIGH],Hctraw,label="1300 MHz",linestyle=":")
    
    # FRBs
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(frbxs,frbys)
    ax2.set_ylabel("$N_{\\rm FRB}$")
    plt.ylim(0,18)
    
    plt.sca(ax)
    plt.ylim(0,ctraw.values[-1])
    plt.xlabel("mjd")
    plt.ylabel("Cumulative $T_{\\rm obs}$ [hr]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"raw_cumulative_fig.png")
    plt.close()
    
    if ks:
        return kstat,kpval
    else:
        return
    
def do_basic_plots(df,LOW,HIGH,prefix):
    """
    Produces basic plots
    
    Args:
        df: pandas dataframe containing info
        LOW: indices corresponding to low frequencies
        HIGH: indices corresponding to high frequencies
        prefix [string]: prefix for outputs
        
    """
    
    #### Number of antennas ####
    plt.figure()
    plt.xlabel("Number of antennas")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0.5,36.5,37)
    plt.hist(df["nant"],bins=bins,weights=df["tobs"]/3600)
    plt.xlim(14,26)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"Nant_hist.png")
    plt.close()
    
    #### Bandwidth ####
    plt.figure()
    plt.xlabel("Bandwidth [MHz]")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(5,345,35)
    
    
    plt.hist(df["nchans"],bins=bins,weights=df["tobs"]/3600)
    #plt.xlim(14,26)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"bw_hist.png")
    plt.close()
    
    ##### central frequency ###
    plt.figure()
    plt.xlabel("Central frequency [MHz]")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0,2000,201)
    plt.hist(df["fbar"],bins=bins,weights=df["tobs"]/3600)
    plt.xlim(750,1500)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"Fbar_hist.png")
    plt.close()
    
    
    
    
    ##### good fraction ######
    plt.figure()
    plt.xlabel("Fraction of good data")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0,1,101)
    plt.hist(df["goodfrac"],bins=bins,weights=df["nsamples"]*df["tsamp"]/3600)
    #plt.xlim(14,26)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"gf_hist.png")
    plt.close()
    
    plt.figure()
    plt.xlabel("Fraction of good data")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0,1,101)
    plt.hist(df["goodfrac"][LOW],bins=bins,weights=df["nsamples"][LOW]*df["tsamp"][LOW]/3600,label="900 MHz")
    plt.hist(df["goodfrac"][HIGH],bins=bins,weights=df["nsamples"][HIGH]*df["tsamp"][HIGH]/3600,label="1300 MHz")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"gf_hist_by_freq.png")
    plt.close()
    
    

def plot_tsamp():
    """
    Produces a dummy plot of the sampling time effect
    """
    
    ICSdt=1.182
    CRACOdt = 13.8
    
    NW=61
    wvals = np.logspace(-2,3,NW)
    ICSthresh = 4.4
    ICS = np.full([NW],4.4*(ICSdt**0.5))
    CRACOthresh = ICSthresh * 25**0.5 / (24*23)**0.5 * (336/288)**0.5
    #print("CRACO thresh is ",CRACOthresh) # why 0.99?
    CRACO = np.full([NW],CRACOthresh*(CRACOdt**0.5))
    
    
    OK = np.where(wvals > ICSdt)[0]
    ICS[OK] *= (wvals[OK]/ICSdt)**0.5
    BAD = np.where(wvals > ICSdt*12.5)[0] # search for at most 12 widths in time
    ICS[BAD] *= 1000
    
    OK = np.where(wvals > CRACOdt)[0]
    CRACO[OK] *= (wvals[OK]/CRACOdt)**0.5
    OK = np.where(wvals > CRACOdt*8)[0]
    CRACO[OK] *= (wvals[OK]/(CRACOdt*8))**0.5
    
    
    plt.figure()
    plt.xlabel("FRB effective width [ms]")
    plt.ylabel("$F_{\\rm th}$ [Jy ms]")
    plt.ylim(1,50)
    plt.plot(wvals,ICS,label="ICS 1.182ms",linestyle="--")
    c1 = plt.gca().lines[-1].get_color()
    plt.plot(wvals,CRACO,label="CRACO 13.8ms")
    c2 = plt.gca().lines[-1].get_color()
    
    
    plt.plot([CRACOdt,CRACOdt],[0.1,CRACO[0]],color="black",linestyle=":")
    plt.plot([ICSdt,ICSdt],[0.1,ICS[0]],color="black",linestyle=":")
    
    # I do not know where these 2.83 and 3.46 come from
    plt.plot([CRACOdt*8,CRACOdt*8],[0.1,CRACO[0]*2.83],color="black",linestyle=":")
    plt.plot([ICSdt*12,ICSdt*12],[0.1,ICS[0]*3.46],color="black",linestyle=":")
    
    plt.text(2.1,5.5,"$F_{\\rm th} \\sim w^{0.5}$",rotation=50,color=c1,fontsize=12)
    plt.text(24,4,"$F_{\\rm th} \\sim w^{0.5}$",rotation=50,color=c2,fontsize=12)
    plt.text(200,15,"$F_{\\rm th} \\sim w$",rotation=70,color=c2,fontsize=12)
    
    
    plt.text(0.75,1.1,"$t_{\\rm obs} = 1.182$ ms",rotation=90,color=c1,fontsize=12)
    plt.text(9,1.1,"$t_{\\rm obs} = 13.8$ ms",rotation=90,color=c2,fontsize=12)
    
    
    plt.text(17,1.4,"$12 \\times t_{\\rm obs}$",rotation=90,color=c1,fontsize=12)
    plt.text(120,2,"$8 \\times t_{\\rm obs}$",rotation=90,color=c2,fontsize=12)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/width_threshold_sketch.png")
    plt.close()


    
main()
