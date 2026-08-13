# searches for a match to 171020 host galaxies
import numpy as np

def main():
    
    file1 = open('object_search_2.csv', 'r')
    Lines = file1.readlines()
    gals=[]
    zs=[]
    for i,line in enumerate(Lines):
        if i==0:
            continue
        words=line.split(',')
        gal=words[1]
        z=words[6]
        zs.append(float(z))
        gals.append(gal)
        
    zs=np.array(zs)
    
    file2 = open('modR-MAG_CANDIDATES.csv', 'r')
    Lines = file2.readlines()
    gals2 = []
    for line in Lines:
        words=line.split()
        pre=words[0]
        gal=words[1]
        string=pre+" "+gal
        gals2.append(string)
    
    for i2,g2 in enumerate(gals2):
        OK=False
        for i1,g1 in enumerate(gals):
            if g1==g2:
                OK=True
                print(g1,zs[i1])
        if not OK:
            print("No matching galaxy found for ",g2)
main()
