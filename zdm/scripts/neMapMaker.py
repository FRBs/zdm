def createRadialNeFits(clusterName, mainTableName, clusterRedshift, numberCells, centreCell):
    header = fits.getheader('Thermo_MACSJ0717_N.fits')
    mt = makeTableFromURL("https://web.pa.msu.edu/astro/MC2/accept/accept_main.tab")
    url = "https://web.pa.msu.edu/astro/MC2/accept/data/"+clusterName+"_profiles.dat"
    df = makeTableFromURL(url)
    sk = astropy.coordinates.SkyCoord([mt['RA'][mt['#Name']==mainTableName].iloc[0]+' '+mt['Dec'][mt['#Name']==mainTableName].iloc[0]], frame='icrs', unit=(u.hourangle, u.deg))
    header['NAXIS1'] = numberCells
    header['NAXIS2'] = numberCells
    header['CRPIX1'] = centreCell
    header['CRPIX2'] = centreCell
    interpFunc = scipy.interpolate.interp1d(df['Rin'],df['nelec'], bounds_error=False,fill_value=0)
    coords = np.meshgrid(np.arange(numberCells),np.arange(numberCells))
    nePlane = np.zeros([numberCells, numberCells])
    physRange = np.linspace(-np.amax(df['Rin']),np.amax(df['Rin']),numberCells)
    coords = np.meshgrid(physRange,physRange)
    nePlane = interpFunc((coords[0]**2+coords[1]**2)**0.5)
    header['CDELT1'] = np.mean(np.diff(physRange))/cosmo.angular_diameter_distance(clusterRedshift).value*180/np.pi
    header['CDELT2'] = np.mean(np.diff(physRange))/cosmo.angular_diameter_distance(clusterRedshift).value*180/np.pi
    header['CRVAL1'] = sk.ra[0].value
    header['CRVAL2'] = sk.dec[0].value
    fits.writeto('Radial'+clusterName+'.fits', nePlane, header, overwrite=True)


 def makeTableFromURL(url):
     response = requests.get(url)
     data = response.text
     lines = data.splitlines()
     headers = lines[0].strip().split()
     units = lines[1].strip().split()
     rows = [line.strip().split() for line in lines[2:]]
     df = pd.DataFrame(rows, columns=headers)
     df = df.apply(pd.to_numeric, errors='ignore')
     return df
