// 'Optimised' function to simulate a Weibull sequence
// compile with g++ toy_weibull.cpp -o tw.exe -std=c++17

// This version is used to simulate a repeating FRB with
// repeition governed by a Weibull distribution with shape k,
// and counts the number of observations appearing in time
// intervals approixmately corresponding to CHIME coverage.

// The calculation is repeated for different assumed
// *observed* (NOT intrinsic) rates, and different coverage
// fractions.

#include<math.h>
#include<fstream>
#include<iostream>
#include<ctime>
#include<random>

using namespace std;

// ############# HEADER BLOCK ################

// standard max file length of character strings
const int mfl=1028;

// calculates a random wait time based on some precalculated parameters. Could be made inline.
double sim_wait_time(const double invk, const double scale);

// simulates an observation sequence. Ndet holds the return info (currently set to 3 length)
int sim_observation(double Tmin, double Tmax, double R, double k, double tfrac);
// Ndet holds return values for detections by each telescope


// Initialise random number generator using the Mersenne Twister algorithm (64 bits)
unsigned seed=time(NULL);
mt19937_64 rndm(seed); //random number generator
const long double drmax=(long double) rndm.max();

// ############# MAIN PROGRAM ################

// expects sw.exe input_file k gamma R0 F0 outfile
// potentially add nrepeats and/or options to output diagnostics
// add option to send/receive SEED for random number gen

const int BASE_ARGS=12;

int main(int nargs, char*args[])
	{
    /*
    Main program to simulate the difference between Poisson and Weibull expectations for example FRB behaviours
    
    Simulates via MC:
        - iteration over Poisson expectation
        - iteration over observed fraction per day
        - Calculate: Nrep / N_poisson
        - Calculate: distribution of Nrep vs Poisson for certain scenarios (convolve a power law?)
    
    
    */
    
	// ####### test the definition of the max random number generator on your machine! ######
	//const unsigned long int rmax=rndm.max();
	//const long double drmax=(long double) rmax;
	//cout.precision(30);
	//cout<<"rmax is "<<rndm.max()<<" "<<rmax<<" "<<drmax<<"\n";
	
    int i,j, Ndet;
    
	// holds other inputs
	double k=0.34;
    //k = 0.9;
    
    double toff = 100.; // coefficient for how far to start the first birst prior to the obs start
    double ttot = 365.; // total observation time
    double dec; // declination: minutes per day
    
    int ndecs = 11;
    double dec0=0., dec1 = 89.;
    double ddec = (dec1 - dec0)/(ndecs-1.);
    
    
    const int maxNdet = 100;
    int Ndets[maxNdet+1];
    
    
    double Tmax = 365;// maximum observation time. Whole days only. For reasons.
    double R=1./Tmax;// units of rate per time interval. Probably should be way lower.
    double Tmin = -toff/R;
    double tfrac;
    double torad = 3.14159/180.;
    
    int nreps = 1000;
    double tfracs[3]={0.01,0.1,1};
    ndecs=3;
    
    int sum=0;
    int Zero = 0;
    int Singles = 0;
    
    
    const int NR = 9;
    double Rmults[NR];
    double thisR;
    int iR;
    
    // fils Rmult array
    for (i=0; i<NR; i++)
        {
        Rmults[i] = pow(10,i*0.25-1.);
        }
    
    
    // loop over declinations
    for (i=0; i<ndecs; i++)
        {
        
        //dec = dec0 + i*ddec;
        
        // determines the fractional on-time
        //tfrac = 2.2 / cos(dec*torad) / 360.;
        tfrac = tfracs[i];
        if (tfrac > 1.)
            {
            tfrac = 1.;
            }
	    cout<<"Using parameters k="<<k<<", tfrac "<<tfrac<<" "<<R/tfrac<<endl;
        
            
		// scale R by tfrac to preserve the FRB rate
        cout<<"\n\n\n\nR is scaled to "<<R/tfrac<<" for tfrac "<<tfrac<<" "<<Tmin*tfrac<<endl;
        
        for (iR=0; iR< NR; iR++)
            {
            // resets Ndets
            for (j=0; j<maxNdet+1; j++)
                {
                Ndets[j]=0;
                }
            // scales used rate by Rmult
            thisR = Rmults[iR] * R / tfrac;
            for (j=0; j<nreps; j++)
		        {
		        Ndet=sim_observation(Tmin*tfrac,Tmax,thisR,k,tfrac);
                if (Ndet < maxNdet)
                    {
                    Ndets[Ndet] ++;
                    }
                else
                    {
                    Ndets[maxNdet]++; // overflow bin
                    }
                }
		    
            // prints Ndets array to files.
            // determines how many are zeros and ones
            sum=0;
            Zero = Ndets[0];
            Singles = Ndets[1];
            for (j=2; j<maxNdet+1; j++)
                {
                sum += Ndets[j];
                }
            cout<<Rmults[iR]<<" "<<Zero<<" "<<Singles<<" "<<sum<<"\n";
            }
        }
	
	return 0;
	}

// fast routine to simulate FRBs and compare to observations
// assume Fths all normalised to be >= 1
int sim_observation(double Tmin, double Tmax, double R, double k, double tfrac)
	{
	// pre-calculate some Weibull data
	const double invk=1./k;
	const double scale=tgamma(1.+invk)*R;
	double tfrb = Tmin; // current FRB time
	double dt; // fraction of a day when an FRB is detected
	int Ndet = 0;// number of detected FRBs
    int count=0;
	// goes forward and generates later bursts
	while (tfrb <= Tmax) // keep doing this until we are too late
	    {
        tfrb += sim_wait_time(invk,scale);
        count++;
        dt = tfrb - int(tfrb); // gets integer remainder
        //cout<<tfrb<<" "<<int(tfrb)<<" "<<dt<<" "<<tfrac<<"\n";
        if (tfrb > 0 && dt < tfrac && tfrb < Tmax) // always assumes observing at "start" of day
            {
            //cout<<"Found one! "<<tfrb<<" "<<dt<<" "<<R<<" "<<tfrac<<endl;
            Ndet += 1;
            }
		}
    //cout<<"Count is "<<count<<" for eff rate of "<<count/(Tmax-Tmin)<<" "<<R<<endl;
	return Ndet;
	}

// simulates a wait time from a Weibull distribution
double sim_wait_time(const double invk, const double scale)
	{
	double r = (double) (rndm()/drmax); // go to double precision from long double division
	double delta=pow(-log(r),invk); // distribution from gamma function
	double dtfrb=delta/scale; // scale distribution to desired rate
	return dtfrb;
	}

