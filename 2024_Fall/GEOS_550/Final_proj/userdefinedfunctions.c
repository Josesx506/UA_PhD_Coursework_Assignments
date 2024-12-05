#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "globalvars.h"

double maximumweatheringratefunction(int i, int j)
/* computes the maximum weathering rate, which may vary with depth to include the variable resistance to weathering of bedded/banded rocks */
{   
    Pl=P_0;
	if (structuralcontrolmode==1)
	 {erodeddepth[i][j]=topoinitial[i][j]-topoold[i][j]+U[i][j]*time;
	  if (erodeddepth[i][j]-structuralelevation[i][j]<0) Pl=P_0; else if (erodeddepth[i][j]-structuralelevation[i][j]<100) Pl=P_0/10; else Pl=P_0;}
	return Pl;
}

double colluvialunitsedimentfluxfunction(double slopel, double totslopesqrl, double soill, double cosfactorl)
/* computes the colluvial unit sediment flux as a function of slope, S_c, and soil thickness */
{	double d_cl;

	factor=1/(1-totslopesqrl/sqrS_c);
	if (totslopesqrl>=sqrnearS_c) factor=1/(1-sqrfractionnearS_c)+2*nearS_c*(sqrt(totslopesqrl)-nearS_c)/(sqrS_c*SQR(1-sqrfractionnearS_c));
	d_cl=d_c*(1-1/factor)+MIN(soill,d_c)/factor;
	return D*slopel*(1-exp(-soill/(cosfactorl*d_cl)))*factor/cosfactorl;
}	

double erosionanddepositionfunction(int i, int j)
/* computes the the advection coefficient in the one-way wave equation for fluvial erosion and deposition */
{   
	if (time + timestep < initFluvialProcs) Kl=0; else Kl=K;
	erodeddepth[i][j]=topoinitial[i][j]-topoold[i][j]+U[i][j]*time;
	if (structuralcontrolmode==1)
	 {erodeddepth[i][j]=topoinitial[i][j]-topoold[i][j]+U[i][j]*time;
	  if (erodeddepth[i][j]-structuralelevation[i][j]<0) Kl=K; else if (erodeddepth[i][j]-structuralelevation[i][j]<100) Kl=K/10; else Kl=K;}
	if (n-1>tiny) entrainment[i][j]=MAX(Kl*pow(area[i][j]/width[i][j],p)*pow(slope[i][j],n-1)-entrainmentthreshold/MAX(slope[i][j],0.001),0); else entrainment[i][j]=MAX(Kl*pow(area[i][j]/width[i][j],p)-entrainmentthreshold/MAX(slope[i][j],0.001),0); 
	if (hillslopevsconfinedvalleyvsfloodplain[i][j]==0) deposition[i][j]=0; 
     else deposition[i][j]=depositionvelocity*erodedvolumerate[i][j]/(discharge[i][j]*pow(MAX(slope[i][j],0.001),1.5));
	return entrainment[i][j]-deposition[i][j];
}