#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "globalvars.h"

void identifydownslopevalleybottom(int i, int j)
/* identifies the downslope nearest neighbor with the largest contributing area */
{
	topovalleybottom=large;areavalleybottom=0;icdown=0;jcdown=0;
	if ((hillslopevsconfinedvalleyvsfloodplain[iup[i][j]][j]==1)&&(areavalleybottom<area[iup[i][j]][j])) {topovalleybottom=topo[iup[i][j]][j];icdown=iup[i][j];jcdown=j;areavalleybottom=area[iup[i][j]][j];}
	if ((hillslopevsconfinedvalleyvsfloodplain[idown[i][j]][j]==1)&&(areavalleybottom<area[idown[i][j]][j])) {topovalleybottom=topo[idown[i][j]][j];icdown=idown[i][j];jcdown=j;areavalleybottom=area[idown[i][j]][j];}
	if ((hillslopevsconfinedvalleyvsfloodplain[i][jup[i][j]]==1)&&(areavalleybottom<area[i][jup[i][j]])) {topovalleybottom=topo[i][jup[i][j]];icdown=i;jcdown=jup[i][j];areavalleybottom=area[i][jup[i][j]];}
	if ((hillslopevsconfinedvalleyvsfloodplain[i][jdown[i][j]]==1)&&(areavalleybottom<area[i][jdown[i][j]])) {topovalleybottom=topo[i][jdown[i][j]];icdown=i;jcdown=jdown[i][j];areavalleybottom=area[i][jdown[i][j]];}
	if ((hillslopevsconfinedvalleyvsfloodplain[iup[i][j]][jup[i][j]]==1)&&(areavalleybottom<area[iup[i][j]][jup[i][j]])) {topovalleybottom=topo[iup[i][j]][jup[i][j]];icdown=iup[i][j];jcdown=jup[i][j];areavalleybottom=area[iup[i][j]][jup[i][j]];}
	if ((hillslopevsconfinedvalleyvsfloodplain[idown[i][j]][jup[i][j]]==1)&&(areavalleybottom<area[idown[i][j]][jup[i][j]])) {topovalleybottom=topo[idown[i][j]][jup[i][j]];icdown=idown[i][j];jcdown=jup[i][j];areavalleybottom=area[idown[i][j]][jup[i][j]];}
	if ((hillslopevsconfinedvalleyvsfloodplain[iup[i][j]][jdown[i][j]]==1)&&(areavalleybottom<area[iup[i][j]][jdown[i][j]])) {topovalleybottom=topo[iup[i][j]][jdown[i][j]];icdown=iup[i][j];jcdown=jdown[i][j];areavalleybottom=area[iup[i][j]][jdown[i][j]];}
	if ((hillslopevsconfinedvalleyvsfloodplain[idown[i][j]][jdown[i][j]]==1)&&(areavalleybottom<area[idown[i][j]][jdown[i][j]])) {topovalleybottom=topo[idown[i][j]][jdown[i][j]];icdown=idown[i][j];jcdown=jdown[i][j];areavalleybottom=area[iup[i][j]][jdown[i][j]];}
}

void calculateslopealongsteepestquadrant(int i, int j)
/* calculates slope along the steepest of the four quadrants */
{	
	slope[i][j]=0;
	slope1=topo[i][j]-topo[iup[i][j]][j];slope2=topo[i][j]-topo[idown[i][j]][j];slope3=topo[i][j]-topo[i][jup[i][j]];slope4=topo[i][j]-topo[i][jdown[i][j]];
    if ((slope1>0)&&(slope3>0)) {slopenew=sqrt(SQR(slope1)+SQR(slope3))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=iup[i][j];draindirj[i][j]=jup[i][j];}}		
	if ((slope1>0)&&(slope4>0)) {slopenew=sqrt(SQR(slope1)+SQR(slope4))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=iup[i][j];draindirj[i][j]=jdown[i][j];}}	
	if ((slope2>0)&&(slope3>0)) {slopenew=sqrt(SQR(slope2)+SQR(slope3))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=idown[i][j];draindirj[i][j]=jup[i][j];}}	
	if ((slope2>0)&&(slope4>0)) {slopenew=sqrt(SQR(slope2)+SQR(slope4))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=idown[i][j];draindirj[i][j]=jdown[i][j];}}
    if (slope[i][j]<tiny) 
	 {if (slope1>0)  {slopenew=slope1*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=iup[i][j];draindirj[i][j]=j;}}		 
      if (slope2>0)  {slopenew=slope2*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=idown[i][j];draindirj[i][j]=j;}}		 
      if (slope3>0)  {slopenew=slope3*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=i;draindirj[i][j]=jup[i][j];}}		 
	  if (slope4>0)  {slopenew=slope4*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=i;draindirj[i][j]=jdown[i][j];}}}		 
}

void mapbankfailureregion(int i, int j)
{   
	 if (((BCmask[i][j]==0)||(BCmask[i][j]==3))&&(drainagebasinmask[i][j]==1)) 
	  {calculateslopealongsteepestquadrant(i,j);
	   if (slope[i][j]>fractionnearS_c*S_c)
	    {topo[i][j]-=bankfailureincrement;
   	     volumeofbankfailure+=deltax*deltax*bankfailureincrement;
	     mapbankfailureregion(iup[i][j],j);mapbankfailureregion(iup[i][j],j);mapbankfailureregion(idown[i][j],j);mapbankfailureregion(i,jup[i][j]);mapbankfailureregion(i,jdown[i][j]);mapbankfailureregion(iup[i][j],jup[i][j]);mapbankfailureregion(idown[i][j],jup[i][j]);mapbankfailureregion(iup[i][j],jdown[i][j]);mapbankfailureregion(idown[i][j],jdown[i][j]);}}
}

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
{
	factor=1/(1-totslopesqrl/sqrS_c);
	if (totslopesqrl>=sqrnearS_c) factor=1/(1-sqrfractionnearS_c)+2*nearS_c*(sqrt(totslopesqrl)-nearS_c)/(sqrS_c*SQR(1-sqrfractionnearS_c));
	if (soil[i][j]>h_s) soill=soill*(1-1/factor)+MIN(soill,h_s)/factor; else soill=h_s;
	return D_0*slopel*soill*factor/cosfactorl;
}

double erosionanddepositionfunction(int i, int j)
/* computes the the advection coefficient in the one-way wave equation for fluvial erosion and deposition */
{   
	Kl=K_0;
	erodeddepth[i][j]=topoinitial[i][j]-topoold[i][j]+U[i][j]*time;
	if (structuralcontrolmode==1)
	 {erodeddepth[i][j]=topoinitial[i][j]-topoold[i][j]+U[i][j]*time;
	  if (erodeddepth[i][j]-structuralelevation[i][j]<0) Kl=K_0; else if (erodeddepth[i][j]-structuralelevation[i][j]<100) Kl=K_0/10; else Kl=K_0;}
	entrainment[i][j]=MAX(Kl*pow(area[i][j]/width[i][j],p)*pow(slope[i][j],n-1)-entrainmentthreshold/MAX(slope[i][j],0.001),0);
	if (hillslopevsconfinedvalleyvsfloodplain[i][j]==0) deposition[i][j]=0; 
     else deposition[i][j]=depositionvelocity*erodedvolumerate[i][j]/(discharge[i][j]*pow(MAX(slope[i][j],0.001),1.5));
	return entrainment[i][j]-deposition[i][j];
}

void bankretreatfunction(int i, int j)
/* performs bank retreat */
{
	if (((BCmask[i][j]==0)||(BCmask[i][j]==3))&&(drainagebasinmask[i][j]==1))
	 {identifydownslopevalleybottom(i,j);
      if ((hillslopevsconfinedvalleyvsfloodplain[i][j]==2)&&(icdown>0)) 
	   {bankheight=topoafterDL[i][j]-topovalleybottom;
		if (bankheight>criticalbankheight) accumulatedbankretreat[i][j]+=bankretreatrate*timestep;
        if (accumulatedbankretreat[i][j]>deltax) 
		 {accumulatedbankretreat[i][j]=0;
	      volumeofbankfailure=deltax*deltax*(topoafterDL[i][j]-topovalleybottom);
	      topoafterDL[i][j]=topovalleybottom;
          mapbankfailureregion(iup[i][j],j);mapbankfailureregion(idown[i][j],j);mapbankfailureregion(i,jup[i][j]);mapbankfailureregion(i,jdown[i][j]);mapbankfailureregion(iup[i][j],jup[i][j]);mapbankfailureregion(idown[i][j],jup[i][j]);mapbankfailureregion(iup[i][j],jdown[i][j]);mapbankfailureregion(idown[i][j],jdown[i][j]);		  
		  erodedvolumerate[icdown][jcdown]+=volumeofbankfailure/timestep;}}}
}