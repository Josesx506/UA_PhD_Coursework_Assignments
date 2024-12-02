#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "util.h"
#include "globalvars.h"
#include "userdefinedfunctions.h"

// Integer Variables
int mini,minj,maxi,maxj,steps,numberofavalancheincrements,spatiallyvariableupliftmode,sweep,mini,minj,sedimentfluxdrivenincisionmode,upslopecount,downslopecount,pass;
int rillspacingmode,structuralcontrolmode,**BCmask,**divideadjacenttomaskedge,**hillslopevsconfinedvalleyvsfloodplain,**faultmask,advected,counter,routingmethod;
int *pitsandflatsis,*pitsandflatsjs,**iup,**idown,**jup,**jdown,**iup2,**idown2,**jup2,**jdown2,**draindiri,**draindirj,**draindiriup,**draindirjup,residi,residj;
int i,j,ic,jc,ic2,icup,jcup,icdown,jcdown,icdowndown,jcdowndown,t,count,lattice_size_x,lattice_size_y,*topovecind,*topovecind2,*neighboringareasind;
// Float Variables
double erode,difference,minupslopeneighbor,maxslope,**timetonextcliffretreat,**timetonextcliffretreatold,S_d,fillincrementl,topoafterfluvialnew,diag,minareaneighbor;
double maxareaneighbor,bankerosion,volumeofbankfailure,bankheight,criticalbankheight,bankretreatrate,**accumulatedbankretreat,**erodeddepth,**entrainment,**deposition;
double *topovec,*topovec2,*neighboringareas,soilporosity,depositionvelocity,maxareaup,maxareadown,meanupslopearea,meandownslopearea,stddevupslopearea,stddevdownslopearea;
double **discharge,**depositionrate,entrainmentthreshold,**topo2,**area2c,**area2cp,**frac,rillspacing,microtopographiccurvature,ratiooflargetosmallareas,maxvelocity;
double sqrfractionnearS_c,nearS_c,timetoremovesoil,fractionaldistance,mincurvatureforwidthestimation,slopenew,a1,b1,maxtopo,distance,timeremaining,erodibility,maxtimestep;
double sqrnearS_c,factor,widthlowest,totalerosion,erosionintobedrock,erodibilitycontrast,slopetrial,totslopesqr1,totslopesqr2,totslopesqr3,totslopesqr4,maxtopo,mintopo;
double maxarea,areavalleybottom,topovalleybottom,velocity,dist,distbetweenup,distbetween,distbetweendown,rdown,rup,phidown,phiup,fup,fdown,topoup,topodown,topodowndown;
double bankretreatparameter,**alluvialfluxinav,**alluvialfluxoutav,alluvialfluxinavl,alluvialfluxoutavl,averageuplanderosionrate,totalfluxout,depositedvolume,downslopeneumannbc;
double **Uinitial,uplandarea,totalerodedupland,totalerodedcolluvial,Pl,Kl,k,maxim,initialsoilbasin,P_0l,erode,**erodedvolumerate,**erodedvolumeratecopy,**meanbasinslope,**flow1;
double **flow2,**flow3,**flow4,**flow5,**flow6,**flow7,**flow8,**flow12,**flow22,**flow32,**flow42,**flow52,**flow62,**flow72,**flow82,**area1,**area2,**area3,**area4,**area5;
double **area6,**area7,**area8,slope1,slope2,slope3,slope4,m,K,k_0,**colluvialunitsedimentfluxin,**colluvialunitsedimentfluxout,**alluvialfluxin,**alluvialfluxout,**erodedvolumerate;
double **meanbasinslope,**structuralelevation,nextadvectiontime,keeptimestep,horizontalslipvector,verticaltohorizontalratio,U_h,U_vfault,K,advectioninterval,advectiontimestep;
double advectioninterval,cosfactor,xsectslope,max1,max2,max3,maxcurv,tot,areai,areaj,down,initialsoilupland,characteristicrunoffrate,manningsn,noiseamplitude,timestepfactorincrease;
double min,max,printcounter,printinterval,**weatheringrate,rhobedrockoverrhosoil,n,U_v,K,p,P_0,h_0,d_c,D,S_c,sqrS_c,deltax,oneoverdeltax,oneoverdeltax2,duration,timestep,time;
double resid,residlast,**topoinitial,**draindirdistanceup,**draindirdistance,**area,**curv,**width,**soil,**bedrock,**soilold,**soilcopy,**slope,**topo,**topocopy,**topoold;
double **topoafterfluvial,**soilafterfluvial,**U,depthxout1,depthxout2,depthyout1,depthyout2,depthxout,depthxin,depthyout,depthyin,slopexout1,slopeyout1,slopexout2,slopeyout2;
double slopexout,slopeyout,slopexin,slopeyin,denomin,denomout,sqrslopein,sqrslopeout,initFluvialProcs;

void printfiles()
/* prints selected state variables to xyz files for later input into a visualization program such as Paraview */
{   FILE *fp0;
    char title0[100];
	 
	if (time>=printcounter)
     {printcounter+=printinterval;
	  sprintf(title0,"./movie/topo_%d.xyz",counter);
	  fp0=fopen(title0,"w");
	  for (i=1;i<=lattice_size_x;i++) for (j=1;j<=lattice_size_y;j++) fprintf(fp0,"%12.1f %12.1f %12.5f\n",(i-1)*deltax,(j-1)*deltax,topo[i][j]);
	  fflush(fp0);fclose(fp0);
	  sprintf(title0,"./movie/hillslopevsconfinedvalleyvsfloodplain_%d.xyz",counter);
	  fp0=fopen(title0,"w");
	  for (i=1;i<=lattice_size_x;i++) for (j=1;j<=lattice_size_y;j++) fprintf(fp0,"%12.1f %12.1f %12.5f\n",(i-1)*deltax,(j-1)*deltax,1.*hillslopevsconfinedvalleyvsfloodplain[i][j]);
	  fflush(fp0);fclose(fp0);
	  sprintf(title0,"./movie/width_%d.xyz",counter);
	  fp0=fopen(title0,"w");
	  for (i=1;i<=lattice_size_x;i++) for (j=1;j<=lattice_size_y;j++) fprintf(fp0,"%12.1f %12.1f %12.5f\n",(i-1)*deltax,(j-1)*deltax,width[i][j]);
	  fflush(fp0);fclose(fp0);
	  sprintf(title0,"./movie/soil_%d.xyz",counter);
	  fp0=fopen(title0,"w");
	  for (i=1;i<=lattice_size_x;i++) for (j=1;j<=lattice_size_y;j++) fprintf(fp0,"%12.1f %12.1f %12.5f\n",(i-1)*deltax,(j-1)*deltax,soil[i][j]);
	  fflush(fp0);fclose(fp0);
	  sprintf(title0,"./movie/area_%d.xyz",counter);
	  fp0=fopen(title0,"w");
	  for (i=1;i<=lattice_size_x;i++) for (j=1;j<=lattice_size_y;j++) fprintf(fp0,"%12.1f %12.1f %f\n",(i-1)*deltax,(j-1)*deltax,sqrt(area[i][j]));
	  fflush(fp0);fclose(fp0);
	  counter++;}
}

void push(int i, int j)
/* pushes grid points to a stack as part of recursive filling and spilling of pits and flats (hydrologic correction) */
{
	count++;
    pitsandflatsis[count]=i;
    pitsandflatsjs[count]=j;
}

void pop()
/* pops grid points from the stack as part of recursive filling and spilling of pits and flats (hydrologic correction) */
{
    ic=pitsandflatsis[count];
    jc=pitsandflatsjs[count];
	count--;
}

void hydrologiccorrection()
/* performs rescursive filling and spilling of pits and flats */ 
{   int i,j;
    double max;

    count=stacksize;
    while (count==stacksize)
    {count=0;
     for (j=1;j<=lattice_size_y;j++)
	  for (i=1;i<=lattice_size_x;i++)
	   {count=0;
	    push(i,j);
	    while (count>0)
	     {pop();
          max=topo[ic][jc];
          if (topo[iup[ic][jc]][jc]<max) max=topo[iup[ic][jc]][jc];if (topo[idown[ic][jc]][jc]<max) max=topo[idown[ic][jc]][jc];if (topo[ic][jup[ic][jc]]<max) max=topo[ic][jup[ic][jc]];if (topo[ic][jdown[ic][jc]]<max) max=topo[ic][jdown[ic][jc]];if (topo[iup[ic][jc]][jup[ic][jc]]<max) max=topo[iup[ic][jc]][jup[ic][jc]];if (topo[idown[ic][jc]][jdown[ic][jc]]<max) max=topo[idown[ic][jc]][jdown[ic][jc]];if (topo[idown[ic][jc]][jup[ic][jc]]<max) max=topo[idown[ic][jc]][jup[ic][jc]];if (topo[iup[ic][jc]][jdown[ic][jc]]<max) max=topo[iup[ic][jc]][jdown[ic][jc]];
          if ((BCmask[ic][jc]!=1)&&(BCmask[ic][jc]!=2)&&(BCmask[i][j]!=5)&&(topo[ic][jc]<=max)&&(count<stacksize))
		   {topo[ic][jc]=max+fillincrementinhydrologiccorrection;
		    push(ic,jc);push(iup[ic][jc],jc);push(idown[ic][jc],jc);push(ic,jup[ic][jc]);push(ic,jdown[ic][jc]);push(iup[ic][jc],jup[ic][jc]);push(idown[ic][jc],jdown[ic][jc]);push(idown[ic][jc],jup[ic][jc]);push(iup[ic][jc],jdown[ic][jc]);}}}}
}

void calculateslopealongsteepestquadrant(int i, int j, double topol, double topoliup, double topolidown, double topoljup, double topoljdown)
/* calculates slope along the steepest of the four quadrants (used for colluvial processes) */
{	
	slope[i][j]=0;slope1=0;slope2=0;slope3=0;slope4=0;
	if (BCmask[iup[i][j]][j]!=5) slope1=topol-topoliup;if (BCmask[idown[i][j]][j]!=5) slope2=topol-topolidown;if (BCmask[i][jup[i][j]]!=5) slope3=topol-topoljup;if (BCmask[i][jdown[i][j]]!=5) slope4=topol-topoljdown;
    if ((slope1>0)&&(slope3>0)) {slopenew=sqrt(SQR(slope1)+SQR(slope3))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=iup[i][j];draindirj[i][j]=jup[i][j];}}		
	if ((slope1>0)&&(slope4>0)) {slopenew=sqrt(SQR(slope1)+SQR(slope4))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=iup[i][j];draindirj[i][j]=jdown[i][j];}}	
	if ((slope2>0)&&(slope3>0)) {slopenew=sqrt(SQR(slope2)+SQR(slope3))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=idown[i][j];draindirj[i][j]=jup[i][j];}}	
	if ((slope2>0)&&(slope4>0)) {slopenew=sqrt(SQR(slope2)+SQR(slope4))*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=idown[i][j];draindirj[i][j]=jdown[i][j];}}
    if (slope[i][j]<tiny) 
	 {if (slope1>0) {slopenew=slope1*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=iup[i][j];draindirj[i][j]=j;}}		 
      if (slope2>0) {slopenew=slope2*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=idown[i][j];draindirj[i][j]=j;}}		 
      if (slope3>0) {slopenew=slope3*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=i;draindirj[i][j]=jup[i][j];}}		 
	  if (slope4>0) {slopenew=slope4*oneoverdeltax;if (slopenew>slope[i][j]) {slope[i][j]=slopenew;draindiri[i][j]=i;draindirj[i][j]=jdown[i][j];}}}		 
}

void calculateslopeD8(int i, int j)
/* calculates slope along the steepest of the 8 nearest neighbor directions (used for fluvial processes) */
{   
    down=0;max=0;
	draindiri[i][j]=i;draindirj[i][j]=j;draindirdistance[i][j]=1;
	if ((BCmask[iup[i][j]][j]!=5)&&(topo[iup[i][j]][j]<topo[i][j])&&((topo[iup[i][j]][j]-topo[i][j])<down)) {down=topo[iup[i][j]][j]-topo[i][j];draindiri[i][j]=iup[i][j];draindirj[i][j]=j;draindirdistance[i][j]=1;}
    if ((BCmask[idown[i][j]][j]!=5)&&(topo[idown[i][j]][j]<topo[i][j])&&((topo[idown[i][j]][j]-topo[i][j])<down)) {down=topo[idown[i][j]][j]-topo[i][j];draindiri[i][j]=idown[i][j];draindirj[i][j]=j;draindirdistance[i][j]=1;}
    if ((BCmask[i][jup[i][j]]!=5)&&(topo[i][jup[i][j]]<topo[i][j])&&((topo[i][jup[i][j]]-topo[i][j])<down)) {down=topo[i][jup[i][j]]-topo[i][j];draindiri[i][j]=i;draindirj[i][j]=jup[i][j];draindirdistance[i][j]=1;}
    if ((BCmask[i][jdown[i][j]]!=5)&&(topo[i][jdown[i][j]]<topo[i][j])&&((topo[i][jdown[i][j]]-topo[i][j])<down)) {down=topo[i][jdown[i][j]]-topo[i][j];draindiri[i][j]=i;draindirj[i][j]=jdown[i][j];draindirdistance[i][j]=1;}
    if ((BCmask[iup[i][j]][jup[i][j]]!=5)&&(topo[iup[i][j]][jup[i][j]]<topo[i][j])&&((topo[iup[i][j]][jup[i][j]]-topo[i][j])*oneoversqrt2<down)) {down=(topo[iup[i][j]][jup[i][j]]-topo[i][j])*oneoversqrt2;draindiri[i][j]=iup[i][j];draindirj[i][j]=jup[i][j];draindirdistance[i][j]=sqrt2;}
    if ((BCmask[idown[i][j]][jup[i][j]]!=5)&&(topo[idown[i][j]][jup[i][j]]<topo[i][j])&&((topo[idown[i][j]][jup[i][j]]-topo[i][j])*oneoversqrt2<down)) {down=(topo[idown[i][j]][jup[i][j]]-topo[i][j])*oneoversqrt2;draindiri[i][j]=idown[i][j];draindirj[i][j]=jup[i][j];draindirdistance[i][j]=sqrt2;}
    if ((BCmask[iup[i][j]][jdown[i][j]]!=5)&&(topo[iup[i][j]][jdown[i][j]]<topo[i][j])&&((topo[iup[i][j]][jdown[i][j]]-topo[i][j])*oneoversqrt2<down)) {down=(topo[iup[i][j]][jdown[i][j]]-topo[i][j])*oneoversqrt2;draindiri[i][j]=iup[i][j];draindirj[i][j]=jdown[i][j];draindirdistance[i][j]=sqrt2;}
    if ((BCmask[idown[i][j]][jdown[i][j]]!=5)&&(topo[idown[i][j]][jdown[i][j]]<topo[i][j])&&((topo[idown[i][j]][jdown[i][j]]-topo[i][j])*oneoversqrt2<down)) {down=(topo[idown[i][j]][jdown[i][j]]-topo[i][j])*oneoversqrt2;draindiri[i][j]=idown[i][j];draindirj[i][j]=jdown[i][j];draindirdistance[i][j]=sqrt2;}
}

void lateraladvection()
/* performs horizontal tectonic displacement */
{
	if (time+timestep>nextadvectiontime) 
	 {timestep=nextadvectiontime-time;
	  nextadvectiontime+=advectioninterval;
      for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) {topocopy[i][j]=topoold[i][j];soilcopy[i][j]=soilold[i][j];}
	  if (fabs(horizontalslipvector)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[idown[i][j]][j];soilold[i][j]=soilcopy[idown[i][j]][j];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];} 
	  if (fabs(horizontalslipvector-45*PI/180)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[idown[i][j]][jup[i][j]];soilold[i][j]=soilcopy[idown[i][j]][jup[i][j]];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];} 
	  if (fabs(horizontalslipvector-90*PI/180)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[i][jup[i][j]];soilold[i][j]=soilcopy[i][jup[i][j]];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];} 
	  if (fabs(horizontalslipvector-135*PI/180)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[iup[i][j]][jup[i][j]];soilold[i][j]=soilcopy[iup[i][j]][jup[i][j]];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];} 
	  if (fabs(horizontalslipvector-180*PI/180)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[iup[i][j]][j];soilold[i][j]=soilcopy[iup[i][j]][j];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];} 
	  if (fabs(horizontalslipvector-225*PI/180)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[iup[i][j]][jdown[i][j]];soilold[i][j]=soilcopy[iup[i][j]][jdown[i][j]];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];} 
	  if (fabs(horizontalslipvector-270*PI/180)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[i][jdown[i][j]];soilold[i][j]=soilcopy[i][jdown[i][j]];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];} 
	  if (fabs(horizontalslipvector-315*PI/180)<tiny) for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (faultmask[i][j]>0) {topoold[i][j]=topocopy[idown[i][j]][jdown[i][j]];soilold[i][j]=soilcopy[idown[i][j]][jdown[i][j]];topo[i][j]=topoold[i][j];soil[i][j]=topoold[i][j];bedrock[i][j]=topo[i][j]-soil[i][j];}}
}

void defineinitialelevationupliftandboundaryconditions()
/* assign initial elevations, uplift, and boundary conditions */
{	FILE *fp0;

	fp0=fopen("./BCmask.txt","r"); //BCmask = 0 (no BC), 1 (Dirichlet), 2 (Neumann), 3 (periodic), 4 (drainage basin), 5 (exterior of drainage basin), 6 (second derivative of elevation normal to boundary is zero)
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) fscanf(fp0,"%d",&BCmask[i][j]);	 
	fclose(fp0);
	for (i=1;i<=lattice_size_x;i++) 
	 for (j=1;j<=lattice_size_y;j++)
	  {iup[i][j]=i+1;jup[i][j]=j+1;idown[i][j]=i-1;jdown[i][j]=j-1;
       if (i==1) idown[i][j]=1;
       if (i==lattice_size_x) iup[i][j]=lattice_size_x;
	   if (j==1) jdown[i][j]=1;
	   if (j==lattice_size_y) jup[i][j]=lattice_size_y;
	   if (BCmask[i][j]==3)
	    {if (i==1) idown[i][j]=lattice_size_x;
         if (i==lattice_size_x) iup[i][j]=1;
	     if (j==1) jdown[i][j]=lattice_size_y;
	     if (j==lattice_size_y) jup[i][j]=1;}}
	for (i=1;i<=2*lattice_size_x;i++) 
	 for (j=1;j<=2*lattice_size_y;j++)
	  {iup2[i][j]=i+1;jup2[i][j]=j+1;idown2[i][j]=i-1;jdown2[i][j]=j-1;
       if (i==1) idown2[i][j]=1;
       if (i==2*lattice_size_x) iup2[i][j]=2*lattice_size_x;
	   if (j==1) jdown2[i][j]=1;
	   if (j==2*lattice_size_y) jup2[i][j]=2*lattice_size_y;
	   if (BCmask[(i-1)/2+1][(j-1)/2]==3)
		{if (i==1) idown2[i][j]=2*lattice_size_x;
         if (i==2*lattice_size_x) iup2[i][j]=1;
	     if (j==1) jdown2[i][j]=2*lattice_size_y;
	     if (j==2*lattice_size_y) jup2[i][j]=1;}}  
	if ((fabs(horizontalslipvector)<tiny)||(fabs(horizontalslipvector-90*PI/180)<tiny)||(fabs(horizontalslipvector-180*PI/180)<tiny)||(fabs(horizontalslipvector-270*PI/180)<tiny)) {advectioninterval=deltax/fabs(U_h);nextadvectiontime=advectioninterval;}
	if ((fabs(horizontalslipvector-45*PI/180)<tiny)||(fabs(horizontalslipvector-135*PI/180)<tiny)||(fabs(horizontalslipvector-225*PI/180)<tiny)||(fabs(horizontalslipvector-315*PI/180)<tiny)) {advectioninterval=deltax*sqrt2/fabs(U_h);nextadvectiontime=advectioninterval;}
	fp0=fopen("./faultmask.txt","r");
	for (j=1;j<=lattice_size_y;j++)
	 for (i=1;i<=lattice_size_x;i++) 
	  {fscanf(fp0,"%d",&faultmask[i][j]); 
	   if (faultmask[i][j]>0) {faultmask[i][j]=1;U[i][j]=U_v;} else U[i][j]=0;
	   if ((BCmask[i][j]==1)||(BCmask[i][j]==5)) U[i][j]=0;}
    fclose(fp0);
	if (spatiallyvariableupliftmode==1)
	 {fp0=fopen("./uplift.txt","r");
	  for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) fscanf(fp0,"%lf",&U[i][j]);
	  fclose(fp0);}
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) Uinitial[i][j]=U[i][j];
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if ((BCmask[i][j]==4)&&((BCmask[iup[i][j]][j]==5)||(BCmask[idown[i][j]][j]==5)||(BCmask[i][jup[i][j]]==5)||(BCmask[i][jdown[i][j]]==5))) divideadjacenttomaskedge[i][j]=1; else divideadjacenttomaskedge[i][j]=0;  
	uplandarea=0;
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if (U[i][j]>tiny) {uplandarea++;soil[i][j]=initialsoilupland;} else soil[i][j]=initialsoilbasin;
	uplandarea*=deltax*deltax;
	if (structuralcontrolmode==1)
	 {fp0=fopen("./structuralelevation.txt","r");
	  for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) fscanf(fp0,"%lf",&structuralelevation[i][j]);
	  fclose(fp0);}
	fp0=fopen("./inputtopo.txt","r");
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) fscanf(fp0,"%lf",&topo[i][j]);
    fclose(fp0);
	hydrologiccorrection(); 
	for (j=1;j<=lattice_size_y;j++)
	 for (i=1;i<=lattice_size_x;i++) 
	  {bedrock[i][j]=topo[i][j]-soil[i][j];
	   calculateslopealongsteepestquadrant(i,j,topo[i][j],topo[iup[i][j]][j],topo[idown[i][j]][j],topo[i][jup[i][j]],topo[i][jdown[i][j]]);
	   timetonextcliffretreat[i][j]=deltax/maximumweatheringratefunction(i,j);}
}

void definevectorsandmatrices()
/* defines all vector and matrix variables */
{
	BCmask=imatrix(1,lattice_size_x,1,lattice_size_y); 
	faultmask=imatrix(1,lattice_size_x,1,lattice_size_y);
	divideadjacenttomaskedge=imatrix(1,lattice_size_x,1,lattice_size_y);
	hillslopevsconfinedvalleyvsfloodplain=imatrix(1,lattice_size_x,1,lattice_size_y);
	draindiri=imatrix(1,lattice_size_x,1,lattice_size_y);draindirj=imatrix(1,lattice_size_x,1,lattice_size_y);
    draindiriup=imatrix(1,lattice_size_x,1,lattice_size_y);draindirjup=imatrix(1,lattice_size_x,1,lattice_size_y);
	draindirdistance=dmatrix(1,lattice_size_x,1,lattice_size_y);draindirdistanceup=dmatrix(1,lattice_size_x,1,lattice_size_y);
	colluvialunitsedimentfluxin=dmatrix(1,lattice_size_x,1,lattice_size_y);colluvialunitsedimentfluxout=dmatrix(1,lattice_size_x,1,lattice_size_y);
	entrainment=dmatrix(1,lattice_size_x,1,lattice_size_y);deposition=dmatrix(1,lattice_size_x,1,lattice_size_y);
	weatheringrate=dmatrix(1,lattice_size_x,1,lattice_size_y);
	meanbasinslope=dmatrix(1,lattice_size_x,1,lattice_size_y);
	erodedvolumerate=dmatrix(1,lattice_size_x,1,lattice_size_y);erodedvolumeratecopy=dmatrix(1,lattice_size_x,1,lattice_size_y);
	structuralelevation=dmatrix(1,lattice_size_x,1,lattice_size_y);
	slope=dmatrix(1,lattice_size_x,1,lattice_size_y);
	area=dmatrix(1,lattice_size_x,1,lattice_size_y);
	U=dmatrix(1,lattice_size_x,1,lattice_size_y);Uinitial=dmatrix(1,lattice_size_x,1,lattice_size_y);
	soil=dmatrix(1,lattice_size_x,1,lattice_size_y);soilcopy=dmatrix(1,lattice_size_x,1,lattice_size_y);soilold=dmatrix(1,lattice_size_x,1,lattice_size_y);soilafterfluvial=dmatrix(1,lattice_size_x,1,lattice_size_y);
	area=dmatrix(1,lattice_size_x,1,lattice_size_y);area2c=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);area2cp=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);
	frac=dmatrix(1,lattice_size_x,1,lattice_size_y);
	curv=dmatrix(1,lattice_size_x,1,lattice_size_y);
	bedrock=dmatrix(1,lattice_size_x,1,lattice_size_y);
	timetonextcliffretreat=dmatrix(1,lattice_size_x,1,lattice_size_y);timetonextcliffretreatold=dmatrix(1,lattice_size_x,1,lattice_size_y);
	discharge=dmatrix(1,lattice_size_x,1,lattice_size_y);
	depositionrate=dmatrix(1,lattice_size_x,1,lattice_size_y);
	width=dmatrix(1,lattice_size_x,1,lattice_size_y);
	erodeddepth=dmatrix(1,lattice_size_x,1,lattice_size_y);
	accumulatedbankretreat=dmatrix(1,lattice_size_x,1,lattice_size_y);
	topo=dmatrix(1,lattice_size_x,1,lattice_size_y);topoinitial=dmatrix(1,lattice_size_x,1,lattice_size_y);topocopy=dmatrix(1,lattice_size_x,1,lattice_size_y);topoold=dmatrix(1,lattice_size_x,1,lattice_size_y);topoafterfluvial=dmatrix(1,lattice_size_x,1,lattice_size_y);topo2=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);
	topovec=dvector(1,lattice_size_x*lattice_size_y);topovecind=ivector(1,lattice_size_x*lattice_size_y);topovec2=dvector(1,4*lattice_size_x*lattice_size_y);topovecind2=ivector(1,4*lattice_size_x*lattice_size_y);
	neighboringareas=dvector(1,8);neighboringareasind=ivector(1,8);
	divideadjacenttomaskedge=imatrix(1,lattice_size_x,1,lattice_size_y);
	area1=dmatrix(1,lattice_size_x,1,lattice_size_y);area2=dmatrix(1,lattice_size_x,1,lattice_size_y);area3=dmatrix(1,lattice_size_x,1,lattice_size_y);area4=dmatrix(1,lattice_size_x,1,lattice_size_y);area5=dmatrix(1,lattice_size_x,1,lattice_size_y);area6=dmatrix(1,lattice_size_x,1,lattice_size_y);area7=dmatrix(1,lattice_size_x,1,lattice_size_y);area8=dmatrix(1,lattice_size_x,1,lattice_size_y);
	flow1=dmatrix(1,lattice_size_x,1,lattice_size_y);flow2=dmatrix(1,lattice_size_x,1,lattice_size_y);flow3=dmatrix(1,lattice_size_x,1,lattice_size_y);flow4=dmatrix(1,lattice_size_x,1,lattice_size_y);flow5=dmatrix(1,lattice_size_x,1,lattice_size_y);flow6=dmatrix(1,lattice_size_x,1,lattice_size_y);flow7=dmatrix(1,lattice_size_x,1,lattice_size_y);flow8=dmatrix(1,lattice_size_x,1,lattice_size_y);
	flow12=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);flow22=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);flow32=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);flow42=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);flow52=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);flow62=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);flow72=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);flow82=dmatrix(1,2*lattice_size_x,1,2*lattice_size_y);
	pitsandflatsis=ivector(1,stacksize);pitsandflatsjs=ivector(1,stacksize);
	idown=imatrix(1,lattice_size_x,1,lattice_size_y);iup=imatrix(1,lattice_size_x,1,lattice_size_y);jup=imatrix(1,lattice_size_x,1,lattice_size_y);jdown=imatrix(1,lattice_size_x,1,lattice_size_y);
    idown2=imatrix(1,2*lattice_size_x,1,2*lattice_size_y);iup2=imatrix(1,2*lattice_size_x,1,2*lattice_size_y);jup2=imatrix(1,2*lattice_size_x,1,2*lattice_size_y);jdown2=imatrix(1,2*lattice_size_x,1,2*lattice_size_y);
}

void establishinitialconditions()
/* read in parameters from input files and hydrologically correct initial topography */
{   FILE *fpin;
    char temp[200];

    fpin=fopen("./input.txt","r");
	//model domain
	fscanf(fpin,"%d %s\n",&lattice_size_x,temp);
	fscanf(fpin,"%d %s\n",&lattice_size_y,temp);
	fscanf(fpin,"%lf %s\n",&deltax,temp);
	//initial soilthickness
	fscanf(fpin,"%lf %s\n",&initialsoilupland,temp);
	fscanf(fpin,"%lf %s\n",&initialsoilbasin,temp);
	//binary flags to toggle between modes
	fscanf(fpin,"%d %s\n",&sedimentfluxdrivenincisionmode,temp);
	fscanf(fpin,"%d %s\n",&structuralcontrolmode,temp);
	fscanf(fpin,"%d %s\n",&spatiallyvariableupliftmode,temp);
	fscanf(fpin,"%d %s\n",&rillspacingmode,temp);
	//tectonic forcing
	fscanf(fpin,"%lf %s\n",&U_v,temp);
	fscanf(fpin,"%lf %s\n",&U_h,temp);
	fscanf(fpin,"%lf %s\n",&horizontalslipvector,temp);
	//weathering and colluvial transport parameters 
	fscanf(fpin,"%lf %s\n",&P_0,temp); 
	fscanf(fpin,"%lf %s\n",&h_0,temp);
	fscanf(fpin,"%lf %s\n",&d_c,temp);
	fscanf(fpin,"%lf %s\n",&D,temp); 
	fscanf(fpin,"%lf %s\n",&S_c,temp); 
	fscanf(fpin,"%lf %s\n",&S_d,temp); 
	fscanf(fpin,"%lf %s\n",&microtopographiccurvature,temp);
	fscanf(fpin,"%lf %s\n",&rillspacing,temp);
	fscanf(fpin,"%lf %s\n",&soilporosity,temp);  
	//fluvial transport parameters
	fscanf(fpin,"%lf %s\n",&K,temp); 
	fscanf(fpin,"%lf %s\n",&erodibilitycontrast,temp); 
	fscanf(fpin,"%lf %s\n",&entrainmentthreshold,temp); 
	fscanf(fpin,"%lf %s\n",&depositionvelocity,temp); 
	fscanf(fpin,"%lf %s\n",&p,temp);
	fscanf(fpin,"%lf %s\n",&n,temp);
	//hydrologic parameters defining characteristic discharge event
	fscanf(fpin,"%lf %s\n",&characteristicrunoffrate,temp);
    fscanf(fpin,"%lf %s\n",&manningsn,temp);
	//model control parameters
	fscanf(fpin,"%lf %s\n",&duration,temp);
	fscanf(fpin,"%lf %s\n",&initFluvialProcs,temp);
	fscanf(fpin,"%lf %s\n",&timestep,temp);
	fscanf(fpin,"%lf %s\n",&timestepfactorincrease,temp);
	fscanf(fpin,"%lf %s\n",&printinterval,temp);
	rhobedrockoverrhosoil=1/(1-soilporosity);
	horizontalslipvector*=PI/180;
	oneoverdeltax=1/deltax;oneoverdeltax2=1/(deltax*deltax);
	sqrfractionnearS_c=SQR(fractionnearS_c);nearS_c=fractionnearS_c*S_c;sqrnearS_c=SQR(nearS_c);sqrS_c=SQR(S_c);
	time=0;
	counter=1;
    definevectorsandmatrices();
	defineinitialelevationupliftandboundaryconditions();
	hydrologiccorrection(); 
	for (j=1;j<=lattice_size_y;j++)
	 for (i=1;i<=lattice_size_x;i++)
	  {topoinitial[i][j]=topo[i][j];
       topoold[i][j]=topo[i][j];
       soilold[i][j]=soil[i][j];
	   draindiri[i][j]=i;
	   draindirj[i][j]=j;
	   timetonextcliffretreatold[i][j]=timetonextcliffretreat[i][j];}
}

void calculatecrosssectionalcurvature(int i, int j)
/* calculate curvature in the direction perpendicular to the dominant discharge direction */
{   
	if (((draindiri[i][j]==iup[i][j])||(draindiri[i][j]==idown[i][j]))&&(draindirj[i][j]==j)) curv[i][j]=(topo[i][jup[i][j]]+topo[i][jdown[i][j]]-2*topo[i][j])*oneoverdeltax2;
	if (((draindirj[i][j]==jup[i][j])||(draindirj[i][j]==jdown[i][j]))&&(draindiri[i][j]==i)) curv[i][j]=(topo[iup[i][j]][j]+topo[idown[i][j]][j]-2*topo[i][j])*oneoverdeltax2;
	if ((draindiri[i][j]==iup[i][j])&&(draindirj[i][j]==jup[i][j])||(draindiri[i][j]==idown[i][j])&&(draindirj[i][j]==jdown[i][j])) curv[i][j]=(topo[idown[i][j]][jup[i][j]]+topo[iup[i][j]][jdown[i][j]]-2*topo[i][j])*oneoverdeltax2*oneoversqrt2*oneoversqrt2;
	if ((draindiri[i][j]==iup[i][j])&&(draindirj[i][j]==jdown[i][j])||(draindiri[i][j]==idown[i][j])&&(draindirj[i][j]==jup[i][j])) curv[i][j]=(topo[iup[i][j]][jup[i][j]]+topo[idown[i][j]][jdown[i][j]]-2*topo[i][j])*oneoverdeltax2*oneoversqrt2*oneoversqrt2;
}

void mfdflowroute(int i, int j)
/* computes contributing area using MFD algorithm of Freeman (1991), pass 1 is for the model DEM and pass 2 is for bilinearly interpolated model DEM */
{   
    tot=0;
    if (pass==1)
	 {if (iup[i][j]!=i) {if (topo[i][j]>topo[iup[i][j]][j]) tot+=pow(topo[i][j]-topo[iup[i][j]][j],mfdweight);} else tot+=flow1[idown[i][j]][j];
	  if (idown[i][j]!=i) {if (topo[i][j]>topo[idown[i][j]][j]) tot+=pow(topo[i][j]-topo[idown[i][j]][j],mfdweight);} else tot+=flow2[iup[i][j]][j];	  
	  if (jup[i][j]!=j) {if (topo[i][j]>topo[i][jup[i][j]]) tot+=pow(topo[i][j]-topo[i][jup[i][j]],mfdweight);} else tot+=flow3[i][jdown[i][j]];	
	  if (jdown[i][j]!=j) {if (topo[i][j]>topo[i][jdown[i][j]]) tot+=pow(topo[i][j]-topo[i][jdown[i][j]],mfdweight);} else tot+=flow4[i][jup[i][j]];
	  if ((iup[i][j]!=i)||(jup[i][j]!=j)) {if (topo[i][j]>topo[iup[i][j]][jup[i][j]]) tot+=pow((topo[i][j]-topo[iup[i][j]][jup[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow5[idown[i][j]][jdown[i][j]];
	  if ((iup[i][j]!=i)||(jdown[i][j]!=j)) {if (topo[i][j]>topo[iup[i][j]][jdown[i][j]]) tot+=pow((topo[i][j]-topo[iup[i][j]][jdown[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow6[idown[i][j]][jup[i][j]];
	  if ((idown[i][j]!=i)||(jup[i][j]!=j)) {if (topo[i][j]>topo[idown[i][j]][jup[i][j]]) tot+=pow((topo[i][j]-topo[idown[i][j]][jup[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow7[iup[i][j]][jdown[i][j]];
	  if ((idown[i][j]!=i)||(jdown[i][j]!=j)) {if (topo[i][j]>topo[idown[i][j]][jdown[i][j]]) tot+=pow((topo[i][j]-topo[idown[i][j]][jdown[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow8[iup[i][j]][jup[i][j]];
      if (topo[i][j]>topo[iup[i][j]][j]) flow1[i][j]=pow(topo[i][j]-topo[iup[i][j]][j],mfdweight)/tot; else flow1[i][j]=0;
	  if (topo[i][j]>topo[idown[i][j]][j]) flow2[i][j]=pow(topo[i][j]-topo[idown[i][j]][j],mfdweight)/tot; else flow2[i][j]=0;
	  if (topo[i][j]>topo[i][jup[i][j]]) flow3[i][j]=pow(topo[i][j]-topo[i][jup[i][j]],mfdweight)/tot; else flow3[i][j]=0;
	  if (topo[i][j]>topo[i][jdown[i][j]]) flow4[i][j]=pow(topo[i][j]-topo[i][jdown[i][j]],mfdweight)/tot; else flow4[i][j]=0;
	  if (topo[i][j]>topo[iup[i][j]][jup[i][j]]) flow5[i][j]=pow((topo[i][j]-topo[iup[i][j]][jup[i][j]])*oneoversqrt2,mfdweight)/tot; else flow5[i][j]=0;
	  if (topo[i][j]>topo[iup[i][j]][jdown[i][j]]) flow6[i][j]=pow((topo[i][j]-topo[iup[i][j]][jdown[i][j]])*oneoversqrt2,mfdweight)/tot; else flow6[i][j]=0;
	  if (topo[i][j]>topo[idown[i][j]][jup[i][j]]) flow7[i][j]=pow((topo[i][j]-topo[idown[i][j]][jup[i][j]])*oneoversqrt2,mfdweight)/tot; else flow7[i][j]=0;
	  if (topo[i][j]>topo[idown[i][j]][jdown[i][j]]) flow8[i][j]=pow((topo[i][j]-topo[idown[i][j]][jdown[i][j]])*oneoversqrt2,mfdweight)/tot; else flow8[i][j]=0;
	  area[iup[i][j]][j]+=area[i][j]*flow1[i][j];area[idown[i][j]][j]+=area[i][j]*flow2[i][j];area[i][jup[i][j]]+=area[i][j]*flow3[i][j];area[i][jdown[i][j]]+=area[i][j]*flow4[i][j];area[iup[i][j]][jup[i][j]]+=area[i][j]*flow5[i][j];area[iup[i][j]][jdown[i][j]]+=area[i][j]*flow6[i][j];area[idown[i][j]][jup[i][j]]+=area[i][j]*flow7[i][j];area[idown[i][j]][jdown[i][j]]+=area[i][j]*flow8[i][j];
	  meanbasinslope[iup[i][j]][j]+=meanbasinslope[i][j]*flow1[i][j];meanbasinslope[idown[i][j]][j]+=meanbasinslope[i][j]*flow2[i][j]; meanbasinslope[i][jup[i][j]]+=meanbasinslope[i][j]*flow3[i][j];meanbasinslope[i][jdown[i][j]]+=meanbasinslope[i][j]*flow4[i][j];meanbasinslope[iup[i][j]][jup[i][j]]+=meanbasinslope[i][j]*flow5[i][j];meanbasinslope[iup[i][j]][jdown[i][j]]+=meanbasinslope[i][j]*flow6[i][j];meanbasinslope[idown[i][j]][jup[i][j]]+=meanbasinslope[i][j]*flow7[i][j];meanbasinslope[idown[i][j]][jdown[i][j]]+=meanbasinslope[i][j]*flow8[i][j];}
	else 
	 {if (iup2[i][j]!=i) {if (topo2[i][j]>topo2[iup2[i][j]][j]) tot+=pow(topo2[i][j]-topo2[iup2[i][j]][j],mfdweight);} else tot+=flow12[idown2[i][j]][j];
	  if (idown2[i][j]!=i) {if (topo2[i][j]>topo2[idown2[i][j]][j]) tot+=pow(topo2[i][j]-topo2[idown2[i][j]][j],mfdweight);} else tot+=flow22[iup2[i][j]][j];	  
	  if (jup2[i][j]!=j) {if (topo2[i][j]>topo2[i][jup2[i][j]]) tot+=pow(topo2[i][j]-topo2[i][jup2[i][j]],mfdweight);} else tot+=flow32[i][jdown2[i][j]];	
	  if (jdown2[i][j]!=j) {if (topo2[i][j]>topo2[i][jdown2[i][j]]) tot+=pow(topo2[i][j]-topo2[i][jdown2[i][j]],mfdweight);} else tot+=flow42[i][jup2[i][j]];
	  if ((iup2[i][j]!=i)||(jup2[i][j]!=j)) {if (topo2[i][j]>topo2[iup2[i][j]][jup2[i][j]]) tot+=pow((topo2[i][j]-topo2[iup2[i][j]][jup2[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow52[idown2[i][j]][jdown2[i][j]];
	  if ((iup2[i][j]!=i)||(jdown2[i][j]!=j)) {if (topo2[i][j]>topo2[iup2[i][j]][jdown2[i][j]]) tot+=pow((topo2[i][j]-topo2[iup2[i][j]][jdown2[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow62[idown2[i][j]][jup2[i][j]];
	  if ((idown2[i][j]!=i)||(jup2[i][j]!=j)) {if (topo2[i][j]>topo2[idown2[i][j]][jup2[i][j]]) tot+=pow((topo2[i][j]-topo2[idown2[i][j]][jup2[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow72[iup2[i][j]][jdown2[i][j]];
	  if ((idown2[i][j]!=i)||(jdown2[i][j]!=j)) {if (topo2[i][j]>topo2[idown2[i][j]][jdown2[i][j]]) tot+=pow((topo2[i][j]-topo2[idown2[i][j]][jdown2[i][j]])*oneoversqrt2,mfdweight);} else tot+=flow82[iup2[i][j]][jup2[i][j]];
	  if (topo2[i][j]>topo2[iup2[i][j]][j]) flow12[i][j]=pow(topo2[i][j]-topo2[iup2[i][j]][j],mfdweight)/tot; else flow12[i][j]=0;
	  if (topo2[i][j]>topo2[idown2[i][j]][j]) flow22[i][j]=pow(topo2[i][j]-topo2[idown2[i][j]][j],mfdweight)/tot; else flow22[i][j]=0;
	  if (topo2[i][j]>topo2[i][jup2[i][j]]) flow32[i][j]=pow(topo2[i][j]-topo2[i][jup2[i][j]],mfdweight)/tot; else flow32[i][j]=0;
	  if (topo2[i][j]>topo2[i][jdown2[i][j]]) flow42[i][j]=pow(topo2[i][j]-topo2[i][jdown2[i][j]],mfdweight)/tot; else flow42[i][j]=0;
	  if (topo2[i][j]>topo2[iup2[i][j]][jup2[i][j]]) flow52[i][j]=pow((topo2[i][j]-topo2[iup2[i][j]][jup2[i][j]])*oneoversqrt2,mfdweight)/tot; else flow52[i][j]=0;
	  if (topo2[i][j]>topo2[iup2[i][j]][jdown2[i][j]]) flow62[i][j]=pow((topo2[i][j]-topo2[iup2[i][j]][jdown2[i][j]])*oneoversqrt2,mfdweight)/tot; else flow62[i][j]=0;
	  if (topo2[i][j]>topo2[idown2[i][j]][jup2[i][j]]) flow72[i][j]=pow((topo2[i][j]-topo2[idown2[i][j]][jup2[i][j]])*oneoversqrt2,mfdweight)/tot; else flow72[i][j]=0;
	  if (topo2[i][j]>topo2[idown2[i][j]][jdown2[i][j]]) flow82[i][j]=pow((topo2[i][j]-topo2[idown2[i][j]][jdown2[i][j]])*oneoversqrt2,mfdweight)/tot; else flow82[i][j]=0;	
      area2cp[iup2[i][j]][j]+=area2cp[i][j]*flow12[i][j];area2cp[idown2[i][j]][j]+=area2cp[i][j]*flow22[i][j];area2cp[i][jup2[i][j]]+=area2cp[i][j]*flow32[i][j];area2cp[i][jdown2[i][j]]+=area2cp[i][j]*flow42[i][j];area2cp[iup2[i][j]][jup2[i][j]]+=area2cp[i][j]*flow52[i][j];area2cp[iup2[i][j]][jdown2[i][j]]+=area2cp[i][j]*flow62[i][j];area2cp[idown2[i][j]][jup2[i][j]]+=area2cp[i][j]*flow72[i][j];area2cp[idown2[i][j]][jdown2[i][j]]+=area2cp[i][j]*flow82[i][j];}
}

void identifyupslopevalleybottom(int i, int j)
/* identifies the downslope nearest neighbor with the largest contributing area */
{
	maxarea=0;icup=i;jcup=j;draindiriup[i][j]=i;draindirjup[i][j]=j;draindirdistanceup[i][j]=1;
	if ((area[iup[i][j]][j]>maxarea)&&(topo[iup[i][j]][j]>topo[i][j])&&(BCmask[iup[i][j]][j]!=5)) {icup=iup[i][j];jcup=j;maxarea=area[iup[i][j]][j];draindiriup[i][j]=iup[i][j];draindirjup[i][j]=j;}
	if ((area[idown[i][j]][j]>maxarea)&&(topo[idown[i][j]][j]>topo[i][j])&&(BCmask[idown[i][j]][j]!=5)) {icup=idown[i][j];jcup=j;maxarea=area[idown[i][j]][j];draindiriup[i][j]=idown[i][j];draindirjup[i][j]=j;} 
	if ((area[i][jup[i][j]]>maxarea)&&(topo[i][jup[i][j]]>topo[i][j])&&(BCmask[i][jup[i][j]]!=5)) {icup=i;jcup=jup[i][j];maxarea=area[i][jup[i][j]];draindiriup[i][j]=i;draindirjup[i][j]=jup[i][j];} 
	if ((area[i][jdown[i][j]]>maxarea)&&(topo[i][jdown[i][j]]>topo[i][j])&&(BCmask[i][jdown[i][j]]!=5)) {icup=i;jcup=jdown[i][j];maxarea=area[i][jdown[i][j]];draindiriup[i][j]=i;draindirjup[i][j]=jdown[i][j];} 
	if ((area[iup[i][j]][jup[i][j]]>maxarea)&&(topo[iup[i][j]][jup[i][j]]>topo[i][j])&&(BCmask[iup[i][j]][jup[i][j]]!=5)) {icup=iup[i][j];jcup=jup[i][j];maxarea=area[iup[i][j]][jup[i][j]];draindiriup[i][j]=iup[i][j];draindirjup[i][j]=jup[i][j];draindirdistanceup[i][j]=sqrt2;} 
	if ((area[iup[i][j]][jdown[i][j]]>maxarea)&&(topo[iup[i][j]][jdown[i][j]]>topo[i][j])&&(BCmask[iup[i][j]][jdown[i][j]]!=5)) {icup=iup[i][j];jcup=jdown[i][j];maxarea=area[iup[i][j]][jdown[i][j]];draindiriup[i][j]=iup[i][j];draindirjup[i][j]=jdown[i][j];draindirdistanceup[i][j]=sqrt2;} 
	if ((area[idown[i][j]][jup[i][j]]>maxarea)&&(topo[idown[i][j]][jup[i][j]]>topo[i][j])&&(BCmask[idown[i][j]][jup[i][j]]!=5)) {icup=idown[i][j];jcup=jup[i][j];maxarea=area[idown[i][j]][jup[i][j]];draindiriup[i][j]=idown[i][j];draindirjup[i][j]=jup[i][j];draindirdistanceup[i][j]=sqrt2;} 
	if ((area[idown[i][j]][jdown[i][j]]>maxarea)&&(topo[idown[i][j]][jdown[i][j]]>topo[i][j])&&(BCmask[idown[i][j]][jdown[i][j]]!=5)) {icup=idown[i][j];jcup=jdown[i][j];maxarea=area[idown[i][j]][jdown[i][j]];draindiriup[i][j]=idown[i][j];draindirjup[i][j]=jdown[i][j];draindirdistanceup[i][j]=sqrt2;}
}

void mapfloodplainupslope(int i, int j)
/* adds floodplains to existing floodplains in the basin in areas with distributary drainage */
{
	if ((area[draindiri[i][j]][draindirj[i][j]]/area[i][j]<1)&&(hillslopevsconfinedvalleyvsfloodplain[i][j]!=2))
	 {hillslopevsconfinedvalleyvsfloodplain[i][j]=2;
      if ((topo[iup[i][j]][j]>topo[i][j])&&(BCmask[iup[i][j]][j]!=5)) mapfloodplainupslope(iup[i][j],j);
      if ((topo[idown[i][j]][j]>topo[i][j])&&(BCmask[idown[i][j]][j]!=5)) mapfloodplainupslope(idown[i][j],j);
      if ((topo[i][jup[i][j]]>topo[i][j])&&(BCmask[i][jup[i][j]]!=5)) mapfloodplainupslope(i,jup[i][j]);
      if ((topo[i][jdown[i][j]]>topo[i][j])&&(BCmask[i][jdown[i][j]]!=5)) mapfloodplainupslope(i,jdown[i][j]);
      if ((topo[iup[i][j]][jup[i][j]]>topo[i][j])&&(BCmask[iup[i][j]][jup[i][j]]!=5)) mapfloodplainupslope(iup[i][j],jup[i][j]);
      if ((topo[iup[i][j]][jdown[i][j]]>topo[i][j])&&(BCmask[iup[i][j]][jdown[i][j]]!=5)) mapfloodplainupslope(iup[i][j],jdown[i][j]);
      if ((topo[idown[i][j]][jup[i][j]]>topo[i][j])&&(BCmask[idown[i][j]][jup[i][j]]!=5)) mapfloodplainupslope(idown[i][j],jup[i][j]);
	  if ((topo[idown[i][j]][jdown[i][j]]>topo[i][j])&&(BCmask[idown[i][j]][jdown[i][j]]!=5)) mapfloodplainupslope(idown[i][j],jdown[i][j]);}	  
}

void calculatecontributingareaandvalleybottomwidth()
/* calculates contributing area and valleybottom width */
{    int i,j,t;
     float max;

	 for (pass=1;pass<=2;pass++)
	  if (pass==1)
	   {for (j=1;j<=lattice_size_y;j++)
	     for (i=1;i<=lattice_size_x;i++)
		  {area[i][j]=deltax*deltax;
	       topo2[i][j]=topo[i][j];
		   frac[i][j]=1;
		   hillslopevsconfinedvalleyvsfloodplain[i][j]=0;
		   topovec[(j-1)*lattice_size_x+i]=topo[i][j];}
		indexx(lattice_size_x*lattice_size_y,topovec,topovecind);
		t=lattice_size_x*lattice_size_y+1;
	    while (t>1)
         {t--;
          i=(topovecind[t])%lattice_size_x;
          if (i==0) i=lattice_size_x;
          j=(topovecind[t])/lattice_size_x+1;
          if (i==lattice_size_x) j--; 
		  mfdflowroute(i,j);}}
	  else
	    {for (j=1;j<=lattice_size_y;j++)
		  for (i=1;i<=lattice_size_x;i++)
		   {if ((i<lattice_size_x)&&(j<lattice_size_y)) topo2[i*2][j*2]=0.5*(topo[i+1][j+1]+topo[i][j]);
            if (j<lattice_size_y) topo2[i*2-1][j*2]=0.5*(topo[i][j+1]+topo[i][j]);
			if (i<lattice_size_x) topo2[i*2][j*2-1]=0.5*(topo[i+1][j]+topo[i][j]);
			topo2[i*2-1][j*2-1]=topo[i][j];}
		 for (j=1;j<=2*lattice_size_y;j++) for (i=1;i<=2*lattice_size_x;i++) area2cp[i][j]=0.25*deltax*deltax;
	     for (j=1;j<=2*lattice_size_y;j++) for (i=1;i<=2*lattice_size_x;i++) topovec2[(j-1)*2*lattice_size_x+i]=topo2[i][j];
         indexx(4*lattice_size_x*lattice_size_y,topovec2,topovecind2);
	     t=4*lattice_size_x*lattice_size_y+1;
	     while (t>1)
          {t--;
           i=(topovecind2[t])%(2*lattice_size_x);
           if (i==0) i=2*lattice_size_x;
           j=(topovecind2[t])/(2*lattice_size_x)+1;
           if (i==2*lattice_size_x) j--; 
		   mfdflowroute(i,j);}
	     for (j=1;j<=lattice_size_y;j++)
          for (i=1;i<=lattice_size_x;i++)
	       {max=area2cp[(i-1)*2+1][(j-1)*2+1];
	        if ((i>1)&&(j>1)&&(i<lattice_size_x)&&(j<lattice_size_y)) if (max<area2cp[(i-1)*2][(j-1)*2+1]) max=area2cp[(i-1)*2][(j-1)*2+1];
		    if ((i>1)&&(j>1)&&(i<lattice_size_x)&&(j<lattice_size_y)) if (max<area2cp[(i-1)*2+1][(j-1)*2]) max=area2cp[(i-1)*2+1][(j-1)*2];
		    if ((i>1)&&(j>1)&&(i<lattice_size_x)&&(j<lattice_size_y)) if (max<area2cp[(i-1)*2+2][(j-1)*2+1]) max=area2cp[(i-1)*2+2][(j-1)*2+1];
		    if ((i>1)&&(j>1)&&(i<lattice_size_x)&&(j<lattice_size_y)) if (max<area2cp[(i-1)*2+1][(j-1)*2+2]) max=area2cp[(i-1)*2+1][(j-1)*2+2];
		    area2c[i][j]=max;}}
	 t=lattice_size_x*lattice_size_y+1;
	 while (t>0)
      {t--;
       i=(topovecind[t])%lattice_size_x;
       if (i==0) i=lattice_size_x;
       j=(topovecind[t])/lattice_size_x+1;
       if (i==lattice_size_x) j--;
	   calculateslopeD8(i,j);
       calculatecrosssectionalcurvature(i,j);
	   if (area[i][j]/area2c[i][j]>2) frac[i][j]=2; else if (area[i][j]/area2c[i][j]<1) frac[i][j]=1; else frac[i][j]=area[i][j]/area2c[i][j];
       if ((frac[i][j]<fthreshold)&&(hillslopevsconfinedvalleyvsfloodplain[i][j]==0)) hillslopevsconfinedvalleyvsfloodplain[i][j]=1; 
	   if (hillslopevsconfinedvalleyvsfloodplain[i][j]==1) hillslopevsconfinedvalleyvsfloodplain[draindiri[i][j]][draindirj[i][j]]=1;		
	   if ((area[draindiri[i][j]][draindirj[i][j]]/area[i][j]<1)&&(Uinitial[i][j]<small)) 
	    {hillslopevsconfinedvalleyvsfloodplain[draindiri[i][j]][draindirj[i][j]]=2; 
		 mapfloodplainupslope(i,j);}
	   if (hillslopevsconfinedvalleyvsfloodplain[i][j]==2) {if (flow1[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[iup[i][j]][j]=2;if (flow2[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[idown[i][j]][j]=2;if (flow3[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[i][jup[i][j]]=2;if (flow4[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[i][jdown[i][j]]=2;if (flow5[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[iup[i][j]][jup[i][j]]=2;if (flow6[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[iup[i][j]][jdown[i][j]]=2;if (flow7[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[idown[i][j]][jup[i][j]]=2;if (flow8[i][j]>0) hillslopevsconfinedvalleyvsfloodplain[idown[i][j]][jdown[i][j]]=2;}		  
	   minareaneighbor=large;maxareaneighbor=0;
	   if (area[iup[i][j]][j]<minareaneighbor) minareaneighbor=area[iup[i][j]][j];if (area[idown[i][j]][j]<minareaneighbor) minareaneighbor=area[idown[i][j]][j];if (area[i][jup[i][j]]<minareaneighbor) minareaneighbor=area[i][jup[i][j]];if (area[i][jdown[i][j]]<minareaneighbor) minareaneighbor=area[i][jdown[i][j]];if (area[iup[i][j]][jup[i][j]]<minareaneighbor) minareaneighbor=area[iup[i][j]][jup[i][j]];if (area[iup[i][j]][jdown[i][j]]<minareaneighbor) minareaneighbor=area[iup[i][j]][jdown[i][j]];if (area[idown[i][j]][jup[i][j]]<minareaneighbor) minareaneighbor=area[idown[i][j]][jup[i][j]];if (area[idown[i][j]][jdown[i][j]]<minareaneighbor) minareaneighbor=area[idown[i][j]][jdown[i][j]];
	   if (area[iup[i][j]][j]>maxareaneighbor) maxareaneighbor=area[iup[i][j]][j];if (area[idown[i][j]][j]>maxareaneighbor) maxareaneighbor=area[idown[i][j]][j];if (area[i][jup[i][j]]>maxareaneighbor) maxareaneighbor=area[i][jup[i][j]];if (area[i][jdown[i][j]]>maxareaneighbor) maxareaneighbor=area[i][jdown[i][j]];if (area[iup[i][j]][jup[i][j]]>maxareaneighbor) maxareaneighbor=area[iup[i][j]][jup[i][j]];if (area[iup[i][j]][jdown[i][j]]>maxareaneighbor) maxareaneighbor=area[iup[i][j]][jdown[i][j]];if (area[idown[i][j]][jup[i][j]]>maxareaneighbor) maxareaneighbor=area[idown[i][j]][jup[i][j]];if (area[idown[i][j]][jdown[i][j]]>maxareaneighbor) maxareaneighbor=area[idown[i][j]][jdown[i][j]];
	   if (maxareaneighbor/minareaneighbor>thresholdarearatiotomakeincisedvalley) hillslopevsconfinedvalleyvsfloodplain[i][j]==1;
       if (curv[i][j]<microtopographiccurvature) curv[i][j]=microtopographiccurvature;
	   xsectslope=deltax*curv[i][j]/2;
	   width[i][j]=2*pow(area[i][j]*characteristicrunoffrate*manningsn*pow(1+1/pow(xsectslope,2.),onethird)/pow(xsectslope,1.5),0.375);
	   if (hillslopevsconfinedvalleyvsfloodplain[i][j]==0) 
	    {if (rillspacingmode==1) rillspacing=1/(slope[i][j]*microtopographiccurvature);
		 if (rillspacing>deltax) rillspacing=deltax;
		 area[i][j]*=rillspacing/deltax;}
	   if (hillslopevsconfinedvalleyvsfloodplain[i][j]==2) width[i][j]=deltax;
	   discharge[i][j]=area[i][j]*characteristicrunoffrate;}
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) meanbasinslope[i][j]*=deltax*deltax/area[i][j];
}

void fluviallyentrainanddepositsediment()
/* computes erosion and deposition by fluvial processes and bank retreat */ 
{   
    t=lattice_size_x*lattice_size_y+1;
	maxvelocity=0;
	for (j=1;j<=lattice_size_y;j++) 
	 for (i=1;i<=lattice_size_x;i++) 
	  {erodedvolumerate[i][j]=0;
       calculateslopeD8(i,j);
	   identifyupslopevalleybottom(i,j);}
	while (t>1)
     {t--;
      i=(topovecind[t])%lattice_size_x;
      if (i==0) i=lattice_size_x;
      j=(topovecind[t])/lattice_size_x+1;
      if (i==lattice_size_x) j--;
	  fup=0;fdown=0;
	  if (((BCmask[i][j]==0)||(BCmask[i][j]==3)||(BCmask[i][j]==4))&&(divideadjacenttomaskedge[i][j]==0))
	   {velocity=erosionanddepositionfunction(i,j);
        if (divideadjacenttomaskedge[i][j]==1) velocity=0;
        if (maxvelocity<fabs(velocity)) maxvelocity=fabs(velocity);
		if (velocity>0) 
		 {icdown=draindiri[i][j];jcdown=draindirj[i][j];
		  icdowndown=draindiri[icdown][jcdown];jcdowndown=draindirj[icdown][jcdown];
		  icup=draindiriup[i][j];jcup=draindirjup[i][j];
		  distbetweenup=draindirdistance[icup][jcup]*deltax;
		  distbetween=draindirdistance[i][j]*deltax;
		  distbetweendown=draindirdistance[draindiri[i][j]][draindirj[i][j]]*deltax;}
	    else 
		 {icdown=draindiriup[i][j];jcdown=draindirjup[i][j];
		  icdowndown=draindiriup[icdown][jcdown];jcdowndown=draindirjup[icdown][jcdown];
		  icup=draindiri[i][j];jcup=draindirj[i][j];
		  distbetweenup=draindirdistance[icup][jcup]*deltax;
		  distbetween=draindirdistance[i][j]*deltax;
		  distbetweendown=draindirdistance[draindiriup[i][j]][draindirjup[i][j]]*deltax;}	 
		topoup=topo[icup][jcup];topodown=topo[icdown][jcdown];topodowndown=topo[icdowndown][jcdowndown];
		if (topoup<small) topoup=topo[ic][jc];
		rdown=(topodowndown-topodown)/(topodown-topo[i][j])*distbetween/distbetweendown;
		if ((rdown>-1*large)&&(rdown<large)); else rdown=1;
		rup=(topodown-topo[i][j])/(topo[i][j]-topoup)*distbetweenup/distbetween;
		if ((rup>-1*large)&&(rup<large)); else rup=1;
		phiup=(rup+fabs(rup))/(1+fabs(rup));
		phidown=(rdown+fabs(rdown))/(1+fabs(rdown));
		fup=-velocity*topo[i][j]-0.5*fabs(velocity)*(1-fabs(velocity)*timestep/distbetween)*phiup*(topoup-topo[i][j]);
	    fdown=-velocity*topodown-0.5*fabs(velocity)*(1-fabs(velocity)*timestep/distbetweendown)*phidown*(topo[i][j]-topodown);}	    
	  if (draindirdistance[i][j]>0) totalerosion=-SIG(velocity)*timestep*oneoverdeltax/draindirdistance[i][j]*(fup-fdown); else totalerosion=0;
	  if (totalerosion<soil[i][j]) 
	   {topoafterfluvial[i][j]=topo[i][j]-totalerosion;
        soilafterfluvial[i][j]=topoafterfluvial[i][j]-bedrock[i][j];
        if (soilafterfluvial[i][j]<0) soilafterfluvial[i][j]=0;		
		erodedvolumerate[i][j]+=deltax*deltax*totalerosion/timestep;}
	  else  
	   {erosionintobedrock=(totalerosion-soil[i][j])*erodibilitycontrast;
		topoafterfluvial[i][j]=bedrock[i][j]-erosionintobedrock;
		soilafterfluvial[i][j]=0;
        erodedvolumerate[i][j]+=deltax*deltax*(soil[i][j]+erosionintobedrock*rhobedrockoverrhosoil)/timestep;
        bedrock[i][j]=topoafterfluvial[i][j];}	
	  if (erodedvolumerate[i][j]<0) erodedvolumerate[i][j]=0;
	  erodedvolumerate[iup[i][j]][j]+=erodedvolumerate[i][j]*flow1[i][j];erodedvolumerate[idown[i][j]][j]+=erodedvolumerate[i][j]*flow2[i][j];erodedvolumerate[i][jup[i][j]]+=erodedvolumerate[i][j]*flow3[i][j];erodedvolumerate[i][jdown[i][j]]+=erodedvolumerate[i][j]*flow4[i][j];erodedvolumerate[iup[i][j]][jup[i][j]]+=erodedvolumerate[i][j]*flow5[i][j];erodedvolumerate[iup[i][j]][jdown[i][j]]+=erodedvolumerate[i][j]*flow6[i][j];erodedvolumerate[idown[i][j]][jup[i][j]]+=erodedvolumerate[i][j]*flow7[i][j];erodedvolumerate[idown[i][j]][jdown[i][j]]+=erodedvolumerate[i][j]*flow8[i][j];}	
}

static double f(double x)
/* function that is solved to calculate the value of topo[i][j] for which colluvial sediment mass is conserved */ 
{   
	colluvialunitsedimentfluxin[i][j]=0;colluvialunitsedimentfluxout[i][j]=0;
	calculateslopealongsteepestquadrant(i,j,topo[i][j],topo[iup[i][j]][j],topo[idown[i][j]][j],topo[i][jup[i][j]],topo[i][jdown[i][j]]);
	cosfactor=sqrt(1+slope[i][j]*slope[i][j]);
	slope1=0;slope2=0;slope3=0;slope4=0;
	if (BCmask[iup[i][j]][j]!=5) slope1=(x-topo[iup[i][j]][j])*oneoverdeltax;if (BCmask[idown[i][j]][j]!=5) slope2=(x-topo[idown[i][j]][j])*oneoverdeltax;if (BCmask[i][jup[i][j]]!=5) slope3=(x-topo[i][jup[i][j]])*oneoverdeltax;if (BCmask[i][jdown[i][j]]!=5) slope4=(x-topo[i][jdown[i][j]])*oneoverdeltax;
	if (slope3>slope4) totslopesqr1=SQR(slope1)+SQR(slope3); else totslopesqr1=SQR(slope1)+SQR(slope4);if (slope3>slope4) totslopesqr2=SQR(slope2)+SQR(slope3); else totslopesqr2=SQR(slope2)+SQR(slope4);if (slope1>slope2) totslopesqr3=SQR(slope3)+SQR(slope1); else totslopesqr3=SQR(slope3)+SQR(slope2);if (slope1>slope2) totslopesqr4=SQR(slope4)+SQR(slope1); else totslopesqr4=SQR(slope4)+SQR(slope2);
	if (slope1>0) colluvialunitsedimentfluxout[i][j]+=colluvialunitsedimentfluxfunction(slope1,totslopesqr1,MAX(x-bedrock[i][j],0),cosfactor); else colluvialunitsedimentfluxin[i][j]+=colluvialunitsedimentfluxfunction(-slope1,totslopesqr1,MAX(x-bedrock[i][j],0),cosfactor);if (slope2>0) colluvialunitsedimentfluxout[i][j]+=colluvialunitsedimentfluxfunction(slope2,totslopesqr2,MAX(x-bedrock[i][j],0),cosfactor); else colluvialunitsedimentfluxin[i][j]+=colluvialunitsedimentfluxfunction(-slope2,totslopesqr2,MAX(x-bedrock[i][j],0),cosfactor);if (slope3>0) colluvialunitsedimentfluxout[i][j]+=colluvialunitsedimentfluxfunction(slope3,totslopesqr3,MAX(x-bedrock[i][j],0),cosfactor); else colluvialunitsedimentfluxin[i][j]+=colluvialunitsedimentfluxfunction(-slope3,totslopesqr3,MAX(x-bedrock[i][j],0),cosfactor);if (slope4>0) colluvialunitsedimentfluxout[i][j]+=colluvialunitsedimentfluxfunction(slope4,totslopesqr4,MAX(x-bedrock[i][j],0),cosfactor); else colluvialunitsedimentfluxin[i][j]+=colluvialunitsedimentfluxfunction(-slope4,totslopesqr4,MAX(x-bedrock[i][j],0),cosfactor);
	if (divideadjacenttomaskedge[i][j]==1) {colluvialunitsedimentfluxout[i][j]*=2;colluvialunitsedimentfluxin[i][j]=0;}
	if (hillslopevsconfinedvalleyvsfloodplain[i][j]==1) return x-topoafterfluvial[i][j]-timestep*(colluvialunitsedimentfluxin[i][j]-colluvialunitsedimentfluxout[i][j])/width[i][j];
	 else return x-topoafterfluvial[i][j]-timestep*(colluvialunitsedimentfluxin[i][j]-colluvialunitsedimentfluxout[i][j])*oneoverdeltax; 
}

void depositlandslidedebris(int i, int j)
/* performs a downslope search along the direction of steepest descent until a location is found where the debris can be deposited, i.e., where slope < S_d */ 
{	 
	calculateslopealongsteepestquadrant(i,j,topo[i][j],topo[iup[i][j]][j],topo[idown[i][j]][j],topo[i][jup[i][j]],topo[i][jdown[i][j]]);
	if ((slope[i][j]>fractionnearS_c*S_d)&&(soil[i][j]>0))
	 {min=topo[i][j];mini=i;minj=j;
	  if ((BCmask[iup[i][j]][j]!=5)&&(topo[iup[i][j]][j]<min)) {min=topo[iup[i][j]][j];mini=iup[i][j];minj=j;}
	  if ((BCmask[idown[i][j]][j]!=5)&&(topo[idown[i][j]][j]<min)) {min=topo[idown[i][j]][j];mini=idown[i][j];minj=j;}
	  if ((BCmask[i][jup[i][j]]!=5)&&(topo[i][jup[i][j]]<min)) {min=topo[i][jup[i][j]];mini=i;minj=jup[i][j];}
	  if ((BCmask[i][jdown[i][j]]!=5)&&(topo[i][jdown[i][j]]<min)) {min=topo[i][jdown[i][j]];mini=i;minj=jdown[i][j];}
	  if ((BCmask[iup[i][j]][jup[i][j]]!=5)&&(topo[iup[i][j]][jup[i][j]]<min)) {min=topo[iup[i][j]][jup[i][j]];mini=iup[i][j];minj=jup[i][j];}
	  if ((BCmask[idown[i][j]][jup[i][j]]!=5)&&(topo[idown[i][j]][jup[i][j]]<min)) {min=topo[idown[i][j]][jup[i][j]];mini=idown[i][j];minj=jup[i][j];}
	  if ((BCmask[iup[i][j]][jdown[i][j]]!=5)&&(topo[iup[i][j]][jdown[i][j]]<min)) {min=topo[iup[i][j]][jdown[i][j]];mini=iup[i][j];minj=jdown[i][j];}
	  if ((BCmask[idown[i][j]][jdown[i][j]]!=5)&&(topo[idown[i][j]][jdown[i][j]]<min)) {min=topo[idown[i][j]][jdown[i][j]];mini=idown[i][j];minj=jdown[i][j];}
	  if (soil[i][j]<fillincrementindebrisblockdeposition) fillincrementl=soil[i][j]; else fillincrementl=fillincrementindebrisblockdeposition;
      topo[i][j]-=fillincrementl;
      soil[i][j]-=fillincrementl;
      if ((BCmask[mini][minj]==0)||(BCmask[mini][minj]==3)||(BCmask[mini][minj]==4))
	   {topo[mini][minj]+=fillincrementl;
	    soil[mini][minj]+=fillincrementl;
	    depositlandslidedebris(mini,minj);}}
} 

void relaxoversteepenedslopestonearS_c()
/* removes soil from any slopes steeper than fractionnearS_c*Sc, and deposits that soil downslope with slope S_d */
{
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) topo[i][j]=topoafterfluvial[i][j];
	t=0;
	while (t<lattice_size_x*lattice_size_y)
     {t++;
      i=(topovecind[t])%lattice_size_x;
      if (i==0) i=lattice_size_x;
      j=(topovecind[t])/lattice_size_x+1;
      if (i==lattice_size_x) j--;
	  if (((BCmask[i][j]==0)||(BCmask[i][j]==3)||(BCmask[i][j]==4))&&(divideadjacenttomaskedge[i][j]==0))
	   {calculateslopealongsteepestquadrant(i,j,topo[i][j],topo[iup[i][j]][j],topo[idown[i][j]][j],topo[i][jup[i][j]],topo[i][jdown[i][j]]);
	    while ((slope[i][j]>nearS_c)&&(soil[i][j]>0))	  
		 {depositlandslidedebris(i,j);
	      calculateslopealongsteepestquadrant(i,j,topo[i][j],topo[iup[i][j]][j],topo[idown[i][j]][j],topo[i][jup[i][j]],topo[i][jdown[i][j]]);}}}
	for (j=1;j<=lattice_size_y;j++) 
	 for (i=1;i<=lattice_size_x;i++) 
	  {topoafterfluvial[i][j]=topo[i][j];
	   topovec[(j-1)*lattice_size_x+i]=topoafterfluvial[i][j];}
}

void upliftproducesoilandretreatcliffs()
/* perform rock uplift, weather bedrock into soil, and retreat areas > nearS_c with semi-Lagrangian advecton */
{
	for (j=1;j<=lattice_size_y;j++) 
	 for (i=1;i<=lattice_size_x;i++) 
	  if ((BCmask[i][j]!=1)&&(BCmask[i][j]!=2)&&(BCmask[i][j]!=5))
	   {calculateslopealongsteepestquadrant(i,j,topoold[i][j],topoold[iup[i][j]][j],topoold[idown[i][j]][j],topoold[i][jup[i][j]],topoold[i][jdown[i][j]]);
	    cosfactor=sqrt(1+slope[i][j]*slope[i][j]);
	    weatheringrate[i][j]=maximumweatheringratefunction(i,j)*exp(-soil[i][j]/(h_0*cosfactor))*cosfactor;
		if (slope[i][j]<=nearS_c)
		 {bedrock[i][j]+=timestep*(U[i][j]-weatheringrate[i][j]);
	      if (bedrock[i][j]<baselevel) bedrock[i][j]=baselevel;
	      soil[i][j]+=timestep*rhobedrockoverrhosoil*weatheringrate[i][j];
		  topo[i][j]=bedrock[i][j]+soil[i][j];}
        else 
		 {bedrock[i][j]+=timestep*U[i][j];
	      topo[i][j]=bedrock[i][j];
	      timetonextcliffretreat[i][j]-=timestep;
		  if ((timetonextcliffretreat[i][j]<tiny)&&(divideadjacenttomaskedge[i][j]==0)) 
		   {numberofavalancheincrements=(int)((topo[i][j]-bedrock[draindiri[i][j]][draindirj[i][j]])/fillincrementindebrisblockdeposition);
            fillincrementl=(topo[i][j]-bedrock[draindiri[i][j]][draindirj[i][j]])/numberofavalancheincrements;
	        topo[i][j]=bedrock[draindiri[i][j]][draindirj[i][j]];
			soil[i][j]=0;
			bedrock[i][j]=topo[i][j];
			for (steps=1;steps<=numberofavalancheincrements;steps++) 
			 {topo[i][j]+=fillincrementindebrisblockdeposition;
              soil[i][j]+=fillincrementindebrisblockdeposition;         
			  depositlandslidedebris(i,j);}
			calculateslopealongsteepestquadrant(iup[i][j],j,topoold[iup[i][j]][j],topoold[iup[iup[i][j]][j]][j],topoold[i][j],topoold[iup[i][j]][jup[i][j]],topoold[iup[i][j]][jdown[i][j]]);
			if ((draindiri[iup[i][j]][j]==i)&&(draindirj[iup[i][j]][j]==j)) timetonextcliffretreat[iup[i][j]][j]-=timetonextcliffretreat[i][j];		
			calculateslopealongsteepestquadrant(idown[i][j],j,topoold[idown[i][j]][j],topoold[i][j],topoold[idown[idown[i][j]][j]][j],topoold[idown[i][j]][jup[i][j]],topoold[idown[i][j]][jdown[i][j]]);
			if ((draindiri[idown[i][j]][j]==i)&&(draindirj[idown[i][j]][j]==j)) timetonextcliffretreat[idown[i][j]][j]-=timetonextcliffretreat[i][j];
			calculateslopealongsteepestquadrant(i,jup[i][j],topoold[i][jup[i][j]],topoold[iup[i][j]][jup[i][j]],topoold[idown[i][j]][jup[i][j]],topoold[i][jup[i][jup[i][j]]],topoold[i][j]);
			if ((draindiri[i][jup[i][j]]==i)&&(draindirj[i][jup[i][j]]==j)) timetonextcliffretreat[i][jup[i][j]]-=timetonextcliffretreat[i][j];
			calculateslopealongsteepestquadrant(i,jdown[i][j],topoold[i][jdown[i][j]],topoold[iup[i][j]][jdown[i][j]],topoold[idown[i][j]][jdown[i][j]],topoold[i][j],topoold[i][jdown[i][jdown[i][j]]]);
			if ((draindiri[i][jdown[i][j]]==i)&&(draindirj[i][jdown[i][j]]==j)) timetonextcliffretreat[i][jdown[i][j]]-=timetonextcliffretreat[i][j];
			timetonextcliffretreat[i][j]=deltax/(maximumweatheringratefunction(i,j)*cos(PI/2-atan(slope[i][j])));}}}
}

static double solveformax(double x)
/* if there is soil at the grid point, max is defined by totslopesqr = sqrnearS_c (any higher elevation would result in an oversteepened condition downslope) */
{   
    calculateslopealongsteepestquadrant(i,j,x,topo[iup[i][j]][j],topo[idown[i][j]][j],topo[i][jup[i][j]],topo[i][jdown[i][j]]);
	if (x-bedrock[i][j]>0) return slope[i][j]-nearS_c; else return 0;
}

void solveforelevationthatconservescolluvialfluxes()
/* solves for the value of topo[i][j] for which colluvial sediment mass is conserved */
{
 	sweep=0;resid=large;
	while ((resid>small)&&(sweep<maxsweep)) 
	 {sweep++;resid=0;
	  for (j=1;j<=lattice_size_y;j++) 
	   for (i=1;i<=lattice_size_x;i++) 
	    {topocopy[i][j]=topo[i][j];
	     topovec[(j-1)*lattice_size_x+i]=topo[i][j];}
	  indexx(lattice_size_x*lattice_size_y,topovec,topovecind);
	  t=0;
	  while (t<lattice_size_x*lattice_size_y)
       {t++;
        i=(topovecind[t])%lattice_size_x;
        if (i==0) i=lattice_size_x;
        j=(topovecind[t])/lattice_size_x+1;
        if (i==lattice_size_x) j--;
		calculateslopealongsteepestquadrant(i,j,topo[i][j],topo[iup[i][j]][j],topo[idown[i][j]][j],topo[i][jup[i][j]],topo[i][jdown[i][j]]);
		if ((BCmask[i][j]!=1)&&(BCmask[i][j]!=2)&&(BCmask[i][j]!=5)&&((slope[i][j]<nearS_c)||(soil[i][j]>0)))
		 {max=zbrent(solveformax,topo[i][j],topo[i][j]+maxcolluvialaggradation,tiny);	      
		  topo[i][j]=zbrent(f,bedrock[i][j],max,tiny);		  
		  soil[i][j]=topo[i][j]-bedrock[i][j];
		  if (soil[i][j]<0) soil[i][j]=0; 
		  if (fabs(topo[i][j]-topocopy[i][j])>resid) resid=fabs(topo[i][j]-topocopy[i][j]);}}}	
}

void applygradientbasedboundaryconditions()
/* apply Neumann boundary conditions where prescribed */
{
	i=1;for (j=1;j<=lattice_size_y;j++) if (BCmask[i][j]==2) {soil[i][j]=soil[iup[i][j]][j];topo[i][j]=topo[iup[i][j]][j];bedrock[i][j]=bedrock[iup[i][j]][j];}
	i=lattice_size_x;for (j=1;j<=lattice_size_y;j++) if (BCmask[i][j]==2) {soil[i][j]=soil[idown[i][j]][j];topo[i][j]=topo[idown[i][j]][j];bedrock[i][j]=bedrock[idown[i][j]][j];}
	j=1;for (i=1;i<=lattice_size_x;i++) if (BCmask[i][j]==2) {soil[i][j]=soil[i][jup[i][j]];topo[i][j]=topo[i][jup[i][j]];bedrock[i][j]=bedrock[i][jup[i][j]];}
	j=lattice_size_y;for (i=1;i<=lattice_size_x;i++) if (BCmask[i][j]==2) {soil[i][j]=soil[i][jdown[i][j]];topo[i][j]=topo[i][jdown[i][j]];bedrock[i][j]=bedrock[i][jdown[i][j]];}
    i=1;for (j=1;j<=lattice_size_y;j++) if (BCmask[i][j]==6) {soil[i][j]=soil[iup[i][j]][j];topo[i][j]=2*topo[iup[i][j]][j]-topo[iup[iup[i][j]][j]][j];bedrock[i][j]=2*bedrock[iup[i][j]][j]-bedrock[iup[iup[i][j]][j]][j];}
	i=lattice_size_x;for (j=1;j<=lattice_size_y;j++) if (BCmask[i][j]==6) {soil[i][j]=soil[idown[i][j]][j];topo[i][j]=2*topo[idown[i][j]][j]-topo[idown[idown[i][j]][j]][j];bedrock[i][j]=2*bedrock[idown[i][j]][j]-topo[idown[idown[i][j]][j]][j];}
	j=1;for (i=1;i<=lattice_size_x;i++) if (BCmask[i][j]==6) {soil[i][j]=soil[i][jup[i][j]];topo[i][j]=2*topo[i][jup[i][j]]-topo[i][jup[i][jup[i][j]]];bedrock[i][j]=2*bedrock[i][jup[i][j]]-bedrock[i][jup[i][jup[i][j]]];}
	j=lattice_size_y;for (i=1;i<=lattice_size_x;i++) if (BCmask[i][j]==6) {soil[i][j]=soil[i][jdown[i][j]];topo[i][j]=2*topo[i][jdown[i][j]]-topo[i][jdown[i][jdown[i][j]]];bedrock[i][j]=2*bedrock[i][jdown[i][j]]-topo[i][jdown[i][jdown[i][j]]];}
}

void evolvetopography()
/* solves for the evolution of topography through a time step by 1) soil production, 2) fluvial erosion, and 3) conservation of mass involving colluvial and alluvial fluxes */
{   
	upliftproducesoilandretreatcliffs();
	hydrologiccorrection();
	calculatecontributingareaandvalleybottomwidth();
	fluviallyentrainanddepositsediment();
	relaxoversteepenedslopestonearS_c();
	solveforelevationthatconservescolluvialfluxes();
	applygradientbasedboundaryconditions();
}

void updatetime()
/* updates time or (if the change required too many iterative sweeps or exceeded the prescribed maximum courant number) reject the change and run the time step again */
{   
	mintopo=large;maxtopo=0;  
	if ((maxvelocity*timestep>courantnumber*deltax)||(sweep>=maxsweep))
	 {for (j=1;j<=lattice_size_y;j++)
	   for (i=1;i<=lattice_size_x;i++)
		{topo[i][j]=topoold[i][j];
		 soil[i][j]=soilold[i][j];
		 timetonextcliffretreat[i][j]=timetonextcliffretreatold[i][j];
		 bedrock[i][j]=topo[i][j]-soil[i][j];}
	  timestep/=2;}
	else
	 {time+=timestep;
      for (j=1;j<=lattice_size_y;j++)
	   for (i=1;i<=lattice_size_x;i++)
	    {if (topo[i][j]<mintopo) mintopo=topo[i][j];if (topo[i][j]>maxtopo) maxtopo=topo[i][j];	 		  
		 topoold[i][j]=topo[i][j];
		 soilold[i][j]=soil[i][j];
		 timetonextcliffretreatold[i][j]=timetonextcliffretreat[i][j];}
	  timestep*=timestepfactorincrease;}
}

int main() 
{   
	establishinitialconditions();
	while (time<duration)
	 {if (time+timestep>duration) timestep=duration-time;
	  if (U_h>tiny) lateraladvection(); 
	  evolvetopography();
	  updatetime();
      printf("elapsed time: %f timestep: %f relief: %f\n",time,timestep,maxtopo-mintopo);
	  printfiles();}	  
	return 0; 
}