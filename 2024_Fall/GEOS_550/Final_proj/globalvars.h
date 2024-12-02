/* Globally available variables and definitions */
#ifndef GLOBALVARS_H
#define GLOBALVARS_H

#define FREE_ARG char*
#define NR_END 1
#define stacksize 10000000
#define PI 3.141592653589793
#define sqrt2 1.414213562373
#define oneoversqrt2 0.707106781186
#define onethird 0.333333333333
#define large 1.e+6
#define tiny 1.e-6
#define verytiny 1.e-12
#define mfdweight 1.1
#define thresholdarearatiotomakeincisedvalley 2
#define thresholdvalleybottombankratio 10
#define fractionnearS_c 0.99
#define fthreshold 1.2
#define baselevel 0
#define small 0.01
#define fillincrementinhydrologiccorrection 0.001
#define fillincrementindebrisblockdeposition 0.1
#define maxsweep 5
#define courantnumber 1
#define maxcolluvialaggradation 0.1

static double sqrarg;
#define SIG(a) ((a>0)-(a<0))
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
#define MAX(i, j) (((i) > (j)) ? (i) : (j))
#define MIN(i, j) (((i) < (j)) ? (i) : (j))

extern int sedimentfluxdrivenincisionmode,upslopecount,downslopecount,pass,rillspacingmode,drainagebasinmaskmode,structuralcontrolmode,**BCmask;
extern int **drainagebasinmask,**divideadjacenttomaskedge,**hillslopevsconfinedvalleyvsfloodplain,**faultmask,advected,sweep,counter,routingmethod;
extern int *pitsandflatsis,*pitsandflatsjs,**iup,**idown,**jup,**jdown,**iup2,**idown2,**jup2,**jdown2,**draindiri,**draindirj,**draindiriup;
extern int **draindirjup,residi,residj,i,j,ic,jc,ic2,icup,jcup,icdown,jcdown,icdowndown,jcdowndown,t,count,lattice_size_x,lattice_size_y,*topovecind,*topovecind2,*neighboringareasind;

// Float
extern double minareaneighbor,maxareaneighbor,bankerosion,volumeofbankfailure,bankheight,criticalbankheight,bankretreatrate,**accumulatedbankretreat;
extern double **erodeddepth,**entrainment,**deposition,*topovec,*topovec2,*neighboringareas,soilporosity,depositionvelocity,maxareaup,maxareadown;
extern double meanupslopearea,meandownslopearea,stddevupslopearea,stddevdownslopearea,**discharge,**depositionrate,entrainmentthreshold,**topo2,**area2c;
extern double **area2cp,**frac,rillspacing,microtopographiccurvature,ratiooflargetosmallareas,maxvelocity,sqrfractionnearS_c,nearS_c,timetoremovesoil;
extern double fractionaldistance,mincurvatureforwidthestimation,slopenew,a1,b1,maxtopo,distance,timeremaining,erodibility,maxtimestep,sqrnearS_c,factor;
extern double widthlowest,totalerosion,erosionintobedrock,erodibilitycontrast,slopetrial,totslopesqr1,totslopesqr2,totslopesqr3,totslopesqr4,maxtopo;
extern double mintopo,maxarea,areavalleybottom,topovalleybottom,velocity,dist,distbetweenup,distbetween,distbetweendown,rdown,rup,phidown,phiup,fup;
extern double fdown,topoup,topodown,topodowndown,bankretreatparameter,**alluvialfluxinav,**alluvialfluxoutav,alluvialfluxinavl,alluvialfluxoutavl;
extern double averageuplanderosionrate,totalfluxout,depositedvolume,downslopeneumannbc,**Uinitial,uplandarea,totalerodedupland,totalerodedcolluvial,Pl;
extern double Kl,k,maxim,initialsoilbasin,P_0l,erode,**erodedvolumerate,**erodedvolumeratecopy,**meanbasinslope,**flow1,**flow2,**flow3,**flow4,**flow5;
extern double **flow6,**flow7,**flow8,**flow12,**flow22,**flow32,**flow42,**flow52,**flow62,**flow72,**flow82,**area1,**area2,**area3,**area4,**area5;
extern double **area6,**area7,**area8,slope1,slope2,slope3,slope4,slope5,slope6,slope7,slope8,m,K_0,k_0,**colluvialunitsedimentfluxin,**colluvialunitsedimentfluxout;
extern double **alluvialfluxin,**alluvialfluxout,**erodedvolumerate,**meanbasinslope,**structuralelevation,nextadvectiontime,keeptimestep,horizontalslipvector;
extern double verticaltohorizontalratio,horizontalsliprate,U_0fault,K,advectioninterval,advectiontimestep,advectioninterval,cosfactor,xsectslope,max1,max2,max3;
extern double maxcurv,tot,areai,areaj,down,initialsoilupland,characteristicrunoffrate,manningsn,noiseamplitude,timestepfactorincrease,min,max,printcounter;
extern double printinterval,**weatheringrate,rhobedrockoverrhosoil,n,U_0,K,p,P_0,h_0,d_c,D,S_c,sqrS_c,deltax,oneoverdeltax,oneoverdeltax2,duration,timestep;
extern double time,resid,residlast,**topoinitial,**draindirdistanceup,**draindirdistance,**area,**curv,**width,**soil,**bedrock,**soilold,**slope,**topo;
extern double **topocopy,**topoold,**topoafterDL,**soilafterDL,**U,depthxout1,depthxout2,depthyout1,depthyout2,depthxout,depthxin,depthyout,depthyin,slopexout1;
extern double slopeyout1,slopexout2,slopeyout2,slopexout,slopeyout,slopexin,slopeyin,denomin,denomout,sqrslopein,sqrslopeout, initFluvialProcs;

#endif /* GLOBALVARS_H*/