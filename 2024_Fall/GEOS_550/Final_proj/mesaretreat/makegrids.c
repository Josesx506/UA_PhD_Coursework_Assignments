#include <stdio.h>
#include <stdlib.h>

#define lattice_size_x 101
#define lattice_size_y 101
#define horizontalsliprate 0

int main() 
{   FILE *fp0,*fp1;
    int i,j,faultmask;

	fp0=fopen("./BCmask.txt","w");
	for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) if ((i>1)&&(j>1)&&(i<lattice_size_x)&&(j<lattice_size_y)) fprintf(fp0,"0\n"); else fprintf(fp0,"2\n"); 
	fclose(fp0);
	fp0=fopen("./faultmask.txt","r");
	fp1=fopen("./inputtopo.txt","w");
    for (j=1;j<=lattice_size_y;j++) for (i=1;i<=lattice_size_x;i++) {fscanf(fp0,"%d",&faultmask);if (faultmask>0) fprintf(fp1,"100.0\n"); else fprintf(fp1,"0.0\n");}
    fclose(fp0);
	fclose(fp1);
	return 0; 
}