#include <stdio.h>
#include <stdlib.h>

#define lattice_size_x 101
#define lattice_size_y 101

int main() 
{   FILE *fp0,*fp1;
    int i,j,faultmask;

	fp0=fopen("./BCmask.txt","w");
	for (j=1;j<=lattice_size_y;j++)
	 for (i=1;i<=lattice_size_x;i++)
	  {if ((i>1)&&(j>1)&&(i<lattice_size_x)&&(j<lattice_size_y)) fprintf(fp0,"0\n"); 
        else if ((j==1)||(j==lattice_size_y)) fprintf(fp0,"2\n"); else fprintf(fp0,"3\n");} 
	fclose(fp0);
	fp0=fopen("./faultmask.txt","w");
    for (j=1;j<=lattice_size_y;j++)
	 for (i=1;i<=lattice_size_x;i++)
	  if (j<lattice_size_y/2) fprintf(fp0,"0\n"); else fprintf(fp0,"1\n");
    fclose(fp0);
	return 0; 
}