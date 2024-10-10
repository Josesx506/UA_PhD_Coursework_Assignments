#ifndef USERDEFINEDFUNCTIONS_H
#define USERDEFINEDFUNCTIONS_H

void identifydownslopevalleybottom(int i, int j);
void calculateslopealongsteepestquadrant(int i, int j);
void mapbankfailureregion(int i, int j);
double maximumweatheringratefunction(int i, int j);
double colluvialunitsedimentfluxfunction(double slopel, double totslopesqrl, double soill, double cosfactorl);
double erosionanddepositionfunction(int i, int j);
void bankretreatfunction(int i, int j);

#endif /* USERDEFINEDFUNCTIONS_H*/