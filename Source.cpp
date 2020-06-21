#include "Rand.h"
#include "Input.h"
#include "SVM.h"
#include "MatrixElement.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include <Eigen> 

#include <iomanip>
#include <string>
#include <stdlib.h> 
#include <ctime> 
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

//#include <mpi.h>

using namespace Eigen;
using namespace std;


int main(int argc, char* argv[]) {
clock_t begin = clock();
string jobname = argv[1];  

/* Read and print input data */
Input input("./input/"+jobname+".inp");
input.print();
ifstream  srcc("./input/"+jobname+".inp");
cout << srcc.rdbuf();
ifstream  src("./input/"+jobname+".inp");

/* Initialize the random numbr generator */
Rand rand(input.irand);

/* Initialize SVM  */
printf("\n\t Initialize SVM \n");
SVM svm(rand, input);
vector<vector<MatrixXd> > Basis;
vector<MatrixXd> NewState;

/* Loading exists basis from file */
std::string basisFile = "./output/"+jobname+".basis";
std::ifstream infile(basisFile);
int existBasis=0, itr=0;
if (infile.good()) infile >> existBasis; 

GeneralizedSelfAdjointEigenSolver<MatrixXd> ges;
MatrixXd Norm;
MatrixXd H;
VectorXd D;
double EE;

if (existBasis > 0) {
  cout << "\t Existing basis is used, with " << existBasis << " states." << endl;
  int N = input.npar;
  for (int i = 0; i < existBasis; i++ ){
    vector<MatrixXd> aState;
	  MatrixXd A = MatrixXd::Zero(N, N);
    double aa[N];
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
	      infile >> A(j,k);
    aState.push_back(A);
    MatrixXd B = MatrixXd::Zero(N, N);
    for (int j = 0; j < N; j++)
	    infile >> B(j,j);
    aState.push_back(B);
    MatrixXd s = MatrixXd::Zero(N, N);
    for (int j = 0; j < N; j++)
	    infile >> s(j,j);
    aState.push_back(s);
    Basis.push_back(aState);
	  svm.UpdateNorm(Basis);
	  svm.UpdateHamiltonian(Basis);
    if (i==existBasis-1 || (i+1)%5==0 && i>1) {
      Norm = svm.NormMatrix(Basis);
      H    = svm.HamiltonianMatrix(Basis);
      ges.compute(H, Norm);
      D = ges.eigenvalues();
      double Ei = D.minCoeff();
      printf("\t iter = %4d     E = %14.8f",i+1,Ei);
      cout << endl;
      if (i<existBasis-1) EE=Ei;
    }
  }
  infile.close();
  itr = Basis.size();
  } else {
  cout << "\t Starting a new basis." << endl;
  NewState = svm.FirstNewState();
  if (NewState[0](0, 0) == 2000) {
	  cout << "\t Finding new state with appropriate overlap failed" << endl << endl;
	  return 0;
	}
  itr = 1; 
  cout << "\t First state found." << endl;
  Basis.push_back(NewState);
  svm.UpdateNorm(Basis);
  svm.UpdateHamiltonian(Basis);
}

/* start SVM iterations */
MatrixXd C;
double E, maxE;
double maxEE;
int n_accuracy=1;
int n_dE=0;
// GeneralizedSelfAdjointEigenSolver<MatrixXd> ges;
SelfAdjointEigenSolver<MatrixXd> es;
VectorXd esD;

cout << "\n\t Starting SVM iterations\n" << endl;
while (itr < input.maxbasis) {
  svm.SaveToFile(basisFile,Basis);
  Norm = svm.NormMatrix(Basis);
  H    = svm.HamiltonianMatrix(Basis);
  ges.compute(H, Norm);
  C = ges.eigenvectors();
  D = ges.eigenvalues();
  E = D.minCoeff();
  maxE = D.maxCoeff();    
  if (itr == 1)  EE = E + abs(E / 2);
  if (itr > existBasis) printf("\t iter = %4d     E = %14.8f \n",itr,E);
  fflush(stdout);
  for (int ntries=0; ntries<10; ntries++) {
    NewState = svm.NewState(Basis, C, D, E, EE);
    if (NewState[0](0, 0) != 2000) break;
  }
  if (NewState[0](0, 0) == 2000) {
	  cout << "\t Finding new state with lower energy failed" << endl << endl;
    return 0;
  }
  Basis.push_back(NewState);
  svm.UpdateNorm(Basis);
  svm.UpdateHamiltonian(Basis);
  EE = E; 
  maxEE=maxE;
  if(itr%5==0) {
    cout<<"   more eigenvalues=  ";
    for(int ii=1; ii<4; ii++) {
      cout<<D(ii)<<"  ";
    }
    cout << endl;
  }
  itr = itr + 1; 
}

clock_t end = clock();
double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
cout << "time=  " << elapsed_secs << endl;

return 0;
}
