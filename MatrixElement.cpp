#include "MatrixElement.h"
#include "Input.h"
#include "Operators.h"
#include "CoordinatsTransformation.h"
#include "Permutation.h"
#include <cmath>
#include <Eigen> 
#include <vector>
using namespace Eigen;
using namespace std;

vector<VectorXd> SumOverCells(int dmax, int npar);
vector<VectorXd> TwoBodyPairs(int npar);

int factorial(int npar);

MatrixElement::MatrixElement(Input &input)
{
    //cout << "\t Initialize MatrixElement\n";
	npar = input.npar;
	h2m = input.h2m;
	dmax = input.dmax;
	nop = input.nop;
	npt = input.npt;
	ibf = input.bosefermi;  // ibf=2 for bosons.  ibf=1 for fermions
        L = input.BoxSize;
        apot3b = input.apot3b;
        vpot3b = input.vpot3b;
	nPairs = npar*(npar - 1) / 2;
	nPerm = factorial(npar);
	nConf = pow(2 * dmax + 1, npar); //number of cells configurations

	Permutation P(npar); 
	PM = P.perm_matrix;
	PV = P.permutation;
	if(ibf==1) parity = P.parity; //fermions
	if (ibf == 2) for (int iperm = 0; iperm < nPerm; iperm++) parity.push_back(1); //bosons
	
	mass=MatrixXd::Zero(npar,npar);
        for (int imas = 0; imas < npar; imas++) mass(imas,imas)=input.mass[imas];

        TBP = TwoBodyPairs(npar);

	PrepareSpinIsospinME(input);
	PreparePotential(input);

	bbox = SumOverCells(dmax, npar);
                        
}
//=============================================================================
double MatrixElement::overlap(std::vector<MatrixXd> state1, std::vector<MatrixXd> state2)
{
  //cout << "overlap start" << endl;
  MatrixXd A1 = state1[0];
  MatrixXd B1 = state1[1];
  VectorXd s1 = state1[2].diagonal();
  
  MatrixXd A2_0 = state2[0];
  MatrixXd B2_0 = state2[1];
  VectorXd s2_0 = state2[2].diagonal();
  
  MatrixXd A2;
  MatrixXd B2;
  VectorXd s2;
  
  double overlap = 0;
  MatrixXd AA;
  MatrixXd InvAA;
  VectorXd d;
  long double x1, x2, x3, x4, x5, x6, x7, x8;
  long double overlapB = 0;
  
  for (int iperm = 0; iperm < nPerm; iperm++) {//sum over permutation
    A2 = PM[iperm].transpose() * A2_0 * PM[iperm];
    B2 = PM[iperm].transpose() * B2_0 * PM[iperm];
    s2 = PM[iperm].transpose() * s2_0;
    AA = A1 + B1 + A2 + B2;
    InvAA = AA.inverse();
    x1 = sqrt(1.0 / AA.determinant());
    x2 = -0.5*s2.transpose()*B2*s2;
    x3 = -0.5*s1.transpose()*B1*s1;
    
    overlapB = 0;
    for (int iConf = 0; iConf < nConf; iConf++) {   //sum over cells
      d = L*(A1 + B1)*bbox[iConf] + B1*s1 + B2*s2;
      x4 = 0.5*d.transpose()*InvAA*d;
      x5 = -0.5*L*L*bbox[iConf].transpose()*A1*bbox[iConf];
      x6 = -0.5*L*L*bbox[iConf].transpose()*B1*bbox[iConf];
      x7 = -0.5*L*bbox[iConf].transpose()*B1*s1;
      x8 = -0.5*L*s1.transpose()*B1*bbox[iConf];
      
      overlapB = overlapB + x1*exp(x2+x3+x4+x5+x6+x7+x8);
    }
    overlap = overlap + parity[iperm] * stme[iperm*nPairs*nop] * pow(overlapB, 3);
  }
  
  //cout << "overlap end" << endl;
  return overlap;
}
//==============================================================================================
double MatrixElement::energy(std::vector<MatrixXd> state1, std::vector<MatrixXd> state2)
{
  MatrixXd A1 = state1[0];
  MatrixXd B1 = state1[1];
  VectorXd s1 = state1[2].diagonal();

  MatrixXd A2_0 = state2[0];
  MatrixXd B2_0 = state2[1];
  VectorXd s2_0 = state2[2].diagonal();

  MatrixXd A2;
  MatrixXd B2;
  VectorXd s2;
  
  MatrixXd AA;
  MatrixXd InvAA;
  
  double PotEnergy = 0, KinEnergy = 0, PotEnergy3B = 0;

  //===========3-body===================
                MatrixXd B(2,2);
                MatrixXd I = MatrixXd::Identity(2,2);
                CoordinatsTransformation v(npar);
                VectorXd Cik(npar);
                VectorXd Cjk(npar);
                int i1, j1, k1;
  //====================================

        long double KinEnergyB, PotEnergyB, overlapK;
	VectorXd d;
        VectorXd yy;
	long double sqInvDetAA, sBs2, sBs1, x4, x5, x6, x7, x8, x9, x10;
	long double v3b;
        long double y1,y2,y3;
        long double z1, z2, z3, z4, z5;
        long double s, r;

                //===========3-body===================
                double PotEnergy3BP = 0;
                MatrixXd BB(2,2);
                MatrixXd BBB(2,2);
                VectorXd e(2);
                double xx1, xx2, xx3, xx4, xx5, xx6, xx7;
                //====================================


  
	for (int iperm = 0; iperm < nPerm; iperm++)   //sum over permutation
	{
	  A2 = PM[iperm].transpose() * A2_0 * PM[iperm];
	  B2 = PM[iperm].transpose() * B2_0 * PM[iperm];
          s2 = PM[iperm].transpose() * s2_0;
	  AA = A1 + B1 + A2 + B2;
	  InvAA = AA.inverse();
	  sqInvDetAA = sqrt(1.0 / AA.determinant());
	  sBs2 = -0.5*s2.transpose()*B2*s2;
	  sBs1 = -0.5*s1.transpose()*B1*s1;
           //2B potential enegry=======================================
	  for (int ipt = 0; ipt < npt; ipt++){
	    for (int iop = 0; iop < nop; iop++){
	      for (int ipair = 0; ipair < nPairs; ipair++)   //sum over pairs
		{
		  s = TBP[ipair].transpose() * InvAA * TBP[ipair];
		  x9 = sqrt(1.0 / (2.0*apot(iop, ipt) *s + 1));
                             
		  PotEnergyB=0;
		  for (int iConf = 0; iConf < nConf; iConf++)   //sum over cells
		    {
		      d = L*(A1 + B1)*bbox[iConf] + B1*s1 + B2*s2;
		      x4 = 0.5*d.transpose()*InvAA*d;
		      x5 = -0.5*L*L*bbox[iConf].transpose()*A1*bbox[iConf];
		      x6 = -0.5*L*L*bbox[iConf].transpose()*B1*bbox[iConf];
		      x7 = -0.5*L*bbox[iConf].transpose()*B1*s1;
		      x8 = -0.5*L*s1.transpose()*B1*bbox[iConf];

		      for (int q = -dmax; q <= dmax; q++){
			r = -L*q + TBP[ipair].transpose() * InvAA * d;
			x10 = -apot(iop, ipt)*r*r *(1.0/(2.0*apot(iop, ipt)*s + 1));
                                                 
			PotEnergyB = PotEnergyB + sqInvDetAA*x9*exp(sBs2+sBs1+x4+x5+x6+x7+x8+x10);
		      }
		    }
		  PotEnergy = PotEnergy + parity[iperm] *stme[(iperm*nPairs+ipair)*nop+iop] 
		              *vpot(iop, ipt) *pow(PotEnergyB,3);
		}
	    }
	  }
                  //========================================================
                  //kinetic energy================================

                  y1 = ((A1 + B1) * InvAA * (A2 + B2)).trace();

                             KinEnergyB = 0;
                             overlapK = 0;
		             for (int iConf = 0; iConf < nConf; iConf++)   //sum over cells
		             {
		         	  d = L*(A1 + B1)*bbox[iConf] + B1*s1;
                                  yy = (A2+B2)*InvAA*d-(A1+B1)*InvAA*(B2*s2);
                                  y2 = yy.transpose()*yy;
		        	  z1 = 0.5*(d+B2*s2).transpose()*InvAA*(d+B2*s2);
			          z2 = -0.5*L*L*bbox[iConf].transpose()*A1*bbox[iConf];
			          z3 = -0.5*L*L*bbox[iConf].transpose()*B1*bbox[iConf];
			          z4 = -0.5*L*bbox[iConf].transpose()*B1*s1;
			          z5 = -0.5*L*s1.transpose()*B1*bbox[iConf];
                                  y3 = sqInvDetAA*exp(sBs2+sBs1+z1+z2+z3+z4+z5);
                                  overlapK = overlapK + y3;
		                  KinEnergyB = KinEnergyB + (y1-y2)*y3;

			      }
                 KinEnergy = KinEnergy + parity[iperm] *stme[iperm*nPairs*nop] *KinEnergyB*overlapK*overlapK;

                  //==============================================

              
                //====================================
                if(npar>2)
                {
		  PotEnergy3BP = 0;
		  for (int i = 0; i < npar; i++){
		  for (int j = i + 1; j < npar; j++){
		  for (int k = j + 1; k < npar; k++){
		    for (int cyc = 0; cyc < 3; cyc++){
		      if (cyc == 0) {i1 = i; j1 = j; k1 = k;}
		      if (cyc == 1) {i1 = j; j1 = k; k1 = i;}
		      if (cyc == 2) {i1 = k; j1 = i; k1 = j;}
		      Cik = v.SingleParticle(i1, k1);
		      Cjk = v.SingleParticle(j1, k1);
		      B(0,0) = Cik.transpose()*InvAA*Cik;
		      B(0,1) = Cik.transpose()*InvAA*Cjk;
		      B(1,0) = Cjk.transpose()*InvAA*Cik;
		      B(1,1) = Cjk.transpose()*InvAA*Cjk;
		      BB = I+2.0*apot3b*B;
		      BBB = B.inverse()*(I-BB.inverse());
		      xx1 = sqrt(1.0 / BB.determinant());
                             
		      PotEnergyB=0;
		      for (int iConf = 0; iConf < nConf; iConf++){   //sum over cells
			d = L*(A1 + B1)*bbox[iConf] + B1*s1 + B2*s2;
			xx2 = 0.5*d.transpose()*InvAA*d;
			xx3 = -0.5*L*L*bbox[iConf].transpose()*A1*bbox[iConf];
			xx4 = -0.5*L*L*bbox[iConf].transpose()*B1*bbox[iConf];
			xx5 = -0.5*L*bbox[iConf].transpose()*B1*s1;
			xx6 = -0.5*L*s1.transpose()*B1*bbox[iConf];
			for (int q1 = -dmax; q1 <= dmax; q1++){
			for (int q2 = -dmax; q2 <= dmax; q2++){    
			  e(0)=d.transpose()*InvAA*Cik-L*q1;
			  e(1)=d.transpose()*InvAA*Cjk-L*q2;
			  xx7=-0.5*e.transpose()*BBB*e;
			  v3b = sqInvDetAA*exp(sBs2+sBs1)*xx1*exp(xx2+xx3+xx4+xx5+xx6+xx7);
			  PotEnergyB = PotEnergyB + v3b;
			}
			}
		      }

		      PotEnergy3BP = PotEnergy3BP + pow(PotEnergyB,3);
		    }
		  }
		  }
                  }

                  PotEnergy3B = PotEnergy3B + parity[iperm] * stme[iperm*nPairs*nop] * PotEnergy3BP;
                                 
                //====================================
                }  //  end if(npar>2) 

               

	}   // end sum over permutation
 
        KinEnergy=0.5*3*h2m*KinEnergy;
        PotEnergy3B = vpot3b * PotEnergy3B;
 

  return PotEnergy+KinEnergy+PotEnergy3B;
}

//===







vector<VectorXd> SumOverCells(int dmax, int npar)
{
	int Nconf = pow(2 * dmax + 1, npar); //number of configuration
	vector<VectorXd> d(Nconf);
	VectorXd b(npar);
	for (int i = 0; i < npar; i++)  b(i) = -dmax; 

	for (int i = 0; i < Nconf; i++)
	{
		d[i] = b;
		b(0)++;
		for (int k = 0; k < npar - 1; k++)
		{
			if (b(k) > dmax)
			{
				b(k) = -dmax;
				b(k + 1)++;
			}
		}
	}
	return d;
}

//=============================================================================

vector<VectorXd> TwoBodyPairs(int npar)
{
	vector<VectorXd> Cij(npar*(npar - 1) / 2);
	CoordinatsTransformation v(npar);
	int ipair = 0;
	for (int i = 0; i < npar; i++)	{
		for (int j = i + 1; j < npar; j++){
			Cij[ipair] = v.SingleParticle(i, j);
			ipair++;
		}
	}
	return Cij;
}
//=============================================================================
void MatrixElement::PreparePotential(Input &input)
{
	vpot = MatrixXd::Zero(nop, npt);
	apot = MatrixXd::Zero(nop, npt);
//	cout << "PreparePotential: nop= " << nop << "  nterms= " << npt << "\n";
	for (int ipt = 0; ipt < npt; ipt++)
	{
		for (int iop = 0; iop < nop; iop++)
		{
			vpot(iop, ipt) = input.potop[iop].vpot[ipt];
			apot(iop, ipt) = input.potop[iop].aquad[ipt];
//			printf("\t\t iop = %4d   iterm = %4d   vpot = %8.3f   "
//		    "aquad = %8.3f   \n",iop,ipt,vpot(iop,ipt),apot(iop, ipt));
		}
	}
}
//=============================================================================
void MatrixElement::PrepareSpinIsospinME(Input &input)
{
        Operators operators(input);
	stme.resize(nPairs*nPerm*nop);
      //  int keypr = 1;

	int ipair = -1;
	for (int ip = 0; ip < npar; ip++){
	  for (int jp = ip + 1; jp < npar; jp++){
	    ipair++;
	//    fprintf(input.printfile,
	//	    "\t\t PrepareSpinIsospinME: ipair = %4d   ipar = %4d   jpar = %4d \n",ipair,ip,jp);
	    for (int iperm = 0; iperm < nPerm; iperm++){
	      for (int iop = 0; iop < nop; iop++){       
		stme[(iperm*nPairs+ipair)*nop+iop] = operators.O(ip, jp, PV[iperm], iop);                               
               
		/* print spin-isospin matrix element */
		//if (keypr == 1) {
		 // fprintf(input.printfile,"\t\t   iperm = %4d  ",iperm);
		 // for (int i = 0; i < npar; i++) fprintf(input.printfile,"%1d",PV[iperm][i]);
		//  fprintf(input.printfile,
		//	 "      iop = %4d  me = %9.5f \n",iop,stme[(iperm*nPairs+ipair)*nop+iop]);
		//}

	      }
	    }
	  }
	}
}
//=============================================================================
int factorial(int npar)
{
	int N;
	if (npar <= 1) return 1;
	N = npar * factorial(npar - 1);
	return N;
}


/*
//harmonic=====================================

//=============================================================================
//double MatrixElement::overlap(std::vector<MatrixXd> state1, std::vector<MatrixXd> state2)
{
	MatrixXd A1 = state1[0];
	MatrixXd B1 = state1[1];
	MatrixXd A2 = state2[0];
	MatrixXd B2 = state2[1];

	return pow((A1 + B1 + A2 + B2).determinant(),-1.5);
}

//=============================================================================
//double MatrixElement::energy(std::vector<MatrixXd> state1, std::vector<MatrixXd> state2)
{
  MatrixXd A1 = state1[0];
  MatrixXd B1 = state1[1];
  MatrixXd A2 = state2[0];
  MatrixXd B2 = state2[1];

  
  MatrixXd AA = A1 + B1 + A2 + B2;
  MatrixXd invAA = AA.inverse(); 
  double overlap = pow(AA.determinant(), -1.5); 
  MatrixXd TT = (A1 + B1) * invAA * (A2 + B2);
  double PotEnergy = 0, KinEnergy = 0;
  Vector2d c(1, -1);
  KinEnergy = 0.5 * h2m * TT.trace();
  PotEnergy = (0.25/h2m) * c.transpose() * invAA * c;

  return overlap*(PotEnergy+KinEnergy);
}

// end harmoinic====================================

*/

MatrixElement::~MatrixElement()
{
}
