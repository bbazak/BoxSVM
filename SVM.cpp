#include "SVM.h"
#include "Input.h"
#include "MatrixElement.h"
#include "Rand.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <Eigen>
using namespace Eigen;
//=============================================================================
SVM::SVM(Rand &r, Input &input) : rr(r), me(input)
{
	N = input.npar;
	bmin = input.rndmin;
	bmax = input.rndmax;
	mm0 = input.mm0;
	kk0 = input.kk0;
	int ndb = input.maxbasis;
	iBoxInf = input.keycontinue;
	Hmatrix = MatrixXd::Zero(ndb, ndb);
	Nmatrix = MatrixXd::Zero(ndb, ndb);
}
//=============================================================================
void SVM::SaveToFile(std::string file, vector<vector<MatrixXd>> Basis)
{
	ofstream myfile;
	myfile.open(file.c_str());
	int itr = Basis.size();
	myfile << itr << "\n";
	for (int i = 0; i < itr; i++)
	{
		MatrixXd A1 = Basis[i][0];
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				myfile << A1(j, k) << " ";
		myfile << "\n";
		VectorXd B1 = Basis[i][1].diagonal();
		for (int j = 0; j < N; j++)
			myfile << B1(j) << " ";
		myfile << "\n";
		VectorXd s1 = Basis[i][2].diagonal();
		for (int j = 0; j < N; j++)
			myfile << s1(j) << " ";
		myfile << "\n\n";
	}
	myfile.close();
}
//=============================================================================
int SVM::CheckOverlap(vector<vector<MatrixXd>> Basis)
{
	int itr = Basis.size() - 1;
	float vnorm, vdotv;
	vnorm = Nmatrix(itr, itr);
	//  printf("\t itr = %4d     vnrom = %12.6f \n",itr,vnorm);
	if (vnorm < 1e-8)
		return 0;
	if (itr > 0)
	{
		for (int i = 0; i < itr; i++)
		{
			vdotv = Nmatrix(itr, i) / sqrt(Nmatrix(i, i) * Nmatrix(itr, itr));
			//      printf("\t\t i = %4d     vdotv = %12.6f \n",i,vdotv);
			if (vdotv > 0.99)
			{
				return 0;
			}
		}
	}
	return 1;
}
//=============================================================================
double EigenValusEquation(int itr, VectorXd D, VectorXd q, double aa, double xx)
{
/*
	double vv = 1;
	double ww = 1;
	double yy = 0;
	double zz = 0;

	for (int n1 = 0; n1 < itr; n1++)
	{
		vv = vv * (D(n1) - xx);
		ww = 1;
		for (int n2 = 0; n2 < itr; n2++)
		{
			if (n2 != n1)
				ww = ww * (D(n2) - xx);
		}
		yy = yy + q(n1) * q(n1) * ww;
	}
	zz = (aa - xx) * vv - yy;

	return zz;
*/

	double w=1.;
    double prod=w;
    w=w*(D(itr-1)-xx);
	for (int j = 0; j < itr-1; j++)
	{
		double ww=q(j)*q(j)*prod/(D(j)-xx);
        w=w-ww;
	}
 	return w;

}
//=============================================================================

double SVM::NewEnergy(vector<vector<MatrixXd>> Basis, MatrixXd C, VectorXd D, double E, double EE)
{
	int itr = Basis.size() - 1;
	VectorXd c = VectorXd::Zero(itr + 1);
	VectorXd Overlap = VectorXd::Zero(itr);
	VectorXd q = VectorXd::Zero(itr);
	double aa = 0;
	double NN = 0;

	//create the vector of i+1 state orthogonal to all previous orthogonal eigenvectors.
	for (int k1 = 0; k1 < itr; k1++)
	{
		for (int k2 = 0; k2 < itr; k2++)
		{
			Overlap(k1) = Overlap(k1) + C(k2, k1) * Nmatrix(itr, k2);
		}
	}

	c(itr) = 1;
	for (int k1 = 0; k1 < itr; k1++)
	{
		for (int k2 = 0; k2 < itr; k2++)
		{
			c(k1) = c(k1) - Overlap(k2) * C(k1, k2);
		}
	}

	//normalize the i+1 vector  ==================================================
	for (int k1 = 0; k1 < itr + 1; k1++)
	{
		for (int k2 = 0; k2 < itr + 1; k2++)
		{
			NN = NN + c(k1) * c(k2) * Nmatrix(k1, k2);
		}
	}
	for (int k1 = 0; k1 < itr + 1; k1++)
	{
		c(k1) = c(k1) / sqrt(NN);
	}

	//create the matrix elements of i+1 vector with all previous orthogonal eigenvectors.
	for (int k1 = 0; k1 < itr; k1++)
	{
		for (int k2 = 0; k2 < itr; k2++)
		{
			for (int k3 = 0; k3 < itr + 1; k3++)
			{
				q(k1) = q(k1) + Hmatrix(k2, k3) * c(k3) * C(k2, k1);
			}
		}
	}
	for (int k1 = 0; k1 < itr + 1; k1++)
	{
		for (int k2 = 0; k2 < itr + 1; k2++)
		{
			aa = aa + c(k1) * c(k2) * Hmatrix(k1, k2);
		}
	}

	//solving the equation for the new eigenvalue===============================================
	// from NAG, p01aaf, mark 7 revised (dec 1978)
	int count = 0;
    double x1, x2, x3, acc=1.e-8, g1, g2, g3;
	x1 =  E; g1 = 1./EigenValusEquation(itr, D, q, aa, x1);
	x3 = E - abs(E - EE); g3 = 1./EigenValusEquation(itr, D, q, aa, x3);
	while (count++<100 && g1 * g2 > 0) 
	{
		x1 =  x3; g1 = g3; 
		x3 = x1 - abs(E - EE); g3 = 1./EigenValusEquation(itr, D, q, aa, x2);
	}
	x2 = 0.5*(x1+x3); g2 =  1./EigenValusEquation(itr, D, q, aa, x2);
	for (int ifu = 0; ifu < 300; ifu++)
	{
		//cout << ifu << ". x1: " << x1 << " g1: " << g1 
		//	 << " x2: " << x2 << " g2: " << g2 
		//	 << " x3: " << x3 << " g3: " << g3 << endl;
        int ic1=1;
     	int ic2=1;
        if ((g1<0 && g2<0) || (g1>0 && g2>0)) ic1=0;
        if ((g2<0 && g3<0) || (g2>0 && g3>0)) ic2=0;
		// no root in the interval
        if (ic1+ic2==0) 
		{
        	//  if(ifu.eq.1) ifa=1
        	return x1;
		}
        double x12=x1-x2;
        double x23=x2-x3;
		double x13=x1-x3;
        double www=g1*x23-g2*x13+g3*x12;
        double x4=1.e30;
        if(www!=0.) 
             x4=(g1*x1*x23-g2*x2*x13+g3*x3*x12)/www;
		// check bounds
		// bisection
        if(ic1==1) 
		{
        	if(x4<x1 || x4>x2) 
			{
         		double xx=x2;
         		x3=x2;
         		g3=g2;
         		x2=0.5*(x1+xx);
         		double f2=EigenValusEquation(itr, D, q, aa, x2);
         		if(abs(f2)<acc) 
					return x2;
				g2=1./f2;
			}
         	else
			{
         		double xx=x2;
         		x3=x2;
         		g3=g2;
         		x2=x4;
				double f2=EigenValusEquation(itr, D, q, aa, x2);
				if(abs(f2)<acc)
					return x2;
				g2=1./f2;
			}
        }
        if(ic2==1)
		{
        	if(x4<x2 || x4>x3)
			{
				x1=x2;
         		g1=g2;
         		x2=0.5*(x2+x3);
				double f2=EigenValusEquation(itr, D, q, aa, x2);
				if(abs(f2)<acc)
					return x2;
				g2=1./f2;
			}
         	else
			{
        		x1=x2;
         		g1=g2;
         		x2=x4;
		 		double f2=EigenValusEquation(itr, D, q, aa, x2);
				if(abs(f2)<acc)
					return x2;
		        g2=1./f2;
			}
		}
		if(abs(x4-x3)<acc) 
			return x4;
	}
	return x1;
/*  
	int count = 0;
    double xm, x0, fx0, x1, x2, eps=1.e-12, fx1, fx2;
	x1 =  E; fx1 = EigenValusEquation(itr, D, q, aa, x1);
	x2 = E - abs(E - EE); fx2 = EigenValusEquation(itr, D, q, aa, x2);
	cout << count << ". x1: " << x1 << " fx1: " << fx1 << " x2: " << x2 << " fx2: " << fx2 << endl;
	while (count++<100 && fx1 * fx2 > 0) { 
		x1 =  x2; fx1 = fx2; 
		x2 = x1 - abs(E - EE); fx2 = EigenValusEquation(itr, D, q, aa, x2);
		cout << count << "; x1: " << x1 << " fx1: " << fx1 << " x2: " << x2 << " fx2: " << fx2 << endl;
	}
    if (fx1 * fx2 < 0) { 
        do {
			// calculate the intermediate value
			x0 = (x1 * fx2 - x2 * fx1) / (fx2 - fx1); 
  
            // check if x0 is root of equation or not 
			fx0 = EigenValusEquation(itr, D, q, aa, x0); 
  
            // update the value of interval 
            x1 = x2; fx1 = fx2; 
            x2 = x0; fx2 = fx0;
			cout << count << ": x1: " << x1 << " fx1: " << fx1 << " x2: " << x2 << " fx2: " << fx2 << endl;
  
            // update number of iteration 
            count++; 
  
            // if x0 is the root of equation then break the loop 
            if (fx0 == 0) 
                break; 
			
            xm = (x1 * fx2 - x2 * fx1) / (fx2 - fx1); 
        } while (fabs(xm - x0) >= eps); // repeat the loop until the convergence 
  
        cout << "Root of the given equation=" << x0 << endl; 
    } else
        cout << "Can not find a root in the given inteval" << endl; 
	
	return x0;
*/

	/* int count = 0;
	double e1 = E;
	double e2 = E - abs(0.5 * (E - EE));
	double e3 = E;
	double Ee1 = EigenValusEquation(itr, D, q, aa, e1);
	//std::cout << "intial e1= " << e1 << ".   intial e2= " << e2<<endl;
	while (count < 101)
	{
		if (Ee1 * EigenValusEquation(itr, D, q, aa, e2) < 0)
			break;
		else
		{
			e1 = e2;
			e2 = e2 - abs(0.5 * (E - EE));
			count++;
		}
	}
	//std::cout  << "counter= " << count << std::endl;
	//if (count > 100)  std::cout << "finding root less then the last fail " << std::endl;
	if (count <= 100)
	{
		count = 0;
		//	std::cout << "e3= ";
		while (abs((e1 - e2) / e2) > abs(1e-5 * (E - EE) / EE))
		{
			e3 = (e1 + e2) / 2;
			//	std::cout << e3 << "  ";
			if (EigenValusEquation(itr, D, q, aa, e3) * EigenValusEquation(itr, D, q, aa, e2) < 0)
				e1 = e3;
			else
				e2 = e3;
			count++;
			if (count > 100)
				break;
		}
		//	std::cout << std::endl;
	}
	return e3; */
}
//=============================================================================
MatrixXd SVM::Dmatrix()
{
	MatrixXd d = MatrixXd::Zero(N, N);
	for (int i = 0; i < N; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			d(i, j) = bmin + (bmax - bmin) * rr.doub();
			d(j, i) = d(i, j);
		}
	}
	return d;
}
//=============================================================================
MatrixXd SVM::diagDmatrix()
{
	MatrixXd dd = MatrixXd::Zero(N, N);
	for (int i = 0; i < N; i++)
	{
		dd(i, i) = bmin + (bmax - bmin) * rr.doub();
	}
	return dd;
}
//=============================================================================
MatrixXd SVM::A(MatrixXd d)
{
	MatrixXd A = MatrixXd::Zero(N, N);
	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			if (i == j)
			{
				for (int k = 0; k < N; k++)
				{
					if (i != k)
						A(i, j) = A(i, j) + 2 * pow(d(i, k), -2);
				}
			}
			else
			{
				A(i, j) = -2 * pow(d(i, j), -2);
				A(j, i) = A(i, j);
			}
		}
	}
	return A;
}
//=============================================================================
MatrixXd SVM::B(MatrixXd dd)
{
	MatrixXd B = MatrixXd::Zero(N, N);
	for (int i = 0; i < N; i++)
	{
		B(i, i) = 1.0 / (dd(i, i) * dd(i, i));
	}
	return B;
}
//=============================================================================
MatrixXd SVM::s(MatrixXd dd)
{
	MatrixXd s = MatrixXd::Zero(N, N);
	for (int i = 0; i < N; i++)
	{
		s(i, i) = -0.5 * bmax + dd(i, i);
	}
	return s;
}
//=============================================================================
vector<MatrixXd> SVM::FirstNewState()
{
	//cout << "FirstNewState" << endl;
	//cout << "FirstNewState2" << endl;
	double e_overlap;
	vector<MatrixXd> NewState;
	int count = 0;
	while (count++ < 10 && e_overlap < 1e-8)
	{
		//cout << "FirstNewState3 " << count << endl;
		NewState.clear();
		NewState.push_back(A(Dmatrix()));
		NewState.push_back(B(diagDmatrix()));
		NewState.push_back(s(diagDmatrix()));
		//	NewState[0] = A(Dmatrix());
		//	cout << "FirstNewState4 " << count << endl;
		e_overlap = me.overlap(NewState, NewState);
		//	cout << "FirstNewState5 " << count << endl;
	}
	//cout << "FirstNewState6" << endl;
	if (count >= 8)
	{
		NewState[0](0, 0) = 2000;
		return NewState;
	}
	else
	{
		double MinE = me.energy(NewState, NewState) / e_overlap;
		double NewE;
		vector<MatrixXd> State(NewState.size());
		State = NewState;
		int count = 0;
		while (count < 1000)
		{
			count++;
			NewState[0] = A(Dmatrix());
			NewState[1] = B(diagDmatrix());
			NewState[2] = s(diagDmatrix());
			e_overlap = me.overlap(NewState, NewState);
			if (e_overlap < 1e-8)
				continue;
			NewE = me.energy(NewState, NewState) / e_overlap;
			if (NewE < MinE)
			{
				MinE = NewE;
				State[0] = NewState[0];
				State[1] = NewState[1];
				State[2] = NewState[2];
			}
		}
		return State;
	}
}

//=============================================================================
vector<MatrixXd> SVM::NewState(vector<vector<MatrixXd>> Basis, MatrixXd C, VectorXd D, double E, double EE)
{
	vector<MatrixXd> NewState;
	MatrixXd d = Dmatrix();
	NewState.push_back(A(d));
	MatrixXd dd = diagDmatrix();
	NewState.push_back(B(dd));
	MatrixXd ss = diagDmatrix();
	NewState.push_back(s(ss));

	Basis.push_back(NewState);
	UpdateNorm(Basis);
	UpdateHamiltonian(Basis);

	int Bsize = Basis.size() - 1;
	int size = NewState.size();
	vector<MatrixXd> State(size);

	MatrixXd mind = d;
	MatrixXd mindd = dd;
	MatrixXd minss = ss;

	int i, j, ii, jj, kk, ll, nn, count1, count2, count3, count4;
	int xx = 0;
	count4 = 0;
	double minE, NewE;

	/*
        int yy=0; 
        int count0=0;
        minE=E; NewE=E;
	while (count0 <= mm0)
	{
		if (CheckOverlap(Basis) == 1)
		{
			NewE = NewEnergy(Basis, C, D, E, EE);
			if (NewE < minE) 
                        {
                                yy=1;
				minE = NewE;
	                        State = NewState;                                         			
		        }
                }
	        NewState[0]=A(Dmatrix());
		Basis[Bsize] = NewState;
		UpdateNorm(Basis);
		UpdateHamiltonian(Basis);
                count0++;
	}
        if(yy==1) 
        {
        Basis[Bsize] = State;
	UpdateNorm(Basis);
	UpdateHamiltonian(Basis);
        }
  */

	while (count4 <= mm0)
	{
		i = 0;
		j = 1;
		ii = 0;
		jj = 0;
		kk = 0;
		ll = 0;
		nn = 0;
		count1 = 0;
		count2 = 0;
		count3 = 0;
		minE = E;
		NewE = E;

		while (count1 < mm0 * kk0 * (N * (N - 1) / 2 + N + N))
		{
			if (CheckOverlap(Basis) == 1)
			{
				NewE = NewEnergy(Basis, C, D, E, EE);
				if (NewE < minE)
				{
					xx = 1;
					State = NewState;
					mind = d;
					mindd = dd;
					minss = ss;
					//std::cout << "Lower energy!" << "   dE= " << std::scientific << NewE-E << std::endl;
					minE = NewE;
				}	
				else
					//std::cout << "counter = " << count1 << "   dE= " << std::scientific << NewE-E << std::endl;
				count2 = 0;
			}
			count1++;
			//============================
			if (count1 % kk0 == 0)
			{
				d = mind;
				dd = mindd;
				ss = minss;
			}
			//============================
			count2++;
			count3++;
			if (count2 > 500)
			{
				State[0] = NewState[0];
				State[0](0, 0) = 2000;
				break;
			}
			if (count3 <= kk0 * N * (N - 1) / 2) // change A
			{
				// d = mind; 
				d(i, j) = bmin + (bmax - bmin) * rr.doub();
				d(j, i) = d(i, j);
				kk++;
				if (kk == kk0)
				{
					kk = 0;
					j++;
					if (j == N)
					{
						i++;
						if (i == N - 1)
							i = 0;
						j = i + 1;
					}
				}
				NewState[0] = A(d);
			}
			else if (count3 <= kk0 * (N * (N - 1) / 2 + N)) // change B
			{
				// dd = mindd;
				dd(ii, ii) = bmin + (bmax - bmin) * rr.doub();
				ll++;
				if (ll == kk0)
				{
					ll = 0;
					ii++;
					if (ii == N)
					{
						ii = 0;
					}
				}
				NewState[1] = B(dd);
			}
			else // change S
			{
				// ss = minss;
				ss(jj, jj) = bmin + (bmax - bmin) * rr.doub();
				nn++;
				if (nn == kk0)
				{
					nn = 0;
					jj++;
					if (jj == N)
					{
						jj = 0;
						//count3=0;
					}
				}
				NewState[2] = s(ss);
			}
			if (count3 == kk0 * (N * (N - 1) / 2 + N + N))
				count3 = 0;
			Basis[Bsize] = NewState;
			UpdateNorm(Basis);
			UpdateHamiltonian(Basis);
		}
		if (xx == 0)
		{
			count4++;
			d = Dmatrix();
			NewState[0] = A(d);
			//dd = diagDmatrix();
			//NewState[1]=B(dd);
			//ss = diagDmatrix();
			//NewState[2]=s(ss);
			Basis[Bsize] = NewState;
			UpdateNorm(Basis);
			UpdateHamiltonian(Basis);
			State[0] = NewState[0];
			State[0](0, 0) = 2000;
		}
		if (xx == 1)
			break;
	} //end while(count4<5)
	  //cout<<"count4=  "<<count4<<endl;
	return State;
}

//=============================================================================
MatrixXd SVM::NormMatrix(vector<vector<MatrixXd>> Basis)
{
	int itr = Basis.size();
	MatrixXd Norm = MatrixXd::Zero(itr, itr);
	for (int n1 = 0; n1 < itr; n1++)
	{
		for (int n2 = n1; n2 < itr; n2++)
		{
			//Norm(n1, n2) = me.overlap(Basis[n1], Basis[n2]);
			Norm(n1, n2) = Nmatrix(n1, n2);
			Norm(n2, n1) = Norm(n1, n2);
		}
	}
	return Norm;
}

//=============================================================================
MatrixXd SVM::HamiltonianMatrix(vector<vector<MatrixXd>> Basis)
{
	int itr = Basis.size();
	MatrixXd H = MatrixXd::Zero(itr, itr);
	for (int n1 = 0; n1 < itr; n1++)
	{
		for (int n2 = n1; n2 < itr; n2++)
		{
			H(n1, n2) = Hmatrix(n1, n2);
			H(n2, n1) = H(n1, n2);
		}
	}
	return H;
}
//=============================================================================
void SVM::UpdateNorm(vector<vector<MatrixXd>> Basis)
{
	int ncur = Basis.size() - 1;
	for (int n1 = 0; n1 <= ncur; n1++)
	{
		Nmatrix(n1, ncur) = me.overlap(Basis[n1], Basis[ncur]);
		Nmatrix(ncur, n1) = Nmatrix(n1, ncur);
	}
}
//=============================================================================
void SVM::UpdateHamiltonian(vector<vector<MatrixXd>> Basis)
{
	int ncur = Basis.size() - 1;
	for (int n1 = 0; n1 <= ncur; n1++)
	{
		Hmatrix(n1, ncur) = me.energy(Basis[n1], Basis[ncur]);
		Hmatrix(ncur, n1) = Hmatrix(n1, ncur);
	}
}
//=============================================================================
SVM::~SVM()
{
}
//=============================================================================
