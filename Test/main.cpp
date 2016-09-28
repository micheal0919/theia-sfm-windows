#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "cholmod.h"

using namespace std;
using namespace Eigen;

void test_1()
{
	Matrix2f A, b;
	LLT<Matrix2f> llt;
	A << 2, -1, -1, 3;
	b << 1, 2, 3, 1;
	cout << "Here is the matrix A:\n" << A << endl;
	cout << "Here is the right hand side b:\n" << b << endl;
	cout << "Computing LLT decomposition..." << endl;
	llt.compute(A);
	cout << "The solution is:\n" << llt.solve(b) << endl;
	A(1, 1)++;
	cout << "The matrix A is now:\n" << A << endl;
	cout << "Computing LLT decomposition..." << endl;
	llt.compute(A);
	cout << "The solution is now:\n" << llt.solve(b) << endl;
	
}

void test_2()
{
	MatrixXd A(3, 3);
	A << 4, -1, 2, -1, 6, 0, 2, 0, 5;
	cout << "The matrix A is" << endl << A << endl;
	LLT<MatrixXd> lltOfA(A); // compute the Cholesky decomposition of A
	MatrixXd L = lltOfA.matrixL(); // retrieve factor L  in the decomposition
	// The previous two lines can also be written as "L = A.llt().matrixL()"
	cout << "The Cholesky factor L is" << endl << L << endl;
	cout << "To check this, let us compute L * L.transpose()" << endl;
	cout << L * L.transpose() << endl;
	cout << "This should equal the matrix A" << endl;
}

void test_3()
{

	typedef Matrix<float, Dynamic, 2> DataMatrix;
	// let's generate some samples on the 3D plane of equation z = 2x+3y (with some noise)
	DataMatrix samples = DataMatrix::Random(12, 2);
	VectorXf elevations = 2 * samples.col(0) + 3 * samples.col(1) + VectorXf::Random(12)*0.1;
	// and let's solve samples * [x y]^T = elevations in least square sense:
	Matrix<float, 2, 1> xy
		= (samples.adjoint() * samples).llt().solve((samples.adjoint()*elevations));
	cout << xy << endl;
}



int test_cholmod(void)
{
	cholmod_sparse *A;
	cholmod_dense *x, *b, *r;
	cholmod_factor *L;
	double one[2] = { 1, 0 }, m1[2] = { -1, 0 }; /* basic scalars */
	cholmod_common c;
	cholmod_start(&c); /* start CHOLMOD */
	A = cholmod_read_sparse(stdin, &c); /* read in a matrix */
	cholmod_print_sparse(A, "A", &c); /* print the matrix */
	if (A == NULL || A->stype == 0) /* A must be symmetric */
	{
		cholmod_free_sparse(&A, &c);
		cholmod_finish(&c);
		return (0);
	}
	b = cholmod_ones(A->nrow, 1, A->xtype, &c); /* b = ones(n,1) */
	L = cholmod_analyze(A, &c); /* analyze */
	cholmod_factorize(A, L, &c); /* factorize */
	x = cholmod_solve(CHOLMOD_A, L, b, &c); /* solve Ax=b */
	r = cholmod_copy_dense(b, &c); /* r = b */
	cholmod_sdmult(A, 0, m1, one, x, r, &c); /* r = r-Ax */
	printf("norm(b-Ax) %8.1e\n",
		cholmod_norm_dense(r, 0, &c)); /* print norm(r) */
	cholmod_free_factor(&L, &c); /* free matrices */
	cholmod_free_sparse(&A, &c);
	cholmod_free_dense(&r, &c);
	cholmod_free_dense(&x, &c);
	cholmod_free_dense(&b, &c);
	cholmod_finish(&c); /* finish CHOLMOD */
	return (0);
}

void test_cereal()
{

}

int test_x()
{
	cout << "Hello World!" << endl;

	test_1();
	test_2();
	test_3();
	test_cholmod();

	cout << "End of main" << endl;
	return 0;
}

