#include <vector>
#include <iostream>

using namespace std;
typedef vector< vector<int> > Matrix;

int calculateOneOutputValue( const Matrix& A, const Matrix& B, int i, int j)
{
	size_t N = A.size();
	int output = 0;
	for (size_t k = 0; k < N; ++k)
	{
		output += A[i][k] * B[k][j];
	}
	return output;
}

Matrix multipleMatrix( const Matrix& A, const Matrix& B)
{
	size_t N = A.size();
	Matrix C(N, vector<int>(N, 0));
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			C[i][j] = calculateOneOutputValue(A, B, i, j);
		}
	}
	return C;
}

void debugPrint(const std::vector<int>& set)
{
	for (std::vector<int>::const_iterator i = set.begin(); i != set.end(); ++i)
	{
		cout << *i << endl;
	}
}

void debugPrint(const Matrix& M)
{
	for (Matrix::const_iterator i = M.begin(); i != M.end(); ++i)
	{
		debugPrint(*i);
	}
}

int main()
{
	Matrix A(2, vector<int>(2, 0));
	Matrix B(2, vector<int>(2, 0));
	A[0][0] = 2;
	A[0][1] = 1;
	A[1][0] = 1;
	A[1][1] = 1;
	B[0][0] = 1;
	B[0][1] = 1;
	B[1][0] = 1;
	B[1][1] = 1;
	Matrix C = multipleMatrix(A, B);
	debugPrint(C);
}
