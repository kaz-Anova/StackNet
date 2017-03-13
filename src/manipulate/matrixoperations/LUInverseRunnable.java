/*Copyright (c) 2017 Marios Michailidis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

package manipulate.matrixoperations;

import exceptions.IllegalStateException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to provide some scalabilty options to the QR decomposition required to calculate the inverse of a matrix </p>
 *
 */
public class LUInverseRunnable  implements Runnable {
	/**
	 * To hold the permutations
	 */
	static public double x[];
	/**
	 * The Square matrix to hold the LU
	 */
	static public double LU[];
	/**
	 * square matrix to pass-to, normally a covariance matrix in a machine-learning context
	 */
	static public double covariancev[];
    /**
     * columns (or rows) of the square matrix
     */
	int n;
	/**
	 * column to permute for
	 */
	int column;
	/**
	 * @param X : Permutations' array
	 * @param lu : The LU (square) matrix
	 * @param cov : The LU (square) matrix to pass results onto
	 * @param n : the side's size of the matrix (rows or columns- does not matter as it is square)
	 * @param column : The column to append results to
	 */
	LUInverseRunnable(double X[], double lu[], double cov[], int N ,int c){
		x=X;
		LU=lu;
		covariancev=cov;
		n=N;
		column=c;
	}
	public void run() {
		
		if ( (double) n == Math.sqrt(LU.length) && 
			n==x.length && LU.length==covariancev.length){
			
			for (int i = 1; i < n; ++ i)
			{
			double sum = x[i];
			for (int j = 0; j < i; ++ j)
				{
				sum -= LU[i * n + j] * x[j];
				}
			x[i] = sum;
			}

		// back substitution.
		x[n-1] /= LU[(n-1)* n + (n-1)];
		for (int i = n-2; i >= 0; -- i)
			{
			double sum = x[i];
			for (int j = i+1; j < n; ++ j)
				{
				sum -= LU[i*n+j] * x[j];
				}
			x[i] = sum / LU[i*n + i];
			}
		
		for (int j = 0; j <n; ++ j)
		{
		covariancev[j*n +column] = x[j];
		}
		
			
		} else {
			
			throw new IllegalStateException("Size pre-condations are not valid for LU decomposition. All dimension of the included variables need to have size of: " +  n);
		}
		
	}
	
}
