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
public class InverseRunnable  implements Runnable {

	/**
	 * Q in QRT decomposition
	 */
	static public double q[];
	/**
	 * T in QRT decomposition
	 */
	static public double t[];	
	/**
	 * the symmetric (n X n) matrix
	 */
	static public double symmetric_matrix[][];
    /**
     * start of the loop
     */
	int start=-1;
	/**
	 * end of loop
	 */
	int end=-1;
	
	/**
	 * 
	 * @param Q : Q in qrt decomposition
	 * @param T : T in qrt decomposition
	 * @param matrix : the symmetric matrix to invert
	 * @param sta : the start of the loop
	 * @param en : the end of the loop
	 */
	InverseRunnable(double Q[], double T[], double matrix[][], int sta, int en ){
		q=Q;
		t=T;
		symmetric_matrix=matrix;
		start=sta;
		end=en;
	}

	/**
	 * @param sta : the start of the loop
	 * @param en : the end of the loop
	 */
	InverseRunnable( int sta, int en){
		start=sta;
		end=en;

	}
	
	/**
	 * null constructor
	 */
	InverseRunnable(){
		
	}
	
	
	public void run() {
		
		if ( q.length == t.length && 
			t.length==symmetric_matrix.length
			&& t.length==symmetric_matrix[0].length && start >=0 && start <symmetric_matrix.length && end <=symmetric_matrix.length
			&& end>=start){
			
			   for (int s = start; s < end; s++){
			   for (int K = s; K < symmetric_matrix.length; K++){
				   symmetric_matrix[s] [K]+= t[s] * q[K];
			   }
			   }
			
		} else {
			
			throw new IllegalStateException("Size pre-condations are not valid for QR decomposition. All dimension of the included variables need to have size of: " + symmetric_matrix.length);
		}
		
	}
	
}
