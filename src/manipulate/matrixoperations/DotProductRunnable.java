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
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to provide a runnable class to dot product operations
 */
public class DotProductRunnable implements Runnable{

	/**
    * first matrix
    */
	double array2done[][];
	/**
	  * 2nd matrix
	  */	
	double array2dtwo[][];
	/**
	 * Matrix that holds the products
	 */
	double dot_matrix[][];
	/**
	 * Row from matrix one to multiply
	 */
	int row_from_matrix_one=-1;
	/**
	 * column from matrix 2
	 */
	int column_from_matrix_two=-1;	
	/**
	 * 0 for normal dot matrix, else transposes the first one
	 */
	int type=0;

/**
 * 
 * @param dotmatrix : the dot matrix to hold the product results;
 * @param array1 : the first matrix that its rows are multiplied
 * @param array2 :  the 2nd matrix that has its columns  multiplied
 * @param row_from_matrix_1 : The row to multiply from matrix 1 
 * @param column_from_2 : The column to multiply with from matrix 2.
 * <p> constructor for dot product implementation.
 */
	DotProductRunnable(double dotmatrix[][],double array1 [][],double array2[][], int row_from_matrix_1, int column_from_2, int types)  {
		array2done=array1;
		array2dtwo=array2;
		dot_matrix=dotmatrix;
		row_from_matrix_one=row_from_matrix_1;
		column_from_matrix_two=column_from_2;
		type=types;
	};
	

	
	
	@Override
	public void run() {
		
		//make some sensible checks
		if (array2done==null || array2dtwo==null || dot_matrix==null ){
			throw new NullObjectException(" One or more arrays needed for the dot product operation are null");
		}
		
		if (type==0){ // normal dot product
			
			if (column_from_matrix_two >= array2dtwo[0].length ||
					row_from_matrix_one>= array2done.length ||
					array2done[row_from_matrix_one].length!=array2dtwo.length
					|| row_from_matrix_one<0 || column_from_matrix_two<0 ||
					row_from_matrix_one>= dot_matrix.length || column_from_matrix_two >= array2dtwo[row_from_matrix_one].length ){
				
				throw new IllegalStateException(" The normal dot product operation cannot be completed as the arrays to multiply or the given rows/columns are incompatible");
				
			} else {
				
			       // make the dot product
		         double dot_pr=0.0;
		         for (int i=0; i < array2dtwo.length; i++) {
		        	 dot_pr+=array2done[row_from_matrix_one][i]*array2dtwo[i][column_from_matrix_two];
		         }
              // pass the value
		         dot_matrix[row_from_matrix_one][column_from_matrix_two]=dot_pr;
			}
			
			
		}


		 else {

				if (column_from_matrix_two >= array2dtwo[0].length ||
						row_from_matrix_one>= array2done[0].length ||
						array2done.length!=array2dtwo.length
						|| row_from_matrix_one<0 || column_from_matrix_two<0 ||
						row_from_matrix_one>= dot_matrix.length || column_from_matrix_two >= array2dtwo[row_from_matrix_one].length ){
					
					throw new IllegalStateException(" The transposed dot product operation cannot be completed as the arrays to multiply or the given rows/columns are incompatible");
					
				} else {
					
				       // make the dot product
			         double dot_pr=0.0;
			         for (int i=0; i < array2dtwo.length; i++) {
			        	 dot_pr+=array2done[i][row_from_matrix_one]*array2dtwo[i][column_from_matrix_two];
			         }
	              // pass the value
			         dot_matrix[row_from_matrix_one][column_from_matrix_two]=dot_pr;
				}       
			
			
			// end of arraylist 2d
		} 
		
		
		
		
		
		
		
		// TODO Auto-generated method stub
		
	}

}
