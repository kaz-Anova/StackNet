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
 *<p> Purpose of the class is to provide a runnable class to the copy operations
 */
public class TransposeRunnable implements Runnable{

	/**
    * 2d Array to copy values from
    */
	double array2d[][];

	/**
	  * 2d Array to copy values to
	  */	
	double array2dtopass[][];
	/**
	 * the column to retrieve
	 */
	int column_to_get=-1;
	/**
	 * the row to send to
	 */
	int row_to_send=-1;	
	

	/**
	 * 
	 * @param array  The array to form as the transposed matrix
	 * @param array_topass  The array to copy values to (by reference) 
	 * @param column To send the data
	 * @param row to retrieve the data
	 * Basic constructor with transposed operations for 2d arrays
	 */
	TransposeRunnable(double array [][],double array_topass[][], int column, int row)  {
		array2d=array;
		array2dtopass=array_topass;
		row_to_send=row;
		column_to_get=column;
	};
	

	
	
	@Override
	public void run() {
		
		//make some sensible checks
		
		if (column_to_get >= array2dtopass[0].length || row_to_send>= array2d.length || array2d[row_to_send].length!=array2dtopass.length){
			
			throw new IllegalStateException(" The transpose operation cannot be completed as the arrays to transpose are incompatible");
		}
		
		// first define the method
		
		if (array2dtopass!=null && array2d!=null && row_to_send>=0 && column_to_get>=0 ){
								
			
				for (int i=0; i < array2dtopass.length; i++) {
					array2d[row_to_send][i]=array2dtopass[i][column_to_get];
				}
			
			
			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" copy operation failed as either the copy-to or copy-from (or both) container-objects were null or of different sizes");
			
		}
		
		
		
		
		
		
		
		// TODO Auto-generated method stub
		
	}

}
