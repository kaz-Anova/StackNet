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

package manipulate.select;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make selections of rows
 */
public class RowSelectRunnable implements Runnable{

    /**
     * The selections for 1-d array will take place in this variable
     */
	double basearray1d[];
	/**
    * The selections for 2d Arrays will take place in this variable
    */
	double basearray2d[][];
    /**
     * 2d Array to select from
     */
	double array2d[][];
   /**
    * column to select
    */
	int row=-1;
	/**
	 * columns to select to for 2d arrays
	 */
	int rows [];
	/**
	 * start location of column loop in 1d arrays
	 */
	int s=0;
	/**
	 * end location of the column in 1d arrays
	 */	
	int e=0;
	/**
	 * start location of row loop in 1d arrays
	 */
	int sr=0;
	/**
	 * end location of the row in 1d arrays
	 */	
	int er=0;



	/**
	 * @param base  The array to chunk in the array(row) to be selected
	 * @param arrays The array that holds the 1 row to select
	 * @param row1 The row to select
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' 1d row selections
	 */
	RowSelectRunnable(double base[], double arrays [][], int row1, int start, int end)  {
		
		basearray1d=base;
		basearray2d=null;
		array2d=arrays;
		row=row1;
		rows=null;
		s=start;
		e=end;
		sr=-1;
		er=-1;
		
	};
	
	/**
	 * @param base  The array to chunk in the array(rows) to be selected
	 * @param arrays The array that holds the rows to select
	 * @param row1 The rows to select
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' 1d row selections
	 */
	RowSelectRunnable(double base[][], double arrays [][], int rows1[], int start, int end)  {
		
		basearray1d=null;
		basearray2d=base;
		array2d=arrays;
		row=-1;
		rows=rows1;
		s=start;
		e=end;
		sr=-1;
		er=-1;

		
	};
	
	/**
	 * @param base  The array to chunk in the array(row) to be selected
	 * @param arrays The array that holds the rows to select
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' 1d row selections
	 */
	RowSelectRunnable(double base[][], double arrays [][],int start2, int end2, int start, int end)  {
		
		basearray1d=null;
		basearray2d=base;
		array2d=arrays;
		row=-1;
		rows=null;
		s=start;
		e=end;
		sr=start2;
		er=end2;

		
	};
	

	@Override
	public void run() {
		
		//make some sensible checks
		
		if (s>e || s<0){
			
			throw new IllegalStateException(" The start of the loop in the selection operation cannot be less/equal than the end");
		}
		
		// first define the method
		
		if (row>=0 && row <array2d.length && basearray1d!=null && array2d!=null && basearray1d.length == array2d[row].length && s< basearray1d.length && e<= basearray1d.length){
								

				for (int i=s ; i < e; i++) {
					basearray1d [i]=array2d[row][i];
				}		

			// end of array 1d
		} else if (basearray2d!=null  && array2d!=null && rows!=null  && rows.length <=array2d.length && e-s<= rows.length && e<= rows.length && s<= rows.length) {
			
				for (int i=s; i < e; i++) {
					basearray2d[i]=manipulate.copies.copies.Copy(array2d[rows[i]]);
				}
				
		} else if (basearray2d!=null  && array2d!=null  && e-s <=array2d.length && e-s== er-sr ) {
			
			for (int i=s,n=sr; i < e; i++,n++) {
				basearray2d[i]=manipulate.copies.copies.Copy(array2d[n]);
			}
			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" row selection failed operation failed as there was no correct combination of object-arrays");
			
		}
		
	}

}
