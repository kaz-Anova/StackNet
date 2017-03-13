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
 *<p> Purpose of the class is to make selections of columns
 */
public class ColumnSelectRunnable implements Runnable{

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
	int col=-1;
	/**
	 * columns to select to for 2d arrays
	 */
	int cols [];
	/**
	 * start location of the loop in 1d arrays
	 */
	int s=0;
	/**
	 * end location of the loop in 1d arrays
	 */	
	int e=0;


	/**
	 * @param base  The array to chunk in the array to be selected
	 * @param arrays The array that holds the 1 variable to select
	 * @param col The column to select
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' 1d merges
	 */
	ColumnSelectRunnable(double base[], double arrays [][], int column, int start, int end)  {
		basearray1d=base;
		basearray2d=null;
		array2d=arrays;
		cols=null;
		col=column;
		s=start;
		e=end;
	};
	
	/**
	 * @param base  The array to chunk in the array to be selected
	 * @param col The column to select
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' 1d merges
	 */
	ColumnSelectRunnable(double base[][], int column[], int start, int end)  {
		basearray1d=null;
		basearray2d=base;
		cols=column;
		col=-1;
		s=start;
		e=end;
	};

	@Override
	public void run() {
		
		//make some sensible checks
		
		if (s>e){
			
			throw new IllegalStateException(" The start of the loop in the de-append operation cannot be less/equal than the end");
		}
		
		// first define the method
		
		if (basearray1d!=null && array2d!=null && basearray1d.length == array2d.length && col>0 && col<array2d[0].length ){
								

				for (int i=s ; i < e; i++) {
					basearray1d [i]=array2d[i][col];
				}		

			// end of array 1d
		} else if (basearray2d!=null && cols.length<=basearray2d[0].length  ) {
			
				for (int i=s; i < e; i++) {
					
					double[] temp= new double [cols.length];
					
					for (int j=0; j < cols.length; j++) {
						temp [j]=basearray2d[i][cols[j]];
					}
					
					basearray2d[i]=temp;

				}
				


			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" column selection failed operation failed as there was no correct combination of object-arrays");
			
		}
		
	}

}
