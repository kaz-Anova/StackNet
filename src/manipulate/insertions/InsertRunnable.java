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

package manipulate.insertions;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to Merge arrays as in concatenating them vertically
 */
public class InsertRunnable implements Runnable{

    /**
     * The insert for 1-d array will take place in this variable
     */
	double basearray1d[];
	/**
    * The insert for 2d Arrays will take place in this variable
    */
	double basearray2d[][];
    /**
     * 1d Array to to be inserted
     */
	double array1d[];
	/**
    * 2d Array to form the merge skeleton
    */
	double array2d[][];
   /**
    * column to insert to for 2d-1d cases
    */
	int col=-1;
	/**
	 * columns to insert to for 2d arrays
	 */
	int cols [];
	/**
	 * columns to retrieve from for 2d arrays
	 */
	int cols2 [];
	/**
	 * start location of the loop in 1d arrays
	 */
	int s=0;
	/**
	 * end location of the loop in 1d arrays
	 */	
	int e=0;
	/**
	 * start location of the loop
	 */
	int s2=0;
	/**
	 * end location of the loop
	 */	
	int e2=0;
	/**
	 * @param base  The array to throw the array to be inserted
	 * @param arrays  The array to insert
	 * @param start The start location 
	 * @param end The end location
	 * @param star2t The start location of the 2nd array 
	 * @param end2 The end location of the 2nd array 
	 * Basic constructor with arrays' 1d merges
	 */
	InsertRunnable(double base[], double arrays [], int start, int end, int start2, int end2)  {
		basearray1d=base;
		array1d=arrays;
		basearray2d=null;
		array2d=null;
		cols=null;
		cols2=null;
		col=-1;
		s=start;
		e=end;
		s2=start2;
		e2=end2;
	};
	
	/**
	 * @param base  The 2d  base of the insertion 
	 * @param arrays The 1d array to insert
	 * @param column The column to insert to
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays'insertion
	 */
	InsertRunnable(double base[][], double arrays [],int column, int start, int end) {
		basearray1d=null;
		array1d=arrays;
		basearray2d=base;
		array2d=null;
		cols=null;
		cols2=null;
		col=column;
		s=start;
		e=end;
		s2=-1;
		e2=-1;
	};
	
	/**
	 * @param base  The 2d  base of the insertion 
	 * @param arrays The 2d array to insert
	 * @param columnbase The columns from the base insert to
	 * @param columns The columns to insert to from the inserted 2d array
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays'insertion
	 */
	InsertRunnable(double base[][], double arrays [][],int columnbase[], int columns[], int start, int end) {
		basearray1d=null;
		array1d=null;
		basearray2d=base;
		array2d=arrays;
		cols=columnbase;
		cols2=columns;
		col=-1;
		s=start;
		e=end;
		s2=-1;
		e2=-1;
	};
	

	@Override
	public void run() {
		
		//make some sensible checks
		
		if (s>e){
			
			throw new IllegalStateException(" The start of the loop in the de-append operation cannot be less/equal than the end");
		}
		
		// first define the method
		
		if (basearray1d!=null && array1d!=null && s2!=-1 && e2!=-1 && (e-s) == (e2-s2) ){
								

				for (int i=s ,n=s2 ; i < e; i++, n++) {
					basearray1d [i]=array1d[n];
				}		

			// end of array 1d
		} else if (basearray2d!=null && array1d!=null && basearray2d.length==array1d.length  &&col!=-1 && col < basearray2d[0].length) {
			
				for (int i=s; i < e; i++) {

					basearray2d [i][col]=array1d[i];

				}
				
		} else if (basearray2d!=null && array2d!=null && basearray2d.length==array2d.length  &&cols!=null &&cols2!=null && cols.length==cols2.length) {
			
			for (int i=s; i < e; i++) {
				
				for (int j=0; j < cols.length; j++) {
					
					basearray2d [i][cols[j]]=array2d[i][cols2[j]];
					
				}

			}				
					

			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" Insertion failed operation failed as there was no correct combination of object-arrays");
			
		}
		
	}

}
