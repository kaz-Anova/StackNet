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

package manipulate.merges;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to Merge arrays as in concatenating them vertically
 */
public class MergeRunnable implements Runnable{

    /**
     * The merge for 1-d arays will take place in this variable
     */
	double basearray1d[];
	/**
    * The merge for 2d Arrays will take place in this variable
    */
	double basearray2d[][];
    /**
     * first 1d Array to form the merge skeleton
     */
	double array1d[];
	/**
    * 2d Array to form the merge skeleton
    */
	double array2d[][];
    /**
     * first 1d Array to be merged under the first one
     */
	double array1d2[];
	/**
    * 2d Array to be merged under the first one
    */
	double array2d2[][];

	/**
	 * start location of the loop
	 */
	int s=0;
	/**
	 * end location of the loop
	 */	
	int e=0;
	/**
	 * @param base  The array to throw the arrays to be merged
	 * @param arrays  The base of the merge
	 * @param arrays2  The concatenated array
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' 1d merges
	 */
	MergeRunnable(double base[], double arrays [],double [] arrays2, int start, int end)  {
		basearray1d=base;
		array1d=arrays;
		array1d2=arrays2;
		basearray2d=null;
		array2d=null;
		array2d2=null;
		s=start;
		e=end;
	};
	/**
	 * @param base  The array to throw the arrays to be merged
	 * @param arrays  The base of the merge
	 * @param arrays2  The concatenated array
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' 2d merges
	 */
	MergeRunnable(double base[][], double arrays [][],double [][] arrays2, int start, int end) {
		basearray1d=null;
		array1d=null;
		array1d2=null;
		basearray2d=base;
		array2d=arrays;
		array2d2=arrays2;
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
		
		if (basearray1d!=null && array1d!=null && array1d2!=null  && basearray1d.length==(array1d.length +array1d2.length)  ){
								

				for (int i=s; i < e; i++) {
					if (i<array1d.length){
						basearray1d [i]=array1d[i];
					}else {
						basearray1d [i]=array1d2[i-array1d.length];
					}
					
				}		

			// end of array 1d
		} else if (basearray2d!=null && array2d!=null && array2d2!=null  && basearray2d.length==(array2d.length +array2d2.length) ) {
			
				for (int i=s; i < e; i++) {
					
					if (i<array2d.length){
						double temp []= manipulate.copies.copies.Copy(array2d[i]);
						basearray2d [i]=temp;
					}else {
						double temp []= manipulate.copies.copies.Copy(array2d2[i-array2d.length]);
						basearray2d [i]=temp;
					}
					
				}
				
				
					

			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" De-Append operation failed as there was no correct combination of object-arrays");
			
		}
		
	}

}
