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

package manipulate.append;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to de-append arrays as in removing the last column of a 2d array
 */
public class DeAppendRunnable implements Runnable{


    /**
     * first 1d Array to de-append to 
     */
	double array1d[];
	/**
    * 2d Array to de-append from
    */
	double array2d[][];

	/**
	 * start location of the loop
	 */
	int s=0;
	/**
	 * end location of the loop
	 */	
	int e=0;
	/**
	 * @param base  The array to form the de-appendings
	 * @param array  The 2d array to apply the de-appending
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' de-appendings for 2d arrays into 1d arrays.
	 */
	DeAppendRunnable(double base[], double arrays [][], int start, int end)  {
		array2d=arrays;
		array1d=base;
		s=start;
		e=end;
	};
	/**
	 * @param base  The array to use AND form the de-appendings
	 * @param array  The 1d array to apply the appending
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' de-appendings for 2d arrays into 2d arrays
	 */
	DeAppendRunnable(double arrays [][], int start, int end) {
		array2d=arrays;
		array1d=null;
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
		
		if (array1d!=null && array2d!=null && array1d.length==array2d.length  ){
								

				for (int i=s; i < e; i++) {
					array1d [i]=array2d[i][0];
				}		

			// end of array 1d
		} else if (array2d!=null  ) {
			
				for (int i=s; i < e; i++) {
					double temp_first[]=array2d[i];
					double temp []=new double [temp_first.length - 1];
					
					for (int j=0; j < temp_first.length-1; j++) {
						temp[j]=temp_first[j];
					}

					array2d[i]=temp;
					
				}
				
				
					

			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" De-Append operation failed as there was no correct combination of object-arrays");
			
		}
		
	}

}
