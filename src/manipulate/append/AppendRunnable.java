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
 *<p> Purpose of the class is to provide a runnable class for array's appends
 */
public class AppendRunnable implements Runnable{

    /**
     * 2d base Array to append to 
     */	
	double array[][];
    /**
     * first 1d Array to append to 
     */
	double array1d[];
	/**
     * 2nd Array to appends to
     */
	double array1dtopass[];
	/**
    * 2d Array to append to 
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
	 * @param base  The array to apply the appends
	 * @param array  The 1d array to apply the appending
	 * @param array_topass The 1d array to append to the "array"
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with arrays' appends for 1d arrays
	 */
	AppendRunnable(double base[][], double arrays [],double array_topass[], int start, int end)  {
		array2d=null;
		array=base;
		array1d=arrays;
		array1dtopass=array_topass;
		s=start;
		e=end;
	};
	/**
	 * @param base  The array to use AND apply the appends
	 * @param array  The 1d array to apply the appending
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with operations for 2d arrays
	 */
	AppendRunnable(double base[][],double array_topass [], int start, int end) {
		array2d=null;
		array1dtopass=null;
		array=base;
		array1d=array_topass;
		s=start;
		e=end;
	};
	 
	/**
	 * @param base  The array to use AND apply the appends
	 * @param array  The 2d array to apply the appending
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor with operations for 2d arrays
	 */
	AppendRunnable(double base[][],double array_topass [][], int start, int end) {
		array1d=null;
		array1dtopass=null;
		array=base;
		array2d=array_topass;
		s=start;
		e=end;	
	};	

	@Override
	public void run() {
		
		//make some sensible checks
		
		if (s>e){
			
			throw new IllegalStateException(" The start of the loop in the append operation cannot be less/equal than the end");
		}
		
		// first define the method
		
		if (array!=null && array1d!=null && array1dtopass!=null && array.length==array1d.length && array1d.length==array1dtopass.length ){
								

				for (int i=s; i < e; i++) {
					array [i][0]=array1d[i];
					array [i][1]=array1dtopass[i];
				}		

			// end of array 1d
		} else if (array!=null&& array1d!=null && array.length==array1d.length ) {
			
				for (int i=s; i < e; i++) {
					double temp_first[]=array[i];
					double temp []=new double [temp_first.length + 1];
					for (int j=0; j < temp_first.length; j++) {
						temp[j]=temp_first[j];
					}
					//add element from single array
					temp[temp.length-1]=array1d[i];
					array[i]=temp;
				}
				
		} else if (array!=null&& array2d!=null && array.length==array2d.length ) {
			
			for (int i=s; i < e; i++) {
				double temp_first[]=array[i];
				double temp_second[]=array2d[i];
				double temp []=new double [temp_first.length +temp_second.length];
				for (int j=0; j < temp_first.length; j++) {
					temp[j]=temp_first[j];
				}
				//add second loop
				for (int j=0; j < temp_second.length; j++) {
					temp[j+temp_first.length]=temp_second[j];
				}
			
				array[i]=temp;
			}				

			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" Append operation failed as there was no correct combination of object-arrays");
			
		}
		
	}

}
