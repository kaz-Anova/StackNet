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
package manipulate.conversions;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to provide a runnable class to the copy operations
 */
public class TypeConversionRunnable implements Runnable{
    /**
     * 1d Array to convert
     */
	double array1d[];
    /**
     * 1d String Array to convert
     */
	String arrays1d[];	
	/**
     * 1d Array to copy convert to
     */
	double array1dtopass[];
	/**
     * 1d Array String convert to
     */
	String arrays1dtopass[];	
	/**
    * 2d Array to convert
    */
	double array2d[][];
	/**
	 * 2d string Array to convert
	 */
	String arrays2d[][];	
	/**
    * 2d Array to convert to
    */	
	double array2dtopass[][];
	/**
	 * 2d string Array to convert to
	 */	
    String arrays2dtopass[][];	

	/**
	 * start location of the loop
	 */
	int s=0;
	/**
	 * end location of the loop
	 */	
	int e=0;
	

	/**
	 * 
	 * @param array  The array to process (by reference)
	 * @param array_topass  The array to convert to (by reference) 
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor that converts a single double array to an equivalent String one
	 */
	TypeConversionRunnable(double array [],String array_topass[], int start, int end)  {
		array1d=array;
		arrays1dtopass=array_topass;
		s=start;
		e=end;
		arrays1d=null;	
		array1dtopass=null;
		array2d=null;
		arrays2d=null;	
		array2dtopass=null;
	    arrays2dtopass=null;

		
	};
	

	/**
	 * 
	 * @param array  The array to process (by reference)
	 * @param array_topass  The array to convert to (by reference) 
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor that converts single String array to an equivalent double one
	 */
	TypeConversionRunnable(String array [], double  array_topass[], int start, int end)  {
		
		arrays1d=array;
		array1dtopass=array_topass;
		s=start;
		e=end;
		array1d=null;	
		arrays1dtopass=null;
		array2d=null;
		arrays2d=null;	
		array2dtopass=null;
	    arrays2dtopass=null;

		
	};
	
	/**
	 * 
	 * @param array  The 2d array to process (by reference)
	 * @param array_topass  The 2d array to convert to (by reference) 
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor that converts a 2d double array to an equivalent String one
	 */
	TypeConversionRunnable(double array [][],String array_topass[][], int start, int end)  {
		array2d=array;
		arrays2dtopass=array_topass;
		s=start;
		e=end;
		arrays1d=null;	
		array1dtopass=null;
		array1d =null;
		arrays2d=null;	
		array2dtopass=null;
		arrays1dtopass =null;
	
	};
	

	/**
	 * 
	 * @param array  The array to process (by reference)
	 * @param array_topass  The array to convert to (by reference) 
	 * @param start The start location 
	 * @param end The end location
	 * Basic constructor that converts 2d double array to an equivalent String one
	 */
	TypeConversionRunnable(String array [][], double  array_topass[][], int start, int end)  {
		
		arrays2d=array;
		array2dtopass=array_topass;
		s=start;
		e=end;
		array1d=null;	
		arrays1dtopass=null;
		array2d=null;
		arrays1d =null;	
		array1dtopass =null;
	    arrays2dtopass=null;

		
	};	
	
	
	@Override
	public void run() {
		
		//make some sensible checks
		
		if (s>e){
			
			throw new IllegalStateException(" The start of the loop in the scalar operation cannot be less/equal than the end");
		}
		
		// first define the method
		
		if (array1d!=null && arrays1dtopass!=null && array1d.length==arrays1dtopass.length){
								
			
				
				for (int i=s; i < e; i++) {
					arrays1dtopass[i]=array1d[i] + "";
				}
			
				

			// end of array 1d
		} else if  (arrays1d!=null&& array1dtopass!=null && arrays1d.length==array1dtopass.length ) {
			
			    for (int i=s; i < e; i++) {
			    	array1dtopass[i]=Double.parseDouble(arrays1d[i]);
			     }

		}
			
		else if (array2d!=null&& arrays2dtopass!=null && array2d.length==arrays2dtopass.length ) {
			

				for (int i=s; i < e; i++) {
					double temp []=array2d[i];
					String temps []=arrays2dtopass[i];
					for (int j=0; j < temps.length; j++) {
						temps[j]=temp[j] +"";
					}
				}
				
		} else if (arrays2d!=null&& array2dtopass!=null && arrays2d.length==array2dtopass.length ) {
					

					for (int i=s; i < e; i++) {
						double temp []=array2dtopass[i];
						String temps []=arrays2d[i];
						for (int j=0; j < temps.length; j++) {
							temp[j]=Double.parseDouble(temps[j]);
						}
					}		
			// end of array2d
		
			
			
			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" Conversion failed because container-objects were null, different sizes or had incompatible types");
			
		}

		
	}

}
