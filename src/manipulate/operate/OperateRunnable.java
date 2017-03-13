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

package manipulate.operate;
import java.util.ArrayList;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to provide a runnable class for array's operations
 */
public class OperateRunnable implements Runnable{
    /**
     * 1d Array to operate to 
     */
	double array1d[];
	/**
     * 1d Array to operates from
     */
	double array1dtopass[];
	/**
    * 2d Array to operate to 
    */
	double array2d[][];
	/**
    * 2d Array tto operate from
    */	
	double array2dtopass[][];
	/**
    * 2d Arraylist to to operate to 
    */	
	ArrayList<ArrayList<Double>> arraylist1d2d;
	/**
    * 2d Arraylist to to operate from
    */		
	ArrayList<ArrayList<Double>> arraylist1d2dtopass;
	/**
	 * defines the operation type and can be any of "add, sub, mul, div"
	 */
	String type;
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
	 * @param array_topass The array to operate with (by reference) 
	 * @param start The start location 
	 * @param end The end location
	 * @param types The type operation can be "add, sub, mul, div" for addition, subtraction, multiplication, division
	 * Basic constructor with arrays' operations for 1d arrays
	 */
	OperateRunnable(double array [],double array_topass[], int start, int end, String types)  {
		array1d=array;
		array1dtopass=array_topass;
		s=start;
		e=end;
        type=types;
		
	};
	

	/**
	 * 
	 * @param array  The 2d array to process (by reference)
	 * @param array_topass  The 2d array to operate values to (by reference)  
	 * @param start The start location 
	 * @param end The end location
	 * @param types The type operation can be "add, sub, mul, div" for addition, subtraction, multiplication, division
	 * Basic constructor with operations for 2d arrays
	 */
	OperateRunnable(double array [][],double array_topass [][], int start, int end, String types) {
		array2d=array;
		array2dtopass=array_topass;
		s=start;
		e=end;
        type=types;
		
	};
	 
	
	/**
	 * 
	 * @param array  The arraylist to process (by reference)
	 * @param array_topass  The 2d arraylist to operate values to (by reference) 
	 * @param start The start location 
	 * @param end The end location
	 * @param types The type operation can be "add, sub, mul, div" for addition, subtraction, multiplication, division
	 * Basic constructor with  operations for 2d arraylists
	 */
	
	OperateRunnable( ArrayList<ArrayList<Double>> array,ArrayList<ArrayList<Double>> array_topass, int start, int end, String types) {
		
		arraylist1d2d=array;
		arraylist1d2dtopass=array_topass;
		s=start;
		e=end;
        type=types;
		
	};	
	
	
	@Override
	public void run() {
		
		//make some sensible checks
		
		if (s>e){
			
			throw new IllegalStateException(" The start of the loop in the scalar operation cannot be less/equal than the end");
		}
		
		// first define the method
		
		if (array1d!=null && array1dtopass!=null && array1d.length==array1dtopass.length){
								
			
				
			if (type.equals("add")){
				
				for (int i=s; i < e; i++) {
					array1d[i]+=array1dtopass[i];
				}
				
				// end of addition
			} else if (type.equals("sub")){
				
				for (int i=s; i < e; i++) {
					array1d[i]-=array1dtopass[i];
				}
				
				
				// end of substraction
			}else if (type.equals("mul")){
				
				for (int i=s; i < e; i++) {
					array1d[i]*=array1dtopass[i];
				}
			
				// end of multiplication
			}
			
		
		else if (type.equals("div")){
			
			for (int i=s; i < e; i++) {
				array1d[i]/=array1dtopass[i];
			}
		
			// end of division
		
		} else {
			
			
			// all object are null, throw error
			throw new NullObjectException (" operation type-String for scalar operation cannot be null ");	
			
			
		}
			
				

			// end of array 1d
		} else if (array2d!=null&& array2dtopass!=null && array2d.length==array2dtopass.length ) {
			
			if (type.equals("add")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					    array2d[i][j]+=array2dtopass[i][j];
					}
				}
				
				// end of addition
			} else if (type.equals("sub")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					    array2d[i][j]-=array2dtopass[i][j];
					}
				}
				
				
				// end of substraction
			}else if (type.equals("mul")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					   array2d[i][j]*=array2dtopass[i][j];
					}
				}
				// end of multiplication
				
			}
			
		
		else if (type.equals("div")){
			
			for (int i=s; i < e; i++) {
				for (int j=0; j < array2d[i].length; j++) {
				     array2d[i][j]/=array2dtopass[i][j];
				}
			}
		
			// end of division
		
		} else {
			
			
			// all object are null, throw error
			throw new NullObjectException (" operation type-String for scalar operation cannot be null ");	
			
			
		}	
			
			

		
			// end of array2d
		} else if (arraylist1d2d!=null && arraylist1d2dtopass!=null && arraylist1d2d.size()==arraylist1d2dtopass.size()) {
			

			if (type.equals("add")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
						arraylist1d2d.get(i).set(j,arraylist1d2d.get(i).get(j) + arraylist1d2d.get(i).get(j));
					}
				}
				
				// end of addition
			} else if (type.equals("sub")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
						arraylist1d2d.get(i).set(j,arraylist1d2d.get(i).get(j) - arraylist1d2d.get(i).get(j));
					}
				}
				
				
				// end of substraction
			}else if (type.equals("mul")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
						arraylist1d2d.get(i).set(j, arraylist1d2d.get(i).get(j) * arraylist1d2d.get(i).get(j));
					}
				}
				// end of multiplication
				
			}
			
		
		else if (type.equals("div")){
			
			for (int i=s; i < e; i++) {
				for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
					arraylist1d2d.get(i).set(j, arraylist1d2d.get(i).get(j) / arraylist1d2d.get(i).get(j));
				}
			}
		
			// end of division
		
		} else {
			
			
			// all object are null, throw error
			throw new NullObjectException (" operation type-String for scalar operation cannot be null ");	
			
			
		}	
				
			
			// end of arraylist 2d
		} else {
			
			// all object are null, throw error
			throw new NullObjectException (" copy operation failed as either the copy-to or copy-from (or both) container-objects were null or of differebr sizes");
			
		}
		
		
		
		
		
		
		
		// TODO Auto-generated method stub
		
	}

}
