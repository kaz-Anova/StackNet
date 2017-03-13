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

package manipulate.scalars;

import java.util.ArrayList;

import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to provide a runnable class to the scalar operations
 */
public class ScalarRunnable implements Runnable{

	double array1d[];
	double array2d[][];
	//ArrayList<Double> arraylist1d1d;
	ArrayList<ArrayList<Double>> arraylist1d2d;
	int s=0;
	int e=0;
	double value=0.0;
	String type;
	

	/**
	 * 
	 * @param array array The array to process (by reference)
	 * @param start The start location 
	 * @param end The end location
	 * @param scalar the value to make an operation with
	 * @param types : in "add, sub, mul, div"
	 * Basic constructor with scalar operations for 1d arrays
	 */
	ScalarRunnable(double array [], int start, int end, double scalar, String types) {
		array1d=array;
		s=start;
		e=end;
		value=scalar;
		type=types;
		
	};
	

	
	/**
	 * 
	 * @param array array The array to process (by reference)
	 * @param start The start location 
	 * @param end The end location
	 * @param scalar the value to make an operation with
	 * @param types : in "add, sub, mul, div"
	 * Basic constructor with scalar operations for 2d arrays
	 */
	ScalarRunnable(double array [][], int start, int end, double scalar, String types) {
		array2d=array;
		s=start;
		e=end;
		value=scalar;
		type=types;
		
	};
	 
	
	
	/**
	 * 
	 * @param array array The array to process (by reference)
	 * @param start The start location 
	 * @param end The end location
	 * @param scalar the value to make an operation with
	 * @param types : in "add, sub, mul, div"
	 * Basic constructor with scalar operations for 2d arraylists
	 */
	
	ScalarRunnable( ArrayList<ArrayList<Double>> array, int start, int end, double scalar, String types) {
		
		arraylist1d2d=array;
		s=start;
		e=end;
		value=scalar;
		type=types;
		
	};	
	
	
	@Override
	public void run() {
		
		//make some sensible checks
		
		if (s>e){
			
			throw new IllegalStateException(" The start of the loop in the scalar operation cannot be less/equal than the end");
		}
		
		// first define the method
		
		if (array1d!=null){
								
			
			if (type.equals("add")){
				
				for (int i=s; i < e; i++) {
					array1d[i]+=value;
				}
				
				// end of addition
			} else if (type.equals("sub")){
				
				for (int i=s; i < e; i++) {
					array1d[i]-=value;
				}
				
				
				// end of substraction
			}else if (type.equals("mul")){
				
				for (int i=s; i < e; i++) {
					array1d[i]*=value;
				}
			
				// end of multiplication
			}
			
		
		else if (type.equals("div")){
			
			for (int i=s; i < e; i++) {
				array1d[i]/=value;
			}
		
			// end of division
		
		} else {
			
			
			// all object are null, throw error
			throw new NullObjectException (" operation type-String for scalar operation cannot be null ");	
			
			
		}
				
				

			// end of array 1d
		} else if (array2d!=null) {
			
			
			
			if (type.equals("add")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					    array2d[i][j]+=value;
					}
				}
				
				// end of addition
			} else if (type.equals("sub")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					    array2d[i][j]-=value;
					}
				}
				
				
				// end of substraction
			}else if (type.equals("mul")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					   array2d[i][j]*=value;
					}
				}
				// end of multiplication
				
			}
			
		
		else if (type.equals("div")){
			
			for (int i=s; i < e; i++) {
				for (int j=0; j < array2d[i].length; j++) {
				     array2d[i][j]/=value;
				}
			}
		
			// end of division
		
		} else {
			
			
			// all object are null, throw error
			throw new NullObjectException (" operation type-String for scalar operation cannot be null ");	
			
			
		}	
			
			
			
			
			
			
			
			
			
			// end of array2d
		} else if (arraylist1d2d!=null) {
			
			if (type.equals("add")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
						arraylist1d2d.get(i).set(j, arraylist1d2d.get(i).get(j) + value);
					}
				}
				
				// end of addition
			} else if (type.equals("sub")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
						arraylist1d2d.get(i).set(j, arraylist1d2d.get(i).get(j) - value);
					}
				}
				
				
				// end of substraction
			}else if (type.equals("mul")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
						arraylist1d2d.get(i).set(j, arraylist1d2d.get(i).get(j) * value);
					}
				}
				// end of multiplication
				
			}
			
		
		else if (type.equals("div")){
			
			for (int i=s; i < e; i++) {
				for (int j=0; j < arraylist1d2d.get(i).size(); j++) {
					arraylist1d2d.get(i).set(j, arraylist1d2d.get(i).get(j) / value);
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
			throw new NullObjectException (" Scalar operation with scalar faild as the container-objects were all null");
			
		}
		
		
		
		
		
		
		
		// TODO Auto-generated method stub
		
	}

}
