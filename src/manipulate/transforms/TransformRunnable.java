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

package manipulate.transforms;

import java.util.ArrayList;

import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to provide a runnable class to the transform operations </p>
 */
public class TransformRunnable implements Runnable{

	double array1d[];
	double array2d[][];
	//ArrayList<Double> arraylist1d1d;
	ArrayList<ArrayList<Double>> arraylist1d2d;
	int s=0;
	int e=0;
	String type;
	public static double max=Double.POSITIVE_INFINITY;
	public static double min=Double.NEGATIVE_INFINITY;
	public static double pow=2.0;
	

	/**
	 * 
	 * @param array array The array to process (by reference)
	 * @param start The start location 
	 * @param end The end location
	 * @param scalar the value to make an operation with
	 * @param types : in 'log, logplusone, sqrt,  exp, abs, min, max, minmax,sin,cos,tan,tanh,sig, pow'
	 * Basic constructor with transform operations for 1d arrays
	 */
	TransformRunnable(double array [], int start, int end, String types) {
		array1d=array;
		s=start;
		e=end;
		type=types;
		
	};
	
	/**
	 * 
	 * @param array array The array to process (by reference)
	 * @param start The start location 
	 * @param end The end location
	 * @param scalar the value to make an operation with
	 * @param types : in 'log, logplusone, sqrt,  exp, abs, min, max, minmax,sin,cos,tan,tanh,sig, pow'
	 * Basic constructor with transform operations for 2d arrays
	 */ 
	
	TransformRunnable(double array [][], int start, int end, String types) {
		array2d=array;
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
		
		if (array1d!=null){
								
			
			if (type.equals("log")){
				
				for (int i=s; i < e; i++) {
					if (array1d[i]>0){
						array1d[i]=Math.log(array1d[i]);
					}
				
				}
				// end of natural logarithm transformation
             } else if (type.equals("logplusone")){
				
				for (int i=s; i < e; i++) {
					if (array1d[i]>0-1){
						array1d[i]=Math.log(array1d[i] + 1.0) ;
					}
				}	
				// end of log + 1
             } else if (type.equals("sqrt")){
 				
				for (int i=s; i < e; i++) {
					if (array1d[i]>0){
						array1d[i]=Math.sqrt(array1d[i]) ;
					}
				}			
				// end of square root
			} else if (type.equals("exp")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.exp(array1d[i]);
				}	
			// end of expotential transformation
			} else if (type.equals("abs")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.abs(array1d[i]);
				}	
			// end of absolute transformation	
				
			} else if (type.equals("abs")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.abs(array1d[i]);
				}	
			// end of absolute transformation		
			} else if (type.equals("min")){
				
				for (int i=s; i < e; i++) {
					if (array1d[i]<min){
						array1d[i]=min ;
					}
				}	
			// end of clip-min transformation
			} else if (type.equals("max")){
				
				for (int i=s; i < e; i++) {
					if (array1d[i]>max){
						array1d[i]=max ;
					}
				}	
			// end of clip-max transformation	
			} else if (type.equals("minmax")){
				
				for (int i=s; i < e; i++) {
					if (array1d[i]>max){
						array1d[i]=max ;
					} else if (array1d[i]<min){
						array1d[i]=min ;
					} 
				}	
			// end of clip-min and max transformation					
			} else if (type.equals("sin")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.sin(array1d[i]);
				}	
			// end of sine transformation
				
			} else if (type.equals("cos")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.cos(array1d[i]);
				}	
			// end of cosine transformation				
			} else if (type.equals("tan")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.tan(array1d[i]);
				}	
			// end of tangent transformation		
			} else if (type.equals("tahn")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.tanh(array1d[i]);
				}	
			// end of Hyperbolic tangent transformation	
				
			} else if (type.equals("sig")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=1.0 /( 1.0 + Math.exp(-array1d[i]) );

				}	
			// end of sigmoid tangent transformation	
			} else if (type.equals("pow")){
				
				for (int i=s; i < e; i++) {
					array1d[i]=Math.pow(array1d[i],pow);
				}	
			// end of power transformation					
				
		} else {
			
			// all object are null, throw error
			throw new IllegalStateException (" operation type-String is not of acceptable type ");	
			
			
		}
					
			// end of array 1d
		} else if (array2d!=null) {
			
			//		for (int j=0; j < array2d[i].length; j++) {
			
			if (type.equals("log")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					if (array2d[i][j]>0){
						array2d[i][j]=Math.log(array2d[i][j]);
					}
					}
				
				}
				// end of natural logarithm transformation
             } else if (type.equals("logplusone")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {					
					if (array2d[i][j]>0-1){
						array2d[i][j]=Math.log(array2d[i][j] + 1.0) ;
					}
					}
				}	
				// end of log + 1
             } else if (type.equals("sqrt")){
 				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {						
					if (array2d[i][j]>0){
						array2d[i][j]=Math.sqrt(array2d[i][j]) ;
					}
					}
				}			
				// end of square root
			} else if (type.equals("exp")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {							
					array2d[i][j]=Math.exp(array2d[i][j]);
					}
				}	
			// end of expotential transformation
			} else if (type.equals("abs")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {						
					array2d[i][j]=Math.abs(array2d[i][j]);
					}
				}	
			// end of absolute transformation	
				
			} else if (type.equals("abs")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {						
					array2d[i][j]=Math.abs(array2d[i][j]);
					}
				}	
			// end of absolute transformation		
			} else if (type.equals("min")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {						
					if (array2d[i][j]<min){
						array2d[i][j]=min ;
					}
					}
				}	
			// end of clip-min transformation
			} else if (type.equals("max")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {						
					if (array2d[i][j]>max){
						array2d[i][j]=max ;
					}
					}
				}	
			// end of clip-max transformation	
			} else if (type.equals("minmax")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
						
					if (array2d[i][j]>max){
						array2d[i][j]=max ;
					} else if (array2d[i][j]<min){
						array2d[i][j]=min ;
					} 
					
					}
				}	
			// end of clip-min and max transformation					
			} else if (type.equals("sin")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {					
					array2d[i][j]=Math.sin(array2d[i][j]);
				}	
				}
			// end of sine transformation
				
			} else if (type.equals("cos")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {					
					array2d[i][j]=Math.cos(array2d[i][j]);
				}	
				}
			// end of cosine transformation				
			} else if (type.equals("tan")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {					
					array2d[i][j]=Math.tan(array2d[i][j]);
				}
				}
			// end of tangent transformation		
			} else if (type.equals("tanh")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {					
					array2d[i][j]=Math.tanh(array2d[i][j]);
				}	
				}
			// end of Hyperbolic tangent transformation	
				
			} else if (type.equals("sig")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {
					array2d[i][j]=1.0 /( 1.0 + Math.exp(-array2d[i][j]) );
					}
				}	
			// end of sigmoid tangent transformation	
			} else if (type.equals("pow")){
				
				for (int i=s; i < e; i++) {
					for (int j=0; j < array2d[i].length; j++) {					
					array2d[i][j]=Math.pow(array2d[i][j],pow);
					}
				}	
			// end of power transformation					
				
		} else {
			
			// all object are null, throw error
			throw new IllegalStateException (" operation type-String is not of acceptable type ");	
	
			
			
		}	
				
			
			
			// end of array2d
		} 
		
		/*
		else if (arraylist1d2d!=null) {
			
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
		} 
			*/
		else {
			
			// all object are null, throw error
			throw new NullObjectException (" Scalar operation with scalar faild as the container-objects were all null");
			
		}
		
		
		
		
		
		
		
		// TODO Auto-generated method stub
		
	}

}
