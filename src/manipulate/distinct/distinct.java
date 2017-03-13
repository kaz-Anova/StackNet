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

package manipulate.distinct;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import exceptions.IllegalStateException;

/**
 * 
 * @author marios
 *<p> find distinct values of arrays </p>
 */
public class distinct {

	/**
	 * 
	 * @param Array : The Array We want to find Distinct values of 
	 * @return : Distinct values of an Array as a double Array
	 */
	
	public static double[] getdoubleDistinctswithlist( double Array[]){
		if (Array==null || Array.length==0){
			throw new IllegalStateException("array is empty");
		}
		

		  ArrayList<Double> sets= new ArrayList<Double>();
			
				for (int j=0; j <Array.length; j++ ){
				int	y=0;
				for (int i=0; i <sets.size(); i++ ){
				if(sets.get(i)==Array[j]) {
					y=1;
					break;
				}
				}
				if (y==0){
					sets.add(Array[j]);	
				}
			}
			
				double[] Arr= new double [sets.size()];
				for (int j=0; j <Arr.length; j++ ){
					Arr[j]=sets.get(j);
				}
			return Arr;

	}
	
	/**
	 * 
	 * @param Array : The Array We want to find Distinct values of 
	 * @return : Distinct values of an Array as a String Array
	 */
	
	public static String[] getStringDistinctswithlist( String Array[]){
		if (Array==null || Array.length==0){
			throw new IllegalStateException("array is empty");
		}
		

		  ArrayList<String> sets= new ArrayList<String>();
			
				for (int j=0; j <Array.length; j++ ){
					int y=0;
				for (int i=0; i <sets.size(); i++ ){
				if(sets.get(i).toString().equals(Array[j])) {
					y=1;
					break;
				}
				}
				if (y==0){
					sets.add(Array[j]);	
				}
			}
			
			return sets.toArray(new String[sets.size()]);

	}
	
	/**
	 * 
	 * @param Array : The Array We want to find Distinct values of 
	 * @return : Distinct values of an Array as a double Array derived from set- more efficient with bigger arrays
	 */
	
	public static double[] getdoubleDistinctset( double Array[]){
		if (Array==null || Array.length==0){
			throw new IllegalStateException("array is empty");
		}
		
		  Set<Double> sets= new HashSet<Double>();			
				for (int j=0; j <Array.length; j++ ){
					sets.add(Array[j]);		
			}
				
				double[] Arr= new double [sets.size()];
				int i=0;
				for (double d: sets){
					Arr[i]=d;
					i++;
				}
			sets=null;
			return Arr;

	}	
	
	/**
	 * 
	 * @param Array : The Array We want to find Distinct values of 
	 * @param column : column to use
	 * @return : Distinct values of an Array as a double Array derived from set- more efficient with bigger arrays
	 */
	
	public static double[] getdoubleDistinctset( double Array[][], int column){
		
		if (Array==null || Array.length==0){
			throw new IllegalStateException("array is empty");
		}
		if (column<0 || column >=Array[0].length){
			throw new IllegalStateException("column exceeds current arrays column dimension");
		}
		  Set<Double> sets= new HashSet<Double>();	
		  
				for (int j=0; j <Array.length; j++ ){
					sets.add(Array[j][column]);		
			}
				
				double[] Arr= new double [sets.size()];
				int i=0;
				for (double d: sets){
					Arr[i]=d;
					i++;
				}
			sets=null;
			return Arr;

	}
	/**
	 * 
	 * @param Array : The Array We want to find Distinct values of 
	 * @param column : column to use
	 * @return : Distinct values of an Array as a double Array derived from set- more efficient with bigger arrays
	 */
	
	public static String[] getdoubleDistinctset( String Array[][], int column){
		
		if (Array==null || Array.length==0){
			throw new IllegalStateException("array is empty");
		}
		if (column<0 || column >=Array[0].length){
			throw new IllegalStateException("column exceeds current arrays column dimension");
		}
		  Set<String> sets= new HashSet<String>();	
		  
				for (int j=0; j <Array.length; j++ ){
					sets.add(Array[j][column]);		
			}
				
				String[] Arr= new String [sets.size()];
				int i=0;
				for (String d: sets){
					Arr[i]=d;
					i++;
				}
			sets=null;
			return Arr;

	}		
	
	/**
	 * 
	 * @param Array : The Array We want to find Distinct values of 
	 * @return : Distinct values of an Array as a string Array derived from set- more efficient with bigger arrays
	 */
	
	public static String[] getstringDistinctset( String Array[]){
		if (Array==null || Array.length==0){
			throw new IllegalStateException("array is empty");
		}
		
		  Set<String> sets= new HashSet<String>();			
				for (int j=0; j <Array.length; j++ ){
					sets.add(Array[j]);		
			}
				
				String[] Arr= new String [sets.size()];
				int i=0;
				for (String d: sets){
					Arr[i]=d;
					i++;
				}
			sets=null;
			return Arr;

	}		
	
}
