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


package preprocess.binning;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import exceptions.IllegalStateException;

/**
 * 
 *Implements binning of a variable based on equal population
 *
 */

public class equalsizebinner implements Serializable {

	/**
	 * serialVersionUID
	 */
	private static final long serialVersionUID = -8759630571881379877L;
	/**
	 * Number of bins to create based on equal populations
	 */
	public static int bins=2;
	/**
	 * Index to be used for when there is a value outside the range of possible values
	 */
	public int other=9999999;
	/**
	 * minimum value to consider when creating the bins
	 */
	public static double minimum_value=Double.NEGATIVE_INFINITY;	
	/**
	 * maximum value to consider when creating the bins
	 */
	public static double maximum_value=Double.POSITIVE_INFINITY;		
	/**
	 * Object that has the same size as @bins and will have [minimum or equal vale to be in the bin, bin type]
	 */
	
	private double bin_holder [][]; 
	
	/**
	 * Object that has the same size as @bins and will have [minimum or equal vale to be in the bin, bin name]
	 */
	
	private String strig_bin_holder [][]; 
	/**
	 * 
	 * @param variable : variable to bin 
	 * @param bin : number of bins to use 
	 */
	public void fit ( double []  variable, int bin){
		
		if (variable.length<=bin){
			throw new IllegalStateException(" The size of the variable cannot be less than the number of bins" );	
		}
		// make a copy of the new array

		double new_array[]=manipulate.copies.copies.Copy(variable);
		// check unique values in the array. 
		// will throw error 
		HashSet<Double> has= new HashSet<Double> ();
		for (int i=0; i < new_array.length; i++){
			has.add(new_array[i]);
		}
		if (has.size()<=1){
			throw new IllegalStateException(" Variable needs to have more 2 or more bins" );	
		}
		// check if number of bins less than in the array
		if (has.size()<bin){
			bins=has.size();
		} else {
			bins=bin;
		}
	
		int calculate_bin_size=(int)Math.floor(((double)(new_array.length)/(double)(bins))); /// every how many rows we have the next bin
		Arrays.sort(new_array); //sort array
		//some default values
	    int row_index=0;
	    int index_holder=0;
	    double previous_value=minimum_value;			
		ArrayList<double []> bin_details= new ArrayList<double []>();
		ArrayList<String []> bin_details_str= new ArrayList<String []>();		


	    for  (int s=0; s< bins; s++){
	    	 row_index+=calculate_bin_size;
	    	 
	    	 if (row_index>=new_array.length){
	    		 row_index=new_array.length-1;
	    	 }
	    	 double value=new_array[row_index];
	    	 if (value>previous_value){

	    		  previous_value=value;
	    		  double [] newarray=new double [] {value,index_holder};
	    		  String [] newarraystr=new String [] {value + "","<=" +value };	
		           bin_details.add(newarray);
		           bin_details_str.add(newarraystr)  	  ; 	
		    	   index_holder+=1 ;		           
	    		  
	    	 }
	    	 if (row_index==new_array.length-1){
	    		 break;
	    	 }
	    }
	    // last values
	    
		  //double [] newarray=new double [] {maximum_value,index_holder};
		 // String [] newarraystr=new String [] {maximum_value + "",">" +maximum_value };	 
		   
	    //bin_details.add(newarray);
	    //bin_details_str.add(newarraystr)    ;
	    
	    // passes the results onto an array
	    bin_holder= new double [bin_details.size()][2];
	    strig_bin_holder= new String [bin_details_str.size()][2];	
	    System.out.println(" Binning parameters ");
	    for (int j=0; j <bin_holder.length; j++ ){
	    	bin_holder[j]=bin_details.get(j);
	    	strig_bin_holder[j]=bin_details_str.get(j);
	    	System.out.println(Arrays.toString(strig_bin_holder[j]));
	    }
	    
	    bins=this.bin_holder.length;
		
	}
	
	/**
	 * 
	 * @param variable : array to be binned based on the @fit method
	 * @return the transformed, binned array
	 */
	public double [] transform ( double []  variable){
		
		if (bin_holder==null){
			throw new IllegalStateException(" The fit() method needs to be run first");	
		}
		double binned_array []= new double [variable.length];
		if (bin_holder.length<=1){
			binned_array=manipulate.copies.copies.Copy(variable);
			
		} else {
			
	          for (int s=0; s<variable.length; s++){
	        	  double val=variable[s];
	        	  boolean isin=false;
	        	  for (int d=0; d< bins; d++){
	        		
	        		 if (val<=bin_holder[d][0]){
	        			 isin=true;
	        			 binned_array[s]=bin_holder[d][1] ; 
	        			 break;
	        		 }
	        	  }
	        	  if (isin==false){
	        		  binned_array[s]=bin_holder[bin_holder.length-1][1]; 
	        	  }
	          }
	               	
		}
		
		return binned_array;
	}

	/**
	 * 
	 * @param variable : array to be binned based on the @fit method
	 * @return the transformed, binned array
	 */
	public int [] transformint ( double []  variable){
		
		if (bin_holder==null){
			throw new IllegalStateException(" The fit() method needs to be run first");	
		}
		int binned_array []= new int [variable.length];
		if (bin_holder.length<=1){
			for (int s=0; s<variable.length; s++){
				binned_array[s]=(int)variable[s];
			}

			
		} else {
			
	          for (int s=0; s<variable.length; s++){
	        	  double val=variable[s];
	        	  boolean isin=false;
	        	  for (int d=0; d< bins; d++){
	        		
	        		 if (val<=bin_holder[d][0]){
	        			 isin=true;
	        			 binned_array[s]=(int)bin_holder[d][1] ; 
	        			 break;
	        		 }
	        	  }
	        	  if (isin==false){
	        		  binned_array[s]=(int) bin_holder[bin_holder.length-1][1]; 
	        	  }
	          }
	               	
		}
		
		return binned_array;
	}	
	/**
	 * 
	 * @param variable : array to be binned based on the @fit method
	 * @return the transformed, binned array
	 */
	public String [] transformstr ( double []  variable){
		
		if (bin_holder==null){
			throw new IllegalStateException(" The fit() method needs to be run first");	
		}
		String binned_array []= new String [variable.length];
		if (bin_holder.length<=1){
			for (int s=0; s<variable.length; s++){
				binned_array[s]=variable[s]+ "";
			}
			
		} else {
			
	          for (int s=0; s<variable.length; s++){
	        	  double val=variable[s];
	        	  boolean isin=false;
	        	  for (int d=0; d< bins; d++){
	        		
	        		 if (val<=bin_holder[d][0]){
	        			 isin=true;
	        			 binned_array[s]=strig_bin_holder[d][1] ; 
	        			 break;
	        		 }
	        	  }
	        	  if (isin==false){
	        		  binned_array[s]=strig_bin_holder[strig_bin_holder.length-1][1]; 
	        	  }
	          }
	               	
		}
		
		return binned_array;
	}
}
