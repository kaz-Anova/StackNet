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


package crossvalidation.metrics;
import java.util.Arrays;
import java.util.Comparator;




/**
 * 
 * <p> Class that computes Precision for top K scored instances, provided 
 * a score array, a label array and a id-type array 
 */


public class PrecisionK {
	/**
	 * 
	 * @param array : The array from which we will create a 2d array with 1 column.
	 * @return : a 2d double array.
	 * <p> the purpose of this method to convert a single double array to one-column 2d double array.
	 */
	
	public static double [][] maked2dfromsingleedouble (double array[]) {
		double arraysingle[][]= new double [array.length][1];
		
		for (int i=0; i < arraysingle.length; i++) {
			arraysingle[i][0]=array[i];
		}
		return arraysingle;
		
	}
	/**
	 * @param one : The main double 2d Array
	 * @param two : The Second double  Array to merged on the right of the first one
	 * <p>This class will perform horizontal stack.E.g a set will be joined to another set on the right
	 * @return : The consolidated 2d double Matrix
	 */
	
	public static double[][] Stackdouble (double one[][], double two[]) {
		
		if(one==null) {
			return maked2dfromsingleedouble(two);
		} else {

		/* Throw Exception if the length of the 2 arrays is not the same*/
	       if (one.length!=two.length ){
		
					throw new IllegalStateException("The two arrays need to have the same length (rows)");
		
			}
	       
	       /* initiate the new array */
	       double consolidated[][] = new double [one.length] [one[0].length +1];
	   	
	   	/* fill in the array */
	   	
	   	for (int i=0; i <consolidated.length; i++ ){
	   		consolidated[i][one[0].length]=two[i];
	   		for (int j=0; j < one[0].length; j++){
	   			consolidated[i][j]=one[i][j];
	   		}
	   		
	   		
	   	}
	       
		return consolidated;
		
		}
		
	}
	
	/**
	 * @param one : The main double single Array
	 * @param two : The Second double 2d Array to merged on the right of the first one
	 * <p>This class will perform horizontal stack.E.g a set will be joined to another set on the right
	 * @return : The consolidated 2d double Matrix
	 */
	
	public static double[][] Stackdouble (double one[], double two[][]) {
		

		/* Throw Exception if the length of the 2 arrays is not the same*/
	       if (one.length!=two.length ){
				
	
					throw new IllegalStateException("The two arrays need to have the same length (rows)");
	
			}
	       
	       /* initiate the new array */
	       double consolidated[][] = new double [one.length] [two[0].length +1];
	   	
	   	/* fill in the array */
	   	
	   	for (int i=0; i <consolidated.length; i++ ){
	   		consolidated[i][0]=one[i];
	   		for (int j=0; j < two[0].length; j++){
	   			consolidated[i][j+1]=two[i][j];
	   		}
	   		
	   		
	   	}
	       
		return consolidated;		
		
	}
	/**
	 * @param one : The main double  Array
	 * @param two : The Second double  Array to merged on the left of the first one
	 * <p>This class will perform horizontal stack.E.g a set will be joined to another set on the right
	 * @return : The consolidated 2d double Matrix
	 */
	
	public static double[][] Stackdouble (double one[], double two[]) {
		

		/* Throw Exception if the length of the 2 arrays is not the same*/
	       if (one.length!=two.length ){
				
	
					throw new IllegalStateException("The two arrays need to have the same length (rows)");
			
			}
	       
	       /* initiate the new array */
	       double consolidated[][] = new double [one.length] [2];
	   	
	   	/* fill in the array */
	   	
	   	for (int i=0; i <consolidated.length; i++ ){
	   		consolidated[i][0]=one[i];
	   		consolidated[i][1]=two[i];
	   		
	   		
	   	}
	       
		return consolidated;		
		
	}
/**
 * 
 * @param prediction : The array holding the score. It is assumed that higher score is associated with higher
 * chance to be the higher values of Y.
 * @param Y : is the label array that needs to be either 0 or 1.
 * @param id : The array referring to the main key.
 * @param k : number of best scores to keep for each key instance when calculating the precision
 * @param assume_not : If true lesser items than k are not counted as false-negatives. E.g. if a key has only three items ,the average for that customer will be based on 3.
 * @return : The average precision for the top k scored elements.
 */
	
	public double compute_precision(double[] prediction, double[] Y, double id[] , int k, boolean assume_not) {
		double precision=0;
		
	// Initially we check if the values of the Target are 1 or 0 as they normally should be. It will throw exception if it doesn't. We also store it as a new array.
		
		for (int i=0 ; i< Y.length; i++){
			
			if (Y[i]!=0.0 && Y[i]!=1.0 ){
					throw new IllegalStateException("Your Y Variable need to be bianry and have"
							+ " only the values of '0' and '1'");
		
			}
		}
		
		/* Merge columns together with the scope of sorting them based on the score */
		
		double sort_set[][]=Stackdouble(id, prediction);
		sort_set=Stackdouble(sort_set, Y);
		
		sort2ddouble2cols(sort_set, 0, 1, true, false);
		
		double key=sort_set[0][0];
		double counter =0;
		int current_k=0;
		precision=precision+sort_set[0][2];
			counter++;
			current_k++;
		
		
		//loop through the whole set
		
		for (int i=1; i <sort_set.length ; i++) {
			// check value
			if (key==sort_set[i][0] && current_k<k){
				precision=precision+sort_set[i][2];
				counter++;
				current_k++;
				
			} else if (key==sort_set[i][0] && current_k>=k){
				//do nothing
			
			}else if (key!=sort_set[i][0] && current_k<k){
				key=sort_set[i][0];
				precision=precision+sort_set[i][2];
				counter++;
				if (!assume_not){
					counter=counter+k-current_k;
				}
				current_k=1;
				
			} else {
				key=sort_set[i][0];
				precision=precision+sort_set[i][2];
				counter++;
				current_k=1;
			}
			
			
		}
		
		precision/=counter;
		
		sort_set=null;

		return precision;
	}
	/** 
	 * @param array : The double array to be sorted
	 * @param x : The  column by which we do the sorting
	 * @param y : The second column by which we do the sorting
	 * @param asc1 : When true, we sort the first column in ascending order else in descending
	 * @param asc2 : When true, we sort the second column in ascending order else in descending
	 * <p> Sorts the same double 2d array that is provided in an descending manner based on two columns.
	 */
	
	public static void sort2ddouble2cols (double array [][], final int x, final int y,final boolean asc1,  final boolean asc2){
		 
		Arrays.sort(array, new Comparator<double[]>() {
		    @Override
		    public int compare(double[] o1, double[] o2) {
		    	if (o1[x]==o2[x]){
		    		if (asc2==true) {
		    			return Double.compare( o1[y],o2[y]);
		    		} else{
		    			return Double.compare( o2[y],o1[y]);
		    		}
		    	} else{
		    		
		    		if (asc1==true) {
		    			return Double.compare( o1[x],o2[x]);
		    		} else{
		    			return Double.compare( o2[x],o1[x]);
		    		}
		    		
		    	}
		    }
		});

	}
	/**
	 * 
	 * @param prediction : The array holding the score. It is assumed that higher score is associated with higher
	 * chance to be the higher values of Y.
	 * @param Y : is the label array that needs to be either 0 or 1.
	 * @param id : The array referring to the main key.
	 * @param k : number of best scores to keep for each key instance when calculating the precision
	 * @return : The average precision for the top k scored elements that have at least k elements.
	 */
		
		public double compute_Literalprecision(double[] prediction, double[] Y, double id[] , int k) {
			
			double precision=0;
			
		// Initially we check if the values of the Target are 1 or 0 as they normally should be. It will throw exception if it doesn't. We also store it as a new array.
			
			for (int i=0 ; i< Y.length; i++){
				
				if (Y[i]!=0.0 && Y[i]!=1.0 ){
						throw new IllegalStateException("Your Y Variable need to be bianry and have"
								+ " only the values of '0' and '1'");
			
				}
			}
			
			/* Merge columns together with the scope of sorting them based on the score */
			
			double sort_set[][]=Stackdouble(id, prediction);
			sort_set=Stackdouble(sort_set, Y);
			
			sort2ddouble2cols(sort_set, 0, 1, true, false);
			
			double grand_precision=0;
			double grand_count=0;
			double key=sort_set[0][0];
			double counter =0;
			int current_k=0;
			precision=precision+sort_set[0][2];
				counter++;
				current_k++;
			
			
			//loop through the whole set
			
			for (int i=1; i <sort_set.length ; i++) {
				// check value
				if (key==sort_set[i][0] && current_k<k){
					precision=precision+sort_set[i][2];
					counter++;
					current_k++;
					
				} else if (key==sort_set[i][0] && current_k>=k){
					//do nothing
				
				}else if (key!=sort_set[i][0] && current_k<k){
					
					//reset, not enough cases
					precision=0;
					counter=0;
					key=sort_set[i][0];
					precision=precision+sort_set[i][2];
					counter++;
					current_k=1;
					
				} else {
					
					grand_precision+=precision;
					grand_count+=counter;
					//resetting
					precision=0;
					counter=0;
					key=sort_set[i][0];
					precision=precision+sort_set[i][2];
					counter++;
					current_k=1;
					
				}
				
				
			}
			
			grand_precision/=grand_count;
			
			sort_set=null;

			return grand_precision;
		}

}