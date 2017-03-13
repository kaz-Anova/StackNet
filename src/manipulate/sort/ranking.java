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

package manipulate.sort;

/**
 * 
 * @author mariosm
 *
 *Class to convert arrays to ranks
 */
public class ranking {
	/**
	 * 
	 * @param array : the array to rank
	 * @param normal : if true ranking is in ascending manner else in descending
	 * @return a ranked array
	 */
	public static  double[] create_rank (double array[], boolean normal){
		
	    double values []=manipulate.copies.copies.Copy(array);
	    if (!normal){
	    	 for (int i=0; i <values.length; i++ ){
	    		 values[i]=-values[i];
	 	    }
	    }
	    double target []=new double [values.length];
	    for (int i=0; i <target.length; i++ ){
	    	target[i]=i;
	    }
	    // Sort based on  value
	    manipulate.sort.mergesorts.mergesort(values,target,0,values.length-1 );
	    double ranking[]=new double [values.length];
	    for (int i=0; i <ranking.length; i++ ){
	    	ranking[i]=i;
	    }
	    // Sort based on initial order
	    manipulate.sort.mergesorts.mergesort(target,ranking,0,target.length-1 );	    
	    target=null;
	    values=null;	    
		return ranking;
	}
	
	
	/**
	 * 
	 * @param values : the array to rank
	 * @param normal : if true ranking is in ascending manner else in descending
       <p> results are saved in the given array 
	 */
	public static void create_rankthis (double values[], boolean normal){
		
	    if (!normal){
	    	 for (int i=0; i <values.length; i++ ){
	    		 values[i]=-values[i];
	 	    }
	    }
	    double target []=new double [values.length];
	    for (int i=0; i <target.length; i++ ){
	    	target[i]=i;
	    }
	    // Sort based on  value
	    manipulate.sort.mergesorts.mergesort(values,target,0,values.length-1 );
	    for (int i=0; i <values.length; i++ ){
	    	values[i]=i;
	    }
	    // Sort based on initial order
	    manipulate.sort.mergesorts.mergesort(target,values,0,target.length-1 );	    
	    target=null;    
	}
	
	
	/**
	 * 
	 * @param array : the array to rank by column
	 * @param normal : if true ranking is in ascending manner else in descending
	 * @return a ranked array
	 */
	public static double[][] create_rank_array (double array[][], boolean normal){
		
		double rank_array [][]= new double [array.length][array[0].length];
		
		for (int j=0; j<array[0].length; j++ ){
			
			double temp []=manipulate.select.columnselect.ColumnSelect(array, j);
			//rank this
			create_rankthis (temp, normal);
			manipulate.insertions.inserts.Insertthis(rank_array, temp, j);
			
		}
		return rank_array;
		
	}
	
	/**
	 * 
	 * @param rank_array : the array to rank by column
	 * @param normal : if true ranking is in ascending manner else in descending
	 */
	public static void create_rank_array_this (double rank_array[][], boolean normal){
		
		
		for (int j=0; j<rank_array[0].length; j++ ){
			
			double temp []=manipulate.select.columnselect.ColumnSelect(rank_array, j);
			//rank this
			create_rankthis (temp, normal);
			manipulate.insertions.inserts.Insertthis(rank_array, temp, j);
			
		}

		
	}	
}
