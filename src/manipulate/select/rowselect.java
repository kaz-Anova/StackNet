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

package manipulate.select;
import exceptions.LessThanMinimum;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make row selections

 */
public class rowselect {
	

	
	//-----------------------------------------1d rows selects on 2d--------------------------------------------------//
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param row  row to select
	 * @param threads number of threads to use
	 * @return the 1d sub-array
	 */
	public static double [] RowSelect( double array_append_to [][], int row,  int threads) {
		

		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0 )   ){
			
			throw new NullObjectException (" The Array to select from is null or empty or the selection ");
		}
		
	    if ( row <0){
	    	throw new LessThanMinimum (row,0);
	    }
	    if ( row >array_append_to.length){
	    	throw new IllegalStateException (" The given row exceeds current array length");
	    }
	    	    
	    //threads' checks
	    
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to[row].length) {
			threads=array_append_to[row].length;
		}	
		
		// initialize the new array to append to
		double new_array_to_copy_to1d []=  new double [array_append_to[row].length];
		
		int length_of_each_threaded_pass = new_array_to_copy_to1d.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=new_array_to_copy_to1d.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new RowSelectRunnable(new_array_to_copy_to1d  , array_append_to,row, locations[n][0], locations[n][1]));
			thread_array[n].start();
		}
		
		//start the threads
		
		for (int n=0; n <threads; n++ ){
			try {
				thread_array[n].join();
			} catch (InterruptedException e) {
			   System.out.println(e.getMessage());
			}
			
		}
		locations=null;
	return new_array_to_copy_to1d;
		//end of copy method
	
		}
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param row  row to select
	 * @param threads number of threads to use
	 * @return the 1d sub-array
	 */
	public static double [] RowSelect( double array_append_to [][], int row) {
		

		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0 )   ){
			
			throw new NullObjectException (" The Array to select from is null or empty or the selection ");
		}
		
	    if ( row <0){
	    	throw new LessThanMinimum (row,0);
	    }
	    if ( row >array_append_to.length){
	    	throw new IllegalStateException (" The given row exceeds current array length");
	    }
	    	    
		
		// initialize the new array to append to
		double new_array_to_copy_to1d []=  manipulate.copies.copies.Copy(array_append_to[row]);
		

	return new_array_to_copy_to1d;
		//end of copy method
	
		}	
	
	
	/**
	 * @param array_append_to 2d Array to select to
	 * @param row  rows to select
	 * @param threads number of threads to use
	 * @return the 2d sub-array
	 */
	public static double [][] RowSelect2d( double array_append_to [][], int row[],  int threads) {
		
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0 )   ){
			
			throw new NullObjectException (" The Array to select from is null or empty or the selection ");
		}
		

	    if ( row.length >array_append_to.length){
	    	throw new IllegalStateException (" The given rows exceed current array length");
	    }
	    	    
	    //threads' checks
	    
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>row.length) {
			threads=row.length;
		}	
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  new double [ row.length][];
		
		int length_of_each_threaded_pass = row.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]= row.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new RowSelectRunnable(new_array_to_copy_to2d  , array_append_to,row, locations[n][0], locations[n][1]));
			thread_array[n].start();
		}
		
		//start the threads
		
		for (int n=0; n <threads; n++ ){
			try {
				thread_array[n].join();
			} catch (InterruptedException e) {
			   System.out.println(e.getMessage());
			}
			
		}
		locations=null;
	return new_array_to_copy_to2d;
		//end of copy method
	
		}
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param row  rows to select
	 * @return the 2d sub-array
	 */
	public static double [][] RowSelect2d( double array_append_to [][], int row[]) {
		
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0 )   ){
			
			throw new NullObjectException (" The Array to select from is null or empty or the selection ");
		}
		

	    if ( row.length >array_append_to.length){
	    	throw new IllegalStateException (" The given rows exceed current array length");
	    }
	    	    
		
		// initialize the new array to append to
	    double new_array_to_copy_to2d [][]=  new double [ row.length][];
	    for (int i=0; i <row.length; i++ ){
	    	new_array_to_copy_to2d[i]=manipulate.copies.copies.Copy(array_append_to[row[i]]);
	    }
		

	return new_array_to_copy_to2d;
		//end of copy method
	
		}	
	
	/**
	 * 
	 * @param array 1d Array to select to
	 * @param row  rows to select
	 * @return the 1d sub-array
	 */
	public static double [] RowSelect( double array [], int row[]) {

		// sensible checks
		if ( (array==null || array.length==0 )   ){
			
			throw new NullObjectException (" The Array to select from is null or empty or the selection ");
		}

	    if ( row.length >array.length){
	    	throw new IllegalStateException (" The given rows exceed current array length");
	    }

		// initialize the new array to append to
	    double new_array_to_copy_to1d []=  new double [ row.length];
	    for (int i=0; i <row.length; i++ ){
	    	new_array_to_copy_to1d[i]=array[row[i]] + 0.0;
	    }

	    return new_array_to_copy_to1d;
		//end of copy method
		}		
	
	/**
	 * @param array_append_to 2d Array to select to
	 * @param start  start location of the row-range
	 * @param end  end location of the row-range
	 * @param threads number of threads to use
	 * @return the 2d sub-array
	 */
	
	public static double [][] RowSelect2d( double array_append_to [][], int start, int end,  int threads) {
		
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0 )   ){
			
			throw new NullObjectException (" The Array to select from is null or empty or the selection ");
		}
		

	    if ( end-start >array_append_to.length){
	    	throw new IllegalStateException (" The given rows (end-start) exceed current array length");
	    }
	    
	    if ( end<start ){
	    	throw new LessThanMinimum (end,start+1);
	    }
	    	    
	    //threads' checks
	    
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>(end-start)) {
			threads=(end-start);
		}	
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  new double [ end-start][];
		
		int length_of_each_threaded_pass = (end-start)/threads;
		int points=0;
		int points2=start;
		
		// the threads of operations
		
		int locations[][] = new int[threads][4];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points2;
			locations[n][1]=points2 + length_of_each_threaded_pass;
			locations[n][2]=points;
			locations[n][3]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
			points2+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points2;
		locations[threads-1][1]= end;
		locations[threads-1][2]=points;
		locations[threads-1][3]= (end-start);		
		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new RowSelectRunnable(new_array_to_copy_to2d  , array_append_to, locations[n][0], locations[n][1], locations[n][2], locations[n][3]));
			thread_array[n].start();
		}
		
		//start the threads
		
		for (int n=0; n <threads; n++ ){
			try {
				thread_array[n].join();
			} catch (InterruptedException e) {
			   System.out.println(e.getMessage());
			}
			
		}
		locations=null;
	return new_array_to_copy_to2d;
		//end of copy method
	
		}	
	
	/**
	 * @param array_append_to 2d Array to select to
	 * @param start  start location of the row-range
	 * @param end  end location of the row-range
	 * @param threads number of threads to use
	 * @return the 2d sub-array
	 */
	
	public static double [][] RowSelect2d( double array_append_to [][], int start, int end) {
		
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0 )   ){
			
			throw new NullObjectException (" The Array to select from is null or empty or the selection ");
		}
		

	    if ( end-start >array_append_to.length){
	    	throw new IllegalStateException (" The given rows (end-start) exceed current array length");
	    }
	    
	    if ( end<start ){
	    	throw new LessThanMinimum (end,start+1);
	    }
	    	    
	    
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  new double [ end-start][];
		
		for (int i=0,n=start; n < end; i++,n++){
			new_array_to_copy_to2d[i]=manipulate.copies.copies.Copy(array_append_to[n]);
			
		}

	return new_array_to_copy_to2d;
		//end of copy method
	
		}	
	
	
	
	
	
}
