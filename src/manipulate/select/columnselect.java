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
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make column selections

 */
public class columnselect {
	

	
	//-----------------------------------------1d column selects on 2d--------------------------------------------------//
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param col  column to select
	 * @param threads number of threads to use
	 * @return the 1d sub-array
	 */
	public static double [] ColumnSelect( double array_append_to [][], int col,  int threads) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1 && col==0){
			
			return manipulate.conversions.dimension.Convert(array_append_to) ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to select from is null or empty ");
		}
		
	
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// initialize the new array to append to
		double new_array_to_copy_to1d []=  new double [array_append_to.length];
		
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
			thread_array[n]= new Thread(new ColumnSelectRunnable(new_array_to_copy_to1d  , array_append_to,col, locations[n][0], locations[n][1]));
			thread_array[n].start();
		}
		
		//start the threads
		
		for (int n=0; n <threads; n++ ){
			try {
				thread_array[n].join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
			   System.out.println(e.getMessage());
			}
			
		}
	return new_array_to_copy_to1d;
		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param col  column to select
	 * @return the 1d sub-array
	 */
	public static double [] ColumnSelect( double array_append_to [][], int col) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1 && col==0){
			
			return manipulate.conversions.dimension.Convert(array_append_to) ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to select from is null or empty ");
		}
		
		
		// initialize the new array to append to
		double new_array_to_copy_to1d []=  new double [array_append_to.length];
		for (int i=0; i < new_array_to_copy_to1d.length; i++){
			new_array_to_copy_to1d[i]= array_append_to[i][col];
		}
	
	return new_array_to_copy_to1d;
		//end of copy method
	
		}
	}
	
	
	
	//-----------------------------------------2d column selections on 2d array--------------------------------------------------//
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param col  columns to select
	 * @param threads number of threads to use
	 * @return the 2d sub-array
	 */
	public static double [][] ColumnSelect( double array_append_to [][], int col [],  int threads) {
		
		// return the second array if the first is null
		if (array_append_to[0].length== col.length){
			
			return array_append_to ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to select from is null or empty ");
		}
		
	
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  manipulate.copies.copies.Copy(array_append_to,threads);
		
		int length_of_each_threaded_pass = new_array_to_copy_to2d.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=new_array_to_copy_to2d.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new ColumnSelectRunnable(new_array_to_copy_to2d  ,col, locations[n][0], locations[n][1]));
			thread_array[n].start();
		}
		
		//start the threads
		
		for (int n=0; n <threads; n++ ){
			try {
				thread_array[n].join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
			   System.out.println(e.getMessage());
			}
			
		}
	return new_array_to_copy_to2d;
		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param col  columns to select
	 * @return the 2d sub-array
	 */
	public static double [][] ColumnSelect( double array_append_to [][], int col[]) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==col.length){
			
			return array_append_to ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to select from is null or empty ");
		}
		
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  new double [array_append_to.length][];
		for (int i=0; i < new_array_to_copy_to2d.length; i++){
			double[] temp= new double [col.length];
			
			for (int j=0; j < col.length; j++) {
				temp [j]=array_append_to[i][col[j]];
			}

			new_array_to_copy_to2d[i]=temp;


		}
	
	return new_array_to_copy_to2d;
		//end of copy method
	
		}
	}	
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param col  columns to select
	 * @param threads number of threads to use
	 */
	public static void ColumnSelectthis( double array_append_to [][], int col [],  int threads) {
		
		// return the second array if the first is null
		if (array_append_to[0].length== col.length){
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to select from is null or empty ");
		}
		
	
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	

		
		int length_of_each_threaded_pass = array_append_to.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array_append_to.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new ColumnSelectRunnable(array_append_to  ,col, locations[n][0], locations[n][1]));
			thread_array[n].start();
		}
		
		//start the threads
		
		for (int n=0; n <threads; n++ ){
			try {
				thread_array[n].join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
			   System.out.println(e.getMessage());
			}
			
		}
		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to 2d Array to select to
	 * @param col  columns to select
	 * @return the 2d sub-array
	 */
	public static void ColumnSelectthis( double array_append_to [][], int col[]) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==col.length){
			
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to select from is null or empty ");
		}
		
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  new double [array_append_to.length][];
		for (int i=0; i < new_array_to_copy_to2d.length; i++){
			double[] temp= new double [col.length];
			
			for (int j=0; j < col.length; j++) {
				temp [j]=array_append_to[i][col[j]];
			}

			array_append_to[i]=temp;


		}
	

	
		}
	}			
}
