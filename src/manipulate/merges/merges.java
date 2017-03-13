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

package manipulate.merges;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make appends on arrays

 */
public class merges {
	
	//--------------------------------------1d merges--------------------------------//
	/**
	 * 
	 * @param array_append_to Array to merge as a base
	 * @param array_append_from Array to merge to
	 * @param threads number of threads to use
	 * @return the merged  array
	 */
	public static double [] Merge( double array_append_to [],  double array_append_from [], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;
			
		} else if (array_append_from!=null && array_append_to==null) {
			return array_append_from;
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Arrays to merge from are null or empty ");
		}
		

		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_from.length) {
			threads=array_append_from.length;
		}	
		
		// initialize the new arry to copy to
		double new_array_to_copy_to1d []= new double [array_append_to.length + array_append_from.length];
		
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
			thread_array[n]= new Thread(new MergeRunnable(new_array_to_copy_to1d,array_append_to , array_append_from, locations[n][0], locations[n][1]));
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
	 * @param array_append_to Array to merge as a base
	 * @param array_append_from Array to merge to
	 * @param threads number of threads to use
	 * @return the merged  array
	 */
	public static double [] Merge( double array_append_to [],  double array_append_from []) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;
			
		} else if (array_append_from!=null && array_append_to==null) {
			return array_append_from;
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Arrays to merge from are null or empty ");
		}
		
		// initialize the new arry to copy to
		double new_array_to_copy_to1d []= new double [array_append_to.length + array_append_from.length];
		for (int i=0; i < array_append_to.length; i++){
			new_array_to_copy_to1d[i]=array_append_to[i];
		}
		for (int i=0; i < array_append_from.length; i++){
			new_array_to_copy_to1d[i+array_append_to.length]=array_append_from[i];
		}
				

	return new_array_to_copy_to1d;
		//end of copy method
	
		}
	}

	


	//--------------------------------------2d merges--------------------------------//
	/**
	 * 
	 * @param array_append_to Array to merge as a base
	 * @param array_append_from Array to merge to
	 * @param threads number of threads to use
	 * @return the 2d merged array
	 */
	public static double [][] Merge( double array_append_to [][],  double array_append_from [][], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;
			
		} else if (array_append_from!=null && array_append_to==null) {
			return array_append_from;
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Arrays to merge from are null or empty ");
		}
		

		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_from.length) {
			threads=array_append_from.length;
		}	
		
		// initialize the new arry to copy to
		double new_array_to_copy_to1d [][]= new double [array_append_to.length + array_append_from.length][];
		
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
			thread_array[n]= new Thread(new MergeRunnable(new_array_to_copy_to1d,array_append_to , array_append_from, locations[n][0], locations[n][1]));
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
	 * @param array_append_to Array to merge as a base
	 * @param array_append_from Array to merge to
	 * @param threads number of threads to use
	 * @return the 2d merged  array
	 */
	public static double [][] Merge( double array_append_to [][],  double array_append_from [][]) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;
			
		} else if (array_append_from!=null && array_append_to==null) {
			return array_append_from;
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Arrays to merge from are null or empty ");
		}
		
		// initialize the new arry to copy to
		double new_array_to_copy_to1d [][]= new double [array_append_to.length + array_append_from.length][];
		for (int i=0; i < array_append_to.length; i++){
			new_array_to_copy_to1d[i]= manipulate.copies.copies.Copy(array_append_to[i]);
		}
		for (int i=0; i < array_append_from.length; i++){
			new_array_to_copy_to1d[i+array_append_to.length]=manipulate.copies.copies.Copy(array_append_from[i]);
		}
				

	return new_array_to_copy_to1d;
		//end of copy method
	
		}
	}

	

	
}
