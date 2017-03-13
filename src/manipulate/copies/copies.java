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
package manipulate.copies;
import java.util.ArrayList;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make hard copies with arrays and arraylists

 */
public class copies {
	
	/**
	 * 
	 * @param array_to_copy_from Array to copy
	 * @param threads number of threads to use
	 * @return the copied array
	 */
	public static double [] Copy( final double array_to_copy_from [], int threads) {
		
		// sensible checks
		if (array_to_copy_from==null || array_to_copy_from.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_to_copy_from.length) {
			threads=array_to_copy_from.length;
		}	
		
		// initialize the new arry to copy to
		double new_array_to_copy_to []= new double [array_to_copy_from.length];
		
		int length_of_each_threaded_pass = array_to_copy_from.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array_to_copy_from.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new CopiesrRunnable(array_to_copy_from, new_array_to_copy_to, locations[n][0], locations[n][1]));
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
	return new_array_to_copy_to;
		//end of copy method
	}
	

	/**
	 * 
	 * @param array2d_to_copy_from 2d Array to copy
	 * @param threads number of threads to use
	 * @return the copied 2d array
	 */
	public static double [][] Copy( final double array2d_to_copy_from [][], int threads) {
		
		// sensible checks
		if (array2d_to_copy_from==null || array2d_to_copy_from.length==0){
			
			throw new NullObjectException (" The 2D Array to copy from is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array2d_to_copy_from.length) {
			threads=array2d_to_copy_from.length;
		}	
		
		// initialize the new arry to copy to

		double new_array2d_to_copy_to [][]= new double [array2d_to_copy_from.length][];
		for (int i=0 ; i <array2d_to_copy_from.length; i++ ){
			new_array2d_to_copy_to[i]= new double [array2d_to_copy_from[i].length];
		}
		
		int length_of_each_threaded_pass = array2d_to_copy_from.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array2d_to_copy_from.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new CopiesrRunnable(array2d_to_copy_from, new_array2d_to_copy_to, locations[n][0], locations[n][1]));
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
		locations=null;
	return new_array2d_to_copy_to;
		//end of multiply this method
	}
	
	
	/**
	 * 
	 * @param array_to_copy_from Array to copy
	 * @return the copied array (default number of threads is 1)
	 */
	public static double [] Copy( final double array_to_copy_from []) {
		
		// sensible checks
		if (array_to_copy_from==null || array_to_copy_from.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		int threads=1;

		
		if (threads>array_to_copy_from.length) {
			threads=array_to_copy_from.length;
		}	
		
		// initialize the new arry to copy to
		double new_array_to_copy_to []= new double [array_to_copy_from.length];
		
		
		for (int i=0; i <new_array_to_copy_to.length; i++ ) {
			new_array_to_copy_to[i]=array_to_copy_from[i];
				
	
		}
	
	return new_array_to_copy_to;
		//end of copy method
	}
	

	/**
	 * 
	 * @param array2d_to_copy_from 2d Array to copy
	 * @return the copied 2d array (default number of threads is 1)
	 */
	public static double [][] Copy( final double array2d_to_copy_from [][]) {
		
		// sensible checks
		if (array2d_to_copy_from==null || array2d_to_copy_from.length==0){
			
			throw new NullObjectException (" The 2D Array to copy from is null or empty ");
		}

		
		// initialize the new arry to copy to
		double new_array2d_to_copy_to [][]= new double [array2d_to_copy_from.length][];
		for (int i=0 ; i <array2d_to_copy_from.length; i++ ){
			new_array2d_to_copy_to[i]= new double [array2d_to_copy_from[i].length];
		}
		
		for (int i=0; i <new_array2d_to_copy_to.length; i++ ) {
			for (int j=0; j <new_array2d_to_copy_to[i].length; j++ ) {
				new_array2d_to_copy_to[i][j]=array2d_to_copy_from[i][j];
				
			}
		}
	return new_array2d_to_copy_to;
		//end of multiply this method
	}
		

	/**
	 * 
	 * @param arraylist2d_to_copy_from 2d Arraylist to copy
	 * @param threads number of threads to use
	 * @return the copied arraylist
	 */
	public static ArrayList<ArrayList<Double>>  Copy( final ArrayList<ArrayList<Double>> arraylist2d_to_copy_from ) {
		
		// sensible checks
		if (arraylist2d_to_copy_from==null || arraylist2d_to_copy_from.size()==0){
			
			throw new NullObjectException (" Array to copy from is null or empty ");
		}

		// initialize the new arry to copy to
		ArrayList<ArrayList<Double>> arraylist2d_to_copy_to= new ArrayList<ArrayList<Double>>(arraylist2d_to_copy_from.size());
		for (int n=0; n <arraylist2d_to_copy_from.size(); n++ ){
			ArrayList<Double> temp= new ArrayList<Double>(arraylist2d_to_copy_from.get(n).size());
			for (int j=0; j <arraylist2d_to_copy_from.get(n).size(); j++ ){
				temp.add(arraylist2d_to_copy_from.get(n).get(j) + 0.0);
			}
			arraylist2d_to_copy_to.add(temp);
			
		}
	
	return arraylist2d_to_copy_to;
		//end of multiply this method
	}
	
	
	
	
}
