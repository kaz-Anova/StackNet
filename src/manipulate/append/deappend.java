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

package manipulate.append;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make appends on arrays

 */
public class deappend {
	

	
	//-----------------------------------------1d De-appendings on 2d--------------------------------------------------//
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to de-append to
	 * @param threads number of threads to use
	 * @return the 1d appended array
	 */
	public static double [] DeAppend( double array_append_to [][], int threads) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1){
			
			return null ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
	
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// Initialise the new array to append to
		double new_array_to_copy_to2d [][]=  manipulate.copies.copies.Copy(array_append_to,threads) ;
		double to_deppend []=new double [array_append_to.length];
		
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
			thread_array[n]= new Thread(new DeAppendRunnable(to_deppend  , new_array_to_copy_to2d, locations[n][0], locations[n][1]));
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
	return to_deppend;
		//end of copy method
	
		}
	}
	

	/**
	 * 
	 * @param array_append_to 2d Array to append to
	 * @return the 1d deppended array
	 */
	public static double [] DeAppend( double array_append_to [][]) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1){
			
			return null ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}

		

		// Initialise the new array to copy to
		double new_array_to_copy_to1d []= new double [array_append_to.length];
		
		for (int i=0; i < new_array_to_copy_to1d.length; i++) {
			
			new_array_to_copy_to1d[i]=array_append_to[i][0];
		}



	return new_array_to_copy_to1d;
		//end of copy method
	
		}
	}
	
	
	
	//-----------------------------------------2d Appendings on 2d--------------------------------------------------//
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to deppend to
	 * @param threads number of threads to use
	 * @return the 2d appended array
	 */
	public static double [][] DeAppend2d( double array_append_to [][], int threads) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1){
			
			return null ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to deppend is null or empty ");
		}
		
	
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// Initialise the new array to append to
		double new_array_to_copy_to2d [][]=  manipulate.copies.copies.Copy(array_append_to,threads) ;
		
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
			thread_array[n]= new Thread(new DeAppendRunnable( new_array_to_copy_to2d, locations[n][0], locations[n][1]));
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
	 * @param array_append_to 2d Array to deappend to
	 * @return the 2d deppended array
	 */
	public static double [][] DeAppend2d( double array_append_to [][]) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1){
			
			return null ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to deppend from is null or empty ");
		}

		// initialize the new arry to copy to
		double new_array_to_copy_to2d [][]= new double [array_append_to.length][];
		
		for (int i=0; i < new_array_to_copy_to2d.length; i++) {
			double temp []=array_append_to[i];
			double temp_minus_1[]=new double [temp.length-1];
			for (int j=0; j < temp_minus_1.length;j++) {
				temp_minus_1[j]=temp[j];
			}
			new_array_to_copy_to2d[i]=temp_minus_1;
		}



	return new_array_to_copy_to2d;
		//end of copy method
	
		}
	}	

	/**
	 * 
	 * @param array_append_to 2d Array to deappend to
	 * @param threads number of threads to use
	 */
	public static void  DeAppend2dthis( double array_append_to [][], int threads) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1){
			
			array_append_to= null ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to deppend is null or empty ");
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
			thread_array[n]= new Thread(new DeAppendRunnable( array_append_to, locations[n][0], locations[n][1]));
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
	
		}
	}
	

	/**
	 * 
	 * @param array_append_to 2d Array to deappend to

	 */
	public static void  DeAppend2dthis( double array_append_to [][]) {
		
		// return the second array if the first is null
		if (array_append_to[0].length==1){
			
			array_append_to= null ;
			
		} 
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)   ){
			
			throw new NullObjectException (" The Array to deppend from is null or empty ");
		}

	
		
		for (int i=0; i < array_append_to.length; i++) {
			double temp []=array_append_to[i];
			double temp_minus_1[]=new double [temp.length-1];
			for (int j=0; j < temp_minus_1.length;j++) {
				temp_minus_1[j]=temp[j];
			}
			array_append_to[i]=temp_minus_1;
		}


		//end of copy method
	
		}
	}		
		
}
