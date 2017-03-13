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

package manipulate.scalars;


import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to implemante basic oeperations of arrays and arraylists with scalars:
 *<ol>
 *<li> Additions</li>
 *<li> Subtractions</li>
 *<li> Multiplications</li>
 *<li> Divisions</li>
 * </ol>
 */
public class scalars {
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the multiplication
	 * @param threads number of threads to use
	 * 
	 */
	public static void MultiplyScalarthis( double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of multiply this method
	}
	

	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the multiplication
	 * 
	 */
	public static void MultiplyScalarthis(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			array[n]*=value;
		}
	
	
		//end of multiply this method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the multiplication
	 * @param threads number of threads to use
	 * @return a new double array multiplied with the scalar
	 * 
	 */
	public static double[] MultiplyScalar(double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of multiply  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the multiplication
	 * @return The copied array multiplied by scalar
	 * 
	 */
	public static double[] MultiplyScalar(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
         
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]*=value;
		}
	
		return coppied_array;
		//end of multiply  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the multiplication
	 * @param threads number of threads to use
	 * 
	 */
	public static void MultiplyScalarthis( double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of multiply this method
	}
	

	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the multiplication
	 * 
	 */
	public static void MultiplyScalarthis(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]*=value;
			}
		
		}
	
	
		//end of multiply this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the multiplication
	 * @param threads number of threads to use
	 * @return a new double 2D array multiplied with the scalar
	 * 
	 */
	public static double[][] MultiplyScalar(double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [][] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of multiply  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the multiplication
	 * @return The copied array multiplied by scalar
	 * 
	 */
	public static double[][] MultiplyScalar(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to multiply with scalar is null or empty ");
		}
         
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]*=value;
			}
		}
	
		return coppied_array;
		//end of multiply  method
	}	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the addition
	 * @param threads number of threads to use
	 * 
	 */
	public static void AddScalarthis( double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Add this method
	}
	

	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the addition
	 * 
	 */
	public static void AddScalarthis(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			array[n]+=value;
		}
	
	
		//end of Add this method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the addition
	 * @param threads number of threads to use
	 * @return a new double array added with the scalar
	 * 
	 */
	public static double[] AddScalar(double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Add  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the addition
	 * @return The copied array added by scalar
	 * 
	 */
	public static double[] AddScalar(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
         
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]+=value;
		}
	
		return coppied_array;
		//end of Add  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the addition
	 * @param threads number of threads to use
	 * 
	 */
	public static void AddScalarthis( double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Add this method
	}
	

	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the addition
	 * 
	 */
	public static void AddScalarthis(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]+=value;
			}
		
		}
	
	
		//end of Add this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the addition
	 * @param threads number of threads to use
	 * @return a new double 2D array added with the scalar
	 * 
	 */
	public static double[][] AddScalar(double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [][] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Add  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the addition
	 * @return The copied array added by scalar
	 * 
	 */
	public static double[][] AddScalar(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Add with scalar is null or empty ");
		}
         
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]+=value;
			}
		}
	
		return coppied_array;
		//end of Add  method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the subtraction
	 * @param threads number of threads to use
	 * 
	 */
	public static void SubtractScalarthis( double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Subtract this method
	}
	

	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the subtraction
	 * 
	 */
	public static void SubtractScalarthis(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			array[n]-=value;
		}
	
	
		//end of Subtract this method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the subtraction
	 * @param threads number of threads to use
	 * @return a new double array subtracted with the scalar
	 * 
	 */
	public static double[] SubtractScalar(double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Subtract  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the subtraction
	 * @return The copied array subtracted by scalar
	 * 
	 */
	public static double[] SubtractScalar(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
         
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]-=value;
		}
	
		return coppied_array;
		//end of Subtract  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the subtraction
	 * @param threads number of threads to use
	 * 
	 */
	public static void SubtractScalarthis( double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Subtract this method
	}
	

	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the subtraction
	 * 
	 */
	public static void SubtractScalarthis(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]-=value;
			}
		
		}
	
	
		//end of Subtract this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the subtraction
	 * @param threads number of threads to use
	 * @return a new double 2D array subtracted with the scalar
	 * 
	 */
	public static double[][] SubtractScalar(double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [][] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Subtract  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the subtraction
	 * @return The copied array subtracted by scalar
	 * 
	 */
	public static double[][] SubtractScalar(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to Subtract with scalar is null or empty ");
		}
         
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]-=value;
			}
		}
	
		return coppied_array;
		//end of Subtract  method
	}	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the division
	 * @param threads number of threads to use
	 * 
	 */
	public static void DivideScalarthis( double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of divide this method
	}
	

	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the division
	 * 
	 */
	public static void DivideScalarthis(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			array[n]/=value;
		}
	
	
		//end of divide this method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the division
	 * @param threads number of threads to use
	 * @return a new double array divided with the scalar
	 * 
	 */
	public static double[] DivideScalar(double array [], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of divide  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param value Value to make the division
	 * @return The copied array divided by scalar
	 * 
	 */
	public static double[] DivideScalar(double array [], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
         
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]/=value;
		}
	
		return coppied_array;
		//end of divide  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the division
	 * @param threads number of threads to use
	 * 
	 */
	public static void DivideScalarthis( double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		
		
		
		int length_of_each_threaded_pass = array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(array, locations[n][0],locations[n][1], value, "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of divide this method
	}
	

	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the division
	 * 
	 */
	public static void DivideScalarthis(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
		
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]/=value;
			}
		
		}
	
	
		//end of divide this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param value Value to make the division
	 * @param threads number of threads to use
	 * @return a new double 2D array divided with the scalar
	 * 
	 */
	public static double[][] DivideScalar(double array [][], double value, int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	
		double [][] coppied_array=manipulate.copies.copies.Copy(array,threads);
		
		int length_of_each_threaded_pass = coppied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=coppied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new ScalarRunnable(coppied_array, locations[n][0],locations[n][1], value, "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of divide  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param value Value to make the division
	 * @return The copied array divided by scalar
	 * 
	 */
	public static double[][] DivideScalar(double array [][], double value) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to divide with scalar is null or empty ");
		}
         
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]/=value;
			}
		}
	
		return coppied_array;
		//end of divide  method
	}	
	
	
	
}
