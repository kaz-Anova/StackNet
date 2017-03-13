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

package manipulate.operate;

import exceptions.DimensionMismatchException;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to implemante basic oeperations of arrays and arraylists with Operates:
 *<ol>
 *<li> Additions</li>
 *<li> Subtractions</li>
 *<li> Multiplications</li>
 *<li> Divisions</li>
 * </ol>
 */
public class operations {
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 Array (this) to multiply with
	 * @param threads number of threads to use
	 * 
	 */
	public static void MultiplyOperatethis( double array [],double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0  ){
			
			throw new NullObjectException (" Arrays to multiply are null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array, array2, locations[n][0],locations[n][1], "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of multiply this method
	}
	

	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 Array (this) to multiply with 
	 * 
	 */
	public static void MultiplyOperatethis(double array [], double array2 []) {
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0 ){
			
			throw new NullObjectException (" Arrays to multiply are null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}	
		
		for (int n=0; n <array.length; n++ ){
			array[n]*=array2[n];
		}
		//end of multiply this method
		
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 Array (this) to multiply with 
	 * @param threads number of threads to use
	 * @return a new double array multiplied
	 * 
	 */
	public static double[] MultiplyOperate(double array [], double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0||array2==null || array2.length==0){
			
			throw new NullObjectException (" Arrays to multiply are null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array, array2, locations[n][0],locations[n][1], "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of multiply  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 Array (this) to multiply with 
	 * @return The copied array multiplied
	 * 
	 */
	public static double[] MultiplyOperate(double array [], double array2[]) {
		
		// sensible checks
		if (array==null || array.length==0||array2==null || array2.length==0){
			
			throw new NullObjectException (" Arrays to multiply are null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}			
         
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]*=array2[n];
		}
	
		return coppied_array;
		//end of multiply  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 Array (this) to multiply with 
	 * @param threads number of threads to use
	 * 
	 */
	public static void MultiplyOperatethis( double array [][], double array2 [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array, array2,locations[n][0],locations[n][1], "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of multiply this method
	}
	

	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 Array to multiply with
	 * 
	 */
	public static void MultiplyOperatethis(double array [][], double array2[][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}	
		
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]*=array2[n][J];
			}
		
		}
	
	
		//end of multiply this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 Array to multiply with
	 * @param threads number of threads to use
	 * @return a new double 2D array multiplied 
	 * 
	 */
	public static double[][] MultiplyOperate(double array [][], double array2[][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array, array2,locations[n][0],locations[n][1], "mul"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of multiply  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 Array to Multiply with
	 * @return The copied array multiplied
	 * 
	 */
	public static double[][] MultiplyOperate(double array [][], double array2[][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}	
         
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]*=array2[n][J];
			}
		}
	
		return coppied_array;
		//end of multiply  method
	}	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for addition
	 * @param threads number of threads to use
	 * 
	 */
	public static void AddOperatethis( double array [], double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array,array2, locations[n][0],locations[n][1], "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Add this method
	}
	

	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for addition
	 * 
	 */
	public static void AddOperatethis(double array [], double array2 []) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}		
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
		
		
		for (int n=0; n <array.length; n++ ){
			array[n]+=array2[n];
		}
	
	
		//end of Add this method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for addition
	 * @param threads number of threads to use
	 * @return a new double array added 
	 * 
	 */
	public static double[] AddOperate(double array [], double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array,array2, locations[n][0],locations[n][1], "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Add  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for addition
	 * @return The copied array added 
	 * 
	 */
	public static double[] AddOperate(double array [], double array2 []) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
         
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]+=array2[n];
		}
	
		return coppied_array;
		//end of Add  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 2nd Array for addition
	 * @param threads number of threads to use
	 * 
	 */
	public static void AddOperatethis( double array [][], double array2 [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array, array2,locations[n][0],locations[n][1], "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Add this method
	}
	

	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 2nd Array for addition
	 * 
	 */
	public static void AddOperatethis(double array [][], double array2 [][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]+=array2[n][J];
			}
		
		}
	
	
		//end of Add this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 2nd Array for addition
	 * @param threads number of threads to use
	 * @return a new double 2D array added
	 * 
	 */
	public static double[][] AddOperate(double array [][], double array2 [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array,array2, locations[n][0],locations[n][1], "add"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Add  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 2nd Array for addition
	 * @return The copied array added 
	 * 
	 */
	public static double[][] AddOperate(double array [][], double array2 [][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
         
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]+=array2[n][J];
			}
		}
	
		return coppied_array;
		//end of Add  method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * @param threads number of threads to use
	 * 
	 */
	public static void SubtractOperatethis( double array [], double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array,array2, locations[n][0],locations[n][1], "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Subtract this method
	}
	

	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * 
	 */
	public static void SubtractOperatethis(double array [], double array2 []) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}		
		
		for (int n=0; n <array.length; n++ ){
			array[n]-=array2[n];
		}
	
	
		//end of Subtract this method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * @param threads number of threads to use
	 * @return a new double array subtracted 
	 * 
	 */
	public static double[] SubtractOperate(double array [], double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array, array2, locations[n][0],locations[n][1], "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Subtract  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * @return The copied array subtracted
	 * 
	 */
	public static double[] SubtractOperate(double array [], double array2 []) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
         
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]-=array2[n];
		}
	
		return coppied_array;
		//end of Subtract  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * @param threads number of threads to use
	 * 
	 */
	public static void SubtractOperatethis( double array [][], double array2 [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array,array2, locations[n][0],locations[n][1], "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of Subtract this method
	}
	

	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * 
	 */
	public static void SubtractOperatethis(double array [][], double array2 [][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]-=array2[n][J];
			}
		
		}
	
	
		//end of Subtract this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * @param threads number of threads to use
	 * @return a new double 2D array subtracted
	 * 
	 */
	public static double[][] SubtractOperate(double array [][], double array2 [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array,array2, locations[n][0],locations[n][1], "sub"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of Subtract  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 2nd Array for subtraction
	 * @return The copied array subtracted
	 * 
	 */
	public static double[][] SubtractOperate(double array [][], double array2 [][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		} 
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]-=array2[n][J];
			}
		}
	
		return coppied_array;
		//end of Subtract  method
	}	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for division
	 * @param threads number of threads to use
	 * 
	 */
	public static void DivideOperatethis( double array [], double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array, array2, locations[n][0],locations[n][1], "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of divide this method
	}
	

	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for division
	 * 
	 */
	public static void DivideOperatethis(double array [], double array2 []) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
		for (int n=0; n <array.length; n++ ){
			array[n]/=array[n];
		}
	
	
		//end of divide this method
	}	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for division
	 * @param threads number of threads to use
	 * @return a new double array divided
	 * 
	 */
	
	public static double[] DivideOperate(double array [], double array2 [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array,array2, locations[n][0],locations[n][1], "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of divide  method
	}
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param array2 2nd Array for division
	 * @return The copied array divided
	 * 
	 */
	public static double[] DivideOperate(double array [], double array2 []) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		} 
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
		double [] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			coppied_array[n]/=array2[n];
		}
	
		return coppied_array;
		//end of divide  method
	}
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 2nd Array for division
	 * @param threads number of threads to use
	 * 
	 */
	public static void DivideOperatethis( double array [][], double array2 [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(array,array2, locations[n][0],locations[n][1], "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		//end of divide this method
	}
	

	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 2nd Array for division
	 * 
	 */
	public static void DivideOperatethis(double array [][], double array2 [][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
		
		for (int n=0; n <array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]/=array2[n][J];
			}
		
		}
	
	
		//end of divide this method
	}	
	
	
	/**
	 * 
	 * @param array 2d Array (this) to process
	 * @param array2 2nd Array for division
	 * @param threads number of threads to use
	 * @return a new double 2D array divided 
	 * 
	 */
	public static double[][] DivideOperate(double array [][], double array2 [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
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
			
			thread_array[n]= new Thread(new OperateRunnable(coppied_array, array2,locations[n][0],locations[n][1], "div"));
			thread_array[n].start();
			points+=length_of_each_threaded_pass;
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
		return coppied_array;
		//end of divide  method
	}
	
	
	/**
	 * 
	 * @param array 2D Array (this) to process
	 * @param array2 2nd Array for division
	 * @return The copied array divided 
	 * 
	 */
	public static double[][] DivideOperate(double array [][], double array2 [][]) {
		
		// sensible checks
		if (array==null || array.length==0 ||array2==null || array2.length==0){
			
			throw new NullObjectException (" Array to multiply with Operate is null or empty ");
		}
		if (array.length!= array2.length){
			
			throw new DimensionMismatchException (array.length, array2.length);
		}
         
		double [][] coppied_array=manipulate.copies.copies.Copy(array);
		
		for (int n=0; n <coppied_array.length; n++ ){
			for (int J=0; J <array[n].length; J++ ){
				array[n][J]/=array2[n][J];
			}
		}
	
		return coppied_array;
		//end of divide  method
	}	
	
	
	
}
