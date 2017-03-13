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

package manipulate.transforms;


import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to implement basic transform operations : </p>
 */
public class transforms {
	
	//---------------------------------log transforms----------------------------------//
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on log</p>
	 * 
	 */
	public static void Transformlogthis( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "log"));
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
	 * @param threads number of threads to use
	 * <p> transform based on log</p>
	 * 
	 */
	public static void Transformlogthis( double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
			if (array[i]>0){
				array[i]=Math.log(array[i]);
			}
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on log</p>
	 * 
	 */
	public static double [] Transformlog( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "log"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on log</p>
	 * 
	 */
	public static double [] Transformlog (double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[]= new double [array.length];
		
		for (int i=0; i < array.length; i++) {
			if (array[i]>0){
				copied_array[i]=Math.log(array[i]);
			} else {
				copied_array[i]=array[i];
			}
		
		}
		return copied_array;
		//end of log this method
	}	
	

	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on log</p>
	 * 
	 */
	public static void Transformlogthis( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "log"));
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
	 * @param threads number of threads to use
	 * <p> transform based on log</p>
	 * 
	 */
	public static void Transformlogthis( double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
			if (array[i][j]>0){
				array[i][j]=Math.log(array[i][j]);
			}
			}
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on log</p>
	 * 
	 */
	public static double [][] Transformlog( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "log"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on log</p>
	 * 
	 */
	public static double [][] Transformlog (double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[][]= new double [array.length][array[0].length];
		
		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
			if (array[i][j]>0){
				copied_array[i][j]=Math.log(array[i][j]);
			} else {
				copied_array[i][j]=array[i][j];
			}
			}
		}
		return copied_array;
		//end of log this method
	}
	
	
	

	
	
	//---------------------------------logplusone transforms----------------------------------//
	
	
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static void Transformlogplusonethis( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "logplusone"));
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
	 * @param threads number of threads to use
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static void Transformlogplusonethis( double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
			if (array[i]>0-1){
				array[i]=Math.log(array[i] + 1);
			}
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static double [] Transformlogplusone( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "logplusone"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static double [] Transformlogplusone (double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[]= new double [array.length];
		
		for (int i=0; i < array.length; i++) {
			if (array[i]>0-1){
				copied_array[i]=Math.log(array[i]+1);
			} else {
				copied_array[i]=array[i];
			}
		
		}
		return copied_array;
		//end of logplusone this method
	}	
	

	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static void Transformlogplusonethis( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "logplusone"));
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
	 * @param threads number of threads to use
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static void Transformlogplusonethis( double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
			if (array[i][j]>0-1){
				array[i][j]=Math.log(array[i][j]+1);
			}
			}
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static double [][] Transformlogplusone( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "logplusone"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on logplusone</p>
	 * 
	 */
	public static double [][] Transformlogplusone (double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[][]= new double [array.length][array[0].length];
		
		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
			if (array[i][j]>0-1){
				copied_array[i][j]=Math.log(array[i][j] +1);
			} else {
				copied_array[i][j]=array[i][j];
			}
			}
		}
		return copied_array;
		//end of logplusone this method
	}
	
	
	
	
	
	
	
	
	
	
	//---------------------------------sqrt transforms----------------------------------//
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static void Transformsqrtthis( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "sqrt"));
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
	 * @param threads number of threads to use
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static void Transformsqrtthis( double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
			if (array[i]>0){
				array[i]=Math.sqrt(array[i]);
			}
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static double [] Transformsqrt( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "sqrt"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static double [] Transformsqrt (double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[]= new double [array.length];
		
		for (int i=0; i < array.length; i++) {
			if (array[i]>0){
				copied_array[i]=Math.sqrt(array[i]);
			} else {
				copied_array[i]=array[i];
			}
		
		}
		return copied_array;
		//end of sqrt this method
	}	
	

	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static void Transformsqrtthis( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "sqrt"));
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
	 * @param threads number of threads to use
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static void Transformsqrtthis( double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
			if (array[i][j]>0){
				array[i][j]=Math.sqrt(array[i][j]);
			}
			}
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static double [][] Transformsqrt( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "sqrt"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on sqrt</p>
	 * 
	 */
	public static double [][] Transformsqrt (double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[][]= new double [array.length][array[0].length];
		
		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
			if (array[i][j]>0){
				copied_array[i][j]=Math.sqrt(array[i][j]);
			} else {
				copied_array[i][j]=array[i][j];
			}
			}
		}
		return copied_array;
		//end of sqrt this method
	}
	
	
	
	
	
	
	
	//---------------------------------exp transforms----------------------------------//
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on exp</p>
	 * 
	 */
	public static void Transformexpthis( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "exp"));
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
	 * @param threads number of threads to use
	 * <p> transform based on exp</p>
	 * 
	 */
	public static void Transformexpthis( double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
				array[i]=Math.exp(array[i]);
			
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on exp</p>
	 * 
	 */
	public static double [] Transformexp( double array [], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "exp"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on exp</p>
	 * 
	 */
	public static double [] Transformexp (double array []) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[]= new double [array.length];
		
		for (int i=0; i < array.length; i++) {
				copied_array[i]=Math.exp(array[i]);
		}
		return copied_array;
		//end of exp this method
	}	
	

	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * <p> transform based on exp</p>
	 * 
	 */
	public static void Transformexpthis( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
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
			
			thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "exp"));
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
	 * @param threads number of threads to use
	 * <p> transform based on exp</p>
	 * 
	 */
	public static void Transformexpthis( double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}

		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
				array[i][j]=Math.exp(array[i][j]);
			
			}
		
		}
		//end of multiply this method
	}
		
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on exp</p>
	 * 
	 */
	public static double [][] Transformexp( double array [][], int threads) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array.length) {
			threads=array.length;
		}	

        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
		int length_of_each_threaded_pass = copied_array.length/threads;
		int points=0;
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=copied_array.length;
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			
			thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "exp"));
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
		return copied_array;
		//end of multiply this method
	}
	
	
	
	/**
	 * 
	 * @param array Array (this) to process
	 * @param threads number of threads to use
	 * @return transformed 1d array
	 * <p> transform based on exp</p>
	 * 
	 */
	public static double [][] Transformexp (double array [][]) {
		
		// sensible checks
		if (array==null || array.length==0){
			
			throw new NullObjectException (" Array to transform is null or empty ");
		}
        
		double  copied_array[][]= new double [array.length][array[0].length];
		
		for (int i=0; i < array.length; i++) {
			for (int j=0; j <array[i].length;j++ ){
				copied_array[i][j]=Math.exp(array[i][j]);
			}
		}
		return copied_array;
		//end of exp this method
	}
	
	
	
	
	
	//---------------------------------abs transforms----------------------------------//
	
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on abs</p>
		 * 
		 */
		public static void Transformabsthis( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "abs"));
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
		 * <p> transform based on abs</p>
		 * 
		 */
		public static void Transformabsthis( double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
					array[i]=Math.abs(array[i]);
				
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on abs</p>
		 * 
		 */
		public static double [] Transformabs( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "abs"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on abs</p>
		 * 
		 */
		public static double [] Transformabs (double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
					copied_array[i]=Math.abs(array[i]);
			}
			return copied_array;
			//end of abs this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on abs</p>
		 * 
		 */
		public static void Transformabsthis( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "abs"));
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

		 * <p> transform based on abs</p>
		 * 
		 */
		public static void Transformabsthis( double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					array[i][j]=Math.abs(array[i][j]);
				
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on abs</p>
		 * 
		 */
		public static double [][] Transformabs( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "abs"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on abs</p>
		 * 
		 */
		public static double [][] Transformabs (double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					copied_array[i][j]=Math.abs(array[i][j]);
				}
			}
			return copied_array;
			//end of abs this method
		}
			

		
		
		
		
		//---------------------------------Min transforms----------------------------------//		
		
		

		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @param min min value
		 * <p> transform based on min</p>
		 * 
		 */
		public static void Transformminthis( double array [], double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "min"));
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
		 * @param threads number of threads to use
         * @param min min value to use
		 * <p> transform based on min</p>
		 * 
		 */
		public static void Transformminthis( double array [], double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				if (array[i]<min){
					array[i]=min;
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
         * @param min min value to use
		 * @return transformed 1d array
		 * <p> transform based on min</p>
		 * 
		 */
		public static double [] Transformmin( double array [],double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "min"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param min min value to use
		 * @return transformed 1d array
		 * <p> transform based on min</p>
		 * 
		 */
		public static double [] Transformmin (double array [], double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
				if (array[i]<min){
					copied_array[i]=min;
				} else {
					copied_array[i]=array[i];
				}
			
			}
			return copied_array;
			//end of min this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param min min value to use
		 * <p> transform based on min</p>
		 * 
		 */
		public static void Transformminthis( double array [][], double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "min"));
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
         * @param min min value to use
		 * <p> transform based on min</p>
		 * 
		 */
		public static void Transformminthis( double array [][], double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
				if (array[i][j]<min){
					array[i][j]=min;
				}
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
         * @param min min value to use
		 * @return transformed 1d array
		 * <p> transform based on min</p>
		 * 
		 */
		public static double [][] Transformmin( double array [][], double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "min"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
         * @param min min value to use
		 * @return transformed 1d array
		 * <p> transform based on min</p>
		 * 
		 */
		public static double [][] Transformmin (double array [][], double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
				if (array[i][j]<min){
					copied_array[i][j]=min;
				} else {
					copied_array[i][j]=array[i][j];
				}
				}
			}
			return copied_array;
			//end of min this method
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		//---------------------------------max transforms----------------------------------//		
		
		

		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @param max max value
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformmaxthis( double array [], double max, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.max=max;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "max"));
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
		 * @param threads number of threads to use
         * @param max max value to use
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformmaxthis( double array [], double max) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				if (array[i]>max){
					array[i]=max;
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
         * @param max max value to use
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [] Transformmax( double array [],double max, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.max=max;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "max"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param max max value to use
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [] Transformmax (double array [], double max) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
				if (array[i]>max){
					copied_array[i]=max;
				} else {
					copied_array[i]=array[i];
				}
			
			}
			return copied_array;
			//end of max this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param max max value to use
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformmaxthis( double array [][], double max, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.max=max;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "max"));
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
         * @param max max value to use
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformmaxthis( double array [][], double max) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
				if (array[i][j]>max){
					array[i][j]=max;
				}
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
         * @param max max value to use
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [][] Transformmax( double array [][], double max, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.max=max;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "max"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
         * @param max max value to use
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [][] Transformmax (double array [][], double max) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
				if (array[i][j]>max){
					copied_array[i][j]=max;
				} else {
					copied_array[i][j]=array[i][j];
				}
				}
			}
			return copied_array;
			//end of max this method
		}
				
		
		
		
		
		
		
		
		
		
		
		//---------------------------------minmax transforms----------------------------------//		
		
		

		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @param max max value
		 * @param min min value
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformminmaxthis( double array [], double max, double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.max=max;
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "minmax"));
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
		 * @param max max value
		 * @param min min value
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformminmaxthis( double array [], double max, double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				if (array[i]>max){
					array[i]=max;
				} else if (array[i]<min){
					array[i]=min;
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @param max max value
		 * @param min min value
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [] Transformminmax( double array [],double max, double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.max=max;
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "minmax"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param max max value
		 * @param min min value
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [] Transformminmax (double array [], double max, double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
				if (array[i]<min){
					copied_array[i]=min;
				} else if (array[i]>max){
					copied_array[i]=max;
				}
				else {
					copied_array[i]=array[i];
				}
			
			}
			return copied_array;
			//end of max this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param max max value
		 * @param min min value
		 * @param threads Number of threads to use
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformminmaxthis( double array [][], double max, double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.max=max;
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "minmax"));
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
		 * @param max max value
		 * @param min min value
		 * <p> transform based on max</p>
		 * 
		 */
		public static void Transformminmaxthis( double array [][], double max, double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
				if (array[i][j]>max){
					array[i][j]=max;
				}else if (array[i][j]<min){
					array[i][j]=min;
				}
				
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param max max value
		 * @param min min value
		 * @param threads Number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [][] Transformminmax( double array [][], double max, double min, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.max=max;
				TransformRunnable.min=min;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "minmax"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param max max value
		 * @param min min value
		 * @return transformed 1d array
		 * <p> transform based on max</p>
		 * 
		 */
		public static double [][] Transforminmax (double array [][], double max, double min) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
				if (array[i][j]>max){
					copied_array[i][j]=max;
				} else if (array[i][j]<min){
					copied_array[i][j]=min;
				}else {
					copied_array[i][j]=array[i][j];
				}
				}
			}
			return copied_array;
			//end of max this method
		}
				

		
		
		
		
		
		//---------------------------------sin transforms----------------------------------//
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on sin</p>
		 * 
		 */
		public static void Transformsinthis( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "sin"));
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
		 * <p> transform based on sin</p>
		 * 
		 */
		public static void Transformsinthis( double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
					array[i]=Math.sin(array[i]);
				
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on sin</p>
		 * 
		 */
		public static double [] Transformsin( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "sin"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on sin</p>
		 * 
		 */
		public static double [] Transformsin (double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
					copied_array[i]=Math.sin(array[i]);
			}
			return copied_array;
			//end of sin this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on sin</p>
		 * 
		 */
		public static void Transformsinthis( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "sin"));
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

		 * <p> transform based on sin</p>
		 * 
		 */
		public static void Transformsinthis( double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					array[i][j]=Math.sin(array[i][j]);
				
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on sin</p>
		 * 
		 */
		public static double [][] Transformsin( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "sin"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on sin</p>
		 * 
		 */
		public static double [][] Transformsin (double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					copied_array[i][j]=Math.sin(array[i][j]);
				}
			}
			return copied_array;
			//end of sin this method
		}
			

		
		
		//---------------------------------cos transforms----------------------------------//
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on cos</p>
		 * 
		 */
		public static void Transformcosthis( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "cos"));
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
		 * <p> transform based on cos</p>
		 * 
		 */
		public static void Transformcosthis( double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
					array[i]=Math.cos(array[i]);
				
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on cos</p>
		 * 
		 */
		public static double [] Transformcos( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "cos"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on cos</p>
		 * 
		 */
		public static double [] Transformcos (double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
					copied_array[i]=Math.cos(array[i]);
			}
			return copied_array;
			//end of cos this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on cos</p>
		 * 
		 */
		public static void Transformcosthis( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "cos"));
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

		 * <p> transform based on cos</p>
		 * 
		 */
		public static void Transformcosthis( double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					array[i][j]=Math.cos(array[i][j]);
				
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on cos</p>
		 * 
		 */
		public static double [][] Transformcos( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "cos"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on cos</p>
		 * 
		 */
		public static double [][] Transformcos (double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					copied_array[i][j]=Math.cos(array[i][j]);
				}
			}
			return copied_array;
			//end of cos this method
		}
			
		
		
		
		
		
		
		//---------------------------------tan transforms----------------------------------//
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on tan</p>
		 * 
		 */
		public static void Transformtanthis( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "tan"));
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
		 * <p> transform based on tan</p>
		 * 
		 */
		public static void Transformtanthis( double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
					array[i]=Math.tan(array[i]);
				
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on tan</p>
		 * 
		 */
		public static double [] Transformtan( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "tan"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on tan</p>
		 * 
		 */
		public static double [] Transformtan (double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
					copied_array[i]=Math.tan(array[i]);
			}
			return copied_array;
			//end of tan this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on tan</p>
		 * 
		 */
		public static void Transformtanthis( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "tan"));
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

		 * <p> transform based on tan</p>
		 * 
		 */
		public static void Transformtanthis( double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					array[i][j]=Math.tan(array[i][j]);
				
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on tan</p>
		 * 
		 */
		public static double [][] Transformtan( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "tan"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on tan</p>
		 * 
		 */
		public static double [][] Transformtan (double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					copied_array[i][j]=Math.tan(array[i][j]);
				}
			}
			return copied_array;
			//end of tan this method
		}
			

		
		
		//---------------------------------tanh transforms----------------------------------//
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on tanh</p>
		 * 
		 */
		public static void Transformtanhthis( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "tanh"));
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
		 * <p> transform based on tanh</p>
		 * 
		 */
		public static void Transformtanhthis( double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
					array[i]=Math.tanh(array[i]);
				
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on tanh</p>
		 * 
		 */
		public static double [] Transformtanh( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "tanh"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on tanh</p>
		 * 
		 */
		public static double [] Transformtanh (double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
					copied_array[i]=Math.tanh(array[i]);
			}
			return copied_array;
			//end of tanh this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on tanh</p>
		 * 
		 */
		public static void Transformtanhthis( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "tanh"));
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

		 * <p> transform based on tanh</p>
		 * 
		 */
		public static void Transformtanhthis( double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
				if (array[i][j]>0){
					array[i][j]=Math.tanh(array[i][j]);
				}
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on tanh</p>
		 * 
		 */
		public static double [][] Transformtanh( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "tanh"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on tanh</p>
		 * 
		 */
		public static double [][] Transformtanh (double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					copied_array[i][j]=Math.tanh(array[i][j]);
				}
			}
			return copied_array;
			//end of tanh this method
		}
			

		
		
		
		
		
		//---------------------------------sig transforms----------------------------------//
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on sig</p>
		 * 
		 */
		public static void Transformsigthis( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "sig"));
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
		 * <p> transform based on sig</p>
		 * 
		 */
		public static void Transformsigthis( double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
					array[i]=1.0/ (1 + Math.exp(-array[i]));
				
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on sig</p>
		 * 
		 */
		public static double [] Transformsig( double array [], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "sig"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on sig</p>
		 * 
		 */
		public static double [] Transformsig (double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {
					copied_array[i]=1.0/ (1 + Math.exp(-array[i]));
			}
			return copied_array;
			//end of sig this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * <p> transform based on sig</p>
		 * 
		 */
		public static void Transformsigthis( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "sig"));
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

		 * <p> transform based on sig</p>
		 * 
		 */
		public static void Transformsigthis( double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					array[i][j]=1.0/ (1 + Math.exp(-array[i][j]));
				
				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @return transformed 1d array
		 * <p> transform based on sig</p>
		 * 
		 */
		public static double [][] Transformsig( double array [][], int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "sig"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @return transformed 1d array
		 * <p> transform based on sig</p>
		 * 
		 */
		public static double [][] Transformsig (double array [][]) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					copied_array[i][j]=1.0/ (1 + Math.exp(-array[i][j]));
				}
			}
			return copied_array;
			//end of sig this method
		}
			
		
		
		
		
		
		//---------------------------------pow transforms----------------------------------//		
		
		

		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
		 * @param pow pow value
		 * <p> transform based on pow</p>
		 * 
		 */
		public static void Transformpowthis( double array [], double pow, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.pow=pow;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "pow"));
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
		 * @param threads number of threads to use
         * @param pow pow value to use
		 * <p> transform based on pow</p>
		 * 
		 */
		public static void Transformpowthis( double array [], double pow) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
					array[i]=Math.pow(array[i],pow);
				
	
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param threads number of threads to use
         * @param pow pow value to use
		 * @return transformed 1d array
		 * <p> transform based on pow</p>
		 * 
		 */
		public static double [] Transformpow( double array [],double pow, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.pow=pow;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "pow"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * @param pow pow value to use
		 * @return transformed 1d array
		 * <p> transform based on pow</p>
		 * 
		 */
		public static double [] Transformpow (double array [], double pow) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			for (int i=0; i < array.length; i++) {

					copied_array[i]=Math.pow(array[i],pow);
				
			
			}
			return copied_array;
			//end of pow this method
		}	
		

		/**
		 * 
		 * @param array Array (this) to process
		 * @param pow pow value to use
		 * <p> transform based on pow</p>
		 * 
		 */
		public static void Transformpowthis( double array [][], double pow, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
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
				TransformRunnable.pow=pow;
				thread_array[n]= new Thread(new TransformRunnable(array, locations[n][0],locations[n][1], "pow"));
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
         * @param pow pow value to use
		 * <p> transform based on pow</p>
		 * 
		 */
		public static void Transformpowthis( double array [][], double pow) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}

			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){
					array[i][j]=Math.pow(array[i][j],pow);

				}
			
			}
			//end of multiply this method
		}
			
		
		
		/**
		 * 
		 * @param array Array (this) to process
         * @param pow pow value to use
		 * @return transformed 1d array
		 * <p> transform based on pow</p>
		 * 
		 */
		public static double [][] Transformpow( double array [][], double pow, int threads) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
			if (threads<=0) {
				threads=1;
			}
			
			if (threads>array.length) {
				threads=array.length;
			}	

	        double[][] copied_array=manipulate.copies.copies.Copy(array,threads);		
			int length_of_each_threaded_pass = copied_array.length/threads;
			int points=0;
			
			int locations[][] = new int[threads][2];
			
			for (int n=0; n <threads-1; n++ ){
				locations[n][0]=points;
				locations[n][1]=points + length_of_each_threaded_pass;
				points+=length_of_each_threaded_pass;
			}
			locations[threads-1][0]=points;
			locations[threads-1][1]=copied_array.length;
			
			Thread[] thread_array= new Thread[threads];
			
			for (int n=0; n <threads; n++ ){
				TransformRunnable.pow=pow;
				thread_array[n]= new Thread(new TransformRunnable(copied_array, locations[n][0],locations[n][1], "pow"));
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
			return copied_array;
			//end of multiply this method
		}
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
         * @param pow pow value to use
		 * @return transformed 1d array
		 * <p> transform based on pow</p>
		 * 
		 */
		public static double [][] Transformpow (double array [][], double pow) {
			
			// sensible checks
			if (array==null || array.length==0){
				
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[][]= new double [array.length][array[0].length];
			
			for (int i=0; i < array.length; i++) {
				for (int j=0; j <array[i].length;j++ ){

					copied_array[i][j]=Math.pow(array[i][j],pow);
			
				}
			}
			return copied_array;
			//end of pow this method
		}
				
		
		
		
		
		
		/**
		 * 
		 * @param array Array (this) to process
		 * <p> Make certain the elements of the array sum to 1</p>
		 * 
		 */
		public static double [] scaleweight (double array []) {
			
			// sensible checks
			if (array==null || array.length==0){
				throw new NullObjectException (" Array to transform is null or empty ");
			}
	        
			double  copied_array[]= new double [array.length];
			
			double min=Double.MAX_VALUE;
			double sum=0.0;
			
			for (int i=0; i < array.length; i++) {
					if (array[i]<min){
						min=array[i];
					}
					sum+=array[i];
				
			}
			if (min<0) {
				sum=0.0;
				for (int i=0; i < array.length; i++) {
					copied_array[i]=	array[i]+Math.abs(min);
					sum+=copied_array[i];
				}
				
				for (int i=0; i < array.length; i++) {
					copied_array[i]/=	sum;
				}				
			} else {
				for (int i=0; i < array.length; i++) {
					copied_array[i]=	array[i]/sum;
				}
			}
			
			return copied_array;
			//end of pow this method
		}
			
		
		
		
		
		
		
	
	
}
