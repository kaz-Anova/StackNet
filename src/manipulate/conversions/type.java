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

package manipulate.conversions;

import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make hard-conversions of arrays with different types

 */
public class type {
	
	/**
	 * 
	 * @param array_to_copy_from double Array to convert
	 * @param threads number of threads to use
	 * @return the String converted array
	 */
	public static String [] Convert( final double array_to_copy_from [], int threads) {
		
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
		
		// Initialise the new array to copy to
		String new_array_to_copy_to []= new String [array_to_copy_from.length];
		
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
			thread_array[n]= new Thread(new TypeConversionRunnable(array_to_copy_from, new_array_to_copy_to, locations[n][0], locations[n][1]));
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
	 * @param array_to_copy_from double Array to convert
	 * @return the String converted array
	 */
	public static String [] Convert( final double array_to_copy_from []) {
		
		// sensible checks
		if (array_to_copy_from==null || array_to_copy_from.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		
		// Initialise the new array to copy to
		String new_array_to_copy_to []= new String [array_to_copy_from.length];
		
		for (int i=0; i < new_array_to_copy_to.length; i++) {
			new_array_to_copy_to[i]=array_to_copy_from[i] + "";
		}
		
	return new_array_to_copy_to;
		//end of copy method
	}
	
	
	
	/**
	 * 
	 * @param array_to_copy_from double Array to convert
	 * @param threads number of threads to use
	 * @return the String converted array
	 */
	public static String [][] Convert( final double array_to_copy_from [][], int threads) {
		
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
		
		// Initialise the new array to copy to
		String new_array_to_copy_to [][]= new String [array_to_copy_from.length][];
		for (int i=0 ; i <array_to_copy_from.length; i++ ){
			new_array_to_copy_to[i]= new String [array_to_copy_from[i].length];
		}
		
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
			thread_array[n]= new Thread(new TypeConversionRunnable(array_to_copy_from, new_array_to_copy_to, locations[n][0], locations[n][1]));
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
	 * @param array_to_copy_from double Array to convert
	 * @return the String converted array
	 */
	public static String [][] Convert( final double array_to_copy_from [][]) {
		
		// sensible checks
		if (array_to_copy_from==null || array_to_copy_from.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		
		
		// Initialise the new array to copy to
		String new_array_to_copy_to [][]= new String [array_to_copy_from.length][];
		
		for (int i=0 ; i <array_to_copy_from.length; i++ ){
			String temp[] = new String [array_to_copy_from[i].length];
			double single[]=array_to_copy_from[i];
			for (int j=0 ; j <temp.length; j++ ){
				temp[j]=single[j]+ "";
			}
			new_array_to_copy_to[i]= temp;
		}
		

	return new_array_to_copy_to;
		//end of copy method
	}
		
	
	
	
	/**
	 * 
	 * @param array_to_copy_from String Array to convert
	 * @param threads number of threads to use
	 * @return the double converted array
	 */
	public static double [] Convert( final String array_to_copy_from [], int threads) {
		
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
		
		// Initialise the new array to copy to
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
			thread_array[n]= new Thread(new TypeConversionRunnable(array_to_copy_from, new_array_to_copy_to, locations[n][0], locations[n][1]));
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
	 * @param array_to_copy_from String Array to convert
	 * @return the double converted array
	 */
	public static double [] Convert( final  String  array_to_copy_from []) {
		
		// sensible checks
		if (array_to_copy_from==null || array_to_copy_from.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		
		// Initialise the new array to copy to
		double new_array_to_copy_to []= new double [array_to_copy_from.length];
		
		for (int i=0; i < new_array_to_copy_to.length; i++) {
			new_array_to_copy_to[i]=Double.parseDouble(array_to_copy_from[i]);
		}
		
	return new_array_to_copy_to;
		//end of copy method
	}
	
	
	
	/**
	 * 
	 * @param array_to_copy_from String Array to convert
	 * @param threads number of threads to use
	 * @return the double converted array
	 */
	public static double [][] Convert( final String array_to_copy_from [][], int threads) {
		
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
		
		// Initialise the new array to copy to
		double new_array_to_copy_to [][]= new double [array_to_copy_from.length][];
		for (int i=0 ; i <array_to_copy_from.length; i++ ){
			new_array_to_copy_to[i]= new double [array_to_copy_from[i].length];
		}
		
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
			thread_array[n]= new Thread(new TypeConversionRunnable(array_to_copy_from, new_array_to_copy_to, locations[n][0], locations[n][1]));
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
	 * @param array_to_copy_from String Array to convert
	 * @return the double converted array
	 */
	public static double [][] Convert( final String  array_to_copy_from [][]) {
		
		// sensible checks
		if (array_to_copy_from==null || array_to_copy_from.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		
		
		// Initialise the new array to copy to
		double new_array_to_copy_to [][]= new double [array_to_copy_from.length][];
		
		for (int i=0 ; i <array_to_copy_from.length; i++ ){
			double  temp[] = new double [array_to_copy_from[i].length];
			String single[]=array_to_copy_from[i];
			for (int j=0 ; j <temp.length; j++ ){
				temp[j]=Double.parseDouble(single[j]);
			}
			new_array_to_copy_to[i]= temp;
		}
		

	return new_array_to_copy_to;
		//end of copy method
	}	
	
	
	

	
	
}
