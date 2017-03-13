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


import exceptions.DimensionMismatchException;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make appends on arrays

 */
public class append {
	
	//--------------------------------------1d appends--------------------------------//
	/**
	 * 
	 * @param array_append_to Array to append to
	 * @param array_append_from Array to append from
	 * @param threads number of threads to use
	 * @return the appended array
	 */
	public static double [][] Append( double array_append_to [],  double array_append_from [], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return manipulate.conversions.dimension.Convert(array_append_to, threads);
		} else if (array_append_from!=null && array_append_to==null) {
			return manipulate.conversions.dimension.Convert(array_append_from, threads);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_from.length) {
			threads=array_append_from.length;
		}	
		
		// Initialise the new array to copy to
		double new_array_to_copy_to2d [][]= new double [array_append_from.length][2];
		
		int length_of_each_threaded_pass = array_append_from.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array_append_from.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new AppendRunnable(new_array_to_copy_to2d,array_append_to , array_append_from, locations[n][0], locations[n][1]));
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
	 * @param array_append_to Array to append to
	 * @param array_append_from Array to append from
	 * @return the appended array
	 */
	public static double [][] Append( double array_append_to [],  double array_append_from []) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return manipulate.conversions.dimension.Convert(array_append_to);
		} else if (array_append_from!=null && array_append_to==null) {
			return manipulate.conversions.dimension.Convert(array_append_from);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
	
		
		// Initialise the new array to copy to
		double new_array_to_copy_to2d [][]= new double [array_append_from.length][2];
		for (int i=0; i <new_array_to_copy_to2d.length; i++ ) {
			new_array_to_copy_to2d[i][0]=array_append_to[i];
			new_array_to_copy_to2d[i][1]=array_append_from[i];			
		}
		

	return new_array_to_copy_to2d;
		//end of copy method
	
		}
	}
	
	
	//-----------------------------------------1d Appendings on 2d--------------------------------------------------//
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from Array to append from
	 * @param threads number of threads to use
	 * @return the appended array
	 */
	public static double [][] Append( double array_append_to [][],  double array_append_from [], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return manipulate.copies.copies.Copy(array_append_to,threads) ;
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			return manipulate.conversions.dimension.Convert(array_append_from,threads);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_from.length) {
			threads=array_append_from.length;
		}	
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  manipulate.copies.copies.Copy(array_append_to,threads) ;
		
		int length_of_each_threaded_pass = array_append_from.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array_append_from.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new AppendRunnable(new_array_to_copy_to2d , array_append_from, locations[n][0], locations[n][1]));
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
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from Array to append from
	 * @return the appended array
	 */
	public static double [][] Append( double array_append_to [][],  double array_append_from []) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return manipulate.copies.copies.Copy(array_append_to) ;
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			return manipulate.conversions.dimension.Convert(array_append_from);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
		// initialize the new arry to copy to
		double new_array_to_copy_to2d [][]= manipulate.copies.copies.Copy(array_append_to);
		

		
		for (int i=0; i < new_array_to_copy_to2d.length; i++) {
			double temp_first[]=new_array_to_copy_to2d[i];
			double temp []=new double [temp_first.length + 1];
			for (int j=0; j < temp_first.length; j++) {
				temp[j]=temp_first[j];
			}
			//add element from single array
			temp[temp.length-1]=array_append_from[i];
			new_array_to_copy_to2d[i]=temp;
			
		}
		

	return new_array_to_copy_to2d;
		//end of copy method
	
		}
	}
	
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from Array to append from
	 * @param threads number of threads to use
	 */
	public static void Appendthis( double array_append_to [][],  double array_append_from [], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			// do nothing
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			array_append_to= manipulate.conversions.dimension.Convert(array_append_from,threads);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_from.length) {
			threads=array_append_from.length;
		}	
		

		
		int length_of_each_threaded_pass = array_append_from.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array_append_from.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new AppendRunnable(array_append_to , array_append_from, locations[n][0], locations[n][1]));
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
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from Array to append from
	 */
	public static void Appendthis( double array_append_to [][],  double array_append_from []) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			// do nothing
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			array_append_to= manipulate.conversions.dimension.Convert(array_append_from);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		

		

		for (int i=0; i < array_append_to.length; i++) {
			double temp_first[]=array_append_to[i];
			double temp []=new double [temp_first.length + 1];
			for (int j=0; j < temp_first.length; j++) {
				temp[j]=temp_first[j];
			}
			//add element from single array
			temp[temp.length-1]=array_append_from[i];
			array_append_to[i]=temp;
		}
		
		//end of copy method
	
		}
	}
		

	
	//---------------------------------------------------------------------2D APPENDINGS---------------------------------------------------------//
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from 2d Array to append from
	 * @param threads number of threads to use
	 * @return the appended array
	 */
	public static double [][] Append( double array_append_to [][],  double array_append_from [][], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return manipulate.copies.copies.Copy(array_append_to,threads) ;
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			return  manipulate.copies.copies.Copy(array_append_from,threads);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_from.length) {
			threads=array_append_from.length;
		}	
		
		// initialize the new array to append to
		double new_array_to_copy_to2d [][]=  manipulate.copies.copies.Copy(array_append_to,threads) ;
		
		int length_of_each_threaded_pass = array_append_from.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array_append_from.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new AppendRunnable(new_array_to_copy_to2d , array_append_from, locations[n][0], locations[n][1]));
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
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from 2d Array to append from
	 * @return the appended array
	 */
	public static double [][] Append( double array_append_to [][],  double array_append_from [][]) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return manipulate.copies.copies.Copy(array_append_to) ;
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			return manipulate.copies.copies.Copy(array_append_from);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
		// initialize the new arry to copy to
		double new_array_to_copy_to2d [][]= manipulate.copies.copies.Copy(array_append_to);
		
		
		for (int i=0; i < new_array_to_copy_to2d.length; i++) {
			double temp_first[]=new_array_to_copy_to2d[i];
			double temp_second[]=array_append_from[i];
			double temp []=new double [temp_first.length + temp_second.length];
			for (int j=0; j < temp_first.length; j++) {
				temp[j]=temp_first[j];
			}
			for (int j=0; j < temp_second.length; j++) {
				temp[j+temp_first.length]=temp_second[j];
			}
			//add element from single array
			new_array_to_copy_to2d[i]=temp;
		}
		

	return new_array_to_copy_to2d;
		//end of copy method
	
		}
	}
	
	
	
	/**
	 * 
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from 2d Array to append from
	 * @param threads number of threads to use
	 */
	public static void Appendthis( double array_append_to [][],  double array_append_from [][], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			// do nothing
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			array_append_to=manipulate.copies.copies.Copy(array_append_from,threads);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_from.length) {
			threads=array_append_from.length;
		}	
		

		
		int length_of_each_threaded_pass = array_append_from.length/threads;
		int points=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][2];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=array_append_from.length;		
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new AppendRunnable(array_append_to , array_append_from, locations[n][0], locations[n][1]));
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
	 * @param array_append_to 2d Array to append to
	 * @param array_append_from 2d Array to append from
	 */
	public static void Appendthis( double array_append_to [][],  double array_append_from [][]) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			// do nothing
			
		} else if (array_append_from!=null && array_append_to==null) {
			
			array_append_to= manipulate.copies.copies.Copy(array_append_from);
		}
		else {
		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && ( array_append_from==null || array_append_from.length==0 )  ){
			
			throw new NullObjectException (" The Arrasy to append from are null or empty ");
		}
		
		if (array_append_to.length!= array_append_from.length){
			
			throw new DimensionMismatchException (array_append_to.length, array_append_from.length);
		}	
		

		

		for (int i=0; i < array_append_to.length; i++) {
			double temp_first[]=array_append_to[i];
			double temp_second[]=array_append_from[i];
			double temp []=new double [temp_first.length + 1];
			for (int j=0; j < temp_first.length; j++) {
				temp[j]=temp_first[j];
			}
			//add element from single array
			for (int j=0; j < temp_second.length; j++) {
				temp[j+temp_first.length]=temp_second[j];
			}
			//add element from single array
			array_append_to[i]=temp;
		}
		
		//end of copy method
	
		}
	}
		
}
