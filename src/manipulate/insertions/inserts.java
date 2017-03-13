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

package manipulate.insertions;
import exceptions.LessThanMinimum;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to make appends on arrays

 */
public class inserts {
	
	//--------------------------------------1d inserts--------------------------------//
	/**
	 * 
	 * @param array_append_to Array to insert to ( base )
	 * @param array_append_from Array to insert from
	 * @param start start location in the base array
	 * @param end end location in the base array
	 * @param threads number of threads to use
	 * @return the array with the insertion included
	 */
	public static double [] Insert( double array_append_to [],  double array_append_from [], int start, int end, int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;

		} else if (array_append_from!=null && array_append_to==null) {
			return array_append_from;
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (start <0) {
				throw new LessThanMinimum (start,0);
		}
		if (end <start) {
			throw new LessThanMinimum (end,start);
	   }
		if (end - start!=array_append_from.length) {
			throw new IllegalStateException(" the length of the given arry needs to match the 'end-start' range as it states where the new array should fit to in the base array ");
	   }
		if (start > array_append_to.length || end> array_append_to.length) {
				throw new IllegalStateException(" the given range exceeds the arrays' dimensions ");
		}		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && (array_append_from==null || array_append_from.length==0)  ){
			
			throw new NullObjectException (" The Arrays to insert to/from are null or empty ");
		}
		//replace number of threads with sensible values
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>end-start) {
			threads=end-start;
		}	
		
		// initialize the new array to copy to
		double new_array_to_copy_to1d []= manipulate.copies.copies.Copy(array_append_to);
		
		int length_of_each_threaded_pass =(end-start)/threads;
		int points=start;
		int points2nd=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][4];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			locations[n][2]=points2nd;
			locations[n][3]=points2nd + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
			points2nd+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=end;	
		locations[threads-1][2]=points2nd;
		locations[threads-1][3]=array_append_from.length;	
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new InsertRunnable(new_array_to_copy_to1d,array_append_from , locations[n][0], locations[n][1], locations[n][2], locations[n][3]));
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
	}
	
	/**
	 * 
	 * @param array_append_to Array to insert to ( base )
	 * @param array_append_from Array to insert from
	 * @param start start location in the base array
	 * @param end end location in the base array
	 * @return the array with the insertion included
	 */
	public static double [] Insert( double array_append_to [],  double array_append_from [], int start, int end) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;

		} else if (array_append_from!=null && array_append_to==null) {
			return array_append_from;
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (start <0) {
				throw new LessThanMinimum (start,0);
		}
		if (end <start) {
			throw new LessThanMinimum (end,start);
	   }
		if (end - start!=array_append_from.length) {
			throw new IllegalStateException(" the length of the given arry needs to match the 'end-start' range as it states where the new array should fit to in the base array ");
	   }
		if (start > array_append_to.length || end> array_append_to.length) {
				throw new IllegalStateException(" the given range exceeds the arrays' dimensions ");
		}		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && (array_append_from==null || array_append_from.length==0)  ){
			
			throw new NullObjectException (" The Arrays to insert to/from are null or empty ");
		}
	
		
		// initialize the new array to copy to
		double new_array_to_copy_to1d []= manipulate.copies.copies.Copy(array_append_to);
		
		for (int i=start ,n=0 ; n < array_append_from.length; i++, n++) {
			new_array_to_copy_to1d [i]=array_append_from[n];
		}			
		
		
    return new_array_to_copy_to1d;
	}	


}

	/**
	 * 
	 * @param array_append_to Array to insert to ( base )
	 * @param array_append_from Array to insert from
	 * @param start start location in the base array
	 * @param end end location in the base array
	 * @param threads number of threads to use
	 */
	public static void Insertthis( double array_append_to [],  double array_append_from [], int start, int end, int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){

		} else if (array_append_from!=null && array_append_to==null) {
			 array_append_to=array_append_from;
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (start <0) {
				throw new LessThanMinimum (start,0);
		}
		if (end <start) {
			throw new LessThanMinimum (end,start);
	   }
		if (end - start!=array_append_from.length) {
			throw new IllegalStateException(" the length of the given arry needs to match the 'end-start' range as it states where the new array should fit to in the base array ");
	   }
		if (start > array_append_to.length || end> array_append_to.length) {
				throw new IllegalStateException(" the given range exceeds the arrays' dimensions ");
		}		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && (array_append_from==null || array_append_from.length==0)  ){
			
			throw new NullObjectException (" The Arrays to insert to/from are null or empty ");
		}
		//replace number of threads with sensible values
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>end-start) {
			threads=end-start;
		}	
		

		
		int length_of_each_threaded_pass =(end-start)/threads;
		int points=start;
		int points2nd=0;
		
		// the threads of operations
		
		int locations[][] = new int[threads][4];
		
		for (int n=0; n <threads-1; n++ ){
			locations[n][0]=points;
			locations[n][1]=points + length_of_each_threaded_pass;
			locations[n][2]=points2nd;
			locations[n][3]=points2nd + length_of_each_threaded_pass;
			points+=length_of_each_threaded_pass;
			points2nd+=length_of_each_threaded_pass;
		}
		locations[threads-1][0]=points;
		locations[threads-1][1]=end;	
		locations[threads-1][2]=points2nd;
		locations[threads-1][3]=array_append_from.length;	
		
		Thread[] thread_array= new Thread[threads];
		
		for (int n=0; n <threads; n++ ){
			thread_array[n]= new Thread(new InsertRunnable(array_append_to,array_append_from , locations[n][0], locations[n][1], locations[n][2], locations[n][3]));
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

		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to Array to insert to ( base )
	 * @param array_append_from Array to insert from
	 * @param start start location in the base array
	 * @param end end location in the base array
	 */
	public static void Insertthis( double array_append_to [],  double array_append_from [], int start, int end) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			

		} else if (array_append_from!=null && array_append_to==null) {
			array_append_to =array_append_from;
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (start <0) {
				throw new LessThanMinimum (start,0);
		}
		if (end <start) {
			throw new LessThanMinimum (end,start);
	   }
		if (end - start!=array_append_from.length) {
			throw new IllegalStateException(" the length of the given arry needs to match the 'end-start' range as it states where the new array should fit to in the base array ");
	   }
		if (start > array_append_to.length || end> array_append_to.length) {
				throw new IllegalStateException(" the given range exceeds the arrays' dimensions ");
		}		
		// sensible checks
		if ( (array_append_to==null || array_append_to.length==0)  && (array_append_from==null || array_append_from.length==0)  ){
			
			throw new NullObjectException (" The Arrays to insert to/from are null or empty ");
		}

		// initialize the new array to copy to
		
		for (int i=start ,n=0 ; n < array_append_from.length; i++, n++) {
			array_append_to [i]=array_append_from[n];
		}			
		

	}	


}	
	
	
	//--------------------------------------1d inserts on 2d--------------------------------//
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 1d Array to insert from
	 * @param column Column to insert to
	 * @param threads number of threads to use
	 * @return the array with the insertion included
	 */
	public static double [][] Insert( double array_append_to [][],  double array_append_from [], int column, int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;

		} else if (array_append_from!=null && array_append_to==null) {
			return manipulate.conversions.dimension.Convert(array_append_from);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (column <0 || column > array_append_to[0].length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		//replace number of threads with sensible values
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// initialize the new array to copy to
		double new_array_to_copy_to2d [][]= manipulate.copies.copies.Copy(array_append_to);	
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
			thread_array[n]= new Thread(new InsertRunnable(new_array_to_copy_to2d, array_append_from, column,locations[n][0], locations[n][1]));
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
		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 1d Array to insert from
	 * @param column Column to insert to
	 * @return the array with the insertion included
	 */
	public static double [][] Insert( double array_append_to [][],  double array_append_from [], int column) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;

		} else if (array_append_from!=null && array_append_to==null) {
			return manipulate.conversions.dimension.Convert(array_append_from);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (column <0 || column > array_append_to[0].length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		
		// initialize the new array to copy to
		double new_array_to_copy_to2d [][]= manipulate.copies.copies.Copy(array_append_to);	
		
		for (int i=0; i <new_array_to_copy_to2d.length; i++ ){
			new_array_to_copy_to2d[i][column]=array_append_from[i];
		}

	return new_array_to_copy_to2d;
		//end of copy method
		//end of copy method
	
		}
	}
	
		
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 1d Array to insert from
	 * @param column Column to insert to
	 * @param threads number of threads to use
	 * @return the array with the insertion included
	 */
	public static void Insertthis( double array_append_to [][],  double array_append_from [], int column, int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			

		} else if (array_append_from!=null && array_append_to==null) {
			 array_append_to=manipulate.conversions.dimension.Convert(array_append_from);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (column <0 || column > array_append_to[0].length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		//replace number of threads with sensible values
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// initialize the new array to copy to
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
			thread_array[n]= new Thread(new InsertRunnable(array_append_to, array_append_from, column,locations[n][0], locations[n][1]));
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

		//end of copy method
		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 1d Array to insert from
	 * @param column Column to insert to
	 */
	public static void Insertthis( double array_append_to [][],  double array_append_from [], int column) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			

		} else if (array_append_from!=null && array_append_to==null) {
			array_append_to= manipulate.conversions.dimension.Convert(array_append_from);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (column <0 || column > array_append_to[0].length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		

		
		for (int i=0; i <array_append_to.length; i++ ){
			array_append_to[i][column]=array_append_from[i];
		}

		//end of copy method
		//end of copy method
	
		}
	}	
	

	
	//--------------------------------------2d inserts on 2d--------------------------------//
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 2d Array to insert from
	 * @param columnto Columns to insert to
	 * @param columnfrom Columns to insert from 
	 * @param threads number of threads to use
	 * @return the array with the insertion included
	 */
	public static double [][] Insert( double array_append_to [][],  double array_append_from [][], int columnto[], int columnfrom[], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;

		} else if (array_append_from!=null && array_append_to==null) {
			return manipulate.select.columnselect.ColumnSelect(array_append_from, columnfrom,threads);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (columnto.length >array_append_to[0].length || columnfrom.length > array_append_from[0].length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		//replace number of threads with sensible values
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// initialize the new array to copy to
		double new_array_to_copy_to2d [][]= manipulate.copies.copies.Copy(array_append_to);	
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
			thread_array[n]= new Thread(new InsertRunnable(new_array_to_copy_to2d, array_append_from, columnto, columnfrom,locations[n][0], locations[n][1]));
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
		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 2d Array to insert from
	 * @param columnto Columns to insert to
	 * @param columnfrom Columns to insert from 

	 * @return the array with the insertion included
	 */
	public static double [][] Insert( double array_append_to [][],  double array_append_from [][], int columnto[], int columnfrom[]) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			return array_append_to;

		} else if (array_append_from!=null && array_append_to==null) {
			return manipulate.select.columnselect.ColumnSelect(array_append_from, columnfrom);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (columnto.length >array_append_to[0].length || columnfrom.length > array_append_from[0].length || columnto.length!=columnfrom.length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		
		// initialize the new array to copy to
		double new_array_to_copy_to2d [][]= manipulate.copies.copies.Copy(array_append_to);	
		
		for (int i=0; i < new_array_to_copy_to2d.length; i++) {
			
			for (int j=0; j < columnto.length; j++) {
				
				new_array_to_copy_to2d [i][columnto[j]]=array_append_from[i][columnfrom[j]];
				
			}

		}				
	return new_array_to_copy_to2d;
		//end of copy method
		//end of copy method
	
		}
	}
		
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 2d Array to insert from
	 * @param columnto Columns to insert to
	 * @param columnfrom Columns to insert from 
	 * @param threads number of threads to use
	 */
	public static void Insertthis( double array_append_to [][],  double array_append_from [][], int columnto[], int columnfrom[], int threads) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			

		} else if (array_append_from!=null && array_append_to==null) {
			//return null;
			array_append_to=manipulate.select.columnselect.ColumnSelect(array_append_from, columnfrom,threads);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (columnto.length >array_append_to[0].length || columnfrom.length > array_append_from[0].length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		//replace number of threads with sensible values
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>array_append_to.length) {
			threads=array_append_to.length;
		}	
		
		// initialize the new array to copy to

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
			thread_array[n]= new Thread(new InsertRunnable(array_append_to, array_append_from, columnto, columnfrom,locations[n][0], locations[n][1]));
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
		//end of copy method
		//end of copy method
	
		}
	}
	
	/**
	 * 
	 * @param array_append_to 2d Array to insert to ( base )
	 * @param array_append_from 2d Array to insert from
	 * @param columnto Columns to insert to
	 * @param columnfrom Columns to insert from 

	 */
	public static void Insertthis( double array_append_to [][],  double array_append_from [][], int columnto[], int columnfrom[]) {
		
		// return the second array if the first is null
		if (array_append_from==null && array_append_to!=null){
			
			//return array_append_to;

		} else if (array_append_from!=null && array_append_to==null) {
			//return null;
			array_append_to=manipulate.select.columnselect.ColumnSelect(array_append_from, columnfrom);
		}
		else {
		// start of the loop cannot be less than zero or less than the end of the loop
		if (columnto.length >array_append_to[0].length || columnfrom.length > array_append_from[0].length || columnto.length!=columnfrom.length) {
			throw new IllegalStateException(" The given column must fall within the range of the given array ");
		}

		for (int i=0; i < array_append_to.length; i++) {
			
			for (int j=0; j < columnto.length; j++) {
				
				array_append_to [i][columnto[j]]=array_append_from[i][columnfrom[j]];
				
			}

		}				

		//end of copy method
		//end of copy method
	
		}
	}	
	/**
	 * 
	 * @param data : provided data
	 * @return new double array with one more column that has the value of 1 in each row
	 */
	public static double [][] addconstant(double data [][]){
		double new_data[][]= new double[data.length][data[0].length+1];
		for (int i=0; i< data.length; i++){
			new_data[i][0]=1.0;
			for (int j=0; j < data[0].length; j++){
				new_data[i][j+1]=data[i][j];
			}
		}
		
		return new_data;
	
	}
}
