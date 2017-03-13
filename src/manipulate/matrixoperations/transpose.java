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

package manipulate.matrixoperations;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to transpose arrays</p>
 *
 **/
public class transpose {
	
	/**
	 * 
	 * @param array_to_copy_from Array to transpose
	 * @param threads number of threads to use
	 * @return the transposed array
	 */
	public static double [][] Copy(  double array_to_copy_from [][], int threads) {
		
		// sensible checks
		if (array_to_copy_from==null || array_to_copy_from.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads> array_to_copy_from.length) {
			threads= array_to_copy_from.length;
		}	
		
		// initialize the new arry to copy to
		double new_array_to_transpose_to [][]= new double [array_to_copy_from[0].length][array_to_copy_from.length];

		
		// the threads of operations
				
		
		Thread[] thread_array= new Thread[threads];
		int count_of_current_threads=0;
		
		
		for (int i=0; i <array_to_copy_from[0].length; i++ ){
			
				thread_array[count_of_current_threads]=new Thread(new TransposeRunnable(new_array_to_transpose_to,array_to_copy_from,i ,i ));
				thread_array[count_of_current_threads].start();
				
				count_of_current_threads++;
				if (count_of_current_threads==threads || i==array_to_copy_from[0].length-1){
					
					for (int n=0; n <count_of_current_threads; n++ ){
						try {
							thread_array[n].join();
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
						   System.out.println(e.getMessage());
						}

					}
					 thread_array= new Thread[threads];
					count_of_current_threads=0;
				}
			}
			
		return new_array_to_transpose_to;
		//end of transpose method		
	}
	

		
		/**
		 * 
		 * @param array_to_copy_from Array to transpose
		 * @param threads number of threads to use
		 * @return the transposed array
		 */
		public static double [][] Copy(  double array_to_copy_from [][]) {
			
			// sensible checks
			if (array_to_copy_from==null || array_to_copy_from.length==0){
				
				throw new NullObjectException (" The  Array to copy from is null or empty ");
			}
			
			// initialize the new arry to copy to
			double new_array_to_transpose_to [][]= new double [array_to_copy_from[0].length][array_to_copy_from.length];

			
			
			for (int i=0; i <array_to_copy_from.length; i++ ){	
				for (int j=0; j <array_to_copy_from[0].length; j++ ){
					new_array_to_transpose_to[j][i]=array_to_copy_from[i][j];
				}
					
				}

				
			return new_array_to_transpose_to;
			//end of transpose method		
		}
		
		
			
	
		


	}
	
