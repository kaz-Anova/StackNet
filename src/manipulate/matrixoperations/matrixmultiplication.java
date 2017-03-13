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
import exceptions.DimensionMismatchException;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 *<p> Purpose of the class is to multipliy arrays as mutrices

 */
public class matrixmultiplication {
	
	
	// -------------------------Normal Multiplication -------------------------//
	/**
	 * 
	 * @param array_1 : The first array that will be multiplied by row
	 * @param array_2 : the second array that will be multiplied by column
	 * @param threads : Number of threads to use
	 * @return the multiplication matrix of arrays 1 and 2
	 */
	public static double [][] multiply(  double array_1[][], double array_2[][] ,int threads) {
		
		// sensible checks
		if (array_1==null || array_1.length==0  || array_2==null || array_2.length==0){
			
			throw new NullObjectException (" The  Arrays to multiply cannot be null or empty ");
		}
		if (array_1[0].length!=array_2.length){
			throw new DimensionMismatchException  (array_1[0].length,array_2.length);
		}
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads> array_1[0].length *array_2.length ) {
			threads= array_1[0].length *array_2.length;
		}	
		
		// initialize the new arry to copy to
		double dotmatrix [][]= new double [array_1.length][array_2[0].length];
		
		// the threads of operations
				
		
		Thread[] thread_array= new Thread[threads];
		int count_of_current_threads=0;
		
		for (int i=0; i <array_1.length; i++ ){
			for (int j=0; i <array_2[0].length; j++ ){
				// create threads
				thread_array[count_of_current_threads]=new Thread(new DotProductRunnable(dotmatrix,array_1,array_2,i,j ,0 ));
				thread_array[count_of_current_threads].start();
				count_of_current_threads++;
				
				// join threads
				if (count_of_current_threads==threads || (i+1)*(j+1) ==array_1[0].length *array_2.length){
					
					for (int n=0; n <count_of_current_threads; n++ ){
						try {
							thread_array[n].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						}

					}
					 thread_array= new Thread[threads];
					count_of_current_threads=0;
				}				
				
				
			}
			
			
		}
		
		
		
			
		return dotmatrix;
		//end of transpose method		
	}
	

		

			
	
		/**
		 * 
		 * @param array_1 : The first array that will be multiplied by row
		 * @param array_2 : the second array that will be multiplied by column
		 * @return the multiplication matrix of arrays 1 and 2
		 */
		public static double [][] multiply(  double array_1[][], double array_2[][] ) {
			
			// sensible checks
			if (array_1==null || array_1.length==0  || array_2==null || array_2.length==0){
				
				throw new NullObjectException (" The  Arrays to multiply cannot be null or empty ");
			}
			if (array_1[0].length!=array_2.length){
				throw new DimensionMismatchException  (array_1[0].length,array_2.length);
			}

			// initialize the new arry to copy to
			double dotmatrix [][]= new double [array_1.length][array_2[0].length];
			
			
			for (int i=0; i <array_1.length; i++ ){
				for (int j=0; i <array_2[0].length; j++ ){
					for (int n=0; n <array_1[i].length; n++ )	{
						dotmatrix [i][j]+=array_1[i][n]* array_2[n][j];
					}
				}	
			}
			
			
			
				
			return dotmatrix;
			//end of transpose method		
		}
		

			
			
			// -------------------------Inverse Multiplication -------------------------//				
		
			/**
			 * 
			 * @param array_1 : The first array that will be multiplied by row
			 * @param array_2 : the second array that will be multiplied by column
			 * @param threads : Number of threads to use
			 * @return the multiplication matrix of arrays 1 and 2
			 */
			public static double [][] multiplytransposed(  double array_1[][], double array_2[][] ,int threads) {
				
				// sensible checks
				if (array_1==null || array_1.length==0  || array_2==null || array_2.length==0){
					
					throw new NullObjectException (" The  Arrays to multiply cannot be null or empty ");
				}
				if (array_1.length!=array_2.length){
					throw new DimensionMismatchException  (array_1.length,array_2.length);
				}
				
				if (threads<=0) {
					threads=1;
				}
				
				if (threads> array_1.length *array_2.length ) {
					threads= array_1.length *array_2.length;
				}	
				
				// initialize the new arry to copy to
				double dotmatrix [][]= new double [array_1[0].length][array_2[0].length];
				
				// the threads of operations
						
				
				Thread[] thread_array= new Thread[threads];
				int count_of_current_threads=0;
				
				for (int i=0; i <array_1[0].length; i++ ){
					for (int j=0; i <array_2[0].length; j++ ){
						// create threads
						thread_array[count_of_current_threads]=new Thread(new DotProductRunnable(dotmatrix,array_1,array_2,i,j ,1 ));
						thread_array[count_of_current_threads].start();
						count_of_current_threads++;
						
						// join threads
						if (count_of_current_threads==threads || (i+1)*(j+1) ==array_1[0].length *array_2.length){
							
							for (int n=0; n <count_of_current_threads; n++ ){
								try {
									thread_array[n].join();
								} catch (InterruptedException e) {
								   System.out.println(e.getMessage());
								}

							}
							 thread_array= new Thread[threads];
							count_of_current_threads=0;
						}				
						
						
					}
					
					
				}
				
				
				
					
				return dotmatrix;
				//end of transpose method		
			}
			

				

					
			
				/**
				 * 
				 * @param array_1 : The first array that will be multiplied by row
				 * @param array_2 : the second array that will be multiplied by column
				 * @return the multiplication matrix of arrays 1 and 2
				 */
				public static double [][] multiplytransposed(  double array_1[][], double array_2[][] ) {
					
					// sensible checks
					if (array_1==null || array_1.length==0  || array_2==null || array_2.length==0){
						
						throw new NullObjectException (" The  Arrays to multiply cannot be null or empty ");
					}
					if (array_1.length!=array_2.length){
						throw new DimensionMismatchException  (array_1.length,array_2.length);
					}

					// initialize the new arry to copy to
					double dotmatrix [][]= new double [array_1[0].length][array_2[0].length];
					
					
					for (int i=0; i <array_1[0].length; i++ ){
						for (int j=0; i <array_2[0].length; j++ ){
							for (int n=0; n <array_1.length; n++ )	{
								dotmatrix [i][j]+=array_1[n][i]* array_2[n][j];
							}
						}	
					}
					
					
					
						
					return dotmatrix;
					//end of transpose method		
				}
				

		


	}
	
