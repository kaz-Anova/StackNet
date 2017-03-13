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

import matrix.fsmatrix;
import exceptions.ConvergenceException;
import exceptions.DimensionMismatchException;
import exceptions.NullObjectException;

public class Inverse {
	
	
	/**
	 * 
	 * @param covariance : symmetric double matrix
	 * @return inverse matrix
	 */
	public static double [][] GetInverse(double covariance[][]) {
	
		if (covariance==null || covariance.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		if (covariance.length!=covariance[0].length){
			
			throw new DimensionMismatchException (covariance.length,covariance[0].length);
		}		
		
	
	 double [][] covariancev=manipulate.copies.copies.Copy(covariance);
		
   	 int row = covariancev.length;
     double[] t = new double[row];
     double[] Q = new double[row];
     double[] R = new double[row];
     double AB;


     // Invert a symetric matrix in V
     for (int M = 0; M < row; M++){
         R[M] = 1;
     }
     
     int K = 0;
     for (int M = 0; M < row; M++)
     {
         double Big = 0;
         for (int s = 0; s < row; s++)
         {
             AB = Math.abs(covariancev[s] [s]);
             if ((AB > Big) && (R[s] != 0))
             {
                 Big = AB;
                 K = s;
             }
         }

         
         R[K] = 0;
         Q[K] = 1 / covariancev[K] [K];
         t[K] = 1;
         
         covariancev[K] [K]= 0.0;
         
         if (K != 0)
         {
             for (int s = 0; s < K; s++)
             {
                 t[s] = covariancev[s] [K];
                 if (R[s] == 0)
                     Q[s] = covariancev[s] [K]* Q[K];
                 else
                     Q[s] = - covariancev[s] [K] * Q[K];
                 covariancev[s] [K]= 0.0;
             }
         }
         if ((K + 1) < row)
         {
             for (int s = K + 1; s< row; s++)
             {
                 if (R[s] != 0)
                     t[s] =   covariancev[K] [s];
                 else
                     t[s] = -covariancev[K] [s];
                 Q[s] = -covariancev[K] [s] * Q[K];
                 covariancev[K] [s]= 0.0;
             }
         }
         for (int s = 0; s < row; s++)
             for (K = s; K < row; K++)
            	 covariancev[s] [K]= covariancev[s] [K]+ t[s] * Q[K];
     }
     
     
     
     int M = row;
     int s = row - 1;
     
     for (K = 1; K < row; K++)
     {
         M = M - 1;
         s = s - 1;
         for (int J = 0; J <= s; J++)
        	 covariancev[M] [J]= covariancev[J] [M];
     }
     
     
 	return covariancev;
	}
	
	/**
	 * 
	 * @param covariance : symmetric double matrix
	 */
	public static void  GetInversethis(double covariancev[][]) {
		
		if (covariancev==null || covariancev.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		if (covariancev.length!=covariancev[0].length){
			
			throw new DimensionMismatchException (covariancev.length,covariancev[0].length);
		}		
		

		
   	 int row = covariancev.length;
     double[] t = new double[row];
     double[] Q = new double[row];
     double[] R = new double[row];
     double AB;


     // Invert a symetric matrix in V
     for (int M = 0; M < row; M++){
         R[M] = 1;
     }
     
     int K = 0;
     for (int M = 0; M < row; M++)
     {
         double Big = 0;
         for (int s = 0; s < row; s++)
         {
             AB = Math.abs(covariancev[s] [s]);
             if ((AB > Big) && (R[s] != 0))
             {
                 Big = AB;
                 K = s;
             }
         }

         
         R[K] = 0;
         Q[K] = 1 / covariancev[K] [K];
         t[K] = 1;
         
         covariancev[K] [K]= 0.0;
         
         if (K != 0)
         {
             for (int s = 0; s < K; s++)
             {
                 t[s] = covariancev[s] [K];
                 if (R[s] == 0)
                     Q[s] = covariancev[s] [K]* Q[K];
                 else
                     Q[s] = - covariancev[s] [K] * Q[K];
                 covariancev[s] [K]= 0.0;
             }
         }
         if ((K + 1) < row)
         {
             for (int s = K + 1; s< row; s++)
             {
                 if (R[s] != 0)
                     t[s] =   covariancev[K] [s];
                 else
                     t[s] = -covariancev[K] [s];
                 Q[s] = -covariancev[K] [s] * Q[K];
                 covariancev[K] [s]= 0.0;
             }
         }
         for (int s = 0; s < row; s++)
             for (K = s; K < row; K++)
            	 covariancev[s] [K]= covariancev[s] [K]+ t[s] * Q[K];
     }
     
     
     
     int M = row;
     int s = row - 1;
     
     for (K = 1; K < row; K++)
     {
         M = M - 1;
         s = s - 1;
         for (int J = 0; J <= s; J++)
        	 covariancev[M] [J]= covariancev[J] [M];
     }
     

	}
	
	
	/**
	 * 
	 * @param covariance : symmetric fixed size matrix
	 */
	public static void  GetInversethis(fsmatrix covariancev) {
		
		if (covariancev==null || covariancev.GetRowDimension()==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		if (covariancev.GetRowDimension()!=covariancev.GetColumnDimension()){
			
			throw new DimensionMismatchException (covariancev.GetRowDimension(),covariancev.GetColumnDimension());
		}		
		

		
   	 int row = covariancev.GetRowDimension();
     double[] t = new double[row];
     double[] Q = new double[row];
     double[] R = new double[row];
     double AB;


     // Invert a symetric matrix in V
     for (int M = 0; M < row; M++){
         R[M] = 1;
     }
     
     int K = 0;
     for (int M = 0; M < row; M++)
     {
         double Big = 0;
         for (int s = 0; s < row; s++)
         {
             AB = Math.abs(covariancev.data[s*covariancev.GetRowDimension() + s]);
             if ((AB > Big) && (R[s] != 0))
             {
                 Big = AB;
                 K = s;
             }
         }

         
         R[K] = 0;
         Q[K] = 1 / covariancev.data[K*covariancev.GetRowDimension() + K];
         t[K] = 1;
         
         covariancev.data[K*covariancev.GetRowDimension() + K]= 0.0;
         
         if (K != 0)
         {
             for (int s = 0; s < K; s++)
             {
                 t[s] = covariancev.data[s*covariancev.GetRowDimension() + K];
                 if (R[s] == 0)
                     Q[s] = covariancev.data[s*covariancev.GetRowDimension() + K]* Q[K];
                 else
                     Q[s] = - covariancev.data[s*covariancev.GetRowDimension() + K] * Q[K];
                 covariancev.data[s*covariancev.GetRowDimension() + K]= 0.0;
             }
         }
         if ((K + 1) < row)
         {
             for (int s = K + 1; s< row; s++)
             {
                 if (R[s] != 0)
                     t[s] = covariancev.data[K *covariancev.GetRowDimension() + s];
                 else
                     t[s] = -covariancev.data[K *covariancev.GetRowDimension() + s];
                 Q[s] = -covariancev.data[K *covariancev.GetRowDimension() + s] * Q[K];
                 covariancev.data[K *covariancev.GetRowDimension() + s]= 0.0;
             }
         }
         for (int s = 0; s < row; s++)
             for (K = s; K < row; K++)
            	 covariancev.data[s*covariancev.GetRowDimension() + K]= covariancev.data[s*covariancev.GetRowDimension() + K]+ t[s] * Q[K];
     }
     
     
     
     int M = row;
     int s = row - 1;
     
     for (K = 1; K < row; K++)
     {
         M = M - 1;
         s = s - 1;
         for (int J = 0; J <= s; J++)
        	 covariancev.data[M*covariancev.GetRowDimension() + J]= covariancev.data[J*covariancev.GetRowDimension() + M];
     }
     

	}
	/**
	 * 
	 * @param data : symmetric 1 single double array representing a 2d array
	 */
	public static void  GetInversethis(double data []) {
		int dimension=(int)Math.sqrt(data.length);
		if (data==null || dimension==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}	

   	 int row = dimension;
     double[] t = new double[row];
     double[] Q = new double[row];
     double[] R = new double[row];
     double AB;


     // Invert a symetric matrix in V
     for (int M = 0; M < row; M++){
         R[M] = 1;
     }
     
     int K = 0;
     for (int M = 0; M < row; M++)
     {
         double Big = 0;
         for (int s = 0; s < row; s++)
         {
             AB = Math.abs(data[s*dimension+ s]);
             if ((AB > Big) && (R[s] != 0))
             {
                 Big = AB;
                 K = s;
             }
         }
         R[K] = 0;
         Q[K] = 1 / data[K*dimension + K];
         t[K] = 1;
         
         data[K*dimension + K]= 0.0;
         
         if (K != 0)
         {
             for (int s = 0; s < K; s++)
             {
                 t[s] = data[s*dimension + K];
                 if (R[s] == 0)
                     Q[s] = data[s*dimension + K]* Q[K];
                 else
                     Q[s] = - data[s*dimension + K] * Q[K];
                 data[s*dimension + K]= 0.0;
             }
         }
         if ((K + 1) < row)
         {
             for (int s = K + 1; s< row; s++)
             {
                 if (R[s] != 0)
                     t[s] = data[K *dimension + s];
                 else
                     t[s] = -data[K *dimension+ s];
                 Q[s] = -data[K *dimension + s] * Q[K];
                 data[K *dimension+ s]= 0.0;
             }
         }
         for (int s = 0; s < row; s++)
             for (K = s; K < row; K++)
            	 data[s*dimension + K]= data[s*dimension+ K]+ t[s] * Q[K];
     }

     int M = row;
     int s = row - 1;
     
     for (K = 1; K < row; K++)
     {
         M = M - 1;
         s = s - 1;
         for (int J = 0; J <= s; J++)
        	 data[M*dimension + J]= data[J*dimension + M];
     }
     

	}	
	
	/**
	 * 
	 * @param covariance : symmetric double matrix
	 * @param threads : Number of threads to use
	 * @return inverse matrix
	 */
	public static double [][] GetInverse(double covariance[][], int threads) {
		
		if (covariance==null || covariance.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		if (covariance.length!=covariance[0].length){
			
			throw new DimensionMismatchException (covariance.length,covariance[0].length);
		}		
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads> covariance.length) {
			threads= covariance.length;
		}	
			
	
	 double [][] covariancev=manipulate.copies.copies.Copy(covariance);
		
   	 int row = covariancev.length;
     double[] t = new double[row];
     double[] Q = new double[row];
     double[] R = new double[row];
     double AB;


     // Invert a symetric matrix in V
     for (int M = 0; M < row; M++){
         R[M] = 1;
     }
     
     int K = 0;
     for (int M = 0; M < row; M++)
     {
         double Big = 0;
         for (int s = 0; s < row; s++)
         {
             AB = Math.abs(covariancev[s] [s]);
             if ((AB > Big) && (R[s] != 0))
             {
                 Big = AB;
                 K = s;
             }
         }

         
         R[K] = 0;
         Q[K] = 1 / covariancev[K] [K];
         t[K] = 1;
         
         covariancev[K] [K]= 0.0;
         
         if (K != 0)
         {
             for (int s = 0; s < K; s++)
             {
                 t[s] = covariancev[s] [K];
                 if (R[s] == 0)
                     Q[s] = covariancev[s] [K]* Q[K];
                 else
                     Q[s] = - covariancev[s] [K] * Q[K];
                 covariancev[s] [K]= 0.0;
             }
         }
         if ((K + 1) < row)
         {
             for (int s = K + 1; s< row; s++)
             {
                 if (R[s] != 0)
                     t[s] =   covariancev[K] [s];
                 else
                     t[s] = -covariancev[K] [s];
                 Q[s] = -covariancev[K] [s] * Q[K];
                 covariancev[K] [s]= 0.0;
             }
         }
         
 		int length_of_each_threaded_pass = covariancev.length/threads;
 		int points=0;
 		
 		int locations[][] = new int[threads][2];
 		
 		for (int n=0; n <threads-1; n++ ){
 			locations[n][0]=points;
 			locations[n][1]=points + length_of_each_threaded_pass;
 			points+=length_of_each_threaded_pass;
 		}
 		locations[threads-1][0]=points;
 		locations[threads-1][1]=covariancev.length;
 		
 		Thread[] thread_array= new Thread[threads];
 		
 		for (int n=0; n <threads; n++ ){
 			
 			thread_array[n]= new Thread(new InverseRunnable(Q,t,covariancev,locations[n][0], locations[n][1]));
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
         
         
     }
    
 
     
     int M = row;
     int s = row - 1;
     
     for (K = 1; K < row; K++)
     {
         M = M - 1;
         s = s - 1;
         for (int J = 0; J <= s; J++)
        	 covariancev[M] [J]= covariancev[J] [M];
     }
     
     
 	return covariancev;
	}
	
	/**
	 * @param covariance : symmetric double matrix
	 * @param threads : Number of threads to use
	 */
	public static void  GetInversethis(double covariancev[][], int threads) {

		if (covariancev==null || covariancev.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		if (covariancev.length!=covariancev[0].length){
			
			throw new DimensionMismatchException (covariancev.length,covariancev[0].length);
		}		
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads> covariancev.length) {
			threads= covariancev.length;
		}	
				
   	 int row = covariancev.length;
     double[] t = new double[row];
     double[] Q = new double[row];
     double[] R = new double[row];
     double AB;


     // Invert a symetric matrix in V
     for (int M = 0; M < row; M++){
         R[M] = 1;
     }
     
     int K = 0;
     for (int M = 0; M < row; M++)
     {
         double Big = 0;
         for (int s = 0; s < row; s++)
         {
             AB = Math.abs(covariancev[s] [s]);
             if ((AB > Big) && (R[s] != 0))
             {
                 Big = AB;
                 K = s;
             }
         }

         
         R[K] = 0;
         Q[K] = 1 / covariancev[K] [K];
         t[K] = 1;
         
         covariancev[K] [K]= 0.0;
         
         if (K != 0)
         {
             for (int s = 0; s < K; s++)
             {
                 t[s] = covariancev[s] [K];
                 if (R[s] == 0)
                     Q[s] = covariancev[s] [K]* Q[K];
                 else
                     Q[s] = - covariancev[s] [K] * Q[K];
                 covariancev[s] [K]= 0.0;
             }
         }
         if ((K + 1) < row)
         {
             for (int s = K + 1; s< row; s++)
             {
                 if (R[s] != 0)
                     t[s] =   covariancev[K] [s];
                 else
                     t[s] = -covariancev[K] [s];
                 Q[s] = -covariancev[K] [s] * Q[K];
                 covariancev[K] [s]= 0.0;
             }
         }
  		int length_of_each_threaded_pass = covariancev.length/threads;
  		int points=0;
  		
  		int locations[][] = new int[threads][2];
  		
  		for (int n=0; n <threads-1; n++ ){
  			locations[n][0]=points;
  			locations[n][1]=points + length_of_each_threaded_pass;
  			points+=length_of_each_threaded_pass;
  		}
  		locations[threads-1][0]=points;
  		locations[threads-1][1]=covariancev.length;
  		
  		Thread[] thread_array= new Thread[threads];
  		
  		for (int n=0; n <threads; n++ ){
  			
  			thread_array[n]= new Thread(new InverseRunnable(Q,t,covariancev,locations[n][0], locations[n][1]));
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
          
          
      }

     int M = row;
     int s = row - 1;
     
     for (K = 1; K < row; K++)
     {
         M = M - 1;
         s = s - 1;
         for (int J = 0; J <= s; J++)
        	 covariancev[M] [J]= covariancev[J] [M];
     }
     

	}
	
	/**
	 * @param covariance : symmetric double matrix
	 * @param threads : Number of threads to use
	 */
	public static void  GetInversethis2(double covariancev[][], int threads) {

		if (covariancev==null || covariancev.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		if (covariancev.length!=covariancev[0].length){
			
			throw new DimensionMismatchException (covariancev.length,covariancev[0].length);
		}		
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads> covariancev.length) {
			threads= covariancev.length;
		}	
				
   	 int row = covariancev.length;
     double[] t = new double[row];
     double[] Q = new double[row];
     double[] R = new double[row];
     double AB;


     // Invert a symetric matrix in V
     for (int M = 0; M < row; M++){
         R[M] = 1;
     }
     
     int K = 0;
     for (int M = 0; M < row; M++)
     {
         double Big = 0;
         for (int s = 0; s < row; s++)
         {
             AB = Math.abs(covariancev[s] [s]);
             if ((AB > Big) && (R[s] != 0))
             {
                 Big = AB;
                 K = s;
             }
         }

         R[K] = 0;
         Q[K] = 1 / covariancev[K] [K];
         t[K] = 1;
         
         covariancev[K] [K]= 0.0;
         
         if (K != 0)
         {
             for (int s = 0; s < K; s++)
             {
                 t[s] = covariancev[s] [K];
                 if (R[s] == 0)
                     Q[s] = covariancev[s] [K]* Q[K];
                 else
                     Q[s] = - covariancev[s] [K] * Q[K];
                 covariancev[s] [K]= 0.0;
             }
         }
         if ((K + 1) < row)
         {
             for (int s = K + 1; s< row; s++)
             {
                 if (R[s] != 0)
                     t[s] =   covariancev[K] [s];
                 else
                     t[s] = -covariancev[K] [s];
                 Q[s] = -covariancev[K] [s] * Q[K];
                 covariancev[K] [s]= 0.0;
             }
         }

         InverseRunnable.q=Q;
         InverseRunnable.t=t;
         InverseRunnable.symmetric_matrix=covariancev;
  		int length_of_each_threaded_pass = covariancev.length/threads;
  		int points=0;
  		
  		int locations[][] = new int[threads][2];
  		
  		for (int n=0; n <threads-1; n++ ){
  			locations[n][0]=points;
  			locations[n][1]=points + length_of_each_threaded_pass;
  			points+=length_of_each_threaded_pass;
  		}
  		locations[threads-1][0]=points;
  		locations[threads-1][1]=covariancev.length;
  		
  		Thread[] thread_array= new Thread[threads];
  		
  		for (int n=0; n <threads; n++ ){
  			
  			thread_array[n]= new Thread(new InverseRunnable(locations[n][0], locations[n][1]));
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
          
          
      }     
     
     int M = row;
     int s = row - 1;
     
     for (K = 1; K < row; K++)
     {
         M = M - 1;
         s = s - 1;
         for (int J = 0; J <= s; J++)
        	 covariancev[M] [J]= covariancev[J] [M];
     }
     

	}	
	/**
	 * @param covariancev : Square matrix to solve
	 */
	public static void LUInversethis (double[][] covariancev){
		//some sensible checking
		
		if (covariancev==null || covariancev.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		if (covariancev.length!=covariancev[0].length){
			
			throw new DimensionMismatchException (covariancev.length,covariancev[0].length);
		}			

		// initialize objects
		double A[][]= manipulate.copies.copies.Copy(covariancev);
		int N=A.length;
		int p[]= new int[N];
		// Initialize sign of permutation.
		double signum = 1.0;

		// Initialize permutation.
		for (int i = 0; i < N; ++ i)
			{
			p[i] = i;
			}

		// Do all columns.
		for (int j = 0; j < N-1; ++ j)
			{
			// Find pivot element (maximum element) in the j-th column.
			double max = Math.abs (A[j][j]);
			int i_pivot = j;
			for (int i = j+1; i < N; ++ i)
				{
				double aij = Math.abs (A[i][j]);
				if (aij > max)
					{
					max = aij;
					i_pivot = i;
					}
				}

			// If the pivot element is not on the diagonal, interchange rows.
			if (i_pivot != j)
				{
				// Swap pivot row with diagonal row.
				double[] swap = A[i_pivot];
				A[i_pivot] = A[j];
				A[j] = swap;

				// Update permutation.
				int swap2 = p[i_pivot];
				p[i_pivot] = p[j];
				p[j] = swap2;
				signum = - signum;
				}

			// Update the decomposition.
			double ajj = A[j][j];
			if (ajj != 0.0)
				{
				for (int i = j+1; i < N; ++ i)
					{
					double aij = A[i][j] / ajj;
					A[i][j] = aij;
					for (int k = j+1; k < N; ++ k)
						{
						A[i][k] -= aij * A[j][k];
						}
					}
				}
			else // (ajj == 0.0)
				{
				throw new ConvergenceException
					("Singularity thresold exceeded");
				}
			}
		
		// for each column
		for (int i = 0; i < N; ++ i)
			{
			double[] x = new double [N];
			for (int j = 0; j < N; ++ j)
			{
			x[j] = i == p[j] ? 1.0 : 0.0;
			}
			Substitute (A, x);
			for (int j = 0; j <N; ++ j)
				{
				covariancev[j][i] = x[j];
				}
			}
        A=null;
        p=null;
		//System.gc();
		
	}	
	/**
	 * @param LU : square matrix
	 * @param x : pointer array of the permutations in a oclumn
	 */
	private static void Substitute(double[][] LU, double[] x)
	{
	// Solve Ly = b using forward substitution. 
		
	for (int i = 1; i < LU.length; ++ i)
		{
		double sum = x[i];
		for (int j = 0; j < i; ++ j)
			{
			sum -= LU[i][j] * x[j];
			}
		x[i] = sum;
		}

	// back substitution.
	x[LU.length-1] /= LU[LU.length-1][LU.length-1];
	for (int i = LU.length-2; i >= 0; -- i)
		{
		double sum = x[i];
		for (int j = i+1; j < LU.length; ++ j)
			{
			sum -= LU[i][j] * x[j];
			}
		x[i] = sum / LU[i][i];
		}
	}
	
	/**
	 * @param covariancev : Square matrix to solve
	 */
	public static void LUInversethis (double[] covariancev){
		
		if (covariancev==null || covariancev.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		// initialize objects
		double A[]= manipulate.copies.copies.Copy(covariancev);
		int N=(int) Math.sqrt(A.length);
		int p[]= new int[N];
		// Initialize sign of permutation.
		double signum = 1.0;

		// Initialize permutation.
		for (int i = 0; i < N; ++ i)
			{
			p[i] = i;
			}

		// Do all columns.
		for (int j = 0; j < N-1; ++ j)
			{
			// Find pivot element (maximum element) in the j-th column.
			double max = Math.abs (A[j*N +j]);
			int i_pivot = j;
			for (int i = j+1; i < N; ++ i)
				{
				double aij = Math.abs (A[i*N +j]);
				if (aij > max)
					{
					max = aij;
					i_pivot = i;
					}
				}

			// If the pivot element is not on the diagonal, interchange rows.
			if (i_pivot != j)
				{
				// Swap pivot row with diagonal row.
				double swap = 0.0;
				for (int k = 0; k < N; ++ k){
					swap=A[i_pivot*N + k];
					A[i_pivot*N + k]=A[j*N + k];
					A[j*N + k]=swap;
				}

				// Update permutation.
				int swap2 = p[i_pivot];
				p[i_pivot] = p[j];
				p[j] = swap2;
				signum = - signum;
				}

			// Update the decomposition.
			double ajj = A[j*N +j];
			if (ajj != 0.0)
				{
				for (int i = j+1; i < N; ++ i)
					{
					double aij = A[i*N +j] / ajj;
					A[i*N +j] = aij;
					for (int k = j+1; k < N; ++ k)
						{
						A[i*N +k] -= aij * A[j*N +k];
						}
					}
				}
			else // (ajj == 0.0)
				{
				throw new ConvergenceException
					("Singularity thresold exceeded");
				}
			}
		
		// for each column
		for (int i = 0; i < N; ++ i)
			{
			double[] x = new double [N];
			for (int j = 0; j < N; ++ j)
			{
			x[j] = i == p[j] ? 1.0 : 0.0;
			}
			Substitute (A, x,N);
			for (int j = 0; j <N; ++ j)
				{
				covariancev[j*N +i] = x[j];
				}
			}
        A=null;
        p=null;
		//System.gc();
		
	}	
	
	/**
	 * @param covariancev : Square matrix to solve
	 * @param threads : Number of threads to use
	 */
	public static void LUInversethis (double[] covariancev, int threads){
		
		if (covariancev==null || covariancev.length==0){
			
			throw new NullObjectException (" The  Array to copy from is null or empty ");
		}
		// threads' checks
		if (threads<=0) {
			threads=1;
		}
		

		// initialize objects
		double A[]= manipulate.copies.copies.Copy(covariancev);
		int N=(int) Math.sqrt(A.length);
		int p[]= new int[N];
		// Initialize sign of permutation.
		double signum = 1.0;
		if (threads> N) {
			threads= N;
		}	
			
		// Initialize permutation.
		for (int i = 0; i < N; ++ i)
			{
			p[i] = i;
			}

		// Do all columns.
		for (int j = 0; j < N-1; ++ j)
			{
			// Find pivot element (maximum element) in the j-th column.
			double max = Math.abs (A[j*N +j]);
			int i_pivot = j;
			for (int i = j+1; i < N; ++ i)
				{
				double aij = Math.abs (A[i*N +j]);
				if (aij > max)
					{
					max = aij;
					i_pivot = i;
					}
				}

			// If the pivot element is not on the diagonal, interchange rows.
			if (i_pivot != j)
				{
				// Swap pivot row with diagonal row.
				double swap = 0.0;
				for (int k = 0; k < N; ++ k){
					swap=A[i_pivot*N + k];
					A[i_pivot*N + k]=A[j*N + k];
					A[j*N + k]=swap;
				}

				// Update permutation.
				int swap2 = p[i_pivot];
				p[i_pivot] = p[j];
				p[j] = swap2;
				signum = - signum;
				}

			// Update the decomposition.
			double ajj = A[j*N +j];
			if (ajj != 0.0)
				{
				for (int i = j+1; i < N; ++ i)
					{
					double aij = A[i*N +j] / ajj;
					A[i*N +j] = aij;
					for (int k = j+1; k < N; ++ k)
						{
						A[i*N +k] -= aij * A[j*N +k];
						}
					}
				}
			else // (ajj == 0.0)
				{
				throw new ConvergenceException
					("Singularity thresold exceeded");
				}
			}

		Thread[] thread_array= new Thread[threads];
		int count_of_current_threads=0;
		
		// for each column
		for (int i = 0; i < N; ++ i)
			{
			double[] x = new double [N];
			for (int j = 0; j < N; ++ j)
			{
			x[j] = i == p[j] ? 1.0 : 0.0;
			}
			// create the thread
			thread_array[count_of_current_threads]=new Thread(new LUInverseRunnable(x,A,covariancev,N,i ));
			thread_array[count_of_current_threads].start();
			count_of_current_threads++;
			// join threads
			if (count_of_current_threads==threads || i==N-1){
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
        A=null;
        p=null;
		//System.gc();
		
	}	
	/**
	 * @param LU : square matrix
	 * @param x : pointer array of the permutations in a oclumn
	 */
	private static void Substitute(double[] LU, double[] x, int n)
	{
	// Solve Ly = b using forward substitution. 
		
	for (int i = 1; i < n; ++ i)
		{
		double sum = x[i];
		for (int j = 0; j < i; ++ j)
			{
			sum -= LU[i * n + j] * x[j];
			}
		x[i] = sum;
		}

	// back substitution.
	x[n-1] /= LU[(n-1)* n + (n-1)];
	for (int i = n-2; i >= 0; -- i)
		{
		double sum = x[i];
		for (int j = i+1; j < n; ++ j)
			{
			sum -= LU[i*n+j] * x[j];
			}
		x[i] = sum / LU[i*n + i];
		}
	}
}
