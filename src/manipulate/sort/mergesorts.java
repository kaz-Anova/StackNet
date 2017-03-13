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

package manipulate.sort;

public class mergesorts {

	/**
	 * 
	 * @param a : int array 1
	 * @param lo : start of the loop
	 * @param hi : end of the loop

	 */
	public static void mergesort(double [] a, int lo, int hi) {
		
	
	      if (hi <= lo) return;
	      
	        int i = lo;
	        int j = hi + 1;
	        double v = a[lo];
	        while (true) { 

	            // find item on lo to swap
	            while (compare(a[++i], v))
	                if (i == hi) break;

	            // find item on hi to swap
	            while (compare(v, a[--j]))
	                if (j == lo) break;      // redundant since a[lo] acts as sentinel

	            // check if pointers cross
	            if (i >= j) break;

	            swap(a, i, j);
	        }

	        // put partitioning item v at a[j]
	        swap(a, lo, j);
	        
	        mergesort(a, lo, j-1);
	        mergesort(a, j+1, hi);

  
		}
	
	/**
	 * 
	 * @param a : double array 1
	 * @param b : double array 2
	 * @param c : double array 3
	 * @param lo : start of the loop
	 * @param hi : end of the loop
	 */
	
	public static void mergesort(double [] a,double [] b,double [] c, int lo, int hi) {
		
		
	      if (hi <= lo) return;
	      
	        int i = lo;
	        int j = hi + 1;
	        double v = a[lo];
	        while (true) { 

	            // find item on lo to swap
	            while (compare(a[++i], v))
	                if (i == hi) break;

	            // find item on hi to swap
	            while (compare(v, a[--j]))
	                if (j == lo) break;      // redundant since a[lo] acts as sentinel

	            // check if pointers cross
	            if (i >= j) break;

	            swap(a, i, j);
	            swap(b, i, j);
	            swap(c, i, j);
	        }

	        // put partitioning item v at a[j]
	        swap(a, lo, j);
	        swap(b,  lo, j);
            swap(c, lo, j);
	        
	        mergesort(a, b ,c, lo, j-1);
	        mergesort(a,b,c, j+1, hi);


		}	
	/**
	 * 
	 * @param a : double array 1
	 * @param b : double array 2
	 * @param lo : start of the loop
	 * @param hi : end of the loop
	 */
	
	public static void mergesort(double [] a,double [] b, int lo, int hi) {
		
		
	      if (hi <= lo) return;
	      
	        int i = lo;
	        int j = hi + 1;
	        double v = a[lo];
	        while (true) { 

	            // find item on lo to swap
	            while (compare(a[++i], v))
	                if (i == hi) break;

	            // find item on hi to swap
	            while (compare(v, a[--j]))
	                if (j == lo) break;      // redundant since a[lo] acts as sentinel

	            // check if pointers cross
	            if (i >= j) break;

	            swap(a, i, j);
	            swap(b, i, j);

	        }

	        // put partitioning item v at a[j]
	        swap(a, lo, j);
	        swap(b,  lo, j);

	        
	        mergesort(a, b , lo, j-1);
	        mergesort(a,b, j+1, hi);


		}	
	/**
	 * 
	 * @param a : int array 1
	 * @param b : int array 2
	 * @param c : double array 3
	 * @param lo : start of the loop
	 * @param hi : end of the loop
	 * <p> useful sorting for smatrix (sparse)
	 */
	public static void mergesort(int [] a,int [] b,double [] c, int lo, int hi) {
		
		
	      if (hi <= lo) return;
	      
	        int i = lo;
	        int j = hi + 1;
	        double v = a[lo];
	        while (true) { 

	            while (compare(a[++i], v))
	                if (i == hi) break;
	            while (compare(v, a[--j]))
	                if (j == lo) break;
	            if (i >= j) break;

	            swap(a, i, j);
	            swap(b, i, j);
	            swap(c, i, j);
	        }

	        // put partitioning item v at a[j]
	      swap(a, lo, j);
	      swap(b,  lo, j);
          swap(c, lo, j);
	        
	        mergesort(a, b ,c, lo, j-1);
	        mergesort(a,b,c, j+1, hi);
		}		
	
	
	static void swap (double array [],int a, int b){
		double temp=array[a];
		array[a]=array[b];
		array[b]=temp; 
		
	}
	static void swap (int array [],int a, int b){
		int temp=array[a];
		array[a]=array[b];
		array[b]=temp; 
		
	}
	
	   private static boolean compare(double  v, double w) {
	        return (v<w) ;
	    }	
	
}
