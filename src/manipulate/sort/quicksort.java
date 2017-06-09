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

public class quicksort {

	/**
	 * 
	 * @param A : double array to sort
	 * @param p : start of the loop
	 * @param q : end of the loop
	 * @param asc : true for ascending sort
	 */
	public static void Quicksort(double A [], int p, int q, boolean asc)
	{
		if (p<q)
		{
			double x = A[p];
			int i = p;
			int j;

			if (asc){
				for (j = p + 1; j<q; j++)
				{
					if (A[j] <= x)
						{
							i = i + 1;
							swap(A,i,j);
						}
				}
			}
			else {
				for (j = p + 1; j<q; j++)
				{
					if (A[j] >= x)
					{
						i = i + 1;
						swap(A,i, j);
					}
				}
			}
			swap(A,i, p);
			int r=i;
			Quicksort(A, p, r, asc);
			Quicksort(A, r + 1, q, asc);
		}
	}
	
	
	/**
	 * @param double : int array to swap elements from
	 * @param a : first index to swap
	 * @param b : second index to swap
	 * 
	 */	
	
	static void swap (double array [],int a, int b){
		double temp=array[a];
		array[a]=array[b];
		array[b]=temp; 
		
	}
	
	/**
	 * @param array : int array to swap elements from
	 * @param a : first index to swap
	 * @param b : second index to swap
	 * 
	 */
	static void swap (int array [],int a, int b){
		int temp=array[a];
		array[a]=array[b];
		array[b]=temp; 
		
	}
	
	/**
	 * 
	 * @param A : double array 1
	 * @param B : double array 2
	 * @param C : double array 3
	 * @param p : start of the loop
	 * @param q : end of the loop
	 * @param asc : true for ascending
	 */
	
	public static void Quicksort(double A [],  double [] B, double [] C, int p, int q, boolean asc){
		if (p<q)
		{
		double x = A[p];
		int i = p;
		int j;
		if (asc){
			for (j = p + 1; j < q; j++)
			{
				if (A[j] < x)
				{
					i = i + 1;
					swap(A ,i, j);
					swap(B, i, j);
					swap(C, i, j);
				}

			}
		}
		else {
			for (j = p + 1; j < q; j++)
			{
				if (A[j] > x)
				{
					i = i + 1;
					swap(A,i, j);
					swap(B,i, j);
					swap(C, i, j);
				}
			}
		}
		swap(A, i , p);
		swap(B, i , p);
		swap(C, i , p );
		
		
		int r=i;
		Quicksort(A,B,C, p, r, asc);
		Quicksort(A,B,C, r + 1, q, asc);
		}
	}
	
	/**
	 * 
	 * @param A : double array 1
	 * @param B : double array 2
	 * @param C : double array 3
	 * @param p : start of the loop
	 * @param q : end of the loop
	 */
	
	public static void Quicksortasc(double A [],  double [] B, double [] C, int p, int q){
		if (p<q)
		{
		double x = A[p];
		int i = p;
		int j;

			for (j = p + 1; j < q; j++)
			{
				if (A[j] < x)
				{
					i = i + 1;
					swap(A ,i, j);
					swap(B, i, j);
					swap(C, i, j);
				}

			}
		swap(A, i , p);
		swap(B, i , p);
		swap(C, i , p );
		
		
		int r=i;
		Quicksortasc(A,B,C, p, r);
		Quicksortasc(A,B,C, r + 1, q);
		}
	}
	/**
	 * 
	 * @param A : double array 1
	 * @param B : double array 2
	 * @param p : start of the loop
	 * @param q : end of the loop
	 * @param asc : true for ascending
	 */
	
	public static void Quicksort(double A [],  double [] B, int p, int q, boolean asc){
		if (p<q)
		{
		double x = A[p];
		int i = p;
		int j;
		if (asc){
			for (j = p + 1; j < q; j++)
			{
				if (A[j] < x)
				{
					i = i + 1;
					swap(A ,i, j);
					swap(B, i, j);
				}

			}
		}
		else {
			for (j = p + 1; j < q; j++)
			{
				if (A[j] > x)
				{
					i = i + 1;
					swap(A,i, j);
					swap(B,i, j);
				}
			}
		}
		swap(A, i , p);
		swap(B, i , p);

		
		int r=i;
		Quicksort(A,B, p, r, asc);
		Quicksort(A,B, r + 1, q, asc);
		}
	}

	/**
	 * 
	 * @param A : double array 1
	 * @param B : double array 2
	 * @param p : start of the loop
	 * @param q : end of the loop
	 */
	
	public static void Quicksortasc(double A [],  double [] B, int p, int q){
		if (p<q)
		{
		double x = A[p];
		int i = p;
		int j;
			for (j = p + 1; j < q; j++)
			{
				if (A[j] < x)
				{
					i = i + 1;
					swap(A ,i, j);
					swap(B, i, j);
				}

			}

		swap(A, i , p);
		swap(B, i , p);

		
		int r=i;
		Quicksortasc(A,B, p, r);
		Quicksortasc(A,B, r + 1, q);
		}
	}
	
	
/*	
    public static void Quicksortasc(double A [],  double [][] B, int p, int r)
    {
        if(p<r)
        {
            int q=partition(A,B,p,r);
            Quicksortasc(A,B,p,q);
            Quicksortasc(A,B,q+1,r);
        }
    }

    private static int partition(double A [],  double [][] B, int p, int r) {

        double x = A[p];
        int i = p-1 ;
        int j = r ;

        while (true) {
            i++;
            while ( i< r && A[i] < x)
                i++;
            j--;
            while (j>p && A[j] > x)
                j--;

            if (i < j){
                swap(A, i, j);
                swap(B, i, j);
            }
            else
                return j;
        }
    }
    
  */  

    public static void Quicksortasc(double numbers [], double B [][], int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (numbers[i] < pivot) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (numbers[j] > pivot) {
            j--;
          }

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, low, j);
        if (i < high)
        	Quicksortasc(numbers, B, i, high);
      }
    
    /**
     * 
     * @param numbers : Array to based on which to sort everything else
     * @param B : array to be sorted based on numbers
     * @param low : start of the loop
     * @param high : end of the loop
     */
    public static void Quicksortasc(double numbers [], int B [], int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (numbers[i] < pivot) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (numbers[j] > pivot) {
            j--;
          }

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, low, j);
        if (i < high)
        	Quicksortasc(numbers, B, i, high);
      }
    /**
     * 
     * @param numbers : Array to based on which to sort everything else
     * @param B : array to be sorted based on numbers
     * @param low : start of the loop
     * @param high : end of the loop
     */
    public static void Quicksortasc(int numbers [], int B [], int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (numbers[i] < pivot) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (numbers[j] > pivot) {
            j--;
          }

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, low, j);
        if (i < high)
        	Quicksortasc(numbers, B, i, high);
      }
    
    /**
     * 
     * @param numbers : Array to based on which to sort everything else
     * @param B : array to be sorted based on numbers
     * @param C : array to be sorted based on numbers
     * @param low : start of the loop
     * @param high : end of the loop
     */
    public static void Quicksortasc(int numbers [], int B [], int C [],int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (numbers[i] < pivot) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (numbers[j] > pivot) {
            j--;
          }
          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
              swap(C, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, C,low, j);
        if (i < high)
        	Quicksortasc(numbers, B, C, i, high);
      }
    /**
     * 
     * @param numbers : Array with double values to be sorted in an ascending way (2nd)
     * @param B : Array with int values to be sorted in an ascending way (1st)
     * @param C :  Array with int values to be sorted based on the other 2 arrays
     * @param low : start location for the sort
     * @param high  end location for the sort
     * <p> This is to be used for sorting trees' indices</p>
     */
    public static void Quicksortasc(double numbers [], int B [],  int C [],int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];
        int intpivot=  B[low + (high-low)/2];
        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (B[i] < intpivot || ( B[i] == intpivot && numbers[i] < pivot) ) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (B[j] > intpivot || ( B[j] == intpivot && numbers[j] > pivot) ) {
            j--;
          } 

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
              swap(C, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, C, low, j);
        if (i < high)
        	Quicksortasc(numbers, B,C, i, high);
      } 
  
    
    
    
    /**
     * 
     * @param numbers : Array with double values to be sorted in an ascending way (2nd)
     * @param B : Array with int values to be sorted in an ascending way (1st)
     * @param C :  Array with int values to be sorted based on the other 2 arrays
     * @param D :  Array with int values to be sorted based on the other 2 arrays
     * @param low : start location for the sort
     * @param high  end location for the sort
     * <p> This is to be used for sorting trees' indices</p>
     */
    public static void Quicksortasc(double numbers [], int B [],  int C [], int D [],int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];
        int intpivot=  B[low + (high-low)/2];
        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (B[i] < intpivot || ( B[i] == intpivot && numbers[i] < pivot) ) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (B[j] > intpivot || ( B[j] == intpivot && numbers[j] > pivot) ) {
            j--;
          } 

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
              swap(C, i, j);
              swap(D, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, C, D,low, j);
        if (i < high)
        	Quicksortasc(numbers, B,C,D, i, high);
      } 
    
    /**
     * 
     * @param numbers : Array with double values to be sorted in an ascending way (2nd)
     * @param B : Array with int values to be sorted in an ascending way (1st)
     * @param C :  Array with int values to be sorted based on the other 2 arrays
     * @param D :  Array with double values to be sorted based on the other 2 arrays
     * @param low : start location for the sort
     * @param high  end location for the sort
     * <p> This is to be used for sorting trees' indices</p>
     */
    public static void Quicksortasc(double numbers [], int B [],  int C [], double D [],int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];
        int intpivot=  B[low + (high-low)/2];
        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (B[i] < intpivot || ( B[i] == intpivot && numbers[i] < pivot) ) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (B[j] > intpivot || ( B[j] == intpivot && numbers[j] > pivot) ) {
            j--;
          } 

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
              swap(C, i, j);
              swap(D, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, C, D,low, j);
        if (i < high)
        	Quicksortasc(numbers, B,C,D, i, high);
      } 
    
    
    public static void Quicksortasc(int numbers [], int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (numbers[i] < pivot) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (numbers[j] > pivot) {
            j--;
          }

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          if (i <= j) {
        	  swap(numbers, i, j);

            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers,  low, j);
        if (i < high)
        	Quicksortasc(numbers,  i, high);
      }   
    public static void Quicksortasc(double numbers [], double B [][], double C[], int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
          // If the current value from the left list is smaller then the pivot
          // element then get the next element from the left list
          while (numbers[i] < pivot) {
            i++;
          }
          // If the current value from the right list is larger then the pivot
          // element then get the next element from the right list
          while (numbers[j] > pivot) {
            j--;
          }

          // If we have found a values in the left list which is larger then
          // the pivot element and if we have found a value in the right list
          // which is smaller then the pivot element then we exchange the
          // values.
          // As we are done we can increase i and j
          if (i <= j) {
        	  swap(numbers, i, j);
              swap(B, i, j);
              swap(C, i, j);
            i++;
            j--;
          }
        }
        // Recursion
        if (low < j)
        	Quicksortasc(numbers, B, C, low, j);
        if (i < high)
        	Quicksortasc(numbers, B, C, i, high);
      }  
    public static void Quicksortasc(double A [], double B [], double [][] C, int p, int r)
    {
        if(p<r)
        {
            int q=partition(A,B,C,p,r);
            Quicksortasc(A,B,C,p,q);
            Quicksortasc(A,B,C,q+1,r);
        }
    }

    private static int partition(double A [], double B [], double [][] C, int p, int r) {

        double x = A[p];
        int i = p-1 ;
        int j = r ;

        while (true) {
            i++;
            while ( i< r && A[i] < x)
                i++;
            j--;
            while (j>p && A[j] > x)
                j--;

            if (i < j){
                swap(A, i, j);
                swap(B, i, j);
                swap(C, i, j);
            }
            else
                return j;
        }
    }
	
	/**
	 * 
	 * @param A : double array 1
	 * @param B : double array 2
	 * @param p : start of the loop
	 * @param q : end of the loop
	 */
	
	public static void Quicksortascs(double A [],  double [][] B, int p, int q){
		
			
		if (p<q)
		{
		double x = A[p];
		int i = p;
		int j;
		int pass=0;
			for (j = p + 1; j < q; j++)
			{
				if (A[j] < x)
				{
					i = i + 1;
					swap(A ,i, j);
					swap(B, i, j);
					pass=1;
				}
				if (pass==0){
					if (A[j] !=x){
						pass=1;
					}
				}

			}
		if (pass==0){
			 //System.out.println(pass);
			 return;
		} else {
			 System.out.println(pass);
			 //System.out.println("");
		swap(A, i , p);
		swap(B, i , p);

		int r=i;
		Quicksortasc(A,B, p, r);
		Quicksortasc(A,B, r + 1, q);
		}
		}
	}
	/**
	 * 
	 * @param A : int array 1
	 * @param B : int array 2
	 * @param C : double array 3
	 * @param p : start of the loop
	 * @param q : end of the loop
	 * @param asc : true for ascending
	 * <p> useful sorting for smatrix (sparse)
	 */
	
	
	
	public static void Quicksort(int A [],  int [] B, double [] C, int p, int q, boolean asc){
		if (p<q)
		{
		double x = A[p];
		int i = p;
		int j;
		if (asc){
			for (j = p + 1; j < q; j++)
			{
				if (A[j] < x)
				{
					i = i + 1;
					swap(A ,i, j);
					swap(B, i, j);
					swap(C, i, j);
				}

			}
		}
		else {
			for (j = p + 1; j < q; j++)
			{
				if (A[j] > x)
				{
					i = i + 1;
					swap(A,i, j);
					swap(B,i, j);
					swap(C, i, j);
				}
			}
		}
		swap(A, i , p);
		swap(B, i , p);
		swap(C, i , p );
		
		
		int r=i;
		Quicksort(A,B,C, p, r, asc);
		Quicksort(A,B,C, r + 1, q, asc);
		}
	}	
	
	static void swap (double array [][],int a, int b){
		double temp []=array[a];
		array[a]=array[b];
		array[b]=temp; 
		
	}
	
	
	
// end of class	
	
}
