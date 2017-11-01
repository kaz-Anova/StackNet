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

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

/**
 * 
 * <p> This class is dedicated to sorting Arrays
 *
 */

public class JavaBasedSort {
	
	/**
	 * 
	 * @param array : The double array to be sorted
	 * <p> Sorts the same double array that is provided in an ascending manner.
	 */
	
	public static void sortdoubleasc (double array []){
		Arrays.sort(array);
	}
	
	/**
	 * 
	 * @param array : The double array to be sorted
	 * @return new double array sorted
	 * <p> Sorts the same double array that is provided in an ascending manner.
	 */
	
	public static double [] sortdoubleascre (double array []){
		double x[]= new double [array.length];
		for (int i=0; i < array.length; i++){
			x[i]=array[i];
		}
		Arrays.sort(x);
		return x;
	}
	
	
	/**
	 * 
	 * @param array : The double array to be sorted
	 * <p> Sorts the same double array that is provided in an descending manner.
	 */
	
	public static void sortdoubledesc (double array []){
		 
		Double arr [] = new Double [array.length];
		for (int i=0; i <arr.length; i++) {
			arr[i]=array[i];
		}
		Arrays.sort(arr, Collections.reverseOrder());
		for (int i=0; i <arr.length; i++) {
			array[i]=arr[i];
		}
		

	}
	
	
	
	
	/**
	 * 
	 * @param array : The double array to be sorted
	 * @param x : The  column by which we do the sorting
	 * <p> Sorts the same double 2d array that is provided in an ascending manner based on one column.
	 */
	
	public static void sort2ddoubleasc (double array [][], final int x){
		 
		Arrays.sort(array, new Comparator<double[]>() {
		    @Override
		    public int compare(double[] o1, double[] o2) {
		        return Double.compare(o1[x], o2[x]);
		    }
		});

	}

	/**
	 * 
	 * @param array : The int array to be sorted
	 * @param x : The  column by which we do the sorting
	 * <p> Sorts the same int 2d array that is provided in an descending manner based on one column.
	 */
	
	public static void sort2dintledesc (int array [][], final int x){
		 
		Arrays.sort(array, new Comparator<int[]>() {
		    @Override
		    public int compare(int[] o1, int[] o2) {
		        return Integer.compare(o2[x], o1[x]);
		    }
		});

	}
	/**
	 * 
	 * @param array : The double array to be sorted
	 * @param x : The  column by which we do the sorting
	 * <p> Sorts the same double 2d array that is provided in an descending manner based on one column.
	 */
	
	public static void sort2ddoubledesc (double array [][], final int x){
		 
		Arrays.sort(array, new Comparator<double[]>() {
		    @Override
		    public int compare(double[] o1, double[] o2) {
		        return Double.compare( o2[x],o1[x]);
		    }
		});

	}
	/**
	 * 
	 * @param one  The first based on which the sorting will be done
	 * @param two : The second array to be sorted based on the first one
	 * <p> This method sorts two arrays (which are sorted internally) in a descending manner.
	 */
	public static void sort2ddoubledesc (double one [],double two [] ){
		double three[][]=manipulate.append.append.Append(one, two);
		
		sort2ddoubledesc(three,0);
		
		for (int i=0; i < three.length; i++){
			one[i]=three[i][0];
			two[i]=three[i][1];
		}
	}
	
	/**
	 * 
	 * @param one  The first based on which the sorting will be done
	 * @param two : The second array to be sorted based on the first one
	 * <p> This method sorts two arrays (which are sorted internally) in a descending manner.
	 * @return a copy of the second array which is sorted based on the first one
	 */
	public static double [] sort2ddoubledescx (double one [],double two [] ){
		double three[][]=manipulate.append.append.Append(one, two);
		double t []= new double [one.length];
		sort2ddoubledesc(three,0);
		
		for (int i=0; i < three.length; i++){
			t[i]=three[i][1];

		}
		
		return t;
	}
	
	/**
	 * 
	 * @param one  The first based on which the sorting will be done
	 * @param two : The second array to be sorted based on the first one
	 * <p> This method sorts two arrays (which are sorted internally) in a descending manner.
	 * @return a copy of the second array which is sorted based on the first one
	 */
	public static double [] sort2ddoubleascx (double one [],double two [] ){
		double three[][]=manipulate.append.append.Append(one, two);
		double t []= new double [one.length];
		sort2ddoubleasc(three,0);
		
		for (int i=0; i < three.length; i++){
			t[i]=three[i][1];

		}
		
		return t;
	}
	
	/**
	 * 
	 * @param one  The first based on which the sorting will be done
	 * @param two : The second array to be sorted based on the first one
	 * <p> This method sorts two arrays (which are sorted internally) in a ascending manner.
	 */
	public static void sort2ddoubleasc (double one [],double two [] ){
		
		double three[][]=manipulate.append.append.Append(one, two);
		
		sort2ddoubleasc(three,0);
		
		for (int i=0; i < three.length; i++){
			one[i]=three[i][0];
			two[i]=three[i][1];
		}
	}
	
	/**
	 * 
	 * @param one  The first based on which the sorting will be done
	 * @param two : The second array to be sorted based on the first one
	 * @param three : The second array to be sorted based on the first one
	 * <p> This method sorts two arrays (which are sorted internally) in a descending manner.
	 */
	public static void sort2ddoubleasc (double one [],double two [],double three[] ){

		double four[][]= new double [one.length][3];
		
		for (int i=0; i < four.length; i++){
			four[i][0]=one[i];
			four[i][1]=two[i];
			four[i][2]=three[i];
		}
		
		 sort2ddoubleasc(four,0);
		
		for (int i=0; i < four.length; i++){
			one[i]=four[i][0];
			two[i]=four[i][1];
			three[i]=four[i][2];
		}
	}
	
	/**
	 * 
	 * @param one  The first based on which the sorting will be done
	 * @param two : The second array to be sorted based on the first one
	 * @param three : The second array to be sorted based on the first one
	 * <p> This method sorts two arrays (which are sorted internally) in a descending manner.
	 */
	
	public static void sort2ddoubleasc (double one [],double two [],double three[][] ){
		//System.out.println(one.length + " " + three.length);
		double four[][]=new double [one.length][2+three[0].length];
		for (int i=0; i <four.length; i++ ){
			four[i][0]=one[i];
			four[i][1]=two[i];
			for (int j=0; j < three[0].length; j++){
				four[i][j+2]=three[i][j];
			}
		}
		 sort2ddoubleasc(four,0);
		
		for (int i=0; i < four.length; i++){
			one[i]=four[i][0];
			two[i]=four[i][1];
			for (int j=0; j < three[0].length; j++){
			three[i][j]=four[i][2+j];
			}
		}
	}
	 /**
	 * 
	 * @param array : The double array to be sorted
	 * @param x : The  column by which we do the sorting
	 * @param y : The second column by which we do the sorting
	 * @param asc1 : When true, we sort the first column in ascending order else in descending
	 * @param asc2 : When true, we sort the second column in ascending order else in descending
	 * <p> Sorts the same double 2d array that is provided in an descending manner based on two columns.
	 */
	
	public static void sort2ddouble2cols (double array [][], final int x, final int y,final boolean asc1,  final boolean asc2){
		 
		Arrays.sort(array, new Comparator<double[]>() {
		    @Override
		    public int compare(double[] o1, double[] o2) {
		    	if (o1[x]==o2[x]){
		    		if (asc2==true) {
		    			return Double.compare( o1[y],o2[y]);
		    		} else{
		    			return Double.compare( o2[y],o1[y]);
		    		}
		    	} else{
		    		
		    		if (asc1==true) {
		    			return Double.compare( o1[x],o2[x]);
		    		} else{
		    			return Double.compare( o2[x],o1[x]);
		    		}
		    		
		    	}
		    }
		});

	}

	 /**
		 * @param array : The double array to be sorted
		 * @param x : The  column by which we do the sorting
		 * @param y : The second column by which we do the sorting
		 * @param z : The third column by which we do the sorting
		 * @param asc1 : When true, we sort the first column in ascending order else in descending
		 * @param asc2 : When true, we sort the second column in ascending order else in descending
		 * @param asc3 : When true, we sort the third column in ascending order else in descending
		 * <p> Sorts the same double 2d array that is provided in an descending manner based on three columns.
		 */
		
		public static void sort2ddouble3cols (double array [][], final int x, final int y, final int z,final boolean asc1,final boolean asc2, final boolean asc3){
			 
			Arrays.sort(array, new Comparator<double[]>() {
			    @Override
			    public int compare(double[] o1, double[] o2) {
			    	if (o1[x]==o2[x]){
			    			if (o1[y]==o2[y]) {
			    				if (asc3==true) {
			    				return Double.compare( o1[z],o2[z]);
			    				}
			    				else {
			    					return Double.compare( o2[z],o1[z]);
			    				}
			    			}
			    			else {
					    		if (asc2==true) {
					    			return Double.compare( o1[y],o2[y]);
					    		}
					    		else{
					    			return Double.compare( o2[y],o1[y]);
					    		}

			    			}		
			    	}
			    	else {
			    		if (asc1==true) {
			    			return Double.compare( o1[x],o2[x]);
			    		}
			    		else{
			    			return Double.compare( o2[x],o1[x]);
			    		}
			    	}
			    }
			});

		}

		/**
		 * 
		 * @param array : The String array to be sorted
		 * <p> Sorts the same String array that is provided in an ascending manner.
		 */
		
		public static void sortStringasc (String array []){
			Arrays.sort(array);
		}
		
		
		/**
		 * 
		 * @param array : The String array to be sorted
		 * <p> Sorts the same String array that is provided in an descending manner.
		 */
		
		public static void sortStringdesc (String array []){ 
			Arrays.sort(array, Collections.reverseOrder());	
		}
		
		
		
		/**
		 * 
		 * @param array : The String array to be sorted
		 * @param x : The  column by which we do the sorting
		 * <p> Sorts the same String 2d array that is provided in an ascending manner based on one column.
		 */
		
		public static void sort2dStringasc (String array [][], final int x){
			 
			Arrays.sort(array, new Comparator<String[]>() {
			    @Override
			    public int compare(String[] o1, String[] o2) {
			    	return o1[x].compareTo(o2[x]);
			    }
			});
}

		/**
		 * 
		 * @param array : The String array to be sorted
		 * @param x : The  column by which we do the sorting
		 * <p> Sorts the same String 2d array that is provided in an descending manner based on one column.
		 */
		
		public static void sort2dStringdesc (String array [][], final int x){
			 
			Arrays.sort(array, new Comparator<String[]>() {
			    @Override
			    public int compare(String[] o1, String[] o2) {
			    	return o2[x].compareTo(o1[x]);
			    }
			});

		}
		
		/**
		 * 
		 * @param array : The String array to be sorted
		 * @param x : The  column by which we do the sorting
		 * @param asc : True for ascending
		 * <p> Sorts the same String 2d array that is provided in an descending manner based on one column.
		 */
		
		public static void sort2dStringdescasdouble (String array [][], final int x ,boolean asc){
			  int on=1;
              int of=-1;
			 if (asc==false){
				 on=-1;
				  of=1;
			 }
			 final int onk=on;
			 final int ofk=of;
			Arrays.sort(array, new Comparator<String[]>() {
			    @Override
			    public int compare(String[] o1, String[] o2) {
			         if ( Double.parseDouble(o1[x]) -  Double.parseDouble(o2[x])>0) {        
			            	return onk;   }           
			            else  {  
			            	return ofk;   }   } 
			    
			});

		}
		
		 /**
		 * 
		 * @param array : The String array to be sorted
		 * @param x : The  column by which we do the sorting
		 * @param y : The second column by which we do the sorting
		 * @param asc1 : When true, we sort the first column in ascending order else in descending
		 * @param asc2 : When true, we sort the second column in ascending order else in descending
		 * <p> Sorts the same String 2d array that is provided in an descending manner based on two columns.
		 */
		
		public static void sort2dString2cols (String array [][], final int x, final int y,final boolean asc1,  final boolean asc2){
			 
			Arrays.sort(array, new Comparator<String[]>() {
			    @Override
			    public int compare(String[] o1, String[] o2) {
			    	if (o1[x]==o2[x]){
			    		if (asc2==true) {
			    			return  o1[y].compareTo(o2[y]);
			    		} else{
			    			return o2[y].compareTo(o1[y]);
			    		}
			    	} else{
			    		
			    		if (asc1==true) {
			    			return o1[x].compareTo(o2[x]);
			    		} else{
			    			return o2[x].compareTo(o1[x]);
			    		}
			    		
			    	}
			    }
			});

		}
		 /**
			 * @param array : The double array to be sorted
			 * @param x : The  column by which we do the sorting
			 * @param y : The second column by which we do the sorting
			 * @param z : The third column by which we do the sorting
			 * @param asc1 : When true, we sort the first column in ascending order else in descending
			 * @param asc2 : When true, we sort the second column in ascending order else in descending
			 * @param asc3 : When true, we sort the third column in ascending order else in descending
			 * <p> Sorts the same double 2d array that is provided in an descending manner based on three columns.
			 */
			
			public static void sort2dString3cols (String array [][], final int x, final int y, final int z,final boolean asc1,final boolean asc2, final boolean asc3){
				 
				Arrays.sort(array, new Comparator<String[]>() {
				    @Override
				    public int compare(String[] o1, String[] o2) {
				    	if (o1[x]==o2[x]){
				    			if (o1[y]==o2[y]) {
				    				if (asc3==true) {
				    				return o1[z].compareTo(o2[z]);
				    				}
				    				else {
				    					return o2[z].compareTo(o1[z]);
				    				}
				    			}
				    			else {
						    		if (asc2==true) {
						    			return  o1[y].compareTo(o2[y]);
						    		} else{
						    			return o2[y].compareTo(o1[y]);
						    		}

				    			}		
				    	}
				    	else {
				    		if (asc1==true) {
				    			return o1[x].compareTo(o2[x]);
				    		} else{
				    			return o2[x].compareTo(o1[x]);
				    		}
				    	}
				    }
				});

			}
	
	//End of Class
}