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
package ml.libffm;

import java.util.Arrays;

import exceptions.IllegalStateException;
import manipulate.sort.JavaBasedSort;
import matrix.smatrix;

/**
 * 
 * Class which uses heuristics based on frequencies to determine which of columns in a sparse matrix form a "field"
 *
 */
public class Fieldfinder {
	
/**
 * 
 * @param matrix : sparse matrix to derive the fields
 * @return an array with size the column size of the sparse matrix and an indicator pointing to the field each column belongs to
 */
	public static int [] get_fileds (smatrix matrix){
		
		if (matrix==null || matrix.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to create fields from" );
		}
		
		int column_fields []= new int [matrix.GetColumnDimension()];
		int freq[]= new int [matrix.GetColumnDimension()];
		
		for (int i=0; i < matrix.mainelementpile.length;i++){
			freq[matrix.mainelementpile[i]]+=1;
		}
		
		int dual [][] = new int [matrix.GetColumnDimension()][2];
		for (int i=0; i < matrix.GetColumnDimension();i++){
			dual[i][0]=i;
			dual[i][1]=freq[i];
		}		
		JavaBasedSort.sort2dintledesc(dual, 1);
		double row_dim =matrix.GetRowDimension();
		int current_count=0;
		int field_counter=0;
		for (int i=0; i <dual.length; i++ ){
			int column=dual[i][0];
			double count=dual[i][1] + 0.0;
			double odds=count/row_dim;
			column_fields[column]=field_counter;
			if (odds>0.5){
				field_counter+=1;
			} else {
				if (odds>0.3){
					current_count+=1;
					if (current_count<3){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}
				} else if (odds>0.2){
					current_count+=1;
					if (current_count<5){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}											
				} else if (odds>0.1){
					current_count+=1;
					if (current_count<10){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}					
				} else if (odds>0.05){
					current_count+=1;
					if (current_count<20){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}						
				} else if (odds>0.025){
					current_count+=1;
					if (current_count<40){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}						
				} else if (odds>0.01){
					current_count+=1;
					if (current_count<100){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}
				} else if (odds>0.005){
					current_count+=1;
					if (current_count<200){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}					
				} else if (odds>0.001){
					current_count+=1;
					if (current_count<1000){
						continue;
					} else {
						current_count=0;
						field_counter+=1;
					}					
				
			} else {
				current_count+=1;	
				if (current_count<5000){
					continue;
				} else {
					current_count=0;
					field_counter+=1;
				}				
				
			}
			}
			
		}
		
		
		return column_fields;
		
		
	}
	
	
	/**
	 * 
	 * @param matrix : sparse matrix to derive the fields
	 * @return an array with size the column size of the sparse matrix and an indicator pointing to the field each column belongs to
	 */
		public static int [] get_fileds_noorder (smatrix matrix){
			
			if (matrix==null || matrix.GetRowDimension()<=0){
				throw new IllegalStateException(" There is nothing to create fields from" );
			}
			
			int column_fields []= new int [matrix.GetColumnDimension()];
			int freq[]= new int [matrix.GetColumnDimension()];
			
			for (int i=0; i < matrix.mainelementpile.length;i++){
				freq[matrix.mainelementpile[i]]+=1;
			}
			
			int dual [][] = new int [matrix.GetColumnDimension()][2];
			for (int i=0; i <matrix.GetColumnDimension();i++){
				dual[i][0]=i;
				dual[i][1]=freq[i];
			}		
			
			double row_dim =matrix.GetRowDimension();
			double current_per=0;
			int field_counter=0;
			for (int i=0; i <dual.length; i++ ){
				int column=dual[i][0];
				double count=dual[i][1] + 0.0;
				double odds=count/row_dim;
				current_per+=odds;
				if (current_per<=1.01 ){
					column_fields[column]=field_counter;
				} else {
						field_counter+=1;
						column_fields[column]=field_counter;
						current_per=odds;	
					}

				
				}
				
			
			return column_fields;
			
			
		}
			
	/**
	 * 
	 * @param range : a string connoting a range of indices in the form of '1,4,7,123,546'. This would mean that the 0 column is a field on its own, {1,2,3} form another field, {4,5,6} another. {7,8...122} form another field and so on.  
	 * @param columm_size : column size 
	 * @return an array with size the given column size and an indicator pointing to the field each column belongs to.
	 */
		public static int [] get_fileds (String range, int columm_size){
			
			if (columm_size<=0 ){
				throw new IllegalStateException("column size cannot be <=0" );
			}			

		    	String splits [] = range.split(",");
		    	int column_fields []= new int [columm_size];
		    	int filed_points [] = new int [splits.length+2];
		    	int size=0;
		    			
		    	for (String ele: splits){
		    		try{
		    			int this_column_index=Integer.parseInt(ele);
		    			if (this_column_index<=0 || this_column_index>=columm_size){
		    				continue;
		    			}
		    			filed_points[size]=this_column_index;
		    			size+=1	;
		    		}catch (Exception e){
		    			throw new IllegalStateException(" range needs to have comma separated integer indices .Here it receied: " + ele  );	
		    		}
		    	}
		    	int new_filed_points [] = new int [size+2];
		    	new_filed_points[0]=0;
		    	new_filed_points[new_filed_points.length-1]=columm_size;
		    	for (int i=0; i <size; i++){
		    		new_filed_points[i+1]=filed_points[i];
		    	}
		    	Arrays.sort(new_filed_points);
		    	for (int i=0; i < new_filed_points.length-1; i++){
		    		for (int j=new_filed_points[i]; j <new_filed_points[i+1];j++ ){
		    			column_fields[j]=i;
		    		}
		    	}
			
			return column_fields;
			
			
			
		}

}
