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

package ml.Tree;

import matrix.fsmatrix;
import matrix.smatrix;


/**
 * <p>This class sorts the features and returns their sorted indices to be used in tree-based models</p>
 */
public class sortcolumnsnomap implements Runnable {
	/**
	 * rows to include
	 */
	
	public int [] subset_of_rows;
	/**
	 * The feature to find best split from
	 */
	public int feature=-1;
	/**
	 * The object that holds the modelling data in double form in cases the user chooses this form
	 */
	private double dataset[][];
	/**
	 * The object that holds the modelling data in fsmatrix form cases the user chooses this form
	 */
	private fsmatrix fsdataset;
	/**
	 * The object that holds the modelling data in smatrix form cases the user chooses this form
	 */
	private smatrix sdataset;	
	/**
	 * Indices to keep and sort
	 */
	private int indice_holder[][];
	/**
	 * to set initial capacity correctly
	 */
	public int row_size;
	/**
	 * Digits to round
	 */
	int rounding=30;
	/**
	 * Holds the maximum rank for each feature - very useful for bucket sort
	 */
	private int rank_holder []; 
	/**
	 * Holds the rank of the 'zero' (e.g. sparse) elements
	 */
	private int zero_rank_holder []; 

	/**
	 * 
	 * @param data : The data to Extract the column from
	 * @param rows : sub-selection of rows to use
	 * @param indice_holder : where to put the indices
	 * @param featuren : the feature to extract (e.g. find best split) from
	 * @param rankholder : array to stored the sorted indices
	 * @param capacity : row size
	 * @param round : rounding to decrease potential size
	 */
	public sortcolumnsnomap(double data [][], int [] rows, int [][]  indice_holder,
			int featuren, int rankholder [] , int capacity, int round){
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (rows==null || rows.length<=0){
			throw new IllegalStateException(" There are no rows!" );
		}
		if (indice_holder==null ){
			throw new IllegalStateException(" Problem with the state of the indice's holder" );
		}	
		if (featuren<0){
			throw new IllegalStateException("The feature cannot be negative" );	
		}
		if (indice_holder.length!=rankholder.length){
			throw new IllegalStateException(" Indice holder and rank holde need to have the same length");
		}
		this.feature=featuren;
		dataset=data;	
		subset_of_rows=rows;
		this.indice_holder=indice_holder;
		this.rank_holder=rankholder;
		this.row_size=capacity;
		this.rounding=round;

	}
	/**
	 * 
	 * @param data : The data to Extract the column from
	 * @param rows : sub-selection of rows to use
	 * @param indice_holder : where to put the indices
	 * @param featuren : the feature to extract (e.g. find best split) from
	 * @param rankholder : array to stored the sorted indices
	 * @param capacity : row size
	 * @param round : rounding to decrease potential size
	 */
 
	public sortcolumnsnomap(fsmatrix data, int []rows, int [][]  indice_holder,
			int featuren, int rankholder [] , int capacity, int round){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (rows==null || rows.length<=0){
			throw new IllegalStateException(" There are no rows!" );
		}
		
		if (indice_holder==null ){
			throw new IllegalStateException(" Problem with the state of the indice's holder" );
		}	
		
		if (featuren<0){
			throw new IllegalStateException("The feature cannot be negative" );	
		}
		if (indice_holder.length!=rankholder.length){
			throw new IllegalStateException(" Indice holder and rank holde need to have the same length");
		}
		this.feature=featuren;
		
		this.rounding=round;
		subset_of_rows=rows;
		fsdataset=data;
		this.indice_holder=indice_holder;
		this.rank_holder=rankholder;
		this.row_size=capacity;
	}

	/**
	 * 
	 * @param data : The data to Extract the column from
	 * @param rows : sub-selection of rows to use
	 * @param indice_holder : where to put the indices
	 * @param featuren : the feature to extract (e.g. find best split) from
	 * @param rankholder : array to stored the sorted indices
	 * @param capacity : row size
	 * @param round : rounding to decrease potential size
	 */
	public sortcolumnsnomap(smatrix data, int [] rows, int [][]  indice_holder,
			int featuren, int rankholder [] , int [] zero_rank_holder, int capacity, int round){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (rows==null || rows.length<=0){
			throw new IllegalStateException(" There are no rows!" );
		}
		
		if (indice_holder==null ){
			throw new IllegalStateException(" Problem with the state of the indice's holder" );
		}	
		
		if (featuren<0){
			throw new IllegalStateException("The feature cannot be negative" );	
		}
		if (indice_holder.length!=rankholder.length){
			throw new IllegalStateException(" Indice holder and rank holde need to have the same length");
		}
		this.rounding=round;
		this.feature=featuren;
		subset_of_rows=rows;
		this.indice_holder=indice_holder;
		sdataset=data;
		this.rank_holder=rankholder;
		this.zero_rank_holder=zero_rank_holder;
		this.row_size=capacity;
		}
	
	

//	/**
//	 * 
//	 * @return the betas
//	 */
//	public double [] Getbetas(){
//		if (betas==null || betas.length<=0){
//			throw new IllegalStateException(" estimator needs to be fitted first" );
//		}
//		return manipulate.copies.copies.Copy(betas);
//	}
	

	private void fit(double[][] data) {
		// make sensible checks

			//compromise memory at the gain of speed...initialise arrays at full row length
			// code block form when weights are provided
			double variable []= new double [subset_of_rows.length];
			int rows_this_fetaure[]= new int [subset_of_rows.length];
			int non_zero_counter=0;
			
			// check all value rows
			for (int i:subset_of_rows){
				variable[non_zero_counter]=Math.round(data[i][this.feature]* 10.0 * this.rounding) / (10.0 * this.rounding); 
				rows_this_fetaure[non_zero_counter]= i;
				non_zero_counter+=1; //increment counter of nonzero fetaures
			}// end of rows loop	
	
				// we need to sort this array up to no_zero_countr, that is important
			manipulate.sort.quicksort.Quicksortasc(variable, rows_this_fetaure , 0,variable.length-1);			
			int [] map= new int [this.row_size];		
			double val=Double.NEGATIVE_INFINITY;
			int rank=0;
			for (int i=0; i < variable.length; i++){
				double vs=variable[i];
				if (vs!=val){
					rank++;
					val=vs;
				}
				map[rows_this_fetaure[i]]=rank;
				//map.put(rows_this_fetaure[i], rank);
			}
			
			indice_holder[this.feature]	=map;
			this.rank_holder[this.feature]	=rank;
			rows_this_fetaure=null;
			variable=null;

	}


	private void fit(fsmatrix data) {

		//compromise memory at the gain of speed...initialise arrays at full row length
		// code block form when weights are provided
		double variable []= new double [subset_of_rows.length];
		int rows_this_fetaure[]= new int [subset_of_rows.length];
		int non_zero_counter=0;
		
		// check all value rows
		for (int i:subset_of_rows){
			variable[non_zero_counter]=Math.round(data.GetElement(i, this.feature)* 10.0 * this.rounding) / (10.0 * this.rounding); 
			rows_this_fetaure[non_zero_counter]= i;
			non_zero_counter+=1; //increment counter of nonzero fetaures
		}// end of rows loop	

			// we need to sort this array up to no_zero_countr, that is important
		manipulate.sort.quicksort.Quicksortasc(variable, rows_this_fetaure , 0,variable.length-1);
		int [] map= new int [this.row_size];
		double val=Double.NEGATIVE_INFINITY;
		int rank=0;
		for (int i=0; i < variable.length; i++){
			double vs=variable[i];
			if (vs!=val){
				rank++;
				val=vs;
			}
			map[rows_this_fetaure[i]]=rank;
			//map.put(rows_this_fetaure[i], rank);
		}
		
		indice_holder[this.feature]	=map;
		this.rank_holder[this.feature]	=rank;
		rows_this_fetaure=null;
		variable=null;

			// end of SGD

	}

	private void fit(smatrix data) {

			//compromise memory at the gain of speed...initialise arrays at full row length
			// code block form when weights are provided
			double variable_nonzero []= new double [subset_of_rows.length];
			int rows_this_fetaure []= new int [subset_of_rows.length];

			// find the sum of the nonzero elements
			int non_zero_counter=0;
			
			// check all value rows
			for (int i:subset_of_rows){
				double vals=Math.round(data.GetElement(i, this.feature)* 10.0 * this.rounding) / (10.0 * this.rounding); 
				if (vals!=0.0){
					variable_nonzero[non_zero_counter]=vals;
					rows_this_fetaure[non_zero_counter]= i;
					non_zero_counter+=1; 
				}
				//increment counter of nonzero fetaures
				/*
				for (int j=data.indexpile[i]; j < data.indexpile[i+1];j++ ){
					int check_feature=data.mainelementpile[j];
					
					if (check_feature<this.feature){ // we found our feature
						continue;// next row - here the feature has zero value
					}
					if (check_feature>this.feature){ // we found our feature
						break;// next row - here the feature has zero value
					}					
					if (check_feature==this.feature && data.valuespile[j]!=0.0){ // we found our feature
						
						variable_nonzero[non_zero_counter]=data.valuespile[j];
						rows_this_fetaure[non_zero_counter]= i;

						non_zero_counter+=1; //increment counter of nonzero fetaures
						break;//found it! no longer need to keep on looping
					} // end of check_feature if statement
					
					
				}// end of columns loop	
				*/
			
			}// end of rows loop	
			

				// we need to sort this array up to no_zero_countr, that is important
				if  (non_zero_counter>=2){
					manipulate.sort.quicksort.Quicksortasc(variable_nonzero,  rows_this_fetaure, 0,non_zero_counter-1);
				}
				
				if (non_zero_counter<=variable_nonzero.length/2) { // use sparse format
					//System.out.println(" sparse column: " + this.feature);
					
				int [] map= new int [non_zero_counter*2];
				boolean zero_is_Not_inserted=true;
				
				double val=Double.NEGATIVE_INFINITY;
				int rank=0;
				int k=0;
				for (int i=0; i < non_zero_counter;i++){
					double vs=variable_nonzero[i];
					
					if ( zero_is_Not_inserted  &&  vs>0.0) {
						rank++;
						this.zero_rank_holder[this.feature]=rank;
						zero_is_Not_inserted=false;
					}
					if (vs!=val){
						rank++;
						val=vs;
					}
					
					map[k++]=rows_this_fetaure[i];
					map[k++]=rank;
					//System.out.println(map[k-2] + " : " + map[k-1]);
					//map.put(rows_this_fetaure[i], rank);
				}

				if ( zero_is_Not_inserted ) {
					rank++;
					this.zero_rank_holder[this.feature]=rank;
					zero_is_Not_inserted=false;
				}	
				indice_holder[this.feature]	=map;
				this.rank_holder[this.feature]	=rank;
				
				} else if(non_zero_counter>variable_nonzero.length/2) {
					
					
					boolean zero_is_Not_inserted=true;
					// find the zero rank
					int rank=0;
					double val=Double.NEGATIVE_INFINITY;
					for (int i=0; i < non_zero_counter;i++){
						double vs=variable_nonzero[i];
						
						if ( zero_is_Not_inserted  &&  vs>0.0) {
							rank++;
							this.zero_rank_holder[this.feature]=rank;
							zero_is_Not_inserted=false;
							break;
						}
						if (vs!=val){
							rank++;
							val=vs;
						}
						
					}

					if ( zero_is_Not_inserted && non_zero_counter<variable_nonzero.length) {
						rank++;
						this.zero_rank_holder[this.feature]=rank;
					}	
					
					int [] map= new int [this.row_size];
					for (int i=0; i < map.length; i++){
						map[i]=this.zero_rank_holder[this.feature];
					}
					
					val=Double.NEGATIVE_INFINITY;
					rank=0;
					for (int i=0; i < variable_nonzero.length; i++){
						double vs=variable_nonzero[i];
						if (vs!=val){
							rank++;
							if (rank==this.zero_rank_holder[this.feature]){
								rank++;
							}
							val=vs;
						}
						map[rows_this_fetaure[i]]=rank;
						//map.put(rows_this_fetaure[i], rank);
					}
					
					indice_holder[this.feature]	=map;
					this.rank_holder[this.feature]	=Math.max(rank, this.zero_rank_holder[this.feature]);
					this.zero_rank_holder[this.feature]=-1;
				}
				
				else {
					throw new IllegalStateException(" Error when processing the nonzero rows of the sparse matrix");
				}
				
				

				rows_this_fetaure=null;
				variable_nonzero=null;
				
				
			// end of SGD

	}



//	public boolean isfitted() {
//		if (betas!=null || betas.length>0){
//			return true;
//		} else {
//		return false;
//		}
//	}


	@Override
	public void run() {
		// check which object was chosen to train on
		if (dataset!=null){
			this.fit(dataset);
		} else if (fsdataset!=null){
			this.fit(fsdataset);	
		} else if (sdataset!=null){
			this.fit(sdataset);	
		} else {
			throw new IllegalStateException(" No data structure specifed in the constructor" );			
		}	
	}
	


}
