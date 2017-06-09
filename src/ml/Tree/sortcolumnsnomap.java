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
	 * The object that holds the modelling data in smatrix form 	 */
	private smatrix sdataset;	
	
	/**
	 * The object that holds the modelling data in fsmatrix form 
	 */
	private fsmatrix fsdataset;	
	/**
	 * Indices to keep and sort
	 */
	private int indice_holder[][];
	/**
	 * to set initial capacity correctly
	 */
	public int rank_holder;
	/**
	 * Digits to round
	 */
	double rounding=30;
	/**
	 * Holds the rank of the 'zero' (e.g. sparse) elements
	 */
	private int zero_rank_holder []; 
	/**
	 * int target values
	 */
	public int target_vales []=null;
	/**
	 * fixes size target values
	 */
	public fsmatrix fstarget=null;
	/**
	 * target proportion change to require a 
	 */
	public double merge_thresold=0.0;
	/**
	 * whether to print stuff or not
	 */
	public boolean verbose=false;
	/**
	 * 
	 * @param data : The data to Extract the column from
	 * @param rows : sub-selection of rows to use
	 * @param indice_holder : where to put the indices
	 * @param featuren : the feature to extract (e.g. find best split) from
	 * @param rankholder : array to stored the sorted indices
	 * @param round : rounding to decrease potential size
	 */
	public sortcolumnsnomap(smatrix data, int [] rows, int [][]  indice_holder
			, int [] zero_rank_holder, double round){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (rows==null || rows.length<=0){
			subset_of_rows= new int [data.GetRowDimension()];
			for (int i=0; i < subset_of_rows.length;i++){
				subset_of_rows[i]=i;
			}
		} else {
			subset_of_rows=rows;
		}
		
		if (indice_holder==null ){
			throw new IllegalStateException(" Problem with the state of the indice's holder" );
		}	

		this.rounding=round;
	
		this.indice_holder=indice_holder;
		sdataset=data;
		this.zero_rank_holder=zero_rank_holder;
		}
	/**
	 * 
	 * @param data : The data to Extract the column from
	 * @param rows : sub-selection of rows to use
	 * @param indice_holder : where to put the indices
	 * @param featuren : the feature to extract (e.g. find best split) from
	 * @param rankholder : array to stored the sorted indices
	 * @param round : rounding to decrease potential size
	 */
	public sortcolumnsnomap(fsmatrix data, int [] rows, int [][]  indice_holder
			, int [] zero_rank_holder, double round){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (rows==null || rows.length<=0){
			subset_of_rows= new int [data.GetRowDimension()];
			for (int i=0; i < subset_of_rows.length;i++){
				subset_of_rows[i]=i;
			}
		} else {
			subset_of_rows=rows;
		}
		
		if (indice_holder==null ){
			throw new IllegalStateException(" Problem with the state of the indice's holder" );
		}	

		this.rounding=round;
	
		this.indice_holder=indice_holder;
		fsdataset=data;
		this.zero_rank_holder=zero_rank_holder;
		}
	/**
	 * @return maximum rank plus 1
	 */
	public int getmaxrank(){
		return this.rank_holder;
	}
	/**
	 * 
	 * @param data create rank indices for the feature set
	 */
	private void fit(smatrix data) {
	
			//compromise memory at the gain of speed...initialise arrays at full row length
			// code block form when weights are provided
		
		for (int j=0; j <data.GetColumnDimension(); j++){
			this.zero_rank_holder[j]=-1;
		}

			//estimate potential initial length
			int startinglength=0;
			for (int i:subset_of_rows){
				startinglength+=data.indexpile[i+1]-data.indexpile[i];


			}
			// create new elements and prepare to sort
			
			int elements[]=new int[startinglength];
			int columns[]=new int[startinglength];
			double values[]=new double[startinglength];
			if (target_vales==null && this.fstarget==null){
			
				int non_zero=0;
				for (int i:subset_of_rows){
					for (int J=data.indexpile[i]; J<data.indexpile[i+1]; J++){

								values[ non_zero ]=Math.round( data.valuespile[J]* 10.0 * rounding) / (10.0 * rounding);
								columns[ non_zero ]=data.mainelementpile[J];				
								elements[ non_zero ]=J;
								non_zero++;			
					}
				}
				
				//Quicksortasc(double numbers [], int B [],  int C [],int low, int high);
				manipulate.sort.quicksort.Quicksortasc(values,  columns,elements, 0,non_zero-1);
				
				int [] holder_of_elements_to_ranks=new int [data.GeLength()];
				int [] holder_of_ranks_elements=new int [non_zero +data.GetColumnDimension()];
				int [] holder_of_rank_to_columns=new int [non_zero+data.GetColumnDimension()];
				
				
				int rank=-1;
				int column_indice=-1;
				double previous_value=Double.NEGATIVE_INFINITY;
				
				for (int i=0; i <non_zero ; i++){
					
					int col=columns[i];
					double value=values[i];
					
					//if same column as before
					
					if (col==column_indice){
						
						// we check if the value is the same as before
						
						if (value==previous_value){
							holder_of_elements_to_ranks[elements[i] ]=rank;
							//holder_of_elements_to_columns[elements[i]]=columns[i];
							// if value is larger
						} else if (value>previous_value){
							previous_value=value;
							// we check id larger than zero and whether zero has been inserted
							
							if (value>0.0 && zero_rank_holder[col]==-1){
								
								//we add the zero value
								rank++;
								zero_rank_holder[col]=rank;
								holder_of_rank_to_columns[rank]=col;
								holder_of_ranks_elements[rank]=-1;
							}
							rank++;
							holder_of_elements_to_ranks[elements[i]]=rank;
							//holder_of_elements_to_columns[elements[i]]=columns[i];	
							holder_of_rank_to_columns[rank]=col;
							holder_of_ranks_elements[rank]=elements[i];
						} else {
							throw new IllegalStateException(" Error in sorting the idices for tree-based methods. ");
						}
						
						// if the new column is higher than the previous one
						
					} else if (col>column_indice){
						
						column_indice=col;
						
						if (value>0.0 && zero_rank_holder[col]==-1){
							//we add the zero value
							rank++;
							zero_rank_holder[col]=rank;
							holder_of_rank_to_columns[rank]=col;
							holder_of_ranks_elements[rank]=-1;
						}
						previous_value=value;
						rank++;
						holder_of_elements_to_ranks[elements[i]]=rank;
						//holder_of_elements_to_columns[elements[i]]=columns[i];	
						holder_of_rank_to_columns[rank]=col;
						holder_of_ranks_elements[rank]=elements[i];
						
						
					} else {
						throw new IllegalStateException(" Error in sorting the columns for tree-based methods. ");
					}
	
					//this.zero_rank_holder[this.feature]=rank;
	
				}
				
				// check if last imported variable has a zero rank if not, we add it.
				if (zero_rank_holder[column_indice]==-1){
					rank++;
					zero_rank_holder[column_indice]=rank;
					holder_of_rank_to_columns[rank]=column_indice;
					holder_of_ranks_elements[rank]=-1;
				}
				//set maximum rank
				this.rank_holder =rank+1;
				int holder_of_ranks_elements_v2 []= new int[this.rank_holder];
				int holder_of_rank_to_columns_v2 []= new int[this.rank_holder];
				for (int i=0; i < holder_of_ranks_elements_v2.length; i++){
					holder_of_ranks_elements_v2[i]=holder_of_ranks_elements[i];
					holder_of_rank_to_columns_v2[i]=holder_of_rank_to_columns[i];			
				}
				if (this.verbose){
					System.out.println(" Total ranks : " + this.rank_holder );
					System.out.println(" percentage of unique ranks versus elements size: " + ((double)this.rank_holder/(double)data.GeLength())*100 +"%") ;
				}
				
				indice_holder[0]=holder_of_elements_to_ranks;
				//indice_holder[1]=holder_of_elements_to_columns;	
				indice_holder[1]=holder_of_ranks_elements_v2;
				indice_holder[2]=holder_of_rank_to_columns_v2;			
				//System.out.println("");
				holder_of_ranks_elements=null;
				holder_of_rank_to_columns=null;
				elements=null;
				columns=null;
				values=null;
			
			
			} else{
				int non_zero=0;				
				double target_values[]=new double[startinglength];
				
					if (this.target_vales!=null){
					for (int i:subset_of_rows){
						
						for (int J=data.indexpile[i]; J<data.indexpile[i+1]; J++){
							/*
								if ( data.valuespile[J]==0.0){
									System.out.println("zero");
								}
								*/
									values[ non_zero ]=Math.round( data.valuespile[J]* 10.0 * rounding) / (10.0 * rounding);
									columns[ non_zero ]=data.mainelementpile[J];				
									elements[ non_zero ]=J;
									target_values[ non_zero ]=target_vales[i];
									non_zero++;			
						}
					}
				} else {

					for (int i:subset_of_rows){
						
						for (int J=data.indexpile[i]; J<data.indexpile[i+1]; J++){
									values[ non_zero ]=Math.round( data.valuespile[J]* 10.0 * rounding) / (10.0 * rounding);
									columns[ non_zero ]=data.mainelementpile[J];				
									elements[ non_zero ]=J;
									for (int j=0; j <this.fstarget.GetColumnDimension();j++){
									target_values[ non_zero ]+=fstarget.GetElement(i, j);
									}
									non_zero++;			
						}
					}
				}
					
				//Quicksortasc(double numbers [], int B [],  int C [],int low, int high);
				manipulate.sort.quicksort.Quicksortasc(values,  columns, elements,target_values, 0,non_zero-1);
				
				int [] holder_of_ranks_counts=new int [non_zero +data.GetColumnDimension()];
				double [] holder_of_rank_to_sums=new double [non_zero+data.GetColumnDimension()];				
				int temp_zero_holder[]= new int [data.GetColumnDimension()];
				int rank=-1;
				int column_indice=-1;
				double previous_value=Double.NEGATIVE_INFINITY;
				
				for (int i=0; i <non_zero ; i++){
					
					int col=columns[i];
					double value=values[i];
					
					//if same column as before
					
					if (col==column_indice){
						
						// we check if the value is the same as before
						if (value==previous_value){
							holder_of_ranks_counts[rank]+=1;
							holder_of_rank_to_sums[rank]+=target_values[ i ];	
							//holder_of_elements_to_columns[elements[i]]=columns[i];
							// if value is larger
						} else if (value>previous_value){
							
							previous_value=value;
							// we check id larger than zero and whether zero has been inserted
							
							if (value>0.0 && temp_zero_holder[col]!=1){
								
								//we add the zero value
								rank++;
								temp_zero_holder[col]=1;
							}
							rank++;
							// to counts and target
							holder_of_ranks_counts[rank]+=1;
							holder_of_rank_to_sums[rank]+=target_values[ i ];							

						} else {
							throw new IllegalStateException(" Error in sorting the idices for tree-based methods. ");
						}
						
						// if the new column is higher than the previous one
						
					} else if (col>column_indice){
						
						column_indice=col;
						
						if (value>0.0 && temp_zero_holder[col]!=1){
							//we add the zero value
							rank++;
							temp_zero_holder[col]=1;
						}
						
						previous_value=value;
						rank++;
						holder_of_ranks_counts[rank]+=1;
						holder_of_rank_to_sums[rank]+=target_values[ i ];		
						
						
						
					} else {
						throw new IllegalStateException(" Error in sorting the columns for tree-based methods. ");
					}
	
					//this.zero_rank_holder[this.feature]=rank;
	
				}
				
				// check if last imported variable has a zero rank if not, we add it.
				if (temp_zero_holder[column_indice]!=1){
					rank++;
					temp_zero_holder[column_indice]=1;
				}
				//set maximum rank
				int old_ranks=rank+1;
				if (this.verbose){
					System.out.println(" Total ranks before target inclusion: " + old_ranks );
					System.out.println(" percentage of unique ranks versus elements size: " + ((double)old_ranks/(double)data.GeLength())*100 +"%") ;
				}
				temp_zero_holder=null;
				
				// start again, this time with focus on merging similar ranks (based on target)
				
				int [] holder_of_elements_to_ranks=new int [data.GeLength()];
				int [] holder_of_ranks_elements=new int [old_ranks];
				int [] holder_of_rank_to_columns=new int [old_ranks];
				boolean last_was_zero=false;
				rank=-1;
				int newrank=-1;
				column_indice=-1;
				previous_value=Double.NEGATIVE_INFINITY;
				
				for (int i=0; i <non_zero ; i++){
					
					int col=columns[i];
					double value=values[i];
					
					//if same column as before
					if (col==column_indice){
						
						// we check if the value is the same as before	
						if (value==previous_value){
							holder_of_elements_to_ranks[elements[i] ]=newrank;
							
							//holder_of_elements_to_columns[elements[i]]=columns[i];
							// if value is larger
						} else if (value>previous_value){
							previous_value=value;
							
							// we check id larger than zero and whether zero has been inserted
							if (value>0.0 && zero_rank_holder[col]==-1){
								
								//we add the zero value
								newrank++;
								rank++;
								zero_rank_holder[col]=newrank;
								holder_of_rank_to_columns[newrank]=col;
								holder_of_ranks_elements[newrank]=-1;
								last_was_zero=true;
							}
							
							if (last_was_zero==false){// && holder_of_ranks_counts[rank]>2 && holder_of_ranks_counts[rank+1]>2 ){
								double current_rank=holder_of_rank_to_sums[rank]/(double)holder_of_ranks_counts[rank];
								double current_rank_tobe=holder_of_rank_to_sums[rank+1]/(double)holder_of_ranks_counts[rank+1];
								double diff=Math.abs(current_rank_tobe-current_rank);
								
								if (diff>this.merge_thresold ){
									newrank++;
								}
								
							} else {
								newrank++;
							}
						
							rank++;
							holder_of_elements_to_ranks[elements[i]]=newrank;
							//holder_of_elements_to_columns[elements[i]]=columns[i];	
							holder_of_rank_to_columns[newrank]=col;
							holder_of_ranks_elements[newrank]=elements[i];
							last_was_zero=false;
						} else {
							throw new IllegalStateException(" Error in sorting the idices for tree-based methods. ");
						}
						
						// if the new column is higher than the previous one
						
					} else if (col>column_indice){
						
						column_indice=col;
						
						if (value>0.0 && zero_rank_holder[col]==-1){
							//we add the zero value
							newrank++;
							rank++;
							zero_rank_holder[col]=newrank;
							holder_of_rank_to_columns[newrank]=col;
							holder_of_ranks_elements[newrank]=-1;
							last_was_zero=true;
						}
						previous_value=value;
						newrank++;
						rank++;
						holder_of_elements_to_ranks[elements[i]]=newrank;
						//holder_of_elements_to_columns[elements[i]]=columns[i];	
						holder_of_rank_to_columns[newrank]=col;
						holder_of_ranks_elements[newrank]=elements[i];
						last_was_zero=false;
						
						
					} else {
						throw new IllegalStateException(" Error in sorting the columns for tree-based methods. ");
					}
	
					//this.zero_rank_holder[this.feature]=rank;
	
				}
				
				// check if last imported variable has a zero rank if not, we add it.
				if (zero_rank_holder[column_indice]==-1){
					newrank++;
					rank++;
					zero_rank_holder[column_indice]=newrank;
					holder_of_rank_to_columns[newrank]=column_indice;
					holder_of_ranks_elements[newrank]=-1;
				}
				//set maximum rank
				this.rank_holder =newrank+1;
				int holder_of_ranks_elements_v2 []= new int[this.rank_holder];
				int holder_of_rank_to_columns_v2 []= new int[this.rank_holder];
				for (int i=0; i < holder_of_ranks_elements_v2.length; i++){
					holder_of_ranks_elements_v2[i]=holder_of_ranks_elements[i];
					holder_of_rank_to_columns_v2[i]=holder_of_rank_to_columns[i];			
				}
				
				if (this.verbose){
					System.out.println(" Total ranks after target inclusion: " + this.rank_holder );
					System.out.println(" Gain from before : " + (old_ranks-this.rank_holder) );
					System.out.println(" percentage of unique ranks versus elements size: " + ((double)this.rank_holder/(double)data.GeLength())*100 + "%" );
					System.out.println(" Gain of percentage of unique ranks versus elements size: " + ((double)old_ranks/(double)data.GeLength() -(double)this.rank_holder/(double)data.GeLength())*100 + "%"  );
				}
				
				indice_holder[0]=holder_of_elements_to_ranks;
				//indice_holder[1]=holder_of_elements_to_columns;	
				indice_holder[1]=holder_of_ranks_elements_v2;
				indice_holder[2]=holder_of_rank_to_columns_v2;			
				//System.out.println("");
				holder_of_ranks_elements=null;
				holder_of_rank_to_columns=null;
				holder_of_rank_to_columns=null;
				holder_of_rank_to_sums=null;
				elements=null;
				columns=null;
				values=null;
			

			}
			
			
			System.gc();
			
			// end of SGD

	}

	/**
	 * 
	 * @param data create rank indices for the feature set in desne format
	 */
	private void fit(fsmatrix data) {
	
			//compromise memory at the gain of speed...initialise arrays at full row length
			// code block form when weights are provided
		zero_rank_holder=null;

			//estimate potential initial length
			int startinglength=subset_of_rows.length*data.GetColumnDimension();

			// create new elements and prepare to sort
			
			int elements[]=new int[startinglength];
			int columns[]=new int[startinglength];
			double values[]=new double[startinglength];
			if (target_vales==null && this.fstarget==null){
			
				int non_zero=0;
				for (int i:subset_of_rows){
					
					for (int j=0; j < data.GetColumnDimension();j++){
								values[ non_zero ]=Math.round( data.GetElement(i, j) * 10.0 * rounding) / (10.0 * rounding);
								columns[ non_zero ]= j;				
								elements[ non_zero ]=i*data.GetColumnDimension() + j;
								non_zero++;			
					}
				}
				//Quicksortasc(double numbers [], int B [],  int C [],int low, int high);
				manipulate.sort.quicksort.Quicksortasc(values,  columns,elements, 0,non_zero-1);
				
				int [] holder_of_elements_to_ranks=new int [startinglength];
				int [] holder_of_ranks_elements=new int [startinglength ];

				int rank=-1;
				int column_indice=-1;
				double previous_value=Double.NEGATIVE_INFINITY;
				
				for (int i=0; i <non_zero ; i++){
					
					int col=columns[i];
					double value=values[i];
					
					//if same column as before
					
					if (col==column_indice){
						
						// we check if the value is the same as before
						
						if (value==previous_value){
							holder_of_elements_to_ranks[elements[i] ]=rank;
							//holder_of_elements_to_columns[elements[i]]=columns[i];
							// if value is larger
						} else if (value>previous_value){
							previous_value=value;
							// we check id larger than zero and whether zero has been inserted
							
							rank++;
							holder_of_elements_to_ranks[elements[i]]=rank;
							//holder_of_elements_to_columns[elements[i]]=columns[i];	
							holder_of_ranks_elements[rank]=elements[i];
						} else {
							throw new IllegalStateException(" Error in sorting the idices for tree-based methods. ");
						}
						
						// if the new column is higher than the previous one
						
					} else if (col>column_indice){
						
						column_indice=col;
						

						previous_value=value;
						rank++;
						holder_of_elements_to_ranks[elements[i]]=rank;
						//holder_of_elements_to_columns[elements[i]]=columns[i];	
						holder_of_ranks_elements[rank]=elements[i];
						
						
					} else {
						throw new IllegalStateException(" Error in sorting the columns for tree-based methods. ");
					}
	
					//this.zero_rank_holder[this.feature]=rank;
	
				}
				
				// check if last imported variable has a zero rank if not, we add it.

				//set maximum rank
				this.rank_holder =rank+1;
				int holder_of_ranks_elements_v2 []= new int[this.rank_holder];
				for (int i=0; i < holder_of_ranks_elements_v2.length; i++){
					holder_of_ranks_elements_v2[i]=holder_of_ranks_elements[i];			
				}
				if (this.verbose){
					System.out.println(" Total ranks : " + this.rank_holder );
					System.out.println(" percentage of unique ranks versus elements size: " + ((double)this.rank_holder/(double)startinglength)*100 +"%") ;
				}
				
				indice_holder[0]=holder_of_elements_to_ranks;
				//indice_holder[1]=holder_of_elements_to_columns;	
				indice_holder[1]=holder_of_ranks_elements_v2;		
				//System.out.println("");
				holder_of_ranks_elements=null;
				elements=null;
				columns=null;
				values=null;
			
			
			} else{
				int non_zero=0;				
				double target_values[]=new double[startinglength];
				
					if (this.target_vales!=null){
						
					for (int i:subset_of_rows){
						
						for (int j=0; j < data.GetColumnDimension();j++){
									values[ non_zero ]=Math.round( data.GetElement(i, j)* 10.0 * rounding) / (10.0 * rounding);
									columns[ non_zero ]=j;				
									elements[ non_zero ]=i*data.GetColumnDimension() + j;
									target_values[ non_zero ]=target_vales[i];
									non_zero++;			
						}
					}
				} else {

					for (int i:subset_of_rows){
						
						for (int j=0; j < data.GetColumnDimension();j++){
									values[ non_zero ]=Math.round( data.GetElement(i, j)* 10.0 * rounding) / (10.0 * rounding);
									columns[ non_zero ]=j;				
									elements[ non_zero ]=i*data.GetColumnDimension() + j;
									for (int s=0; s <this.fstarget.GetColumnDimension();s++){
									target_values[ non_zero ]+=fstarget.GetElement(i, s);
									}
									non_zero++;			
						}
					}
				}
					
				//Quicksortasc(double numbers [], int B [],  int C [],int low, int high);
				manipulate.sort.quicksort.Quicksortasc(values,  columns, elements,target_values, 0,non_zero-1);
				
				int [] holder_of_ranks_counts=new int [non_zero];
				double [] holder_of_rank_to_sums=new double [non_zero];				
				int rank=-1;
				int column_indice=-1;
				double previous_value=Double.NEGATIVE_INFINITY;
				
				for (int i=0; i <non_zero ; i++){
					
					int col=columns[i];
					double value=values[i];
					
					//if same column as before
					
					if (col==column_indice){
						
						// we check if the value is the same as before
						if (value==previous_value){
							holder_of_ranks_counts[rank]+=1;
							holder_of_rank_to_sums[rank]+=target_values[ i ];	
							//holder_of_elements_to_columns[elements[i]]=columns[i];
							// if value is larger
						} else if (value>previous_value){
							
							previous_value=value;
							// we check id larger than zero and whether zero has been inserted
							

							rank++;
							// to counts and target
							holder_of_ranks_counts[rank]+=1;
							holder_of_rank_to_sums[rank]+=target_values[ i ];							

						} else {
							throw new IllegalStateException(" Error in sorting the idices for tree-based methods. ");
						}
						
						// if the new column is higher than the previous one
						
					} else if (col>column_indice){
						
						column_indice=col;
						
						
						previous_value=value;
						rank++;
						holder_of_ranks_counts[rank]+=1;
						holder_of_rank_to_sums[rank]+=target_values[ i ];		
						
						
						
					} else {
						throw new IllegalStateException(" Error in sorting the columns for tree-based methods. ");
					}
	
					//this.zero_rank_holder[this.feature]=rank;
	
				}
				
				//set maximum rank
				int old_ranks=rank+1;
				if (this.verbose){
					System.out.println(" Total ranks before target inclusion: " + old_ranks );
					System.out.println(" percentage of unique ranks versus elements size: " + ((double)old_ranks/(double)startinglength)*100 +"%") ;
				}
				
				// start again, this time with focus on merging similar ranks (based on target)
				
				int [] holder_of_elements_to_ranks=new int [startinglength];
				int [] holder_of_ranks_elements=new int [old_ranks];
				rank=-1;
				int newrank=-1;
				column_indice=-1;
				previous_value=Double.NEGATIVE_INFINITY;
				
				for (int i=0; i <non_zero ; i++){
					
					int col=columns[i];
					double value=values[i];
					
					//if same column as before
					if (col==column_indice){
						
						// we check if the value is the same as before	
						if (value==previous_value){
							holder_of_elements_to_ranks[elements[i] ]=newrank;
							
							//holder_of_elements_to_columns[elements[i]]=columns[i];
							// if value is larger
						} else if (value>previous_value){
							previous_value=value;
							
							// we check id larger than zero and whether zero has been inserted
							
							double current_rank=holder_of_rank_to_sums[rank]/(double)holder_of_ranks_counts[rank];
							double current_rank_tobe=holder_of_rank_to_sums[rank+1]/(double)holder_of_ranks_counts[rank+1];
							double diff=Math.abs(current_rank_tobe-current_rank);
							
							if (diff>this.merge_thresold ){
								newrank++;
							}
							rank++;
							holder_of_elements_to_ranks[elements[i]]=newrank;
							//holder_of_elements_to_columns[elements[i]]=columns[i];	
							holder_of_ranks_elements[newrank]=elements[i];
						} else {
							throw new IllegalStateException(" Error in sorting the idices for tree-based methods. ");
						}
						
						// if the new column is higher than the previous one
						
					} else if (col>column_indice){
						
						column_indice=col;
						
						previous_value=value;
						newrank++;
						rank++;
						holder_of_elements_to_ranks[elements[i]]=newrank;
						//holder_of_elements_to_columns[elements[i]]=columns[i];	
						holder_of_ranks_elements[newrank]=elements[i];
						
						
					} else {
						throw new IllegalStateException(" Error in sorting the columns for tree-based methods. ");
					}
	
					//this.zero_rank_holder[this.feature]=rank;
	
				}
				

				//set maximum rank
				this.rank_holder =newrank+1;
				int holder_of_ranks_elements_v2 []= new int[this.rank_holder];
				for (int i=0; i < holder_of_ranks_elements_v2.length; i++){
					holder_of_ranks_elements_v2[i]=holder_of_ranks_elements[i];		
				}
				
				if (this.verbose){
					System.out.println(" Total ranks after target inclusion: " + this.rank_holder );
					System.out.println(" Gain from before : " + (old_ranks-this.rank_holder) );
					System.out.println(" percentage of unique ranks versus elements size: " + ((double)this.rank_holder/(double)startinglength)*100 + "%" );
					System.out.println(" Gain of percentage of unique ranks versus elements size: " + ((double)old_ranks/(double)startinglength -(double)this.rank_holder/(double)startinglength)*100 + "%"  );
				}
				
				indice_holder[0]=holder_of_elements_to_ranks;
				//indice_holder[1]=holder_of_elements_to_columns;	
				indice_holder[1]=holder_of_ranks_elements_v2;		
				//System.out.println("");
				holder_of_ranks_elements=null;
				holder_of_rank_to_sums=null;
				elements=null;
				columns=null;
				values=null;
			

			}
			
			
			System.gc();
			
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
		if (sdataset!=null){
			this.fit(sdataset);	
		}else if (fsdataset!=null){
				this.fit(fsdataset);				
		} else {
			throw new IllegalStateException(" No data structure specifed in the constructor" );			
		}	
	}
	


}
