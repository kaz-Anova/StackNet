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
import java.util.Random;

import utilis.XorShift128PlusRandom;
import matrix.fsmatrix;

/**
 * <p>This class will perform the splitting for the tree regression algorithms .</p>
 */
public class splitintadjustednomap2 implements Runnable {
	/**
	 * The total weighted count
	 */
	public double sum_weighted_count;
	/**
	 * the overall sum values (of the target)
	 */
	public double total_sum_values[];
	/**
	 * the metric gain for the best variable
	 */
	public double best_gamma_array []; 
	/**
	 * counts of less-than-feature-value count 
	 */
	public double sum_weighted_less_array [];
	/**
	 * counts of less-than-feature-value count 
	 */
	public double count_weighted_less_array [];
	/**
	 * counts of less-than-feature-value count 
	 */
	public int countless_array [];
	/**
	 * Sorted_indices for this feture
	 */
	public int [] sorted_indices;
	
	/**
	 * Holds the maximum rank
	 */
	public int maximum_rank=1;
	/**
	 * General Subset of rows
	 */
	public int [] subset;
	
	/**
	 * Bets Rank for current feature
	 */
	private int bestrank=-1;
	
	public int getbestrank(){
		return bestrank;
	}
	/**
	 * Holds the rank of the 'zero' (e.g. sparse) elements
	 */
	private int zero_rank_holder []; 
	
	/**
	 * Bets Rank for current feature
	 */
	private int bestrow=-1;
	
	public int getbestrow(){
		return bestrow;
	}
	/**
	 * The distance to use. It has to be one of  RMSE,MAE,QUANTILE
	 */
	public String criterion="RMSE";
	/**
	 * The feature to find best split from
	 */
	public int feature=-1;
	/**
	 * minimum allowed weight in a leaf
	 */
	public double min_leaf=1;
	/**
	 * Tau for quantile
	 */
	public double tau=0.5;
	/**
	 * % of possible cuttoffs to keep
	 */
	public double cut_off_subsample=1.0;
    /**
     * seed to use
     */
	public int seed=1;
	/**
	 * Random number generator to use
	 */
	private Random random;
	/**
	 * offset for divisions
	 */
	public double offset=0.00001;
	/**
	 * double target use , an array of 0 and 1
	 */
	public fsmatrix target;
	/**
	 * Where the weight is stored 
	 */
	public double weight[];



	/**
	 * 
	 * @param data : The data to Extract the column from
	 * @param rows : Sorted row indices
	 * @param maintarget : the target variable in fsmatrix format
	 * @param featuren : the feature to extract (e.g. find best split) from
	 * @param criterion : criterion to use (like RMSE, MAE...)
	 * @param cut_off_subsample : percentage of cut offs to keep
	 * @param min_leaf : minimum number of weighted count (cases) to allow to exist in a node
	 * @param best_cuttof_array : Array to hold the best cuttof (has length 1)
	 * @param best_gamma_array : array to hold the best metric value (has length 1)
	 * @param countless_array : integer array to hold the number of cases less equal to the best cut-toff
	 * @param seed : seed to use
	 * @param sum_weighted_count : total weight counted for this subselection of rows
	 * @param sum_values : total weight sum for subselection of rows
	 * @param sum_weighted_less_array : initially empty-then filled array of size [target columns] to assign the best weighted sum per value of value lower than the best cuttoff
	 * @param count_weighted_less_array : initially empty-then filled array [size of 1] holding the weighted count of value lower than the best cuttoff
	 * @param subset : Subset of rows to use
	 
	 */
	public splitintadjustednomap2(int [] rows,  fsmatrix maintarget, 
			int featuren ,String criterion, double cut_off_subsample, double min_leaf,
			double best_gamma_array [], int countless_array [] , 
			 int seed ,double sum_weighted_count, double sum_values[], double sum_weighted_less_array []
					 , double count_weighted_less_array [],int []  subset, int max_rank, int [] zerorank){
		/*
		if (subset==null || subset.length<=0){
			throw new IllegalStateException(" There are no rows!" );
		}

		if ( !criterion.equals("RMSE")  &&  !criterion.equals("MAE") && !criterion.equals("QUANTILE") ){
			throw new IllegalStateException("  distance  has to be one of RMSE,MAE or QUANTILE" );	
		}			
		if (featuren<0){
			throw new IllegalStateException("The feature cannot be negative" );	
		}
		if (cut_off_subsample<0){
			cut_off_subsample=1.0;
		}	
		if (sum_values.length!=maintarget.GetColumnDimension()){
			throw new DimensionMismatchException(sum_values.length,target.GetColumnDimension());
		}
		if (sum_weighted_less_array.length!=maintarget.GetColumnDimension()){
			throw new DimensionMismatchException(sum_weighted_less_array.length,target.GetColumnDimension());
		}
		*/
		this.sum_weighted_less_array=sum_weighted_less_array;
		this.count_weighted_less_array=count_weighted_less_array;
		this.sum_weighted_count=sum_weighted_count;
		this.total_sum_values=sum_values;		
		this.best_gamma_array=best_gamma_array;
		this.countless_array =countless_array;
		this.cut_off_subsample=cut_off_subsample;
		this.feature=featuren;
		this.target=maintarget;
		this.criterion=criterion;
		//this.min_leaf=min_leaf;
		this.seed=seed;
		this.sorted_indices=rows;
		this.subset=subset;
		this.zero_rank_holder=zerorank;
		this.maximum_rank=max_rank;
		}
	
/**
 * <p> Finds the best split streaming through a single feature (column).</p>
 */
	private void fit() {
	
		// set random number generator 
		random = new  XorShift128PlusRandom(seed);
		int rank_to_rows[]= new int[this.maximum_rank+1];
		double target_vals[]=new double[this.target.GetColumnDimension()*(this.maximum_rank+1)];
		int counters[]= new int [this.maximum_rank+1];
		double weighted_counters[]=new double[this.maximum_rank+1];
		double current_weighted_count=0.0;
		int current_count_int=0;
		double current_targetvalue [] = new double [this.target.GetColumnDimension()];
		int thresoldcut=utilis.util.get_random_integer(this.cut_off_subsample);
		if (this.zero_rank_holder==null){ // e.g if we dont have a sparse matrix
			
		if (this.weight==null ){
		
		for (int row: subset){
			int rank=this.sorted_indices[row];
			counters[rank]+=1;
			weighted_counters[rank]+=1.0;
			  for (int v=0; v <this.target.GetColumnDimension(); v++ ){
				  target_vals[(rank*this.target.GetColumnDimension())+v]+=this.target.GetElement(row, v);///zero_counter ;
					}
			  rank_to_rows[rank]=row+1;
		}	
		} else {
			for (int row: subset){
				int rank=this.sorted_indices[row];
				counters[rank]+=1;
				weighted_counters[rank]+=this.weight[row];
				  for (int v=0; v <this.target.GetColumnDimension(); v++ ){
					  target_vals[(rank*this.target.GetColumnDimension())+v]+=this.target.GetElement(row, v)*this.weight[row];///zero_counter ;
						}
				  rank_to_rows[rank]=row+1;
			}
		}

		} else { // We do have a Sparse matrix ...unfortunately (the coder suffers from it)

			int the_zero_rank=this.zero_rank_holder[this.feature];
			
			if (the_zero_rank!=-1){
				
				int data_rows=this.target.GetRowDimension();
				int densened_sorted_indices []= new int [data_rows];
				
				for (int i=0; i <sorted_indices.length; ){
					densened_sorted_indices[sorted_indices[i]]=sorted_indices[i+1];
					i+=2;
				}
				
				
				if (this.weight==null ){
					for (int row: subset){
						int rank=densened_sorted_indices[row];
						if (rank==0){
							rank=the_zero_rank;
						}
						counters[rank]+=1;
						weighted_counters[rank]+=1.0;
						  for (int v=0; v <this.target.GetColumnDimension(); v++ ){
							  target_vals[(rank*this.target.GetColumnDimension())+v]+=this.target.GetElement(row, v);///zero_counter ;
								}
						  rank_to_rows[rank]=row+1;
	
					}	
					} else {
						for (int row: subset){
							int rank=densened_sorted_indices[row];
							if (rank==0){
								rank=the_zero_rank;
							}						
							counters[rank]+=1;
							weighted_counters[rank]+=this.weight[row];
							  for (int v=0; v <this.target.GetColumnDimension(); v++ ){
								  target_vals[(rank*this.target.GetColumnDimension())+v]+=this.target.GetElement(row, v)*this.weight[row];///zero_counter ;
									}
							  rank_to_rows[rank]=row+1;
	
						}
					}	
			
			
			} else {
				
				
				if (this.weight==null ){
					for (int row: subset){
						int rank=this.sorted_indices[row];
						counters[rank]+=1;
						weighted_counters[rank]+=1.0;
						  for (int v=0; v <this.target.GetColumnDimension(); v++ ){
							  target_vals[(rank*this.target.GetColumnDimension())+v]+=this.target.GetElement(row, v);///zero_counter ;
								}
						  rank_to_rows[rank]=row+1;

					}	
					} else {
						for (int row: subset){
							int rank=this.sorted_indices[row];
							counters[rank]+=1;
							weighted_counters[rank]+=this.weight[row];
							  for (int v=0; v <this.target.GetColumnDimension(); v++ ){
								  target_vals[(rank*this.target.GetColumnDimension())+v]+=this.target.GetElement(row, v)*this.weight[row];///zero_counter ;
									}
							  rank_to_rows[rank]=row+1;

						}
					}
				
				
				
				
				
				
			}
			
			
			
			
			
	
		}
		
		
		
		 for (int s=1; s < rank_to_rows.length; s++){
				
				if (counters[s]==0){
					continue;
				}
				
				//int i=rank_to_rows[s]-1;
			 	 //current_rank=s;
				 //currentvalue=data.GetElement(i, this.feature);
				 current_weighted_count+=weighted_counters[s];
				 current_count_int+=counters[s];
				 for (int k=0; k <current_targetvalue.length; k++ ){
					 current_targetvalue[k]+=target_vals[(s*this.target.GetColumnDimension())+k];
				 } 					
					 
			 // check if the new value is different than the previous one
			 if (random.nextInt()<=thresoldcut) {
					 // we check if we need to add the zero. It is painful, but could not thing of better way to do that-sorry users!
					 // on happier note, I do not think people would want to use it anyway such a mess I've made so we-re good

						 double tempmetric=Double.NEGATIVE_INFINITY;
						 if (this.criterion.equals("RMSE")){
							 tempmetric= calculateRMSE( current_targetvalue,this.total_sum_values,
									 current_weighted_count, this.sum_weighted_count);
						 } else {
							 tempmetric= calculateMAE( current_targetvalue,this.total_sum_values,
									 current_weighted_count, this.sum_weighted_count); 
						 }
						 if (tempmetric> this.best_gamma_array[0]) { // metric better, gets inserted
							 this.countless_array [0] =current_count_int;
							 this.count_weighted_less_array[0]= current_weighted_count;
						     this.best_gamma_array[0]=tempmetric; // since it is the best, it is definitely better than the default
						     this.bestrank=s;
						     bestrow=  rank_to_rows[s]-1;
						     sum_weighted_less_array=current_targetvalue.clone();
					 
						 }
					 // end of check whether the new cut-off is better metric-wise than the previous
				 }

				 // end of loop
		}	
		
		target_vals=null;
		counters=null;
		rank_to_rows=null;
		weighted_counters=null;
		current_targetvalue=null;
		//this.weight=null;
		 

		//System.gc();

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

			this.fit();
	
			
	}
	/**
	 * 
	 * @param current_target_values_sum :  sum of all targets until the current split
	 * @param total_target_values_sum :total sum of all target
	 * @param current_count : number of (weighted) cases for the current split
	 * @param total_count : total number of (weighted) cases for the current split
	 * @return double value of the RMSE metric for that split
	 */
	
	public double calculateRMSE(double current_target_values_sum [],double total_target_values_sum [],
		   double current_count, double total_count){
		   
		   double other_count=total_count-current_count;
		   current_count+=this.offset;
		   other_count+=this.offset;
		   /*
		   double other_proportion=other_count/total_count;
		   double current_proportion=current_count/total_count;
		   
		   //we are not allowing min_leaf cases so they should be inlcuded in finding the best split
		   if (other_count< this.min_leaf){
			   other_proportion=0;
		   }
		   if (current_count< this.min_leaf){
			   current_proportion=0;
		   }
		   */
		 double metrix=0.0;
		 
		 for (int k=0; k <current_target_values_sum.length; k++ ){
			 /*			 
				 double diff=(current_target_values_sum[k]/(current_count) - total_target_values_sum[k]/(total_count) );
				 metrix+= current_count * diff * diff; 
				 diff=( ( total_target_values_sum[k]-current_target_values_sum[k])/(other_count)  - total_target_values_sum[k]/(total_count) );
				 metrix+= other_count * diff * diff;
				
			  	 */
			 	double diff=(current_target_values_sum[k]/current_count) -( total_target_values_sum[k]-current_target_values_sum[k])/(other_count) ;	 
			  	metrix+=	 other_count *current_count* diff * diff;

		 }
		return metrix;
		
	}
/**
 * 
 * @param current_target_values_sum :  sum of all targets until the current split
 * @param total_target_values_sum :total sum of all target
 * @param current_count : number of (weighted) cases for the current split
 * @param total_count : total number of (weighted) cases for the current split
 * @return double value of the MAE metric for that split
 */
	public double calculateMAE(double current_target_values_sum [],double total_target_values_sum [],
			double current_count, double total_count){
		
		   double other_count=total_count-current_count;
		   current_count+=this.offset;
		   other_count+=this.offset;
		   /*
		   double other_proportion=other_count/total_count;
		   double current_proportion=current_count/total_count;
		   
		   //we are not allowing min_leaf cases so they should be inlcuded in finding the best split
		   if (other_count< this.min_leaf){
			   other_proportion=0;
		   }
		   if (current_count< this.min_leaf){
			   current_proportion=0;
		   }
		 double metrix=0.0;
		 for (int k=0; k <current_target_values_sum.length; k++ ){
			 if (current_proportion>0 && current_count >0 && total_count>0) {
				 double diff=Math.abs((current_target_values_sum[k]/current_count - total_target_values_sum[k]/total_count ));
				 metrix+= current_proportion * diff ;
			 } 
			 if (other_proportion>0 && other_count >0 && total_count>0) {
				 double diff=Math.abs(( ( total_target_values_sum[k]-current_target_values_sum[k])/other_count - total_target_values_sum[k]/total_count ));
				 metrix+= other_proportion * diff;
			 } 			 
		 }
		 */
			 double metrix=0.0;
			 
			 for (int k=0; k <current_target_values_sum.length; k++ ){
				 /*
					 double diff=Math.abs((current_target_values_sum[k]/(current_count) - total_target_values_sum[k]/(total_count) ));
					 metrix+= current_count * diff * diff; 
					 diff=Math.abs(( ( total_target_values_sum[k]-current_target_values_sum[k])/(other_count)  - total_target_values_sum[k]/(total_count) ));
					 metrix+= other_count * diff * diff;
					 */
				  	metrix+=other_count *current_count*Math.abs((current_target_values_sum[k]/current_count) -( total_target_values_sum[k]-current_target_values_sum[k])/(other_count) );	 
				  			 
			 } 
			 
			 
		return metrix;
		
	}

}
