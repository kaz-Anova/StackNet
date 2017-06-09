package ml.Tree;

import java.util.Random;

import utilis.XorShift128PlusRandom;

/**
 * 
 * Used to find best split for regression in parallel
 *
 */
public class regressionmetric implements Runnable {

	/**
	 * The total weighted count
	 */
	public  double sum_weighted_count;
	/**
	 * the overall sum values (of the target)
	 */
	public  double total_sum_values[];
    /**
     *  thread number to assign
     */
	public int thread_number=0;
	/**
	 * Holds the best metric found by each one of the threads
	 */
	public  double best_gamma[];
	/**
	 * Holds the best rank found by each one of the threads
	 */
	public  int [] best_ranks;
	/**
	 * General Subset of ranks
	 */
	public  int [] subset_of_ranks;
	/**
	 * Holds the weighted sums per rank 
	 */
	public  double weighted_sum[];
	/**
	 * Holds the weighted count per rank 
	 */
	public  double weighted_count[];

	/**
	 * The distance to use. It has to be one of  ENTROPY,GINI,AUC
	 */
	public  String criterion="RMSE";
	/**
	 * start of the ranks' loop
	 */
	private int start=-1;
	/**
	 * end of the ranks' loop
	 */	
	private int end=-1;

	/**
	 * small offset to set a regularization
	 */
	public double offset=0.0001;
	/**
	 *threshold for cut-off subsample
	 **/
	int thresold;
	/**
	 * count of ranks
	 */
	public  int [] count;
	/**
	 * links ranks with columns
	 */
	public int [] rank_to_column;
	
	/**
	 * column dimension for desne data
	 */
	public int dimension=-1;
	
	/**
	 * 
	 * @param sum_weighted_count : The total weighted count
	 * @param total_sum_values : the overall sum values (of the target)
	 * @param best_gamma : Holds the best metric found by each one of the threads
	 * @param best_ranks : Holds the best rank found by each one of the threads
	 * @param subset_of_ranks : General Subset of ranks
	 * @param weighted_sum : Holds the weighted sums per rank 
	 * @param weighted_count : Holds the weighted count per rank 
	 * @param criterion : The distance to use. It has to be one of  ENTROPY,GINI,AUC
	 * @param number : index
	 * @param start : start of the rows
	 * @param end : end of the rows
	 * @param dimension : column dimension
	 */
	
	public regressionmetric(
			double sum_weighted_count,
			double total_sum_values[],
			double best_gamma[],
			int [] best_ranks,
			int thresold,
			int [] rank_to_column,			
			int [] count,
			double weighted_sum[],
			double weighted_count[],
			String criterion,
			int number, 
			int start,
			int end,
			int dimension){	
		
		this.sum_weighted_count=sum_weighted_count;
		this.total_sum_values=total_sum_values;
		this.best_gamma=best_gamma;
		this.best_ranks=best_ranks;
		this.thresold=thresold;
		this.rank_to_column=rank_to_column;
		this.count=count;
		this.weighted_sum=weighted_sum;
		this.weighted_count=weighted_count;
		this.criterion=criterion;	
		this.thread_number=number;
		this.start=start;
		this.end=end;
		this.dimension=dimension;
	}
	
	/**
	 * 
	 * @param sum_weighted_count : The total weighted count
	 * @param total_sum_values : the overall sum values (of the target)
	 * @param best_gamma : Holds the best metric found by each one of the threads
	 * @param best_ranks : Holds the best rank found by each one of the threads
	 * @param subset_of_ranks : General Subset of ranks
	 * @param weighted_sum : Holds the weighted sums per rank 
	 * @param weighted_count : Holds the weighted count per rank 
	 * @param criterion : The distance to use. It has to be one of  ENTROPY,GINI,AUC
	 * @param number : index
	 * @param start : start of the rows
	 * @param end : end of the rows
	 */
	
	public regressionmetric(
			double sum_weighted_count,
			double total_sum_values[],
			double best_gamma[],
			int [] best_ranks,
			int thresold,
			int [] rank_to_column,			
			int [] count,
			double weighted_sum[],
			double weighted_count[],
			String criterion,
			int number, 
			int start,
			int end){	
		
		this.sum_weighted_count=sum_weighted_count;
		this.total_sum_values=total_sum_values;
		this.best_gamma=best_gamma;
		this.best_ranks=best_ranks;
		this.thresold=thresold;
		this.rank_to_column=rank_to_column;
		this.count=count;
		this.weighted_sum=weighted_sum;
		this.weighted_count=weighted_count;
		this.criterion=criterion;	
		this.thread_number=number;
		this.start=start;
		this.end=end;
	}
	
	@Override
	public void run() {
		//System.out.println(thread_number + " " + this.start + " " + this.end + " " + best_ranks.length + " " + best_gamma.length);
		double best_current_gamma=Double.NEGATIVE_INFINITY;
		int best_current_rank=-1;
		int previous_col=-1;
		double current_targetvalue[] =new double [this.total_sum_values.length];

		double current_weighted_count=0.0;
		int current_int=0;	
		Random random = new XorShift128PlusRandom(this.thread_number);
		
		if (this.dimension>0){	
			
			// sum all values and prepare to find the best cut_off	
			
			for (int z=start; z<end; z++ ){
				
				int rank=z;
				if (weighted_count[rank]==0.0){
					continue;
				}
				
				int el=this.rank_to_column[rank];
				int newcol=el%this.dimension;
	
				if (newcol>previous_col){
					previous_col=newcol;
					current_targetvalue =new double [this.total_sum_values.length];
					current_weighted_count=0.0;
					current_int=0;	
				}
				
				current_weighted_count+=weighted_count[rank];
				current_int+=count[rank];
				weighted_count[rank]=current_weighted_count;
				count[rank]=current_int;
				
				for (int c=0; c <this.total_sum_values.length;c++){
					current_targetvalue[c]+=weighted_sum[rank*this.total_sum_values.length + c];
					weighted_sum[rank*this.total_sum_values.length + c]=current_targetvalue[c];
				}
				if (random.nextInt()<=this.thresold){					
	
							 double tempmetric=Double.NEGATIVE_INFINITY;
							 if (this.criterion.equals("RMSE")){
								 tempmetric= calculateRMSE( current_targetvalue,this.total_sum_values,
										 current_weighted_count, this.sum_weighted_count);
							 }  else {
								 tempmetric= calculateMAE( current_targetvalue,this.total_sum_values,
										 current_weighted_count, this.sum_weighted_count); 
							 }
							 if (tempmetric> best_current_gamma) { // metric better, gets inserted
								 best_current_gamma=tempmetric; 
							    best_current_rank=rank;
							 
							 }
						 // end of check whether the new cut-off is better metric-wise than the previous
					
				}
			}
			best_ranks[this.thread_number]=best_current_rank;
			best_gamma[this.thread_number]=best_current_gamma;
			
		} else {
			
			for (int z=start; z<end; z++ ){
				
				int rank=z;
				if (weighted_count[rank]==0.0){
					continue;
				}
				
				int newcol=this.rank_to_column[rank];
	
				if (newcol>previous_col){
					previous_col=newcol;
					current_targetvalue =new double [this.total_sum_values.length];
					current_weighted_count=0.0;
					current_int=0;	
				}
				
				current_weighted_count+=weighted_count[rank];
				current_int+=count[rank];
				weighted_count[rank]=current_weighted_count;
				count[rank]=current_int;
				
				for (int c=0; c <this.total_sum_values.length;c++){
					current_targetvalue[c]+=weighted_sum[rank*this.total_sum_values.length + c];
					weighted_sum[rank*this.total_sum_values.length + c]=current_targetvalue[c];
				}
				if (random.nextInt()<=this.thresold){					
	
							 double tempmetric=Double.NEGATIVE_INFINITY;
							 if (this.criterion.equals("RMSE")){
								 tempmetric= calculateRMSE( current_targetvalue,this.total_sum_values,
										 current_weighted_count, this.sum_weighted_count);
							 }  else {
								 tempmetric= calculateMAE( current_targetvalue,this.total_sum_values,
										 current_weighted_count, this.sum_weighted_count); 
							 }
							 if (tempmetric> best_current_gamma) { // metric better, gets inserted
								 best_current_gamma=tempmetric; 
							    best_current_rank=rank;
							 
							 }
						 // end of check whether the new cut-off is better metric-wise than the previous
					
				}
			}
			best_ranks[this.thread_number]=best_current_rank;
			best_gamma[this.thread_number]=best_current_gamma;
			
			
		}
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
