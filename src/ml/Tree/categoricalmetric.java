package ml.Tree;

import java.util.Random;

import utilis.XorShift128PlusRandom;

/**
 * 
 * Used to find best split for classification in parallel
 *
 */
public class categoricalmetric implements Runnable {
	/**
	 * for entropy estimations
	 */
	 double math_log_2= Math.log(2.0);
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
	 * count of ranks
	 */
	public  int [] count;
	/**
	 * links ranks with columns
	 */
	public int [] rank_to_column;
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
	public  String criterion="ENTROPY";
	/**
	 * start of the ranks' loop
	 */
	private int start=-1;
	/**
	 * end of the ranks' loop
	 */	
	private int end=-1;
	/**
	 *threshold for cut-off subsample
	 **/
	int thresold;
	/**
	 * small offset to set a regularization
	 */
	public double offset=0.0001;
	/**
	 * minimum leaf size 
	 */
	public double min_leaf=2.0;

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
	 * @param thresold : threshold for cut-off subsample
	 * @param rank_to_column : links ranks with columns
	 * @param count : counts per rank
	 * @param weighted_sum : Holds the weighted sums per rank 
	 * @param weighted_count : Holds the weighted count per rank 
	 * @param criterion : The distance to use. It has to be one of  ENTROPY,GINI,AUC
	 * @param number : index
	 * @param start : start of the rows
	 * @param end : end of the rows
	 * @param dimension : column dimension
	 */
	
	public categoricalmetric(
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
	 * @param thresold : threshold for cut-off subsample
	 * @param rank_to_column : links ranks with columns
	 * @param count : counts per rank
	 * @param weighted_sum : Holds the weighted sums per rank 
	 * @param weighted_count : Holds the weighted count per rank 
	 * @param criterion : The distance to use. It has to be one of  ENTROPY,GINI,AUC
	 * @param number : index
	 * @param start : start of the rows
	 * @param end : end of the rows
	 */
	
	public categoricalmetric(
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

		double best_current_gamma=Double.NEGATIVE_INFINITY;
		int best_current_rank=-1;
		int previous_col=-1;
		double current_targetvalue[] =new double [this.total_sum_values.length];
		double current_weighted_count=0.0;
		int current_int=0;	
		Random random = new XorShift128PlusRandom(this.thread_number);
		
		if (this.dimension>0){
			
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
						 if (criterion.equals("ENTROPY")){
							 tempmetric= calculateENTROPY( current_targetvalue,total_sum_values,
									 current_weighted_count, sum_weighted_count);
						 } else if (criterion.equals("AUC")){
							 tempmetric= calculateAUC( current_targetvalue,total_sum_values,
									 current_weighted_count, sum_weighted_count); 
						 } else {
							 tempmetric= calculateGINI( current_targetvalue,total_sum_values,
									 current_weighted_count, sum_weighted_count); 
						 }
						 if (tempmetric> best_current_gamma) { // metric better, gets inserted
							 best_current_gamma=tempmetric; 
						    best_current_rank=rank;
						 
						 }
				}	 // end of check whether the new cut-off is better metric-wise than the previous
				
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
					 if (criterion.equals("ENTROPY")){
						 tempmetric= calculateENTROPY( current_targetvalue,total_sum_values,
								 current_weighted_count, sum_weighted_count);
					 } else if (criterion.equals("AUC")){
						 tempmetric= calculateAUC( current_targetvalue,total_sum_values,
								 current_weighted_count, sum_weighted_count); 
					 } else {
						 tempmetric= calculateGINI( current_targetvalue,total_sum_values,
								 current_weighted_count, sum_weighted_count); 
					 }
					 if (tempmetric> best_current_gamma) { // metric better, gets inserted
						 best_current_gamma=tempmetric; 
					    best_current_rank=rank;
					 
					 }
			}	 // end of check whether the new cut-off is better metric-wise than the previous
			
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
	 * @return double value of the ENTROPY metric for that split
	 */
	public double calculateENTROPY(double current_target_values_sum [],double total_target_values_sum [],
		   double current_count, double total_count){
		   
		   double other_count=total_count-current_count;

		   //current_count+=this.offset;
		   //other_count+=this.offset;
		   //total_count+=this.offset;
		   if (current_count>0 || other_count>0 ){
			 double metrix=0.0;
      		 double statistic2=0;
      		 double statistic1=0;
      		 double temp=0.0;
 			 //statistic1-=temp[t]/(sum+0.001) * Math.log((temp[t]+0.001)/(sum+0.001))/Math.log(2);
 			 //statistic2-=((kepecounts[t]-temp[t]))/((sumy-sum+0.001)) * Math.log(((kepecounts[t]-temp[t]+0.001))/((sumy-sum+0.001)))/Math.log(2);

      		 
      		 for (int t=0; t < current_target_values_sum.length; t++){
      			 if (current_count!=0.0){
      			temp=current_target_values_sum[t]/current_count;
      			statistic1-=temp * Math.log(temp);
      			 }
      			 if (other_count!=0.0){
      			temp= (total_target_values_sum[t]-current_target_values_sum[t])/other_count;
      			statistic2-= temp * Math.log(temp);
      			 }
      			//temp=(current_count-current_target_values_sum[t])/(current_count);
      			//statistic1+=temp* Math.log(temp)/math_log_2;
      			//temp=(((total_count-total_target_values_sum[t])-(current_count-current_target_values_sum[t]))/(other_count));
      			//statistic2+=temp* Math.log(temp)/math_log_2;

      		 } 
      		if (statistic1!=0 || statistic2!=0){
      		metrix=1.0/(statistic1*(current_count)+statistic2*((other_count)));
      		return metrix;
      		} else {
      		  return Double.NEGATIVE_INFINITY;  
      		}
      		//System.out.println(metrix);
      		
		
		   } else {
			   return Double.NEGATIVE_INFINITY; 
		   }
		
	}
	/**
	 * 
	 * @param current_target_values_sum :  sum of all targets until the current split
	 * @param total_target_values_sum :total sum of all target
	 * @param current_count : number of (weighted) cases for the current split
	 * @param total_count : total number of (weighted) cases for the current split
	 * @return double value of the GINI metric for that split
	 */
	public double calculateGINI(double current_target_values_sum [],double total_target_values_sum [],
			double current_count, double total_count){
		
		   double other_count=total_count-current_count;
		   if (current_count>0 || other_count>0 ){

			 double metrix=0.0;
			 double temp=0.0;
      		 double statistic2=0;
      		 double statistic1=0;
      		 
  			 for (int t=0; t < current_target_values_sum.length; t++){
  			if (current_count!=0.0){ 
				 temp=current_target_values_sum[t]/(current_count);
	  			 statistic1+=temp*temp ;
  			 }
 			 if (other_count!=0.0){  			
				 temp=((total_target_values_sum[t]-current_target_values_sum[t])/(other_count));
	  			 statistic2+=temp*temp ; 
 			 }
			 //temp=(current_count-current_target_values_sum[t])/(current_count);
  			 //statistic1+=temp*temp  ;
			 //temp=(((total_count-total_target_values_sum[t])-(current_count-current_target_values_sum[t]))/(other_count));
  			 //statistic2+=temp*temp  ;       			 
  			
  			 } 
  			if (statistic1!=0 || statistic2!=0){
  			metrix=(statistic1*(current_count)+statistic2*(other_count));
  			return metrix;
     		} else {
     		  return Double.NEGATIVE_INFINITY;  
     		}
     		//System.out.println(metrix);
     		
		
		   } else {
			   return Double.NEGATIVE_INFINITY; 
		   }
		
	}
	/**
	 * 
	 * @param current_target_values_sum :  sum of all targets until the current split
	 * @param total_target_values_sum :total sum of all target
	 * @param current_count : number of (weighted) cases for the current split
	 * @param total_count : total number of (weighted) cases for the current split
	 * @return double value of the AUC metric for that split
	 */
	public double calculateAUC(double current_target_values_sum [],double total_target_values_sum [],
			double current_count, double total_count){
		  
		   double good_before=current_target_values_sum[1] ;
		   double bad_before=current_target_values_sum[0] ;		
		   double good_after=total_target_values_sum[1] - current_target_values_sum[1] ;
		   double bad_after=total_target_values_sum [0] - current_target_values_sum[0] ;	
		   double total_combos=total_target_values_sum[0]*total_target_values_sum[1];
		   double odds_before=good_before/(good_before+bad_before);
		   double odds_after=good_after/(good_after+bad_after);
		   double constant=0.5*good_before*bad_before + 0.5*good_after*bad_after; 
		   double metrix=0.5;
		   
		   if (odds_before>odds_after ){
			   metrix=good_before*bad_after+ constant ;
		   } else {
			   metrix=good_after*bad_before+constant;
		   }
		   
		   return metrix/total_combos;
		   
		   /*
		   double metrix=0.5;
		   double constant=0.5*current_target_values_sum[1]*current_target_values_sum[0] + 0.5*(total_target_values_sum[1] - current_target_values_sum[1])*(total_target_values_sum [0] - current_target_values_sum[0]);
		   if (current_target_values_sum[1]/(current_target_values_sum[1]+current_target_values_sum[0])>(total_target_values_sum[1] - current_target_values_sum[1])/((total_target_values_sum[1] - current_target_values_sum[1])+(total_target_values_sum [0] - current_target_values_sum[0]) ) ){
			   metrix=current_target_values_sum[1]*(total_target_values_sum [0] - current_target_values_sum[0])+ constant ;
		   } else {
			   metrix=(total_target_values_sum[1] - current_target_values_sum[1])*current_target_values_sum[0]+constant;
		   }		   
		   	
		return metrix/(total_target_values_sum[0]*total_target_values_sum[1]);
		*/
		   
		
	}

}
