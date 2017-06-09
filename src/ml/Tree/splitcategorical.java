package ml.Tree;


/**
 * <p>This class will perform the splitting for the tree regression algorithms .</p>
 */
public class splitcategorical implements Runnable {
	
		/**
		 * Sorted_indices from element to rank
		 */
	
		private int [] ranks_for_accessing;
		/**
		 * Columns' indexer for the ranks
		 */		
		private int []  cols_for_accessing;
		/**
		 * Rows' indexer for the ranks
		 */				
		private int []  rows_for_accessing;

		/**
		 * Holds the weighted sums per rank 
		 */
		public  double weighted_sum[];
		/**
		 * Holds the weighted count per rank 
		 */
		public  double weighted_count[];
		/**
		 * Holds the count per rank 
		 */
		public  int count[];

		/**
		 * Holds indices of zero columns
		 */
		public  int zero_rank_holder[];
		/**
		 * distinct classes
		 */
		public  int  dimsnions;

		/**
		 * double target use
		 */
		private  int [] target;
		/**
		 * Where the weight is stored 
		 */
		private  double weight[];		
		/**
		 * start of the rows' loop
		 */
		int start=-1;
		/**
		 * end of the rows' loop
		 */	
		int end=-1;

		/**
		 * 
		 * @param weight_sum : holder for the total sum of target variable per activated rank
		 * @param weight_count : holder for the total weighted count per activated rank
		 * @param count :  holder for the total count per activated rank
		 * @param zero_rank_holder :  holds the rank of zero elements
		 * @param ranks_for_accessing : Sorted_indices from element to rank
		 * @param cols_for_accessing : Columns' indexer for the ranks
		 * @param rows_for_accessing : Rows' indexer for the ranks
		 * @param target : holds the target variable values
		 * @param weight : holds the weight per row
		 * @param dimsnions :distinct classes
		 * @param start : start of the subset_columns
		 * @param end : end of the subset_columns
		 */
		
		public splitcategorical(
				double weight_sum[],
				double weight_count[],
				int count[],
				int zero_rank_holder[],
				int [] ranks_for_accessing, 
				int []  cols_for_accessing,
				int []  rows_for_accessing,
				int target [],
				double weight [],
				int dimsnions,
				int start,
				int end){				
			this.weighted_sum=weight_sum;
			this.weighted_count=weight_count;
			this.count=count;
			this.zero_rank_holder=zero_rank_holder;
			this.ranks_for_accessing=ranks_for_accessing;
			this.cols_for_accessing=cols_for_accessing;
			this.rows_for_accessing=rows_for_accessing;
			this.target=target;
			this.weight=weight;
			this.dimsnions=dimsnions;
			this.start=start;
			this.end=end;
		}
		


		@Override
		public void run() {	
			if (weight==null ){
				// for all rows in the subset
				for (int i=start; i <end;i++){
					//get the row
				    int row=rows_for_accessing[i];
				    int column=cols_for_accessing[i];
				    int rank=ranks_for_accessing[i];
				    //loop through all non zero elements

			    	count[rank]+=1;
			    	weighted_count[rank]+=1.0;
			    	weighted_sum[(rank*dimsnions+target[row])]+=1.0;

					int zerorank=this.zero_rank_holder[column];
					
					if (zerorank==-1){
						continue;
					}	
					count[zerorank]-=1;
					weighted_count[zerorank]-=1.0;
					weighted_sum[(zerorank*dimsnions+target[row])]-=1.0;



				}	
				} else {
					
					// for all rows in the subset
					for (int i=start; i <end;i++){
						//get the row
					    int row=rows_for_accessing[i];
					    int column=cols_for_accessing[i];
					    int rank=ranks_for_accessing[i];

				    	count[rank]+=1;
				    	weighted_count[rank]+=weight[row];			    	
				    	weighted_sum[(rank*dimsnions+target[row])]+=weight[row];
				    	
						int zerorank=this.zero_rank_holder[column];
						
						if (zerorank==-1){
							continue;
						}	
						count[zerorank]-=1;
						weighted_count[zerorank]-=weight[row];	
						weighted_sum[(zerorank*dimsnions+target[row])]-=weight[row];
					    	 
					    

					}					
					
				}
			
		}

	
	
}
