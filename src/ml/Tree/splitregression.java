package ml.Tree;
import matrix.fsmatrix;

/**
 * <p>This class will perform the splitting for the tree regression algorithms .</p>
 */
public class splitregression implements Runnable {
		/**
		 * Sorted_indices from element to rank
		 */
		private  int [] holder_of_elements_to_ranks;	
		/**
		 * valid columns . Invalid columns will have -1
		 */
		private  int [] valid_columns;
		/**
		 * General Subset of rows
		 */
		private  int [] subset_of_rows;
		/**
		 * subset columns . Like valid columns but n different format
		 */
		private  int [] subset_columns;
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
		public  int [] zero_rank_holder;
		/**
		 * sparse matrix rows
		 */
		private  int [] indexpile;
		/**
		 * sparse matrix columns
		 */		
		private  int [] mainpile;
		/**
		 * double target use
		 */
		private  fsmatrix target;
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
		 * @param indexpile : holds column indices per row (as in the sparse matrix)
		 * @param mainpile : holds column indices (as in the sparse matrix)
		 * @param subset_of_rows : rows to consider
		 * @param valid_columns : columns to consider
		 * @param subset_columns : indexed subset of columns
		 * @param holder_of_elements_to_ranks : : elements to ranks 
		 * @param target : holds the target variable values
		 * @param weight : holds the weight per row
		 * @param start : start of the rows
		 * @param end : end of the rows
		 */
		
		public splitregression(
				double weight_sum[],
				double weight_count[],
				int count[],
				int zero_rank_holder[],
				int indexpile[],
				int mainpile[],
				int subset_of_rows[],
				int valid_columns[],
				int subset_columns[],				
				int  holder_of_elements_to_ranks [],
				fsmatrix target,
				double weight [], 
				int start,
				int end){				
			this.weighted_sum=weight_sum;
			this.weighted_count=weight_count;
			this.count=count;
			this.zero_rank_holder=zero_rank_holder;
			this.indexpile=indexpile;
			this.mainpile=mainpile;
			this.subset_of_rows=subset_of_rows;
			this.valid_columns=valid_columns;
			this.subset_columns=subset_columns;			
			this.holder_of_elements_to_ranks=holder_of_elements_to_ranks;
			this.target=target;
			this.weight=weight;
			this.start=start;
			this.end=end;
		}
		


		@Override
		public void run() {
			int min=this.subset_columns[start];
			int max=this.subset_columns[end-1];	
			int dimsnions=this.target.GetColumnDimension();
			if (this.weight==null ){
				// for all rows in the subset
				for (int i=0; i <subset_of_rows.length;i++){
					//get the row
				    int row=subset_of_rows[i];
				    //loop through all non zero elements
				    for (int el=this.indexpile[row];el<this.indexpile[row+1];el++ ){
				    	//retrieve the column
				    	int column=this.mainpile[el];
				    	
				    	// check if column is subset
				    	if (column<min || valid_columns[column]!=1){
				    		continue;
				    	}
				    	if (column> max){
				    		break;
				    	}					    	

				    	//retrieve the rank
				    	int rank=this.holder_of_elements_to_ranks[el];
				    		 this.count[rank]+=1;
				    		 this.weighted_count[rank]+=1.0;
				    		 
							int zerorank=this.zero_rank_holder[column];
							
							if (zerorank!=-1){
								count[zerorank]-=1;
								weighted_count[zerorank]-=1.0;
							}	
							
							
							for (int v=0; v <dimsnions; v++ ){
								 double temp=this.target.GetElement(row, v);
								 this.weighted_sum[(rank*dimsnions+v)]+=temp;
								 if (zerorank!=-1){
									 weighted_sum[(zerorank*dimsnions+v)]-=temp;
								 }
								}
				    	 
				    }

				}	
				} else {
					
					// for all rows in the subset
					for (int i=0; i <subset_of_rows.length;i++){
						//get the row
					    int row=subset_of_rows[i];
					    //loop through all non zero elements
					    for (int el=this.indexpile[row];el<this.indexpile[row+1];el++ ){
					    	//retrieve the column
					    	int column=this.mainpile[el];
					    	// check if column is subset
					    	if (column<min || valid_columns[column]!=1){
					    		continue;
					    	}
					    	if (column> max){
					    		break;
					    	}						    	

					    	//retrieve the rank
					    	int rank=this.holder_of_elements_to_ranks[el];
					    		 this.count[rank]+=1;
					    		 this.weighted_count[rank]+=this.weight[row];
					    		 
							  	 int zerorank=this.zero_rank_holder[column];
							  	 if (zerorank!=-1){
									 count[zerorank]-=1;
									 weighted_count[zerorank]-=this.weight[row];
							  	}
								 for (int v=0; v <dimsnions; v++ ){
									 double temp=this.target.GetElement(row, v)*this.weight[row];
									 this.weighted_sum[(rank*dimsnions+v)]+=temp;
									 if (zerorank!=-1){
										 weighted_sum[(zerorank*dimsnions+v)]-=temp;
									 }
									}				    		 

					    	 
					    }

					}					
					
				}
			
		}

	
	
}
