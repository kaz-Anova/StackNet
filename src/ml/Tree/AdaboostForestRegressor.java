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

import Stats.DescriptiveStatistics;
import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;

/**
 * 
 * @author mariosm
 *<p> This class will perform addaboost regression ,where an ensemble of learners is created via 
 *adjusting the weights of the f<sub>m+1</sub> based on the error of the previous f<sub>m</sub> learner.</p>
 <p> 
 *ref : Freund, Y., & Schapire, R. E. (1995, March). A desicion-theoretic generalization of on-line learning and an application to boosting. In European conference on computational learning theory (pp. 23-37). Springer Berlin Heidelberg.
Chicago	 </p>

 */

public class AdaboostForestRegressor implements estimator,regressor {

	/**
	 * This keeps the sorted indices ranks
	 */
	private int sorted_indices [][];

	public void set_sorted_indices (int indices [][]){
		if (indices==null || indices.length !=this.columndimension) {
			throw new IllegalStateException(" The sorted indices need to have the same dimension as the feature input ");
		}
		this.sorted_indices=indices;
	}
	/**
	 * This keeps the sorted indices for each column
	 */
	private int maximum_ranks;

	public void set_ranked_scores (int indices){

		this.maximum_ranks=indices;
	}
	  /**
	   * This holds all the trees'nodes
	   */
	private RandomForestRegressor tree_body [] ;
	/**
	 * Number of trees to build
	 */
	public int estimators=10;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * weight on each estimator . Smaller values prevent overfitting. 
	 */
	//public double shrinkage=0.1;
	/**
	 * Trees in Random forest.
	 */
	public int trees=1;

	/**
	 * use samples with replacement or not
	 */
	public boolean bootsrap=false;
	
	/*************tree specific from here on *****************/
	/**
	 * maximum number of nodes allowed
	 */
	public double max_tree_size=-1;
	 
	  /**
	   * digits of rounding to prevent overfitting
	   */
	  public double rounding=6;
	/**
	 * offset for divisions
	 */
	public double offset=0.00001;
	/**
	 * maximum depth of the tree
	 */
	public double max_depth=3;
	/**
	 * Minimum gain to allow for a node to split
	 */
	public double gamma=1E-30;
	/**
	 * Minimum weighted sum to split a node
	 */
	public double min_split=2.0;
	/**
	 * Minimum weighted sum to keep a splitted node
	 */
	public double min_leaf=1.0;		
	/**
	 * Proportions of columns (features) to consider
	 */
	public double max_features=1.0;
	/**
	 * Proportions of columns (features) to consider
	 */
	public double feature_subselection=1.0;
	/**
	 * Proportions of best cut offs to consider
	 */
	public double cut_off_subsample=1.0;
	/**
	 * Proportions of best cut offs to consider
	 */
	public double row_subsample=1.0;	

	/**
	 * Rows to use
	 */
	private int rows [];
	
	public  RandomForestRegressor [] Get_tree(){
		if (this.tree_body==null || this.tree_body.length<=0){
			throw new IllegalStateException(" There is NO tree" );
		}
		return tree_body;
	}
	
	public double [] get_importances(){
		if (this.feature_importances==null || feature_importances.length<=0){
			throw new IllegalStateException(" There no importances (yet)" );
		}
		return feature_importances;
	}
	
	public void set_rows(int rows []){
		if (rows==null || rows.length<=0){
			throw new IllegalStateException(" The row indices are empty" );
		}
		this.rows=rows;
	}
	/**
	 * columns to use
	 */
	private int columns [];
	
	public void set_columns(int columns []){
		if (columns==null || columns.length<=0){
			throw new IllegalStateException(" The columns indices are empty" );
		}
		this.columns=columns;
	}
	/**
	 * Holds the rank of the 'zero' (e.g. sparse) elements
	 */
	private int zero_rank_holder [];
	public void set_zero_rank (int [] indices){

		this.zero_rank_holder=indices;
	}
	/**
	 * The objective to optimise in split . It may be RMSE 
	 *  , MAE or QUANTILE 
	 */
	public String Objective="RMSE";	
	/**
	 * Tau is not used so far
	 */
	public double tau=0.5;

	/**
	 * The coefficients of each classifier
	 */
	private double coeffs [];

	/**
	 * size of ensemble
	 */
	private  int ensemble_size=0;
	/**
	 * weight threshold to avoid early stopping
	 */
	public double weight_thresold=0.1;
	/**
	 * scale the copy the dataset
	 */
	public boolean copy=true;
	
    /**
     * seed to use
     */
	public int seed=1;
	/**
	 * Random number generator to use
	 */
	private Random random;
	/**
	 * weighst to used per row(sample)
	 */
	public double [] weights;
	/**
	 * if true, it prints stuff
	 */
	public boolean verbose=true;
	/**
	 * Target variable in double format
	 */
	public double target[];
	
	/**
	 * Target variable in 2d double format
	 */	
	public double target2d[][];
	/**
	 * Target variable in fixed-size matrix format
	 */	
	public fsmatrix fstarget;	
	/**
	 * Target variable in sparse matrix format
	 */	
	public smatrix starget;	
	/**
	 * Hold feature importance for the tree
	 */
	 double feature_importances [];
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
	}
	/**
	 * Number of target-variable columns. The name is left as n_classes(same as classification for consistency)
	 */
	private int n_classes=0;
	
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
	 * Default constructor for LinearRegression with no data
	 */
	public AdaboostForestRegressor(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public AdaboostForestRegressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public AdaboostForestRegressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public AdaboostForestRegressor(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
	}

	public void setdata(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}

	public void setdata(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}

	public void setdata(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
		}
		
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
	

	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8611561535854392960L;
	@Override
	public double[][] predict2d(double[][] data) {
		 
		/*  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		

		double predictions[][]= new double [data.length][this.n_classes];

		double shrinks[]= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
		Thread[] thread_array= new Thread[(this.ensemble_size<this.threads)?this.ensemble_size:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(this.ensemble_size <this.threads)?this.ensemble_size:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <this.ensemble_size;tree++ ){
				shrinks[count_of_live_threads]=this.coeffs[tree];	
				
				arrays[count_of_live_threads]=new fsmatrix(data.length,this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperfv2 (data, arrays[count_of_live_threads], tree_body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==tree_body.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
					
					//extract the values and see if we got a gamma better than required one
					for (int s=0; s <count_of_live_threads;s++ ){
						
						append_predictions_score(predictions,  arrays[s],shrinks[s]);
						
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(tree_body.length <this.threads)?tree_body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(tree_body.length <this.threads)?tree_body.length:this.threads];
			        shrinks= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
				}

			}	
		double sumes=DescriptiveStatistics.getSum(this.coeffs);
		for (int i=0; i < predictions.length; i++){
			for (int j=0; j < predictions[0].length; j++){
				predictions[i][j]/=sumes;
				
			}
		}
			
		return predictions;
			
			}

	@Override
	public double[][] predict2d(fsmatrix data) {
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		double shrinks[]= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
		Thread[] thread_array= new Thread[(this.ensemble_size<this.threads)?this.ensemble_size:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(this.ensemble_size <this.threads)?this.ensemble_size:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <this.ensemble_size;tree++ ){
				shrinks[count_of_live_threads]=this.coeffs[tree];	
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperfv2 (data, arrays[count_of_live_threads], tree_body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==tree_body.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
					
					//extract the values and see if we got a gamma better than required one
					for (int s=0; s <count_of_live_threads;s++ ){
						
						append_predictions_score(predictions,  arrays[s],shrinks[s]);
						
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(tree_body.length <this.threads)?tree_body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(tree_body.length <this.threads)?tree_body.length:this.threads];
			        shrinks= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
				}

			}	
			double sumes=DescriptiveStatistics.getSum(this.coeffs);
			for (int i=0; i < predictions.length; i++){
				for (int j=0; j < predictions[0].length; j++){
					predictions[i][j]/=sumes;
					
				}
			}
		return predictions;
		
			}

	@Override
	public double[][] predict2d(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		double shrinks[]= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
		Thread[] thread_array= new Thread[(this.ensemble_size<this.threads)?this.ensemble_size:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(this.ensemble_size <this.threads)?this.ensemble_size:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <this.ensemble_size;tree++ ){
				shrinks[count_of_live_threads]=this.coeffs[tree];	
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperfv2 (data, arrays[count_of_live_threads], tree_body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==tree_body.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
					
					//extract the values and see if we got a gamma better than required one
					for (int s=0; s <count_of_live_threads;s++ ){
						
						append_predictions_score(predictions,  arrays[s],shrinks[s]);
						
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(tree_body.length <this.threads)?tree_body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(tree_body.length <this.threads)?tree_body.length:this.threads];
			        shrinks= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
				}

			}	
			double sumes=DescriptiveStatistics.getSum(this.coeffs);
			for (int i=0; i < predictions.length; i++){
				for (int j=0; j < predictions[0].length; j++){
					predictions[i][j]/=sumes;
					
				}
			}
		return predictions;
	}

	@Override
	public double[] predict_Row2d(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	

		double predictions[]= new double [this.n_classes];
		
		for (int j=0; j < tree_body.length; j++){
			double newpredictions[]=tree_body[j].predict_Row2d(data);
				for (int s=0; s < newpredictions.length; s++){
					predictions[s]+=this.coeffs[j]*newpredictions[s];
				}
			
		}


		double sumes=DescriptiveStatistics.getSum(this.coeffs);
		for (int i=0; i < predictions.length; i++){
				predictions[i]/=sumes;
				
			
		}
			// return the 1st prediction
			return predictions;
			}


	@Override
	public double[] predict_Row2d(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	

		double predictions[]= new double [this.n_classes];
		
		for (int j=0; j < tree_body.length; j++){
			double newpredictions[]=tree_body[j].predict_Row2d(data,rows);
				for (int s=0; s < newpredictions.length; s++){
					predictions[s]+=this.coeffs[j]*newpredictions[s];
				}
			
		}

		double sumes=DescriptiveStatistics.getSum(this.coeffs);
		for (int i=0; i < predictions.length; i++){
				predictions[i]/=sumes;
				
			
		}


			// return the 1st prediction
			return predictions;
			
			
	}

	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		double predictions[]= new double [this.n_classes];
		
		for (int j=0; j < tree_body.length; j++){
			double newpredictions[]=tree_body[j].predict_Row2d(data,start,end);
				for (int s=0; s < newpredictions.length; s++){
					predictions[s]+=newpredictions[s]*this.coeffs[j];
				}
			
		}

		double sumes=DescriptiveStatistics.getSum(this.coeffs);
		for (int i=0; i < predictions.length; i++){
				predictions[i]/=sumes;
				
			
		}

			// return the 1st prediction
			return predictions;
			}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}	
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		
		double predictions[]= new double [data.GetRowDimension()];
		
		double shrinks[]= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
		Thread[] thread_array= new Thread[(this.ensemble_size<this.threads)?this.ensemble_size:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(this.ensemble_size <this.threads)?this.ensemble_size:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <this.ensemble_size;tree++ ){
				shrinks[count_of_live_threads]=this.coeffs[tree];	
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperfv2 (data, arrays[count_of_live_threads], tree_body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==tree_body.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
					
					//extract the values and see if we got a gamma better than required one
					for (int s=0; s <count_of_live_threads;s++ ){
						
						append_predictions_score(predictions,  arrays[s],shrinks[s]);
						
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(tree_body.length <this.threads)?tree_body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(tree_body.length <this.threads)?tree_body.length:this.threads];
			        shrinks= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
				}

			}	

			double sumes=DescriptiveStatistics.getSum(this.coeffs);
			for (int i=0; i < predictions.length; i++){
					predictions[i]/=sumes;
					
				
			}

			// return the 1st prediction
			return predictions;
			
			}
			

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}	
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

	double predictions[]= new double [data.GetRowDimension()];
		
		double shrinks[]= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
		Thread[] thread_array= new Thread[(this.ensemble_size<this.threads)?this.ensemble_size:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(this.ensemble_size <this.threads)?this.ensemble_size:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <this.ensemble_size;tree++ ){
				shrinks[count_of_live_threads]=this.coeffs[tree];	
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperfv2 (data, arrays[count_of_live_threads], tree_body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==tree_body.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
					
					//extract the values and see if we got a gamma better than required one
					for (int s=0; s <count_of_live_threads;s++ ){
						
						append_predictions_score(predictions,  arrays[s],shrinks[s]);
						
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(tree_body.length <this.threads)?tree_body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(tree_body.length <this.threads)?tree_body.length:this.threads];
			        shrinks= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
				}

			}	


			// return the 1st prediction
			return predictions;
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}	
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		
	double predictions[]= new double [data.length];
		
		double shrinks[]= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
		Thread[] thread_array= new Thread[(this.ensemble_size<this.threads)?this.ensemble_size:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(this.ensemble_size <this.threads)?this.ensemble_size:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <this.ensemble_size;tree++ ){
				shrinks[count_of_live_threads]=this.coeffs[tree];	
				
				arrays[count_of_live_threads]=new fsmatrix(data.length,this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperfv2 (data, arrays[count_of_live_threads], tree_body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==tree_body.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
					
					//extract the values and see if we got a gamma better than required one
					for (int s=0; s <count_of_live_threads;s++ ){
						
						append_predictions_score(predictions,  arrays[s],shrinks[s]);
						
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(tree_body.length <this.threads)?tree_body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(tree_body.length <this.threads)?tree_body.length:this.threads];
			        shrinks= new double [(this.ensemble_size<this.threads)?this.ensemble_size:this.threads];
				}

			}	

			double sumes=DescriptiveStatistics.getSum(this.coeffs);
			for (int i=0; i < predictions.length; i++){
					predictions[i]/=sumes;
					
				
			}
			// return the 1st prediction
			return predictions;
			
			}

	@Override
	public double predict_Row(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	


		double predictions= 0.0;
		
		
		for (int j=0; j < tree_body.length; j++){
			double newpredictions=tree_body[j].predict_Row(data);
					predictions+=newpredictions*this.coeffs[j];
			
		}

		double sumes=DescriptiveStatistics.getSum(this.coeffs);
				predictions/=sumes;
				
			
		
			
		

			// return the 1st prediction
			return predictions;
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	


		double predictions= 0.0;
		
		
		for (int j=0; j < tree_body.length; j++){
			double newpredictions=tree_body[j].predict_Row(data,rows);
					predictions+=newpredictions*this.coeffs[j];
			
		}


		double sumes=DescriptiveStatistics.getSum(this.coeffs);

				predictions/=sumes;


			// return the 1st prediction
			return predictions;
			}
			
	

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	

		double predictions= 0.0;
		
		
		for (int j=0; j < tree_body.length; j++){
			double newpredictions=tree_body[j].predict_Row(data,start,  end);
					predictions+=newpredictions*this.coeffs[j];
			
		}
		double sumes=DescriptiveStatistics.getSum(this.coeffs);


				predictions/=sumes;
				
			
		

			// return the 1st prediction
			return predictions;
			}

	
	
	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		dataset=data;
		
		if (max_tree_size<=0){
			max_tree_size=Double.MAX_VALUE;
		}		
		if (min_split<2){
			min_split=2;
		}
		if (min_leaf<1){
			min_leaf=1;
		}	
		if (max_features<=0){
			max_features=1;
		}			
		if (feature_subselection<=0){
			feature_subselection=1;
		}	
		if (this.offset<=0){
			this.offset=0.0000001;
		}
		if (cut_off_subsample<=0){
			cut_off_subsample=1;
		}			
		if (row_subsample<=0){
			row_subsample=1;
		}	
		if (weight_thresold<=0){
			weight_thresold=0.1;
		}	
		if (tau<=0 || tau>=1){
			tau=0.1;
		}	
		if (this.estimators<1){
			estimators=1;
		}
		if (this.trees<1){
			trees=1;
		}
		/*
		if ( this.shrinkage <=0){
			throw new IllegalStateException("Shrinkage cannot be negative");
		}
		*/
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}			
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.length) && (target2d==null || target2d.length!=data.length) && (fstarget==null || fstarget.GetRowDimension()!=data.length)  && (starget==null || starget.GetRowDimension()!=data.length)  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		}
		if (weights==null) {
			
			weights=new double [data.length];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/weights.length;
			}
			
		} else {
			if (weights.length!=data.length){
				throw new DimensionMismatchException(weights.length,data.length);
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			for (int i=0; i < weights.length; i++){
				weights[i]*= weights.length;
			}
		}

		//hard copy
		if (copy){
			data= manipulate.copies.copies.Copy( data);
		}
		// Initialise randomizer
		fsdataset=new fsmatrix(data);
		
		
		this.random = new XorShift128PlusRandom(this.seed);

		
		
		n_classes=0;
		if (target!=null){
			n_classes=1;
			fstarget=new fsmatrix(target,target.length,1);
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
			fstarget=new fsmatrix(target2d);
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
			fstarget=starget.ConvertToFixedSizeMatrix();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		

		/**
		 *  generate rows required by the algorithm
		 */
		
		columndimension=data[0].length;
		feature_importances= new double [columndimension];
		
		if (this.sorted_indices==null){
			this.sorted_indices=new int  [3] [];
			this.maximum_ranks= 0;
			this.zero_rank_holder= new int [this.columndimension];

		// create indices

			sortcolumnsnomap sorty= new sortcolumnsnomap (this.fsdataset,
														this.rows,
														this.sorted_indices,
														this.zero_rank_holder,
														this.rounding );
			sorty.verbose=this.verbose;
			sorty.merge_thresold=this.gamma;
			sorty.fstarget=this.fstarget;			
			sorty.run();
			this.maximum_ranks=	sorty.getmaxrank();
		}
		
		if (this.verbose){
			System.out.println("Sorting is done");
		}
		


		// Initialise the tree structure
		this.coeffs=new double [this.estimators];
		tree_body= new RandomForestRegressor[this.estimators];
		// start the loop to find the support vectors 
		int targetcolumns=n_classes;
		fsmatrix predictions=new fsmatrix(fstarget.GetRowDimension(), targetcolumns);
		double weight[]=new double [fstarget.GetRowDimension()];
		for (int i=0; i < weights.length; i++){
			weight[i]= weights[i];
		}
		ensemble_size=0;

		
		for (int n=0; n <this.estimators; n++ ){
			RandomForestRegressor model = new RandomForestRegressor(fsdataset);
			//general
			model.set_sorted_indices(this.sorted_indices);
			model.set_ranked_scores(this.maximum_ranks);
			model.set_zero_rank(this.zero_rank_holder);
			model.internal_threads=this.threads;
			model.verbose=false;
			
			model.estimators=this.trees;
			model.copy=false;
			model.cut_off_subsample=this.cut_off_subsample;
			model.feature_subselection=this.feature_subselection;
			if (this.rows!=null){
				model.set_rows(this.rows);
			}
			if (this.columns!=null){
				model.set_columns(this.columns);
			}
			model.offset=this.offset;
			model.rounding=this.rounding;
			model.gamma=this.gamma;
			model.max_depth=this.max_depth;
			model.max_features=this.max_features;
			model.max_tree_size=-1;
			model.bootsrap=this.bootsrap;
			model.min_leaf=this.min_leaf;
			model.min_split=this.min_split;
			model.Objective=this.Objective;
			model.row_subsample=this.row_subsample;
			model.seed=this.seed+ n;
			model.weights=this.weights;
			model.fstarget=this.fstarget;
			tree_body[n]=model;	

			model.fit(fsdataset);
			ensemble_size+=1;
			
			predictions=tree_body[n].predictfs(fsdataset);
			
			double error=0.0;
			//iterate through predictions
			double errors[]= new double [predictions.GetRowDimension()];
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
				for (int j=0; j<targetcolumns; j++ ){
						error+=  Math.abs( fstarget.GetElement(m, j)-predictions.GetElement(m, j)  );
						
				}
				errors[m]=error;
				}
			double val_thresold=DescriptiveStatistics.getPercentile(errors, this.weight_thresold, false);
			weight=new double[this.weights.length];
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
				weight[m]=this.weight_thresold;
			}
			
			double adjust=1/val_thresold;
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
								if ( errors[m] > val_thresold){
									weight[m]+= errors[m] *adjust;
									
								}
				}
		    
			// if error too big or too small , stop.
			
		if (verbose) {
			System.out.println(" Iteration: "+ n+ " with error: " + error + " and adjust: " + adjust );
		}	
		

		
			//Assign new coefficient
	   		coeffs[n] =adjust;//*shrinkage;
		
			
			
		
			
			
		}
		for (int i=0; i <ensemble_size; i++){
			double importances[]=tree_body[i].get_importances();
			for (int j=0; j < importances.length; j++){
				feature_importances[j]+=importances[j];
			}
		}
		

		double sum_importances=get_sum(this.feature_importances);
		for (int i=0; i <feature_importances.length; i++ ){
			feature_importances[i]/=sum_importances;
			
		}
		dataset=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		
	}
	@Override
	public void fit(fsmatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		fsdataset=data;
		
		if (this.offset<=0){
			this.offset=0.0000001;
		}
		if (max_tree_size<=0){
			max_tree_size=Double.MAX_VALUE;
		}
				
		if (min_split<2){
			min_split=2;
		}
		if (min_leaf<1){
			min_leaf=1;
		}	
		if (max_features<=0){
			max_features=1;
		}			
		if (feature_subselection<=0){
			feature_subselection=1;
		}		
		if (cut_off_subsample<=0){
			cut_off_subsample=1;
		}			
		if (row_subsample<=0){
			row_subsample=1;
		}	
		if (this.estimators<1){
			estimators=1;
		}
		if (this.trees<1){
			trees=1;
		}
		if (weight_thresold<=0 ){
			weight_thresold=0.1;
		}	
		if (tau<=0 || tau>=1){
			tau=0.1;
		}
		/*
		if ( this.shrinkage <=0){
			throw new IllegalStateException("Shrinkage cannot be negative");
		}
		*/
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}			
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (target2d==null || target2d.length!=data.GetRowDimension()) && (fstarget==null || fstarget.GetRowDimension()!=data.GetRowDimension())  && (starget==null || starget.GetRowDimension()!=data.GetRowDimension())  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		}
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/weights.length;
			}
			
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			for (int i=0; i < weights.length; i++){
				weights[i]*= weights.length;
			}
		}

		//hard copy
		if (copy){
			data= (fsmatrix)( data.Copy());
		}
		// Initialise randomizer

		
		n_classes=0;
		if (target!=null){
			n_classes=1;
			fstarget=new fsmatrix(target,target.length,1);
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
			fstarget=new fsmatrix(target2d);
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
			fstarget=starget.ConvertToFixedSizeMatrix();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		
		

		/**
		 *  generate rows required by the algorithm
		 */
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		columndimension=data.GetColumnDimension();
		feature_importances= new double [columndimension];
		if (this.sorted_indices==null){
			this.sorted_indices=new int  [3] [];
			this.maximum_ranks= 0;
			this.zero_rank_holder= new int [this.columndimension];

		// create indices

			sortcolumnsnomap sorty= new sortcolumnsnomap (this.fsdataset,
														this.rows,
														this.sorted_indices,
														this.zero_rank_holder,
														this.rounding );
			sorty.run();
			this.maximum_ranks=	sorty.getmaxrank();
		}
		if (this.verbose){
			System.out.println("Sorting is done");
		}
				

		// Initialise the tree structure
		this.coeffs=new double [this.estimators];
		tree_body= new RandomForestRegressor[this.estimators];
		// start the loop to find the support vectors 
		int targetcolumns=n_classes;
		fsmatrix predictions=new fsmatrix(fstarget.GetRowDimension(), targetcolumns);
		
		double weight[]=new double [fstarget.GetRowDimension()];
		for (int i=0; i < weights.length; i++){
			weight[i]= weights[i];
		}
		ensemble_size=0;

		
		for (int n=0; n <this.estimators; n++ ){
			RandomForestRegressor model = new RandomForestRegressor(fsdataset);
			//general
			model.set_sorted_indices(this.sorted_indices);
			model.set_ranked_scores(this.maximum_ranks);
			model.set_zero_rank(this.zero_rank_holder);
			model.internal_threads=this.threads;
			model.verbose=false;
			model.rounding=this.rounding;
			model.estimators=this.trees;
			model.copy=false;
			model.cut_off_subsample=this.cut_off_subsample;
			model.feature_subselection=this.feature_subselection;
			if (this.rows!=null){
				model.set_rows(this.rows);
			}
			if (this.columns!=null){
				model.set_columns(this.columns);
			}
			model.offset=this.offset;
			model.gamma=this.gamma;
			model.max_depth=this.max_depth;
			model.max_features=this.max_features;
			model.max_tree_size=-1;
			model.bootsrap=this.bootsrap;
			model.min_leaf=this.min_leaf;
			model.min_split=this.min_split;
			model.Objective=this.Objective;
			model.row_subsample=this.row_subsample;
			model.seed=this.seed+ n;
			model.weights=this.weights;
			model.fstarget=this.fstarget;
			tree_body[n]=model;	

			model.fit(fsdataset);
			ensemble_size+=1;
			
			predictions=tree_body[n].predictfs(fsdataset);
			
			double error=0.0;
			//iterate through predictions
			double errors[]= new double [predictions.GetRowDimension()];
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
				for (int j=0; j<targetcolumns; j++ ){
						error+=  Math.abs( fstarget.GetElement(m, j)-predictions.GetElement(m, j)  );
						
				}
				errors[m]=error;
				}
			double val_thresold=DescriptiveStatistics.getPercentile(errors, this.weight_thresold, false);
			weight=new double[this.weights.length];
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
				weight[m]=this.weight_thresold;
			}
			
			double adjust=1/val_thresold;
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
								if ( errors[m] > val_thresold){
									weight[m]+= errors[m] *adjust;
									
								}
				}
		    
			// if error too big or too small , stop.
			
		if (verbose) {
			System.out.println(" Iteration: "+ n+ " with error: " + error + " and adjust: " + adjust );
		}	
		

		
			//Assign new coefficient
	   		coeffs[n] =adjust;//*shrinkage;
		
			
			
		}

		for (int i=0; i <ensemble_size; i++){
			double importances[]=tree_body[i].get_importances();
			for (int j=0; j < importances.length; j++){
				feature_importances[j]+=importances[j];
			}
		}
		

		double sum_importances=get_sum(this.feature_importances);
		for (int i=0; i <feature_importances.length; i++ ){
			feature_importances[i]/=sum_importances;
			
		}
		fsdataset=null;
		sdataset=null;
		
		System.gc();

		
	}
	
	@Override
	public void fit(smatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		sdataset=data;
		if (this.offset<=0){
			this.offset=0.0000001;
		}
		if (max_tree_size<=0){
			max_tree_size=Double.MAX_VALUE;
		}
				
		if (min_split<2){
			min_split=2;
		}
		if (min_leaf<1){
			min_leaf=1;
		}	
		if (max_features<=0){
			max_features=1;
		}			
		if (feature_subselection<=0){
			feature_subselection=1;
		}		
		if (cut_off_subsample<=0){
			cut_off_subsample=1;
		}			
		if (row_subsample<=0){
			row_subsample=1;
		}	
		if (weight_thresold<=0 ){
			weight_thresold=0.1;
		}	
		if (tau<=0 || tau>=1){
			tau=0.1;
		}
		if (this.estimators<1){
			estimators=1;
		}
		if (this.trees<1){
			trees=1;
		}
		/*
		if ( this.shrinkage <=0){
			throw new IllegalStateException("Shrinkage cannot be negative");
		}
		*/
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}			
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (target2d==null || target2d.length!=data.GetRowDimension()) && (fstarget==null || fstarget.GetRowDimension()!=data.GetRowDimension())  && (starget==null || starget.GetRowDimension()!=data.GetRowDimension())  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		}
		if (weights==null) {
			
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/weights.length;
			}
			
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			for (int i=0; i < weights.length; i++){
				weights[i]*= weights.length;
			}
		}

		//hard copy
		if (copy){
			data= (smatrix)( data.Copy());
		}
		// Initialise randomizer


		
		n_classes=0;
		if (target!=null){
			n_classes=1;
			fstarget=new fsmatrix(target,target.length,1);
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
			fstarget=new fsmatrix(target2d);
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
			fstarget=starget.ConvertToFixedSizeMatrix();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		sdataset.trim();
		
		if (!this.sdataset.IsSortedByRow()){
			this.sdataset.convert_type();
			//System.out.println("built sort");
			}

		/**
		 *  generate rows required by the algorithm
		 */
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		columndimension=data.GetColumnDimension();
		feature_importances= new double [columndimension];
		if (this.sorted_indices==null){
			this.sorted_indices=new int  [3] [];
			this.maximum_ranks= 0;
			this.zero_rank_holder= new int [this.columndimension];

		// create indices

			sortcolumnsnomap sorty= new sortcolumnsnomap (this.sdataset,
														this.rows,
														this.sorted_indices,
														this.zero_rank_holder,
														this.rounding );
			sorty.run();
			this.maximum_ranks=	sorty.getmaxrank();
		}
		if (this.verbose){
			System.out.println("Sorting is done");
		}
						
	
		
		// Initialise the tree structure
		this.coeffs=new double [this.estimators];
		tree_body= new RandomForestRegressor[this.estimators];
		// start the loop to find the support vectors 
		int targetcolumns=n_classes;
		fsmatrix predictions=new fsmatrix(fstarget.GetRowDimension(), targetcolumns);
		double weight[]=new double [fstarget.GetRowDimension()];
		for (int i=0; i < weights.length; i++){
			weight[i]= weights[i];
		}
		ensemble_size=0;

		
		for (int n=0; n <this.estimators; n++ ){
			RandomForestRegressor model = new RandomForestRegressor(sdataset);
			//general
			model.set_sorted_indices(this.sorted_indices);
			model.set_ranked_scores(this.maximum_ranks);
			model.set_zero_rank(this.zero_rank_holder);
			model.internal_threads=this.threads;
			model.verbose=false;
			model.rounding=this.rounding;
			model.estimators=this.trees;
			model.copy=false;
			model.cut_off_subsample=this.cut_off_subsample;
			model.feature_subselection=this.feature_subselection;
			if (this.rows!=null){
				model.set_rows(this.rows);
			}
			if (this.columns!=null){
				model.set_columns(this.columns);
			}
			model.offset=this.offset;
			model.gamma=this.gamma;
			model.max_depth=this.max_depth;
			model.max_features=this.max_features;
			model.max_tree_size=-1;
			model.min_leaf=this.min_leaf;
			model.min_split=this.min_split;
			model.Objective=this.Objective;
			model.row_subsample=this.row_subsample;
			model.seed=this.seed+ n;
			model.bootsrap=this.bootsrap;
			model.weights=this.weights;
			model.fstarget=this.fstarget;
			tree_body[n]=model;	

			model.fit(sdataset);
			ensemble_size+=1;
			
			predictions=tree_body[n].predictfs(sdataset);
			
			double error=0.0;
			//iterate through predictions
			double errors[]= new double [predictions.GetRowDimension()];
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
				for (int j=0; j<targetcolumns; j++ ){
						error+=  Math.abs( fstarget.GetElement(m, j)-predictions.GetElement(m, j)  );
						
				}
				errors[m]=error;
				}
			double val_thresold=DescriptiveStatistics.getPercentile(errors, this.weight_thresold, false);
			weight=new double[this.weights.length];
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
				weight[m]=this.weight_thresold;
			}
			
			double adjust=1/val_thresold;
			for (int m=0; m <predictions.GetRowDimension(); m++ ){
								if ( errors[m] > val_thresold){
									weight[m]+= errors[m] *adjust;
									
								}
				}
		    
			// if error too big or too small , stop.
			
		if (verbose) {
			System.out.println(" Iteration: "+ n+ " with error: " + error + " and adjust: " + adjust );
		}	
		

		
			//Assign new coefficient
	   		coeffs[n] =adjust;//*shrinkage;
		
			
			
		
			
			
		}
		

		for (int i=0; i <ensemble_size; i++){
			double importances[]=tree_body[i].get_importances();
			for (int j=0; j < importances.length; j++){
				feature_importances[j]+=importances[j];
			}
		}
		

		double sum_importances=get_sum(this.feature_importances);
		for (int i=0; i <feature_importances.length; i++ ){
			feature_importances[i]/=sum_importances;
			
		}

		sdataset=null;
		System.gc();

		

		
		// calculate first node
			
	}
  
	/**
	 * Retrieve the number of target variables
	 */
	public int getnumber_of_targets(){
		return n_classes;
	}
	
	
	public double get_sum(double array []){
		double a=0.0;
		for (int i=0; i <array.length; i++ ){
			a+=array[i];
		}
		return a;
	}
	
	/**
	 * 
	 * @returns the closest integer that reflects this percentage!
	 * <p> it may sound strange, random.nextint can be significantly faster than nextdouble()
	 */
	public int get_random_integer(double percentage){
		
		double per= Math.min(Math.max(0, percentage),1.0);
		double difference=2147483647.0+(2147483648.0);
		int point=(int)(-2147483648.0 +  (per*difference ));
		
		
		return point;
		
	}

	@Override
	public String GetType() {
	
		return "regressor";
	}

	@Override
	public boolean SupportsWeights() {
		return true;
	}

	@Override
	public String GetName() {
		return "AdaboostForestRegressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor: AdaboostForestRegressor");
		System.out.println("Targets: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		System.out.println("Estimators: " + this.estimators);		
		System.out.println("Bootsrapping: " + this.bootsrap);		
		System.out.println("cut_off_subsample: "+ this.cut_off_subsample);
		System.out.println("Objective: "+ this.Objective);
		System.out.println("tau: "+ this.tau);
		System.out.println("feature_subselection: "+ this.feature_subselection);
		System.out.println("gamma: "+ this.gamma);	
		System.out.println("offset: "+ this.offset);		
		System.out.println("rounding: "+ this.rounding);
		System.out.println("trees: "+ this.trees);		
		//System.out.println("Shrinkage: "+ this.shrinkage);				
		System.out.println("max_depth: "+ this.max_depth);			
		System.out.println("max_features : "+ this.max_features);	
		System.out.println("max_tree_size : "+ this.max_tree_size);
		System.out.println("min_leaf : "+ this.min_leaf);	
		System.out.println("min_leaf : "+ this.min_split);	
		System.out.println("row_subsample : "+ this.row_subsample);			
		System.out.println("threads : "+ this.threads);			
		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (this.tree_body==null){
			System.out.println("Trained: False");	
		} else {
			System.out.println("Trained: True");
		}
		
	}
	
	@Override
	public double [][] predict_proba(double data [][]){
		return predict2d(data);
	}
	@Override
	public double [][] predict_proba(fsmatrix f){
		return predict2d( f);
	}
	@Override
	public double [][] predict_proba(smatrix f){
		return predict2d( f);
	}
	
	@Override	
	public void set_params(String params){
		
		String splitted_params []=params.split(" " + "+");
		
		for (int j=0; j<splitted_params.length; j++ ){
			String mini_split []=splitted_params[j].split(":");
			if (mini_split.length>=2){
				String metric=mini_split[0];
				String value=mini_split[1];
				
				if (metric.equals("cut_off_subsample")) {this.cut_off_subsample=Double.parseDouble(value);}
				else if (metric.equals("feature_subselection")) {this.feature_subselection=Double.parseDouble(value);}
				else if (metric.equals("row_subsample")) {this.row_subsample=Double.parseDouble(value);}	
				else if (metric.equals("estimators")) {this.estimators=Integer.parseInt(value);}
				else if (metric.equals("min_leaf")) {this.min_leaf=Double.parseDouble(value);}				
				else if (metric.equals("trees")) {this.trees=Integer.parseInt(value);}				
				else if (metric.equals("max_depth")) {this.max_depth=Integer.parseInt(value);}
				else if (metric.equals("Objective")) {this.Objective=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("rounding")) {this.rounding=Double.parseDouble(value);}	
				else if (metric.equals("weight_thresold")) {this.weight_thresold=Double.parseDouble(value);}				
				else if (metric.equals("offset")) {this.offset=Double.parseDouble(value);}						
				else if (metric.equals("max_tree_size")) {this.max_tree_size=Integer.parseInt(value);}
				else if (metric.equals("gamma")) {this.gamma=Double.parseDouble(value);}
				else if (metric.equals("max_features")) {this.max_features=Double.parseDouble(value);}
				else if (metric.equals("bootsrap")) {this.bootsrap=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("min_split")) {this.min_split=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}			
				
			}
			
		}
		

	}

	@Override
	public boolean HasTheSametype(estimator a) {
		if (a.GetType().equals(this.GetType())){
			return true;
		} else {
		return false;
		}
	}

	@Override
	public boolean isfitted() {
		if (this.tree_body!=null || tree_body.length>0){
			return true;
		} else {
		return false;
		}
	}

	@Override
	public boolean IsRegressor() {
		return true ;
	}

	@Override
	public boolean IsClassifier() {
		return false;
	}

	@Override
	public void reset() {
		this.tree_body= null;
		n_classes=0;
		tau=0.5;
		Objective="RMSE";
		threads=1;
		rounding=30;
		offset=0.0001;
		this.estimators=10;
		this.bootsrap=false;
		//this.shrinkage=0.1;
		this.columns=null;
		this.random=null;
		this.trees=1;
		this.cut_off_subsample=1.0;
		this.feature_importances.clone();
		this.feature_subselection=1.0;
		this.gamma=1E-30;
		this.max_depth=3;
		this.max_features=1.0;
		this.max_tree_size=-1;
		this.min_leaf=1.0;
		this.min_split=2.0;
		this.row_subsample=1.0;
		columndimension=0;
		copy=true;
		seed=1;
		random=null;
		target=null;
		target2d=null;
		fstarget=null;
		starget=null;
		weights=null;
		verbose=true;
		
	}

	@Override
	public estimator copy() {
		AdaboostForestRegressor br = new AdaboostForestRegressor();
		RandomForestRegressor[] tree_bodys= new RandomForestRegressor[this.tree_body.length];
        for (int i=0; i <tree_bodys.length; i++ ){
        	tree_bodys[i]=(RandomForestRegressor) tree_body[i].copy();
        }
        br.tree_body=tree_bodys;
        br.estimators=this.estimators;
        br.bootsrap=this.bootsrap;
        //br.shrinkage=this.shrinkage;
        br.trees=this.trees;
		br.n_classes=this.n_classes;
		br.columns=this.columns.clone();
		br.offset=this.offset;
		br.rounding=this.rounding;
		br.cut_off_subsample=this.cut_off_subsample;
		br.tau= this.tau;
		br.feature_importances=this.feature_importances.clone();
		br.feature_subselection=this.feature_subselection;
		br.gamma=this.gamma;
		br.max_depth=this.max_depth;
		br.max_features=this.max_features;
		br.max_tree_size=this.max_tree_size;
		br.Objective=this.Objective;
		br.min_leaf=this.min_leaf;
		br.min_split=this.min_split;
		br.row_subsample=this.row_subsample;
		br.threads=this.threads;
		br.columndimension=this.columndimension;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.target2d=manipulate.copies.copies.Copy(this.target2d.clone());	
		br.fstarget=(fsmatrix) this.fstarget.Copy();
		br.starget=(smatrix) this.starget.Copy();
		br.weights=manipulate.copies.copies.Copy(this.weights.clone());
		br.verbose=this.verbose;
		return br;
	}

	@Override
	public scaler ReturnScaler() {
		return null;
	}
	@Override
	public void setScaler(scaler sc) {

	}
	@Override
	public void setSeed(int seed) {
		this.seed=seed;}	
	
	@Override	
	public void set_target(double data []){
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		this.target=data;
	}
	/**
	 * 
	 * @param previous_predictions : Previous predictions 
	 * @param new_predictions : New predictions to be added to the new ones
	 */
		public void append_predictions_score(double previous_predictions [][],  fsmatrix new_predictions , double shrink){
			
			if (new_predictions.GetColumnDimension()==1){		
				for (int i=0; i <previous_predictions.length; i++ ){
							previous_predictions[i][0]+= new_predictions.GetElement(i, 0)*shrink;			
					
				}
				
			}else {
			
				for (int i=0; i <previous_predictions.length; i++ ){
					for (int j=0; j <previous_predictions[0].length; j++ ){
						previous_predictions[i][j]+= new_predictions.GetElement(i, j)*shrink;

					}
					

		
				}
				
			}
		}

		/**
		 * 
		 * @param previous_predictions : Previous predictions 
		 * @param new_predictions : New predictions to be added to the new ones
		 */
			public void append_predictions_score(double previous_predictions [],  fsmatrix new_predictions , double shrink){
					
					for (int i=0; i <previous_predictions.length; i++ ){
								previous_predictions[i]+= new_predictions.GetElement(i, 0)*shrink;			
						
					
					
				}
			}
			
			@Override
			public int getSeed() {
				return this.seed;}
			
			@Override
			public void AddClassnames(String names[]){
				//none
			}
			@Override
			public void set_target(fsmatrix fstarget){
				this.fstarget=fstarget;
			}
}

	  

