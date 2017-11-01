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

package ml.Bagging;
import java.util.Random;

import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import exceptions.DimensionMismatchException;
import exceptions.IllegalStateException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;
import ml.Kernel.copy.KernelmodelRegressor;
import ml.LSVR.LSVR;
import ml.LibFm.LibFmRegressor;
import ml.LibFm.OriginalLibFMRegressor;
import ml.LinearRegression.LinearRegression;
import ml.Tree.AdaboostForestRegressor;
import ml.Tree.DecisionTreeRegressor;
import ml.Tree.GradientBoostingForestRegressor;
import ml.Tree.RandomForestRegressor;
import ml.fastrgf.FRGFRegressor;
import ml.h2o.H2ODeepLearningRegressor;
import ml.h2o.H2ODrfRegressor;
import ml.h2o.H2OGbmRegressor;
import ml.h2o.H2OGlmRegressor;
import ml.knn.knnRegressor;
import ml.lightgbm.LightgbmRegressor;
import ml.nn.Vanilla2hnnregressor;
import ml.nn.multinnregressor;
import ml.python.PythonGenericRegressor;
import ml.python.keras.KerasnnRegressor;
import ml.python.sklearn.SklearnAdaBoostRegressor;
import ml.python.sklearn.SklearnDecisionTreeRegressor;
import ml.python.sklearn.SklearnExtraTreesRegressor;
import ml.python.sklearn.SklearnMLPRegressor;
import ml.python.sklearn.SklearnRandomForestRegressor;
import ml.python.sklearn.SklearnSGDRegressor;
import ml.python.sklearn.SklearnknnRegressor;
import ml.python.sklearn.SklearnsvmRegressor;
import ml.stacknet.StackNetClassifier;
import ml.vowpalwabbit.VowpaLWabbitRegressor;
import ml.xgboost.XgboostRegressor;

/**
 * <p> Regressor Base for bagging

 */

public class BaggingRegressor implements estimator,regressor {

	  /**
	   * This holds all the classifiers
	   */
	private regressor body [] ;
	/**
	 * Number of bags to build
	 */
	public int estimators=3;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * This is String defines the model and each parameters - for example : <b>LinearRegression</b> <em>Type:Routine C:0.01 threads:1 usescale:True seed:1 verbose:false</em>
	 */
	private String model_parameters="";
	

	public  regressor [] Get_tree(){
		if (this.body==null || this.body.length<=0){
			throw new IllegalStateException(" There is NO tree" );
		}
		return body;
	}
	
	/**
	 * 
	 * @param params : model parameters
	 */
	public void set_model_parameters(String params){
		//check if parameters are valid
		if (StackNetClassifier.containsRegressor( params)){
			model_parameters=params;
		} else {
			throw new IllegalStateException(" no recognised regressor in " + params );
		}
	}	
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
	 * Default constructor for BaggingRegressor with no data
	 */
	public BaggingRegressor(){
	
	}	
	/**
	 * Default constructor for BaggingRegressor with double data
	 */
	public BaggingRegressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for BaggingRegressor with fsmatrix data
	 */
	public BaggingRegressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for BaggingRegressor with smatrix data
	 */
	public BaggingRegressor(smatrix data){
		
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
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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

		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.length,this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.length; i++){
							for (int j=0; j < predictions[0].length; j++){
								predictions[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.length; i++){
				for (int s=0; s < predictions[0].length; s++){
					predictions[i][s]/=body.length;
				}
			}

			// return the 1st prediction
			return predictions;
			
			}

	@Override
	public double[][] predict2d(fsmatrix data) {
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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
		
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.length; i++){
							for (int j=0; j < predictions[0].length; j++){
								predictions[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.length; i++){
				for (int s=0; s < predictions[0].length; s++){
					predictions[i][s]/=body.length;
				}
			}


		return predictions;
		
			}

	
	public fsmatrix predictfs(fsmatrix data) {
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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
		int oldthread=this.threads;
		this.threads=1;
		
		fsmatrix predictions= new fsmatrix (data.GetRowDimension(),this.n_classes);
		
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.data.length; i++){
							
							predictions.data[i]+=arrays[s].data[i];
							
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.data.length; i++){
				predictions.data[i]/=body.length;
				//System.out.println(predictions.data[i]);
				
			}
		
			this.threads=oldthread;

		return predictions;
		
			}
	public fsmatrix predictfs(smatrix data) {
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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
		int oldthread=this.threads;
		this.threads=1;
		
		fsmatrix predictions= new fsmatrix (data.GetRowDimension(),this.n_classes);
		
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.data.length; i++){
							
							predictions.data[i]+=arrays[s].data[i];
							
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.data.length; i++){
				predictions.data[i]/=body.length;
				
			}
		
			this.threads=oldthread;

		return predictions;
		
			}
	
	public fsmatrix predictfs(double [][] data) {
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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
		int oldthread=this.threads;
		this.threads=1;
		
		fsmatrix predictions= new fsmatrix (data.length,this.n_classes);
		
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.length,this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.data.length; i++){
							
							predictions.data[i]+=arrays[s].data[i];
							
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.data.length; i++){
				predictions.data[i]/=body.length;
				
			}
		
			this.threads=oldthread;

		return predictions;
		
			}
	@Override
	public double[][] predict2d(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.length; i++){
							for (int j=0; j < predictions[0].length; j++){
								predictions[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.length; i++){
				for (int s=0; s < predictions[0].length; s++){
					predictions[i][s]/=body.length;
				}
			}

			// return the 1st prediction
			return predictions;
	}

	@Override
	public double[] predict_Row2d(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.body==null || this.body.length<=0  ){
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
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_Row2d(data);
				for (int s=0; s < newpredictions.length; s++){
					predictions[s]+=newpredictions[s];
				}
			
		}


			for (int s=0; s < predictions.length; s++){
				predictions[s]/=body.length;
			
		}

			// return the 1st prediction
			return predictions;
			}


	@Override
	public double[] predict_Row2d(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.body==null || this.body.length<=0  ){
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
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_Row2d(data,rows);
				for (int s=0; s < newpredictions.length; s++){
					predictions[s]+=newpredictions[s];
				}
			
		}


			for (int s=0; s < predictions.length; s++){
				predictions[s]/=body.length;
			
		}

			// return the 1st prediction
			return predictions;
			
			
	}

	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.body==null || this.body.length<=0  ){
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
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_Row2d(data,start,end);
				for (int s=0; s < newpredictions.length; s++){
					predictions[s]+=newpredictions[s];
				}
			
		}


			for (int s=0; s < predictions.length; s++){
				predictions[s]/=body.length;
			
		}

			// return the 1st prediction
			return predictions;
			}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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
		
		double predictionss[]= new double [data.GetRowDimension()];
		
		
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.length; i++){
							for (int j=0; j < predictions[0].length; j++){
								predictions[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.length; i++){
					predictionss[i]=predictions[i][0]/body.length;
				
			}
			predictions=null;
			// return the 1st prediction
			return predictionss;
			
			}
			

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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

		double predictionss[]= new double [data.GetRowDimension()];
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.length; i++){
							for (int j=0; j < predictions[0].length; j++){
								predictions[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.length; i++){
					predictionss[i]=predictions[i][0]/body.length;
				
			}
			predictions=null;
			// return the 1st prediction
			return predictionss;
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.body==null || this.body.length<=0 ){
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
		double predictionss[]= new double [data.length];
		double predictions[][]= new double [data.length][this.n_classes];
		
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.length,this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelperbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
				thread_array[count_of_live_threads].start();
				
				count_of_live_threads++;
				if (count_of_live_threads==threads || tree==body.length-1){
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
						
						for (int i=0; i < predictions.length; i++){
							for (int j=0; j < predictions[0].length; j++){
								predictions[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		
			for (int i=0; i < predictions.length; i++){
					predictionss[i]=predictions[i][0]/body.length;
				
			}
			predictions=null;
			// return the 1st prediction
			return predictionss;
			
			}

	@Override
	public double predict_Row(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.body==null || this.body.length<=0  ){
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
		
		
		for (int j=0; j < body.length; j++){
			double newpredictions=body[j].predict_Row(data);
					predictions+=newpredictions;
			
		}

				predictions/=body.length;
			
		

			// return the 1st prediction
			return predictions;
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.body==null || this.body.length<=0  ){
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
		
		
		for (int j=0; j < body.length; j++){
			double newpredictions=body[j].predict_Row(data,rows);
					predictions+=newpredictions;
			
		}

				predictions/=body.length;
			
		

			// return the 1st prediction
			return predictions;
			}
			
	

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.body==null || this.body.length<=0  ){
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
		
		
		for (int j=0; j < body.length; j++){
			double newpredictions=body[j].predict_Row(data,start,  end);
					predictions+=newpredictions;
			
		}

				predictions/=body.length;
			
		

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
			
		if (this.model_parameters.equals("")){
			throw new IllegalStateException(" Valid model parameters need to be set prior to fitting the method" );
		}
		
		if (this.estimators<1){
			estimators=1;
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
			/*
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
			*/
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
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		
		columndimension=data[0].length;

		// Initialise the tree structure

		Thread[] thread_array= new Thread[this.threads];
		body= new regressor[this.estimators];
		// start the loop to find the support vectors 

		int count_of_live_threads=0;
		for (int n=0; n <this.estimators; n++ ){
			
			regressor model =null;
			if (model_parameters.contains("AdaboostForestRegressor")) {
				model= new AdaboostForestRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnRandomForestRegressor")) {
			model= new SklearnRandomForestRegressor(fsdataset);					
			}else if (model_parameters.contains("DecisionTreeRegressor")) {
				model= new DecisionTreeRegressor(fsdataset);
			}else if (model_parameters.contains("GradientBoostingForestRegressor")) {
				model= new GradientBoostingForestRegressor(fsdataset);
			}else if (model_parameters.contains("RandomForestRegressor")) {
				model= new RandomForestRegressor(fsdataset);
			}else if (model_parameters.contains("Vanilla2hnnregressor")) {
				model= new Vanilla2hnnregressor(fsdataset);
			}else if (model_parameters.contains("multinnregressor")) {
				model= new multinnregressor(fsdataset);
			}else if (model_parameters.contains("LSVR")) {
				model= new LSVR(fsdataset);
			}else if (model_parameters.contains("LinearRegression")) {
				model= new LinearRegression(fsdataset);
			}else if (model_parameters.contains("OriginalLibFMRegressor")) {
				model= new OriginalLibFMRegressor(fsdataset);						
			}else if (model_parameters.contains("LibFmRegressor")) {
				model= new LibFmRegressor(fsdataset);
			}else if (model_parameters.contains("knnRegressor")) {
				model= new knnRegressor(fsdataset);
			}else if (model_parameters.contains("KernelmodelRegressor")) {
				model= new KernelmodelRegressor(fsdataset);
			}else if (model_parameters.contains("XgboostRegressor")) {
				model= new XgboostRegressor(fsdataset);									
			}else if (model_parameters.contains("LightgbmRegressor")) {
			model= new LightgbmRegressor(fsdataset);							
			}else if (model_parameters.contains("H2OGbmRegressor")) {
			model= new H2OGbmRegressor(fsdataset);									
			}else if (model_parameters.contains("H2ODeepLearningRegressor")) {
			model= new H2ODeepLearningRegressor(fsdataset);	
			}else if (model_parameters.contains("H2OGlmRegressor")) {
			model= new H2OGlmRegressor(fsdataset);									
			}else if (model_parameters.contains("H2ODrfRegressor")) {
			model= new H2ODrfRegressor(fsdataset);				
			}else if (model_parameters.contains("FRGFRegressor")) {
			model= new FRGFRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnAdaBoostRegressor")) {
				model= new SklearnAdaBoostRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnDecisionTreeRegressor")) {
				model= new SklearnDecisionTreeRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnExtraTreesRegressor")) {
				model= new SklearnExtraTreesRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnknnRegressor")) {
				model= new SklearnknnRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnMLPRegressor")) {
				model= new SklearnMLPRegressor(fsdataset);														
			}else if (model_parameters.contains("SklearnSGDRegressor")) {
			model= new SklearnSGDRegressor(fsdataset);									
			}else if (model_parameters.contains("SklearnsvmRegressor")) {
			model= new SklearnsvmRegressor(fsdataset);	
			}else if (model_parameters.contains("KerasnnRegressor")) {
			model= new KerasnnRegressor(fsdataset);									
			}else if (model_parameters.contains("PythonGenericRegressor")) {
			model= new PythonGenericRegressor(fsdataset);				
			}else if (model_parameters.contains("VowpaLWabbitRegressor")) {
			model= new VowpaLWabbitRegressor(fsdataset);
			
			
			
			
			
			
			
			} else {
				throw new IllegalStateException(" The selected model '" + model_parameters + " is not recognizable as valid classifier" );
			}
			
			model.set_params(this.model_parameters);
			model.setSeed(model.getSeed() + this.seed +  n);
			model.set_target(this.fstarget);
			body[n]=model;
			
					
				thread_array[count_of_live_threads]= new Thread(model);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting batch Tree: " + n);
					
				}				
				if (count_of_live_threads==threads || n==this.estimators-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {

							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
				
					

					count_of_live_threads=0;
				}
				
				
			}

		


		fstarget=null;
		fsdataset=null;
		dataset=null;
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
		
		if (this.model_parameters.equals("")){
			throw new IllegalStateException(" Valid model parameters need to be set prior to fitting the method" );
		}
		
		if (this.estimators<1){
			estimators=1;
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
			/*
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
			*/
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
			data=  (fsmatrix) data.Copy();
		}
		// Initialise randomizer

		
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
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		
		columndimension=data.GetColumnDimension();

		// Initialise the tree structure

		Thread[] thread_array= new Thread[this.threads];
		body= new regressor[this.estimators];
		// start the loop to find the support vectors 

		int count_of_live_threads=0;
		for (int n=0; n <this.estimators; n++ ){
			
			regressor model =null;
			if (model_parameters.contains("AdaboostForestRegressor")) {
				model= new AdaboostForestRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnRandomForestRegressor")) {
			model= new SklearnRandomForestRegressor(fsdataset);					
			}else if (model_parameters.contains("DecisionTreeRegressor")) {
				model= new DecisionTreeRegressor(fsdataset);
			}else if (model_parameters.contains("GradientBoostingForestRegressor")) {
				model= new GradientBoostingForestRegressor(fsdataset);
			}else if (model_parameters.contains("RandomForestRegressor")) {
				model= new RandomForestRegressor(fsdataset);
			}else if (model_parameters.contains("Vanilla2hnnregressor")) {
				model= new Vanilla2hnnregressor(fsdataset);
			}else if (model_parameters.contains("multinnregressor")) {
				model= new multinnregressor(fsdataset);
			}else if (model_parameters.contains("LSVR")) {
				model= new LSVR(fsdataset);
			}else if (model_parameters.contains("LinearRegression")) {
				model= new LinearRegression(fsdataset);
			}else if (model_parameters.contains("OriginalLibFMRegressor")) {
				model= new OriginalLibFMRegressor(fsdataset);					
			}else if (model_parameters.contains("LibFmRegressor")) {
				model= new LibFmRegressor(fsdataset);
			}else if (model_parameters.contains("knnRegressor")) {
				model= new knnRegressor(fsdataset);
			}else if (model_parameters.contains("KernelmodelRegressor")) {
				model= new KernelmodelRegressor(fsdataset);
			}else if (model_parameters.contains("XgboostRegressor")) {
				model= new XgboostRegressor(fsdataset);									
			}else if (model_parameters.contains("LightgbmRegressor")) {
			model= new LightgbmRegressor(fsdataset);							
			}else if (model_parameters.contains("H2OGbmRegressor")) {
			model= new H2OGbmRegressor(fsdataset);									
			}else if (model_parameters.contains("H2ODeepLearningRegressor")) {
			model= new H2ODeepLearningRegressor(fsdataset);	
			}else if (model_parameters.contains("H2OGlmRegressor")) {
			model= new H2OGlmRegressor(fsdataset);									
			}else if (model_parameters.contains("H2ODrfRegressor")) {
			model= new H2ODrfRegressor(fsdataset);	
			}else if (model_parameters.contains("FRGFRegressor")) {
			model= new FRGFRegressor(fsdataset);			
			}else if (model_parameters.contains("SklearnAdaBoostRegressor")) {
				model= new SklearnAdaBoostRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnDecisionTreeRegressor")) {
				model= new SklearnDecisionTreeRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnExtraTreesRegressor")) {
				model= new SklearnExtraTreesRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnknnRegressor")) {
				model= new SklearnknnRegressor(fsdataset);
			}else if (model_parameters.contains("SklearnMLPRegressor")) {
				model= new SklearnMLPRegressor(fsdataset);								
						
			}else if (model_parameters.contains("SklearnSGDRegressor")) {
			model= new SklearnSGDRegressor(fsdataset);									
			}else if (model_parameters.contains("SklearnsvmRegressor")) {
			model= new SklearnsvmRegressor(fsdataset);	
			}else if (model_parameters.contains("KerasnnRegressor")) {
			model= new KerasnnRegressor(fsdataset);									
			}else if (model_parameters.contains("PythonGenericRegressor")) {
			model= new PythonGenericRegressor(fsdataset);				
			}else if (model_parameters.contains("VowpaLWabbitRegressor")) {
			model= new VowpaLWabbitRegressor(fsdataset);			
			} else {
				throw new IllegalStateException(" The selected model '" + model_parameters + " is not recognizable as valid classifier" );
			}
			
			model.set_params(this.model_parameters);
			model.setSeed(model.getSeed() + this.seed +  n);
			model.set_target(this.fstarget);
			body[n]=model;
			
					
				thread_array[count_of_live_threads]= new Thread(model);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting batch Tree: " + n);
					
				}				
				if (count_of_live_threads==threads || n==this.estimators-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {

							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
				
					

					count_of_live_threads=0;
				}
				
				
			}

		


		fstarget=null;
		fsdataset=null;
		dataset=null;
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
		if (this.model_parameters.equals("")){
			throw new IllegalStateException(" Valid model parameters need to be set prior to fitting the method" );
		}
		
		if (this.estimators<1){
			estimators=1;
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
			/*
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
			*/
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
			data=  (smatrix) data.Copy();
		}
		// Initialise randomizer

		
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
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		
		columndimension=data.GetColumnDimension();

		// Initialise the tree structure

		Thread[] thread_array= new Thread[this.threads];
		body= new regressor[this.estimators];
		// start the loop to find the support vectors 

		int count_of_live_threads=0;
		for (int n=0; n <this.estimators; n++ ){
			
			regressor model =null;
			if (model_parameters.contains("AdaboostForestRegressor")) {
				model= new AdaboostForestRegressor(sdataset);
			}else if (model_parameters.contains("SklearnRandomForestRegressor")) {
			model= new SklearnRandomForestRegressor(sdataset);					
			}else if (model_parameters.contains("DecisionTreeRegressor")) {
				model= new DecisionTreeRegressor(sdataset);
			}else if (model_parameters.contains("GradientBoostingForestRegressor")) {
				model= new GradientBoostingForestRegressor(sdataset);
			}else if (model_parameters.contains("RandomForestRegressor")) {
				model= new RandomForestRegressor(sdataset);
			}else if (model_parameters.contains("Vanilla2hnnregressor")) {
				model= new Vanilla2hnnregressor(sdataset);
			}else if (model_parameters.contains("multinnregressor")) {
				model= new multinnregressor(sdataset);
			}else if (model_parameters.contains("LSVR")) {
				model= new LSVR(sdataset);
			}else if (model_parameters.contains("LinearRegression")) {
				model= new LinearRegression(sdataset);
			}else if (model_parameters.contains("OriginalLibFMRegressor")) {
				model= new OriginalLibFMRegressor(sdataset);				
				
			}else if (model_parameters.contains("LibFmRegressor")) {
				model= new LibFmRegressor(sdataset);
			}else if (model_parameters.contains("knnRegressor")) {
				model= new knnRegressor(sdataset);
			}else if (model_parameters.contains("KernelmodelRegressor")) {
				model= new KernelmodelRegressor(sdataset);
			}else if (model_parameters.contains("XgboostRegressor")) {
				model= new XgboostRegressor(sdataset);									
			}else if (model_parameters.contains("LightgbmRegressor")) {
			model= new LightgbmRegressor(sdataset);							
			}else if (model_parameters.contains("H2OGbmRegressor")) {
			model= new H2OGbmRegressor(sdataset);									
			}else if (model_parameters.contains("H2ODeepLearningRegressor")) {
			model= new H2ODeepLearningRegressor(sdataset);	
			}else if (model_parameters.contains("H2OGlmRegressor")) {
			model= new H2OGlmRegressor(sdataset);									
			}else if (model_parameters.contains("H2ODrfRegressor")) {
			model= new H2ODrfRegressor(sdataset);
			}else if (model_parameters.contains("FRGFRegressor")) {
			model= new FRGFRegressor(sdataset);		
			}else if (model_parameters.contains("SklearnAdaBoostRegressor")) {
				model= new SklearnAdaBoostRegressor(sdataset);
			}else if (model_parameters.contains("SklearnDecisionTreeRegressor")) {
				model= new SklearnDecisionTreeRegressor(sdataset);
			}else if (model_parameters.contains("SklearnExtraTreesRegressor")) {
				model= new SklearnExtraTreesRegressor(sdataset);
			}else if (model_parameters.contains("SklearnknnRegressor")) {
				model= new SklearnknnRegressor(sdataset);
			}else if (model_parameters.contains("SklearnMLPRegressor")) {
				model= new SklearnMLPRegressor(sdataset);														
			}else if (model_parameters.contains("SklearnSGDRegressor")) {
			model= new SklearnSGDRegressor(sdataset);									
			}else if (model_parameters.contains("SklearnsvmRegressor")) {
			model= new SklearnsvmRegressor(sdataset);	
			}else if (model_parameters.contains("KerasnnRegressor")) {
			model= new KerasnnRegressor(sdataset);									
			}else if (model_parameters.contains("PythonGenericRegressor")) {
			model= new PythonGenericRegressor(sdataset);				
			}else if (model_parameters.contains("VowpaLWabbitRegressor")) {
			model= new VowpaLWabbitRegressor(sdataset);			
			
			} else {
				throw new IllegalStateException(" The selected model '" + model_parameters + " is not recognizable as valid classifier" );
			}
			
			model.set_params(this.model_parameters);
			model.setSeed(model.getSeed() + this.seed +  n);
			model.set_target(this.fstarget);
			body[n]=model;
			
					
				thread_array[count_of_live_threads]= new Thread(model);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting batch Tree: " + n);
					
				}				
				if (count_of_live_threads==threads || n==this.estimators-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {

							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}
				
					

					count_of_live_threads=0;
				}
				
				
			}

		


		fstarget=null;
		fsdataset=null;
		dataset=null;
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
		return "BaggingRegressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor: BaggingRegressor");
		System.out.println("Targets: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		System.out.println("Estimators: " + this.estimators);		
		System.out.println("model_parameters: " + this.model_parameters);		
		System.out.println("threads : "+ this.threads);			
		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (this.body==null){
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
	public boolean HasTheSametype(estimator a) {
		if (a.GetType().equals(this.GetType())){
			return true;
		} else {
		return false;
		}
	}

	@Override
	public boolean isfitted() {
		if (this.body!=null || body.length>0){
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
		this.body= null;
		n_classes=0;
		threads=1;
		this.estimators=10;
		this.random=null;
		this.feature_importances.clone();
		columndimension=0;
		this.model_parameters="";
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
		BaggingRegressor br = new BaggingRegressor();
		regressor[] bodys= new regressor[this.body.length];
        for (int i=0; i <bodys.length; i++ ){
        	bodys[i]=(regressor) body[i].copy();
        }
        br.body=bodys;
        br.estimators=this.estimators;
		br.n_classes=this.n_classes;
		br.feature_importances=this.feature_importances.clone();
		br.threads=this.threads;
		br.columndimension=this.columndimension;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.model_parameters=this.model_parameters;
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.target2d=manipulate.copies.copies.Copy(this.target2d.clone());	
		br.fstarget=(fsmatrix) this.fstarget.Copy();
		br.starget=(smatrix) this.starget.Copy();
		br.weights=manipulate.copies.copies.Copy(this.weights.clone());
		br.verbose=this.verbose;
		return br;
	}
	
	@Override	
	public void set_params(String params){
		
		String splitted_params []=params.split(" " + "+");
		
		for (int j=0; j<splitted_params.length; j++ ){
			String mini_split []=splitted_params[j].split(":");
			if (mini_split.length>=2){
				String metric=mini_split[0];
				String value=mini_split[1];
				
				if (metric.equals("estimators")) {this.estimators=Integer.parseInt(value);}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}			
				
			}
			
		}
		

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

	  

