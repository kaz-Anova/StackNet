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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import utilis.map.intint.StringIntMap4a;
import exceptions.DimensionMismatchException;
import exceptions.IllegalStateException;
import exceptions.LessThanMinimum;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;
import ml.Kernel.copy.KernelmodelClassifier;
import ml.LSVC.LSVC;
import ml.LibFm.LibFmClassifier;
import ml.LibFm.OriginalLibFMClassifier;
import ml.LogisticRegression.LogisticRegression;
import ml.NaiveBayes.NaiveBayesClassifier;
import ml.Tree.AdaboostRandomForestClassifier;
import ml.Tree.DecisionTreeClassifier;
import ml.Tree.GradientBoostingForestClassifier;
import ml.Tree.RandomForestClassifier;
import ml.fastrgf.FRGFClassifier;
import ml.h2o.H2ODeepLearningClassifier;
import ml.h2o.H2ODrfClassifier;
import ml.h2o.H2OGbmClassifier;
import ml.h2o.H2OGlmClassifier;
import ml.h2o.H2ONaiveBayesClassifier;
import ml.knn.knnClassifier;
import ml.libffm.libffmClassifier;
import ml.lightgbm.LightgbmClassifier;
import ml.nn.Vanilla2hnnclassifier;
import ml.nn.softmaxnnclassifier;
import ml.python.PythonGenericClassifier;
import ml.python.keras.KerasnnClassifier;
import ml.python.sklearn.SklearnAdaBoostClassifier;
import ml.python.sklearn.SklearnDecisionTreeClassifier;
import ml.python.sklearn.SklearnExtraTreesClassifier;
import ml.python.sklearn.SklearnMLPClassifier;
import ml.python.sklearn.SklearnRandomForestClassifier;
import ml.python.sklearn.SklearnSGDClassifier;
import ml.python.sklearn.SklearnknnClassifier;
import ml.python.sklearn.SklearnsvmClassifier;
import ml.stacknet.StackNetClassifier;
import ml.vowpalwabbit.VowpaLWabbitClassifier;
import ml.xgboost.XgboostClassifier;

/**
 * <p> Classifier Base for bagging
 */

public class BaggingClassifier implements estimator,classifier {


	  /**
	   * This holds all the classifiers
	   */
	private classifier body [] ;
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

	/**
	 * 
	 * @param params : model parameters
	 */
	public void set_model_parameters(String params){
		//check if parameters are valid
		if (StackNetClassifier.containsClassifier( params)){
			model_parameters=params;
		} else {
			throw new IllegalStateException(" no recognised classifier in " + params );
		}
	}

	
	public  classifier [] Get_tree(){
		if (this.body==null || this.body.length<=0){
			throw new IllegalStateException(" There is NO tree" );
		}
		return body;
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
	 * Name of the unique classes
	 */
	private String classes[];
	/**
	 * Target variable in String format
	 */	
	public String Starget[];
	
	public int getnumber_of_classes(){
		return n_classes;
	}
	/**
	 * 
	 * @param classes : the classes to put
	 */
	public void setclasses(String[] classes ) {
		if (classes==null || classes.length<=0){
			throw new  IllegalStateException (" No classes are found");
		} else {
		this.classes= classes;
		}
	}
	
	@Override
	public String[] getclasses() {
		if (classes==null || classes.length<=0){
			throw new  IllegalStateException (" No classes are found, the model needs to be fitted first");
		} else {
		return classes;
		}
	}
	@Override
	public void AddClassnames(String[] names) {
		
		String distinctnames[]=manipulate.distinct.distinct.getstringDistinctset(names);
		if (distinctnames.length<2){
			throw new LessThanMinimum(names.length,2);
		}
		if (distinctnames.length!=names.length){
			throw new  IllegalStateException (" There are duplicate values in the names of the addClasses method, dedupe before adding them");
		}
		classes=new String[names.length];
		for (int j=0; j < names.length; j++){
			classes[j]=names[j];
		}
	}
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
	 * Default constructor for BaggingClassifier with no data
	 */
	public BaggingClassifier(){
	
	}	
	/**
	 * Default constructor for BaggingClassifier with double data
	 */
	public BaggingClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for BaggingClassifier with fsmatrix data
	 */
	public BaggingClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for BaggingClassifier with smatrix data
	 */
	public BaggingClassifier(smatrix data){
		
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
	public double[][] predict_proba(double[][] data) {
		 
		/*  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.body==null || this.body.length<=0 ){
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

				thread_array[count_of_live_threads]= new Thread(new scoringhelpercatbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
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

	@Override
	public double[][] predict_proba(fsmatrix data) {
		if (n_classes<2 ||this.body==null || this.body.length<=0 ){
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

				thread_array[count_of_live_threads]= new Thread(new scoringhelpercatbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
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

	public fsmatrix predict_probafs(fsmatrix data) {
		if (n_classes<2 ||this.body==null || this.body.length<=0 ){
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

				thread_array[count_of_live_threads]= new Thread(new scoringhelpercatbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
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
	public double[][] predict_proba(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.body==null || this.body.length<=0 ){
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

				thread_array[count_of_live_threads]= new Thread(new scoringhelpercatbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
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

	@Override
	public double[] predict_probaRow(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.body==null || this.body.length<=0  ){
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
			double newpredictions[]=body[j].predict_probaRow(data);
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
	public double[] predict_probaRow(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.body==null || this.body.length<=0  ){
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
		}	

		double predictions[]= new double [this.n_classes];
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_probaRow(data,rows);
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
	public double[] predict_probaRow(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.body==null || this.body.length<=0  ){
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
		}	
		double predictions[]= new double [this.n_classes];
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_probaRow(data,start,end);
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
		if (n_classes<2 ||this.body==null || this.body.length<=0 ){
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
		
		double predictions[]= new double [data.GetRowDimension()];
		double predictions_probas[][]= new double [data.GetRowDimension()][n_classes];
		
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelpercatbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
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
							for (int j=0; j < predictions_probas[0].length; j++){
								predictions_probas[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}
		for (int i=0; i < predictions.length; i++) {
			double temp[]=predictions_probas[i];
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions[i]=maxi;
	    	  }

		}		
		
		predictions_probas=null;

			// return the 1st prediction
			return predictions;
			
			}
			

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.body==null || this.body.length<=0 ){
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

		double predictions[]= new double [data.GetRowDimension()];
		double predictions_probas[][]= new double [data.GetRowDimension()][n_classes];

		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.GetRowDimension(),this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelpercatbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
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
							for (int j=0; j < predictions_probas[0].length; j++){
								predictions_probas[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}	
		

		for (int i=0; i < predictions.length; i++) {
			double temp[]=predictions_probas[i];
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions[i]=maxi;
	    	  }

		}
	predictions_probas=null;

			// return the 1st prediction
			return predictions;
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.body==null || this.body.length<=0 ){
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
		
		double predictions[]= new double [data.length];
		double predictions_probas[][]= new double [data.length][n_classes];
		Thread[] thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
        fsmatrix arrays []= new fsmatrix[(body.length <this.threads)?body.length:this.threads];
        
			int count_of_live_threads=0;

			for (int tree =0; tree <body.length;tree++ ){
				
				arrays[count_of_live_threads]=new fsmatrix(data.length,this.n_classes);

				thread_array[count_of_live_threads]= new Thread(new scoringhelpercatbagv2 (data, arrays[count_of_live_threads], body[tree])); ;
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
							for (int j=0; j < predictions_probas[0].length; j++){
								predictions_probas[i][j]+=arrays[s].GetElement(i, j);
							}
						}
						
					}
						
					count_of_live_threads=0;
					thread_array= new Thread[(body.length <this.threads)?body.length:this.threads]; // generate threads' array
			        arrays = new fsmatrix[(body.length <this.threads)?body.length:this.threads];
			        
				}

			}
		for (int i=0; i < predictions.length; i++) {
			double temp[]=predictions_probas[i];
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions[i]=maxi;
	    	  }

		}
		predictions_probas=null;

			// return the 1st prediction
			return predictions;
			
			}

	@Override
	public double predict_Row(double[] data) {
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


		double predictions= 0.0;
		double predictions_probas[]= new double [n_classes];
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_probaRow(data);
				for (int b=0; b < newpredictions.length; b++){
					predictions_probas[b]+=newpredictions[b];
				}
		}
			double temp[]=predictions_probas;
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions=maxi;
	    	  }

		
	predictions_probas=null;
		

			// return the 1st prediction
			return predictions;
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.body==null || this.body.length<=0  ){
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
		}	


		double predictions= 0.0;
		double predictions_probas[]= new double [n_classes];
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_probaRow(data,rows);
				for (int b=0; b < newpredictions.length; b++){
					predictions_probas[b]+=newpredictions[b];
				}
		}
			double temp[]=predictions_probas;
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions=maxi;
	    	  }

		
	    	  predictions_probas=null;
		

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
		double predictions_probas[]= new double [n_classes];
		
		for (int j=0; j < body.length; j++){
			double newpredictions[]=body[j].predict_probaRow(data, start,  end);
				for (int b=0; b < newpredictions.length; b++){
					predictions_probas[b]+=newpredictions[b];
				}
		}
			double temp[]=predictions_probas;
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions=maxi;
	    	  }

		
	    	  predictions_probas=null;
		


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
		if ( (target==null || target.length!=data.length)  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else if (target!=null && (classes==null ||  classes.length<=1) ){
				
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
			}
			double uniquevalues[]= new double[has.size()];
			
			int k=0;
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		    	uniquevalues[k]= it.next();
		    	k++;
		    	}
		    // sort values
		    Arrays.sort(uniquevalues);
		    
		    classes= new String[uniquevalues.length];
		    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
		    int index=0;
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    	mapper.put(classes[j], index);
		    	index++;
		    }
		   
		    for (int i=0; i < target.length; i++){
		    	target[i]=mapper.get(target[i] + "");
		    }
			   
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
		fsdataset=new fsmatrix(data);
			
		
		// Initialise randomizer

		this.random = new XorShift128PlusRandom(this.seed);
		this.n_classes=classes.length;			
		

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
		body= new classifier[this.estimators];
		// start the loop to find the support vectors 

		int count_of_live_threads=0;
		for (int n=0; n <this.estimators; n++ ){
			classifier model =null;
			if (model_parameters.contains("AdaboostRandomForestClassifier")) {
				model= new AdaboostRandomForestClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnAdaBoostClassifier")) {
				model= new SklearnAdaBoostClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnDecisionTreeClassifier")) {
				model= new SklearnDecisionTreeClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnExtraTreesClassifier")) {
				model= new SklearnExtraTreesClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnknnClassifier")) {
				model= new SklearnknnClassifier(fsdataset);							
			}else if (model_parameters.contains("SklearnMLPClassifier")) {
				model= new SklearnMLPClassifier(fsdataset);	
			}else if (model_parameters.contains("SklearnRandomForestClassifier")) {
				model= new SklearnRandomForestClassifier(fsdataset);					
			}else if (model_parameters.contains("DecisionTreeClassifier")) {
				model= new DecisionTreeClassifier(fsdataset);
			}else if (model_parameters.contains("GradientBoostingForestClassifier")) {
				model= new GradientBoostingForestClassifier(fsdataset);
			}else if (model_parameters.contains("RandomForestClassifier")) {
				model= new RandomForestClassifier(fsdataset);
			}else if (model_parameters.contains("Vanilla2hnnclassifier")) {
				model= new Vanilla2hnnclassifier(fsdataset);
			}else if (model_parameters.contains("softmaxnnclassifier")) {
				model= new softmaxnnclassifier(fsdataset);
			}else if (model_parameters.contains("NaiveBayesClassifier")) {
				model= new NaiveBayesClassifier(fsdataset);
			}else if (model_parameters.contains("OriginalLibFMClassifier")) {
				model= new OriginalLibFMClassifier(fsdataset);				
			}else if (model_parameters.contains("LSVC")) {
				model= new LSVC(fsdataset);
			}else if (model_parameters.contains("LogisticRegression")) {
				model= new LogisticRegression(fsdataset);
			}else if (model_parameters.contains("LibFmClassifier")) {
				model= new LibFmClassifier(fsdataset);
			}else if (model_parameters.contains("knnClassifier")) {
				model= new knnClassifier(fsdataset);
			}else if (model_parameters.contains("KernelmodelClassifier")) {
				model= new KernelmodelClassifier(fsdataset);							
			}else if (model_parameters.contains("XgboostClassifier")) {
				model= new XgboostClassifier(fsdataset);	
			}else if (model_parameters.contains("LightgbmClassifier")) {
				model= new LightgbmClassifier(fsdataset);							
			}else if (model_parameters.contains("H2OGbmClassifier")) {
				model= new H2OGbmClassifier(fsdataset);								
			}else if (model_parameters.contains("H2ODeepLearningClassifier")) {
				model= new H2ODeepLearningClassifier(fsdataset);
			}else if (model_parameters.contains("H2ODrfClassifier")) {
				model= new H2ODrfClassifier(fsdataset);			
			}else if (model_parameters.contains("H2OGlmClassifier")) {
				model= new H2OGlmClassifier(fsdataset);					
			}else if (model_parameters.contains("H2ONaiveBayesClassifier")) {
				model= new H2ONaiveBayesClassifier(fsdataset);	
			}else if (model_parameters.contains("FRGFClassifier")) {
				model= new FRGFClassifier(fsdataset);	
						
			}else if (model_parameters.contains("SklearnSGDClassifier")) {
				model= new SklearnSGDClassifier(fsdataset);								
			}else if (model_parameters.contains("SklearnsvmClassifier")) {
				model= new SklearnsvmClassifier(fsdataset);
			}else if (model_parameters.contains("KerasnnClassifier")) {
				model= new KerasnnClassifier(fsdataset);			
			}else if (model_parameters.contains("PythonGenericClassifier")) {
				model= new PythonGenericClassifier(fsdataset);					
			}else if (model_parameters.contains("VowpaLWabbitClassifier")) {
				model= new VowpaLWabbitClassifier(fsdataset);					
			}else if (model_parameters.contains("libffmClassifier")) {
				model= new libffmClassifier(fsdataset);	
				
				
				
					
				
				
				
				
				
			} else {
				throw new IllegalStateException(" The selected model '" + model_parameters + " is not recognizable as valid classifier" );
			}
			
			model.AddClassnames(this.classes);
			model.set_params(this.model_parameters);
			model.setSeed(model.getSeed() + this.seed +  n);
			model.set_target(this.target);
			body[n]=model;
					
				thread_array[count_of_live_threads]= new Thread(model);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting batch model: " + n);
					
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

		
		target=null;
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
		if ( (target==null || target.length!=data.GetRowDimension())){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else if (target!=null && (classes==null ||  classes.length<=1) ){
				
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
			}
			double uniquevalues[]= new double[has.size()];
			
			int k=0;
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		    	uniquevalues[k]= it.next();
		    	k++;
		    	}
		    // sort values
		    Arrays.sort(uniquevalues);
		    
		    classes= new String[uniquevalues.length];
		    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
		    int index=0;
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    	mapper.put(classes[j], index);
		    	index++;
		    }
		    for (int i=0; i < target.length; i++){
		    	target[i]=mapper.get(target[i] + "");
		    }
		    

			  
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
		fsdataset=data;
			
		// Initialise randomizer

		this.random = new XorShift128PlusRandom(this.seed);
		this.n_classes=classes.length;			
		

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
		body= new classifier[this.estimators];
		// start the loop to find the support vectors 

		int count_of_live_threads=0;
		for (int n=0; n <this.estimators; n++ ){
			classifier model =null;
			if (model_parameters.contains("AdaboostRandomForestClassifier")) {
				model= new AdaboostRandomForestClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnAdaBoostClassifier")) {
				model= new SklearnAdaBoostClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnDecisionTreeClassifier")) {
				model= new SklearnDecisionTreeClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnExtraTreesClassifier")) {
				model= new SklearnExtraTreesClassifier(fsdataset);
			}else if (model_parameters.contains("SklearnknnClassifier")) {
				model= new SklearnknnClassifier(fsdataset);							
			}else if (model_parameters.contains("SklearnMLPClassifier")) {
				model= new SklearnMLPClassifier(fsdataset);	
			}else if (model_parameters.contains("SklearnRandomForestClassifier")) {
				model= new SklearnRandomForestClassifier(fsdataset);					
			}else if (model_parameters.contains("DecisionTreeClassifier")) {
				model= new DecisionTreeClassifier(fsdataset);
			}else if (model_parameters.contains("GradientBoostingForestClassifier")) {
				model= new GradientBoostingForestClassifier(fsdataset);
			}else if (model_parameters.contains("RandomForestClassifier")) {
				model= new RandomForestClassifier(fsdataset);
			}else if (model_parameters.contains("Vanilla2hnnclassifier")) {
				model= new Vanilla2hnnclassifier(fsdataset);
			}else if (model_parameters.contains("softmaxnnclassifier")) {
				model= new softmaxnnclassifier(fsdataset);
			}else if (model_parameters.contains("NaiveBayesClassifier")) {
				model= new NaiveBayesClassifier(fsdataset);
			}else if (model_parameters.contains("OriginalLibFMClassifier")) {
				model= new OriginalLibFMClassifier(fsdataset);				
			}else if (model_parameters.contains("LSVC")) {
				model= new LSVC(fsdataset);
			}else if (model_parameters.contains("LogisticRegression")) {
				model= new LogisticRegression(fsdataset);
			}else if (model_parameters.contains("LibFmClassifier")) {
				model= new LibFmClassifier(fsdataset);
			}else if (model_parameters.contains("knnClassifier")) {
				model= new knnClassifier(fsdataset);
			}else if (model_parameters.contains("KernelmodelClassifier")) {
				model= new KernelmodelClassifier(fsdataset);							
			}else if (model_parameters.contains("XgboostClassifier")) {
				model= new XgboostClassifier(fsdataset);	
			}else if (model_parameters.contains("LightgbmClassifier")) {
				model= new LightgbmClassifier(fsdataset);							
			}else if (model_parameters.contains("H2OGbmClassifier")) {
				model= new H2OGbmClassifier(fsdataset);								
			}else if (model_parameters.contains("H2ODeepLearningClassifier")) {
				model= new H2ODeepLearningClassifier(fsdataset);
			}else if (model_parameters.contains("H2ODrfClassifier")) {
				model= new H2ODrfClassifier(fsdataset);			
			}else if (model_parameters.contains("H2OGlmClassifier")) {
				model= new H2OGlmClassifier(fsdataset);					
			}else if (model_parameters.contains("H2ONaiveBayesClassifier")) {
				model= new H2ONaiveBayesClassifier(fsdataset);	
			}else if (model_parameters.contains("FRGFClassifier")) {
				model= new FRGFClassifier(fsdataset);					
			}else if (model_parameters.contains("SklearnSGDClassifier")) {
				model= new SklearnSGDClassifier(fsdataset);								
			}else if (model_parameters.contains("SklearnsvmClassifier")) {
				model= new SklearnsvmClassifier(fsdataset);
			}else if (model_parameters.contains("KerasnnClassifier")) {
				model= new KerasnnClassifier(fsdataset);			
			}else if (model_parameters.contains("PythonGenericClassifier")) {
				model= new PythonGenericClassifier(fsdataset);					
			}else if (model_parameters.contains("VowpaLWabbitClassifier")) {
				model= new VowpaLWabbitClassifier(fsdataset);					
			}else if (model_parameters.contains("libffmClassifier")) {
				model= new libffmClassifier(fsdataset);				
				
			} else {
				throw new IllegalStateException(" The selected model '" + model_parameters + " is not recognizable as valid classifier" );
			}
			
			model.AddClassnames(this.classes);
			model.set_params(this.model_parameters);
			model.setSeed(model.getSeed() + this.seed +  n);
			model.set_target(this.target);
			
			body[n]=model;
					
				thread_array[count_of_live_threads]= new Thread(model);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting batch model: " + n);
					
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
		target=null;
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
		if ( (target==null || target.length!=data.GetRowDimension())  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else if (target!=null  ){
				
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
			}
			double uniquevalues[]= new double[has.size()];
			
			int k=0;
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		    	uniquevalues[k]= it.next();
		    	k++;
		    	}
		    // sort values
		    Arrays.sort(uniquevalues);
		    
		    classes= new String[uniquevalues.length];
		    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
		    int index=0;
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    	mapper.put(classes[j], index);
		    	index++;
		    }

		    for (int i=0; i < target.length; i++){
		    	target[i]=mapper.get(target[i] + "");
		    }
		    

			  
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
		sdataset=data;
			
		// Initialise randomizer

		this.random = new XorShift128PlusRandom(this.seed);
		this.n_classes=classes.length;			
		

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
		body= new classifier[this.estimators];
		// start the loop to find the support vectors 

		int count_of_live_threads=0;
		for (int n=0; n <this.estimators; n++ ){
			classifier model =null;
			if (model_parameters.contains("AdaboostRandomForestClassifier")) {
				model= new AdaboostRandomForestClassifier(sdataset);
			}else if (model_parameters.contains("SklearnAdaBoostClassifier")) {
				model= new SklearnAdaBoostClassifier(sdataset);
			}else if (model_parameters.contains("SklearnDecisionTreeClassifier")) {
				model= new SklearnDecisionTreeClassifier(sdataset);
			}else if (model_parameters.contains("SklearnExtraTreesClassifier")) {
				model= new SklearnExtraTreesClassifier(sdataset);
			}else if (model_parameters.contains("SklearnknnClassifier")) {
				model= new SklearnknnClassifier(sdataset);							
			}else if (model_parameters.contains("SklearnMLPClassifier")) {
				model= new SklearnMLPClassifier(sdataset);	
			}else if (model_parameters.contains("SklearnRandomForestClassifier")) {
				model= new SklearnRandomForestClassifier(sdataset);						
			}else if (model_parameters.contains("DecisionTreeClassifier")) {
				model= new DecisionTreeClassifier(sdataset);
			}else if (model_parameters.contains("GradientBoostingForestClassifier")) {
				model= new GradientBoostingForestClassifier(sdataset);
			}else if (model_parameters.contains("RandomForestClassifier")) {
				model= new RandomForestClassifier(sdataset);
			}else if (model_parameters.contains("Vanilla2hnnclassifier")) {
				model= new Vanilla2hnnclassifier(sdataset);
			}else if (model_parameters.contains("softmaxnnclassifier")) {
				model= new softmaxnnclassifier(sdataset);
			}else if (model_parameters.contains("NaiveBayesClassifier")) {
				model= new NaiveBayesClassifier(sdataset);
			}else if (model_parameters.contains("OriginalLibFMClassifier")) {
				model= new OriginalLibFMClassifier(sdataset);				
			}else if (model_parameters.contains("LSVC")) {
				model= new LSVC(sdataset);
			}else if (model_parameters.contains("LogisticRegression")) {
				model= new LogisticRegression(sdataset);
			}else if (model_parameters.contains("LibFmClassifier")) {
				model= new LibFmClassifier(sdataset);
			}else if (model_parameters.contains("knnClassifier")) {
				model= new knnClassifier(sdataset);
			}else if (model_parameters.contains("KernelmodelClassifier")) {
				model= new KernelmodelClassifier(sdataset);							
			}else if (model_parameters.contains("XgboostClassifier")) {
				model= new XgboostClassifier(sdataset);	
			}else if (model_parameters.contains("LightgbmClassifier")) {
				model= new LightgbmClassifier(sdataset);							
			}else if (model_parameters.contains("H2OGbmClassifier")) {
				model= new H2OGbmClassifier(sdataset);								
			}else if (model_parameters.contains("H2ODeepLearningClassifier")) {
				model= new H2ODeepLearningClassifier(sdataset);
			}else if (model_parameters.contains("H2ODrfClassifier")) {
				model= new H2ODrfClassifier(sdataset);			
			}else if (model_parameters.contains("H2OGlmClassifier")) {
				model= new H2OGlmClassifier(sdataset);					
			}else if (model_parameters.contains("H2ONaiveBayesClassifier")) {
				model= new H2ONaiveBayesClassifier(sdataset);	
			}else if (model_parameters.contains("FRGFClassifier")) {
				model= new FRGFClassifier(sdataset);	
					
			}else if (model_parameters.contains("SklearnSGDClassifier")) {
				model= new SklearnSGDClassifier(sdataset);								
			}else if (model_parameters.contains("SklearnsvmClassifier")) {
				model= new SklearnsvmClassifier(sdataset);
			}else if (model_parameters.contains("KerasnnClassifier")) {
				model= new KerasnnClassifier(sdataset);			
			}else if (model_parameters.contains("PythonGenericClassifier")) {
				model= new PythonGenericClassifier(sdataset);					
			}else if (model_parameters.contains("VowpaLWabbitClassifier")) {
				model= new VowpaLWabbitClassifier(sdataset);				
			}else if (model_parameters.contains("libffmClassifier")) {
				model= new libffmClassifier(sdataset);				
				
			} else {
				throw new IllegalStateException(" The selected model '" + model_parameters + " is not recognizable as valid classifier" );
			}
			
			model.AddClassnames(this.classes);
			model.set_params(this.model_parameters);
			model.setSeed(model.getSeed() + this.seed +  n);
			model.set_target(this.target);
			body[n]=model;
					
				thread_array[count_of_live_threads]= new Thread(model);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting batch model: " + n);
					
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
		target=null;
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
		return "classifier";
	}
	@Override
	public boolean SupportsWeights() {
		return true;
	}

	@Override
	public String GetName() {
		return "BaggingClassifier";
	}
	@Override	
	public void set_target(double data []){
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}

		this.target=data;
	}
	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: BaggingClassifier");
		System.out.println("Classes: " + n_classes);
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
		return false  ;
	}

	@Override
	public boolean IsClassifier() {
		return true;
	}

	@Override
	public void reset() {
		this.body= null;
		n_classes=0;
		threads=1;
		this.estimators=10;
		this.model_parameters="";
		this.random=null;
		columndimension=0;
		this.classes=null;
		copy=true;
		seed=1;
		random=null;
		target=null;
		target=null;
		weights=null;
		verbose=true;
		
	}

	@Override
	public estimator copy() {
		BaggingClassifier br = new BaggingClassifier();
		classifier[] bodys= new classifier[this.body.length];
        for (int i=0; i <bodys.length; i++ ){
        	bodys[i]=(classifier) body[i].copy();
        }
        br.body=bodys;
        br.estimators=this.estimators;
        br.model_parameters=this.model_parameters;
		br.n_classes=this.n_classes;
		br.threads=this.threads;
		br.columndimension=this.columndimension;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.target=manipulate.copies.copies.Copy(this.target.clone());
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
				
				if 	 (metric.equals("estimators")) {this.estimators=Integer.parseInt(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}			
				else if 	 (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
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
	public int getSeed() {
		return this.seed;}
	}

	  

