package stacknetrun;

import io.Serialized_Object;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashSet;

import crossvalidation.metrics.auc;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.stacknet.StackNetClassifier;

public class runstacknet {

	
	/**
	 * If it is train or predict . defaults is true
	 */
	private static boolean is_train=true;
	/**
	 * If input files are in sparse format. defaults is false
	 */
	private static boolean is_sparse=false;	
	/**
	 * If train and test files have headers. defaults to false
	 */
	private static boolean has_head=false;
	/**
	 * If the test file has a column in the beginning which corresponds to the target. defaults to false
	 */
	private static boolean test_file_has_target=false;	
	/**
	 * name of model file defaults to 'stacknet.model'
	 */
	private static String model_file="stacknet.model";
	/**
	 * name of prediction file, defaults to <em>stacknet_pred.csv</em>
	 */
	private static String pred_file="stacknet_pred.csv";
	/**
	 * prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing
	 */
	private static String output_name="";	
	/**
	 * name of training file
	 */
	private static String train_file="";
	/**
	 * name of test file 
	 */
	private static String test_file="";	
	/**
	 * name of parameters' file 
	 */
	private static String params_file="";		
	/**
	 * To allow StackNet to print stuff
	 */
	private static boolean verbose=true;
	/**
	 * number of model to run in parallel
	 */
	private static int threads=1;	
	/**
	 * Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) .defaults to 'logloss'
	 */
	private static String  metric="logloss";
	/**
	 * To allow StackNet to use the <em>restacking</em> option. defaults is true
	 */
	private static boolean restacking=true;
	/**
	 * state for randomised procedures 
	 */
	private static int seed=1;	
	/**
	 * number of folds to be used for kfold during training. defaults is 5
	 */
	private static int folds=5;	
	
	/**
	 * stackNet object to be used
	 */
	private static StackNetClassifier stacknet;
	/**
	 * @param args : arguments to train StackNet model
	 * <ul>
	 * <li> 'train' or 'predict' : to train or predict </li>
	 * <li> 'sparse' : if the data to be imported are in sparse format (libsvm) or dense . defaults to false</li>
	 * <li> 'has_head' : true if train_file and test_file have headers else false </li>
	 * <li> 'model' : prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing' </li>
	 * <li> 'output_name' : prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing </li>
	 * <li> 'pred_file' : name of the output prediction file. defaults to 'stacknet_pred.csv' </li>
	 * <li> 'train_file' : name of the training file. </li>
	 * <li> 'test_file' : name of the test  file. </li>
	 * <li> 'test_target' : true if the test file has a target variable in the beginning (left) else false (only predictors in the file). </li> 
	 * <li> 'params' : parameter file where each line is a model. empty lines correspond to the creation of new levels </li>
	 * <li> 'verbose' : true if we need StackNet to output its progress .defaults to true. </li>
	 * <li> 'threads' : number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected).   </li>
	 * <li> 'metric' : Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) .defaults to 'logloss' </li>
	 * <li> 'stackdata' :True for <em>restacking</em>. defaults to true </li>
	 * <li> 'seed' : integer for randomised procedures.defaults to 1</li>
	 * <li> 'folds' : number of folds for re-usable kfold . defaults to 5</li>
	 * <li> 'help' : gives a few helpful tips</li>
	 * </ul>
	 */

	public static void main(String[] args) {
		// check if the first command is 'help' or empty
	    if (args.length==0 || args[0].toLowerCase().indexOf("help")!=-1 || args[0].toLowerCase().indexOf("h")!=-1 || args[0]==null){
	    	System.out.println("'train' or 'predict' : to train or predict \n"+
	    "'sparse' : true if the data to be imported are in sparse format (libsvm) or dense (false) \n" +
	    "'has_head' : true if train_file and test_file have headers else false\n"+
	    "'model' : name of the output model file. \n"+
	    "'output_name' : prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing. \n"+	    
	    "'pred_file' : name of the output prediction file. \n"+
	    "'train_file' : name of the training file. \n" +
	    "'test_file' : name of the test file. \n"+
	    "'test_target' : true if the test file has a target variable in the beginning (left) else false (only predictors in the file).\n" +
	    "'params' : parameter file where each line is a model. empty lines correspond to the creation of new levels \n"+
	    "'verbose' : true if we need StackNet to output its progress else false \n"+
	    "'threads' : number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected). \n"+
	    "'metric' : Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) \n"+
	    "'stackdata' :true for restacking else false\n"+
	    "'seed' : integer for randomised procedures \n"+
	    "'folds' : number of folds for re-usable kfold\n\n"+
	    "example of parameter file :\n\n"+
	    "LogisticRegression C:1 Type:Liblinear maxim_Iteration:100 scale:true verbose:false\n"+
	    "RandomForestClassifier bootsrap:false estimators:100 threads:5 logit.offset:0.00001 verbose:false cut_off_subsample:1.0 feature_subselection:1.0 gamma:0.00001 max_depth:8 max_features:0.25 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1\n"+
	    "GradientBoostingForestClassifier estimators:100 threads: offset:0.00001 verbose:false trees:1 rounding:2 shrinkage:0.05 cut_off_subsample:1.0 feature_subselection:0.8 gamma:0.00001 max_depth:8 max_features:1.0 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:RMSE row_subsample:0.9 seed:1\n"+
	    "Vanilla2hnnclassifier UseConstant:true usescale:true seed:1 Type:SGD maxim_Iteration:50 C:0.000001 learn_rate:0.009 smooth:0.02 h1:30 h2:20 connection_nonlinearity:Relu init_values:0.02\n"+
	    "LSVC Type:Liblinear threads:1 C:1.0 maxim_Iteration:100 seed:1\n"+
	    "LibFmClassifier lfeatures:3 init_values:0.035 smooth:0.05 learn_rate:0.1 threads:1 C:0.00001 maxim_Iteration:15 seed:1\n"+
	    "NaiveBayesClassifier usescale:true threads:1 Shrinkage:0.1 seed:1 verbose:false\n\n"+
	    "RandomForestClassifier estimators=1000 rounding:3 threads:4 max_depth:6 max_features:0.6 min_leaf:2.0 Objective:ENTROPY gamma:0.000001 row_subsample:1.0 verbose:false copy=false");
	    
	    System.exit(-1); // exiting the system
	      }
	     
	    if (args[0].toLowerCase().indexOf("train")!=-1 || args[0].toLowerCase().indexOf("tr")!=-1 ){ 
	    	is_train=true;
	    } else if (args[0].toLowerCase().indexOf("predict")!=-1 || args[0].toLowerCase().indexOf("pred")!=-1 ){
	    	is_train=false;
	    } else {
	    	System.out.println(" first command needs to be either 'train' or 'predict'. Please see the documentation or type 'help'");
		    System.exit(-1); // exiting the system
	    }
	    HashSet<String> params_contained = new HashSet<String>();
	    
	    for (int j=1; j<args.length; j++ ){
	    	String option=args[j].toLowerCase();
	    	// check of the format is right
	    	if (option.indexOf("=")==-1){
	    		System.out.println(" Every option needs to have a '='. The format is 'option_name:option_value'. for example 'verbose=false'");
			    System.exit(-1); // exiting the system
	    	}
	    	String [] splits=option.split("=");
	    	if (splits.length!=2){
	    		System.out.println("There needs to be exactly one '=' in each option");
			    System.exit(-1); // exiting the system
	    	}
	    	// extract parameters' elements
	    	String parameter_name=splits[0];
	    	String parameter_value=splits[1];
	    	
	    	if (parameter_value.trim().length() <= 0){
	    		System.out.println("parameter value cannot be only white spaces");
			    System.exit(-1); // exiting the system
	    	}
	    	
	    	//check if parameter was set before
	    	if (params_contained.contains(parameter_name)){
	    		System.out.println("parameter name : " + parameter_name + " has already been set. A parameter must be specified only once." );
			    System.exit(-1); // exiting the system
	    	} else {
	    		System.out.println("parameter name : " + parameter_name + " value :  " + parameter_value );
	    	}
	    	
	    	if (parameter_name.equals("sparse")){
	    		  if (parameter_value.indexOf("false")!=-1  ){ 
	    			  is_sparse=false; 
	    		  } else if (parameter_value.indexOf("true")!=-1  ){ 
	    			  is_sparse=true; 
	    	} else {
	    		System.out.println("the 'is_sparse' parameters needs to be either 'true' or 'false' ");
			    System.exit(-1); // exiting the system
	    	}
	    	
	    	
	    } else if (parameter_name.equals("model")){
	    	model_file=parameter_value;	
	    }else if (parameter_name.equals("pred_file")){
	    	pred_file=parameter_value;	
	    }else if (parameter_name.equals("train_file")){
	    	train_file=parameter_value;	
	    }else if (parameter_name.equals("test_file")){
	    	test_file=parameter_value;	
	    }else if (parameter_name.equals("output_name")){
	    	output_name=parameter_value;	
	    }else if (parameter_name.equals("test_target")){   
    		if (parameter_value.indexOf("false")!=-1  ){ 
    			test_file_has_target=false; 
    		} else if (parameter_value.indexOf("true")!=-1  ){ 
    			test_file_has_target=true; 
    		}else {
	    		System.out.println("the 'test_target' parameters needs to be either 'true' or 'false' ");
			    System.exit(-1); // exiting the system
	    	}
	    		
	    }else if (parameter_name.equals("params")){
	    	params_file=parameter_value;	
	    }else if (parameter_name.equals("verbose")){
	    		if (parameter_value.indexOf("false")!=-1  ){ 
	    			verbose=false; 
	    		} else if (parameter_value.indexOf("true")!=-1  ){ 
	    			verbose=true; 
	    		}else {
		    		System.out.println("the 'verbose' parameters needs to be either 'true' or 'false' ");
				    System.exit(-1); // exiting the system
		    	}

	    }else if (parameter_name.equals("has_head")){
    		if (parameter_value.indexOf("false")!=-1  ){ 
    			has_head=false; 
    		} else if (parameter_value.indexOf("true")!=-1  ){ 
    			has_head=true; 
    		}else {
	    		System.out.println("the 'has_head' parameters needs to be either 'true' or 'false' ");
			    System.exit(-1); // exiting the system
	    	}

	    }else if (parameter_name.equals("threads")){
	    	try{
	    		threads=Integer.parseInt(parameter_value);
	    	} catch (Exception e){
	    		System.out.println("parameter 'threads' needs to have an integer value . here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    	if (threads<1){
	    		System.out.println("parameter 'threads' needs to have an integer value higher equal (>=) to 1 . here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    } else if (parameter_name.equals("metric")){
	    	if ( !parameter_value.equals("logloss")  && !parameter_value.equals("accuracy") && !parameter_value.equals("auc")){
	    		System.out.println("metric valid values are : logloss, accuracy, auc (for binary target)" );
			    System.exit(-1); // exiting the system	    		
	    	} else {
	    		metric=parameter_value;
	    	}
	    	
	    }else if (parameter_name.equals("stackdata")){
    		if (parameter_value.indexOf("false")!=-1  ){ 
    			restacking=false; 
    		} else if (parameter_value.indexOf("true")!=-1  ){ 
    			restacking=true; 
    		}else {
	    		System.out.println("the 'stackdata' parameters needs to be either 'true' or 'false' ");
			    System.exit(-1); // exiting the system
	    	}
	    		
	    }else if (parameter_name.equals("seed")){
	    	try{
	    		seed=Integer.parseInt(parameter_value);
	    	} catch (Exception e){
	    		System.out.println("parameter 'seed' needs to have an integer value . here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    	if (seed<0){
	    		System.out.println("parameter 'seed' needs to have an integer value higher equal (>=) to 0 . here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    	
	    	
	    }else if (parameter_name.equals("folds")){
	    	try{
	    		folds=Integer.parseInt(parameter_value);
	    	} catch (Exception e){
	    		System.out.println("parameter 'folds' needs to have an integer value . here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    	if (folds<=1){
	    		System.out.println("parameter 'folds' needs to have an integer value higher equal (>=) to 2 . here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    }
	    	
	}// end of arguments' loop
	    
	// check if there are adequate parameters 
	if (is_train && (train_file.equals("") || params_file.equals(""))){
		System.out.println("a train method needs to have at least a 'train_file' and a 'params_file' parameter ");
	    System.exit(-1); // exiting the system	
	}
	if (is_train==false && (test_file.equals("") || model_file.equals("")   )){
		System.out.println("a train method needs to have at least a 'train_file' and a 'model_file' parameter ");
	    System.exit(-1); // exiting the system	
	}
	
	//do train if "is train" command is activated
	if (is_train){
		
		// modelling objects
		fsmatrix X=null;
		fsmatrix X_test=null;
		smatrix Xsparse=null;
		smatrix Xsparse_test=null;
		double  y []= null;
		//input the data based on the format
		double starttime=System.currentTimeMillis()	;
		
		if (is_sparse==false){	
			 io.input in = new io.input();
			 in.delimeter=",";
	         in.HasHeader=has_head;
			in.targets_columns= new int[] {0};
			X= in.Readfmatrix(train_file);
			y= in.GetTarget();
			if (verbose){
				System.out.println("Loaded dense train data with " + X.GetRowDimension() + " and columns " + X.GetColumnDimension() );	
			}
			
		} else {
			 io.input in = new io.input();
			 Xsparse=in.readsmatrixdata(train_file, ":", has_head, true);
			 y=io.input.Retrievecolumn(train_file, " ", 0, 0.0, has_head, verbose);
			if (verbose){
					System.out.println("Loaded sparse train data with " + Xsparse.GetRowDimension() + " and columns " + Xsparse.GetColumnDimension() );	
			}
			
		}
		double load_data_time=System.currentTimeMillis()	;
		
		if (verbose){
			System.out.format(" loaded data in : %f\n", (load_data_time-starttime)/1000.0 );
		}
		// start the training 
		
		 stacknet= new StackNetClassifier();
		 
			String modellings[][]=io.input.StackNet_Configuration(params_file);
			/*an example of params
					new String[][] {
					{"LogisticRegression C:1 Type:Liblinear maxim_Iteration:100 scale:true verbose:false",
					"RandomForestClassifier bootsrap:false estimators:100 threads:5 logit.offset:0.00001 verbose:false cut_off_subsample:1.0 feature_subselection:1.0 gamma:0.00001 max_depth:8 max_features:0.25 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1",
					"GradientBoostingForestClassifier estimators:100 threads: offset:0.00001 verbose:false trees:1 rounding:2 shrinkage:0.05 cut_off_subsample:1.0 feature_subselection:0.8 gamma:0.00001 max_depth:8 max_features:1.0 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:RMSE row_subsample:0.9 seed:1" ,
					"Vanilla2hnnclassifier UseConstant:true usescale:true seed:1 Type:SGD maxim_Iteration:50 C:0.000001 learn_rate:0.009 smooth:0.02 h1:30 h2:20 connection_nonlinearity:Relu init_values:0.02"	,
					"LSVC Type:Liblinear threads:1 C:1.0 maxim_Iteration:100 seed:1",
					"LibFmClassifier lfeatures:3 init_values:0.035 smooth:0.05 learn_rate:0.1 threads:1 C:0.00001 maxim_Iteration:15 seed:1",
					"NaiveBayesClassifier usescale:true threads:1 Shrinkage:0.1 seed:1 verbose:false"
					},
					
					{"RandomForestClassifier estimators=1000 rounding:3 threads:4 max_depth:6 max_features:0.6 min_leaf:2.0 Objective:ENTROPY gamma:0.000001 row_subsample:1.0 verbose:false copy=false"}
					
			};
			*/
			stacknet.parameters=modellings;
			stacknet.threads=threads;
			stacknet.metric=metric;
			stacknet.folds=folds;
			stacknet.stackdata=restacking;
			stacknet.seed=seed;
			if (!output_name.equals("")){
				stacknet.print=true;
				stacknet.output_name=output_name;
			}
			stacknet.verbose=verbose;
			stacknet.target=y;	
			
			if (is_sparse){
				stacknet.fit(Xsparse);	
			} else {
				stacknet.fit(X);
			}
			double modelling_time=System.currentTimeMillis()	;
			
			if (verbose){
				System.out.format(" modelling lasted : %f\n", (modelling_time -load_data_time)/1000.0 );
			}
			
			try {
				Serialized_Object.save(model_file, (Serializable) stacknet);
			} catch (IOException e1) {
				System.out.println("model could not be exported at " + model_file);
			}
			
			// check if there is a test file. if there is one, then it gets loaded and predictions are made
			X=null;
			y=null;
			System.gc();// release memory via calling the garbage colector
			
			if (!test_file.equals("")){
				// there is a test file provided
				double load_time=System.currentTimeMillis()	;
				
				
				if (is_sparse==false){	
					 io.input in = new io.input();
					 in.delimeter=",";
			         in.HasHeader=has_head;
			         if (test_file_has_target){
					in.targets_columns= new int[] {0};
			         }
					X_test= in.Readfmatrix(test_file);
					y= in.GetTarget();
					if (verbose){
						System.out.println("Loaded dense test data with " + X_test.GetRowDimension() + " and columns " + X_test.GetColumnDimension() );	
					}
					//check if number of columns of the test data, matches the training data, if test data is more, it gets trimmed, BUT a message is displayed
					// ... and there is nothing you can do to stop it! It s for your own safety.
					if (X_test.GetColumnDimension()!=stacknet.get_predictors()){
						System.out.println("Warning : training column dimension is not the same with test " +X_test.GetColumnDimension()  + " <> " +  stacknet.get_predictors());
						if (X_test.GetColumnDimension()>stacknet.get_predictors()){
							System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknet.get_predictors());
							int cols[]= new int [stacknet.get_predictors()];
							for (int j=0; j <cols.length; j++ ){
								cols[j]=j;
							}
							X_test= X_test.makecolumnubset( cols );
						
						}
						
						
						
					}
					
					
				} else {
					 io.input in = new io.input();
					 Xsparse_test=in.readsmatrixdata(test_file, ":", has_head, test_file_has_target);
					 if (test_file_has_target){
						 y=io.input.Retrievecolumn(test_file, " ", 0, 0.0, has_head, verbose);
					 }
					if (verbose){
							System.out.println("Loaded sparse test data with " + Xsparse_test.GetRowDimension() + " and columns " + Xsparse_test.GetColumnDimension() );	
					}
					if (Xsparse_test.GetColumnDimension()!=stacknet.get_predictors()){
						System.out.println("Warning : training column dimension is not the same with test " +Xsparse_test.GetColumnDimension()  + " <> " +  stacknet.get_predictors());
						if (Xsparse_test.GetColumnDimension()>stacknet.get_predictors()){
							System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknet.get_predictors());
							Xsparse_test= Xsparse_test.makesubmatrixcols( stacknet.get_predictors());
						
						}
						
					
					}
					
				}
				double done_load_time=System.currentTimeMillis()	;
				if (verbose){
					System.out.format(" loading test data lasted : %f\n", (done_load_time -load_time)/1000.0 );
				}
				
				double [][] predictions=null;
				try {
				if (is_sparse==false){	
					predictions=stacknet.predict_proba(X_test);
				}else {
					predictions=stacknet.predict_proba(Xsparse_test);
				}
				}catch (Exception e){
					System.out.println(" prediction to has failed due to " + e.getMessage());
				} 
				
				//print predictions
				try {
				io.output out= new io.output();
				out.printdouble2d(predictions, pred_file);}
				catch (Exception e){
					System.out.println("printing prediction to  " + pred_file + " has failed due to " + e.getMessage());
				}
				
				double predict_test_time=System.currentTimeMillis()	;
				if (verbose){
					System.out.format(" predicting on test data lasted : %f\n", (predict_test_time-done_load_time )/1000.0 );
				}
				
				
				if (predictions[0].length==2){
					predictions=manipulate.select.columnselect.ColumnSelect(predictions, new int [] {1});

				}
				
				// metrics' calculation
				if (verbose && y!=null && predictions!=null){
					try{
						if(stacknet.getnumber_of_classes()==2 && metric.equals("auc")){
								double pr [] = manipulate.conversions.dimension.Convert(predictions);
								crossvalidation.metrics.Metric ms =new auc();
								double auc=ms.GetValue(pr,y ); // the auc for the current fold	
								System.out.println("Test AUC: " + auc);
							} else if (metric.equals("logloss")){
								
								double log=stacknet.logloss (predictions,y); // the logloss for the current fold	
								System.out.println("Test logloss : " + log);
								
								
							} else if (metric.equals("accuracy")){
								
								double acc=stacknet.accuracy (predictions,y); // the accuracy for the current fold	
								System.out.println("Test accuracy : " + acc);
								
							}
					} catch (Exception e){
						System.out.println("Metric could not be calculated on the test ");
					}
				}
				
				
				
				// end of checking the test file
			}
			
			
			double final_time=System.currentTimeMillis()	;
			if (verbose){
				System.out.format(" The whole StackNet procedure lasted: %f\n", (final_time-starttime )/1000.0 );
			}	
			
			
		
		
	} else {
		//if it is 'predict'
		
		try {
			stacknet = (StackNetClassifier) Serialized_Object.load(model_file);
		} catch (ClassNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			System.out.println("Loading  " + model_file + " has failed due to " + e1.getMessage());
			System.exit(-1);
		}
		
		fsmatrix X_test=null;
		smatrix Xsparse_test=null;
		double  y []= null;
		//input the data based on the format
		double starttime=System.currentTimeMillis()	;
		
		
		if (!test_file.equals("")){
			// there is a test file provided
			double load_time=System.currentTimeMillis()	;
			
			
			if (is_sparse==false){	
				 io.input in = new io.input();
				 in.delimeter=",";
		         in.HasHeader=has_head;
		         if (test_file_has_target){
				in.targets_columns= new int[] {0};
		         }
				X_test= in.Readfmatrix(test_file);
				y= in.GetTarget();
				if (verbose){
					System.out.println("Loaded dense test data with " + X_test.GetRowDimension() + " and columns " + X_test.GetColumnDimension() );	
				}
				//check if number of columns of the test data, matches the training data, if test data is more, it gets trimmed, BUT a message is displayed
				// ... and there is nothing you can do to stop it! It s for your own safety.
				if (X_test.GetColumnDimension()!=stacknet.get_predictors()){
					System.out.println("Warning : training column dimension is not the same with test " +X_test.GetColumnDimension()  + " <> " +  stacknet.get_predictors());
					if (X_test.GetColumnDimension()>stacknet.get_predictors()){
						System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknet.get_predictors());
						int cols[]= new int [stacknet.get_predictors()];
						for (int j=0; j <cols.length; j++ ){
							cols[j]=j;
						}
						X_test= X_test.makecolumnubset( cols );
					
					}
					
					
					
				}
				
				
			} else {
				 io.input in = new io.input();
				 Xsparse_test=in.readsmatrixdata(test_file, ":", has_head, test_file_has_target);
				 if (test_file_has_target){
					 y=io.input.Retrievecolumn(test_file, " ", 0, 0.0, has_head, verbose);
				 }
				if (verbose){
						System.out.println("Loaded sparse test data with " + Xsparse_test.GetRowDimension() + " and columns " + Xsparse_test.GetColumnDimension() );	
				}
				if (Xsparse_test.GetColumnDimension()!=stacknet.get_predictors()){
					System.out.println("Warning : training column dimension is not the same with test " +Xsparse_test.GetColumnDimension()  + " <> " +  stacknet.get_predictors());
					if (Xsparse_test.GetColumnDimension()>stacknet.get_predictors()){
						System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknet.get_predictors());
						Xsparse_test= Xsparse_test.makesubmatrixcols( stacknet.get_predictors());
					
					}
					
				
				}
				
			}
			double done_load_time=System.currentTimeMillis()	;
			if (verbose){
				System.out.format(" loading test data lasted : %f\n", (done_load_time -load_time)/1000.0 );
			}
			
			double [][] predictions=null;
			try {
			if (is_sparse==false){	
				predictions=stacknet.predict_proba(X_test);
			}else {
				predictions=stacknet.predict_proba(Xsparse_test);
			}
			}catch (Exception e){
				System.out.println(" prediction to has failed due to " + e.getMessage());
			} 
			
			//print predictions
			try {
			io.output out= new io.output();
			out.printdouble2d(predictions, pred_file);}
			catch (Exception e){
				System.out.println("printing prediction to  " + pred_file + " has failed due to " + e.getMessage());
			}
			
			double predict_test_time=System.currentTimeMillis()	;
			if (verbose){
				System.out.format(" predicting on test data lasted : %f\n", (predict_test_time-done_load_time )/1000.0 );
			}
			
			
			if (predictions[0].length==2){
				predictions=manipulate.select.columnselect.ColumnSelect(predictions, new int [] {1});

			}
			
			// metrics' calculation
			if (verbose && y!=null && predictions!=null){
				try{
					if(stacknet.getnumber_of_classes()==2 && metric.equals("auc")){
							double pr [] = manipulate.conversions.dimension.Convert(predictions);
							crossvalidation.metrics.Metric ms =new auc();
							double auc=ms.GetValue(pr,y ); // the auc for the current fold	
							System.out.println("Test AUC: " + auc);
						} else if (metric.equals("logloss")){
							
							double log=stacknet.logloss (predictions,y); // the logloss for the current fold	
							System.out.println("Test logloss : " + log);
							
							
						} else if (metric.equals("accuracy")){
							
							double acc=stacknet.accuracy (predictions,y); // the accuracy for the current fold	
							System.out.println("Test accuracy : " + acc);
							
						}
				} catch (Exception e){
					System.out.println("Metric could not be calculated on the test ");
				}
			}
			
			
			
			// end of checking the test file
		}
		
		
		double final_time=System.currentTimeMillis()	;
		if (verbose){
			System.out.format(" The whole StackNet predict procedure lasted: %f\n", (final_time-starttime )/1000.0 );
		}
		
		
		
		
	}
	
	

}
}	
	
