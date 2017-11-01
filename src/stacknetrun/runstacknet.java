package stacknetrun;

import io.Serialized_Object;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashSet;

import crossvalidation.metrics.auc;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.stacknet.StackNetClassifier;
import ml.stacknet.StackNetRegressor;

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
	 * name of file to load in order to form the train and test indices. This overrides the internal process for generating K-folds and ignores the given folds. 
	 */
	private static  String input_index="";	
	/**
	 * prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing
	 */
	private static String output_name="";	
	/**
	 * suffix for the names of kfold indices to be printed as .csvs . It will print as many files as the selected kfold with names [indices_name][fold_number].csv . It will have the format of "index,[0 if training else 1]"
	 */
	private static String indices_name="";
	/**
	 * prefix to be used when the user supplies own pairs of [X_train,X_cv] datasets for each fold  as well as an X file for the whole training data. Each train/valid pair is identified by prefix_'train'[fold_index_starting_from_zero]'.txt'/prefix_'cv'[fold_index_starting_from_zero]'.txt' and prefix_'train.txt' for the final set. 
	 */
	private static String data_prefix="";
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
	 * could be regression or classification .
	 */
	private static String task="";
	/**
	 * A parameter that allows classifiers to be used in regression problems. It first bins (digitises) the target variable and then runs classifiers on the transformed variable. Defaults to 2
	 */
	private static int bins=2;
	/**
	 * To allow StackNet to print stuff
	 */
	private static boolean verbose=true;
	/**
	 * True to enable printing the target column in the output file for train holdout predictions (when output_name is not empty). 
	 */
	private static  boolean include_target=false;
	/**
	 * number of model to run in parallel
	 */
	private static int threads=1;	
	/**
	 * Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) for classification and rmse ,rsquared or mae for regerssion .defaults to 'logloss' for classification and 'rmse' for regression
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
	 * stackNet classifier object to be used
	 */
	private static StackNetClassifier stacknet;
	/**
	 * stackNet regressor object to be used
	 */
	private static StackNetRegressor stacknetreg;
	
	/**
	 * @param args : arguments to train StackNet model
	 * <ul>
	 * <li> 'train' or 'predict' : to train or predict </li>
	 * <li> 'task' : could be either 'regression' or 'classification'.</li>
	 * <li> 'sparse' : if the data to be imported are in sparse format (libsvm) or dense . defaults to false</li>
	 * <li> 'has_head' : true if train_file and test_file have headers else false </li>
	 * <li> 'model' : prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing' </li>
	 * <li> 'output_name' : prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing </li>
	 * <li> 'indices_name' : suffix for the names of kfold indices to be printed as .csvs . It will print as many files as the selected kfold with names [indices_name][fold_number].csv . It will have the format of 'index,[0 if training else 1]' </li>
	 * <li> 'input_index' : name of file to load in order to form the train and cv indices during kfold cross validation. This overrides the internal process for generating kfolds and ignores the given folds.  </li>
	 * <li> 'pred_file' : name of the output prediction file. defaults to 'stacknet_pred.csv' </li>
	 * <li> 'data_prefix' : prefix to be used when the user supplies own pairs of [X_train,X_cv] datasets for each fold as well as a pair of whole [X,X_test] files. Each train/valid pair is identified by prefix_'train'[fold_index_starting_from_zero]'.txt'/prefix_'cv'[fold_index_starting_from_zero]'.txt' and prefix_'train.txt'/prefix_'test.txt' for the final sets. 
	 * <li> 'train_file' : name of the training file. </li>
	 * <li> 'test_file' : name of the test  file. </li>
	 * <li> 'test_target' : true if the test file has a target variable in the beginning (left) else false (only predictors in the file). </li> 
	 * <li> 'params' : parameter file where each line is a model. empty lines correspond to the creation of new levels </li>
	 * <li> 'verbose' : True if we need StackNet to output its progress .defaults to true. </li>
	 * <li> 'include_target' : True to enable printing the target column in the output file for train holdout predictions (when output_name is not empty). </li>
	 * <li> 'threads' : number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected).   </li>
	 * <li> 'metric' : Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) for classification and rmse ,rsquared or mae for regerssion .defaults to 'logloss' for classification and 'rmse' for regression </li>
	 * <li> 'stackdata' :True for <em>restacking</em>. defaults to true </li>
	 * <li> 'seed' : integer for randomised procedures.defaults to 1</li>
	 * <li> 'folds' : number of folds for re-usable kfold . defaults to 5</li>
	 * <li> 'bins' : A parameter that allows classifiers to be used in regression problems. It first bins (digitises) the target variable and then runs classifiers on the transformed variable. Defaults to 2</li>
	 * <li> 'help' : gives a few helpful tips</li>
	 * </ul>
	 */

	public static void main(String[] args) {
		// check if the first command is 'help' or empty
	    if (args.length==0 || args[0].toLowerCase().indexOf("help")!=-1 || args[0].toLowerCase().indexOf("h")!=-1 || args[0]==null){
	    	System.out.println("'train' or 'predict' : to train or predict \n"+
	    "'sparse' : true if the data to be imported are in sparse format (libsvm) or dense (false) \n" +	    
	    "'task' : could be either 'regression' or 'classification'. \n" +
	    "'has_head' : true if train_file and test_file have headers else false\n"+
	    "'model' : name of the output model file. \n"+
	    "'output_name' : prefix of the models to be printed per iteration. this is to allows the meta features of each iterations to be printed. defaults to nothing. \n"+	    
	    "'indices_name' : suffix for the names of kfold indices to be printed as .csvs . It will print as many files as the selected kfold with names [indices_name][fold_number].csv . It will have the format of 'index,[0 if training else 1]'\n"+	    
	    "'input_index' : name of file to load in order to form the train and cv indices during kfold cross validation. This overrides the internal process for generating kfolds and ignores the given folds. It needs to have the same size as the training data.The smallest indice forms the first cv dataset. The second smallest forms the second cv dataset and so on. \n"+	    
	    "'pred_file' : name of the output prediction file. \n"+
	    "'data_prefix' : prefix to be used when the user supplies own pairs of [X_train,X_cv] datasets for each fold as well as an X file for the whole training data. Each train/valid pair is identified by prefix_'train'[fold_index_starting_from_zero]'.txt'/prefix_'cv'[fold_index_starting_from_zero]'.txt' and prefix_'train.txt' for the final set. For example if prefix='mystack' and folds=2 then stacknet is expecting 2 pairs of train/cv files. e.g [[mystack_train0.txt,mystack_cv0.txt],[mystack_train1.txt,mystack_cv1.txt]]. It also expects a [mystack_train.txt]  for the final train set \n"+	    
	    "'train_file' : name of the training file. \n" +
	    "'test_file' : name of the test file. \n"+
	    "'test_target' : true if the test file has a target variable in the beginning (left) else false (only predictors in the file).\n" +
	    "'params' : parameter file where each line is a model. empty lines correspond to the creation of new levels \n"+
	    "'verbose' : true if we need StackNet to output its progress else false \n"+
	    "'include_target' : True to enable printing the target column in the output file for train holdout predictions (when output_name is not empty).\n"+	    
	    "'threads' : number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected) \n"+
	    "'metric' : Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) for classification and rmse ,rsquared or mae for regerssion .defaults to 'logloss' for classification and 'rmse' for regression \n"+
	    "'stackdata' :true for restacking else false\n"+
	    "'seed' : integer for randomised procedures \n"+
	    "'folds' : number of folds for re-usable kfold\n"+
	    "'bins' :  A parameter that allows classifiers to be used in regression problems. It first bins (digitises) the target variable and then runs classifiers on the transformed variable. Defaults to 2\n\n"+	    
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
	    	String option_upper=args[j];
	    	// check of the format is right
	    	if (option.indexOf("=")==-1){
	    		System.out.println(" Every option needs to have a '='. The format is 'option_name:option_value'. for example 'verbose=false'");
			    System.exit(-1); // exiting the system
	    	}
	    	String [] splits=option.split("=");
	    	String [] splits_upper=option_upper.split("=");
	    	if (splits.length!=2){
	    		System.out.println("There needs to be exactly one '=' in each option");
			    System.exit(-1); // exiting the system
	    	}
	    	// extract parameters' elements
	    	String parameter_name=splits[0];
	    	String parameter_value=splits[1];
	    	//String parameter_name_upper=splits_upper[0];
	    	String parameter_value_upper=splits_upper[1];
	    	
	    	
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
	    	model_file=parameter_value_upper;	
	    } else if (parameter_name.equals("task")){
	    	if (!parameter_value_upper.equals("regression") && !parameter_value_upper.equals("classification")){
	    		System.out.println("Valid values for 'task' are  : regression or  classification. Here it received " +  parameter_value_upper);
			    System.exit(-1); // exiting the system		    		
	    	}
	    	task=parameter_value_upper;	    	
	    }else if (parameter_name.equals("pred_file")){
	    	pred_file=parameter_value_upper;	
	    }else if (parameter_name.equals("train_file")){
	    	train_file=parameter_value_upper;	
	    }else if (parameter_name.equals("test_file")){
	    	test_file=parameter_value_upper;	
	    }else if (parameter_name.equals("output_name")){
	    	output_name=parameter_value_upper;	
	    }else if (parameter_name.equals("indices_name")){
	    	indices_name=parameter_value_upper;	
	    }else if (parameter_name.equals("data_prefix")){
	    	data_prefix=parameter_value_upper;	    
	    }else if (parameter_name.equals("input_index")){
	    	input_index=parameter_value_upper;	
	    }else if (parameter_name.equals("test_target")){   
    		if (parameter_value.indexOf("false")!=-1  ){ 
    			test_file_has_target=false; 
    		} else if (parameter_value.indexOf("true")!=-1  ){ 
    			test_file_has_target=true; 
    		}else {
	    		System.out.println("the 'test_target' parameters needs to be either 'true' or 'false' ");
			    System.exit(-1); // exiting the system
	    	}
	    }else if (parameter_name.equals("include_target")){   
    		if (parameter_value.indexOf("false")!=-1  ){ 
    			include_target=false; 
    		} else if (parameter_value.indexOf("true")!=-1  ){ 
    			include_target=true; 
    		}else {
	    		System.out.println("the'include_target' parameters needs to be either 'true' or 'false' ");
			    System.exit(-1); // exiting the system
	    	}	   
    		
	    }else if (parameter_name.equals("params")){
	    	params_file=parameter_value_upper;	
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
	    	if ( !parameter_value.equals("logloss")  && !parameter_value.equals("accuracy") && !parameter_value.equals("auc")&& !parameter_value.equals("rmse") && !parameter_value.equals("mae") && !parameter_value.equals("rsquared")){
	    		System.out.println("metric valid values are : logloss, accuracy, auc (for binary target) for classification and rmse and mae for regression" );
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
	    		System.out.println("parameter 'seed' needs to have an integer value higher equal (>=) to 0 . Here it received : " + parameter_value);
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
	    		System.out.println("parameter 'folds' needs to have an integer value higher equal (>=) to 2 . Here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    }else if (parameter_name.equals("bins")){
	    	try{
	    		bins=Integer.parseInt(parameter_value);
	    	} catch (Exception e){
	    		System.out.println("parameter 'bins' needs to have an integer value . here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    	if (bins<=1){
	    		System.out.println("parameter 'bins' needs to have an integer value higher equal (>=) to 2 . Here it received : " + parameter_value);
			    System.exit(-1); // exiting the system	
	    	}
	    }	    	
	    	
	}// end of arguments' loop
	if (is_train  && !task.equals("regression") && !task.equals("classification")) {
		System.out.println("a train method needs to have a task which may be regression or classification");
	    System.exit(-1); // exiting the system			
	}
	// check if there are adequate parameters 
	if (is_train && (train_file.equals("") || params_file.equals("")) && data_prefix.equals("") ){
		System.out.println("a train method needs to have at least a 'train_file', a 'params_file'");
	    System.exit(-1); // exiting the system	
	}
	if (is_train==false && (test_file.equals("") || model_file.equals("")   )){
		System.out.println("a predict method needs to have at least a 'train_file' and a 'model_file' parameter ");
	    System.exit(-1); // exiting the system	
	}
	if (is_train && !data_prefix.equals("")){
		// check if the files exist
		for (int f=0; f <folds;f++){
			File varTrain = new File(data_prefix +"_train" + f + ".txt");
			if (!varTrain.exists()){
				System.err.println(data_prefix +"_train" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
				System.exit(-1); // exiting the system	
			}
			File varcv = new File(data_prefix +"_cv" + f + ".txt");
			if (!varcv.exists()){
				System.err.println(data_prefix +"_cv" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
				System.exit(-1); // exiting the system	
			}			
		}
		File varTrain = new File(data_prefix +"_train.txt");
		if (!varTrain.exists()){
			System.err.println(data_prefix +"_train.txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/test files too apart from the train/cv files. Please use 'help' to see more details.");
			System.exit(-1); // exiting the system	
		}
		/*
		File varTest = new File(data_prefix +"_test.txt");
		if (!varTest.exists()){
			System.err.println(data_prefix +"_test.txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/test files too apart from the train/cv files. Please use 'help' to see more details.");
			System.exit(-1); // exiting the system	
		}	*/	
	}
	
	//do train if "is train" command is activated
	if (is_train){
		if (!input_index.equals("")){
			File indexTrain = new File(input_index);
			if (!indexTrain.exists()){
				System.err.println(indexTrain +" does not exist. since 'input_index' is given, StackNet expects a file with indices.");
				System.exit(-1); // exiting the system	
			}			
		}
		if (task.equals("regression") && !metric.equals("rmse") && !metric.equals("mae") && !metric.equals("rsquared") ){
			System.out.println(" metric will be set to rmse");
			metric="rmse";
		}
		if (task.equals("classification") && !metric.equals("logloss")  && !metric.equals("accuracy") && !metric.equals("auc") ){
			System.out.println(" metric will be set to logloss");
			metric="logloss";
		}		
		// modelling objects
		fsmatrix X=null;
		fsmatrix X_test=null;
		smatrix Xsparse=null;
		smatrix Xsparse_test=null;
		double  y []= null;
		//input the data based on the format
		double starttime=System.currentTimeMillis()	;
		
		if (!data_prefix.equals("")){
			if (verbose){
				System.out.println(" Data files are expected at the " + data_prefix + " location ");
			}
		}
		else if (is_sparse==false){	
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
		if (task.equals("classification")){
			 stacknet= new StackNetClassifier();
			 
				String modellings[][]=io.input.StackNet_Configuration(params_file);

				stacknet.parameters=modellings;
				stacknet.threads=threads;
				stacknet.metric=metric;
				stacknet.folds=folds;
				stacknet.input_index=input_index;
				stacknet.include_target=include_target;
				stacknet.stackdata=restacking;
				stacknet.seed=seed;
				if (!output_name.equals("")){
					stacknet.print=true;
					stacknet.output_name=output_name;
				}
				if (!indices_name.equals("")){
					stacknet.print_indices=true;
					stacknet.indices_name=indices_name;
				}			
				
				stacknet.verbose=verbose;
				stacknet.target=y;	
				
				if (!data_prefix.equals("")){
					if (is_sparse){
						stacknet.fit_sparse(data_prefix);
					} else {
					stacknet.fit_dense(data_prefix);
					}
				}
				else if (is_sparse){
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
					System.out.println("model could not be exported at " + model_file  + " because " + e1.getMessage());
				}
				
				// check if there is a test file. if there is one, then it gets loaded and predictions are made
				X=null;
				y=null;
				System.gc();// release memory via calling the garbage colector
			} else {
				
				 stacknetreg= new StackNetRegressor();
				 
					String modellings[][]=io.input.StackNet_Configuration(params_file);

					stacknetreg.parameters=modellings;
					stacknetreg.threads=threads;
					stacknetreg.metric=metric;
					stacknetreg.folds=folds;
					stacknetreg.input_index=input_index;
					stacknetreg.include_target=include_target;
					stacknetreg.bins=bins;
					stacknetreg.stackdata=restacking;
					stacknetreg.seed=seed;
					if (!output_name.equals("")){
						stacknetreg.print=true;
						stacknetreg.output_name=output_name;
					}
					if (!indices_name.equals("")){
						stacknetreg.print_indices=true;
						stacknetreg.indices_name=indices_name;
					}			
					
					stacknetreg.verbose=verbose;
					stacknetreg.target=y;	
					
					if (!data_prefix.equals("")){
						if (is_sparse){
							stacknetreg.fit_sparse(data_prefix);
						} else {
							stacknetreg.fit_dense(data_prefix);
						}
					}
					else if (is_sparse){
						stacknetreg.fit(Xsparse);	
					} else {
						stacknetreg.fit(X);
					}
					double modelling_time=System.currentTimeMillis()	;
					
					if (verbose){
						System.out.format(" modelling lasted : %f\n", (modelling_time -load_data_time)/1000.0 );
					}
					
					try {
						Serialized_Object.save(model_file, (Serializable) stacknetreg);
					} catch (IOException e1) {
						System.out.println("model could not be exported at " + model_file + " because " + e1.getMessage());
					}
					
					// check if there is a test file. if there is one, then it gets loaded and predictions are made
					X=null;
					y=null;
					System.gc();// release memory via calling the garbage colector			
				
				
				
			}
			if (!test_file.equals("")){
				// there is a test file provided
				double load_time=System.currentTimeMillis()	;
				double done_load_time=System.currentTimeMillis();
				double [][] predictions=null;
				if (task.equals("classification")){
					if (is_sparse==false){	
						 io.input in = new io.input();
						 in.delimeter=",";
				         in.HasHeader=has_head;
				         if (test_file_has_target){
						in.targets_columns= new int[] {0};
				         }
						X_test= in.Readfmatrix(test_file);
						  if (test_file_has_target){
						y= in.GetTarget();
						   }
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
							
							}else{
								int current_dim=Xsparse_test.GetColumnDimension();
								Xsparse_test.set_column_dimension(stacknet.get_predictors());
								System.out.println("Warning : test matrix will increase its column dimension from  " +current_dim + " to " +   stacknet.get_predictors());
							}
							
						}
						
					}
					done_load_time=System.currentTimeMillis()	;
					if (verbose){
						System.out.format(" loading test data lasted : %f\n", (done_load_time -load_time)/1000.0 );
					}
					
					
					try {
					if (is_sparse==false){	
						predictions=stacknet.predict_proba(X_test);
					}else {
						predictions=stacknet.predict_proba(Xsparse_test);
					}
					}catch (Exception e){
						System.out.println(" prediction to has failed due to " + e.getMessage());
					} 
				} else {
					if (is_sparse==false){	
						 io.input in = new io.input();
						 in.delimeter=",";
				         in.HasHeader=has_head;
				         if (test_file_has_target){
						in.targets_columns= new int[] {0};
				         }
						X_test= in.Readfmatrix(test_file);
						  if (test_file_has_target){
						y= in.GetTarget();
						   }
						if (verbose){
							System.out.println("Loaded dense test data with " + X_test.GetRowDimension() + " and columns " + X_test.GetColumnDimension() );	
						}
						//check if number of columns of the test data, matches the training data, if test data is more, it gets trimmed, BUT a message is displayed
						// ... and there is nothing you can do to stop it! It s for your own safety.
						if (X_test.GetColumnDimension()!=stacknetreg.get_predictors()){
							System.out.println("Warning : training column dimension is not the same with test " +X_test.GetColumnDimension()  + " <> " +  stacknetreg.get_predictors());
							if (X_test.GetColumnDimension()>stacknetreg.get_predictors()){
								System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknetreg.get_predictors());
								int cols[]= new int [stacknetreg.get_predictors()];
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
						if (Xsparse_test.GetColumnDimension()!=stacknetreg.get_predictors()){
							System.out.println("Warning : training column dimension is not the same with test " +Xsparse_test.GetColumnDimension()  + " <> " +  stacknetreg.get_predictors());
							if (Xsparse_test.GetColumnDimension()>stacknetreg.get_predictors()){
								System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknetreg.get_predictors());
								Xsparse_test= Xsparse_test.makesubmatrixcols( stacknetreg.get_predictors());
							
							}else{
								int current_dim=Xsparse_test.GetColumnDimension();
								Xsparse_test.set_column_dimension(stacknetreg.get_predictors());
								System.out.println("Warning : test matrix will increase its column dimension from  " +current_dim + " to " +   stacknetreg.get_predictors());
							}
							
						}
						
					}
					done_load_time=System.currentTimeMillis()	;
					if (verbose){
						System.out.format(" loading test data lasted : %f\n", (done_load_time -load_time)/1000.0 );
					}
					
					try {
					if (is_sparse==false){	
						predictions=stacknetreg.predict_proba(X_test);
					}else {
						predictions=stacknetreg.predict_proba(Xsparse_test);
					}
					}catch (Exception e){
						System.out.println(" prediction to has failed due to " + e.getMessage());
					} 				
					
					
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
						if(metric.equals("auc") && stacknet.getnumber_of_classes()==2 ){
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
								
							} else if (metric.equals("rmse")){
								
								double acc=stacknetreg.rmse (predictions,y); // the accuracy for the current fold	
								System.out.println("Test " + metric+ " : " + acc);	
							} else if (metric.equals("mae")){
								
								double acc=stacknetreg.mae (predictions,y); // the accuracy for the current fold	
								System.out.println("Test " + metric+ " : " + acc);	
								
							} else if (metric.equals("rsquared")){
								
								double acc=stacknetreg.rsquared (predictions,y); // the accuracy for the current fold	
								System.out.println("Test " + metric+ " : " + acc);									
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
			task="classification";
		} catch (Exception e) {
			// TODO Auto-generated catch block
		}
		try {
			stacknetreg = (StackNetRegressor) Serialized_Object.load(model_file);
			task="regression";
		} catch (Exception e) {
			System.out.format(" Loading Stacknet failed due to : " + e.getMessage());
		}
		
		if (stacknetreg==null && stacknet==null){
			System.out.println("failed to Load  " + model_file );
			System.exit(-1);
		}
		if (task.equals("regression") && !metric.equals("rmse") && !metric.equals("mae") && !metric.equals("rsquared") ){
			System.out.println(" metric will be set to rmse");
			metric="rmse";
		}
		if (task.equals("classification") && !metric.equals("logloss")  && !metric.equals("accuracy") && !metric.equals("auc") ){
			System.out.println(" metric will be set to logloss");
			metric="logloss";
		}	
		
		
		
		fsmatrix X_test=null;
		smatrix Xsparse_test=null;
		double  y []= null;
		//input the data based on the format
		double starttime=System.currentTimeMillis()	;
		
		
		if (!test_file.equals("")){
			// there is a test file provided
			double load_time=System.currentTimeMillis()	;
			double [][] predictions=null;
			double done_load_time=System.currentTimeMillis();
			if (task.equals("classification")){
			
				if (is_sparse==false){	
					 io.input in = new io.input();
					 in.delimeter=",";
			         in.HasHeader=has_head;
			         if (test_file_has_target){
					in.targets_columns= new int[] {0};
			         }
					X_test= in.Readfmatrix(test_file);
					  if (test_file_has_target){
					y= in.GetTarget();
					   }
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
						
						} else{
							int current_dim=Xsparse_test.GetColumnDimension();
							Xsparse_test.set_column_dimension(stacknet.get_predictors());
							System.out.println("Warning : test matrix will increase its column dimension from  " +current_dim + " to " +   stacknet.get_predictors());
						}
					}
					
				}
				done_load_time=System.currentTimeMillis()	;
				if (verbose){
					System.out.format(" loading test data lasted : %f\n", (done_load_time -load_time)/1000.0 );
				}
				
				
				try {
				if (is_sparse==false){	
					predictions=stacknet.predict_proba(X_test);
				}else {
					predictions=stacknet.predict_proba(Xsparse_test);
				}
				}catch (Exception e){
					System.out.println(" prediction to has failed due to " + e.getMessage());
				} 
			} else {
				
				if (is_sparse==false){	
					 io.input in = new io.input();
					 in.delimeter=",";
			         in.HasHeader=has_head;
			         if (test_file_has_target){
					in.targets_columns= new int[] {0};
			         }
					X_test= in.Readfmatrix(test_file);
					  if (test_file_has_target){
					y= in.GetTarget();
					   }
					if (verbose){
						System.out.println("Loaded dense test data with " + X_test.GetRowDimension() + " and columns " + X_test.GetColumnDimension() );	
					}
					//check if number of columns of the test data, matches the training data, if test data is more, it gets trimmed, BUT a message is displayed
					// ... and there is nothing you can do to stop it! It s for your own safety.
					if (X_test.GetColumnDimension()!=stacknetreg.get_predictors()){
						System.out.println("Warning : training column dimension is not the same with test " +X_test.GetColumnDimension()  + " <> " +  stacknetreg.get_predictors());
						if (X_test.GetColumnDimension()>stacknetreg.get_predictors()){
							System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknetreg.get_predictors());
							int cols[]= new int [stacknetreg.get_predictors()];
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
					if (Xsparse_test.GetColumnDimension()!=stacknetreg.get_predictors()){
						System.out.println("Warning : training column dimension is not the same with test " +Xsparse_test.GetColumnDimension()  + " <> " +  stacknetreg.get_predictors());
						if (Xsparse_test.GetColumnDimension()>stacknetreg.get_predictors()){
							System.out.println("Warning : test matrix gets its columns trimmed down to " +  stacknetreg.get_predictors());
							Xsparse_test= Xsparse_test.makesubmatrixcols( stacknetreg.get_predictors());
						
						} else{
							int current_dim=Xsparse_test.GetColumnDimension();
							Xsparse_test.set_column_dimension(stacknetreg.get_predictors());
							System.out.println("Warning : test matrix will increase its column dimension from  " +current_dim + " to " +   stacknetreg.get_predictors());
						}
						
					
					}
					
				}
				done_load_time=System.currentTimeMillis()	;
				if (verbose){
					System.out.format(" loading test data lasted : %f\n", (done_load_time -load_time)/1000.0 );
				}
				
				
				try {
				if (is_sparse==false){	
					predictions=stacknetreg.predict_proba(X_test);
				}else {
					predictions=stacknetreg.predict_proba(Xsparse_test);
				}
				}catch (Exception e){
					System.out.println(" prediction to has failed due to " + e.getMessage());
				}				

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
					if(metric.equals("auc") &&stacknet.getnumber_of_classes()==2 ){
							double pr [] = manipulate.conversions.dimension.Convert(predictions);
							crossvalidation.metrics.Metric ms =new auc();
							double auc=ms.GetValue(pr,y ); // the auc for the current fold	
							System.out.println("Test AUC: " + auc);
						} else if (metric.equals("logloss")){
							
							double log=stacknet.logloss (predictions,y); // the logloss for the current fold	
							System.out.println("Test " + metric+ " : " + log);
							
							
						} else if (metric.equals("accuracy")){
							
							double acc=stacknet.accuracy (predictions,y); // the accuracy for the current fold	
							System.out.println("Test " + metric+ " : " + acc);
							
						} else if (metric.equals("rmse")){
							
							double acc=stacknetreg.rmse (predictions,y); // the accuracy for the current fold	
							System.out.println("Test " + metric+ " : " + acc);	
						} else if (metric.equals("mae")){
							
							double acc=stacknetreg.mae (predictions,y); // the accuracy for the current fold	
							System.out.println("Test " + metric+ " : " + acc);	
							
						} else if (metric.equals("rsquared")){
							
							double acc=stacknetreg.rsquared (predictions,y); // the accuracy for the current fold	
							System.out.println("Test " + metric+ " : " + acc);									
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
	
System.exit(-1);
}
	
}	
	
