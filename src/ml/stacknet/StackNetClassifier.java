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

package ml.stacknet;
import io.input;
import io.readcsv;

import java.io.File;
import java.io.FileWriter;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import crossvalidation.metrics.auc;
import crossvalidation.splits.kfold;
import matrix.fsmatrix;
import matrix.smatrix;
import misc.print;
import ml.classifier;
import ml.estimator;
import ml.Bagging.BaggingClassifier;
import ml.Bagging.BaggingRegressor;
import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import utilis.map.intint.StringIntMap4a;
import exceptions.DimensionMismatchException;
import exceptions.IllegalStateException;
import exceptions.LessThanMinimum;

/**
 * 
 * @author Marios Michailidis
 * 
 * (equations removed because due to formating problems)
 * 
 * <H2> INTRODUCTION TO STACKNET </H2>
 * 
 * <p>implements STACKNET for classification given any number of input classifiers and/or regressors.
 * STACKNET uses stacked generalization into a feedforward neural network architecture to ensemble multiple models in classification problems.
 * <H2> THE LOGIC </H2>
 *<p>Given some input data , a neural network normally applies a perceptron along with a transformation function like relu or sigmoid, tanh or others.
 *<p> The STACKNET model assumes that this function can take the form of any supervised machine learning algorithm.
 * <p> Logically the outputs of each neuron , can be fed onto next layers.
 * <p> This logic could be generalised to any layer.
* <p> To create an output prediction score for any number of unique categories of the response variable, all selected algorithms in the last layer need to have output dimensionality equal to the number those unique classes.
 * In case where there are many such classifiers, the results is the scaled average of all these output predictions.
 * <H2> THE MODES </H2>
 * <p>The <em>stacking</em> element of the StackNet model could be run with 2 different modes. The first mode (e.g. the default) is the one already mentioned and assumes that in each layer uses the predictions (or output scores) of the direct previous one similar with a typical feedforward neural network or equivalently:
 * <p><b> Normal stacking mode</b> 
* <p>The second mode (also called restacking) assumes that each layer uses previous neurons activations as well as all previous layers neurons (including the input layer).
 * <p><b> Restacking mode</b> 
 * <p> Assuming the algorithm is located in layer n>1, to activate each neuron h in that layer, all outputs from all neurons from the previous n-1
 *  (or k) layers need to be accumulated (or stacked .The intuition behind this mode is drive from the fact that the higher level algorithm have extracted information from the input data, but rescanning the input space may yield new information not obvious from the first passes. This is also driven from the forward training methodology discussed below and assumes that convergence needs to happen within one model iteration
 * <H2> K-FOLD TRAINING</H2>
 * <p>The typical neural networks are most commonly trained with a form of back propagation, however stacked generalization requites a forward training methodology that splits the data into two parts – one of which is used for training and the other for predictions. The reason this split is necessary is to avoid the over fitting that could be a factor of the kind of algorithms being used as inputs as well as the absolute count of them.
 * <p> However splitting the data in just 2 parts would mean that in each new layer the second part needs to be further dichotomized increasing the bias of overfitting even more as each algorithm will have to be trained and validated on increasingly less data. 
To overcome this drawback the algorithm utilises a k-fold cross validation (where k is a hyper parameter) so that all the original training data is scored in different k batches thereby outputting n shape training predictions where n is the size of the samples in the training data. Therefore the training process is consisted of 2 parts:
<p> 1. Split the data k times and run k models to output predictions for each k part and then bring the k parts back together to the original order so that the output predictions can be used in later stages of the model. This process is illustrated below : 
<p> 2. Rerun the algorithm on the whole training data to be used later on for scoring the external test data. There is no reason to limit the ability of the model to learn using 100% of the training data since the output scoring is already unbiased (given that it is always scored as a holdout set).
<p> It should be noted that (1) is only applied during training to create unbiased predictions for the second layer model to fit one. During scoring time (and after model training is complete) only (2) is in effect.
<p> All models must be run sequentially based on the layers, but the order of the models within the layer does not matter. In other words all models of layer one need to be trained in order to proceed to layer two but all models within the layer can be run asynchronously and in parallel to save time.  
 The k-fold may also be viewed as a form of regularization where smaller number of folds (but higher than 1) ensure that the validation data is big enough to demonstrate how well a single model could generalize. On the other hand higher k means that the models come closer to running with 100% of the training and may yield more unexplained information. The best values could be found through cross validation. 
Another possible way to implement this could be to save all the k models and use the average of their predicting to score the unobserved test data, but this have all the models never trained with 100% of the training data and may be suboptimal. 
 * <H3> Final Notes</H3>
 * <p> STACKNET is commonly a better than the best single model it contains, however its ability to perform well still relies on a mix of string and diverse single models in order to get  the best out of this meta-modelling methodology.
 * 
 */
public class StackNetClassifier implements estimator,classifier, Serializable {

	/**
	 * list of classifiers to build for different levels
	 */
	private estimator[][]  tree_body ;
	
	/**
	 * column counts of each level
	 */
	private int column_counts[];
	/**
	 * list of classifiers's parameters to build for different levels
	 */
	public  String[] [] parameters ;
	/**
	 * threads to use
	 */
	public int threads=1;
	
	/**
	 * Print datasets after each level if True
	 */
	public boolean print=false;
	/**
	 * Print indices of kfold for the training data
	 */
	public boolean print_indices=false;
	/**
	 * Suffix for output files
	 */
	public String output_name="stacknet";
	/**
	 * name of file to load in order to form the train and test indices. This overrides the internal process for generating K-folds and ignores the given folds. 
	 */
	public String input_index="";	
	/**
	 * True to enable printing the target column in the left of the output file for train holdout predictions (when output_name is not empty). 
	 */
	public boolean include_target=false;
	/**
	 * prefix to be used when the user supplies own pairs of [X_train,X_cv] datasets for each fold  as well as a pair of whole [X,X_test] files. Each train/valid pair is identified by prefix_'train'[fold_index_starting_from_zero]'.txt'/prefix_'cv'[fold_index_starting_from_zero]'.txt' and prefix_'train.txt'/prefix_'test.txt' for the final sets. 
	 */
	private static String data_prefix="";
	/**
	 * Suffix for indices' files
	 */
	public String indices_name="stacknet_index";
	/**
	 * The metric to validate the results on. can be either logloss,  accuracy or auc (for binary only)
	 */
	public String metric="logloss";	
	/**
	 * stack the previous level data
	 */
	public boolean stackdata=false;
	/**
	 * To be used in tandem with 'data_prefix' to specify the type of files to load 
	 */
	public boolean is_sparse=false;	
	/**
	 * number of kfolds to run cv for
	 */
	public int folds=5;

	
	public  estimator[][] Get_tree(){
		if (this.tree_body==null || this.tree_body.length<=0){
			throw new IllegalStateException(" There is NO tree" );
		}
		return tree_body;
	}
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
	public int [] fstarget;	
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
	 * Default constructor for LinearRegression with no data
	 */
	public StackNetClassifier(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public StackNetClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public StackNetClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public StackNetClassifier(smatrix data){
		
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
		}	else if (!data_prefix.equals("") && this.is_sparse==true){
				this.fit_sparse(data_prefix);		
		} else {
			throw new IllegalStateException(" No data structure specifed in the constructor" );			
		}	
	}		

	/**
	 * 
	 * @param data_prefix2 : prefix of file to use to load sparse data
	 */

	public void fit_dense(String data_prefix2) {
		// make sensible checks

		data_prefix=data_prefix2;
		if (this.parameters.length<1 || (this.parameters[0].length<1) ){
			throw new IllegalStateException(" Parameters need to be provided in string format as model_name parameter_m:value_n ... " );
		}	
		if (parameters.length<2 && parameters[0].length==1){
			throw new IllegalStateException("StackNet cannot have only 1 model" );
		}	
		
		if ( !metric.equals("logloss")  && !metric.equals("accuracy") && !metric.equals("auc")){
			throw new IllegalStateException(" The metric to validate on needs to be one of logloss, accuracy or auc (for binary only) " );	
		}
		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}
		
		// make sensible checks on the target data
		File varTrain = new File(data_prefix +"_train.txt");
		if (!varTrain.exists()){
			System.err.println(data_prefix +"_train.txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/test files too apart from the train/cv files. Please use 'help' to see more details.");
			System.exit(-1); // exiting the system	
		}
		target=io.input.Retrievecolumn(data_prefix2 +"_train.txt", ",", 0, 0.0, false, verbose);	
		
		// temporary target
		int target_dimenson=0;
		for (int f=0; f < this.folds; f++){
			File varcv = new File(data_prefix +"_cv" + f + ".txt");
			if (!varcv.exists()){
				System.err.println(data_prefix +"_cv" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
				System.exit(-1); // exiting the system	
			}	
			
			int n_rows=input.GetRowCount(data_prefix +"_cv" + f + ".txt",false);
			target_dimenson+=n_rows;
		}
		
		double temp_target []= new double [target_dimenson];		
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

		// Initialise randomiser
		
		this.random = new XorShift128PlusRandom(this.seed);

		this.n_classes=classes.length;			

		if (!this.metric.equals("auc") && this.n_classes!=2){
			String last_case []=parameters[parameters.length-1];
			for (int d=0; d <last_case.length;d++){
				String splits[]=last_case[d].split(" " + "+");	
				String str_estimator=splits[0];
				boolean has_regressor_in_last_layer=false;
				if (str_estimator.contains("AdaboostForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("DecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("GradientBoostingForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("RandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("Vanilla2hnnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("multinnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LSVR")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LinearRegression")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LibFmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("knnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KernelmodelRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("XgboostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LightgbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODeepLearningRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGlmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODrfRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnAdaBoostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnDecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnExtraTreesRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnknnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnMLPRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnRandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnSGDRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnsvmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("PythonGenericRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KerasnnRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("FRGFRegressor")) {
					has_regressor_in_last_layer=true;
					
				}else if (str_estimator.contains("VowpaLWabbitRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("OriginalLibFMRegressor")) {
					has_regressor_in_last_layer=true;					
				}
				
				if (has_regressor_in_last_layer){
					throw new IllegalStateException("The last layer of StackNet cannot have a regressor unless the metric is auc and it is a binary problem" );
				}
			}
		}		
		fsmatrix data =null;
		fsmatrix trainstacker=null;
		tree_body= new estimator[parameters.length][];
		column_counts = new int[parameters.length];
		int kfolder [][][]=new int [this.folds][2][];
		for(int level=0; level<parameters.length; level++){
			
			// change the data 
			if (level>0){
				if (this.stackdata){
					
					double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+trainstacker.GetColumnDimension()];
					int ccc=0;
					for (int i=0; i <data.GetRowDimension(); i++ ){ 
						ccc=0;
						for (int j=0; j <data.GetColumnDimension(); j++ ){
							temp[i][ccc]=data.GetElement(i, j);
							ccc++;
						}
						for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
							temp[i][ccc]=trainstacker.GetElement(i, j);
							ccc++;
						}
					}
					
					data=new fsmatrix(temp);	
				}
				else {
					int ccc=0;
					 data =new fsmatrix(data.GetRowDimension(),trainstacker.GetColumnDimension());
					 for (int i=0; i <data.GetRowDimension(); i++ ){ 
							ccc=0;
							for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
								data.SetElement(i, ccc, trainstacker.GetElement(i, j));
								ccc++;
							}
						}					
					
				}
				
				
			}
			
			String [] level_grid=parameters[level];
			estimator[] mini_batch_tree= new estimator[level_grid.length];
			
			double metric_averages[]=new double [level_grid.length]; //holds stats for the level
			int model_count=0;
			
			Thread[] thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimator [] estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			int count_of_live_threads=0;


			int temp_class=estimate_classes(level_grid,  this.n_classes, level==(parameters.length-1));
			column_counts[level] = temp_class;
			if (this.verbose){
				System.out.println(" Level: " +  (level+1) + " dimensionality: " + temp_class);
				System.out.println(" Starting cross validation ");
			}
			if (level<parameters.length -1){
			trainstacker=new fsmatrix(temp_target.length, temp_class);

			int n_counter=0;
			

			// begin cross validation
			for (int f=0; f < this.folds; f++){
				
					int train_indices[]=null; // train indices
					int test_indices[]=null; // test indices	
					//System.out.println(" start!");
					fsmatrix X_train = null;
					fsmatrix X_cv  =null;
					double [] y_train=null;
					double [] y_cv= null;	
					
					int column_counter=0;
					
					if (level==0){
						
						File varTrainmini = new File(data_prefix +"_train" + f + ".txt");
						if (!varTrainmini.exists()){
							System.err.println(data_prefix +"_train" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
							System.exit(-1); // exiting the system	
						}
						File varcv = new File(data_prefix +"_cv" + f + ".txt");
						if (!varcv.exists()){
							System.err.println(data_prefix +"_cv" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
							System.exit(-1); // exiting the system	
						}	
						 io.input in = new io.input();
						 in.delimeter=",";
				         in.HasHeader=false;
				         in.verbose=false;
						 in.targets_columns= new int[] {0};
						 X_train= in.Readfmatrix(data_prefix +"_train" + f + ".txt");
						 y_train= in.GetTarget();
						 					
						 in = new io.input();
						 in.delimeter=",";
				         in.HasHeader=false;
				         in.verbose=false;
						 in.targets_columns= new int[] {0};
						 X_cv = in.Readfmatrix(data_prefix +"_cv" + f + ".txt");
						 y_cv= in.GetTarget();
						 if (X_train.GetColumnDimension()!=X_cv.GetColumnDimension()){
								System.err.println("ERROR : training column dimension at fold "  + f + " is not the same with cv " +X_train.GetColumnDimension()  + " <> " +  X_cv.GetColumnDimension());
								System.err.println("StackNet will terminate");
								System.exit(-1);
						 }
						 
						train_indices= new int [temp_target.length-y_cv.length];
						test_indices= new int [y_cv.length];
						int future_counter=n_counter+y_cv.length;
						int c_temp=0;
						for (int b=0; b <temp_target.length ; b++){
							if (b<n_counter || b>=future_counter){
								train_indices[c_temp]=b;
								c_temp++;	
								
							}
						}							
						for (int b=0; b <y_cv.length ; b++){
							test_indices[b]=n_counter;
							temp_target[n_counter]=y_cv[b];
							n_counter++;	
						}
						kfolder[f][0]=	train_indices;
						kfolder[f][1]=	test_indices;
						
					} else {
						train_indices=kfolder[f][0]; // train indices
						test_indices=kfolder[f][1]; // test indices	
						//System.out.println(" start!");
						X_train = data.makerowsubset(train_indices);
						X_cv  =data.makerowsubset(test_indices);
						y_train=manipulate.select.rowselect.RowSelect(temp_target, train_indices);
						y_cv=manipulate.select.rowselect.RowSelect(temp_target, test_indices);				
					}					
					
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
					
					
					for (int es=0; es <level_grid.length; es++ ){
						String splits[]=level_grid[es].split(" " + "+");	
						String str_estimator=splits[0];
						int bags=find_bags(level_grid[es]);
						if (containsClassifier(str_estimator)){
							BaggingClassifier mod = new BaggingClassifier(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
							mini_batch_tree[es].AddClassnames(this.classes);
						} else if (containsRegressor(str_estimator)){
							BaggingRegressor mod = new BaggingRegressor(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
						} else {
							throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
						}
						mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");
						
						mini_batch_tree[es].set_target(y_train);
		
						estimators[count_of_live_threads]=mini_batch_tree[es];
						thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
						thread_array[count_of_live_threads].start();
						count_of_live_threads++;
						if (this.verbose==true){
							System.out.println("Fitting model: " + (es+1));
							
						}
						
						if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
							for (int s=0; s <count_of_live_threads;s++ ){
								try {
									thread_array[s].join();
								} catch (InterruptedException e) {
								   System.out.println(e.getMessage());
								   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
								}
							}
							
							
							for (int s=0; s <count_of_live_threads;s++ ){
								double predictions[][]=estimators[s].predict_proba(X_cv);
								
								boolean is_regerssion=estimators[s].IsRegressor();
								if (predictions[0].length==2){
									predictions=manipulate.select.columnselect.ColumnSelect(predictions, new int [] {1});

								}
								// metrics' calculation
								if (this.verbose){
									if(this.n_classes==2 && this.metric.equals("auc")){
											double pr [] = manipulate.conversions.dimension.Convert(predictions);
											crossvalidation.metrics.Metric ms =new auc();
											double auc=ms.GetValue(pr,y_cv ); // the auc for the current fold	
											metric_averages[model_count]+=auc;
											System.out.println(" AUC: " + auc);
										} else if ( this.metric.equals("logloss")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												metric_averages[model_count]+=rms;
												System.out.println(" rmse : " + rms);
											}else {
											double log=logloss (predictions,y_cv ); // the logloss for the current fold	
											System.out.println(" logloss : " + log);
											metric_averages[model_count]+=log;
											}
											
										} else if (this.metric.equals("accuracy")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println(" rmse : " + rms);
												metric_averages[model_count]+=rms;
											}else {
												double acc=accuracy (predictions,y_cv ); // the accuracy for the current fold	
												System.out.println(" accuracy : " + acc);
												metric_averages[model_count]+=acc;
											}
										}
							}
								
								
								for (int j=0; j <predictions[0].length; j++ ){
									for (int i=0; i <predictions.length; i++ ){
										trainstacker.SetElement(test_indices[i], column_counter, predictions[i][j]);
									}
									column_counter+=1;
								}
								
								model_count+=1;
								
							}							
							
							System.gc();
							count_of_live_threads=0;
							thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
							estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
						}
						
						
				
					}
					if (this.verbose==true){
						System.out.println("Done with fold: " + (f+1) + "/" + this.folds);
						
					}
					model_count=0;	
			}
			if (this.print){
				
				if (this.verbose){
					
					System.out.println("Printing reusable train for level: " + (level+1) + " as : " + this.output_name +  (level+1)+ ".csv" );
				}
				if (include_target){
					trainstacker.ToFileTarget(this.output_name +  (level+1)+ ".csv",temp_target);
				}else {
					trainstacker.ToFile(this.output_name +  (level+1)+ ".csv");
				}
				
			}

			}

			
			if (this.verbose){
				for (int jj=0; jj< metric_averages.length;jj++ ){
					System.out.println(" Average of all folds model " +jj + " : "  + metric_averages[jj]/this.folds);
				}
				System.out.println(" Level: " +  (level+1)+ " start output modelling ");
			}
			
			if (level==0){
				File varTrainwhole = new File(data_prefix +"_train.txt");
				if (!varTrainwhole.exists()){
					System.err.println(data_prefix +"_train.txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/test files too apart from the train/cv files. Please use 'help' to see more details.");
					System.exit(-1); // exiting the system	
				}
				 io.input in = new io.input();
				 in.delimeter=",";
		         in.HasHeader=false;
		         in.verbose=false;
				 in.targets_columns= new int[] {0};
				 data= in.Readfmatrix(data_prefix +"_train.txt");
				 this.columndimension=data.GetColumnDimension();
			}
			
			thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			mini_batch_tree= new estimator[level_grid.length];
			count_of_live_threads=0;
			/* Final modelling */
			
			for (int es=0; es <level_grid.length; es++ ){
				String splits[]=level_grid[es].split(" " + "+");	
				String str_estimator=splits[0];
				int bags=find_bags(level_grid[es]);
				if (containsClassifier(str_estimator)){
					BaggingClassifier mod = new BaggingClassifier(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
					mini_batch_tree[es].AddClassnames(this.classes);
				} else if (containsRegressor(str_estimator)){
					BaggingRegressor mod = new BaggingRegressor(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
				} else {
					throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
				}
				mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");
				
				
				if (level==0){
				mini_batch_tree[es].set_target(this.target);
				} else {
					mini_batch_tree[es].set_target(temp_target); 
				}

				estimators[count_of_live_threads]=mini_batch_tree[es];
				thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting model : " + (es+1));
					
				}
				
				if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}

					System.gc();
					count_of_live_threads=0;
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
				}
				
			}			
			
			if (this.verbose==true){
				System.out.println("Completed level: " + (level+1) + " out of " + parameters.length);
				
			}
			
			// assign trained models in the main body
			
			this.tree_body[level]=mini_batch_tree;
			
			
		}
		target=null;
		trainstacker=null;
		sdataset=null;
		fsdataset=null;
		dataset=null;
		fstarget=null;

		System.gc();


		
	}	
	
/**
 * 
 * @param data_prefix2 : prefix of file to use to load sparse data
 */
	public void fit_sparse(String data_prefix2) {
	
		data_prefix=data_prefix2;
		
		if (this.parameters.length<1 || (this.parameters[0].length<1) ){
			throw new IllegalStateException(" Parameters need to be provided in string format as model_name parameter_m:value_n ... " );
		}	
		if (parameters.length<2 && parameters[0].length==1){
			throw new IllegalStateException("StackNet cannot have only 1 model" );
		}
		
		if ( !metric.equals("logloss")  && !metric.equals("accuracy") && !metric.equals("auc")){
			throw new IllegalStateException(" The metric to validate on needs to be one of logloss, accuracy or auc (for binary only) " );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}
		File varTrain = new File(data_prefix +"_train.txt");
		if (!varTrain.exists()){
			System.err.println(data_prefix +"_train.txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/test files too apart from the train/cv files. Please use 'help' to see more details.");
			System.exit(-1); // exiting the system	
		}
		target=io.input.Retrievecolumn(data_prefix2 +"_train.txt", " ", 0, 0.0, false, verbose);
		// temporary target
		int target_dimenson=0;
		for (int f=0; f < this.folds; f++){
			File varcv = new File(data_prefix +"_cv" + f + ".txt");
			if (!varcv.exists()){
				System.err.println(data_prefix +"_cv" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
				System.exit(-1); // exiting the system	
			}	
			
			int n_rows=input.GetRowCount(data_prefix +"_cv" + f + ".txt",false);
			target_dimenson+=n_rows;
		}
		
		double temp_target []= new double [target_dimenson];
		
		
		
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


		// Initialise randomiser
		
		this.random = new XorShift128PlusRandom(this.seed);

		this.n_classes=classes.length;			

		if (!this.metric.equals("auc") && this.n_classes!=2){
			String last_case []=parameters[parameters.length-1];
			for (int d=0; d <last_case.length;d++){
				String splits[]=last_case[d].split(" " + "+");	
				String str_estimator=splits[0];
				boolean has_regressor_in_last_layer=false;
				if (str_estimator.contains("AdaboostForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("DecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("GradientBoostingForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("RandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("Vanilla2hnnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("multinnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LSVR")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LinearRegression")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LibFmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("knnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KernelmodelRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("XgboostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LightgbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODeepLearningRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGlmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODrfRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnAdaBoostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnDecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnExtraTreesRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnknnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnMLPRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnRandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnSGDRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnsvmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("PythonGenericRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KerasnnRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("FRGFRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("VowpaLWabbitRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("OriginalLibFMRegressor")) {
					has_regressor_in_last_layer=true;					
				}
				
				
				if (has_regressor_in_last_layer){
					throw new IllegalStateException("The last layer of StackNet cannot have a regressor unless the metric is auc and it is a binary problem" );
				}
			}
		}		
		
		smatrix data =null;
		fsmatrix trainstacker=null;
		tree_body= new estimator[parameters.length][];
		column_counts = new int[parameters.length];
		int kfolder [][][]=new int [this.folds][2][];
		
		for(int level=0; level<parameters.length; level++){
			
			// change the data 
			if (level>0){
				/*
				if (this.stackdata){
					
					double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+trainstacker.GetColumnDimension()];
					int ccc=0;
					for (int i=0; i <data.GetRowDimension(); i++ ){ 
						ccc=0;
						for (int j=0; j <data.GetColumnDimension(); j++ ){
							temp[i][ccc]=data.GetElement(i, j);
							ccc++;
						}
						for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
							temp[i][ccc]=trainstacker.GetElement(i, j);
							ccc++;
						}
					}
					
					data=new smatrix(temp);	
				}
				else {*/
					 data =new smatrix(trainstacker);
					
					
				//}
				
				
			}
			
			String [] level_grid=parameters[level];
			estimator[] mini_batch_tree= new estimator[level_grid.length];
			double metric_averages[]=new double [level_grid.length]; //holds stats for the level
			int model_count=0;
			Thread[] thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimator [] estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			int count_of_live_threads=0;
			
			int temp_class=estimate_classes(level_grid,  this.n_classes, level==(parameters.length-1));
			column_counts[level] = temp_class;
			
			if (this.verbose){
				System.out.println(" Level: " +  (level+1) + " dimensionality: " + temp_class);
				System.out.println(" Starting cross validation ");
			}
			if (level<parameters.length -1){
			trainstacker=new fsmatrix(temp_target.length, temp_class);

			int n_counter=0;
			//print indices if selected
		
			// begin cross validation
			for (int f=0; f < this.folds; f++){
					int column_counter=0;	
					
					int train_indices[]=null; // train indices
					int test_indices[]=null; // test indices	
					//System.out.println(" start!");
					smatrix X_train = null;
					smatrix X_cv  =null;
					double [] y_train=null;
					double [] y_cv= null;	
					
					if (level==0){
						
						File varTrainmini = new File(data_prefix +"_train" + f + ".txt");
						if (!varTrainmini.exists()){
							System.err.println(data_prefix +"_train" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
							System.exit(-1); // exiting the system	
						}
						File varcv = new File(data_prefix +"_cv" + f + ".txt");
						if (!varcv.exists()){
							System.err.println(data_prefix +"_cv" + f + ".txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/cv files. Please use 'help' to see more details.");
							System.exit(-1); // exiting the system	
						}	
						 io.input in = new io.input();
						 X_train=in.readsmatrixdata(data_prefix +"_train" + f + ".txt", ":", false, true);
						 y_train=io.input.Retrievecolumn(data_prefix +"_train" + f + ".txt", " ", 0, 0.0, false, false);						
							if (verbose){
								System.out.println("Loaded sparse train data at fold " + f + " with " + X_train.GetRowDimension() + " and columns " + X_train.GetColumnDimension() );	
						}					
							in = new io.input();
							X_cv=in.readsmatrixdata(data_prefix +"_cv" + f + ".txt", ":", false, true);
							y_cv=io.input.Retrievecolumn(data_prefix +"_cv" + f + ".txt", " ", 0, 0.0, false, false);
							if (verbose){
								System.out.println("Loaded sparse cv data at fold " + f + " with " + X_cv.GetRowDimension() + " and columns " + X_cv.GetColumnDimension() );	
						}	
							
							train_indices= new int [temp_target.length-y_cv.length];
							test_indices= new int [y_cv.length];
							int future_counter=n_counter+y_cv.length;
							int c_temp=0;
							for (int b=0; b <temp_target.length ; b++){
								if (b<n_counter || b>=future_counter){
									train_indices[c_temp]=b;
									c_temp++;	
									
								}
							}							
							for (int b=0; b <y_cv.length ; b++){
								test_indices[b]=n_counter;
								temp_target[n_counter]=y_cv[b];
								n_counter++;	
							}
							kfolder[f][0]=	train_indices;
							kfolder[f][1]=	test_indices;
						
					} else {
						train_indices=kfolder[f][0]; // train indices
						test_indices=kfolder[f][1]; // test indices	
						//System.out.println(" start!");
						X_train = data.makesubmatrix(train_indices);
						X_cv  =data.makesubmatrix(test_indices);
						y_train=manipulate.select.rowselect.RowSelect(temp_target, train_indices);
						y_cv=manipulate.select.rowselect.RowSelect(temp_target, test_indices);				
					}
					
					if (X_train.GetColumnDimension()!=X_cv.GetColumnDimension()){
						if (verbose){
						System.out.println("Warning : training column dimension at fold "  + f + " is not the same with cv " +X_train.GetColumnDimension()  + " <> " +  X_cv.GetColumnDimension());
						}
						if (X_cv.GetColumnDimension()>X_train.GetColumnDimension()){
							if (verbose){
							System.out.println("Warning : cv matrix  at fold " + f + " gets its columns trimmed down to " +  X_train.GetColumnDimension());
							}
							X_cv= X_cv.makesubmatrixcols( X_train.GetColumnDimension());
						
						}else{
							int current_dim=X_cv.GetColumnDimension();
							X_cv.set_column_dimension(X_train.GetColumnDimension());
							if (verbose){
							System.out.println("Warning : cv matrix  at fold " + f + " will increase its column dimension from  " +current_dim + " to " +   X_train.GetColumnDimension());
							}
						}
						
					}

					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
					
					
					for (int es=0; es <level_grid.length; es++ ){
						String splits[]=level_grid[es].split(" " + "+");	
						String str_estimator=splits[0];
 						int bags=find_bags(level_grid[es]);
						if (containsClassifier(str_estimator)){
							BaggingClassifier mod = new BaggingClassifier(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
							mini_batch_tree[es].AddClassnames(this.classes);
						} else if (containsRegressor(str_estimator)){
							BaggingRegressor mod = new BaggingRegressor(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
						} else {
							throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
						}
						mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");
						mini_batch_tree[es].set_target(y_train);
		
						estimators[count_of_live_threads]=mini_batch_tree[es];
						thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
						thread_array[count_of_live_threads].start();
						count_of_live_threads++;
						if (this.verbose==true){
							System.out.println("Fitting model : " + es);
							
						}
						
						if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
							for (int s=0; s <count_of_live_threads;s++ ){
								try {
									thread_array[s].join();
								} catch (InterruptedException e) {
								   System.out.println(e.getMessage());
								   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
								}
							}
							
							
							for (int s=0; s <count_of_live_threads;s++ ){
								double predictions[][]=estimators[s].predict_proba(X_cv);
								boolean is_regerssion=estimators[s].IsRegressor();
								if (predictions[0].length==2){
									predictions=manipulate.select.columnselect.ColumnSelect(predictions, new int [] {1});
								}
								
								if (this.verbose){
									if(this.n_classes==2 && this.metric.equals("auc")){
											double pr [] = manipulate.conversions.dimension.Convert(predictions);
											crossvalidation.metrics.Metric ms =new auc();
											double auc=ms.GetValue(pr,y_cv ); // the auc for the current fold	
											System.out.println(" AUC: " + auc);
											metric_averages[model_count]+=auc;											
										} else if ( this.metric.equals("logloss")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println("rmse : " + rms);
												metric_averages[model_count]+=rms;												
											}else {
											double log=logloss (predictions,y_cv ); // the logloss for the current fold	
											System.out.println("logloss : " + log);
											metric_averages[model_count]+=log;			
											}
											
										} else if (this.metric.equals("accuracy")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println("rmse : " + rms);
												metric_averages[model_count]+=rms;	
											}else {
												double acc=accuracy (predictions,y_cv ); // the accuracy for the current fold	
												System.out.println("accuracy : " + acc);
												metric_averages[model_count]+=acc;	
											}
										}
							}
								for (int j=0; j <predictions[0].length; j++ ){
									for (int i=0; i <predictions.length; i++ ){
										trainstacker.SetElement(test_indices[i], column_counter, predictions[i][j]);
									}
									column_counter+=1;
								}
								
								model_count+=1;
								
							}							
							
							System.gc();
							count_of_live_threads=0;
							thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
							estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
						}
						
						

					}
					
					if (this.verbose==true){
						System.out.println("Done with fold: " + (f+1) + "/" + this.folds);
						
					}
					model_count=0;	
			}
			if (this.print){
				
				if (this.verbose){
				
					
					System.out.println("Printing reusable train for level: " + (level+1) + " as : " + this.output_name +  (level+1)+ ".csv" );
				}
				if (include_target){
					trainstacker.ToFileTarget(this.output_name +  (level+1)+ ".csv",temp_target);
				}else {
					trainstacker.ToFile(this.output_name +  (level+1)+ ".csv");
				}
				
			}
			}

			
			if (this.verbose){
				
				for (int jj=0; jj< metric_averages.length;jj++ ){
					System.out.println(" Average of all folds model " +jj + " : "  + metric_averages[jj]/this.folds);
				}				
				
				System.out.println(" Level: " +  (level+1)+ " start output modelling ");
			}
			if (level==0){
				File varTrainwhole = new File(data_prefix +"_train.txt");
				if (!varTrainwhole.exists()){
					System.err.println(data_prefix +"_train.txt does not exist. since 'data_prefix is given, stacknet expects pairs of train/test files too apart from the train/cv files. Please use 'help' to see more details.");
					System.exit(-1); // exiting the system	
				}
				 io.input in = new io.input();
				 data=in.readsmatrixdata(data_prefix +"_train.txt", ":", false, true);	
				 this.columndimension=data.GetColumnDimension();
			}
			thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			mini_batch_tree= new estimator[level_grid.length];
			count_of_live_threads=0;
			/* Final modelling */
			
			for (int es=0; es <level_grid.length; es++ ){
				String splits[]=level_grid[es].split(" " + "+");	
				String str_estimator=splits[0];
				
				int bags=find_bags(level_grid[es]);
				if (containsClassifier(str_estimator)){
					BaggingClassifier mod = new BaggingClassifier(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
					mini_batch_tree[es].AddClassnames(this.classes);
				} else if (containsRegressor(str_estimator)){
					BaggingRegressor mod = new BaggingRegressor(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
				} else {
					throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
				}
				mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");
				
				if (level==0){
				mini_batch_tree[es].set_target(this.target);
				} else {
					mini_batch_tree[es].set_target(temp_target); 
				}

				estimators[count_of_live_threads]=mini_batch_tree[es];
				thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting model : " + es);
					
				}
				
				if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}

					System.gc();
					count_of_live_threads=0;
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
				}
				
			}			
			
			if (this.verbose==true){
				System.out.println("Completed level: " + (level+1) + " out of " + parameters.length);
				
			}
			
			// assign trained models in the main body
			
			this.tree_body[level]=mini_batch_tree;
			
			
		}
		target=null;
		trainstacker=null;
		sdataset=null;
		fsdataset=null;
		dataset=null;
		fstarget=null;
		System.gc();	
	}
	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8611561535854392960L;
	@Override
	public double[][] predict_proba(double[][] data) {
		 
		/*  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
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
		fsmatrix arrays =null;
		
		for(int level=0; level<tree_body.length; level++){
			int column_counter=0;
			arrays= new fsmatrix(predictions.length, this.column_counts[level]);
			for (estimator k : tree_body[level]){
				double preds[][]=k.predict_proba(data);
				if (preds[0].length==2 && level <tree_body.length-1){
					preds=manipulate.select.columnselect.ColumnSelect(preds, new int [] {1});
				}
				for (int j=0; j <preds[0].length; j++ ){
					for (int i=0; i <preds.length; i++ ){
						arrays.SetElement(i, column_counter, preds[i][j]);
					}
					column_counter+=1;
				}				
			}
			
			if (this.stackdata){
				
				double temp[][] = new double [data.length][data[0].length+arrays.GetColumnDimension()];
				int ccc=0;
				for (int i=0; i <data.length; i++ ){ 
					ccc=0;
					for (int j=0; j <data[0].length; j++ ){
						temp[i][ccc]=data[i][j];
						ccc++;
					}
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						temp[i][ccc]=arrays.GetElement(i, j);
						ccc++;
					}
				}
				
				data=temp;	
			}
			else {
				int ccc=0;
				 data =new double [data.length][arrays.GetColumnDimension()] ;
				 for (int i=0; i <data.length; i++ ){ 
						ccc=0;
						for (int j=0; j <arrays.GetColumnDimension(); j++ ){
							data[i][ccc]=arrays.GetElement(i, j);
							ccc++;
						}
					}					
				
			}
			
			if (this.print){
				
				if (this.verbose){
					
					System.out.println("Printing reusable test for level: " + (level+1) + " as : " + this.output_name +"_test" +  (level+1)+ ".csv");
				}
				arrays.ToFile(this.output_name +"_test" +  (level+1)+ ".csv");
				
			}		
			
		}
		
		if (arrays.GetColumnDimension()%this.n_classes!=0){
			 throw new IllegalStateException("Number of final model's output columns need to be a factor of the used classes");  
		}
		int multi=arrays.GetColumnDimension()/this.n_classes;
		
			for (int i=0; i <predictions.length; i++ ){
				for (int m=0; m <multi; m++ ){				
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						int col=arrays.GetColumnDimension() * m + j ;
						predictions[i][j]+=arrays.GetElement(i, col);
					}
				}
			}
		
			scale_scores(predictions);
		
			// return the 1st prediction
			return predictions;
			
			}

	@Override
	public double[][] predict_proba(fsmatrix data) {
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
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
		fsmatrix arrays =null;
		
		for(int level=0; level<tree_body.length; level++){
			int column_counter=0;
			arrays= new fsmatrix(predictions.length, this.column_counts[level]);
			for (estimator k : tree_body[level]){
				double preds[][]=k.predict_proba(data);
				if (preds[0].length==2 && level <tree_body.length-1){
					preds=manipulate.select.columnselect.ColumnSelect(preds, new int [] {1});
				}
				for (int j=0; j <preds[0].length; j++ ){
					for (int i=0; i <preds.length; i++ ){
						arrays.SetElement(i, column_counter, preds[i][j]);
					}
					column_counter+=1;
				}				
			}
			
			if (this.stackdata){
				
				
				double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+arrays.GetColumnDimension()];
				int ccc=0;
				for (int i=0; i <data.GetRowDimension(); i++ ){ 
					ccc=0;
					for (int j=0; j <data.GetColumnDimension(); j++ ){
						temp[i][ccc]=data.GetElement(i, j);
						ccc++;
					}
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						temp[i][ccc]=arrays.GetElement(i, j);
						ccc++;
					}
				}
				
				data=new fsmatrix(temp);	
			}
			else {
				int ccc=0;
				 data =new fsmatrix(data.GetRowDimension(),arrays.GetColumnDimension());
				 for (int i=0; i <data.GetRowDimension(); i++ ){ 
						ccc=0;
						for (int j=0; j <arrays.GetColumnDimension(); j++ ){
							data.SetElement(i, ccc, arrays.GetElement(i, j));
							ccc++;
						}
					}					
				
			}
			if (this.print){
				
				if (this.verbose){
					
					System.out.println("Printing reusable test for level: " + (level+1) + " as : " + this.output_name +"_test" +  (level+1)+ ".csv");
				}
				arrays.ToFile(this.output_name +"_test" +  (level+1)+ ".csv");
				
			}	
			
		}
		
		if (arrays.GetColumnDimension()%this.n_classes!=0){
			 throw new IllegalStateException("Number of final model's output columns need to be a factor of the used classes");  
		}
		int multi=arrays.GetColumnDimension()/this.n_classes;
		
			for (int i=0; i <predictions.length; i++ ){
				for (int m=0; m <multi; m++ ){				
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						int col=arrays.GetColumnDimension() * m + j ;
						predictions[i][j]+=arrays.GetElement(i, col);
					}
				}
			}
		
			scale_scores(predictions);
		

			// return the 1st prediction
			return predictions;
		
			}

	@Override
	public double[][] predict_proba(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
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
		if (data.indexer==null){
			data.buildmap();;
		}
		
		this.stackdata=false;
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		fsmatrix arrays =null;
		
		for(int level=0; level<tree_body.length; level++){
			int column_counter=0;
			arrays= new fsmatrix(predictions.length, this.column_counts[level]);
			for (estimator k : tree_body[level]){
				double preds[][]=k.predict_proba(data);
				if (preds[0].length==2 && level <tree_body.length-1){
					preds=manipulate.select.columnselect.ColumnSelect(preds, new int [] {1});
				}
				for (int j=0; j <preds[0].length; j++ ){
					for (int i=0; i <preds.length; i++ ){
						arrays.SetElement(i, column_counter, preds[i][j]);
					}
					column_counter+=1;
				}				
			}
			
			if (this.stackdata){
				
				
				double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+arrays.GetColumnDimension()];
				int ccc=0;
				for (int i=0; i <data.GetRowDimension(); i++ ){ 
					ccc=0;
					for (int j=0; j <data.GetColumnDimension(); j++ ){
						temp[i][ccc]=data.GetElement(i, j);
						ccc++;
					}
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						temp[i][ccc]=arrays.GetElement(i, j);
						ccc++;
					}
				}
				
				data=new smatrix(temp);	
			}
			else {

				data=new smatrix(arrays);
					
				
			}
			
			if (this.print){
				
				if (this.verbose){
					
					System.out.println("Printing reusable test for level: " + (level+1) + " as : " + this.output_name +"_test" +  (level+1)+ ".csv");
				}
				arrays.ToFile(this.output_name +"_test" +  (level+1)+ ".csv");
				
			}	
		}
		
		if (arrays.GetColumnDimension()%this.n_classes!=0){
			 throw new IllegalStateException("Number of final model's output columns need to be a factor of the used classes");  
		}
		int multi=arrays.GetColumnDimension()/this.n_classes;
		
			for (int i=0; i <predictions.length; i++ ){
				for (int m=0; m <multi; m++ ){				
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						int col=arrays.GetColumnDimension() * m + j ;
						predictions[i][j]+=arrays.GetElement(i, col);
					}
				}
			}
		
			scale_scores(predictions);
		

			// return the 1st prediction
			return predictions;
	}

	@Override
	public double[] predict_probaRow(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	

		
		double predictions[]= new double [this.n_classes];


			// return the 1st prediction
			return predictions;
			}


	@Override
	public double[] predict_probaRow(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.tree_body==null || this.tree_body.length<=0  ){
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


		
		// return the 1st prediction
		return predictions;		
			
	}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.tree_body==null || this.tree_body.length<=0  ){
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

		// return the 1st prediction
		return predictions;
			}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
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
		
		double predictionsclass[]= new double [data.GetRowDimension()];
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		fsmatrix arrays =null;
		
		for(int level=0; level<tree_body.length; level++){
			int column_counter=0;
			arrays= new fsmatrix(predictions.length, this.column_counts[level]);
			for (estimator k : tree_body[level]){
				double preds[][]=k.predict_proba(data);
				if (preds[0].length==2 && level <tree_body.length-1){
					preds=manipulate.select.columnselect.ColumnSelect(preds, new int [] {1});
				}
				for (int j=0; j <preds[0].length; j++ ){
					for (int i=0; i <preds.length; i++ ){
						arrays.SetElement(i, column_counter, preds[i][j]);
					}
					column_counter+=1;
				}				
			}
			
			if (this.stackdata){
				
				
				double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+arrays.GetColumnDimension()];
				int ccc=0;
				for (int i=0; i <data.GetRowDimension(); i++ ){ 
					ccc=0;
					for (int j=0; j <data.GetColumnDimension(); j++ ){
						temp[i][ccc]=data.GetElement(i, j);
						ccc++;
					}
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						temp[i][ccc]=arrays.GetElement(i, j);
						ccc++;
					}
				}
				
				data=new fsmatrix(temp);	
			}
			else {
				int ccc=0;
				 data =new fsmatrix(data.GetRowDimension(),arrays.GetColumnDimension());
				 for (int i=0; i <data.GetRowDimension(); i++ ){ 
						ccc=0;
						for (int j=0; j <arrays.GetColumnDimension(); j++ ){
							data.SetElement(i, ccc, arrays.GetElement(i, j));
							ccc++;
						}
					}					
				
			}
			
			
		}
		
		if (arrays.GetColumnDimension()%this.n_classes!=0){
			 throw new IllegalStateException("Number of final model's output columns need to be a factor of the used classes");  
		}
		int multi=arrays.GetColumnDimension()/this.n_classes;
		
			for (int i=0; i <predictions.length; i++ ){
				for (int m=0; m <multi; m++ ){				
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						int col=arrays.GetColumnDimension() * m + j ;
						predictions[i][j]+=arrays.GetElement(i, col);
					}
				}
			}
		


		
			// return the 1st prediction

		for (int i=0; i < predictionsclass.length; i++) {
			double temp[]=predictions[i];
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictionsclass[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictionsclass[i]=maxi;
	    	  }

		}		
		
		predictions=null;

			// return the 1st prediction
			return predictionsclass;
			
			}
			

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
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
		if (data.indexer==null){
			data.buildmap();
		}
		this.stackdata=false;
		double predictionsclass[]= new double [data.GetRowDimension()];
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		fsmatrix arrays =null;
		
		for(int level=0; level<tree_body.length; level++){
			int column_counter=0;
			arrays= new fsmatrix(predictions.length, this.column_counts[level]);
			for (estimator k : tree_body[level]){
				double preds[][]=k.predict_proba(data);
				if (preds[0].length==2 && level <tree_body.length-1){
					preds=manipulate.select.columnselect.ColumnSelect(preds, new int [] {1});
				}
				for (int j=0; j <preds[0].length; j++ ){
					for (int i=0; i <preds.length; i++ ){
						arrays.SetElement(i, column_counter, preds[i][j]);
					}
					column_counter+=1;
				}				
			}
			
			if (this.stackdata){
				
				
				double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+arrays.GetColumnDimension()];
				int ccc=0;
				for (int i=0; i <data.GetRowDimension(); i++ ){ 
					ccc=0;
					for (int j=0; j <data.GetColumnDimension(); j++ ){
						temp[i][ccc]=data.GetElement(i, j);
						ccc++;
					}
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						temp[i][ccc]=arrays.GetElement(i, j);
						ccc++;
					}
				}
				
				data=new smatrix(temp);	
			}
			else {
				data=new smatrix(arrays);	
				
				
			}
			
			
		}
		
		if (arrays.GetColumnDimension()%this.n_classes!=0){
			 throw new IllegalStateException("Number of final model's output columns need to be a factor of the used classes");  
		}
		int multi=arrays.GetColumnDimension()/this.n_classes;
		
			for (int i=0; i <predictions.length; i++ ){
				for (int m=0; m <multi; m++ ){				
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						int col=arrays.GetColumnDimension() * m + j ;
						predictions[i][j]+=arrays.GetElement(i, col);
					}
				}
			}
		


		
			// return the 1st prediction

		for (int i=0; i < predictionsclass.length; i++) {
			double temp[]=predictions[i];
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictionsclass[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictionsclass[i]=maxi;
	    	  }

		}		
		
		predictions=null;

			// return the 1st prediction
			return predictionsclass;
			
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
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
		
		double predictionsclass[]= new double [data.length];
		double predictions[][]= new double [data.length][n_classes];

		fsmatrix arrays =null;
		
		for(int level=0; level<tree_body.length; level++){
			int column_counter=0;
			arrays= new fsmatrix(predictions.length, this.column_counts[level]);
			for (estimator k : tree_body[level]){
				double preds[][]=k.predict_proba(data);
				if (preds[0].length==2 && level <tree_body.length-1){
					preds=manipulate.select.columnselect.ColumnSelect(preds, new int [] {1});
				}
				for (int j=0; j <preds[0].length; j++ ){
					for (int i=0; i <preds.length; i++ ){
						arrays.SetElement(i, column_counter, preds[i][j]);
					}
					column_counter+=1;
				}				
			}
			
			if (this.stackdata){
				
				double temp[][] = new double [data.length][data[0].length+arrays.GetColumnDimension()];
				int ccc=0;
				for (int i=0; i <data.length; i++ ){ 
					ccc=0;
					for (int j=0; j <data[0].length; j++ ){
						temp[i][ccc]=data[i][j];
						ccc++;
					}
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						temp[i][ccc]=arrays.GetElement(i, j);
						ccc++;
					}
				}
				
				data=temp;	
			}
			else {
				int ccc=0;
				 data =new double [data.length][arrays.GetColumnDimension()] ;
				 for (int i=0; i <data.length; i++ ){ 
						ccc=0;
						for (int j=0; j <arrays.GetColumnDimension(); j++ ){
							data[i][ccc]=arrays.GetElement(i, j);
							ccc++;
						}
					}					
				
			}
			
			
		}
		
		if (arrays.GetColumnDimension()%this.n_classes!=0){
			 throw new IllegalStateException("Number of final model's output columns need to be a factor of the used classes");  
		}
		int multi=arrays.GetColumnDimension()/this.n_classes;
		
			for (int i=0; i <predictions.length; i++ ){
				for (int m=0; m <multi; m++ ){				
					for (int j=0; j <arrays.GetColumnDimension(); j++ ){
						int col=arrays.GetColumnDimension() * m + j ;
						predictions[i][j]+=arrays.GetElement(i, col);
					}
				}
			}
		
			// return the 1st prediction

		for (int i=0; i < predictionsclass.length; i++) {
			double temp[]=predictions[i];
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictionsclass[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictionsclass[i]=maxi;
	    	  }

		}		
			
		predictions=null;

			// return the 1st prediction
			return predictionsclass;


			
			}
	@Override
	public double predict_Row(double[] data) {
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
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	


		double predictions= 0.0;
	

			// return the 1st prediction
			return predictions;
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.tree_body==null || this.tree_body.length<=0  ){
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
		columndimension=data[0].length;
		if (this.parameters.length<1 || (this.parameters[0].length<1) ){
			throw new IllegalStateException(" Parameters need to be provided in string format as model_name parameter_m:value_n ... " );
		}
		if (parameters.length<2 && parameters[0].length==1){
			throw new IllegalStateException("StackNet cannot have only 1 model" );
		}


		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}
		if ( !metric.equals("logloss")  && !metric.equals("accuracy") && !metric.equals("auc")){
			throw new IllegalStateException(" The metric to validate on needs to be one of logloss, accuracy or auc (for binary only) " );	
		}
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.length)  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} 	
		
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
	    fstarget=new int[target.length];
	    for (int i=0; i < fstarget.length; i++){
	    	fstarget[i]=mapper.get(target[i] + "");
	    }
		
		
		if (weights==null) {
			
			weights=new double [data.length];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
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

		// Initialise randomiser
		
		this.random = new XorShift128PlusRandom(this.seed);

		this.n_classes=classes.length;	
		
		
		if (!this.metric.equals("auc") && this.n_classes!=2){
			String last_case []=parameters[parameters.length-1];
			for (int d=0; d <last_case.length;d++){
				String splits[]=last_case[d].split(" " + "+");	
				String str_estimator=splits[0];
				boolean has_regressor_in_last_layer=false;
				if (str_estimator.contains("AdaboostForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("DecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("GradientBoostingForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("RandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("Vanilla2hnnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("multinnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LSVR")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LinearRegression")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LibFmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("knnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KernelmodelRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("XgboostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LightgbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODeepLearningRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGlmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODrfRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnAdaBoostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnDecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnExtraTreesRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnknnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnMLPRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnRandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnSGDRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnsvmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("PythonGenericRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KerasnnRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("FRGFRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("VowpaLWabbitRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("OriginalLibFMRegressor")) {
					has_regressor_in_last_layer=true;					
				}
				
				
				if (has_regressor_in_last_layer){
					throw new IllegalStateException("The last layer of StackNet cannot have a regressor unless the metric is auc and it is a binary problem" );
				}
			}
		}
		

		fsmatrix trainstacker=null;
		tree_body= new estimator[parameters.length][];
		column_counts = new int[parameters.length];
		int kfolder [][][]=null;
		if (!this.input_index.equals("")){
			kfolder=readcsv.get_kfolder(this.input_index);
			this.folds=kfolder.length;
		}else {
			kfolder=kfold.getindices(this.target.length, this.folds);
		}
		if ( (kfolder[0][0].length+ kfolder[0][1].length)!=this.target.length){
			throw new IllegalStateException("The kfold indices do not have the proper size" );
		}
		for(int level=0; level<parameters.length; level++){
			
			// change the data 
			if (level>0){
				if (this.stackdata){
					
					double temp[][] = new double [data.length][data[0].length+trainstacker.GetColumnDimension()];
					int ccc=0;
					for (int i=0; i <data.length; i++ ){ 
						ccc=0;
						for (int j=0; j <data[0].length; j++ ){
							temp[i][ccc]=data[i][j];
							ccc++;
						}
						for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
							temp[i][ccc]=trainstacker.GetElement(i, j);
							ccc++;
						}
					}
					
					data=temp;	
				}
				else {
					int ccc=0;
					 data = new double [data.length][trainstacker.GetColumnDimension()];
					 for (int i=0; i <data.length; i++ ){ 
							ccc=0;
							for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
								data[i][ccc]=trainstacker.GetElement(i, j);
								ccc++;
							}
						}					
					
				}
				
				
			}
			
			String [] level_grid=parameters[level];
			estimator[] mini_batch_tree= new estimator[level_grid.length];
			double metric_averages[]=new double [level_grid.length]; //holds stats for the level
			int model_count=0;
			Thread[] thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimator [] estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			int count_of_live_threads=0;


			int temp_class=estimate_classes(level_grid,  this.n_classes, level==(parameters.length-1));
			column_counts[level]=temp_class;
			if (this.verbose){
				System.out.println(" Level: " +  (level+1) + " dimensionality: " + temp_class);
				System.out.println(" Starting cross validation ");
			}
			if (level<parameters.length -1){
			trainstacker=new fsmatrix(target.length, temp_class);

			
			if (print_indices && level==0 && !this.indices_name.equals("")){
				
				try{  // Catch errors in I/O if necessary.
					  // Open a file to write to.
						String saveFile = indices_name + ".csv";
						
						@SuppressWarnings("resource")
						FileWriter writer = new FileWriter(saveFile);
						for (int n=0; n <this.folds;n++){
							for (int m=0; m < kfolder[n][1].length;m++){
								writer.append(kfolder[n][1][m] +"," + n + "\n");
							}
							
						}

				} catch (Exception e){
					System.out.println(" Failed to write indices at: " +  indices_name + ".csv");
				}
			}
			// begin cross validation
			for (int f=0; f < this.folds; f++){
				
					int train_indices[]=kfolder[f][0]; // train indices
					int test_indices[]=kfolder[f][1]; // test indices	
					//System.out.println(" start!");
					double X_train [][]= manipulate.select.rowselect.RowSelect2d(data, train_indices);
					double X_cv [][] = manipulate.select.rowselect.RowSelect2d(data, test_indices);
					double [] y_train=manipulate.select.rowselect.RowSelect(this.target, train_indices);
					double [] y_cv=manipulate.select.rowselect.RowSelect(this.target, test_indices);
					//double [] y_cv=manipulate.select.rowselect.RowSelect(this.target, test_indices);	
					int column_counter=0;
					
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
					
					
					for (int es=0; es <level_grid.length; es++ ){
						String splits[]=level_grid[es].split(" " + "+");	
						String str_estimator=splits[0];
 						int bags=find_bags(level_grid[es]);
						if (containsClassifier(str_estimator)){
							BaggingClassifier mod = new BaggingClassifier(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
							mini_batch_tree[es].AddClassnames(this.classes);
						} else if (containsRegressor(str_estimator)){
							BaggingRegressor mod = new BaggingRegressor(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
						} else {
							throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
						}
						mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");
						mini_batch_tree[es].set_target(y_train);
		
						estimators[count_of_live_threads]=mini_batch_tree[es];
						thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
						thread_array[count_of_live_threads].start();
						count_of_live_threads++;
						if (this.verbose==true){
							System.out.println("fitting model : " + (es+1));
							
						}
						
						if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
							for (int s=0; s <count_of_live_threads;s++ ){
								try {
									thread_array[s].join();
								} catch (InterruptedException e) {
								   System.out.println(e.getMessage());
								   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
								}
							}
							
							
							for (int s=0; s <count_of_live_threads;s++ ){
								double predictions[][]=estimators[s].predict_proba(X_cv);
								boolean is_regerssion=estimators[s].IsRegressor();
								if (predictions[0].length==2){
									predictions=manipulate.select.columnselect.ColumnSelect(predictions, new int [] {1});
								}
								
								if (this.verbose){
									if(this.n_classes==2 && this.metric.equals("auc")){
											double pr [] = manipulate.conversions.dimension.Convert(predictions);
											crossvalidation.metrics.Metric ms =new auc();
											double auc=ms.GetValue(pr,y_cv ); // the auc for the current fold	
											System.out.println(" AUC: " + auc);
											metric_averages[model_count]+=auc;
										} else if ( this.metric.equals("logloss")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println(" rmse : " + rms);
												metric_averages[model_count]+=rms;
											}else {
											double log=logloss (predictions,y_cv ); // the logloss for the current fold	
											System.out.println(" logloss : " + log);
											metric_averages[model_count]+=log;
											}
											
										} else if (this.metric.equals("accuracy")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println(" rmse : " + rms);
												metric_averages[model_count]+=rms;
											}else {
												double acc=accuracy (predictions,y_cv ); // the accuracy for the current fold	
												System.out.println(" accuracy : " + acc);
												metric_averages[model_count]+=acc;
											}
										}
							}						
								
								
								for (int j=0; j <predictions[0].length; j++ ){
									for (int i=0; i <predictions.length; i++ ){
										trainstacker.SetElement(test_indices[i], column_counter, predictions[i][j]);
									}
									column_counter+=1;
								}
								
								model_count+=1;
								
							}							
							
							System.gc();
							count_of_live_threads=0;
							thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
							estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
						}
						
						
		
					}
					if (this.verbose==true){
						System.out.println("Done with fold: " + (f+1) + "/" + this.folds);
						
					}
					model_count=0;
			}
			
			if (this.print){
				
				if (this.verbose){
					
					System.out.println("Printing reusable train for level: " + (level+1) + " as : " + this.output_name +  (level+1)+ ".csv" );
				}
				trainstacker.ToFile(this.output_name +  (level+1)+ ".csv");
				
			}
			
			}
			// we print file

			if (this.verbose){
				for (int jj=0; jj< metric_averages.length;jj++ ){
					System.out.println(" Average of all folds model " +jj + " : "  + metric_averages[jj]/this.folds);
				}				
				System.out.println(" Level: " +  (level+1)+ " start output modelling ");
			}
			
			thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			mini_batch_tree= new estimator[level_grid.length];
			count_of_live_threads=0;
			/* Final modelling */
			
			for (int es=0; es <level_grid.length; es++ ){
				String splits[]=level_grid[es].split(" " + "+");	
				String str_estimator=splits[0];
				int bags=find_bags(level_grid[es]);
				if (containsClassifier(str_estimator)){
					BaggingClassifier mod = new BaggingClassifier(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
					mini_batch_tree[es].AddClassnames(this.classes);
				} else if (containsRegressor(str_estimator)){
					BaggingRegressor mod = new BaggingRegressor(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
				} else {
					throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
				}
				mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");	
				mini_batch_tree[es].set_target(this.target);

				estimators[count_of_live_threads]=mini_batch_tree[es];
				thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting model: " + (es+1));
					
				}
				
				if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}

					System.gc();
					count_of_live_threads=0;
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
				}
				
			}			
			
			if (this.verbose==true){
				System.out.println("Completed level: " + (level+1) + " out of " + parameters.length);
				
			}
			
			// assign trained models in the main body
			
			this.tree_body[level]=mini_batch_tree;
			
			
		}
		target=null;
		trainstacker=null;
		sdataset=null;
		fsdataset=null;
		dataset=null;
		fstarget=null;
		System.gc();
		
	}
	@Override
	public void fit(fsmatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		fsdataset=data;
		columndimension=data.GetColumnDimension();
		
		if (this.parameters.length<1 || (this.parameters[0].length<1) ){
			throw new IllegalStateException(" Parameters need to be provided in string format as model_name parameter_m:value_n ... " );
		}	
		if (parameters.length<2 && parameters[0].length==1){
			throw new IllegalStateException("StackNet cannot have only 1 model" );
		}	
		
		if ( !metric.equals("logloss")  && !metric.equals("accuracy") && !metric.equals("auc")){
			throw new IllegalStateException(" The metric to validate on needs to be one of logloss, accuracy or auc (for binary only) " );	
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
		} 	
		
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
	    fstarget=new int[target.length];
	    for (int i=0; i < fstarget.length; i++){
	    	fstarget[i]=mapper.get(target[i] + "");
	    }		
		if (weights==null) {
			
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
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

		// Initialise randomiser
		
		this.random = new XorShift128PlusRandom(this.seed);

		this.n_classes=classes.length;			

		if (!this.metric.equals("auc") && this.n_classes!=2){
			String last_case []=parameters[parameters.length-1];
			for (int d=0; d <last_case.length;d++){
				String splits[]=last_case[d].split(" " + "+");	
				String str_estimator=splits[0];
				boolean has_regressor_in_last_layer=false;
				if (str_estimator.contains("AdaboostForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("DecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("GradientBoostingForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("RandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("Vanilla2hnnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("multinnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LSVR")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LinearRegression")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LibFmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("knnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KernelmodelRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("XgboostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LightgbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODeepLearningRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGlmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODrfRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnAdaBoostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnDecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnExtraTreesRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnknnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnMLPRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnRandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnSGDRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnsvmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("PythonGenericRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KerasnnRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("FRGFRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("VowpaLWabbitRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("OriginalLibFMRegressor")) {
					has_regressor_in_last_layer=true;					
				}
				
				
				if (has_regressor_in_last_layer){
					throw new IllegalStateException("The last layer of StackNet cannot have a regressor unless the metric is auc and it is a binary problem" );
				}
			}
		}		
		
		fsmatrix trainstacker=null;
		tree_body= new estimator[parameters.length][];
		column_counts = new int[parameters.length];
		int kfolder [][][]=null;
		if (!this.input_index.equals("")){
			kfolder=readcsv.get_kfolder(this.input_index);
			this.folds=kfolder.length;
		}else {
			kfolder=kfold.getindices(this.target.length, this.folds);
		}
		if ( (kfolder[0][0].length+ kfolder[0][1].length)!=this.target.length){
			throw new IllegalStateException("The kfold indices do not have the proper size" );
		}
		for(int level=0; level<parameters.length; level++){
			
			// change the data 
			if (level>0){
				if (this.stackdata){
					
					double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+trainstacker.GetColumnDimension()];
					int ccc=0;
					for (int i=0; i <data.GetRowDimension(); i++ ){ 
						ccc=0;
						for (int j=0; j <data.GetColumnDimension(); j++ ){
							temp[i][ccc]=data.GetElement(i, j);
							ccc++;
						}
						for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
							temp[i][ccc]=trainstacker.GetElement(i, j);
							ccc++;
						}
					}
					
					data=new fsmatrix(temp);	
				}
				else {
					int ccc=0;
					 data =new fsmatrix(data.GetRowDimension(),trainstacker.GetColumnDimension());
					 for (int i=0; i <data.GetRowDimension(); i++ ){ 
							ccc=0;
							for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
								data.SetElement(i, ccc, trainstacker.GetElement(i, j));
								ccc++;
							}
						}					
					
				}
				
				
			}
			
			String [] level_grid=parameters[level];
			estimator[] mini_batch_tree= new estimator[level_grid.length];
			double metric_averages[]=new double [level_grid.length]; //holds stats for the level
			int model_count=0;
			Thread[] thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimator [] estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			int count_of_live_threads=0;


			int temp_class=estimate_classes(level_grid,  this.n_classes, level==(parameters.length-1));
			column_counts[level] = temp_class;
			if (this.verbose){
				System.out.println(" Level: " +  (level+1) + " dimensionality: " + temp_class);
				System.out.println(" Starting cross validation ");
			}
			if (level<parameters.length -1){
			trainstacker=new fsmatrix(target.length, temp_class);

			
			//print indices if selected
			if (print_indices && level==0 && !this.indices_name.equals("")){
				
				try{  // Catch errors in I/O if necessary.
					  // Open a file to write to.
						String saveFile = indices_name + ".csv";
						
						@SuppressWarnings("resource")
						FileWriter writer = new FileWriter(saveFile);
						for (int n=0; n <this.folds;n++){
							for (int m=0; m < kfolder[n][1].length;m++){
								writer.append(kfolder[n][1][m] +"," + n + "\n");
							}
							
						}

				} catch (Exception e){
					System.out.println(" Failed to write indices at: " +  indices_name + ".csv");
				}
			}
			
			   //System.out.println(" unique values : " + Arrays.toString(this.target));
			// begin cross validation
			for (int f=0; f < this.folds; f++){
				
					int train_indices[]=kfolder[f][0]; // train indices
					int test_indices[]=kfolder[f][1]; // test indices	
					//System.out.println(" start!");
					fsmatrix X_train = data.makerowsubset(train_indices);
					fsmatrix X_cv  =data.makerowsubset(test_indices);
					double [] y_train=manipulate.select.rowselect.RowSelect(this.target, train_indices);
					double [] y_cv=manipulate.select.rowselect.RowSelect(this.target, test_indices);	
					int column_counter=0;
					
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
					
					
					for (int es=0; es <level_grid.length; es++ ){
						String splits[]=level_grid[es].split(" " + "+");	
						String str_estimator=splits[0];
 						int bags=find_bags(level_grid[es]);
						if (containsClassifier(str_estimator)){
							BaggingClassifier mod = new BaggingClassifier(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
							mini_batch_tree[es].AddClassnames(this.classes);
						} else if (containsRegressor(str_estimator)){
							BaggingRegressor mod = new BaggingRegressor(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
						} else {
							throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
						}
						mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");
						mini_batch_tree[es].set_target(y_train);

						
						estimators[count_of_live_threads]=mini_batch_tree[es];
						thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
						thread_array[count_of_live_threads].start();
						count_of_live_threads++;
						if (this.verbose==true){
							System.out.println("Fitting model: " + (es+1));
							
						}
						
						if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
							for (int s=0; s <count_of_live_threads;s++ ){
								try {
									thread_array[s].join();
								} catch (InterruptedException e) {
								   System.out.println(e.getMessage());
								   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
								}
							}
							
							
							for (int s=0; s <count_of_live_threads;s++ ){
								double predictions[][]=estimators[s].predict_proba(X_cv);
								boolean is_regerssion=estimators[s].IsRegressor();
								if (predictions[0].length==2){
									predictions=manipulate.select.columnselect.ColumnSelect(predictions, new int [] {1});

								}
								// metrics' calculation
								if (this.verbose){
									if(this.n_classes==2 && this.metric.equals("auc")){
											double pr [] = manipulate.conversions.dimension.Convert(predictions);
											crossvalidation.metrics.Metric ms =new auc();
											double auc=ms.GetValue(pr,y_cv ); // the auc for the current fold	
											System.out.println(" AUC: " + auc);
											metric_averages[model_count]+=auc;
										} else if ( this.metric.equals("logloss")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println(" rmse : " + rms);
												metric_averages[model_count]+=rms;
											}else {
											double log=logloss (predictions,y_cv ); // the logloss for the current fold	
											System.out.println(" logloss : " + log);
											metric_averages[model_count]+=log;
											}
											
										} else if (this.metric.equals("accuracy")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println(" rmse : " + rms);
												metric_averages[model_count]+=rms;
											}else {
												double acc=accuracy (predictions,y_cv ); // the accuracy for the current fold	
												System.out.println(" accuracy : " + acc);
												metric_averages[model_count]+=acc;
											}
										}
							}
								
								
								for (int j=0; j <predictions[0].length; j++ ){
									for (int i=0; i <predictions.length; i++ ){
										trainstacker.SetElement(test_indices[i], column_counter, predictions[i][j]);
									}
									column_counter+=1;
								}
								
								model_count+=1;
								
							}							
							
							System.gc();
							count_of_live_threads=0;
							thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
							estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
						}
						
						
				
					}
					if (this.verbose==true){
						System.out.println("Done with fold: " + (f+1) + "/" + this.folds);
						
					}
					model_count=0;
			}
			if (this.print){
				
				if (this.verbose){
					
					System.out.println("Printing reusable train for level: " + (level+1) + " as : " + this.output_name +  (level+1)+ ".csv" );
				}
				if (include_target){
					trainstacker.ToFileTarget(this.output_name +  (level+1)+ ".csv",this.target);
				}else {
					trainstacker.ToFile(this.output_name +  (level+1)+ ".csv");
				}
				
			}

			}

			
			if (this.verbose){
				
				for (int jj=0; jj< metric_averages.length;jj++ ){
					System.out.println(" Average of all folds model " +jj + " : "  + metric_averages[jj]/this.folds);
				}
				
				System.out.println(" Level: " +  (level+1)+ " start output modelling ");
			}
			
			thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			mini_batch_tree= new estimator[level_grid.length];
			count_of_live_threads=0;
			/* Final modelling */
			
			for (int es=0; es <level_grid.length; es++ ){
				String splits[]=level_grid[es].split(" " + "+");	
				String str_estimator=splits[0];
				int bags=find_bags(level_grid[es]);
				if (containsClassifier(str_estimator)){
					BaggingClassifier mod = new BaggingClassifier(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
					mini_batch_tree[es].AddClassnames(this.classes);
				} else if (containsRegressor(str_estimator)){
					BaggingRegressor mod = new BaggingRegressor(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
				} else {
					throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
				}
				mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");					
				mini_batch_tree[es].set_target(this.target);

				estimators[count_of_live_threads]=mini_batch_tree[es];
				thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting model : " + (es+1));
					
				}
				
				if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}

					System.gc();
					count_of_live_threads=0;
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
				}
				
			}			
			
			if (this.verbose==true){
				System.out.println("Completed level: " + (level+1) + " out of " + parameters.length);
				
			}
			
			// assign trained models in the main body
			
			this.tree_body[level]=mini_batch_tree;
			
			
		}
		
		trainstacker=null;
		sdataset=null;
		fsdataset=null;
		dataset=null;
		fstarget=null;

		System.gc();


		
	}
	
	@Override
	public void fit(smatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		sdataset=data;
		columndimension=data.GetColumnDimension();
		if (this.parameters.length<1 || (this.parameters[0].length<1) ){
			throw new IllegalStateException(" Parameters need to be provided in string format as model_name parameter_m:value_n ... " );
		}	
		if (parameters.length<2 && parameters[0].length==1){
			throw new IllegalStateException("StackNet cannot have only 1 model" );
		}
		
		if ( !metric.equals("logloss")  && !metric.equals("accuracy") && !metric.equals("auc")){
			throw new IllegalStateException(" The metric to validate on needs to be one of logloss, accuracy or auc (for binary only) " );	
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
		} 	
		
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
	    fstarget=new int[target.length];
	    for (int i=0; i < fstarget.length; i++){
	    	fstarget[i]=mapper.get(target[i] + "");
	    }
	    
		if (weights==null) {
			
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
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

		// Initialise randomiser
		
		this.random = new XorShift128PlusRandom(this.seed);

		this.n_classes=classes.length;			

		if (!this.metric.equals("auc") && this.n_classes!=2){
			String last_case []=parameters[parameters.length-1];
			for (int d=0; d <last_case.length;d++){
				String splits[]=last_case[d].split(" " + "+");	
				String str_estimator=splits[0];
				boolean has_regressor_in_last_layer=false;
				if (str_estimator.contains("AdaboostForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("DecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("GradientBoostingForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("RandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("Vanilla2hnnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("multinnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LSVR")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LinearRegression")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LibFmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("knnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KernelmodelRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("XgboostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LightgbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODeepLearningRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGlmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODrfRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnAdaBoostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnDecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnExtraTreesRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnknnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnMLPRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnRandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnSGDRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnsvmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("PythonGenericRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KerasnnRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("FRGFRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("VowpaLWabbitRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("OriginalLibFMRegressor")) {
					has_regressor_in_last_layer=true;					
				}
				
				
				if (has_regressor_in_last_layer){
					throw new IllegalStateException("The last layer of StackNet cannot have a regressor unless the metric is auc and it is a binary problem" );
				}
			}
		}		
		
		
		fsmatrix trainstacker=null;
		tree_body= new estimator[parameters.length][];
		column_counts = new int[parameters.length];
		int kfolder [][][]=null;
		if (!this.input_index.equals("")){
			kfolder=readcsv.get_kfolder(this.input_index);
			this.folds=kfolder.length;
		}else {
			kfolder=kfold.getindices(this.target.length, this.folds);
		}
		if ( (kfolder[0][0].length+ kfolder[0][1].length)!=this.target.length){
			throw new IllegalStateException("The kfold indices do not have the proper size" );
		}
		for(int level=0; level<parameters.length; level++){
			
			// change the data 
			if (level>0){
				/*
				if (this.stackdata){
					
					double temp[][] = new double [data.GetRowDimension()][data.GetColumnDimension()+trainstacker.GetColumnDimension()];
					int ccc=0;
					for (int i=0; i <data.GetRowDimension(); i++ ){ 
						ccc=0;
						for (int j=0; j <data.GetColumnDimension(); j++ ){
							temp[i][ccc]=data.GetElement(i, j);
							ccc++;
						}
						for (int j=0; j <trainstacker.GetColumnDimension(); j++ ){
							temp[i][ccc]=trainstacker.GetElement(i, j);
							ccc++;
						}
					}
					
					data=new smatrix(temp);	
				}
				else {*/
					 data =new smatrix(trainstacker);
					
					
				//}
				
				
			}
			
			String [] level_grid=parameters[level];
			estimator[] mini_batch_tree= new estimator[level_grid.length];
			double metric_averages[]=new double [level_grid.length]; //holds stats for the level			
			Thread[] thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimator [] estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			int count_of_live_threads=0;
			int model_count=0;
			int temp_class=estimate_classes(level_grid,  this.n_classes, level==(parameters.length-1));
			column_counts[level] = temp_class;
			
			if (this.verbose){
				System.out.println(" Level: " +  (level+1) + " dimensionality: " + temp_class);
				System.out.println(" Starting cross validation ");
			}
			if (level<parameters.length -1){
			trainstacker=new fsmatrix(target.length, temp_class);


			//print indices if selected
			if (print_indices && level==0 && !this.indices_name.equals("")){
				
				try{  // Catch errors in I/O if necessary.
					  // Open a file to write to.
						String saveFile = indices_name + ".csv";
						
						@SuppressWarnings("resource")
						FileWriter writer = new FileWriter(saveFile);
						for (int n=0; n <this.folds;n++){
							for (int m=0; m < kfolder[n][1].length;m++){
								writer.append(kfolder[n][1][m] +"," + n + "\n");
							}
							
						}

				} catch (Exception e){
					System.out.println(" Failed to write indices at: " +  indices_name + ".csv");
				}
			}			
			// begin cross validation
			for (int f=0; f < this.folds; f++){
				
					int train_indices[]=kfolder[f][0]; // train indices
					int test_indices[]=kfolder[f][1]; // test indices	
					//System.out.println(" start!");
					smatrix X_train = data.makesubmatrix(train_indices);
					smatrix X_cv  =data.makesubmatrix(test_indices);
					double [] y_train=manipulate.select.rowselect.RowSelect(this.target, train_indices);
					double [] y_cv=manipulate.select.rowselect.RowSelect(this.target, test_indices);	
					int column_counter=0;
					
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
					
					
					for (int es=0; es <level_grid.length; es++ ){
						String splits[]=level_grid[es].split(" " + "+");	
						String str_estimator=splits[0];
 						int bags=find_bags(level_grid[es]);
						if (containsClassifier(str_estimator)){
							BaggingClassifier mod = new BaggingClassifier(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
							mini_batch_tree[es].AddClassnames(this.classes);
						} else if (containsRegressor(str_estimator)){
							BaggingRegressor mod = new BaggingRegressor(X_train);
							mod.set_model_parameters(level_grid[es]);
							mini_batch_tree[es]=mod;
						} else {
							throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
						}
						mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");
						mini_batch_tree[es].set_target(y_train);
		
						estimators[count_of_live_threads]=mini_batch_tree[es];
						thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
						thread_array[count_of_live_threads].start();
						count_of_live_threads++;
						if (this.verbose==true){
							System.out.println("Fitting model : " + es);
							
						}
						
						if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
							for (int s=0; s <count_of_live_threads;s++ ){
								try {
									thread_array[s].join();
								} catch (InterruptedException e) {
								   System.out.println(e.getMessage());
								   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
								}
							}
							
							
							for (int s=0; s <count_of_live_threads;s++ ){
								double predictions[][]=estimators[s].predict_proba(X_cv);
								boolean is_regerssion=estimators[s].IsRegressor();
								if (predictions[0].length==2){
									predictions=manipulate.select.columnselect.ColumnSelect(predictions, new int [] {1});
								}
								
								if (this.verbose){
									if(this.n_classes==2 && this.metric.equals("auc")){
											double pr [] = manipulate.conversions.dimension.Convert(predictions);
											crossvalidation.metrics.Metric ms =new auc();
											double auc=ms.GetValue(pr,y_cv ); // the auc for the current fold	
											System.out.println(" AUC: " + auc);
											metric_averages[model_count]+=auc;
										} else if ( this.metric.equals("logloss")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println(" rmse : " + rms);
												metric_averages[model_count]+=rms;
											}else {
											double log=logloss (predictions,y_cv ); // the logloss for the current fold	
											System.out.println(" logloss : " + log);
											metric_averages[model_count]+=log;
											}
											
										} else if (this.metric.equals("accuracy")){
											if (is_regerssion){
												double rms=rmse(predictions,y_cv);
												System.out.println(" rmse : " + rms);
												metric_averages[model_count]+=rms;
											}else {
												double acc=accuracy (predictions,y_cv ); // the accuracy for the current fold	
												System.out.println(" accuracy : " + acc);
												metric_averages[model_count]+=acc;
											}
										}
							}
								for (int j=0; j <predictions[0].length; j++ ){
									for (int i=0; i <predictions.length; i++ ){
										trainstacker.SetElement(test_indices[i], column_counter, predictions[i][j]);
									}
									column_counter+=1;
								}
								
								model_count+=1;
								
							}							
							
							System.gc();
							count_of_live_threads=0;
							thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
							estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
						}
						
						

					}
					
					if (this.verbose==true){
						System.out.println("Done with fold: " + (f+1) + "/" + this.folds);
						
					}
					model_count=0;
			}
			if (this.print){
				
				if (this.verbose){
					
					System.out.println("Printing reusable train for level: " + (level+1) + " as : " + this.output_name +  (level+1)+ ".csv" );
				}
				if (include_target){
					trainstacker.ToFileTarget(this.output_name +  (level+1)+ ".csv",this.target);
				}else {
					trainstacker.ToFile(this.output_name +  (level+1)+ ".csv");
				}
				
			}
			}

			
			if (this.verbose){
				
				for (int jj=0; jj< metric_averages.length;jj++ ){
					System.out.println(" Average of all folds model " +jj + " : "  + metric_averages[jj]/this.folds);
				}
				
				System.out.println(" Level: " +  (level+1)+ " start output modelling ");
			}
			
			thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
			estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
			mini_batch_tree= new estimator[level_grid.length];
			count_of_live_threads=0;
			/* Final modelling */
			
			for (int es=0; es <level_grid.length; es++ ){
				String splits[]=level_grid[es].split(" " + "+");	
				String str_estimator=splits[0];
				int bags=find_bags(level_grid[es]);
				if (containsClassifier(str_estimator)){
					BaggingClassifier mod = new BaggingClassifier(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
					mini_batch_tree[es].AddClassnames(this.classes);
				} else if (containsRegressor(str_estimator)){
					BaggingRegressor mod = new BaggingRegressor(data);
					mod.set_model_parameters(level_grid[es]);
					mini_batch_tree[es]=mod;
				} else {
					throw new IllegalStateException(" Only regressors and classifiers supported in StackNet for the meantime." );	
				}
						mini_batch_tree[es].set_params("estimators:" +bags +  " verbose:false seed:1");										
						
				mini_batch_tree[es].set_target(this.target);

				estimators[count_of_live_threads]=mini_batch_tree[es];
				thread_array[count_of_live_threads]= new Thread(mini_batch_tree[es]);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (this.verbose==true){
					System.out.println("Fitting model : " + es);
					
				}
				
				if (count_of_live_threads==thread_array.length || es==level_grid.length-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
					}

					System.gc();
					count_of_live_threads=0;
					thread_array= new Thread[(this.threads>level_grid.length)?level_grid.length: this.threads];
					estimators= new estimator[(this.threads>level_grid.length)?level_grid.length: this.threads];
				}
				
			}			
			
			if (this.verbose==true){
				System.out.println("Completed level: " + (level+1) + " out of " + parameters.length);
				
			}
			
			// assign trained models in the main body
			
			this.tree_body[level]=mini_batch_tree;
			
			
		}
		target=null;
		trainstacker=null;
		sdataset=null;
		fsdataset=null;
		dataset=null;
		fstarget=null;
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
		return "StackNetClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: StackNetClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);						
		System.out.println("threads : "+ this.threads);			
		System.out.println("Seed: "+ seed);	
		System.out.println("print at each level: "+ this.print);		
		System.out.println("output suffix: "+ this.output_name);		
		System.out.println("Verbality: "+ verbose);			
		if (this.tree_body==null){
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
		if (this.tree_body!=null || tree_body.length>0){
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
		this.tree_body= null;
		n_classes=0;
		threads=1;
		this.print=false;
		this.output_name="stacknet";
		this.random=null;
		this.feature_importances.clone();
		columndimension=0;
		this.classes=null;
		seed=1;
		random=null;
		target=null;
		fstarget=null;
		target=null;
		fstarget=null;
		starget=null;
		weights=null;
		verbose=true;

		
		
	}

	@Override
	public estimator copy() {
		StackNetClassifier br = new StackNetClassifier();
		estimator[][] tree_bodys= new estimator[this.tree_body.length][];
        for (int i=0; i <tree_bodys.length; i++ ){
        	tree_bodys[i]= tree_body[i];
        }
        br.tree_body=tree_bodys;
        //br.shrinkage=this.shrinkage;
		br.n_classes=this.n_classes;
		br.threads=this.threads;
		br.columndimension=this.columndimension;
		br.seed=this.seed;
		br.print=this.print;
		br.output_name=this.output_name;
		br.random=this.random;
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.target2d=manipulate.copies.copies.Copy(this.target2d.clone());	
		br.fstarget=(this.fstarget.clone());
		br.starget=(smatrix) this.starget.Copy();
		br.weights=manipulate.copies.copies.Copy(this.weights.clone());
		br.verbose=this.verbose;
		return br;
	}
	
	@Override	
	public void set_params(String params){

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
			public void append_predictions(double previous_predictions [][],  double new_predictions [][], double shrink){
				
				if (previous_predictions.length==1){
					for (int i=0; i <previous_predictions[0].length; i++ ){
							previous_predictions[0][i]+=  new_predictions[i][0]*shrink;
					}
					
				}else {
				
					for (int i=0; i <previous_predictions[0].length; i++ ){
						for (int j=0; j <previous_predictions.length; j++ ){
							previous_predictions[j][i]+= new_predictions[i][j]*shrink;
							

						}	
						
				}
			}
			}



		/**
		 * 
		 * @param previous_predictions : Previous predictions 
		 * @param new_predictions : New predictions to be added to the new ones
		 */
			public void append_predictions_score(double previous_predictions [][],  double new_predictions [][], double shrink){
				
				if (new_predictions[0].length==1){		
					for (int i=0; i <previous_predictions.length; i++ ){
						for (int j=0; j <previous_predictions[0].length; j++ ){
								previous_predictions[i][0]+= new_predictions[i][0]*shrink;			
						} 
					}
					
				}else {
				
					for (int i=0; i <previous_predictions.length; i++ ){
						for (int j=0; j <previous_predictions[0].length; j++ ){
							previous_predictions[i][j]+= new_predictions[i][j]*shrink;

						}
						

			
					}
					
				}
			}
			/**
			 * 
			 * @param previous_predictions : Previous predictions 
			 * @param new_predictions : New predictions to be added to the new ones
			 */
				public void append_predictions_score(double previous_predictions [],  double new_predictions [], double shrink){
					
					if (new_predictions.length==1){		
							for (int j=0; j <previous_predictions.length; j++ ){
									previous_predictions[0]+= new_predictions[0]*shrink;			
							} 
						
						
					}else {
					
							for (int j=0; j <previous_predictions.length; j++ ){
								previous_predictions[j]+= new_predictions[j]*shrink;


						}
						
					}
				}	
			/**
			 * 
			 * @param previous_predictions : raw scores output to be transformed into probabilities
			 */
			public void scale_scores(double previous_predictions [][]){
				
				for (int i=0; i <previous_predictions.length; i++ ){
					double sum=0.0;

		            for (int j = 0; j < previous_predictions[0].length; j++) {
		            	sum += previous_predictions[i][j];
		            }

		            for (int j = 0; j <  previous_predictions[0].length; j++) {
		            	previous_predictions[i][j] /= sum;
		            }
		            
				}
				}

			/**
			 * 
			 * @param previous_predictions : raw scores output to be transformed into probabilities
			 */
			public int scale_scores(double previous_predictions []){
				
					double sum=0.0;
					double max=Double.MIN_VALUE;
					int cla=-1;
					for (int j = 0; j < previous_predictions.length; j++) {
						if (previous_predictions[j]>max ){
							max=previous_predictions[j];
							cla=j;
						}
					}
			        for (int j = 0; j < previous_predictions.length; j++) {
			        	previous_predictions[j] = Math.exp(previous_predictions[j] - max);
			        	sum += previous_predictions[j];
			        }

			        for (int j = 0; j <  previous_predictions.length; j++) {
			        	previous_predictions[j] /= sum;
			        }
			        
			        return cla;
				
				}	
			
			/**
			 * 
			 * @param str_estimator : string of parameters
			 * @return True if String sequence contains a classifier
			 */
			public static boolean containsClassifier(String str_estimator){
				boolean has_classifier_in_last_layer=false;
				if (str_estimator.contains("AdaboostForestClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("DecisionTreeClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("GradientBoostingForestClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("RandomForestClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("Vanilla2hnnclassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("softmaxnnclassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("LSVC")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("LogisticRegression")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("LibFmClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("knnClassifier")) {
					has_classifier_in_last_layer=true;
					

				}else if (str_estimator.contains("KernelmodelClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("NaiveBayesClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("XgboostClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("LightgbmClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("H2OGbmClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("H2ODeepLearningClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("H2ODrfClassifier")) {
					has_classifier_in_last_layer=true;			
				}else if (str_estimator.contains("H2OGlmClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("H2ONaiveBayesClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("FRGFClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("SklearnAdaBoostClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("SklearnDecisionTreeClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("SklearnExtraTreesClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("SklearnknnClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("SklearnMLPClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("SklearnRandomForestClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("SklearnSGDClassifier")) {
					has_classifier_in_last_layer=true;			
				}else if (str_estimator.contains("SklearnsvmClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("KerasnnClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("PythonGenericClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("FRGFClassifier")) {
					has_classifier_in_last_layer=true;
					
				}else if (str_estimator.contains("VowpaLWabbitClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("OriginalLibFMClassifier")) {
					has_classifier_in_last_layer=true;
				}else if (str_estimator.contains("libffmClassifier")) {
					has_classifier_in_last_layer=true;
				}					
					
				
				
				
				
				
				return has_classifier_in_last_layer;
			}
			
			/**
			 * 
			 * @param str_estimator : string of parameters
			 * @return True if String sequence contains a regressor
			 */
			public static boolean containsRegressor(String str_estimator){
				boolean has_regressor_in_last_layer=false;
				if (str_estimator.contains("AdaboostForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("DecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("GradientBoostingForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("RandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("Vanilla2hnnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("multinnregressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LSVR")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LinearRegression")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LibFmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("knnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KernelmodelRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("XgboostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("LightgbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGbmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODeepLearningRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2ODrfRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("H2OGlmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("FRGFRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("SklearnAdaBoostRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnDecisionTreeRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnExtraTreesRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnknnRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnMLPRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnRandomForestRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnSGDRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("SklearnsvmRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("PythonGenericRegressor")) {
					has_regressor_in_last_layer=true;
				}else if (str_estimator.contains("KerasnnRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("FRGFRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("OriginalLibFMRegressor")) {
					has_regressor_in_last_layer=true;					
				}else if (str_estimator.contains("VowpaLWabbitRegressor")) {
					has_regressor_in_last_layer=true;					
				}
				return has_regressor_in_last_layer;
			}			
			
			
			/**
			 * 
			 * @param array : Array of string parameters for the given estimators in one level
			 * @param number_of_classes : number of distinct classes of the target variable
			 * @param islastlevel : True if it is the output level
			 * @return total number of columns to output for the given stacker
			 */
			public static int estimate_classes(String array [], int number_of_classes, boolean islastlevel){
				
				int no=0;
				int add=(number_of_classes<=2?1:number_of_classes);
				if (islastlevel && number_of_classes==2){
					add=2;
				}
				for (int k=0; k <array.length; k++ ){
					String x=array[k];
					if (x.contains("AdaboostForestRegressor") ||
							x.contains("DecisionTreeRegressor")	||
							x.contains("GradientBoostingForestRegressor")	||
							x.contains("RandomForestRegressor")	||				
							x.contains("multinnregressor")	||	
							x.contains("Vanilla2hnnregressor")	||
							x.contains("LSVR")	||
							x.contains("LinearRegression")	||							
							x.contains("OriginalLibFMRegressor")	||
							x.contains("LibFmRegressor")	||
							x.contains("knnRegressor")	||
							x.contains("XgboostRegressor")	||								
							x.contains("LightgbmRegressor")	||
							x.contains("H2OGbmRegressor")	||
							x.contains("H2ODeepLearningRegressor")	||		
							x.contains("H2ODrfRegressor")	||
							x.contains("H2OGlmRegressor")	||
							x.contains("SklearnAdaBoostRegressor")	||
							x.contains("SklearnDecisionTreeRegressor")	||
							x.contains("SklearnExtraTreesRegressor")	||
							x.contains("SklearnknnRegressor")	||								
							x.contains("SklearnMLPRegressor")	||
							x.contains("SklearnRandomForestRegressor")	||
							x.contains("SklearnsvmRegressor")	||		
							x.contains("SklearnSGDRegressor")	||
							x.contains("KerasnnRegressor")	||							
							x.contains("PythonGenericRegressor")	||									
							x.contains("FRGFRegressor")	||
							x.contains("VowpaLWabbitRegressor")	||
							x.contains("KernelmodelRegressor")
							) {
						no++;
					} else {
						no+=add;
					}
				}
				
					return no;
			}
			
			/**
			 * 
			 * @param preds : 2 dimensional predictions
			 * @param target : one dimensional target variable
			 * @return : the logloss metric
			 */
			public double logloss (double preds[][], double target []){
				double metr=0.0;
				double errorlog=0;
				double len=preds.length;
			    // Throw exception if the size is not 2
				if (preds[0].length==1){
					for (int i=0; i <preds.length; i++ ) {
						double value=preds[i][0];
						if (value>1.0-(1E-14)){
							value=1.0-(1E-14);
						} else if (value<0+(1E-14)){
							value=0.0+(1E-14);
						}
						if (target[i]==0){
							errorlog-=(1-target[i]) * Math.log(1-value);
						} else {
							errorlog-=target[i]*Math.log(value) ;
						}
					
					}
				} else {
					
					for (int i=0; i <preds.length; i++ ) {
						double value=preds[i][(int) (target[i]) ];
						if (value>1.0-(1E-14)){
							value=1.0-(1E-14);
						} else if (value<0+(1E-14)){
							value=0.0+(1E-14);
						}
							errorlog-=1.0*Math.log(value) ;
						
					
					}				
					
				}
				errorlog=errorlog/len;
				metr=errorlog;
				return metr;
			}
			
			/**
			 * 
			 * @param preds : 2 dimensional predictions
			 * @param target : one dimensional target variable
			 * @return : the accuracy metric
			 */
			public  double accuracy (double preds[][], double target []){
				double metr=0.0;
				double errorlog=0;
				double count_of_correct=0.0;
				double len=preds.length;
			    // Throw exception if the size is not 2
				if (preds[0].length==1){
					for (int i=0; i <preds.length; i++ ) {
						double value=preds[i][0];
						if (value>=0.5f){
							value=1.0;
						} else {
							value=0.0;
						}
						if (target[i]==value){
							count_of_correct+=1.0;
						} 
					
					}
				} else {
					
					for (int i=0; i <preds.length; i++ ) {
						double maximum=0.0;
						double proba=preds[i][0];
						for (int j=1; j <preds[0].length;j++ ){
							if (preds[i][j]>proba){
								proba=preds[i][j];
								maximum=j;
							}
						}
						if (target[i]==maximum){
							count_of_correct+=1.0;
						} 
						
					
					}				
					
				}
				errorlog=count_of_correct/len;
				metr=errorlog;
				return metr;
			}
			/**
			 * 
			 * @param params : model parameters
			 * @return : number of bags if specified (else 1 )
			 */
			public static int find_bags(String params ){
				int bg=1;
				try{
					String []splits= params.split("bags:");
					String []splits2= splits[1].split(" ");
					bg=Integer.parseInt(splits2[0]);
				} catch (Exception e){
					bg=1;
				}
				
				return bg;
				
			}
			/**
			 * 
			 * @param preds : 2 dimensional predictions
			 * @param target : one dimensional target variable
			 * @return : the rmse metric
			 */
			public  double rmse (double preds[][], double target []){
				double metr=0.0;
				double errorlog=0;
				double len=preds.length;
			    // Throw exception if the size is not 2
					for (int i=0; i <preds.length; i++ ) {
						double value=preds[i][0];
						errorlog=value-target[i];
						errorlog*=errorlog	;
						metr+=errorlog;
					
					}
				 
					metr=Math.sqrt(metr/len);
				return metr;
			}
			
			@Override
			public int getSeed() {
				return this.seed;}
			
			}




