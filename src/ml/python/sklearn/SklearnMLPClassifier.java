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

package ml.python.sklearn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import exceptions.DimensionMismatchException;
import exceptions.LessThanMinimum;
import io.output;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;
import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import utilis.map.intint.StringIntMap4a;

/**
*<p>Wraps the newest sklearn's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier">MLPClassifier</a>.
*Multilayer Perceptron models models are being trained via a subprocess using python and it is assumed that sklearn is installed in the system and it is accessible from the commandline via <b>python</b>.
* <b>It is expected that files will be created and their size will vary based on the volume of the training data.</b></p>
*
*<p>Information about the tunable parameters can be found <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">here</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all MLPClassifier features and the user is advised to use it directly from the source.
*Also the <em>sklearn version</em> used to test the class is <b>0.18.2</b> and it is not certain whether it will be compatible with future or older versions. The performance and memory consumption will also be worse than running directly . Additionally
*the description of the parameters may not match the ones in the website. </p></em> 
 */


public class SklearnMLPClassifier implements estimator,classifier {

	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * The initial learning rate used. It controls the step-size in updating the weights. Only used when optimizer='sgd' or 'adam'.
	 */
	public double learning_rate_init=0.01;
	/**
	 * Number of hidden neurons, comma separated.The length connotes the number of hidden layers too
	 */
	public String hidden="50,25";	
	/**
	 * Tolerance for the optimization. When the epsilon or score is not improving by at least tol for two consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops
	 */
	public double tol=0.00001;	
	/**
	 *Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.
	 */
	public int epochs=10;
	/**
	 * minimum number of cases in batch 
	 */
	public int batch_size=64;
	/**
	 * momentum in sgd
	 */
	public double momentum=0.90;
	/**
	 * Split percentage to use for early stopping.
	 */
	public double validation_split=0.0;
	/**
	 * L2 regularization on the weights
	 */
	public double alpha=0.0;				
	/**
	 * use dense data inside the python module
	 */
	public boolean use_dense=false;	
	/**
	 * Enable shuffling of training data (on each node). This option is recommended if training data is replicated on N nodes, and the number of training samples per iteration is close to N times the dataset size, where all nodes train will (almost) all the data
	 */
	public boolean shuffle=true;
	/**
	 * standardizes data in a batch
	 */
	public boolean standardize=true;	
	/**
	 * converts the data matrix to log plus 1. Negative values are made zeros 
	 */
	public boolean use_log1p=false;
	/**
	 * Learning rate schedule for weight updates. Possible values : constant, invscaling, adaptive
	 */
	public String learning_rate="adaptive";
	/**
	 * activation functions. Has to be relu, identity, tanh or logistic
	 */
	public String activation="relu";
	/**
	 * The solver for weight optimization. Has to be lbfgs, sgd, adam.
	 */
	public String optimizer="adam";	
	/**
	 * Value for numerical stability in adam. Only used when solver=’adam’
	 */
	public double epsilon=1e-8;	
	/**
	 * scale the copy the dataset
	 */
	public boolean copy=false;
    /**
     * seed to use
     */
	public int seed=1;

	/**
	 * Random number generator to use
	 */
	private Random random;
	/**
	 * holds the name of the output model. All files produced (like predictions and model dumps) will use this as prefix.  
	 */
	private String model_name="";
	/**
	 * The directory where all files should be saved
	 */
	private String usedir="";
	/**
	 * Name of the python file
	 */
	private String script_name="SklearnMLPClassifier.py";
	/**
	 * weighst to used per row(sample)
	 */
	public double [] weights;
	/**
	 * if true, it prints stuff
	 */
	public boolean verbose=false;
	/**
	 * Target variable in double format
	 */
	public double target[];
	/*****
	 * Initial estimates
	 */
	double initial_estimates [];
	/**
	 * Target variable in 2d double format
	 */	
	public double target2d[][];
	/**
	 * Target variable in fixed-size matrix format
	 */	
	public double [] fstarget;	
	/**
	 * Target variable in sparse matrix format
	 */	
	public smatrix starget;	
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
	}
	/**
	 * @param name : name to used as predix for all files produced from this
	 */
	public void set_model_name(String name){
		this.model_name=name;
	}
	/**
	 * @param usedir : Set the directory where all files should be saved
	 */
	public void set_usedir(String usedir){
		this.usedir=usedir;
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
	

	public final class SessionIdentifierGenerator {
	    private SecureRandom random = new SecureRandom();

	    public String nextSessionId() {
	        return new BigInteger(130, random).toString(32);
	    }
	}
	/**
	 * 
	 * @param filename : the conifiguration file name for required to run xgboost from the command line
	 * @param datset : the dataset to be used
	 * @param model : model dump name
	 * @param columns : Number of columns in the data
	 */
   private void create_config_file(String filename , String datset, String model, int columns){

		try{  // Catch errors in I/O if necessary.
			  // Open a file to write to.
				String saveFile = filename;
				
				FileWriter writer = new FileWriter(saveFile);
				writer.append("task=train\n");
			    writer.append("columns=" + columns+ "\n");					
			    writer.append("usedense=" + this.use_dense+ "\n");	
			    writer.append("standardize=" + this.standardize+ "\n");	
			    writer.append("use_log1p=" + this.use_log1p+ "\n");				    
			    writer.append("model=" +model+ "\n");	
			    writer.append("data=" +datset+ "\n");				    
			    writer.append("alpha=" + this.alpha+ "\n");
			    writer.append("momentum=" + this.momentum  + "\n");
			    writer.append("max_iter=" +this.epochs + "\n");
			    writer.append("batch_size=" + this.batch_size + "\n");
			    writer.append("validation_fraction=" + this.validation_split + "\n");
			    writer.append("random_state=" + this.seed  + "\n");
			    writer.append("learning_rate_init=" + this.learning_rate_init + "\n");
			    writer.append("shuffle=" + this.shuffle + "\n");
			    writer.append("hidden_layer_sizes=" + this.hidden + "\n");
			    writer.append("tol=" +  this.tol + "\n");
			    writer.append("learning_rate=" +  this.learning_rate + "\n");		    
			    writer.append("activation=" + this.activation + "\n");
			    writer.append("solver=" +  this.optimizer + "\n");
			    writer.append("epsilon=" +  this.epsilon + "\n");			    
			    if (this.verbose){
			    	writer.append("verbose=1" +  "\n");
			    }else {
			    	writer.append("verbose=0" +  "\n");			    	
			    }		    
				writer.close();

	    	} catch (Exception e) {
	    		throw new IllegalStateException(" failed to write the config file at: " + filename);
	    	}   
   }
   
   
	/**
	 * 
	 * @param filename : the configuration file name for required to run python from the command line
	 * @param datset : the dataset to be used
	 * @param model : model dump name
	 * @param predictionfile : where the predictions will be saved
	 * @param columns : Number of columns in the data
	 */
  private void create_config_file_pred(String filename , String datset, String model, String predictionfile, int columns){

		try{  // Catch errors in I/O if necessary.
			  // Open a file to write to.
				String saveFile = filename;
				
				FileWriter writer = new FileWriter(saveFile);
				writer.append("task=predict\n");
			    writer.append("columns=" + columns+ "\n");				
			    writer.append("usedense=" + this.use_dense+ "\n");	
			    writer.append("standardize=" + this.standardize+ "\n");		    
			    writer.append("use_log1p=" + this.use_log1p+ "\n");			    
			    writer.append("model=" +model+ "\n");	
			    writer.append("data=" +datset+ "\n");				
			    writer.append("prediction=" +predictionfile+ "\n");	
			    if (this.verbose){
			    	writer.append("verbose=1" +  "\n");
			    }else {
			    	writer.append("verbose=0" +  "\n");			    	
			    }				    
			   
		    
				writer.close();

	    	} catch (Exception e) {
	    		throw new IllegalStateException(" failed to write the config file at: " + filename);
	    	}   
  }
  
   /**
    * 
    * @param confingname : full path and name of the config file
    */
   private void create_light_suprocess(String confingname ) {
	   
	   // check if file exists
	   if (new File(confingname).exists()==false){
		   throw new IllegalStateException("Config file does not exist at: " + confingname);
	   }
	   // create the subprocess
		try {

			 String python_path="python";
			 String s_path="lib" + File.separator + "python" + File.separator +  this.script_name;
			 List<String> list = new ArrayList<String>();
			 list.add(python_path);	
			 list.add(s_path);				 
			 list.add( confingname); 

			 //start the process		 
			 ProcessBuilder p = new ProcessBuilder(list);		  
		     p.redirectErrorStream(true);
		     Process sp = p.start();
		     BufferedReader r = new BufferedReader(new InputStreamReader(sp.getInputStream()));
		     String line;
		        while(true){
		            line = r.readLine();
		            if(line == null) { break; }
		            if (this.verbose){
		            	System.out.println(line);
		            }
		        }

		} catch (IOException e) {
			throw new IllegalStateException(" failed to create sklearn subprocess with config name " + confingname);
		}
	   
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
	 * Default constructor for LinearRegression with no data
	 */
	public SklearnMLPClassifier(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public SklearnMLPClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public SklearnMLPClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public SklearnMLPClassifier(smatrix data){
		
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
		
		File directory = new File(this.usedir +  File.separator +  "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		/*  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()==false ){
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
		
		
		//genelearning_rate_init dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,new double [X.GetRowDimension()] ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		 
		 
		 int cols [] = new int [this.n_classes];
		 for (int j=0 ; j <this.n_classes; j++){
			 cols[j]=j;
		 }
		 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", cols, "0.0", false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
		 }
		 for (int i =0; i <predictions.length;i++ ){
			 for (int j =0; j <this.n_classes;j++ ){
				 predictions[i][j]=temp[i][j];
			 }
		 } 
	        temp=null;
		 
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete(); 

		System.gc();

		// return the 1st prediction
		return predictions;
			
			}

	@Override
	public double[][] predict_proba(fsmatrix data) {
		
		File directory = new File(this.usedir +  File.separator +  "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()==false ){
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
		
		
		//genelearning_rate_init dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,new double [X.GetRowDimension()] ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		 
		 
		 int cols [] = new int [this.n_classes];
		 for (int j=0 ; j <this.n_classes; j++){
			 cols[j]=j;
		 }
		 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", cols, "0.0", false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
		 }
		 for (int i =0; i <predictions.length;i++ ){
			 for (int j =0; j <this.n_classes;j++ ){
				 predictions[i][j]=temp[i][j];
			 }
		 } 
	        temp=null;
		 
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete(); 

		System.gc();

		// return the 1st prediction
		return predictions;
		
			}

	@Override
	public double[][] predict_proba(smatrix data) {
		
		File directory = new File(this.usedir +  File.separator +  "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()==false ){
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
		
		
		//genelearning_rate_init dataset
		output out = new output();
		out.verbose=false;
		out.printsmatrix(data,new double [data.GetRowDimension()] ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		data=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		 
		 
		 int cols [] = new int [this.n_classes];
		 for (int j=0 ; j <this.n_classes; j++){
			 cols[j]=j;
		 }
		 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", cols, "0.0", false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
		 }
		 for (int i =0; i <predictions.length;i++ ){
			 for (int j =0; j <this.n_classes;j++ ){
				 predictions[i][j]=temp[i][j];
			 }
		 } 
	        temp=null;
		 
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete(); 

		System.gc();

		// return the 1st prediction
		return predictions;
	}

	@Override
	public double[] predict_probaRow(double[] data) {

			return null;
			}


	@Override
	public double[] predict_probaRow(fsmatrix data, int rows) {
		return null;
			
	}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		return null;
			}

	@Override
	public double[] predict(fsmatrix data) {
		
		File directory = new File(this.usedir +  File.separator +  "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()==false ){
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
		double prediction_probas[][]= new double [data.GetRowDimension()][n_classes];


		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, new double [X.GetRowDimension()] , this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );

		 int cols [] = new int [this.n_classes];
		 for (int j=0 ; j <this.n_classes; j++){
			 cols[j]=j;
		 }
		 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", cols, "0.0", false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
		 }
		 for (int i =0; i <predictions.length;i++ ){
			 for (int j =0; j <this.n_classes;j++ ){
				 prediction_probas[i][j]=temp[i][j];
			 }
		 } 
	        temp=null;
	
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		double temps[]=null;       
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			 temps=prediction_probas[i];
	    	  int maxi=0;
	    	  double max=temps[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temps[k]>max){
	    			 max=temps[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions[i]=maxi;
	    	  }

		}		
		
		prediction_probas=null;

			// return the 1st prediction
			return predictions;
			
			}
			

	@Override
	public double[] predict(smatrix data) {
		
		File directory = new File(this.usedir +  File.separator +  "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()==false ){
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
		double prediction_probas[][]= new double [data.GetRowDimension()][n_classes];

		output out = new output();
		out.verbose=false;
		out.printsmatrix(data, new double [data.GetRowDimension()] , this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		data=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );

		 int cols [] = new int [this.n_classes];
		 for (int j=0 ; j <this.n_classes; j++){
			 cols[j]=j;
		 }
		 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", cols, "0.0", false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
		 }
		 for (int i =0; i <predictions.length;i++ ){
			 for (int j =0; j <this.n_classes;j++ ){
				 prediction_probas[i][j]=temp[i][j];
			 }
		 } 
	        temp=null;
	
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		double temps[]=null;       
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			 temps=prediction_probas[i];
	    	  int maxi=0;
	    	  double max=temps[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temps[k]>max){
	    			 max=temps[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions[i]=maxi;
	    	  }

		}		
		
		prediction_probas=null;

			// return the 1st prediction
			return predictions;
			
			
	}

	@Override
	public double[] predict(double[][] data) {
		
		File directory = new File(this.usedir +  File.separator +  "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()==false ){
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
		double prediction_probas[][]= new double [data.length][n_classes];

		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, new double [X.GetRowDimension()] , this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );

		 int cols [] = new int [this.n_classes];
		 for (int j=0 ; j <this.n_classes; j++){
			 cols[j]=j;
		 }
		 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", cols, "0.0", false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
		 }
		 for (int i =0; i <predictions.length;i++ ){
			 for (int j =0; j <this.n_classes;j++ ){
				 prediction_probas[i][j]=temp[i][j];
			 }
		 } 
	        temp=null;
	
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		double temps[]=null;       
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			 temps=prediction_probas[i];
	    	  int maxi=0;
	    	  double max=temps[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temps[k]>max){
	    			 max=temps[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions[i]=maxi;
	    	  }

		}		
		
		prediction_probas=null;

			// return the 1st prediction
			return predictions;

			
			}
	@Override
	public double predict_Row(double[] data) {

			return 0.0;
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {
		return 0.0;
			}
			
	

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		return 0.0;
			}

	
	
	@Override
	public void fit(double[][] data) {
		
		
		// make sensible checks
		if (data==null || data.length<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		dataset=data;
		
		this.random = new XorShift128PlusRandom(this.seed);
		
		//check model name
		if (this.model_name.equals("")){
			SessionIdentifierGenerator session = new SessionIdentifierGenerator();
			this.model_name=session.nextSessionId();
		}
		// check diretcory
		if (this.usedir.equals("")){
			usedir=System.getProperty("user.dir"); // working directory
		}
		
		File directory = new File(this.usedir +  File.separator + "models");
		
		if (! directory.exists()){
			directory.mkdir();
		}
		if ( !learning_rate.equals("constant")  && !learning_rate.equals("invscaling") && !learning_rate.equals("adaptive")){
			throw new IllegalStateException(" learning_rate has to be between 'constant', 'invscaling', 'adaptive'");	
		}
		if ( !optimizer.equals("lbfgs")  && !optimizer.equals("adam") && !optimizer.equals("sgd")){
			throw new IllegalStateException(" optimizer has to be between 'adam', 'lbfgs' or 'sgd'" );	
		}
		if ( !activation.equals("relu")  && !activation.equals("tanh") && !activation.equals("identity")&& !activation.equals("logistic")){
			throw new IllegalStateException(" activation has to be between 'relu', 'tanh', 'identity' or 'logistic'" );	
		}
		
		
    	String splits [] = this.hidden.replace(" ","").split(",");
		int k=0;
    	for (String ele: splits){
    		try{
    			int hidden_neurons=Integer.parseInt(ele);
	
    			if (hidden_neurons<0){
    				throw new IllegalStateException(" hidden neurons in a layer cannot be less than /equal to zero " );		
    			}
    			k+=1;	
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated integer indices .Here it receied: " + ele  );	
    		}
    	}
    	
		if ( epsilon <=0){
			epsilon=0.00000001;
		}	
		
		if ( tol <=0){
			tol=0.0001;
		}
		
		if ( alpha <=0){
			alpha=0.00000001;
		}

		if (this.batch_size<=0){
			this.batch_size=1;
		}	
		if (this.momentum<=0 || this.momentum>=1.0 ){
			this.momentum=0.9;
		}	
		if (this.validation_split<=0 || this.validation_split>=1.0 ){
			this.validation_split=0.0;
		}
		if (this.learning_rate_init<=0 || this.learning_rate_init>=1.0 ){
			this.learning_rate_init=0.01;
		}	
		if (this.epochs<1){
			this.epochs=1;
		}

		if (this.threads<0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}


		// make sensible checks on the target data
		if ( (target==null || target.length!=data.length) && (Starget==null || Starget.length!=data.length) ){
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
			
			k=0;
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
		    fstarget=new double[target.length];
		    for (int i=0; i < fstarget.length; i++){
		    	fstarget[i]=mapper.get(target[i] + "");
		    }
		    

			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);
			    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
			    int index=0;
			    for (int j=0; j < classes.length; j++){
			    	mapper.put(classes[j], index);
			    	index++;
			    }
			    
			    fstarget=new double[Starget.length];
			    for (int i=0; i < fstarget.length; i++){
			    	fstarget[i]=mapper.get(Starget[i]);
			    }    
			
		}else {
			fstarget=this.target;	
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
			data= manipulate.copies.copies.Copy(data);
		}

		this.n_classes=classes.length;	

		
		columndimension=data[0].length;
		//genelearning_rate_init config file
		
		//genelearning_rate_init dataset
		smatrix X= new smatrix(data);
		System.out.println(X.GetColumnDimension() + X.GetRowDimension());
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		
		create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				columndimension);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
        // tries to delete a non-existing file
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".conf" );
        // tries to delete a non-existing file
        f.delete();	
        
		//deletes all temporary modelling files : 
		File folder = new File(this.usedir +  File.separator +  "models");
		File[] listOfFiles = folder.listFiles();

		    for (int i = 0; i < listOfFiles.length; i++) {
		      if (listOfFiles[i].isFile()) {
		    	  if (listOfFiles[i].getName().contains(".snapshot_iter")){
		    		  listOfFiles[i].delete();
		    	  }
		      } 
		    }

	}
	@Override
	public void fit(fsmatrix data) {
		
		
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		fsdataset=data;

		this.random = new XorShift128PlusRandom(this.seed);
		
		//check model name
		if (this.model_name.equals("")){
		
			SessionIdentifierGenerator session = new SessionIdentifierGenerator();
			this.model_name=session.nextSessionId();
			
		}
		// check diretcory
		if (this.usedir.equals("")){
			usedir=System.getProperty("user.dir"); // working directory
			
		}
		
		File directory = new File(this.usedir +  File.separator + "models");
		
		if (! directory.exists()){
			directory.mkdir();
		}

		if ( !learning_rate.equals("constant")  && !learning_rate.equals("invscaling") && !learning_rate.equals("adaptive")){
			throw new IllegalStateException(" learning_rate has to be between 'constant', 'invscaling', 'adaptive'");	
		}
		if ( !optimizer.equals("lbfgs")  && !optimizer.equals("adam") && !optimizer.equals("sgd")){
			throw new IllegalStateException(" optimizer has to be between 'adam', 'lbfgs' or 'sgd'" );	
		}
		if ( !activation.equals("relu")  && !activation.equals("tanh") && !activation.equals("identity")&& !activation.equals("logistic")){
			throw new IllegalStateException(" activation has to be between 'relu', 'tanh', 'identity' or 'logistic'" );	
		}
		
		
    	String splits [] = this.hidden.replace(" ","").split(",");
		int k=0;
    	for (String ele: splits){
    		try{
    			int hidden_neurons=Integer.parseInt(ele);
	
    			if (hidden_neurons<0){
    				throw new IllegalStateException(" hidden neurons in a layer cannot be less than /equal to zero " );		
    			}
    			k+=1;	
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated integer indices .Here it receied: " + ele  );	
    		}
    	}
    	
		if ( epsilon <=0){
			epsilon=0.00000001;
		}	
		
		if ( tol <=0){
			tol=0.0001;
		}
		
		if ( alpha <=0){
			alpha=0.00000001;
		}

		if (this.batch_size<=0){
			this.batch_size=1;
		}	
		if (this.momentum<=0 || this.momentum>=1.0 ){
			this.momentum=0.9;
		}	
		if (this.validation_split<=0 || this.validation_split>=1.0 ){
			this.validation_split=0.0;
		}
		if (this.learning_rate_init<=0 || this.learning_rate_init>=1.0 ){
			this.learning_rate_init=0.01;
		}	
		if (this.epochs<1){
			this.epochs=1;
		}

		if (this.threads<0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}


		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) ){
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
			
			k=0;
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
		    fstarget=new double[target.length];
		    for (int i=0; i < fstarget.length; i++){
		    	fstarget[i]=mapper.get(target[i] + "");
		    }
		    

			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);
			    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
			    int index=0;
			    for (int j=0; j < classes.length; j++){
			    	mapper.put(classes[j], index);
			    	index++;
			    }
			    
			    fstarget=new double[Starget.length];
			    for (int i=0; i < fstarget.length; i++){
			    	fstarget[i]=mapper.get(Starget[i]);
			    }    
			
		}else {
			fstarget=this.target;	
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
			data= (fsmatrix) data.Copy();
		}

		this.n_classes=classes.length;	
		

		
		columndimension=data.GetColumnDimension();
		//genelearning_rate_init config file
		
		//genelearning_rate_init dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		
		create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				columndimension);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );

	        // create new file
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();
			f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
	        // tries to delete a non-existing file
	        f.delete();
			//deletes all temporary modelling files : 
			File folder = new File(this.usedir +  File.separator +  "models");
			File[] listOfFiles = folder.listFiles();

			    for (int i = 0; i < listOfFiles.length; i++) {
			      if (listOfFiles[i].isFile()) {
			    	  if (listOfFiles[i].getName().contains(".snapshot_iter")){
			    		  listOfFiles[i].delete();
			    	  }
			      } 
			    }	
	}
	
	@Override
	public void fit(smatrix data) {
		

		
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		sdataset=data;
		this.random = new XorShift128PlusRandom(this.seed);
		
		//check model name
		if (this.model_name.equals("")){
		
			SessionIdentifierGenerator session = new SessionIdentifierGenerator();
			this.model_name=session.nextSessionId();
			
		}
		// check directory
		if (this.usedir.equals("")){
			usedir=System.getProperty("user.dir"); // working directory
			
		}
		
		File directory = new File(this.usedir +  File.separator + "models");
		
		if (! directory.exists()){
			directory.mkdir();
		}
		if ( !learning_rate.equals("constant")  && !learning_rate.equals("invscaling") && !learning_rate.equals("adaptive")){
			throw new IllegalStateException(" learning_rate has to be between 'constant', 'invscaling', 'adaptive'");	
		}
		if ( !optimizer.equals("lbfgs")  && !optimizer.equals("adam") && !optimizer.equals("sgd")){
			throw new IllegalStateException(" optimizer has to be between 'adam', 'lbfgs' or 'sgd'" );	
		}
		if ( !activation.equals("relu")  && !activation.equals("tanh") && !activation.equals("identity")&& !activation.equals("logistic")){
			throw new IllegalStateException(" activation has to be between 'relu', 'tanh', 'identity' or 'logistic'" );	
		}
		
		
    	String splits [] = this.hidden.replace(" ","").split(",");
		int k=0;
    	for (String ele: splits){
    		try{
    			int hidden_neurons=Integer.parseInt(ele);
	
    			if (hidden_neurons<0){
    				throw new IllegalStateException(" hidden neurons in a layer cannot be less than /equal to zero " );		
    			}
    			k+=1;	
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated integer indices .Here it receied: " + ele  );	
    		}
    	}
    	
		if ( epsilon <=0){
			epsilon=0.00000001;
		}	
		
		if ( tol <=0){
			tol=0.0001;
		}
		
		if ( alpha <=0){
			alpha=0.00000001;
		}

		if (this.batch_size<=0){
			this.batch_size=1;
		}	
		if (this.momentum<=0 || this.momentum>=1.0 ){
			this.momentum=0.9;
		}	
		if (this.validation_split<=0 || this.validation_split>=1.0 ){
			this.validation_split=0.0;
		}
		if (this.learning_rate_init<=0 || this.learning_rate_init>=1.0 ){
			this.learning_rate_init=0.01;
		}	
		if (this.epochs<1){
			this.epochs=1;
		}

		if (this.threads<0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}


		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) ){
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
			
			k=0;
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
		    fstarget=new double[target.length];
		    for (int i=0; i < fstarget.length; i++){
		    	fstarget[i]=mapper.get(target[i] + "");
		    }
		    

			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);
			    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
			    int index=0;
			    for (int j=0; j < classes.length; j++){
			    	mapper.put(classes[j], index);
			    	index++;
			    }
			    
			    fstarget=new double[Starget.length];
			    for (int i=0; i < fstarget.length; i++){
			    	fstarget[i]=mapper.get(Starget[i]);
			    }    
			
		}else {
			fstarget=this.target;	
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
			data= (smatrix) data.Copy();
		}

		this.n_classes=classes.length;	
		
		columndimension=data.GetColumnDimension();
		//genelearning_rate_init config file
		
		//genelearning_rate_init dataset

		output out = new output();
		out.verbose=false;
		out.printsmatrix(data, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		data=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		
		create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				columndimension);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		sdataset=null;
		System.gc();
		
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
        // tries to delete a non-existing file
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        // tries to delete a non-existing file
        f.delete();
		
		// calculate first node
		//deletes all temporary modelling files : 
		File folder = new File(this.usedir +  File.separator +  "models");
		File[] listOfFiles = folder.listFiles();

		    for (int i = 0; i < listOfFiles.length; i++) {
		      if (listOfFiles[i].isFile()) {
		    	  if (listOfFiles[i].getName().contains(".snapshot_iter")){
		    		  listOfFiles[i].delete();
		    	  }
		      } 
		    }	
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
		return "SklearnMLPClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: SklearnMLPClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);			
	    System.out.println("objective: " + this.optimizer);
	    System.out.println("epsilon: " + this.epsilon  );
	    System.out.println("learning_rate: " + this.learning_rate);
	    System.out.println("activation: " + this.activation  );  
	    System.out.println("momentum: " + this.momentum);
	    System.out.println("validation_split: " + this.validation_split );
	    System.out.println("alpha: " + this.alpha );
	    System.out.println("use_dense: " + this.use_dense );
	    System.out.println("nthread: " + this.threads );
	    System.out.println("epochs: " +  this.epochs );
	    System.out.println("batch_size: " + this.batch_size );
	    System.out.println("hidden: " + this.hidden  );
	    System.out.println("tol: " + this.tol  );
	    System.out.println("validation_split" + this.validation_split);
	    System.out.println("shuffle: " + this.shuffle ); 
	    System.out.println("standardize: " +  this.standardize );
	    System.out.println("use_log1p: " +  this.use_log1p );    
	    if (this.verbose){
	    	System.out.println("verbose: true");
	    }else {
	    	System.out.println("verbose: false");
	    }
		if (new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()){
			System.out.println("Trained: True");	
		} else {
			System.out.println("Trained: False");
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
		if (new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod").exists()){
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
		
		File directory = new File(this.usedir +  File.separator +  "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		n_classes=0;
		threads=1;
		columndimension=0;
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".mod" );
        f.delete();  
        optimizer = "adam";
        epsilon=0.00000001;
        momentum=0.9;
        use_dense =false;
		validation_split = 0.0;
        threads=1;
        epochs = 10;
        batch_size=1;
    	shuffle=true;
    	standardize=false;	
    	use_log1p=true;
    	learning_rate="constant";
    	activation="relu";
    	hidden="50,25";
    	tol=0.0001;
    	alpha=0.0001;
		classes=null;
		copy=true;
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
		
		return this;
	}
	
	@Override	
	public void set_params(String params){
		
		String splitted_params []=params.split(" " + "+");
		
		for (int j=0; j<splitted_params.length; j++ ){
			String mini_split []=splitted_params[j].split(":");
			if (mini_split.length>=2){
				String metric=mini_split[0];
				String value=mini_split[1];
				//System.out.println("'" + metric + "'" + " value=" + value);
				if (metric.equals("alpha")) {this.alpha=Double.parseDouble(value);}
				else if (metric.equals("momentum")) {this.momentum=Double.parseDouble(value);}	
				else if (metric.equals("epochs")) {this.epochs=Integer.parseInt(value);}			
				else if (metric.equals("batch_size")) {this.batch_size=Integer.parseInt(value);}				
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("use_dense")) {this.use_dense=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("validation_split")) {this.validation_split=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("learning_rate_init")) {this.learning_rate_init=Double.parseDouble(value);}					
				else if (metric.equals("shuffle")) {this.shuffle=(value.toLowerCase().equals("true")?true:false);}				
				else if (metric.equals("standardize")) {this.standardize=(value.toLowerCase().equals("true")?true:false)   ;}								
				else if (metric.equals("use_log1p")) {this.use_log1p=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("hidden")) {this.hidden=value;}				
				else if (metric.equals("tol")) {this.tol=Double.parseDouble(value);}				
				else if (metric.equals("learning_rate")) {this.learning_rate=value;}				
				else if (metric.equals("activation")) {this.activation=value;}
				else if (metric.equals("optimizer")) {this.optimizer=value;}	
				else if (metric.equals("epsilon")) {this.epsilon=Double.parseDouble(value);}	
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}					
			}
			
		}
		

	}
	@Override	
	public void set_target(double data []){
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		this.target=data;
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
			  