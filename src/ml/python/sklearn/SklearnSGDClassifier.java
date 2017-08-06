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
*<p>Wraps the newest sklearn's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier">SGDClassifier</a>.
*sgd-based linear models models are being trained via a subprocess using python and it is assumed that sklearn is installed in the system and it is accessible from the commandline via <b>python</b>.
* <b>It is expected that files will be created and their size will vary based on the volume of the training data.</b></p>
*
*<p>Information about the tunable parameters can be found <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html">here</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all SGDClassifier features and the user is advised to use it directly from the source.
*Also the <em>sklearn version</em> used to test the class is <b>0.18.2</b> and it is not certain whether it will be compatible with future or older versions. The performance and memory consumption will also be worse than running directly . Additionally
*the description of the parameters may not match the ones in the website. </p></em> 
 */


public class SklearnSGDClassifier implements estimator,classifier {

	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * The initial learning rate for the 'constant' or 'invscaling' schedules. The default value is 0.0 as eta0 is not used by the default schedule 'optimal'.
	 */
	public double eta0=0.01;
	/**
	 * The exponent for inverse scaling learning rate [default 0.5].
	 */
	public double power_t=0.00001;	
	/**
	 *The number of passes over the training data (aka epochs). The number of iterations is set to 1 if using partial_fit. Defaults to 5.
	 */
	public int n_iter=10;
	/**
	 * The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.
	 */
	public double l1_ratio=0.0;
	/**
	 * L2 regularization on the weights
	 */
	public double alpha=0.0;				
	/**
	 * use dense data inside the python module
	 */
	public boolean use_dense=false;	
	/**
	 * Enable shuffling of training data (on each iter).
	 */
	public boolean shuffle=true;
	/**
	 * standardizes/scales data before modelling
	 */
	public boolean standardize=true;	
	/**
	 * converts the data matrix to log plus 1. Negative values are made zeros 
	 */
	public boolean use_log1p=false;
	/**
	 * The learning rate schedule:
	 * <ul>
		<li>constant: eta = eta0 </li>
		<li>optimal: eta = 1.0 / (alpha * (t + t0))</li>
		<li>invscaling: eta = eta0 / pow(t, power_t)</li>
		</ul>
	 */
	public String learning_rate="optimal";
	/**
	 * loss functions. Has to be log or modified_huber
	 */
	public String loss="log";
	/**
	 * The penalty (aka regularization term) to be used. Defaults to 'l2' which is the standard regularizer for linear SVM models. 'l1' and 'elasticnet' might bring sparsity to the model (feature selection) not achievable with 'l2'.
	 */
	public String penalty="l2";	
	/**
	 *  For 'huber', determines the threshold at which it becomes less important to get the prediction exactly right
	 */
	public double epsilon=0.1;	
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
	private String script_name="SklearnSGDClassifier.py";
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
			    writer.append("n_iter=" +this.n_iter + "\n");
			    writer.append("l1_ratio=" + this.l1_ratio + "\n");
			    writer.append("random_state=" + this.seed  + "\n");
			    writer.append("eta0=" + this.eta0 + "\n");
			    writer.append("shuffle=" + this.shuffle + "\n");
			    writer.append("power_t=" +  this.power_t + "\n");
			    writer.append("learning_rate=" +  this.learning_rate + "\n");		    
			    writer.append("loss=" + this.loss + "\n");
			    writer.append("penalty=" +  this.penalty + "\n");
			    writer.append("epsilon=" +  this.epsilon + "\n");		
			    writer.append("n_jobs=" +  this.threads + "\n");				    
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
	public SklearnSGDClassifier(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public SklearnSGDClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public SklearnSGDClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public SklearnSGDClassifier(smatrix data){
		
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
		
		
		//geneeta0 dataset
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
		
		
		//geneeta0 dataset
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
		
		
		//geneeta0 dataset
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
		if ( !learning_rate.equals("constant")  && !learning_rate.equals("invscaling") && !learning_rate.equals("optimal")){
			throw new IllegalStateException(" learning_rate has to be between 'constant', 'invscaling', 'optimal'");	
		}
		if ( !penalty.equals("l2")  && !penalty.equals("l1") && !penalty.equals("elasticnet")){
			throw new IllegalStateException(" penalty has to be between 'l2', 'l1' or 'elasticnet'" );	
		}
		if ( !loss.equals("log")  && !loss.equals("modified_huber")){
			throw new IllegalStateException(" loss has to be between 'log' and 'modified_huber'" );	
		}
		
    	
		if ( epsilon <=0){
			epsilon=0.00000001;
		}	
		
		if ( power_t <=0){
			power_t=0.25;
		}
		
		if ( alpha <=0){
			alpha=0.00000001;
		}

		if (this.l1_ratio<=0 || this.l1_ratio>=1.0 ){
			this.l1_ratio=0.0;
		}
		if (this.eta0<=0 || this.eta0>=1.0 ){
			this.eta0=0.01;
		}	
		if (this.n_iter<1){
			this.n_iter=1;
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
		//geneeta0 config file
		
		//geneeta0 dataset
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

		if ( !learning_rate.equals("constant")  && !learning_rate.equals("invscaling") && !learning_rate.equals("optimal")){
			throw new IllegalStateException(" learning_rate has to be between 'constant', 'invscaling', 'optimal'");	
		}
		if ( !penalty.equals("l2")  && !penalty.equals("l1") && !penalty.equals("elasticnet")){
			throw new IllegalStateException(" penalty has to be between 'l2', 'l1' or 'elasticnet'" );	
		}
		if ( !loss.equals("log")  && !loss.equals("modified_huber")){
			throw new IllegalStateException(" loss has to be between 'log' and 'modified_huber'" );	
		}
		
    	
		if ( epsilon <=0){
			epsilon=0.00000001;
		}	
		
		if ( power_t <=0){
			power_t=0.25;
		}
		
		if ( alpha <=0){
			alpha=0.00000001;
		}

		if (this.l1_ratio<=0 || this.l1_ratio>=1.0 ){
			this.l1_ratio=0.0;
		}
		if (this.eta0<=0 || this.eta0>=1.0 ){
			this.eta0=0.01;
		}	
		if (this.n_iter<1){
			this.n_iter=1;
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
		//geneeta0 config file
		
		//geneeta0 dataset
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
		if ( !learning_rate.equals("constant")  && !learning_rate.equals("invscaling") && !learning_rate.equals("optimal")){
			throw new IllegalStateException(" learning_rate has to be between 'constant', 'invscaling', 'optimal'");	
		}
		if ( !penalty.equals("l2")  && !penalty.equals("l1") && !penalty.equals("elasticnet")){
			throw new IllegalStateException(" penalty has to be between 'l2', 'l1' or 'elasticnet'" );	
		}
		if ( !loss.equals("log")  && !loss.equals("modified_huber")){
			throw new IllegalStateException(" loss has to be between 'log' and 'modified_huber'" );	
		}
		
    	
		if ( epsilon <=0){
			epsilon=0.00000001;
		}	
		
		if ( power_t <=0){
			power_t=0.25;
		}
		
		if ( alpha <=0){
			alpha=0.00000001;
		}

		if (this.l1_ratio<=0 || this.l1_ratio>=1.0 ){
			this.l1_ratio=0.0;
		}
		if (this.eta0<=0 || this.eta0>=1.0 ){
			this.eta0=0.01;
		}	
		if (this.n_iter<1){
			this.n_iter=1;
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
		//geneeta0 config file
		
		//geneeta0 dataset

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
		return "SklearnSGDClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: SklearnSGDClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);			
	    System.out.println("objective: " + this.penalty);
	    System.out.println("epsilon: " + this.epsilon  );
	    System.out.println("learning_rate: " + this.learning_rate);
	    System.out.println("loss: " + this.loss  );  
	    System.out.println("l1_ratio: " + this.l1_ratio );
	    System.out.println("alpha: " + this.alpha );
	    System.out.println("use_dense: " + this.use_dense );
	    System.out.println("nthread: " + this.threads );
	    System.out.println("n_iter: " +  this.n_iter );
	    System.out.println("power_t: " + this.power_t  );
	    System.out.println("l1_ratio" + this.l1_ratio);
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
        penalty = "l2";
        epsilon=0.00000001;
        use_dense =false;
		l1_ratio = 0.0;
        threads=1;
        n_iter = 10;
    	shuffle=true;
    	standardize=false;	
    	use_log1p=true;
    	learning_rate="optimal";
    	loss="log";
    	power_t=0.0001;
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
				else if (metric.equals("n_iter")) {this.n_iter=Integer.parseInt(value);}						
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("use_dense")) {this.use_dense=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("l1_ratio")) {this.l1_ratio=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("eta0")) {this.eta0=Double.parseDouble(value);}					
				else if (metric.equals("shuffle")) {this.shuffle=(value.toLowerCase().equals("true")?true:false);}				
				else if (metric.equals("standardize")) {this.standardize=(value.toLowerCase().equals("true")?true:false)   ;}								
				else if (metric.equals("use_log1p")) {this.use_log1p=(value.toLowerCase().equals("true")?true:false)   ;}							
				else if (metric.equals("power_t")) {this.power_t=Double.parseDouble(value);}				
				else if (metric.equals("learning_rate")) {this.learning_rate=value;}				
				else if (metric.equals("loss")) {this.loss=value;}
				else if (metric.equals("penalty")) {this.penalty=value;}	
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
			  