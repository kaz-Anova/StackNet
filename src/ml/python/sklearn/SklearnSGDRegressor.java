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
import io.output;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;

/**
*<p>Wraps the newest sklearn's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor">SGDRegressor</a>.
*sgd-based linear models models are being trained via a subprocess using python and it is assumed that sklearn is installed in the system and it is accessible from the commandline via <b>python</b>.
* <b>It is expected that files will be created and their size will vary based on the volume of the training data.</b></p>
*
*<p>Information about the tunable parameters can be found <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html">here</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all SGDRegressor features and the user is advised to use it directly from the source.
*Also the <em>sklearn version</em> used to test the class is <b>0.18.2</b> and it is not certain whether it will be compatible with future or older versions. The performance and memory consumption will also be worse than running directly . Additionally
*the description of the parameters may not match the ones in the website. </p></em> 
 */


public class SklearnSGDRegressor implements estimator,regressor {
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
	public String learning_rate="constant";
	/**
	 * loss functions. Has to be squared_loss , huber, epsilon_insensitive or squared_epsilon_insensitive
	 */
	public String loss="squared_loss";
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
	private String script_name="SklearnSGDRegressor.py";
	
	/**
	 * weight to used per row(sample)
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
	public SklearnSGDRegressor(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public SklearnSGDRegressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public SklearnSGDRegressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public SklearnSGDRegressor(smatrix data){
		
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
		File directory = new File(this.usedir +  File.separator + "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		if (n_classes<1 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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
		//generate dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, new double [X.GetRowDimension()] , this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		for (int n=0; n < this.n_classes;n++) {

				create_config_file_pred(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n +  ".pred"	,
						this.columndimension
						);
		
				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".conf" );
				 
				 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", ",", 0, 0.0, false, false);

					 if (temp.length!=predictions.length){
						 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of dataset length" );
					 }
					 for (int i =0; i <predictions.length;i++ ){
							 predictions[i][n]=temp[i];
						 
					 } 
				        // create new file
						File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
				        f.delete();
						f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
				        f.delete();     
				        temp=null;
						System.gc();				  
				 }
				 
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
			// return the 1st prediction
			return predictions;	 

			
			}

	@Override
	public double[][] predict2d(fsmatrix data) {
		
		File directory = new File(this.usedir +  File.separator + "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		if (n_classes<1 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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
		
		//generate dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		for (int n=0; n < this.n_classes;n++) {

				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ n + ".conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n +  ".pred"	,
						this.columndimension
						);
		
				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".conf" );
				 
				 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", ",", 0, 0.0, false, false);

					 if (temp.length!=predictions.length){
						 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".pred " + " is not of dataset length" );
					 }
					 for (int i =0; i <predictions.length;i++ ){
							 predictions[i][n]=temp[i];
						 
					 } 
				        // create new file

						File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ n + ".conf" );
				        f.delete();
						f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
				        f.delete();     
				        temp=null;
						System.gc();				 
					 
					 
				 }
				 
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
			// return the 1st prediction
			return predictions;	 
		
			}

	@Override
	public double[][] predict2d(smatrix data) {
		
		File directory = new File(this.usedir +  File.separator + "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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
		
		//generate dataset
		output out = new output();
		out.verbose=false;
		out.printsmatrix(data, new double [data.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 

		System.gc();
		
		for (int n=0; n < this.n_classes;n++) {

				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n +  ".pred",
						this.columndimension
						);
		
				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".conf" );
				 
				 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", ",", 0, 0.0, false, false);

					 if (temp.length!=predictions.length){
						 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of dataset length" );
					 }
					 for (int i =0; i <predictions.length;i++ ){
							 predictions[i][n]=temp[i];
						 
					 } 
				        // create new file

						File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
				        f.delete();
						f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
				        f.delete();     
				        temp=null;
						System.gc();				  
				 }
				 
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
			// return the 1st prediction
			return predictions;	 
	}

	@Override
	public double[] predict_Row2d(double[] data) {

			return null;
			}


	@Override
	public double[] predict_Row2d(fsmatrix data, int rows) {
		return null;
			
	}

	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		return null;
			}

	@Override
	public double[] predict(fsmatrix data) {
		
		File directory = new File(this.usedir +  File.separator + "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if ( n_classes>1) {
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


		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				this.columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of dataset length" );
		 }		 
		 
		 
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i]= temp[i];
			  
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
        temp=null;
		System.gc();
			// return the 1st prediction



		// return the 1st prediction
		return predictions;
			
			}
			

	@Override
	public double[] predict(smatrix data) {
		
		File directory = new File(this.usedir +  File.separator + "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if ( n_classes>1) {
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

		output out = new output();
		out.verbose=false;
		out.printsmatrix(data, new double [data.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				this.columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf");
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of dataset length" );
		 }		 		 

		for (int i =0; i <predictions.length;i++ ){
				 predictions[i]=temp[i];

			 } 
		 
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
        temp=null;
		System.gc();
			// return the 1st prediction



		// return the 1st prediction
		return predictions;
			
			
	}

	@Override
	public double[] predict(double[][] data) {
		
		File directory = new File(this.usedir +  File.separator + "models");
		if (! directory.exists()){
			directory.mkdir();
		}
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if ( n_classes>1) {
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

		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
				this.columndimension
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of dataset length" );
		 }			 

			for (int i =0; i <predictions.length;i++ ){
				predictions[i]=temp[i];
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
        temp=null;
		System.gc();
			// return the 1st prediction


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
		if ( !loss.equals("squared_loss")  && !loss.equals("huber") && !loss.equals("epsilon_insensitive") && !loss.equals("squared_epsilon_insensitive")){
			throw new IllegalStateException(" loss has to be in 'squared_loss' ,'huber', 'epsilon_insensitive','squared_epsilon_insensitive' " );	
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
		if ( (target==null || target.length!=data.length) && (target2d==null || target2d.length!=data.length) && (fstarget==null || fstarget.GetRowDimension()!=data.length)  && (starget==null || starget.GetRowDimension()!=data.length)  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		}
		if (weights==null) {
			weights=new double [data.length];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/(double) weights.length;
			}
		} else {
			if (weights.length!=data.length){
				throw new DimensionMismatchException(weights.length,data.length);
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			
		}

		//hard copy
		if (copy){
			data= manipulate.copies.copies.Copy(data);
		}
		// Initialise scaler

		n_classes=0;
		if (target!=null){
			n_classes=1;
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}

			columndimension=data[0].length;

			//generate config file
			
			//generate dataset

			smatrix X= new smatrix(data);
			
			if (n_classes==1){
				
				double label []= new double [data.length];
				if (target!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target[i];
					}
				}else if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][0];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i, 0);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();
					    }


					    
			            for (int i=starget.indexpile[0]; i <starget.indexpile[1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}
				
				output out = new output();
				out.verbose=false;
				out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train");//this.usedir +  File.separator +  "models"+File.separator + 
				X=null;
				label=null;
				System.gc();
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ "0.conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.columndimension);

				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" );
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" );
		        // tries to delete a non-existing file
		        f.delete();
			


			}else {
			
			
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [data.length];
				
				
				if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][n];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i,n);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();
					    }

				        
					    
			            for (int i=starget.indexpile[n]; i <starget.indexpile[n+1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}
				
				output out = new output();
				out.verbose=false;
				out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train");//this.usedir +  File.separator +  "models"+File.separator + 
				X=null;
				label=null;
				System.gc();
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + + n +".train",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".mod",
						this.columndimension);

				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".conf" );
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".conf" );
		        // tries to delete a non-existing file
		        f.delete();


			}		
			
			}
			File folder = new File(this.usedir +  File.separator +  "models");
			File[] listOfFiles = folder.listFiles();
		    for (int i = 0; i < listOfFiles.length; i++) {
			      if (listOfFiles[i].isFile()) {
			    	  if (listOfFiles[i].getName().contains(".snapshot_iter")){
			    		  listOfFiles[i].delete();
			    	  }
			      } 
			    }
			fstarget=null;
			fsdataset=null;
			sdataset=null;
			target2d=null;
			System.gc();
	}
	@Override
	public void fit(fsmatrix data) {

		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		fsdataset=data;

		
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
		if ( !loss.equals("squared_loss")  && !loss.equals("huber") && !loss.equals("epsilon_insensitive") && !loss.equals("squared_epsilon_insensitive")){
			throw new IllegalStateException(" loss has to be in 'squared_loss' ,'huber', 'epsilon_insensitive','squared_epsilon_insensitive' " );	
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
		if ( (target==null || target.length!=data.GetRowDimension()) && (target2d==null || target2d.length!=data.GetRowDimension()) && (fstarget==null || fstarget.GetRowDimension()!=data.GetRowDimension())  && (starget==null || starget.GetRowDimension()!=data.GetRowDimension())  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		}
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/(double) weights.length;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			
		}

		//hard copy
		if (copy){
			data= (fsmatrix) (data.Copy());
		}
		// Initialise scaler

		n_classes=0;
		if (target!=null){
			n_classes=1;
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		

		
			columndimension=data.GetColumnDimension();

			//generate config file
			
			//generate dataset

			smatrix X= new smatrix(data);
			
			if (n_classes==1){
				
				double label []= new double [data.GetRowDimension()];
				if (target!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target[i];
					}
				}else if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][0];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i, 0);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();
					    }


					    
			            for (int i=starget.indexpile[0]; i <starget.indexpile[1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}
				
				output out = new output();
				out.verbose=false;
				out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train");//this.usedir +  File.separator +  "models"+File.separator + 
				X=null;
				label=null;
				System.gc();
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.columndimension);

				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" );
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ "0.conf" );
		        // tries to delete a non-existing file
		        f.delete();
			


			}else {
			
			
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [data.GetRowDimension()];
				
				
				if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][n];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i,n);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();
					    }

				        
					    
			            for (int i=starget.indexpile[n]; i <starget.indexpile[n+1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}
				
				output out = new output();
				out.verbose=false;
				out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train");//this.usedir +  File.separator +  "models"+File.separator + 
				X=null;
				label=null;
				System.gc();
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + + n +".train",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".mod",
						this.columndimension);

				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".conf" );
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".conf" );
		        // tries to delete a non-existing file
		        f.delete();


			}		
			
			}
			File folder = new File(this.usedir +  File.separator +  "models");
			File[] listOfFiles = folder.listFiles();
		    for (int i = 0; i < listOfFiles.length; i++) {
			      if (listOfFiles[i].isFile()) {
			    	  if (listOfFiles[i].getName().contains(".snapshot_iter")){
			    		  listOfFiles[i].delete();
			    	  }
			      } 
			    }
			fstarget=null;
			fsdataset=null;
			sdataset=null;
			target2d=null;
			System.gc();
		
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
		if ( !loss.equals("squared_loss")  && !loss.equals("huber") && !loss.equals("epsilon_insensitive") && !loss.equals("squared_epsilon_insensitive")){
			throw new IllegalStateException(" loss has to be in 'squared_loss' ,'huber', 'epsilon_insensitive','squared_epsilon_insensitive' " );	
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
		if ( (target==null || target.length!=data.GetRowDimension()) && (target2d==null || target2d.length!=data.GetRowDimension()) && (fstarget==null || fstarget.GetRowDimension()!=data.GetRowDimension())  && (starget==null || starget.GetRowDimension()!=data.GetRowDimension())  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		}
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/(double) weights.length;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			
		}

		//hard copy
		if (copy){
			data= (smatrix) (data.Copy());
		}
		// Initialise scaler

		n_classes=0;
		if (target!=null){
			n_classes=1;
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		
		
		if (!data.IsSortedByRow()){
			data.convert_type();
			}

		
			columndimension=data.GetColumnDimension();

			//generate config file
			
			//generate dataset


			
			if (n_classes==1){
				
				double label []= new double [data.GetRowDimension()];
				if (target!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target[i];
					}
				}else if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][0];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i, 0);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();
					    }


					    
			            for (int i=starget.indexpile[0]; i <starget.indexpile[1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}
				
				output out = new output();
				out.verbose=false;
				out.printsmatrix(data, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train");//this.usedir +  File.separator +  "models"+File.separator + 
				data=null;
				label=null;
				System.gc();
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.columndimension);

				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" );
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" );
		        // tries to delete a non-existing file
		        f.delete();
			


			}else {
			
			
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [data.GetRowDimension()];
				
				
				if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][n];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i,n);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();
					    }

				        
					    
			            for (int i=starget.indexpile[n]; i <starget.indexpile[n+1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}
				
				output out = new output();
				out.verbose=false;
				out.printsmatrix(data, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train");//this.usedir +  File.separator +  "models"+File.separator + 
				data=null;
				label=null;
				System.gc();
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + + n +".train",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".mod",
						this.columndimension);

				//make subprocess
				 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".conf" );
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".conf" );
		        // tries to delete a non-existing file
		        f.delete();


			}		
			
			}
			File folder = new File(this.usedir +  File.separator +  "models");
			File[] listOfFiles = folder.listFiles();
		    for (int i = 0; i < listOfFiles.length; i++) {
			      if (listOfFiles[i].isFile()) {
			    	  if (listOfFiles[i].getName().contains(".snapshot_iter")){
			    		  listOfFiles[i].delete();
			    	  }
			      } 
			    }
			fstarget=null;
			fsdataset=null;
			sdataset=null;
			target2d=null;
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
		return "SklearnSGDRegressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor: SklearnSGDRegressor");
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
		if (new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()){
			return true;	
		} else {
			return false;
		}		

	}

	@Override
	public boolean IsRegressor() {
		return  true  ;
	}

	@Override
	public boolean IsClassifier() {
		return false;
	}

	@Override
	public void reset() {

		n_classes=0;
		threads=1;
		columndimension=0;
		File directory = new File(this.usedir +  File.separator + "models");
		if (! directory.exists()){
			directory.mkdir();
		}
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.mod" );
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
    	learning_rate="constant";
    	loss="log";
    	power_t=0.0001;
    	alpha=0.0001;
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
				String value="";
				for (int s=1; s<mini_split.length; s++ ){
					if (s==1){
						value= mini_split[s];
					}else {
						value=value + ":" + mini_split[s];
					}
				}

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
			  

