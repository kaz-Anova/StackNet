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

package ml.python.keras;
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
*<p>Wraps <a href="https://keras.io/models/about-keras-models/">Keras</a>.
*Keras' (multilayer perceptron) models are being trained via a subprocess using python and it is assumed that keras is installed in the system and it is accessible from the commandline via <b>python</b>.
* <b>It is expected that files will be created and their size will vary based on the volume of the training data.</b></p>
*
*<p>Information about the tunable parameters can be found <a href="https://keras.io/models/about-keras-models/">here</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all Keras' features and the user is advised to use it directly from the source.
*The implementation will be based primarily on experience on what has worked in the past. The backend of keras (for example whether it will be run on Tensorflow or Theano) is irrelevant to this class.
*If the user wants to use Theano with certain flags, Gpu etc, he/she needs to specify in the equivalent <a href="https://keras.io/backend/">keras.json</a> and <a href="https://keras.io/getting-started/faq/">.theanorc</a>  files that normally reside in the user directory after installing keras and theano.
*Also the  <u>Keras version used to test the class is <b>2.0.6</b></u> and  <u>theano was <b>0.9.0</b></u>.
* It was not tested with tensorflow, but theoretically it should work.
*   it is not certain whether it will be compatible with future or older versions. The performance and memory consumption will also be worse than running directly . Additionally
*the description of the parameters may not match the ones in the website. </p></em> 
 */



public class KerasnnRegressor implements estimator,regressor {

	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * learning rate
	 */
	public double lr=0.01;
	/**
	 * Number of hidden neurons, comma separated.The length connotes the number of hidden layers too
	 */
	public String hidden="50,25";	
	/**
	 * dropout ratin for each hidden layer,comma sepalrd .Has to match in length the 'hidden' parameter
	 */
	public String droupouts="0.4,0.2";	
	/**
	 * Number of iterations to train the DL model
	 */
	public int epochs=10;
	/**
	 * minimum number of cases in batch 
	 */
	public int batch_size=64;
	/**
	 * Stop the model after X rounds (requires validation_split>0)
	 */
	public int stopping_rounds=250;	
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
	public String l2="0.0,0.0";				
	/**
	 * Enable shuffling of training data (on each node). This option is recommended if training data is replicated on N nodes, and the number of training samples per iteration is close to N times the dataset size, where all nodes train will (almost) all the data
	 */
	public boolean shuffle=true;
	/**
	 * standardizes data in a batch
	 */
	public boolean standardize=true;	
	/**
	 * True to use batch normalisation
	 */
	public boolean batch_normalization=false;	
	/**
	 * converts the data matrix to log plus 1. Negative values are made zeros 
	 */
	public boolean use_log1p=false;
	/**
	 * the distribution from which initial weights are to be drawn. Has to be RandomNormal, RandomUniform, TruncatedNormal,VarianceScaling, Orthogonal, Identity, lecun_uniform, glorot_normal,glorot_uniform, he_normal, lecun_normal, he_uniform, he_normal
	 */
	public String weight_init="lecun_uniform";
	/**
	 * activation functions. Has to be Relu, Tanh, sigmoid
	 */
	public String activation="Relu,Relu";
	/**
	 * The objective has to be adagrad,adam,nadam,adadelta,sgd.
	 */
	public String optimizer="adam";	
	/**
	 * The loss has to be mean_squared_error, mean_absolute_error, mean_squared_logarithmic_error, squared_hinge, hinge, poisson
	 */
	private String loss="mean_squared_error";	
	/**
	 * scale the copy the dataset
	 */
	public boolean copy=false;
    /**
     * seed to use
     */
	public int seed=1;
	/**
	 * True to Use dense data
	 */
	public boolean use_dense=false; 	
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
	private String script_name="KerasnnRegressor.py";
	
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
			    writer.append("l2=" + this.l2+ "\n");
			    writer.append("momentum=" + this.momentum  + "\n");
			    writer.append("epochs=" +this.epochs + "\n");
			    writer.append("batch_size=" + this.batch_size + "\n");
			    writer.append("stopping_rounds=" + this.stopping_rounds+ "\n");
			    writer.append("threads=" + this.threads + "\n");
			    writer.append("validation_split=" + this.validation_split + "\n");
			    writer.append("seed=" + this.seed  + "\n");
			    writer.append("lr=" + this.lr + "\n");
			    writer.append("shuffle=" + this.shuffle + "\n");
			    writer.append("hidden=" + this.hidden + "\n");
			    writer.append("droupouts=" +  this.droupouts + "\n");
			    writer.append("weight_init=" +  this.weight_init + "\n");
			    writer.append("batch_normalization=" +  this.batch_normalization + "\n");			    
			    writer.append("activation=" + this.activation + "\n");
			    writer.append("optimizer=" +  this.optimizer + "\n");
			    writer.append("loss=" +  this.loss + "\n");			    
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
	public KerasnnRegressor(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public KerasnnRegressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public KerasnnRegressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public KerasnnRegressor(smatrix data){
		
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

		if ( !weight_init.equals("RandomNormal")  && !weight_init.equals("RandomUniform") && !weight_init.equals("TruncatedNormal")
				&& !weight_init.equals("VarianceScaling") && !weight_init.equals("Orthogonal")
				&& !weight_init.equals("Identity") && !weight_init.equals("lecun_uniform")
				&& !weight_init.equals("glorot_normal") && !weight_init.equals("glorot_uniform")
				&& !weight_init.equals("he_normal") && !weight_init.equals("lecun_normal")&& !weight_init.equals("he_uniform")){
			throw new IllegalStateException(" weight_init has to be between 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'Identity', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform'" );	
		}
		if ( !optimizer.equals("adagrad")  && !optimizer.equals("adam") && !optimizer.equals("nadam")&& !optimizer.equals("adadelta")&& !optimizer.equals("sgd")){
			throw new IllegalStateException(" optimizer has to be between 'adagrad', 'adam', 'nadam', 'adadelta' or 'sgd'" );	
		}
		if ( !loss.equals("mean_squared_error")  && !loss.equals("mean_absolute_error") && !loss.equals("mean_squared_logarithmic_error")&& !loss.equals("squared_hinge")&& !loss.equals("hinge")&& !loss.equals("poisson")){
			throw new IllegalStateException(" loss has to be between 'categorical_crossentropy', 'categorical_hinge', 'logcosh' or 'Kullback–Leibler divergence'" );	
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
    			throw new IllegalStateException(" hidden needs to have comma sepalrd integer indices .Here it receied: " + ele  );	
    		}
    	}
    	
    	String splits_drops []= this.droupouts.replace(" ","").split(",");
    	if (splits_drops.length!=splits.length){
    		throw new IllegalStateException(" dropouts and hidden neuron units need to have the same length .Here : " + splits_drops.length + "!=" + splits.length  );	
    	}
		for (k=0; k < splits_drops.length; k++){

			String ele=splits_drops[k];
    		try{
    			double hidden_dropout=Double.parseDouble(ele);
    			if (hidden_dropout<0 || hidden_dropout >=1){
    				throw new IllegalStateException(" hidden neurons dropouts' in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated double values .Here it receied: " + ele  );	
    		}
    	}
		
    	String splits_activations []= this.activation.replace(" ","").split(",");
    	if (splits_activations.length!=splits.length){
    		throw new IllegalStateException(" activations and hidden neuron units need to have the same length .Here : " + splits_activations.length + "!=" + splits.length  );	
    	}	
    	
		for (k=0; k < splits_activations.length; k++){
			String ele=splits_activations[k];
			if ( !ele.equals("relu")  && !ele.equals("tanh") & !ele.equals("sigmoid")){
				throw new IllegalStateException(" activations has to be between 'relu', 'tanh' or 'sigmoid'" );	
			}
			
		}
		
    	String splits_l2 []= this.l2.replace(" ","").split(",");
    	if (splits_l2.length!=splits.length){
    		throw new IllegalStateException(" l2 values and hidden neuron units need to have the same length .Here : " + splits_l2.length + "!=" + splits.length  );	
    	}
		for (k=0; k < splits_l2.length; k++){

			String ele=splits_l2[k];
    		try{
    			double hidden_l2=Double.parseDouble(ele);
    			if (hidden_l2<0 || hidden_l2 >=1){
    				throw new IllegalStateException(" l2 regularization in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" l2 needs to have comma separated double values .Here it receied: " + ele  );	
    		}
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
		if (this.lr<=0 || this.lr>=1.0 ){
			this.lr=0.01;
		}	
		if (this.epochs<1){
			this.epochs=1;
		}
		if (this.validation_split<=0){
			this.validation_split=0;
		}	
			
		if (this.stopping_rounds<0){
			this.stopping_rounds=0;
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
		if ( !weight_init.equals("RandomNormal")  && !weight_init.equals("RandomUniform") && !weight_init.equals("TruncatedNormal")
				&& !weight_init.equals("VarianceScaling") && !weight_init.equals("Orthogonal")
				&& !weight_init.equals("Identity") && !weight_init.equals("lecun_uniform")
				&& !weight_init.equals("glorot_normal") && !weight_init.equals("glorot_uniform")
				&& !weight_init.equals("he_normal") && !weight_init.equals("lecun_normal")&& !weight_init.equals("he_uniform")){
			throw new IllegalStateException(" weight_init has to be between 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'Identity', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform'" );	
		}
		if ( !optimizer.equals("adagrad")  && !optimizer.equals("adam") && !optimizer.equals("nadam")&& !optimizer.equals("adadelta")&& !optimizer.equals("sgd")){
			throw new IllegalStateException(" optimizer has to be between 'adagrad', 'adam', 'nadam', 'adadelta' or 'sgd'" );	
		}
		if ( !loss.equals("mean_squared_error")  && !loss.equals("mean_absolute_error") && !loss.equals("mean_squared_logarithmic_error")&& !loss.equals("squared_hinge")&& !loss.equals("hinge")&& !loss.equals("poisson")){
			throw new IllegalStateException(" loss has to be between 'categorical_crossentropy', 'categorical_hinge', 'logcosh' or 'Kullback–Leibler divergence'" );	
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
    			throw new IllegalStateException(" hidden needs to have comma sepalrd integer indices .Here it receied: " + ele  );	
    		}
    	}
    	
    	String splits_drops []= this.droupouts.replace(" ","").split(",");
    	if (splits_drops.length!=splits.length){
    		throw new IllegalStateException(" dropouts and hidden neuron units need to have the same length .Here : " + splits_drops.length + "!=" + splits.length  );	
    	}
		for (k=0; k < splits_drops.length; k++){

			String ele=splits_drops[k];
    		try{
    			double hidden_dropout=Double.parseDouble(ele);
    			if (hidden_dropout<0 || hidden_dropout >=1){
    				throw new IllegalStateException(" hidden neurons dropouts' in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated double values .Here it receied: " + ele  );	
    		}
    	}
		
    	String splits_activations []= this.activation.replace(" ","").split(",");
    	if (splits_activations.length!=splits.length){
    		throw new IllegalStateException(" activations and hidden neuron units need to have the same length .Here : " + splits_activations.length + "!=" + splits.length  );	
    	}	
    	
		for (k=0; k < splits_activations.length; k++){
			String ele=splits_activations[k];
			if ( !ele.equals("relu")  && !ele.equals("tanh") & !ele.equals("sigmoid")){
				throw new IllegalStateException(" activations has to be between 'relu', 'tanh' or 'sigmoid'" );	
			}
			
		}
		
    	String splits_l2 []= this.l2.replace(" ","").split(",");
    	if (splits_l2.length!=splits.length){
    		throw new IllegalStateException(" l2 values and hidden neuron units need to have the same length .Here : " + splits_l2.length + "!=" + splits.length  );	
    	}
		for (k=0; k < splits_l2.length; k++){

			String ele=splits_l2[k];
    		try{
    			double hidden_l2=Double.parseDouble(ele);
    			if (hidden_l2<0 || hidden_l2 >=1){
    				throw new IllegalStateException(" l2 regularization in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" l2 needs to have comma separated double values .Here it receied: " + ele  );	
    		}
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
		if (this.lr<=0 || this.lr>=1.0 ){
			this.lr=0.01;
		}	
		if (this.epochs<1){
			this.epochs=1;
		}
		if (this.validation_split<=0){
			this.validation_split=0;
		}	
			
		if (this.stopping_rounds<0){
			this.stopping_rounds=0;
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
		
		if ( !weight_init.equals("RandomNormal")  && !weight_init.equals("RandomUniform") && !weight_init.equals("TruncatedNormal")
				&& !weight_init.equals("VarianceScaling") && !weight_init.equals("Orthogonal")
				&& !weight_init.equals("Identity") && !weight_init.equals("lecun_uniform")
				&& !weight_init.equals("glorot_normal") && !weight_init.equals("glorot_uniform")
				&& !weight_init.equals("he_normal") && !weight_init.equals("lecun_normal")&& !weight_init.equals("he_uniform")){
			throw new IllegalStateException(" weight_init has to be between 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'Identity', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform'" );	
		}
		if ( !optimizer.equals("adagrad")  && !optimizer.equals("adam") && !optimizer.equals("nadam")&& !optimizer.equals("adadelta")&& !optimizer.equals("sgd")){
			throw new IllegalStateException(" optimizer has to be between 'adagrad', 'adam', 'nadam', 'adadelta' or 'sgd'" );	
		}
		if ( !loss.equals("mean_squared_error")  && !loss.equals("mean_absolute_error") && !loss.equals("mean_squared_logarithmic_error")&& !loss.equals("squared_hinge")&& !loss.equals("hinge")&& !loss.equals("poisson")){
			throw new IllegalStateException(" loss has to be between 'categorical_crossentropy', 'categorical_hinge', 'logcosh' or 'Kullback–Leibler divergence'" );	
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
    			throw new IllegalStateException(" hidden needs to have comma sepalrd integer indices .Here it receied: " + ele  );	
    		}
    	}
    	
    	String splits_drops []= this.droupouts.replace(" ","").split(",");
    	if (splits_drops.length!=splits.length){
    		throw new IllegalStateException(" dropouts and hidden neuron units need to have the same length .Here : " + splits_drops.length + "!=" + splits.length  );	
    	}
		for (k=0; k < splits_drops.length; k++){

			String ele=splits_drops[k];
    		try{
    			double hidden_dropout=Double.parseDouble(ele);
    			if (hidden_dropout<0 || hidden_dropout >=1){
    				throw new IllegalStateException(" hidden neurons dropouts' in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated double values .Here it receied: " + ele  );	
    		}
    	}
		
    	String splits_activations []= this.activation.replace(" ","").split(",");
    	if (splits_activations.length!=splits.length){
    		throw new IllegalStateException(" activations and hidden neuron units need to have the same length .Here : " + splits_activations.length + "!=" + splits.length  );	
    	}	
    	
		for (k=0; k < splits_activations.length; k++){
			String ele=splits_activations[k];
			if ( !ele.equals("relu")  && !ele.equals("tanh") & !ele.equals("sigmoid")){
				throw new IllegalStateException(" activations has to be between 'relu', 'tanh' or 'sigmoid'" );	
			}
			
		}
		
    	String splits_l2 []= this.l2.replace(" ","").split(",");
    	if (splits_l2.length!=splits.length){
    		throw new IllegalStateException(" l2 values and hidden neuron units need to have the same length .Here : " + splits_l2.length + "!=" + splits.length  );	
    	}
		for (k=0; k < splits_l2.length; k++){

			String ele=splits_l2[k];
    		try{
    			double hidden_l2=Double.parseDouble(ele);
    			if (hidden_l2<0 || hidden_l2 >=1){
    				throw new IllegalStateException(" l2 regularization in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" l2 needs to have comma separated double values .Here it receied: " + ele  );	
    		}
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
		if (this.lr<=0 || this.lr>=1.0 ){
			this.lr=0.01;
		}	
		if (this.epochs<1){
			this.epochs=1;
		}
		if (this.validation_split<=0){
			this.validation_split=0;
		}	
			
		if (this.stopping_rounds<0){
			this.stopping_rounds=0;
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
		return "KerasnnRegressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor: KerasnnRegressor");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		
	    System.out.println("objective: " + this.optimizer);
	    System.out.println("loss: " + this.loss  );
	    System.out.println("weight_init: " + this.weight_init);
	    System.out.println("activation: " + this.activation  );  
	    System.out.println("momentum: " + this.momentum);
	    System.out.println("validation_split: " + this.validation_split );
	    System.out.println("l2: " + this.l2 );
	    System.out.println("use_dense: " + this.use_dense );
	    System.out.println("nthread: " + this.threads );
	    System.out.println("epochs: " +  this.epochs );
	    System.out.println("batch_size: " + this.batch_size );
	    System.out.println("hidden: " + this.hidden  );
	    System.out.println("droupouts: " + this.droupouts  );
	    System.out.println("stopping_rounds: " + this.stopping_rounds  );
	    System.out.println("validation_split" + this.validation_split);
	    System.out.println("shuffle: " + this.shuffle ); 
	    System.out.println("standardize: " +  this.standardize );
	    System.out.println("batch_normalization: " + this.batch_normalization );
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
        optimizer = "adam";
        loss="mean_squared_error";
        momentum=0.95;
        use_dense =false;
		validation_split = 0.2;
        threads=1;
        epochs = 10;
        batch_size=1;
        stopping_rounds=0;
    	shuffle=true;
    	standardize=false;	
    	batch_normalization=false;	
    	use_log1p=true;
    	weight_init="lecun_uniform"; //UniformAdaptive, Uniform or Normal
    	activation="relu,relu"; //Has to be Rectifier, Tanh, Maxout or ExpRectifier
    	hidden="50,25";
    	droupouts="0.4,0.2";
    	l2="0.0,0.0";
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
				if (metric.equals("l2")) {this.l2=value;}
				else if (metric.equals("momentum")) {this.momentum=Double.parseDouble(value);}	
				else if (metric.equals("epochs")) {this.epochs=Integer.parseInt(value);}			
				else if (metric.equals("batch_size")) {this.batch_size=Integer.parseInt(value);}				
				else if (metric.equals("stopping_rounds")) {this.stopping_rounds=Integer.parseInt(value);}		
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("use_dense")) {this.use_dense=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("validation_split")) {this.validation_split=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("lr")) {this.lr=Double.parseDouble(value);}					
				else if (metric.equals("shuffle")) {this.shuffle=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("standardize")) {this.standardize=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("batch_normalization")) {this.batch_normalization=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("use_log1p")) {this.use_log1p=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("hidden")) {this.hidden=value;}				
				else if (metric.equals("droupouts")) {this.droupouts=value;}			
				else if (metric.equals("weight_init")) {this.weight_init=value;}				
				else if (metric.equals("activation")) {this.activation=value;}
				else if (metric.equals("optimizer")) {this.optimizer=value;}	
				else if (metric.equals("loss")) {this.loss=value;}	
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
			  

