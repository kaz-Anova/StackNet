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

package ml.h2o;
import io.output;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.math.BigInteger;
import java.net.URL;
import java.net.URLConnection;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import utilis.detectos;
import water.AutoBuffer;
import water.H2O;
import water.H2OApp;
import water.Key;
import water.Keyed;
import water.fvec.FileVec;
import water.fvec.Frame;
import water.fvec.NFSFileVec;
import water.parser.ParseDataset;
import exceptions.DimensionMismatchException;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.tree.gbm.GBM;
import hex.tree.gbm.GBMModel;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;

/**
*<p>Wraps analytics' community favourite  <a href="http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html">GLM</a> from  <a href="https://github.com/h2oai/h2o-3">H2O-3</a>.
*This particular instance is allowing only regression results. H2OGlm models are being trained instantiating an H2O thread-enabled cluster environment 
 <b>It is expected that files will be created</b> and their size will vary based on the volume of the training data.This happens because it is 
 safer to create H2O vectors and frames directly from csvs. Therefore permission to read/write files is mandatory. </p>
*
*<p>Reference : [  Nykodym T.,  Rao A.,  Wang A.,  Kraljevic T.,  Lanford J.,  Hussami N., 2015], <em>Generalized Linear Modeling with H2O</em>
  <a href="http://s3.amazonaws.com/h2o-release/h2o/master/3154/docs-website/h2o-docs/booklets/GLM_Vignette.pdf">Link</a><p>
*
*<p>Information about the tunable parameters can be found <a href="http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html">here</a> </p> 
*
<p> An explanatory video using H2O Glm w can be found <a href="https://www.youtube.com/watch?v=VJPltxh5Q6Q">here</a> . There is another from usage within <a href="https://www.youtube.com/watch?v=g7drhm_SdbQ">Python and R</a>  

*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all H2O Glm's features and the user is advised to use it directly from the <a href="https://github.com/h2oai/h2o-3">source</a>.
*Also the version included is 3.8 and it is based on the <a href="https://github.com/h2oai/h2o-droplets">H2O droplet</a> . The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website, hence it is advised to look for them there for more intuition about what they do.
 */

public class H2OGlmRegressor implements estimator,regressor {
	/**
	 * Parameter for internal use to allow the system to print in the console
	 */
	private transient PrintStream originalStream = System.out;
	/**
	 * Parameter for internal use to block the system to print in the console
	 */
	private transient PrintStream dummyStream  = new PrintStream(new OutputStream(){
	    public void write(int b) {}
		});
	/**
	 * Number of iterations to build teh model
	 */
	public int max_iterations=100;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * Whether to standardize input features or not
	 */
	public boolean standardize=true;
	/**
	 * Proportion of l1-l2. If 0 , it is ridge . If 1.o, it is Lasso
	 */
	public double alpha=0.0;
	/**
	 * Regularization parameter
	 */
	public double lambda=0.001;
	/**
	 * tolerance of the objective function
	 */
	public double objective_epsilon=0.001;
	/**
	 * tolerance of the coefficients
	 */
	public double beta_epsilon=0.95;	
	/**
	 * Tolerance specific to the family distribution
	 */
	public double family_epsilon=0.001;		
	/**
	 * (Only applicable if Tweedie is specified for distribution) Specify the Tweedie power. The range is from 1 to 2. For a normal distribution, enter 0. For Poisson distribution, enter 1. For a gamma distribution, enter 2. For a compound Poisson-gamma distribution, enter a value greater than 1 but less than 2.
	 */
	public double tweedie_power=1.5;	
	/**
	 * (Only applicable if Quantile is specified for distribution) Specify the quantile to be used for Quantile Regression.
	 */
	public double quantile_alpha=0;	
	/**
	 * The family has to be auto, gamma gaussian poisson tweedie
	 */
	public String family="gaussian";	
	/**
	 * The link has to be auto, log identity inverse  tweedie
	 */
	public String link="auto";	
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
	    * @param name : name of the file to load to train the model
	    * @param model_name : name of the output model
	    */
	   private void create_h2o_cluster(String name , String model_name) {
		   

		   // create the subprocess
			try {
				System.setOut(dummyStream);
				
				String url= "http://localhost:54321/";
				try {
				    URL myURL = new URL(url);
				    // also you can put a port 
				   //  URL myURL = new URL("http://localhost:8080");
				    URLConnection myURLConnection = myURL.openConnection();
				    myURLConnection.connect();
				} 
				catch (Exception e) {
					  H2OApp.main(new String[]{"-name", this.model_name + "1", "-nthreads",this.threads + "" });
					  H2O.waitForCloudSize(1, 100*100/* ms */);
				}

				  
				  FileVec vs = NFSFileVec.make(new File (name)); 
				  Frame fr = ParseDataset.parse(Key.make(this.model_name + "_train"),vs._key);
				  //fr.replace(0,fr.vec("C1").toCategoricalVec()).remove();  
				  vs.remove();
				  /*
				  for (int j=1; j < fr.numCols(); j++){
					  if (fr.vec(j).isCategorical()){
						  fr.replace(j,fr.vec(j).toNumericVec()).remove();  
					  }
				  }
				  */
				  
				  
				  GLMModel.GLMParameters glmp = new GLMModel.GLMParameters();
				  
			      glmp._balance_classes=false;
			      glmp._max_iterations=this.max_iterations;
			      glmp._standardize=this.standardize;
			      glmp._alpha=new double []{this.alpha};
			      glmp._lambda=new double []{this.lambda};

			      glmp._beta_epsilon=this.beta_epsilon;
			      glmp._quantile_alpha=this.quantile_alpha;
			      glmp._tweedie_power=this.tweedie_power; 
			      glmp._ignore_const_cols=true;
			      glmp._max_confusion_matrix_size=0;      
			      glmp._stopping_tolerance=0;     
			      glmp._score_each_iteration=false;
			      glmp._stopping_rounds=0;
			      glmp._max_runtime_secs=0; 
			      glmp._keep_cross_validation_predictions=false; 
			      
			      glmp._family=glmp._family.multinomial; 
			      glmp._link=glmp._link.multinomial; 

			      if (this.n_classes==2){
				      glmp._family=glmp._family.binomial; //gamma binomial multinomial gaussian poisson tweedie
				      glmp._link=glmp._link.logit; //logit log identity inverse family_default multinomial tweedie

			      }
				  glmp._link=glmp._link.family_default;
			      if (this.link.equals("identity")){
			    	  glmp._link=glmp._link.identity;
			      }else if (this.link.equals("log")){
			    	  glmp._link=glmp._link.log;
			      }else if (this.link.equals("inverse")){
			    	  glmp._link=glmp._link.inverse;			    	  
			      }else if (this.link.equals("inverse")){
			    	  glmp._link=glmp._link.inverse;
			      }
			      
			      if (this.family.equals("gamma")){
			    	  glmp._family=glmp._family.gamma;
			      }else if(this.family.equals("gaussian")){
			    	  glmp._family=glmp._family.gaussian;
			      }else if(this.family.equals("poisson")){
			    	  glmp._family=glmp._family.poisson;
			      }else if(this.family.equals("tweedie")){
			    	  glmp._family=glmp._family.tweedie;
			      }
			      glmp._train = fr._key;
			      glmp._response_column = "C1";
			      
			      GLM glm = new GLM(glmp);
			      GLMModel m = null;
			      
					try {
						if (this.verbose){
							System.setOut(this.originalStream);
						}
						 m = glm.trainModel().get();
						 fr.delete();

						 System.setOut(this.dummyStream);
					      OutputStream  os = new FileOutputStream(new File (model_name));
					      m.writeAll(new AutoBuffer(os,true)).close();
						} catch (Exception e1) {
							System.setOut(this.originalStream);
							//H2O.shutdown(-1);
							throw new IllegalStateException(" failed to make a H2O GLM model... due to " + e1.getLocalizedMessage());
						}
			      

			} catch (Exception e) {
				System.setOut(this.originalStream);
				//H2O.shutdown(-1);
				throw new IllegalStateException(" failed to create an H2O cluster at trining time .. due to " + e.getLocalizedMessage());
			}
			//H2O.die(this.model_name + "1");
			
			//H2O.shutdown(-1);
			//H2O.requestShutdown();
			System.setOut(this.originalStream);
	   }
	   /**
	    * 
	    * @param name : name of the file to load to predict the model
	    * @param model_name : name of the input model
	    */
	   private double [] create_h2o_cluster_predict(String name, String model_name ) {
		   
		   System.setOut(dummyStream);
		   double pred []=null;
		   // create the subprocess
			try {
				String url= "http://localhost:54321/";
				try {
				    URL myURL = new URL(url);
				    // also you can put a port 
				   //  URL myURL = new URL("http://localhost:8080");
				    URLConnection myURLConnection = myURL.openConnection();
				    myURLConnection.connect();
				} 
				catch (Exception e) {
				      H2OApp.main(new String[]{"-name", this.model_name+ "1", "-nthreads",this.threads + "" });
					  H2O.waitForCloudSize(1, 100*100/* ms */);
				}
				  
				  FileVec vs = NFSFileVec.make(new File (name)); 
				  Frame test = ParseDataset.parse(Key.make(this.model_name + "_test"),vs._key);
				  test.remove("C1");
				  int rows=(int) test.numRows();
				  
				  pred = new double [rows];
				  //String namesss [] = new String[test.numCols()];
				  /*
				  for (int j=0; j < namesss.length; j++){
					  namesss[j]="C" + (j+2);
				  }
				  System.out.println(Arrays.toString(test.names())); 
				  test.setNames(namesss);
				 
				  System.out.println(Arrays.toString(test.names()));
				  for (int j=0; j < test.numCols(); j++){
					  if (test.vec(j).isCategorical()){
						  test.replace(j,test.vec(j).toNumericVec()).remove();  
					  }
				  }		  
				   */

				
				  GLMModel m = null;
				  
				  try {
						InputStream in= new FileInputStream(new File (model_name));
						AutoBuffer ab = new AutoBuffer(in);
						m = (GLMModel)Keyed.readAll(ab);
						Frame predictions = m.score(test);
						test.delete();
						
						  for (int i=0; i < rows; i++){
								  pred[i]=predictions.vec(0).at(i);
					      }
						predictions.delete();
					  } catch (Exception e1) {
						  //H2O.shutdown(-1);
							throw new IllegalStateException(" failed to score with a H2O GLM model... due to... " + e1.getMessage());
							
					  }


			} catch (Exception e) {
				//H2O.shutdown(-1);
				throw new IllegalStateException(" failed to create an H2O cluster and successfully at predict time due to... " + e.getMessage() );
			}
			//H2O.shutdown(-1);
			System.setOut(this.originalStream);
			return pred;
		   
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
	public H2OGlmRegressor(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public H2OGlmRegressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public H2OGlmRegressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public H2OGlmRegressor(smatrix data){
		
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
		out.printsmatrix(X,new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		for (int n=0; n < this.n_classes;n++) {


				 double temp []=create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".mod");
						 
						 
						 io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", ",", 0, 0.0, false, false);

					 if (temp.length!=predictions.length){
						 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of dataset length" );
					 }
					 for (int i =0; i <predictions.length;i++ ){
							 predictions[i][n]=temp[i];
						 
					 } 
				        // create new file
    
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
		out.printsmatrix(X,new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		for (int n=0; n < this.n_classes;n++) {

					double temp []=create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".mod");
				
			
					 if (temp.length!=predictions.length){
						 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".pred " + " is not of dataset length" );
					 }
					 for (int i =0; i <predictions.length;i++ ){
							 predictions[i][n]=temp[i];
						 
					 } 
				        // create new file
 
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
		out.printsmatrix(data,new double [data.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 

		System.gc();
		
		for (int n=0; n < this.n_classes;n++) {

			 		double temp []=create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",this.usedir +  File.separator +  "models"+File.separator + this.model_name  + n + ".mod");
				
					 if (temp.length!=predictions.length){
						 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of dataset length" );
					 }
					 for (int i =0; i <predictions.length;i++ ){
							 predictions[i][n]=temp[i];
						 
					 } 
				        // create new file

						File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + n + ".conf" );
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
		out.printsmatrix(X,new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		 double temp []=create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",this.usedir +  File.separator +  "models"+File.separator + this.model_name  + "0.mod");

		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of dataset length" );
		 }		 
		 
		 
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i]= temp[i];
			  
		 }
		 
		    
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
		out.printsmatrix(data,new double [data.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 

		System.gc();
		
		 double temp []=create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",this.usedir +  File.separator +  "models"+File.separator + this.model_name  + "0.mod");
		 
		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of dataset length" );
		 }		 		 

		for (int i =0; i <predictions.length;i++ ){
				 predictions[i]=temp[i];

			 } 
		 
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
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
		out.printsmatrix(X,new double [X.GetRowDimension()] ,  this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		 double temp []=create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",this.usedir +  File.separator +  "models"+File.separator + this.model_name  + "0.mod");		 if (temp.length!=predictions.length){
			 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of dataset length" );
		 }			 

			for (int i =0; i <predictions.length;i++ ){
				predictions[i]=temp[i];
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
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

		if ( !family.equals("gaussian") && !family.equals("auto")  && !family.equals("gamma")  && !family.equals("poisson")&&  !family.equals("tweedie") ){
			throw new IllegalStateException(" family has to be in [auto,gaussian,gamma ,poisson ,tweedie]  " );	
		}
		if ( !link.equals("log") && !link.equals("auto")  && !link.equals("identity")  && !link.equals("inverse")&&  !link.equals("tweedie") ){
			throw new IllegalStateException(" link has to be in [auto,log,identity ,inverse ,tweedie]  " );	
		}		
		if (this.alpha<0){
			this.alpha=0;
		}	
		if (this.beta_epsilon<=0){
			this.beta_epsilon=0.000001;
		}
		if (this.max_iterations<1){
			this.max_iterations=1;
		}
		if (this.lambda<=0){
			this.lambda=1;
		}		
		if (this.tweedie_power<=1 ||this.tweedie_power>=2  ){
			throw new IllegalStateException(" tweedie_power has to be in in (1,2) " );	
		}	
		if (this.quantile_alpha<=0){
			this.quantile_alpha=0.1;
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
				
				//make subprocess
				create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod");
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" );
		        // tries to delete a non-existing file
		        f.delete();
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
				

				//make subprocess
				 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".mod");
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" );
		        // tries to delete a non-existing file
		        f.delete();


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
		if ( !family.equals("gaussian") && !family.equals("auto")  && !family.equals("gamma")  && !family.equals("poisson")&&  !family.equals("tweedie") ){
			throw new IllegalStateException(" family has to be in [auto,gaussian,gamma ,poisson ,tweedie]  " );	
		}
		if ( !link.equals("log") && !link.equals("auto")  && !link.equals("identity")  && !link.equals("inverse")&&  !link.equals("tweedie") ){
			throw new IllegalStateException(" link has to be in [auto,log,identity ,inverse ,tweedie]  " );	
		}		
		if (this.alpha<0){
			this.alpha=0;
		}	
		if (this.beta_epsilon<=0){
			this.beta_epsilon=0.000001;
		}
		if (this.max_iterations<1){
			this.max_iterations=1;
		}
		if (this.lambda<=0){
			this.lambda=1;
		}		
		if (this.tweedie_power<=1 ||this.tweedie_power>=2  ){
			throw new IllegalStateException(" tweedie_power has to be in in (1,2) " );	
		}	
		if (this.quantile_alpha<=0){
			this.quantile_alpha=0.1;
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
				
				//make subprocess
				 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod");
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" );
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
				
				//make subprocess
				 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".mod");
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" );
		        // tries to delete a non-existing file
		        f.delete();



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
		
		if ( !family.equals("gaussian")  && !family.equals("auto")  && !family.equals("gamma") && !family.equals("laplace") && !family.equals("huber")&& !family.equals("poisson")&& !family.equals("quantile")&& !family.equals("tweedie") ){
			throw new IllegalStateException(" family has to be in [auto,gaussian,gamma ,laplace , huber ,poisson, quantile ,tweedie]  " );	
		}
		
		if ( !family.equals("gaussian") && !family.equals("auto")  && !family.equals("gamma")  && !family.equals("poisson")&&  !family.equals("tweedie") ){
			throw new IllegalStateException(" family has to be in [auto,gaussian,gamma ,poisson ,tweedie]  " );	
		}
		if ( !link.equals("log") && !link.equals("auto")  && !link.equals("identity")  && !link.equals("inverse")&&  !link.equals("tweedie") ){
			throw new IllegalStateException(" link has to be in [auto,log,identity ,inverse ,tweedie]  " );	
		}		
		if (this.alpha<0){
			this.alpha=0;
		}	
		if (this.beta_epsilon<=0){
			this.beta_epsilon=0.000001;
		}
		if (this.max_iterations<1){
			this.max_iterations=1;
		}
		if (this.lambda<=0){
			this.lambda=1;
		}		
		if (this.tweedie_power<=1 ||this.tweedie_power>=2  ){
			throw new IllegalStateException(" tweedie_power has to be in in (1,2) " );	
		}	
		if (this.quantile_alpha<=0){
			this.quantile_alpha=0.1;
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
				
				//make subprocess
				 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod");
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.train" );
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
				
				//make subprocess
				 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".mod");
				sdataset=null;
				System.gc();
				
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +  n +".train" );
		        // tries to delete a non-existing file
		        f.delete();



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
		return "H2OGlmRegressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor: H2OGlmRegressor");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
	    System.out.println("quantile_alpha: " + this.quantile_alpha );	
	    System.out.println("tweedie_power: " + this.tweedie_power  );
	    System.out.println("family: " + this.family);
	    System.out.println("link: " + this.link );	    
	    
	    System.out.println("alpha: " + this.alpha);
	    System.out.println("lambda: " + this.lambda );
	    System.out.println("beta_epsilon: " + this.beta_epsilon );

	    System.out.println("nthread: " + this.threads );
	    System.out.println("max_iterations: " +  this.max_iterations );
	    System.out.println("standardize: " + this.standardize );
	    
	    if (this.verbose){
	    	System.out.println("verbose: true");
	    }else {
	    	System.out.println("verbose: false");
	    }
		if (new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()){
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
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.mod" );
        f.delete();  

        family = "auto";
        link = "auto";

        alpha=0.0;
        lambda=0.0001;
        beta_epsilon =0.001;
        quantile_alpha =0.1;
        tweedie_power =1.2;    
        threads=4;
        max_iterations = 100;
        standardize=false;
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
				else if (metric.equals("beta_epsilon")) {this.beta_epsilon=Double.parseDouble(value);}	
				else if (metric.equals("max_iterations")) {this.max_iterations=Integer.parseInt(value);}
				else if (metric.equals("objective_epsilon")) {this.objective_epsilon=Double.parseDouble(value);}					
				else if (metric.equals("standardize")) {this.standardize=(value.toLowerCase().equals("true")?true:false)   ;}					
						
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("lambda")) {this.lambda=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("tweedie_power")) {this.tweedie_power=Double.parseDouble(value);}
				else if (metric.equals("quantile_alpha")) {this.quantile_alpha=Double.parseDouble(value);}								
				else if (metric.equals("family")) {this.family=value;}	
				else if (metric.equals("link")) {this.link=value;}				

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
	public void set_target(fsmatrix fstarget){
		this.fstarget=fstarget;
	}
	@Override
	public void AddClassnames(String names[]){
		//none
	}		
}
			  

