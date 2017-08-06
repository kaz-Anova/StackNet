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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.math.BigInteger;
import java.net.URL;
import java.net.URLConnection;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import utilis.map.intint.StringIntMap4a;
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
import exceptions.LessThanMinimum;
import hex.deeplearning.DeepLearning;
import hex.deeplearning.DeepLearningModel;
import hex.tree.gbm.GBM;
import hex.tree.gbm.GBMModel;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;

/**
*<p>Wraps analytics' community favourite  <a href="https://s3.amazonaws.com/h2o-release/h2o/rel-markov/1/docs-website/datascience/deeplearning.html">Deep Learning</a> from  <a href="https://github.com/h2oai/h2o-3">H2O-3</a>.
*This particular instance is allowing only classification results. H2O Deep Learning models are being trained instantiating an H2O thread-enabled cluster environment 
 <b>It is expected that files will be created</b> and their size will vary based on the volume of the training data.This happens because it is 
 safer to create H2O vectors and frames directly from csvs. Therefore permission to read/write files is mandatory. </p>
*
*<p>Reference : [ Parmar V., Candel A. , 2015], <em>Deep Learning with H2O</em>
  <a href="https://raw.githubusercontent.com/h2oai/h2o/master/docs/deeplearning/DeepLearningBookletV1.pdf">Link</a><p>
*
*<p>Information about the tunable parameters can be found <a href="https://s3.amazonaws.com/h2o-release/h2o/rel-markov/1/docs-website/datascience/deeplearning.html">here</a> </p> 
*
<p> An explanatory video using H2O Deepleanring and tips can be found <a href="https://www.youtube.com/watch?v=LM255qs8Zsk">here</a> . There is another from usage within <a href="https://www.youtube.com/watch?v=ZMcbjL2U34Y">R</a>  

*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all H2O Deep Learning's features and the user is advised to use it directly from the <a href="https://github.com/h2oai/h2o-3">source</a>.
*Also the version included is 3.8 and it is based on the <a href="https://github.com/h2oai/h2o-droplets">H2O droplet</a> . The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website, hence it is advised to look for them there for more intuition about what they do.
 */

public class H2ODeepLearningClassifier implements estimator,classifier {
	
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
	 * Number of hidden neurons. The length connotes the number of hidden layers too
	 */
	private int [] hidden_neurons=new int []{50,25};
	/**
	 * Number of hidden neurons, comma separated.The length connotes the number of hidden layers too
	 */
	public String hidden="50,25";	
	/**
	 * dropout ratio for each hidden layer .Has to match in length the 'hidden' parameter
	 */
	private double [] hidden_dropout=new double []{0.4,0.2};
	/**
	 * dropout ratin for each hidden layer,comma separated .Has to match in length the 'hidden' parameter
	 */
	public String droupouts="0.4,0.2";	
	/**
	 * Number of iterations to train the DL model
	 */
	public int epochs=10;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * minimum number of cases in batch
	 */
	public int mini_batch_size=1;
	/**
	 * dropout from to the input layer
	 */
	public double input_dropout_ratio=0.0;	
	/**
	 * The first of two hyper parameters for ADADELTA. It is similar to momentum and relates to the memory to prior weight updates. Typical values are between 0.9 and 0.999. This parameter is only active if adaptive learning rate is enabled.
	 */
	public double rho=0.95;
	/**
	 * The second of two hyper parameters for ADADELTA. It is similar to learning rate annealing during initial training and momentum at later stages where it allows forward progress. Typical values are between 1e-10 and 1e-4. This parameter is only active if adaptive learning rate is enabled.
	 */
	public double epsilon=1e-8;
	/**
	 * L1 regularization on the weights
	 */
	public double l1=0;
	/**
	 * Proportions of rows consider in each epoc
	 */
	public double sample_rate=0.95;	
	/**
	 * L2 regularization on the weights
	 */
	public double l2=0;	
	/**
	 * A maximum on the sum of the squared incoming weights into any one neuron. This tuning parameter is especially useful for unbound activation functions such as Maxout or Rectifier.
	 */
	public double max_w2=1.0;	
	/**
	 * When adaptive learning rate is disabled, the magnitude of the weight updates are determined by the user specified learning rate (potentially annealed), and are a function of the difference between the predicted value and the target value. 
	 */
	public double rate=0.1;	
	/**
	 * Learning rate annealing reduces the learning rate to "freeze" into local minima in the optimization landscape
	 */
	public double rate_annealing=1e-6 ;	
	/**
	 * The learning rate decay parameter controls the change of learning rate across layers.
	 */
	public double rate_decay=0.01;	
	/**
	 * The momentum_start parameter controls the amount of momentum at the beginning of training. This parameter is only active if adaptive learning rate is disabled.
	 */
	public double momentum_start=0.95;		
	/**
	 * The momentum_ramp parameter controls the amount of learning for which momentum increases (assuming momentum_stable is larger than momentum_start).
	 */
	public double momentum_ramp=0.92;		
	/**
	 * The momentum_stable parameter controls the final momentum value reached after momentum_ramp training samples.
	 */
	public double momentum_stable=0.9;			
	/**
	 * Specify whether to oversample the minority classes to balance the class distribution.
	 */
	public boolean balance_classes=false;	
	/**
	 * Enable shuffling of training data (on each node). This option is recommended if training data is replicated on N nodes, and the number of training samples per iteration is close to N times the dataset size, where all nodes train will (almost) all the data
	 */
	public boolean shuffle=true;
	/**
	 * standardizes data in a batch
	 */
	public boolean standardize=false;	
	/**
	 * for faster convergence (but potential loss in accuracy)
	 */
	public boolean fast_mode=false;	
	/**
	 * The implemented adaptive learning rate algorithm (ADADELTA) automatically combines the benefits of learning rate annealing and momentum training to avoid slow convergence. Specification of only two parameters (rho and epsilon) simplifies hyper-parameter search
	 */
	public boolean adaptive_rate=true;
	/**
	 * to enable Nesterov accelerated gradient descent method
	 */
	public boolean nesterov_accelerated_gradient=false;
	/**
	 * he distribution from which initial weights are to be drawn. Has to be UniformAdaptive, Uniform or Normal
	 */
	public String weight_init="UniformAdaptive";
	/**
	 * activation functions. Has to be Rectifier, Tanh, Maxout or ExpRectifier
	 */
	public String activation="Rectifier";
	/**
	 * The objective has to be bernouli.
	 */
	public String Objective="bernouli";	
	/**
	 * The loss has to be CrossEntropy
	 */
	private String loss="CrossEntropy";		
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
			  fr.replace(0,fr.vec("C1").toCategoricalVec()).remove();  
			  vs.remove();
			  /*
			  for (int j=1; j < fr.numCols(); j++){
				  if (fr.vec(j).isCategorical()){
					  fr.replace(j,fr.vec(j).toNumericVec()).remove();  
				  }
			  }
			  */
			  
			  DeepLearningModel.DeepLearningParameters glmp = new DeepLearningModel.DeepLearningParameters();
			 
			  glmp._epochs=this.epochs;
			  glmp._activation=glmp._activation.RectifierWithDropout;
		      if (this.activation.equals("Maxout")){
		    	  glmp._activation=glmp._activation.MaxoutWithDropout;
		      }else if (this.activation.equals("Tanh")){
		    	  glmp._activation=glmp._activation.TanhWithDropout;
		      }else if (this.activation.equals("ExpRectifierWithDropout")){
		    	  glmp._activation=glmp._activation.ExpRectifierWithDropout;
		      }
		      
			  glmp._initial_weight_distribution=glmp._initial_weight_distribution.UniformAdaptive;
			  if (this.weight_init.equals("Uniform")){
		    	  glmp._initial_weight_distribution=glmp._initial_weight_distribution.Uniform;
		      }else  if (this.weight_init.equals("Normal")){
		    	  glmp._initial_weight_distribution=glmp._initial_weight_distribution.Normal;		      }
		     
			  glmp._distribution=glmp._distribution.bernoulli;
			  glmp._loss=glmp._loss.Automatic;

			  //Tanh Rectifier Maxout Dropout ExpRectifier
			  glmp._input_dropout_ratio=this.input_dropout_ratio;
			  glmp._hidden=this.hidden_neurons;
			  glmp._hidden_dropout_ratios=this.hidden_dropout;

			  glmp._shuffle_training_data=this.shuffle;
			  glmp._mini_batch_size=this.mini_batch_size;
			  glmp._standardize=this.standardize;
			  //UniformAdaptive Uniform Normal
			  glmp._train_samples_per_iteration=(long) Math.max(4,(this.sample_rate*fr.numRows()));
			  glmp._fast_mode=this.fast_mode;

			  //CrossEntropy Absolute Huber Quadratic Quantile
			  glmp._adaptive_rate=this.adaptive_rate;
			  glmp._rho=this.rho;
			  glmp._epsilon=this.epsilon;
			  glmp._l1=this.l1;
			  glmp._l2=this.l2;	  
			  glmp._max_w2=(float) this.max_w2;
			  glmp._rate=this.rate;
			  glmp._rate_annealing=this.rate_annealing ;
			  glmp._rate_decay= this.rate_decay;
			  glmp._momentum_start=this.momentum_start;
			  glmp._momentum_ramp=this.momentum_ramp;
			  glmp._momentum_stable=this.momentum_stable;
			  glmp._nesterov_accelerated_gradient=this.nesterov_accelerated_gradient;

		      glmp._balance_classes=this.balance_classes;

		      //glmp._distribution.gamma gaussian huber laplace multinomial poisson quantile tweedie
		      glmp._seed=1;

		      
		      glmp._score_interval=2000000000;
		      glmp._max_confusion_matrix_size=0;      
		      glmp._stopping_tolerance=0;     
		      glmp._score_each_iteration=false;
		      glmp._stopping_rounds=0;
		      glmp._max_runtime_secs=0; 
		      glmp._keep_cross_validation_predictions=false;       
		      
		      
		      glmp._train = fr._key;
		      glmp._response_column = "C1";
		      
		      
		      DeepLearning glm = new DeepLearning(glmp);

		      DeepLearningModel m = null;
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
						throw new IllegalStateException(" failed to make a H2O DL model.... due to " + e1.getLocalizedMessage());
					}
		} catch (Exception e) {
			System.setOut(this.originalStream);
			//H2O.shutdown(-1);
			throw new IllegalStateException(" failed to create an H2O cluster at trining time.. due to " + e.getLocalizedMessage());
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
   private double [][] create_h2o_cluster_predict(String name, String model_name ) {
	   
	   System.setOut(dummyStream);
	   double pred [][]=null;
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
			  
			  pred = new double [rows][this.n_classes];
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

			
			  DeepLearningModel m = null;
			  
			  try {
					InputStream in= new FileInputStream(new File (model_name));
					AutoBuffer ab = new AutoBuffer(in);
					m = (DeepLearningModel)Keyed.readAll(ab);
					Frame predictions = m.score(test);
					test.delete();
					
					  for (int i=0; i < rows; i++){
						  for (int j=0; j < this.n_classes; j++){
							  pred[i][j]=predictions.vec(j+1).at(i);
							  //System.out.println("pred[" + i + "][" +j+"]: " + pred[i][j]);
						  }
				      }
					predictions.delete();
				  } catch (Exception e1) {
					  //H2O.shutdown(-1);
						throw new IllegalStateException(" failed to score with a H2O DL model... due to... " + e1.getMessage());
						
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
	public H2ODeepLearningClassifier(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public H2ODeepLearningClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public H2ODeepLearningClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public H2ODeepLearningClassifier(smatrix data){
		
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

		
		//generate dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,new double [X.GetRowDimension()] , this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		double predictions[][]=this.create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test", this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod"); 

        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
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
		


		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		
		System.gc();
		
		double predictions[][]=this.create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test", this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod"); 

        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
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


		output out = new output();
		out.verbose=false;
		out.printsmatrix(data,new double [data.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		
		double predictions[][]=this.create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test", this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod"); 

        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
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
		


		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		double predictions[]= new double [data.GetRowDimension()];
		double prediction_probas[][]=this.create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test", this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod"); 
		double temp[]=null;
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();    
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			 temp=prediction_probas[i];
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

		output out = new output();
		out.verbose=false;
		out.printsmatrix(data,new double [data.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 

		System.gc();
		
		
		double predictions[]= new double [data.GetRowDimension()];
		double prediction_probas[][]=this.create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test", this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod"); 
		double temp[]=null;
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();    
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			 temp=prediction_probas[i];
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
		
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,new double [X.GetRowDimension()],this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		double predictions[]= new double [data.length];
		double prediction_probas[][]=this.create_h2o_cluster_predict(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test", this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod"); 
		double temp[]=null;
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();    
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			 temp=prediction_probas[i];
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
		
		if ( !weight_init.equals("UniformAdaptive")  && !weight_init.equals("Uniform") & !weight_init.equals("Normal")){
			throw new IllegalStateException(" weight_init has to be between 'UniformAdaptive', 'Uniform' or 'Normal'" );	
		}
		if ( !activation.equals("Rectifier")  && !activation.equals("Tanh") & !activation.equals("Maxout")& !activation.equals("ExpRectifier")){
			throw new IllegalStateException(" activation has to be between 'Rectifier', 'Tanh', 'ExpRectifier' or 'Maxout'" );	
		}
    	String splits [] = this.hidden.replace(" ","").split(",");
		hidden_neurons= new int [splits.length];
		int k=0;
    	for (String ele: splits){
    		try{
    			hidden_neurons[k]=Integer.parseInt(ele);
	
    			if (hidden_neurons[k]<0){
    				throw new IllegalStateException(" hidden neurons in a layer cannot be less than /equal to zero " );		
    			}
    			k+=1;
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated integer indices .Here it receied: " + ele  );	
    		}
    	}
    	splits = this.droupouts.replace(" ","").split(",");
		hidden_dropout= new double [hidden_neurons.length];
		for (k=0; k < splits.length; k++){
			if (k>=hidden_neurons.length){
				break;
			}
			String ele=splits[k];
    		try{
    			hidden_dropout[k]=Double.parseDouble(ele);
    			if (hidden_dropout[k]<0 || hidden_dropout[k] >=1){
    				throw new IllegalStateException(" hidden neurons dropouts' in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated double values .Here it receied: " + ele  );	
    		}
    	}	    
		
		if (this.mini_batch_size<=0){
			this.mini_batch_size=1;
		}	
		if (this.rho<=0 || this.rho>=1.0 ){
			this.rho=0.9;
		}	
		if (this.l2<0){
			this.l2=0;
		}
		if (this.sample_rate<=0 || this.sample_rate>=1.0){
			this.sample_rate=1;
		}
		if (this.epochs<1){
			this.epochs=1;
		}
		if (this.epsilon<=0){
			this.epsilon=1e-4;
		}	
		if (this.l1<0){
			this.l1=0.0;
		}			
		if (this.input_dropout_ratio<0){
			this.input_dropout_ratio=0.1;
		}
		if (this.max_w2<0){
			this.max_w2=1.;
		}	
		if (this.rate<0){
			this.rate=0.1;
		}	
		if (this.rate_annealing<0){
			this.rate_annealing=1e-6;
		}	
		if (this.rate_decay<0){
			this.rate_decay=0.01;
		}			
		if (this.momentum_start<0 || this.momentum_start>=1.0){
			this.momentum_start=0.95;
		}		
		if (this.momentum_ramp<0 || this.momentum_ramp>=1.0){
			this.momentum_ramp=0.95;
		}	
		if (this.momentum_stable<0 || this.momentum_stable>=1.0){
			this.momentum_stable=0.95;
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
		
		this.Objective="bernouli";
		if (this.n_classes>2){
			this.Objective="multiclass";
		}

		
		columndimension=data[0].length;
		//generate config file
		
		//generate dataset
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
		
		//make subprocess
		 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
        // tries to delete a non-existing file
        f.delete();	
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

		if ( !weight_init.equals("UniformAdaptive")  && !weight_init.equals("Uniform") & !weight_init.equals("Normal")){
			throw new IllegalStateException(" weight_init has to be between 'UniformAdaptive', 'Uniform' or 'Normal'" );	
		}
		if ( !activation.equals("Rectifier")  && !activation.equals("Tanh") & !activation.equals("Maxout")& !activation.equals("ExpRectifier")){
			throw new IllegalStateException(" activation has to be between 'Rectifier', 'Tanh', 'ExpRectifier' or 'Maxout'" );	
		}
    	String splits [] = this.hidden.replace(" ","").split(",");
		hidden_neurons= new int [splits.length];
		int k=0;
    	for (String ele: splits){
    		try{
    			hidden_neurons[k]=Integer.parseInt(ele);
    			if (hidden_neurons[k]<0){
    				throw new IllegalStateException(" hidden neurons in a layer cannot be less than /equal to zero " );		
    			}
    			k+=1;
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated integer indices .Here it received: " + ele  );	
    		}
    	}
    	splits = this.droupouts.replace(" ","").split(",");
		hidden_dropout= new double [hidden_neurons.length];
		for (k=0; k < splits.length; k++){
			if (k>=hidden_neurons.length){
				break;
			}
			String ele=splits[k];
    		try{
    			hidden_dropout[k]=Double.parseDouble(ele);
    			if (hidden_dropout[k]<0 || hidden_dropout[k] >=1){
    				throw new IllegalStateException(" hidden neurons dropouts' in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated double values .Here it receied: " + ele  );	
    		}
    	}	    
		
		if (this.mini_batch_size<=0){
			this.mini_batch_size=1;
		}	
		if (this.rho<=0 || this.rho>=1.0 ){
			this.rho=0.9;
		}	
		if (this.l2<0){
			this.l2=0;
		}
		if (this.sample_rate<=0 || this.sample_rate>=1.0){
			this.sample_rate=1;
		}
		if (this.epochs<1){
			this.epochs=1;
		}
		if (this.epsilon<=0){
			this.epsilon=1e-4;
		}	
		if (this.l1<0){
			this.l1=0.0;
		}			
		if (this.input_dropout_ratio<0){
			this.input_dropout_ratio=0.1;
		}
		if (this.max_w2<0){
			this.max_w2=1.;
		}	
		if (this.rate<0){
			this.rate=0.1;
		}	
		if (this.rate_annealing<0){
			this.rate_annealing=1e-6;
		}	
		if (this.rate_decay<0){
			this.rate_decay=0.01;
		}			
		if (this.momentum_start<0 || this.momentum_start>=1.0){
			this.momentum_start=0.95;
		}		
		if (this.momentum_ramp<0 || this.momentum_ramp>=1.0){
			this.momentum_ramp=0.95;
		}	
		if (this.momentum_stable<0 || this.momentum_stable>=1.0){
			this.momentum_stable=0.95;
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
			
		} else {
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
		
		this.Objective="bernouli";
		if (this.n_classes>2){
			this.Objective="multiclass";
		}

		
		columndimension=data.GetColumnDimension();
		//generate config file
		
		//generate dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		
		//make subprocess
		 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
		 
       // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
       // tries to delete a non-existing file
       f.delete();	
		
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

		if ( !weight_init.equals("UniformAdaptive")  && !weight_init.equals("Uniform") & !weight_init.equals("Normal")){
			throw new IllegalStateException(" weight_init has to be between 'UniformAdaptive', 'Uniform' or 'Normal'" );	
		}
		if ( !activation.equals("Rectifier")  && !activation.equals("Tanh") & !activation.equals("Maxout")& !activation.equals("ExpRectifier")){
			throw new IllegalStateException(" activation has to be between 'Rectifier', 'Tanh', 'ExpRectifier' or 'Maxout'" );	
		}
    	String splits [] = this.hidden.replace(" ","").split(",");
		hidden_neurons= new int [splits.length];
		int k=0;
    	for (String ele: splits){
    		try{
    			hidden_neurons[k]=Integer.parseInt(ele);
	
    			if (hidden_neurons[k]<0){
    				throw new IllegalStateException(" hidden neurons in a layer cannot be less than /equal to zero " );		
    			}
    			k+=1;	
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated integer indices .Here it receied: " + ele  );	
    		}
    	}
    	splits = this.droupouts.replace(" ","").split(",");
		hidden_dropout= new double [hidden_neurons.length];
		for (k=0; k < splits.length; k++){
			if (k>=hidden_neurons.length){
				break;
			}
			String ele=splits[k];
    		try{
    			hidden_dropout[k]=Double.parseDouble(ele);
    			if (hidden_dropout[k]<0 || hidden_dropout[k] >=1){
    				throw new IllegalStateException(" hidden neurons dropouts' in a layer cannot be less than  zero or higher than /equal to 1" );		
    			}
    		}catch (Exception e){
    			throw new IllegalStateException(" hidden needs to have comma separated double values .Here it receied: " + ele  );	
    		}
    	}	    
		
		if (this.mini_batch_size<=0){
			this.mini_batch_size=1;
		}	
		if (this.rho<=0 || this.rho>=1.0 ){
			this.rho=0.9;
		}	
		if (this.l2<0){
			this.l2=0;
		}
		if (this.sample_rate<=0 || this.sample_rate>=1.0){
			this.sample_rate=1;
		}
		if (this.epochs<1){
			this.epochs=1;
		}
		if (this.epsilon<=0){
			this.epsilon=1e-4;
		}	
		if (this.l1<0){
			this.l1=0.0;
		}			
		if (this.input_dropout_ratio<0){
			this.input_dropout_ratio=0.1;
		}
		if (this.max_w2<0){
			this.max_w2=1.;
		}	
		if (this.rate<0){
			this.rate=0.1;
		}	
		if (this.rate_annealing<0){
			this.rate_annealing=1e-6;
		}	
		if (this.rate_decay<0){
			this.rate_decay=0.01;
		}			
		if (this.momentum_start<0 || this.momentum_start>=1.0){
			this.momentum_start=0.95;
		}		
		if (this.momentum_ramp<0 || this.momentum_ramp>=1.0){
			this.momentum_ramp=0.95;
		}	
		if (this.momentum_stable<0 || this.momentum_stable>=1.0){
			this.momentum_stable=0.95;
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
		
		this.Objective="bernouli";
		if (this.n_classes>2){
			this.Objective="multiclass";
		}

		
		columndimension=data.GetColumnDimension();
		//generate config file
		
		//generate dataset

		output out = new output();
		out.verbose=false;
		out.printsmatrix(data, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		data=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		
		//make subprocess
		 create_h2o_cluster(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" , this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
		 
       // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
       // tries to delete a non-existing file
       f.delete();	
		
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
		return "H2ODeepLearningClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: H2ODeepLearningClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		
	    System.out.println("objective: " + this.Objective);
	    System.out.println("loss: " + this.loss  );
	    System.out.println("weight_init: " + this.weight_init);
	    System.out.println("activation: " + this.activation  );
	    
	    System.out.println("l1: " + this.l1  );
	    System.out.println("rho: " + this.rho);
	    System.out.println("epsilon: " + this.epsilon );
	    System.out.println("sample_rate: " + this.sample_rate );
	    System.out.println("l2: " + this.l2 );
	    System.out.println("balance_classes: " + this.balance_classes );
	    System.out.println("input_dropout_ratio: " +  this.input_dropout_ratio );
	    System.out.println("nthread: " + this.threads );
	    System.out.println("epochs: " +  this.epochs );
	    System.out.println("mini_batch_size: " + this.mini_batch_size );
	    
	    System.out.println("hidden: " + this.hidden  );
	    System.out.println("droupouts: " + this.droupouts  );
	    System.out.println("max_w2: " + this.max_w2  );
	    System.out.println("rate: " + this.rate);
	    System.out.println("rate_annealing: " + this.rate_annealing );
	    System.out.println("rate_decay: " + this.rate_decay );
	    System.out.println("momentum_start: " + this.momentum_start );
	    System.out.println("momentum_ramp: " + this.momentum_ramp );
	    System.out.println("momentum_stable: " +  this.momentum_stable );
	    System.out.println("shuffle: " + this.shuffle );
	    
	    System.out.println("standardize: " +  this.standardize );
	    System.out.println("fast_mode: " + this.fast_mode );
	    
	    System.out.println("adaptive_rate: " +  this.adaptive_rate );
	    System.out.println("nesterov_accelerated_gradient: " + this.nesterov_accelerated_gradient );	    
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
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".mod" );
        f.delete();  

        Objective = "";
        loss="CrossEntropy";
        l1 = 0;
        rho=0.95;
        epsilon=1e-8;
        sample_rate =0.95;
        l2 =0;
        balance_classes =false;
        input_dropout_ratio = 0.1;
        threads=1;
        epochs = 10;
        mini_batch_size=1;
        
    	max_w2=1.0;	
    	rate=0.1;	
    	rate_annealing=1e-6 ;	
    	rate_decay=0.01;	
    	momentum_start=0.95;		
    	momentum_ramp=0.92;		
    	momentum_stable=0.9;			
    	
    	shuffle=true;
    	standardize=false;	
    	fast_mode=false;	
    	adaptive_rate=true;
    	nesterov_accelerated_gradient=true;
    	weight_init="UniformAdaptive"; //UniformAdaptive, Uniform or Normal
    	activation="Rectifier"; //Has to be Rectifier, Tanh, Maxout or ExpRectifier
    	loss="CrossEntropy";	     
        
        
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
				if (metric.equals("l2")) {this.l2=Double.parseDouble(value);}
				else if (metric.equals("rho")) {this.rho=Double.parseDouble(value);}
				else if (metric.equals("sample_rate")) {this.sample_rate=Double.parseDouble(value);}	
				else if (metric.equals("epochs")) {this.epochs=Integer.parseInt(value);}
				else if (metric.equals("l1")) {this.l1=Double.parseDouble(value);}					
				else if (metric.equals("mini_batch_size")) {this.mini_batch_size=Integer.parseInt(value);}				
				else if (metric.equals("input_dropout_ratio")) {this.input_dropout_ratio=Double.parseDouble(value);}		
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("balance_classes")) {this.balance_classes=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("epsilon")) {this.epsilon=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				
				else if (metric.equals("max_w2")) {this.max_w2=Double.parseDouble(value);}	
				else if (metric.equals("rate")) {this.rate=Double.parseDouble(value);}	
				else if (metric.equals("rate_annealing")) {this.rate_annealing=Double.parseDouble(value);}	
				else if (metric.equals("rate_decay")) {this.rate_decay=Double.parseDouble(value);}	
				else if (metric.equals("momentum_start")) {this.momentum_start=Double.parseDouble(value);}	
				else if (metric.equals("momentum_ramp")) {this.momentum_ramp=Double.parseDouble(value);}	
				else if (metric.equals("momentum_stable")) {this.momentum_stable=Double.parseDouble(value);}					
				else if (metric.equals("shuffle")) {this.shuffle=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("standardize")) {this.standardize=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("fast_mode")) {this.fast_mode=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("adaptive_rate")) {this.adaptive_rate=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.equals("nesterov_accelerated_gradient")) {this.nesterov_accelerated_gradient=(value.toLowerCase().equals("true")?true:false)   ;}						
				else if (metric.equals("hidden")) {this.hidden=value;}				
				else if (metric.equals("droupouts")) {this.droupouts=value;}			
				else if (metric.equals("weight_init")) {this.weight_init=value;}				
				else if (metric.equals("activation")) {this.activation=value;}
				
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
			  

