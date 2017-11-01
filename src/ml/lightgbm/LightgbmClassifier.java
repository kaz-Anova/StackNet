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

package ml.lightgbm;
import io.output;

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

import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import utilis.detectos;
import utilis.map.intint.StringIntMap4a;
import exceptions.DimensionMismatchException;
import exceptions.LessThanMinimum;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;

/**
*<p>Wraps another kaggle's favourite  <a href="https://github.com/Microsoft/LightGBM">LightGBM</a>.
*This particular instance is allowing only classification results. LightGBM models are being trained via a subprocess based on the operating systems
*executing the class. <b>It is expected that files will be created and their size will vary based on the volumne of the training data.</b></p>
*
*
*<p>Information about the tunable parameters can be found <a href="https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md">here</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all LightGBM features and the user is advised to use it directly from the source.
*Also the version may not be the final and it is not certain whether it will be updated in the future as it required manual work to find all libraries and
*files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website, hence it is advised to <a href="https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md">use LightGBM's online parameter thread in
*github</a> for more information about them. </p></em> 
 */

public class LightgbmClassifier implements estimator,classifier {

	/**
	 * Number of trees to build
	 */
	public int num_iterations=100;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * maximum number of leaves
	 */
	public int num_leaves=0;
	/**
	 * to safeguard optimisation
	 */
	public double poission_max_delta_step=0;
	/**
	 * maximum depth of the tree
	 */
	public int max_depth=4;
	/**
	 * Minimum sum hessian in one leaf
	 */
	public double min_sum_hessian_in_leaf=1e-3;
	/**
	 * Minimum  number of data in a leaf
	 */
	public int min_data_in_leaf=20;		
	/**
	 * Proportions of columns (features) to consider within a tree
	 */
	public double feature_fraction=1.0;
	/**
	 * Minimum  gain to split a node
	 */
	public double min_gain_to_split=1.0;
	/**
	 * weight on each estimator . Smaller values prevent overfitting. 
	 */
	public double learning_rate=0.1;
	/**
	 * Proportions of rows consider
	 */
	public double bagging_fraction=0.95;	
	/**
	 *  Every how many iters it will perform bagging 
	 */
	public int bagging_freq=1;	
	/**
	 * scale weight for binary class
	 */
	public double scale_pos_weight=1.0;
	/**
	 * L1 regularization on the weights
	 */
	public double lambda_l1=0;	
	/**
	 * L2 regularization on the weights
	 */
	public double lambda_l2=0;		
	/**
	 * The objective has to be multiclass or binary.
	 */
	private String Objective="multiclass";	
	/**
	 * comma separated features to be treated as categorical
	 */
	public String categorical_feature="";
	/**
	 * Type of boosting. could be one of gbdt, dart or goss
	 */
	public String boosting="gbdt";
	/**
	 * Specify whether to use xgboost dart mode or not
	 */
	public boolean xgboost_dart_mode=false;
	/**
	 * Specify whether to use uniform dropout
	 */
	public boolean uniform_drop=false;
	/**
	 * dropout rate in dart boosting
	 */
	public double drop_rate=0.1;	
	/**
	 * probability of skipping drop
	 */
	public double skip_drop=0.5;	
	/**
	 * max number of dropped trees on one iteration
	 */
	public int max_drop=50;
	/**
	 * used in boosting goss, the retain ratio of large gradient data
	 */
	public double top_rate=0.1;	
	/**
	 * only used in boosting goss, the retain ratio of small gradient data
	 */
	public double other_rate=0.1;	
	/**
	 * max number of bin that feature values will bucket in. Small bin may reduce training accuracy but may increase general power (deal with over-fit).
	 */
	public int max_bin=255;
	/**
	 * in number of data inside one bin, use this to avoid one-data-one-bin (may over-fitting).
	 */
	public int min_data_in_bin=5;
	/**
	 * by default, LightGBM will map data file to memory and load features from memory. This will provide faster data loading speed. But it may out of memory when the data file is very big.
		Set this to true if data file is too big to fit in memory.
	 */
	public boolean two_round=false;
	/**
	 * scale the copy the dataset
	 */
	public boolean is_unbalance=false;
	/**
	 *Sample number of rows to create histograms. More normally gives more precision , but it will take more time.
	 */
	public int bin_construct_sample_cnt=1000000;
	/**
	 * parameter for sigmoid function. Will be used in binary classification
	 */
	public double sigmoid=0.1;

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
	 * @param filename : the conifiguration file name for required to run xgboost from the command line
	 * @param datset : the dataset to be used
	 * @param model : model dump name
	 */
   private void create_config_file(String filename , String datset, String model){

		try{  // Catch errors in I/O if necessary.
			  // Open a file to write to.
				String saveFile = filename;
				
				FileWriter writer = new FileWriter(saveFile);
				writer.append("boosting=" + this.boosting + "\n");
			    writer.append("objective=" + this.Objective+ "\n");
			    if (this.Objective.equals("multiclass")){
			    	writer.append("num_class=" + this.n_classes  + "\n"); 
			    }
			    writer.append("learning_rate=" + this.learning_rate  + "\n");
			    writer.append("min_sum_hessian_in_leaf=" +this.min_sum_hessian_in_leaf + "\n");
			    writer.append("min_data_in_leaf=" + this.min_data_in_leaf + "\n");
			    writer.append("feature_fraction=" + this.feature_fraction+ "\n");
			    writer.append("min_gain_to_split=" + this.min_gain_to_split + "\n");
			    writer.append("bagging_fraction=" + this.bagging_fraction + "\n");
			    writer.append("poission_max_delta_step=" + this.poission_max_delta_step  + "\n");
			    writer.append("lambda_l1=" + this.lambda_l1 + "\n");
			    writer.append("lambda_l2=" + this.lambda_l2 + "\n");
			    writer.append("scale_pos_weight=" + this.scale_pos_weight + "\n");
			    writer.append("max_depth=" +  this.max_depth + "\n");
			    writer.append("num_threads=" + this.threads + "\n");
			    writer.append("num_iterations=" +  this.num_iterations + "\n");
			    writer.append("feature_fraction_seed=" +  this.seed + "\n");	
			    writer.append("bagging_seed=" +  this.seed + "\n");				    
			    writer.append("drop_seed=" +  this.seed + "\n");				    
			    writer.append("data_random_seed=" +  this.seed + "\n");				    
			    writer.append("num_leaves=" + this.num_leaves + "\n");
			    writer.append("bagging_freq=" + this.bagging_freq + "\n");	
			    writer.append("xgboost_dart_mode=" + this.xgboost_dart_mode + "\n");				    
			    writer.append("drop_rate=" + this.drop_rate + "\n");
			    writer.append("skip_drop=" + this.skip_drop + "\n");
			    writer.append("max_drop=" + this.max_drop + "\n");				    
			    writer.append("top_rate=" + this.top_rate + "\n");
			    writer.append("other_rate=" + this.other_rate + "\n");		
			    writer.append("sigmoid=" + this.sigmoid + "\n");
			    writer.append("max_bin=" + this.max_bin + "\n");	
			    writer.append("min_data_in_bin=" + this.min_data_in_bin + "\n");				    			    
			    writer.append("uniform_drop=" + this.uniform_drop + "\n");				    
			    writer.append("two_round=" + this.two_round + "\n");	
			    writer.append("is_unbalance=" + this.is_unbalance + "\n");
			    writer.append("categorical_feature=" + this.categorical_feature + "\n");
			    writer.append("bin_construct_sample_cnt=" + this.bin_construct_sample_cnt + "\n");		
			    writer.append("is_sparse=true" + "\n");				    
			    
			    if (this.verbose){
			    	writer.append("verbosity=1" +  "\n");
			    }else {
			    	writer.append("verbosity=0" +  "\n");			    	
			    }
			    //file details
			    writer.append(datset+ "\n");
			    writer.append(model+ "\n");			    
				writer.close();

	    	} catch (Exception e) {
	    		throw new IllegalStateException(" failed to write the config file at: " + filename);
	    	}   
   }
   
   
	/**
	 * 
	 * @param filename : the conifiguration file name for required to run xgboost from the command line
	 * @param datset : the dataset to be used
	 * @param model : model dump name
	 * @param predictionfile : where the predictions will eb saved
	 */
  private void create_config_file_pred(String filename , String datset, String model, String predictionfile){

		try{  // Catch errors in I/O if necessary.
			  // Open a file to write to.
				String saveFile = filename;
				
				FileWriter writer = new FileWriter(saveFile);
			    writer.append("objective=" + this.Objective+ "\n");
			    if (this.Objective.equals("multiclass")){
			    	writer.append("num_class=" + this.n_classes  + "\n"); 
			    }
			    writer.append("is_sparse=true" + "\n");				  
			    writer.append("num_threads=" + this.threads + "\n");
			    if (this.verbose){
			    	writer.append("verbosity=1" +  "\n");
			    }else {
			    	writer.append("verbosity=0" +  "\n");			    	
			    }
			    //file details
			    writer.append(datset+ "\n");
			    writer.append( model+ "\n");
			    writer.append( "output_result=" + predictionfile + "\n");
		    
				writer.close();

	    	} catch (Exception e) {
	    		throw new IllegalStateException(" failed to write the config file at: " + filename);
	    	}   
  }
  
   /**
    * 
    * @param confingname : full path and name of the config file
    * @param istrain : if this is a train task or a prediction task
    */
   private void create_light_suprocess(String confingname, boolean istrain ) {
	   
	   // check if file exists
	   if (new File(confingname).exists()==false){
		   throw new IllegalStateException("Config file does not exist at: " + confingname);
	   }
	   // create the subprocess
		try {
			 String operational_system=detectos.getOS();
			 if (!operational_system.equals("win") && !operational_system.equals("linux")&& !operational_system.equals("mac")){
				 throw new IllegalStateException(" The operational system is not identified as win, linux or mac which is required to run xgboost" ); 
			 }
			 String xgboost_path="lib" + File.separator + operational_system + File.separator + "lightgbm" + File.separator + "lightgbm";
			 List<String> list = new ArrayList<String>();
			 list.add(xgboost_path);			 
			 list.add("config=" + confingname);
			 if (istrain){
				 list.add("task=train");	 
			 }
			 else {
				 list.add("task=prediction");	 
			 }			 

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
			throw new IllegalStateException(" failed to create LIGHTgbm subprocess with config name " + confingname);
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
	 * Default constructor with no data
	 */
	public LightgbmClassifier(){
	
	}	
	/**
	 * Default constructor with double data
	 */
	public LightgbmClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor with fsmatrix data
	 */
	public LightgbmClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor with smatrix data
	 */
	public LightgbmClassifier(smatrix data){
		
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
		
		
		//generate dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"input_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
		 }else {
			 int cols [] = new int [this.n_classes];
			 for (int j=0 ; j <this.n_classes; j++){
				 cols[j]=j;
			 }
			 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", cols, "0.0", false, false);
			 if (temp.length!=predictions.length){
				 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
			 }
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]=temp[i][j];
				 }
			 } 
		        temp=null;
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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
		

		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		
		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"input_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);
		
		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
		 }else {
			 int cols [] = new int [this.n_classes];
			 for (int j=0 ; j <this.n_classes; j++){
				 cols[j]=j;
			 }
			 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", cols, "0.0", false, false);
			 if (temp.length!=predictions.length){
				 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
			 }
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]=temp[i][j];
				 }
			 } 
		        temp=null;
		 }
		 
		 
		// create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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
		
		output out = new output();
		out.verbose=false;
		out.printsmatrix(data,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"input_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);
		
		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
		 }else {
			 int cols [] = new int [this.n_classes];
			 for (int j=0 ; j <this.n_classes; j++){
				 cols[j]=j;
			 }
			 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", cols, "0.0", false, false);
			 if (temp.length!=predictions.length){
				 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
			 }
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]=temp[i][j];
				 }
			 } 
		        temp=null;
		 }
		 
		 
		// create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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
		out.printsmatrix(X,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"input_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
		 }else {
			 int cols [] = new int [this.n_classes];
			 for (int j=0 ; j <this.n_classes; j++){
				 cols[j]=j;
			 }
			 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", cols, "0.0", false, false);
			 if (temp.length!=predictions.length){
				 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
			 }
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 prediction_probas[i][j]=temp[i][j];
				 }
			 } 
		        temp=null;
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		double temp[]=null;       
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

		double predictions[]= new double [data.GetRowDimension()];
		double prediction_probas[][]= new double [data.GetRowDimension()][n_classes];

		output out = new output();
		out.verbose=false;
		out.printsmatrix(data,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"input_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
		 }else {
			 int cols [] = new int [this.n_classes];
			 for (int j=0 ; j <this.n_classes; j++){
				 cols[j]=j;
			 }
			 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", cols, "0.0", false, false);
			 if (temp.length!=predictions.length){
				 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
			 }
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 prediction_probas[i][j]=temp[i][j];
				 }
			 } 
		        temp=null;
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		double temp []=null;       
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
		
		double predictions[]= new double [data.length];
		double prediction_probas[][]= new double [data.length][n_classes];

		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		
		create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"input_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
		 }else {
			 int cols [] = new int [this.n_classes];
			 for (int j=0 ; j <this.n_classes; j++){
				 cols[j]=j;
			 }
			 double temp [][]=io.readcsv.putfiletoarraydoublestatic(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", cols, "0.0", false, false);
			 if (temp.length!=predictions.length){
				 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of correct size" );
			 }
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 prediction_probas[i][j]=temp[i][j];
				 }
			 } 
		        temp=null;
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		double temp []=null;       
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
		if ( !boosting.equals("gbdt")  && !boosting.equals("dart") & !boosting.equals("goss")){
			throw new IllegalStateException(" booster has to be between 'gbdt', 'dart' or 'goss'" );	
		}
		
	    if (!categorical_feature.equals("")){
	    	String splits [] = categorical_feature.split(",");
	    	for (String ele: splits){
	    		try{
	    			Integer.parseInt(ele);
	    		}catch (Exception e){
	    			throw new IllegalStateException(" categorical_feature needs to have comma separated integer indices .Here it receied: " + ele  );	
	    		}
	    	}
	    }
		if (this.min_data_in_bin<0){
			this.min_data_in_bin=5;
		}		    
		if (this.max_bin<0){
			this.max_bin=255;
		}	    
		if (this.top_rate<0){
			this.top_rate=0.1;
		}	    
		if (this.other_rate<0){
			this.other_rate=0.1;
		}	    
		if (this.max_drop<0){
			this.max_drop=50;
		}	    
		if (this.skip_drop<0){
			this.skip_drop=0.5;
		}
		if (this.drop_rate<0){
			this.drop_rate=0.1;
		}
		if (this.sigmoid<0){
			this.sigmoid=1.;
		}	
		if (this.bagging_freq<0){
			this.bagging_freq=0;
		}
		
		if (this.bin_construct_sample_cnt<0){
			this.bin_construct_sample_cnt=100000;
		}		
		if (this.num_leaves<0){
			this.num_leaves=10;
		}
		if (this.poission_max_delta_step<0){
			this.poission_max_delta_step=0.7;
		}	
		
		if (this.min_data_in_leaf<0){
			this.min_data_in_leaf=20;
		}	
		if (this.feature_fraction<=0){
			this.feature_fraction=1.0;
		}	
		if (this.lambda_l1<=0){
			this.lambda_l1=0.0;
		}
		if (this.lambda_l2<0){
			this.lambda_l2=0;
		}
		if (this.bagging_fraction<=0){
			this.bagging_fraction=1.0;
		}
		if (this.num_iterations<1){
			this.num_iterations=1;
		}
		if (this.min_gain_to_split<0){
			this.min_gain_to_split=0.;
		}	
		if (this.learning_rate<=0){
			this.learning_rate=0.01;
		}		
		if (this.min_sum_hessian_in_leaf<0){
			this.min_sum_hessian_in_leaf=1e-3;
		}		
		if (this.scale_pos_weight<=0){
			this.scale_pos_weight=1.;
		}
		if (this.max_depth<=0){
			this.max_depth=1;
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
		
		this.Objective="binary";
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
		
		create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				"output_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
        // tries to delete a non-existing file
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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

		if ( !boosting.equals("gbdt")  && !boosting.equals("dart") & !boosting.equals("goss")){
			throw new IllegalStateException(" booster has to be between 'gbdt', 'dart' or 'goss'" );	
		}
		
	    if (!categorical_feature.equals("")){
	    	String splits [] = categorical_feature.split(",");
	    	for (String ele: splits){
	    		try{
	    			Integer.parseInt(ele);
	    		}catch (Exception e){
	    			throw new IllegalStateException(" categorical_feature needs to have comma separated integer indices .Here it receied: " + ele  );	
	    		}
	    	}
	    }
		if (this.min_data_in_bin<0){
			this.min_data_in_bin=5;
		}		    
		if (this.max_bin<0){
			this.max_bin=255;
		}	    
		if (this.top_rate<0){
			this.top_rate=0.1;
		}	    
		if (this.other_rate<0){
			this.other_rate=0.1;
		}	    
		if (this.max_drop<0){
			this.max_drop=50;
		}	    
		if (this.skip_drop<0){
			this.skip_drop=0.5;
		}
		if (this.drop_rate<0){
			this.drop_rate=0.1;
		}
		if (this.bagging_freq<0){
			this.bagging_freq=0;
		}
		if (this.sigmoid<0){
			this.sigmoid=1.;
		}	
		if (this.bin_construct_sample_cnt<0){
			this.bin_construct_sample_cnt=100000;
		}		
		if (this.num_leaves<0){
			this.num_leaves=10;
		}
		if (this.poission_max_delta_step<0){
			this.poission_max_delta_step=0.7;
		}	
		
		if (this.min_data_in_leaf<0){
			this.min_data_in_leaf=20;
		}	
		if (this.feature_fraction<=0){
			this.feature_fraction=1.0;
		}	
		if (this.lambda_l1<=0){
			this.lambda_l1=0.0;
		}
		if (this.lambda_l2<0){
			this.lambda_l2=0;
		}
		if (this.bagging_fraction<=0){
			this.bagging_fraction=1.0;
		}
		if (this.num_iterations<1){
			this.num_iterations=1;
		}
		if (this.min_gain_to_split<0){
			this.min_gain_to_split=0.;
		}	
		if (this.learning_rate<=0){
			this.learning_rate=0.01;
		}		
		if (this.min_sum_hessian_in_leaf<0){
			this.min_sum_hessian_in_leaf=1e-3;
		}		
		if (this.scale_pos_weight<=0){
			this.scale_pos_weight=1.;
		}
		if (this.max_depth<=0){
			this.max_depth=1;
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
		
		this.Objective="binary";
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
		
		create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				"output_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);

	        // create new file
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();
			f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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

		if ( !boosting.equals("gbdt")  && !boosting.equals("dart") & !boosting.equals("goss")){
			throw new IllegalStateException(" booster has to be between 'gbdt', 'dart' or 'goss'" );	
		}
		
	    if (!categorical_feature.equals("")){
	    	String splits [] = categorical_feature.split(",");
	    	for (String ele: splits){
	    		try{
	    			Integer.parseInt(ele);
	    		}catch (Exception e){
	    			throw new IllegalStateException(" categorical_feature needs to have comma separated integer indices .Here it receied: " + ele  );	
	    		}
	    	}
	    }
		if (this.min_data_in_bin<0){
			this.min_data_in_bin=5;
		}		    
		if (this.max_bin<0){
			this.max_bin=255;
		}	    
		if (this.top_rate<0){
			this.top_rate=0.1;
		}	    
		if (this.other_rate<0){
			this.other_rate=0.1;
		}	    
		if (this.max_drop<0){
			this.max_drop=50;
		}
		if (this.sigmoid<0){
			this.sigmoid=1.;
		}	
		if (this.skip_drop<0){
			this.skip_drop=0.5;
		}
		if (this.drop_rate<0){
			this.drop_rate=0.1;
		}
		if (this.bagging_freq<0){
			this.bagging_freq=0;
		}
		if (this.bin_construct_sample_cnt<0){
			this.bin_construct_sample_cnt=100000;
		}		
		if (this.num_leaves<0){
			this.num_leaves=10;
		}
		if (this.poission_max_delta_step<0){
			this.poission_max_delta_step=0.7;
		}	
		
		if (this.min_data_in_leaf<0){
			this.min_data_in_leaf=20;
		}	
		if (this.feature_fraction<=0){
			this.feature_fraction=1.0;
		}	
		if (this.lambda_l1<=0){
			this.lambda_l1=0.0;
		}
		if (this.lambda_l2<0){
			this.lambda_l2=0;
		}
		if (this.bagging_fraction<=0){
			this.bagging_fraction=1.0;
		}
		if (this.num_iterations<1){
			this.num_iterations=1;
		}
		if (this.min_gain_to_split<0){
			this.min_gain_to_split=0.;
		}	
		if (this.learning_rate<=0){
			this.learning_rate=0.01;
		}		
		if (this.min_sum_hessian_in_leaf<0){
			this.min_sum_hessian_in_leaf=1e-3;
		}		
		if (this.scale_pos_weight<=0){
			this.scale_pos_weight=1.;
		}
		if (this.max_depth<=0){
			this.max_depth=1;
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
		
		this.Objective="binary";
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
		
		create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
				"data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				"output_model=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");

		//make subprocess
		 create_light_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);
		sdataset=null;
		System.gc();
		
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
        // tries to delete a non-existing file
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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
		return "LightgbmClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: LightgbmClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		
		System.out.println("boosting: " + this.boosting );
	    System.out.println("objective: " + this.Objective);
	    if (this.Objective.equals("multiclass")){
	    	System.out.println("Objective: multiclass" ); 
	    }
	    else {
	    	System.out.println("Objective: binary" ); 
	    }
	    

	    System.out.println("learning_rate=" + this.learning_rate  );
	    System.out.println("min_sum_hessian_in_leaf=" +this.min_sum_hessian_in_leaf );
	    System.out.println("min_data_in_leaf=" + this.min_data_in_leaf );
	    System.out.println("feature_fraction=" + this.feature_fraction);
	    System.out.println("min_gain_to_split=" + this.min_gain_to_split);
	    System.out.println("bagging_fraction=" + this.bagging_fraction );
	    System.out.println("poission_max_delta_step=" + this.poission_max_delta_step );
	    System.out.println("lambda_l1=" + this.lambda_l1 );
	    System.out.println("lambda_l2=" + this.lambda_l2 );
	    System.out.println("scale_pos_weight=" + this.scale_pos_weight );
	    System.out.println("max_depth=" +  this.max_depth );
	    System.out.println("threads=" + this.threads );
	    System.out.println("num_iterations=" +  this.num_iterations );
	    System.out.println("seed=" +  this.seed );				    
	    System.out.println("bagging_freq=" + this.bagging_freq );	
	    System.out.println("xgboost_dart_mode=" + this.xgboost_dart_mode );				    
	    System.out.println("drop_rate=" + this.drop_rate );
	    System.out.println("skip_drop=" + this.skip_drop );
	    System.out.println("max_drop=" + this.max_drop );				    
	    System.out.println("top_rate=" + this.top_rate );
	    System.out.println("other_rate=" + this.other_rate );					    
	    System.out.println("max_bin=" + this.max_bin );	
	    System.out.println("sigmoid=" + this.sigmoid );		    
	    System.out.println("min_data_in_bin=" + this.max_bin );				    			    
	    System.out.println("uniform_drop=" + this.uniform_drop );				    
	    System.out.println("two_round=" + this.two_round );	
	    System.out.println("is_unbalance=" + this.is_unbalance );
	    System.out.println("categorical_feature=" + this.categorical_feature );
	    System.out.println("bin_construct_sample_cnt=" + this.bin_construct_sample_cnt );		

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
        sigmoid=1.0;
        boosting = "gbdt";
        Objective = "";
        num_iterations=100;
		num_leaves=10;
		poission_max_delta_step=0.7;
		max_depth=5;
		min_sum_hessian_in_leaf=1e-3;
		min_data_in_leaf=20;
		feature_fraction=1.0;
		min_gain_to_split=0;
		learning_rate=0.1;
		bagging_fraction=0.8;
		bagging_freq=1;
		scale_pos_weight=1.0;
		lambda_l1=0;
		lambda_l2=0;
		categorical_feature="";
		xgboost_dart_mode=false;
		uniform_drop=false;
		drop_rate=0.1;
		skip_drop=0.5;
		max_drop=50;
		top_rate=0.1;
		other_rate=0.1;
		max_bin=255;
		min_data_in_bin=5;
		two_round=false;
		bin_construct_sample_cnt=1000000;
		is_unbalance=true;
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
				if (metric.equals("lambda_l1")) {this.lambda_l1=Double.parseDouble(value);}
				else if (metric.equals("lambda_l2")) {this.lambda_l2=Double.parseDouble(value);}				
				else if (metric.equals("feature_fraction")) {this.feature_fraction=Double.parseDouble(value);}
				else if (metric.equals("bagging_fraction")) {this.bagging_fraction=Double.parseDouble(value);}	
				else if (metric.equals("num_iterations")) {this.num_iterations=Integer.parseInt(value);}
				else if (metric.equals("scale_pos_weight")) {this.scale_pos_weight=Double.parseDouble(value);}	
				else if (metric.equals("learning_rate")) {this.learning_rate=Double.parseDouble(value);}
				else if (metric.equals("sigmoid")) {this.sigmoid=Double.parseDouble(value);}				
				else if (metric.equals("num_leaves")) {this.num_leaves=Integer.parseInt(value);}				
				else if (metric.equals("max_depth")) {this.max_depth=Integer.parseInt(value);}
				else if (metric.equals("boosting")) {this.boosting=value;}				
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("min_gain_to_split")) {this.min_gain_to_split=Double.parseDouble(value);}	
				else if (metric.equals("min_data_in_leaf")) {this.min_data_in_leaf=Integer.parseInt(value);}
				else if (metric.equals("scale_pos_weight")) {this.scale_pos_weight=Double.parseDouble(value);}						
				else if (metric.equals("min_sum_hessian_in_leaf")) {this.min_sum_hessian_in_leaf=Double.parseDouble(value);}
				else if (metric.equals("drop_rate")) {this.drop_rate=Double.parseDouble(value);}				
				else if (metric.equals("skip_drop")) {this.skip_drop=Double.parseDouble(value);}				
				else if (metric.equals("top_rate")) {this.top_rate=Double.parseDouble(value);}				
				else if (metric.equals("other_rate")) {this.other_rate=Double.parseDouble(value);}	
				else if (metric.equals("categorical_feature")) {this.categorical_feature=value;}	
				else if (metric.equals("uniform_drop")) {this.uniform_drop=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.equals("xgboost_dart_mode")) {this.xgboost_dart_mode=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.equals("is_unbalance")) {this.is_unbalance=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.equals("two_round")) {this.two_round=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.equals("bagging_freq")) {this.bagging_freq=Integer.parseInt(value);}
				else if (metric.equals("max_drop")) {this.max_drop=Integer.parseInt(value);}				
				else if (metric.equals("max_bin")) {this.max_bin=Integer.parseInt(value);}				
				else if (metric.equals("min_data_in_bin")) {this.min_data_in_bin=Integer.parseInt(value);}	
				else if (metric.equals("bin_construct_sample_cnt")) {this.bin_construct_sample_cnt=Integer.parseInt(value);}					
				else if (metric.equals("poission_max_delta_step")) {this.poission_max_delta_step=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
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
			  

