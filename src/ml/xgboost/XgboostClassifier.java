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

package ml.xgboost;
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
*<p>Wraps kaggle's favourite (and StackNet's creator favourite <a href="https://github.com/dmlc/xgboost/">Xgboost</a>).
*This particular instance is allowing only classification results. Xgboost models are being trained via a subprocess based on the operating systems
*executing the class. <b>It is expected that files will be created and their size will vary based on the volumne of the training data.</b></p>
*
*<p>Reference : Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In 22nd SIGKDD Conference on Knowledge 
*Discovery and Data Mining, 2016<p>
*
*<p>Information about the tunable parameters can be found <a href="https://github.com/dmlc/xgboost/blob/master/doc/parameter.md">here</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all Xgboost's features and the user is advised to use it directly from the source.
*Also the version included is 6.0 and it is not certain whether it will be updated in the future as it required manual work to find all libraries and
*files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website, hence it is advised to <a href="https://github.com/dmlc/xgboost/blob/master/doc/parameter.md">use xgboost's online parameter thread in
*github</a> for more information about them. </p></em> 
 */

public class XgboostClassifier implements estimator,classifier {

	/**
	 * Number of trees to build
	 */
	public int num_round=100;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * maximum number of leaves
	 */
	public int max_leaves=0;
	/**
	 * offset for divisions
	 */
	public double max_delta_step=0;
	/**
	 * maximum depth of the tree
	 */
	public int max_depth=4;
	/**
	 * Minimum gain to allow for a node to split
	 */
	public double gamma=1.0;
	/**
	 * Minimum weighted sum to keep a splitted node
	 */
	public double min_child_weight=1.0;		
	/**
	 * Proportions of columns (features) to consider within a tree
	 */
	public double colsample_bytree=1.0;
	/**
	 * Proportions of columns (features) to consider at each level of a given tree
	 */
	public double colsample_bylevel=1.0;
	/**
	 * weight on each estimator . Smaller values prevent overfitting. 
	 */
	public double eta=0.1;
	/**
	 * Proportions of rows consider
	 */
	public double subsample=0.95;	
	/**
	 * scale weight for binary class
	 */
	public double scale_pos_weight=1.0;
	/**
	 * Lambda regularization on the weights
	 */
	public double lambda=0;	
	/**
	 * alpha regularization on the weights
	 */
	public double alpha=0;		
	/**
	 * The objective has to be logloss.
	 */
	private String Objective="logloss";	
	/**
	 * Should be either gbtree or gblinear
	 */
	public String booster="gbtree";	
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
				writer.append("booster=" + this.booster + "\n");
			    writer.append("objective=" + this.Objective+ "\n");
			    if (this.Objective.equals("multi:softprob")){
			    	writer.append("num_class=" + this.n_classes  + "\n"); 
			    }
			    writer.append("eta=" + this.eta  + "\n");
			    writer.append("gamma=" +this.gamma + "\n");
			    writer.append("min_child_weight=" + this.min_child_weight + "\n");
			    writer.append("colsample_bytree=" + this.colsample_bytree+ "\n");
			    writer.append("colsample_bylevel=" + this.colsample_bylevel + "\n");
			    writer.append("subsample=" + this.subsample + "\n");
			    writer.append("max_delta_step=" + this.max_delta_step  + "\n");
			    writer.append("lambda=" + this.lambda + "\n");
			    writer.append("alpha=" + this.alpha + "\n");
			    writer.append("scale_pos_weight=" + this.scale_pos_weight + "\n");
			    writer.append("max_depth=" +  this.max_depth + "\n");
			    writer.append("nthread=" + this.threads + "\n");
			    writer.append("num_round=" +  this.num_round + "\n");
			    writer.append("seed=" +  this.seed + "\n");				    
			    writer.append("max_leaves=" + this.max_leaves + "\n");
			    writer.append("save_period=0" + "\n");
			    if (this.verbose){
			    	writer.append("silent=0" +  "\n");
			    }else {
			    	writer.append("silent=1" +  "\n");			    	
			    }
			    //file details
			    writer.append(datset+ "\n");
			    writer.append( model+ "\n");			    
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
			    if (this.Objective.equals("multi:softprob")){
			    	writer.append("num_class=" + this.n_classes  + "\n"); 
			    }
			    writer.append("nthread=" + this.threads + "\n");
			    writer.append("max_leaves=" + this.max_leaves + "\n");
			    writer.append("save_period=0" + "\n");
			    if (this.verbose){
			    	writer.append("silent=0" +  "\n");
			    }else {
			    	writer.append("silent=1" +  "\n");			    	
			    }
			    
			    //file details
			    writer.append(datset+ "\n");
			    writer.append( model+ "\n");
			    writer.append( "name_pred=" + predictionfile + "\n");
		    
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
   private void create_xg_suprocess(String confingname, boolean istrain ) {
	   
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
			 String xgboost_path="lib" + File.separator + operational_system + File.separator + "xg" + File.separator + "xgboost";
			 List<String> list = new ArrayList<String>();
			 list.add(xgboost_path);			 
			 list.add(confingname);
			 if (istrain){
				 list.add("task=train");	 
			 }
			 else {
				 list.add("task=pred");	 
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
			throw new IllegalStateException(" failed to create Xgboost subprocess with config name " + confingname);
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
	public XgboostClassifier(){
	
	}	
	/**
	 * Default constructor with double data
	 */
	public XgboostClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor with fsmatrix data
	 */
	public XgboostClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor with smatrix data
	 */
	public XgboostClassifier(smatrix data){
		
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
				"test:data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"model_in=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (this.n_classes==2){
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		 }else {
			 if (temp.length!=this.n_classes*predictions.length){
				 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of length n_class*length" );
			 }
			 int s=0;
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]=temp[s];
					 s++;
				 }
			 } 
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
        temp=null;
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
				"test:data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"model_in=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);
		
		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (this.n_classes==2){
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		 }else {
			 if (temp.length!=this.n_classes*predictions.length){
				 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of length n_class*length" );
			 }
			 int s=0;
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]=temp[s];
					 s++;
				 }
			 } 
		 }
		 
		 
		// create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
		f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		f.delete();     
		temp=null;
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
				"test:data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"model_in=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);
		
		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (this.n_classes==2){
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		 }else {
			 if (temp.length!=this.n_classes*predictions.length){
				 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of length n_class*length" );
			 }
			 int s=0;
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]=temp[s];
					 s++;
				 }
			 } 
		 }
		 
		 
		// create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
		f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		f.delete();     
		temp=null;		
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
				"test:data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"model_in=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (this.n_classes==2){
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		 }else {
			 if (temp.length!=this.n_classes*predictions.length){
				 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of length n_class*length" );
			 }
			 int s=0;
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 prediction_probas[i][j]=temp[s];
					 s++;
				 }
			 } 
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		temp=null;       
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
				"test:data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"model_in=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (this.n_classes==2){
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		 }else {
			 if (temp.length!=this.n_classes*predictions.length){
				 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of length n_class*length" );
			 }
			 int s=0;
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 prediction_probas[i][j]=temp[s];
					 s++;
				 }
			 } 
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		temp=null;       
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
				"test:data="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				"model_in=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
				);

		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);
		 
		 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", ",", 0, 0.0, false, false);
		 if (this.n_classes==2){
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		 }else {
			 if (temp.length!=this.n_classes*predictions.length){
				 throw new IllegalStateException(" There produced score in temporartu file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred " + " is not of length n_class*length" );
			 }
			 int s=0;
			 for (int i =0; i <predictions.length;i++ ){
				 for (int j =0; j <this.n_classes;j++ ){
					 prediction_probas[i][j]=temp[s];
					 s++;
				 }
			 } 
		 }
		 
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
        f.delete();     
		temp=null;       
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
		if ( !booster.equals("gbtree")  && !booster.equals("gblinear") ){
			throw new IllegalStateException(" booster has to be between 'gbtree' and gblinear' " );	
		}
		
		if (this.max_leaves<0){
			this.max_leaves=0;
		}
		if (this.max_delta_step<0){
			this.max_delta_step=0;
		}	
		if (this.min_child_weight<0){
			this.min_child_weight=0;
		}	
		if (this.colsample_bytree<=0){
			this.colsample_bytree=1.0;
		}	
		if (this.lambda<=0){
			this.lambda=0.0;
		}
		if (this.alpha<0){
			this.alpha=0;
		}
		if (this.subsample<=0){
			this.subsample=1;
		}
		if (this.num_round<1){
			this.num_round=1;
		}
		if (this.colsample_bylevel<=0){
			this.colsample_bylevel=1;
		}	
		if (this.eta<=0){
			this.eta=0.01;
		}		
		if (this.gamma<=0){
			this.gamma=0.01;
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
			
		}	else {
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
		
		this.Objective="binary:logistic";
		if (this.n_classes>2){
			this.Objective="multi:softprob";
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
				"model_out=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");

		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);
		 
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
        // tries to delete a non-existing file
        f.delete();
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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

		if ( !booster.equals("gbtree")  && !booster.equals("gblinear") ){
			throw new IllegalStateException(" booster has to be between 'gbtree' and gblinear' " );	
		}
		
		if (this.max_leaves<0){
			this.max_leaves=0;
		}
		if (this.max_delta_step<0){
			this.max_delta_step=0;
		}	
		if (this.min_child_weight<0){
			this.min_child_weight=0;
		}	
		if (this.colsample_bytree<=0){
			this.colsample_bytree=1.0;
		}	
		if (this.lambda<=0){
			this.lambda=0.0;
		}
		if (this.alpha<0){
			this.alpha=0;
		}
		if (this.subsample<=0){
			this.subsample=1;
		}
		if (this.num_round<1){
			this.num_round=1;
		}
		if (this.colsample_bylevel<=0){
			this.colsample_bylevel=1;
		}	
		if (this.eta<=0){
			this.eta=0.01;
		}		
		if (this.gamma<=0){
			this.gamma=0.01;
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
			
		}	else {
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
		
		this.Objective="binary:logistic";
		if (this.n_classes>2){
			this.Objective="multi:softprob";
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
				"model_out=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");

		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);

	        // create new file
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();
			f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
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

		if ( !booster.equals("gbtree")  && !booster.equals("gblinear") ){
			throw new IllegalStateException(" booster has to be between 'gbtree' and gblinear' " );	
		}
		
		if (this.max_leaves<0){
			this.max_leaves=0;
		}
		if (this.max_delta_step<0){
			this.max_delta_step=0;
		}	
		if (this.min_child_weight<0){
			this.min_child_weight=0;
		}	
		if (this.colsample_bytree<=0){
			this.colsample_bytree=1.0;
		}	
		if (this.lambda<=0){
			this.lambda=0.0;
		}
		if (this.alpha<0){
			this.alpha=0;
		}
		if (this.subsample<=0){
			this.subsample=1;
		}
		if (this.num_round<1){
			this.num_round=1;
		}
		if (this.colsample_bylevel<=0){
			this.colsample_bylevel=1;
		}	
		if (this.eta<=0){
			this.eta=0.01;
		}		
		if (this.gamma<=0){
			this.gamma=0.01;
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
			
		}	else {
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
		
		this.Objective="binary:logistic";
		if (this.n_classes>2){
			this.Objective="multi:softprob";
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
				"model_out=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");

		//make subprocess
		 create_xg_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);
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
		return "XgboostClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: XgboostClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		
		System.out.println("booster: " + this.booster );
	    System.out.println("objective: " + this.Objective);
	    if (this.Objective.equals("multi:softprob")){
	    	System.out.println("Objective: multi:softprob" ); 
	    }
	    else {
	    	System.out.println("Objective: binary:logistic" ); 
	    }
	    System.out.println("eta: " + this.eta  );
	    System.out.println("gamma: " +this.gamma );
	    System.out.println("min_child_weight: " + this.min_child_weight );
	    System.out.println("colsample_bytree: " + this.colsample_bytree);
	    System.out.println("colsample_bylevel: " + this.colsample_bylevel );
	    System.out.println("subsample: " + this.subsample );
	    System.out.println("max_delta_step: " + this.max_delta_step  );
	    System.out.println("lambda: " + this.lambda );
	    System.out.println("alpha: " + this.alpha );
	    System.out.println("scale_pos_weight: " + this.scale_pos_weight );
	    System.out.println("max_depth: " +  this.max_depth );
	    System.out.println("nthread: " + this.threads );
	    System.out.println("num_round: " +  this.num_round );
	    System.out.println("max_leaves: " + this.max_leaves );
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

        booster = "gbtree";
        Objective = "";
        eta = 0.01;
        gamma = 1.0;
        min_child_weight =1.0;
        colsample_bytree=0.5;
        colsample_bylevel=1.0;
        subsample =0.8;
        max_delta_step =1. ;
        lambda =0.1;
        alpha =0.1;
        scale_pos_weight=1.0;
        max_depth = 4;
        threads=4;
        num_round = 100;
        max_leaves=0;
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
				if (metric.equals("lambda")) {this.lambda=Double.parseDouble(value);}
				else if (metric.equals("colsample_bytree")) {this.colsample_bytree=Double.parseDouble(value);}
				else if (metric.equals("subsample")) {this.subsample=Double.parseDouble(value);}	
				else if (metric.equals("num_round")) {this.num_round=Integer.parseInt(value);}
				else if (metric.equals("min_child_weight")) {this.min_child_weight=Double.parseDouble(value);}
				else if (metric.equals("scale_pos_weight")) {this.scale_pos_weight=Double.parseDouble(value);}	
				else if (metric.equals("eta")) {this.eta=Double.parseDouble(value);}					
				else if (metric.equals("max_leaves")) {this.max_leaves=Integer.parseInt(value);}				
				else if (metric.equals("max_depth")) {this.max_depth=Integer.parseInt(value);}
				else if (metric.equals("objective")) {this.Objective=value;}
				else if (metric.equals("booster")) {this.booster=value;}				
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("alpha")) {this.alpha=Double.parseDouble(value);}				
				else if (metric.equals("scale_pos_weight")) {this.scale_pos_weight=Double.parseDouble(value);}						
				else if (metric.equals("gamma")) {this.gamma=Double.parseDouble(value);}
				else if (metric.equals("colsample_bylevel")) {this.colsample_bylevel=Double.parseDouble(value);}
				else if (metric.equals("max_delta_step")) {this.max_delta_step=Double.parseDouble(value);}
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
			  

