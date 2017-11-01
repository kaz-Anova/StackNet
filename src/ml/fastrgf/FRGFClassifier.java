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

package ml.fastrgf;
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
*<p>Wraps <a href="https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiLya_dmq3VAhVHBcAKHRPFDxQQFggoMAA&url=https%3A%2F%2Fgithub.com%2Fbaidu%2Ffast_rgf&usg=AFQjCNEK2Se2nHYT5y-KS9TUnD1TJJafUg">fast_rgf</a>).
*This particular instance is allowing only classification results. fast_rgf models are being trained via a subprocess based on the operating systems
*executing the class. <b>It is expected that files will be created and their size will vary based on the volumne of the training data.</b></p>
*
*
*<p>Information about the tunable parameters can be found <a href="https://github.com/baidu/fast_rgf/tree/master/examples">here</a> </p> 
*
*Reference :  <em>Rie Johnson and Tong Zhang. Learning Nonlinear Functions Using Regularized Greedy Forest, IEEE Trans. on Pattern Analysis and Machine Intelligence, 36:942-954, 2014.</em>
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all fast_rgf features and the user is advised to use it directly from the source.
*Also the version may not be the final and it is not certain whether it will be updated in the future as it required manual work to find all libraries and
*files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website, hence it is advised to <a href="https://github.com/baidu/fast_rgf/tree/master/examples">use fast_rgf online parameter thread in
*github</a> for more information about them. </p></em> 
 */

public class FRGFClassifier implements estimator,classifier {

	/**
	 * Number of trees to build
	 */
	public int ntrees=100;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * maximum number of nodes
	 */
	public int max_nodes=10;
	/**
	 * maximum depth of the tree
	 */
	public int max_level=4;
	/**
	 * new tree is created when leaf-nodes gain < this value * estimated gain of creating new three
	 */
	public double new_tree_gain_ratio=1.0;
	/**
	 * Step size of epsilon-greedy boosting (inactive for rgf)
	 */
	public double stepsize=0.1;
	/**
	 * Minimum sum of data weights for each discretized value
	 */
	public int min_bucket_weights=5;	
	/**
	 *  Maximum bins for dense data
	 */
	public int dense_max_buckets=250;	
	/**
	 * Histogram bins for sparse data
	 */
	public int sparse_max_buckets=250;
	/**
	 * you may try a different value in [1000,10000000]
	 */
	public double sparse_max_features=1000;
	/**
	 * L1 regularization on the weights
	 */
	public double lamL1=0;	
	/**
	 * L2 regularization on the weights
	 */
	public double lamL2=0;	
	/**
	 * L2 regularization parameter for sparse data
	 */
	public double sparse_lamL2=0;	
	/**
	 * minimum samples in node
	 */
	public int min_sample=0;
	/**
	 * minimum number of occurrences for a feature to be selected
	 */
	public int min_occurrences=0;	
	/**
	 * The objective has to be BINARY.
	 */
	private String Objective="BINARY";	
	/**
	 * optimization method for training forest (rgf or epsilon-greedy)
	 */
	public String opt="rgf";	
	/**
	 * Type of loss. could be LS, MODLS (modified least squares loss), or LOGISTIC
	 */
	public String loss="LOGISTIC";
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
	 * @param target : file containing only target
	 */
   private void create_config_file(String filename , String datset, String model , String target){

		try{  // Catch errors in I/O if necessary.
			  // Open a file to write to.
				String saveFile = filename;
				
				FileWriter writer = new FileWriter(saveFile);
				writer.append("dtree.loss=" + this.loss + "\n");
			    writer.append("trn.target=" + this.Objective+ "\n");
			    writer.append("forest.stepsize=" + this.stepsize  + "\n");
			    writer.append("discretize.sparse.max_buckets=" + this.sparse_max_buckets + "\n");
			    writer.append("dtree.min_sample=" + this.min_sample+ "\n");
			    writer.append("dtree.new_tree_gain_ratio=" + this.new_tree_gain_ratio + "\n");
			    writer.append("discretize.sparse.min_bucket_weights=" + this.min_bucket_weights + "\n");
			    writer.append("dtree.lamL1=" + this.lamL1 + "\n");
			    writer.append("dtree.lamL2=" + this.lamL2 + "\n");
			    writer.append("discretize.sparse.max_features=" + this.sparse_max_features + "\n");
			    writer.append("dtree.max_level=" +  this.max_level + "\n");
			    writer.append("set.nthreads=" + this.threads + "\n");
			    writer.append("forest.ntrees=" +  this.ntrees + "\n");				    			    			    
			    writer.append("dtree.max_nodes=" + this.max_nodes + "\n");
			    writer.append("discretize.dense.max_buckets=" + this.dense_max_buckets + "\n");	
			    writer.append("discretize.sparse.lamL2=" + this.sparse_lamL2 + "\n");			
			    writer.append("discretize.sparse.min_occrrences=" + this.min_occurrences + "\n");				    			    
			    writer.append("forest.opt=" + this.opt + "\n");					    
			    writer.append("trn.x-file_format=sparse\n");				    
			    if (this.verbose){
			    	writer.append("set.verbose=2" +  "\n");
			    }else {
			    	writer.append("set.verbose=0" +  "\n");			    	
			    }
			    //file details
			    writer.append(datset+ "\n");
			    writer.append(model+ "\n");		
			    writer.append(target+ "\n");				    
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
	 * @param predictionfile : where the predictions will be saved
	 */
  private void create_config_file_pred(String filename , String datset, String model, String predictionfile){

		try{  // Catch errors in I/O if necessary.
			  // Open a file to write to.
				String saveFile = filename;
				
				FileWriter writer = new FileWriter(saveFile);	  
			    writer.append("set.nthreads=" + this.threads + "\n");
			    if (this.verbose){
			    	writer.append("set.verbose=2" +  "\n");
			    }else {
			    	writer.append("set.verbose=0" +  "\n");			    	
			    }
			    //file details
			    writer.append(datset+ "\n");
			    writer.append( model+ "\n");
			    writer.append("tst.x-file_format=sparse\n");
			    writer.append( "tst.output-prediction=" + predictionfile + "\n");
		    
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
   private void create_frgf_suprocess(String confingname, boolean istrain ) {
	   
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
			 String frgf_path="lib" + File.separator + operational_system + File.separator + "frgf" + File.separator + "forest_train";
			 if (istrain==false){
				 frgf_path="lib" + File.separator + operational_system + File.separator + "frgf" + File.separator + "forest_predict";
			 }
			 List<String> list = new ArrayList<String>();
			 list.add(frgf_path);			 
			 list.add("-config=" + confingname);			 

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
			throw new IllegalStateException(" failed to create fast_rgf subprocess with config name " + confingname);
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
	public FRGFClassifier(){
	
	}	
	/**
	 * Default constructor with double data
	 */
	public FRGFClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor with fsmatrix data
	 */
	public FRGFClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor with smatrix data
	 */
	public FRGFClassifier(smatrix data){
		
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
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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
		
		 if (this.n_classes==2){
			 
				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".conf" ,
						"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
						);

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);			 
			 
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,1.0/(1+Math.exp(-temp[i]) ) );
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" );
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
					create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
							"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
							"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".mod",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred"	
							);

					//make subprocess
					 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" , false);
					 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
						 predictions[i][n]=1.0/(1+Math.exp(-temp[i]) ); 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
			        f.delete();
					f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
			        f.delete(); 

					System.gc();        
			 }
			 
			 for (int i =0; i <predictions.length;i++ ){
				 double sum=0.0;
				 for (int j =0; j <this.n_classes;j++ ){
					 sum+=predictions[i][j];
				 }
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]/=sum;
				 }				 
			 } 
		 }
		 
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
		
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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

		 if (this.n_classes==2){
			 
				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
						"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
						);

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);			 
			 
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,1.0/(1+Math.exp(-temp[i]) ) );
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".conf" );
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
					create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
							"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
							"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".mod",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred"	
							);

					//make subprocess
					 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" , false);
					 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
						 predictions[i][n]=1.0/(1+Math.exp(-temp[i]) ); 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
			        f.delete();
					f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
			        f.delete(); 

					System.gc();        
			 }
			 
			 for (int i =0; i <predictions.length;i++ ){
				 double sum=0.0;
				 for (int j =0; j <this.n_classes;j++ ){
					 sum+=predictions[i][j];
				 }
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]/=sum;
				 }				 
			 } 
		 }
		 
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
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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
		
		 if (this.n_classes==2){
			 
				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
						"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
						);

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);			 
			 
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,1.0/(1+Math.exp(-temp[i]) ) );
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
					create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
							"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
							"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".mod",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred"	
							);

					//make subprocess
					 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" , false);
					 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
						 predictions[i][n]=1.0/(1+Math.exp(-temp[i]) ); 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
			        f.delete();
					f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
			        f.delete(); 

					System.gc();        
			 }
			 
			 for (int i =0; i <predictions.length;i++ ){
				 double sum=0.0;
				 for (int j =0; j <this.n_classes;j++ ){
					 sum+=predictions[i][j];
				 }
				 for (int j =0; j <this.n_classes;j++ ){
					 predictions[i][j]/=sum;
				 }				 
			 } 
		 }
		 
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
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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

		 if (this.n_classes==2){
			 
				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name.replace(" ", "") + ".conf" ,
						"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
						);

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);			 
			 
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,1.0/(1+Math.exp(-temp[i]) ) );
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".conf" );
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
					create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
							"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
							"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".mod",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred"	
							);

					//make subprocess
					 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" , false);
					 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
					 prediction_probas[i][n]=temp[i]; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
			        f.delete();
					f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
			        f.delete(); 

					System.gc();        
			 }
			 
		 }
		 
     // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		f.delete();

		System.gc();
         
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			double [] temp=prediction_probas[i];
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
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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
		
		 if (this.n_classes==2){
			 
				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
						"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
						);

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);			 
			 
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,1.0/(1+Math.exp(-temp[i]) ) );
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
					create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
							"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
							"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".mod",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred"	
							);

					//make subprocess
					 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" , false);
					 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
					 prediction_probas[i][n]=temp[i]; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
			        f.delete();
					f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
			        f.delete(); 

					System.gc();        
			 }
			 
		 }
		 
  // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		f.delete();

		System.gc();
      
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			double [] temp=prediction_probas[i];
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
		if (n_classes<2 ||new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod").exists()==false ){
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
		 
		 if (this.n_classes==2){
			 
				create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
						"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred"	
						);

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , false);			 
			 
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,1.0/(1+Math.exp(-temp[i]) ) );
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" );
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
					create_config_file_pred(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" ,
							"tst.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
							"model.load=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".mod",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred"	
							);

					//make subprocess
					 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" , false);
					 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
					 prediction_probas[i][n]=temp[i]; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".conf" );
			        f.delete();
					f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
			        f.delete(); 

					System.gc();        
			 }
			 
		 }
		 
  // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		f.delete();

		System.gc();
      
		System.gc();
			// return the 1st prediction

		for (int i=0; i < predictions.length; i++) {
			double [] temp=prediction_probas[i];
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
		if ( !loss.equals("LOGISTIC")  && !loss.equals("LS") & !loss.equals("MODLS")){
			throw new IllegalStateException(" loss has to be in 'LOGISTIC', 'LS' or 'MODLS'" );	
		}
		if ( !opt.equals("rgf")  && !opt.equals("epsilon-greedy") ){
			throw new IllegalStateException(" opt has to be between 'rgf' and 'epsilon-greedy'" );	
		}		

		if (this.min_occurrences<0){
			this.min_occurrences=1;
		}		    
    
		if (this.sparse_lamL2<0){
			this.sparse_lamL2=0.1;
		}	    
	
		if (this.dense_max_buckets<0){
			this.dense_max_buckets=250;
		}
		
		if (this.max_nodes<0){
			this.max_nodes=10;
		}	
	
		if (this.lamL1<=0){
			this.lamL1=0.1;
		}
		if (this.lamL2<0){
			this.lamL2=0.1;
		}
		if (this.min_bucket_weights<=0){
			this.min_bucket_weights=250;
		}
		if (this.ntrees<1){
			this.ntrees=50;
		}
		if (this.new_tree_gain_ratio<0){
			this.new_tree_gain_ratio=1.0;
		}	
		if (this.stepsize<=0){
			this.stepsize=0.1;
		}				
		if (this.sparse_max_features<=0){
			this.sparse_max_features=1000;
		}
		if (this.max_level<=0){
			this.max_level=4;
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
		//generate config file
		
		//generate dataset
		smatrix X= new smatrix(data);
		System.out.println(X.GetColumnDimension() + X.GetRowDimension());
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		if (n_classes==2){
			out = new output();
			out.verbose=false;
			out.printSingledouble(fstarget, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label");
			fstarget=null;
			
			create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
					"trn.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					"model.save=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
					"trn.y-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label");

			//make subprocess
			 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);
			 
	        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label" );
	        // tries to delete a non-existing file
	        f.delete();
			f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".conf" );
	        // tries to delete a non-existing file
	        f.delete();			

		}else {	
		
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [fstarget.length];
				for (int i=0; i < label.length; i++){
					if ( fstarget[i]==Double.parseDouble(classes[n]) ){
							label[i]=1.0;
						} else {
							label[i]=0.0;	
						}
				}
				
				out = new output();
				out.verbose=false;
				out.printSingledouble(label, this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".label");
				label=null;
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".conf" ,
						"trn.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
						"model.save=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+".mod",
						"trn.y-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+".label");

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".conf" , true);
				 

		        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".label" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+n+ ".conf" );
		        // tries to delete a non-existing file
		        f.delete();					
				
				
	
	
				}
		}	
		
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".train" );
        // tries to delete a non-existing file
        f.delete();		
		
		
		
		fsdataset=null;
		sdataset=null;
		System.gc();
		

        

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

		if ( !loss.equals("LOGISTIC")  && !loss.equals("LS") & !loss.equals("MODLS")){
			throw new IllegalStateException(" loss has to be in 'LOGISTIC', 'LS' or 'MODLS'" );	
		}
		if ( !opt.equals("rgf")  && !opt.equals("epsilon-greedy") ){
			throw new IllegalStateException(" opt has to be between 'rgf' and 'epsilon-greedy'" );	
		}		

		if (this.min_occurrences<0){
			this.min_occurrences=1;
		}		    
    
		if (this.sparse_lamL2<0){
			this.sparse_lamL2=0.1;
		}	    
	
		if (this.dense_max_buckets<0){
			this.dense_max_buckets=250;
		}
		
		if (this.max_nodes<0){
			this.max_nodes=10;

		}	

		if (this.lamL1<=0){
			this.lamL1=0.1;
		}
		if (this.lamL2<0){
			this.lamL2=0.1;
		}
		if (this.min_bucket_weights<=0){
			this.min_bucket_weights=250;
		}
		if (this.ntrees<1){
			this.ntrees=50;
		}
		if (this.new_tree_gain_ratio<0){
			this.new_tree_gain_ratio=1.0;
		}	
		if (this.stepsize<=0){
			this.stepsize=0.1;
		}				
		if (this.sparse_max_features<=0){
			this.sparse_max_features=1000;
		}
		if (this.max_level<=0){
			this.max_level=4;
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
		//generate config file
		
		//generate dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrix(X,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		if (n_classes==2){
			out = new output();
			out.verbose=false;
			out.printSingledouble(fstarget, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label");
			fstarget=null;
			
			create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
					"trn.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					"model.save=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
					"trn.y-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label");

			//make subprocess
			 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);
			 
	        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label" );
	        // tries to delete a non-existing file
	        f.delete();
			f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".conf" );
	        // tries to delete a non-existing file
	        f.delete();			

		}else {	
		
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [fstarget.length];
				for (int i=0; i < label.length; i++){
					if ( fstarget[i]==Double.parseDouble(classes[n]) ){
							label[i]=1.0;
						} else {
							label[i]=0.0;	
						}
				}
				
				out = new output();
				out.verbose=false;
				out.printSingledouble(label, this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".label");
				label=null;
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".conf" ,
						"trn.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
						"model.save=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+".mod",
						"trn.y-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+".label");

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".conf" , true);
				 

		        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".label" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+n+ ".conf" );
		        // tries to delete a non-existing file
		        f.delete();					
				
				
	
	
				}
		}	
		
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".train" );
        // tries to delete a non-existing file
        f.delete();	
		
		fsdataset=null;
		sdataset=null;
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
		// check directory
		if (this.usedir.equals("")){
			usedir=System.getProperty("user.dir"); // working directory
			
		}
		
		File directory = new File(this.usedir +  File.separator + "models");
		
		if (! directory.exists()){
			directory.mkdir();
		}

		if ( !loss.equals("LOGISTIC")  && !loss.equals("LS") & !loss.equals("MODLS")){
			throw new IllegalStateException(" loss has to be in 'LOGISTIC', 'LS' or 'MODLS'" );	
		}
		if ( !opt.equals("rgf")  && !opt.equals("epsilon-greedy") ){
			throw new IllegalStateException(" opt has to be between 'rgf' and 'epsilon-greedy'" );	
		}		

		if (this.min_occurrences<0){
			this.min_occurrences=1;
		}		    
    
		if (this.sparse_lamL2<0){
			this.sparse_lamL2=0.1;
		}	    
	
		if (this.dense_max_buckets<0){
			this.dense_max_buckets=250;
		}
		
		if (this.max_nodes<0){
			this.max_nodes=10;
		}	
	
		if (this.lamL1<=0){
			this.lamL1=0.1;
		}
		if (this.lamL2<0){
			this.lamL2=0.1;
		}
		if (this.min_bucket_weights<=0){
			this.min_bucket_weights=250;
		}
		if (this.ntrees<1){
			this.ntrees=50;
		}
		if (this.new_tree_gain_ratio<0){
			this.new_tree_gain_ratio=1.0;
		}	
		if (this.stepsize<=0){
			this.stepsize=0.1;
		}				
		if (this.sparse_max_features<=0){
			this.sparse_max_features=1000;
		}
		if (this.max_level<=0){
			this.max_level=4;
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
		//generate config file
		
		//generate dataset

		output out = new output();
		out.verbose=false;

		out.printsmatrix(data,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		data=null;
		if (n_classes==2){
			out = new output();
			out.verbose=false;
			out.printSingledouble(fstarget, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label");
			fstarget=null;
			
			create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" ,
					"trn.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					"model.save=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
					"trn.y-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label");

			//make subprocess
			 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true);
			 
	        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".label" );
	        // tries to delete a non-existing file
	        f.delete();
			f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".conf" );
	        // tries to delete a non-existing file
	        f.delete();			

		}else {	
		
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [fstarget.length];
				for (int i=0; i < label.length; i++){
					if ( fstarget[i]==Double.parseDouble(classes[n]) ){
							label[i]=1.0;
						} else {
							label[i]=0.0;	
						}
				}
				
				out = new output();
				out.verbose=false;
				out.printSingledouble(label, this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".label");
				label=null;
				
				create_config_file(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".conf" ,
						"trn.x-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
						"model.save=" +this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+".mod",
						"trn.y-file="+this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+".label");

				//make subprocess
				 create_frgf_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".conf" , true);
				 

		        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".label" );
		        // tries to delete a non-existing file
		        f.delete();
				f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+n+ ".conf" );
		        // tries to delete a non-existing file
		        f.delete();					
				
				
	
	
				}
		}	
		
        // create new file
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name+ ".train" );
        // tries to delete a non-existing file
        f.delete();

		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
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
		return "FRGFClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: FRGFClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		
		System.out.println("loss: " + this.loss );
	    System.out.println("objective: " + this.Objective);
	    System.out.println("stepsize=" + this.stepsize  );
	    System.out.println("sparse_max_buckets=" + this.sparse_max_buckets );
	    System.out.println("new_tree_gain_ratio=" + this.new_tree_gain_ratio);
	    System.out.println("min_bucket_weights=" + this.min_bucket_weights );
	    System.out.println("lamL1=" + this.lamL1 );
	    System.out.println("lamL2=" + this.lamL2 );
	    System.out.println("sparse_max_features=" + this.sparse_max_features );
	    System.out.println("max_level=" +  this.max_level );
	    System.out.println("min_sample=" +  this.min_sample );	    
	    System.out.println("threads=" + this.threads );
	    System.out.println("ntrees=" +  this.ntrees );
	    System.out.println("seed=" +  this.seed );				    
	    System.out.println("dense_max_buckets=" + this.dense_max_buckets );	
	    System.out.println("sparse_lamL2=" + this.sparse_lamL2 );					    			    
	    System.out.println("opt=" + this.opt );
		

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
        loss = "LOGISTIC";
        Objective = "";
        ntrees=100;
		max_nodes=10;
		max_level=5;
		sparse_max_buckets=250;
		new_tree_gain_ratio=0;
		stepsize=0.1;
		min_bucket_weights=5;
		dense_max_buckets=250;
		sparse_max_features=1000;
		lamL1=0;
		lamL2=0;
		opt="rgf";
		sparse_lamL2=1;
		min_occurrences=5;
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
				if (metric.equals("lamL1")) {this.lamL1=Double.parseDouble(value);}
				else if (metric.equals("lamL2")) {this.lamL2=Double.parseDouble(value);}				
				else if (metric.equals("ntrees")) {this.ntrees=Integer.parseInt(value);}
				else if (metric.equals("sparse_max_features")) {this.sparse_max_features=Integer.parseInt(value);}	
				else if (metric.equals("stepsize")) {this.stepsize=Double.parseDouble(value);}				
				else if (metric.equals("max_nodes")) {this.max_nodes=Integer.parseInt(value);}				
				else if (metric.equals("max_level")) {this.max_level=Integer.parseInt(value);}
				else if (metric.equals("min_sample")) {this.min_sample=Integer.parseInt(value);}				
				else if (metric.equals("loss")) {this.loss=value;}				
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("new_tree_gain_ratio")) {this.new_tree_gain_ratio=Double.parseDouble(value);}	
				else if (metric.equals("sparse_max_buckets")) {this.sparse_max_buckets=Integer.parseInt(value);}
				else if (metric.equals("sparse_max_features")) {this.sparse_max_features=Integer.parseInt(value);}										
				else if (metric.equals("sparse_lamL2")) {this.sparse_lamL2=Double.parseDouble(value);}	
				else if (metric.equals("opt")) {this.opt=value;}	
				else if (metric.equals("dense_max_buckets")) {this.dense_max_buckets=Integer.parseInt(value);}					
				else if (metric.equals("min_occurrences")) {this.min_occurrences=Integer.parseInt(value);}						
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
			  

