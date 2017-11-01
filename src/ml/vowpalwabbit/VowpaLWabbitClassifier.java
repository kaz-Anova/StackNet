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

package ml.vowpalwabbit;
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
*<p>Wraps the popular  <a href="https://github.com/JohnLangford/vowpal_wabbit">Vowpal Wabbit</a>.
*This particular instance is allowing only classification results. Vowpal Wabbit models are being trained via a subprocess based on the operating systems
*executing the class. <b>It is expected that files will be created and their size will vary based on the volumne of the training data.</b></p>
*
*
*<p>Information about the tunable parameters can be found <a href="https://github.com/JohnLangford/vowpal_wabbit/wiki/Command-line-arguments">here</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all Vowpal Wabbit features and the user is advised to use it directly from the source.
*Also the version may not be the final and it is not certain whether it will be updated in the future as it required manual work to find all libraries and
*files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website, hence it is advised to <a href="https://github.com/JohnLangford/vowpal_wabbit/wiki/Command-line-arguments">use VW's online parameter thread in
*github</a> for more information about them. </p></em> 
 */

public class VowpaLWabbitClassifier implements estimator,classifier {

	/**
	 * Number of Training Passes
	 */
	public int passes=3;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * number of bits in the feature table
	 */
	public int bit_precision=18;
	/**
	 * Decay factor for learning_rate between passes
	 */
	public double decay_learning_rate=0.97;
	/**
	 * Number of hidden units to use in a sigmoidal feedforward network with nn hidden units
	 */
	public int nn=0;
	/**
	 * Initial t value. Affects learning rate's updates
	 */
	public double initial_t=0.8;
	/**
	 * t power value. Affects learning rate's updates
	 */
	public double power_t=0.5;
	/**
	 * ftrl alpha parameter when using ftrl
	 */
	public double ftrl_alpha=0.005;
	/**
	 * learning Rate for gradient-based updates
	 */
	public double learning_rate=0.5;
	/**
	 *ftrl beta stability patameter when using ftrl
	 */
	public double ftrl_beta=0.95;		
	/**
	 * L1 regularization 
	 */
	public double l1=0.00001 ;	
	/**
	 * L2 regularization
	 */
	public double l2=0.00001 ;		
	/**
	 * The objective has to be multiclass or binary.
	 */
	private String Objective="multiclass";	
	/**
	 * To use the ftrl optimization option (instead of adaptive). It is on by default.
	 */
	public boolean use_ftrl=true;
	/**
	 * if true it creates all possible 2-way interactions of all features
	 */
	public boolean make2way=false;
	/**
	 * if true it creates all possible 3-way interactions of all features
	 */
	public boolean make3way=false;
	/**
	 * when nn>0, train or test sigmoidal feedforward network using dropout.
	 */
	public boolean use_dropout=false;
	/**
	 * when nn>0, train or test sigmoidal feedforward network using mean field.
	 */
	public boolean use_meanfield=false;
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
    * @param confingname : name of cache file
    * @param istrain : if this is a train task or a prediction task
	* @param datset : the dataset to be used
	* @param model : model dump name
	* @param predictionfile : where the predictions will be saved
    */
   private void create_vw_suprocess(String confingname, boolean istrain , String datset, String model, String predictionfile ) {
	   
	   // create the subprocess
		try {
			 String operational_system=detectos.getOS();
			 if (!operational_system.equals("win") && !operational_system.equals("linux")&& !operational_system.equals("mac")){
				 throw new IllegalStateException(" The operational system is not identified as win, linux or mac which is required to run vw" ); 
			 }
			 String vw_path="lib" + File.separator + operational_system + File.separator + "vw" + File.separator + "vw";
			 List<String> list = new ArrayList<String>();
			 list.add(vw_path);	
			 //String Str_build="-d "  + datset;

			 list.add("--data="  + datset);	
			 if (istrain){
				 list.add("--final_regressor="  + model);			 
				 list.add("--cache_file=" + confingname);	
				 list.add("--link=logistic");	
				 //list.add("-k" );				 
			 } else {
				 list.add("--testonly" ); 
				 list.add("--initial_regressor="  + model); 
				 list.add("--predictions="  + predictionfile); 
				 if (this.n_classes>2){
					 list.add("--probabilities");
				 }
				
			 }

				
				 if (istrain){ 				 
	
				
					list.add("--learning_rate=" +  this.learning_rate );	
					list.add("--bit_precision=" +  this.bit_precision );	
					
					list.add("--initial_t=" +  this.learning_rate );
					list.add("--power_t=" +  this.power_t );
					list.add("--decay_learning_rate=" +  this.decay_learning_rate );				
					
					list.add("--passes=" +  this.passes );	
					list.add("--l1=" +  this.l1 );					
					list.add("--l2=" +  this.l2 );		
				

					if (this.make2way){
						list.add("--quadratic=ff");
					}
					if (this.make3way){
						list.add("--cubic=fff");
					}					
					if (this.use_ftrl){
						list.add("--ftrl");
						list.add("--ftrl_alpha=" +  this.ftrl_alpha );					
						list.add("--ftrl_beta=" +  this.ftrl_beta );	
					}					
					if (this.nn>0){
						list.add("--nn=" +  this.nn );	
						if (this.n_classes>2){
							list.add("--multitask");	 
						}
						if (this.use_dropout){
							list.add("--dropout");	
						}
						if (this.use_meanfield){
							list.add("--meanfield");	
						}						
					}	
					
					if (this.Objective.equals("binary")){
						list.add("--binary");
					} else {
						list.add("--oaa=" +  this.n_classes );	
						//list.add("--probabilities");
					} 
			 }	
				 				 
				 list.add("--loss_function=logistic");				 
				 list.add("--holdout_off");		
				 list.add("--random_seed=" +  this.seed );	
				 
			 
			 //start the process		
			 //list.add(Str_build);	
			 ProcessBuilder p = new ProcessBuilder(list);
			 //System.out.println(Str_build);
			 //System.out.println(p.command());
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
			throw new IllegalStateException(" failed to create vw subprocess with config name " + confingname);
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
	 * Default constructor 
	 */
	public VowpaLWabbitClassifier(){
	
	}	
	/**
	 * Default constructor with double data
	 */
	public VowpaLWabbitClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor  with fsmatrix data
	 */
	public VowpaLWabbitClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor  with smatrix data
	 */
	public VowpaLWabbitClassifier(smatrix data){
		
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
		double temp_in [] =new double [predictions.length];
		Arrays.fill(temp_in, 1);
		out.printsmatrixvw(X, temp_in, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		temp_in=null;	
		
		System.gc();

		//make subprocess
		 create_vw_suprocess("" , false,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models" + File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
		 }else {

			 double temp [][]=io.readcsv.getvowpalpreds(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
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
		double temp_in [] =new double [predictions.length];
		Arrays.fill(temp_in, 1);
		out.printsmatrixvw(X,temp_in  , this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		temp_in=null;	
		
		System.gc();
		
		 create_vw_suprocess("" , false,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
		 }else {

			 double temp [][]=io.readcsv.getvowpalpreds(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
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
		double temp_in [] =new double [predictions.length];
		Arrays.fill(temp_in, 1);		
		out.printsmatrixvw(data,temp_in, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		data=null;
		temp_in=null;		
		 create_vw_suprocess("" , false,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= Math.min(1.0,temp[i]);
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
		 }else {

			 double temp [][]=io.readcsv.getvowpalpreds(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
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
		double temp_in [] =new double [predictions.length];
		Arrays.fill(temp_in, 1);		
		out.printsmatrixvw(X, temp_in, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		temp_in=null;
		System.gc();
		
		 create_vw_suprocess("" , false,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
		 }else {

			 double temp [][]=io.readcsv.getvowpalpreds(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
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
		double temp_in [] =new double [predictions.length];
		Arrays.fill(temp_in, 1);		
		out.printsmatrixvw(data,temp_in,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		data=null;
		temp_in=null;
		System.gc();
		
		
		 create_vw_suprocess("" , false,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
		 }else {

			 double temp [][]=io.readcsv.getvowpalpreds(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
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
		double temp_in [] =new double [predictions.length];
		Arrays.fill(temp_in, 1);			
		out.printsmatrixvw(X, temp_in, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		temp_in=null;
		System.gc();
		
		 create_vw_suprocess("" , false,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
		 
		 if (this.n_classes==2){
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,temp[i]);
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
		 }else {

			 double temp [][]=io.readcsv.getvowpalpreds(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred");
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

	
		if (this.bit_precision<18){
			this.bit_precision=18;
		}
		if (this.decay_learning_rate<0){
			this.decay_learning_rate=0.97;
		}	

		if (this.power_t<=0){
			this.power_t=1.0;
		}	
		if (this.l1<=0){
			this.l1=0.0;
		}
		if (this.l2<0){
			this.l2=0.0;
		}
		if (this.ftrl_beta<=0){
			this.ftrl_beta=0.1;
		}
		if (this.passes<1){
			this.passes=1;
		}
		if (this.ftrl_alpha<0){
			this.ftrl_alpha=0.001;
		}	
		if (this.learning_rate<=0){
			this.learning_rate=0.01;
		}		
		if (this.initial_t<0){
			this.initial_t=0.5;
		}		
		if (this.nn<0){
			this.nn=0;
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
			fstarget=new double [this.target.length];
			for (int i=0; i < this.target.length;i++){
				fstarget[i]=target[i];
			}
						
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
			for (int i=0; i < fstarget.length; i++){
				fstarget[i]+=1.0;
			}
		}else {
			for (int i=0; i < fstarget.length; i++){
				if (fstarget[i]==0){
					fstarget[i]=-1.0;
				}
				
			}			
		}

		
		columndimension=data[0].length;
		//generate config file
		
		//generate dataset
		smatrix X= new smatrix(data);
		//System.out.println(X.GetColumnDimension() + X.GetRowDimension());
		output out = new output();
		out.verbose=false;
		out.printsmatrixvw(X, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		
		//make subprocess
		 create_vw_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 "");
		 
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

		
		if (this.bit_precision<18){
			this.bit_precision=18;
		}
		if (this.decay_learning_rate<0){
			this.decay_learning_rate=0.97;
		}	

		if (this.power_t<=0){
			this.power_t=1.0;
		}	
		if (this.l1<=0){
			this.l1=0.0;
		}
		if (this.l2<0){
			this.l2=0.0;
		}
		if (this.ftrl_beta<=0){
			this.ftrl_beta=0.1;
		}
		if (this.passes<1){
			this.passes=1;
		}
		if (this.ftrl_alpha<0){
			this.ftrl_alpha=0.001;
		}	
		if (this.learning_rate<=0){
			this.learning_rate=0.01;
		}		
		if (this.initial_t<0){
			this.initial_t=0.5;
		}		
		if (this.nn<0){
			this.nn=0;
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
			fstarget=new double [this.target.length];
			for (int i=0; i < this.target.length;i++){
				fstarget[i]=target[i];
			}
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
			for (int i=0; i < fstarget.length; i++){
				fstarget[i]+=1.0;
			}
		}else {
			for (int i=0; i < fstarget.length; i++){
				if (fstarget[i]==0){
					fstarget[i]=-1.0;
				}
				
			}			
		}

		
		columndimension=data.GetColumnDimension();
		//generate config file
		
		//generate dataset
		smatrix X= new smatrix(data);
		output out = new output();
		out.verbose=false;
		out.printsmatrixvw(X, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		//System.out.println(" file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" + " is printed");

		X=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();
		

		//make subprocess
		 create_vw_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 "");

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
		// check directory
		if (this.usedir.equals("")){
			usedir=System.getProperty("user.dir"); // working directory
			
		}
		
		File directory = new File(this.usedir +  File.separator + "models");
		
		if (! directory.exists()){
			directory.mkdir();
		}

		
		if (this.bit_precision<18){
			this.bit_precision=18;
		}
		if (this.decay_learning_rate<0){
			this.decay_learning_rate=0.97;
		}	

		if (this.power_t<=0){
			this.power_t=1.0;
		}	
		if (this.l1<=0){
			this.l1=0.0;
		}
		if (this.l2<0){
			this.l2=0.0;
		}
		if (this.ftrl_beta<=0){
			this.ftrl_beta=0.1;
		}
		if (this.passes<1){
			this.passes=1;
		}
		if (this.ftrl_alpha<0){
			this.ftrl_alpha=0.001;
		}	
		if (this.learning_rate<=0){
			this.learning_rate=0.01;
		}		
		if (this.initial_t<0){
			this.initial_t=0.5;
		}		
		if (this.nn<0){
			this.nn=0;
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
			fstarget=new double [this.target.length];
			for (int i=0; i < this.target.length;i++){
				fstarget[i]=target[i];
			}	
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
			for (int i=0; i < fstarget.length; i++){
				fstarget[i]+=1.0;
			}
		}else {
			for (int i=0; i < fstarget.length; i++){
				if (fstarget[i]==0){
					fstarget[i]=-1.0;
				}
				
			}			
		}
		
		columndimension=data.GetColumnDimension();
		//generate config file
		
		//generate dataset

		output out = new output();
		out.verbose=false;
		out.printsmatrixvw(data, fstarget,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
		//System.out.println(" file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" + " is printed");
		data=null;
		fstarget=null;
		fsdataset=null;
		sdataset=null;
		System.gc();


		//make subprocess
		 create_vw_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".conf" , true,
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
				 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
				 "");
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
		return "VowpaLWabbitClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: VowpaLWabbitClassifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
	    if (this.Objective.equals("multiclass")){
	    	System.out.println("Objective: multiclass" ); 
	    }
	    else {
	    	System.out.println("Objective: binary" ); 
	    }
	    System.out.println("learning_rate=" + this.learning_rate  );
	    System.out.println("initial_t=" +this.initial_t );
	    System.out.println("power_t=" + this.power_t);
	    System.out.println("ftrl_alpha=" + this.ftrl_alpha);
	    System.out.println("ftrl_beta=" + this.ftrl_beta );
	    System.out.println("decay_learning_rate=" + this.decay_learning_rate );
	    System.out.println("l1=" + this.l1 );
	    System.out.println("l2=" + this.l2 );
	    System.out.println("nn=" +  this.nn );
	    System.out.println("threads=" + this.threads );
	    System.out.println("passes=" +  this.passes );
	    System.out.println("seed=" +  this.seed );				    	
	    System.out.println("use_ftrl=" + this.use_ftrl );				    			    			    
	    System.out.println("make2way=" + this.make2way );				    
	    System.out.println("make3way=" + this.make3way );	
	    System.out.println("use_meanfield=" + this.use_meanfield );				    
	    System.out.println("use_dropout=" + this.use_dropout );	    


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
        Objective = "";
        passes=3;
		bit_precision=18;
		decay_learning_rate=0.97;
		nn=0;
		initial_t=0.5;
		power_t=0.5;
		ftrl_alpha=0.005;
		learning_rate=0.8;
		ftrl_beta=0.1;

		l1=0.00001;
		l2=0.00001;
		use_ftrl=false;
		make2way=false;
		make3way=false;
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
		use_meanfield=false;
		use_dropout=false;		
		
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
				if (metric.toLowerCase().equals("l1")) {this.l1=Double.parseDouble(value);}
				else if (metric.toLowerCase().equals("l2")) {this.l2=Double.parseDouble(value);}				
				else if (metric.toLowerCase().equals("power_t")) {this.power_t=Double.parseDouble(value);}
				else if (metric.toLowerCase().equals("ftrl_beta")) {this.ftrl_beta=Double.parseDouble(value);}	
				else if (metric.toLowerCase().equals("passes")) {this.passes=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("learning_rate")) {this.learning_rate=Double.parseDouble(value);}			
				else if (metric.toLowerCase().equals("bit_precision")) {this.bit_precision=Integer.parseInt(value);}				
				else if (metric.toLowerCase().equals("nn")) {this.nn=Integer.parseInt(value);}		
				else if (metric.toLowerCase().equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("ftrl_alpha")) {this.ftrl_alpha=Double.parseDouble(value);}						
				else if (metric.toLowerCase().equals("initial_t")) {this.initial_t=Double.parseDouble(value);}					
				else if (metric.toLowerCase().equals("make2way")) {this.make2way=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.toLowerCase().equals("use_ftrl")) {this.use_ftrl=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.toLowerCase().equals("make3way")) {this.make3way=(value.toLowerCase().equals("true")?true:false)   ;}													
				else if (metric.toLowerCase().equals("decay_learning_rate")) {this.decay_learning_rate=Double.parseDouble(value);}
				else if (metric.toLowerCase().equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.toLowerCase().equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.toLowerCase().equals("use_dropout")) {this.use_dropout=(value.toLowerCase().equals("true")?true:false)   ;}				
				else if (metric.toLowerCase().equals("use_meanfield")) {this.use_meanfield=(value.toLowerCase().equals("true")?true:false)   ;}				
				
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
			  

