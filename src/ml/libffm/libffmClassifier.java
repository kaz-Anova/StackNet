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

package ml.libffm;
import io.output;

import java.io.BufferedReader;
import java.io.File;
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
import misc.print;
import ml.classifier;
import ml.estimator;

/**
*<p>Wraps <a href="https://github.com/guestwalk/libffm">libffm</a> which has won many kaggle competitions.
*This particular instance is allowing only classification results. There is no wrapper available for regression. libffm models are being trained via a subprocess based on the operating systems
*executing the class. <b>It is expected that files will be created and their size will vary based on the volumne of the training data.</b></p>
*
*
*<p>Information about the tunable parameters can be found <a href="https://github.com/guestwalk/libffm">here</a> </p> 
*
*Reference : Juan, Y., Zhuang, Y., Chin, W. S., & Lin, C. J. (2016, September). Field-aware factorization machines for CTR prediction. In <em>Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50)</em>. ACM. 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all libffm features and the user is advised to use it directly from the source.
*Also the version may not be the final and it is not certain whether it will be updated in the future as it required manual work to find all libraries and
*files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website, hence it is advised to use libffm online parameter thread in github for more information about them. </p></em> 
 *<p> Multiclass problems are formed as binary 1-vs-all. 
 *
 */

public class libffmClassifier implements estimator,classifier {

	/**
	 *number of latent factors
	 */
	public int factor=4;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * number of iterations
	 */
	public int iteration=15;
	/**
	 * learning rate 
	 */
	public double learn_rate=0.1;
	/**
	 * regularization parameter
	 */
	public double lambda=0.00002;
	/**
	 * allow instance-wise normalization
	 */
	public boolean use_norm=true;	
	/**
	 * method for determining the factors. The best way (but not the default) is to provide a list with comma separated indices. Consider this String '1,4,7,123,546'. This would mean that the 0 column is a field on its own, {1,2,3} form another field, {4,5,6} another. {7,8...122} form another field and so on.  
	 *<p> Another possible value is 'no_order' (default). This looks at the proportion of zeros in neighbouring columns to determine if the form a field. 
	 *<p> The last possible value is 'order'. This calculates frequencies of non-zero values for all columns and then orders them based on frequency. Columns that have a few missing values form their own fields. Weaker columns (frequency-wise) are joined together to form fields.  
	 */
	public String opt="no_order";	
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
	 * field each column corresponds to
	 */
	private int fields[];
	/**
	 * 
	 * @return a vector that contains the field of each column.
	 */
	public int [] get_fileds(){
		return fields.clone();
	}
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
		* @param datset : the dataset to be used
		* @param model : model dump name
		* @param prediction : filename to save predictions if istrain==false
        * @param istrain : if this is a train task or a prediction task
	    */
	   private void create_libffm_suprocess(String datset, String model,String prediction,  boolean istrain) {
		   
		   // create the subprocess
			try {
				 String operational_system=detectos.getOS();
				 if (!operational_system.equals("win") && !operational_system.equals("linux")&& !operational_system.equals("mac")){
					 throw new IllegalStateException(" The operational system is not identified as win, linux or mac which is required to run libfm" ); 
				 }
				 String libfm_path="lib" + File.separator + operational_system + File.separator + "libffm" + File.separator;
				 if(istrain){
					 libfm_path= libfm_path+ "ffm-train";
				 } else {
					 libfm_path= libfm_path+ "ffm-predict";
				 }

				 List<String> list = new ArrayList<String>();
				 list.add(libfm_path);	
				 //String Str_build="-d "  + datset;
				 if (istrain){
					 list.add("-l" );
					 list.add(this.lambda+"");				 
					 list.add("-k" );
					 list.add(this.factor+"");	
					 list.add("-t" );
					 list.add(this.iteration+"");	
					 list.add("-r" );
					 list.add(this.learn_rate+"");	
					 list.add("-s" );
					 list.add(this.threads+"");	
					 if (this.use_norm==false){
						 list.add("--no-norm" ); 
					 }
					 if (this.verbose==false){
						 list.add("--quiet" ); 
					 }				 
					 list.add(datset);				 						 					 
					 list.add(model);
				 }else {
					 
					 list.add(datset);				 						 					 
					 list.add(model); 
					 list.add(prediction); 
				 }
				 

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
				throw new IllegalStateException(" failed to create libffm subprocess");
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
	public libffmClassifier(){
	
	}	
	/**
	 * Default constructor with double data
	 */
	public libffmClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor with fsmatrix data
	 */
	public libffmClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor with smatrix data
	 */
	public libffmClassifier(smatrix data){
		
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
		out.printsmatrix(X,new double [predictions.length], fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		System.gc();
		
		 if (this.n_classes==2){
			 
			 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
						false
						);
		 
			 
			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]=temp[i];
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 
				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				 
				 create_libffm_suprocess(
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".mod",
							this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred",
							false
							);
				


				double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
						 predictions[i][n]=temp[i] ; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
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
		out.printsmatrix(X,new double [predictions.length], fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;
		
		System.gc();

		 if (this.n_classes==2){
			 
			 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
						false
						);


			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]=temp[i];
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
				 create_libffm_suprocess(
					this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
					this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".mod",
					this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred",
					false
					);


					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
						 predictions[i][n]=temp[i] ; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
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
		out.printsmatrix(data,new double [predictions.length], fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		
		 if (this.n_classes==2){
			 
			 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
						false
						);


			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 predictions[i][1]= temp[i];
				 predictions[i][0]= 1.0-predictions[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
				 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred",
						false
						);

					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
						 predictions[i][n]=temp[i]; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
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
		out.printsmatrix(X,new double [predictions.length], fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();

		 if (this.n_classes==2){
			 
			 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
						false
						);


			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= temp[i];
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
				 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred",
						false
						);


					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
					 prediction_probas[i][n]=temp[i]; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
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
		out.printsmatrix(data,new double [predictions.length], fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 

		System.gc();
		
		 if (this.n_classes==2){
			 
			 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
						false
						);


			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= temp[i];
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
				 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred",
						false
						);
				 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
					 prediction_probas[i][n]=temp[i]; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
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
		out.printsmatrix(X,new double [predictions.length], fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");//this.usedir +  File.separator +  "models"+File.separator + 
		X=null;

		System.gc();
		 
		 if (this.n_classes==2){
			 
			 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred",
						false
						);


			 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred", "\t", 0, 0.0, false, false);
			 for (int i =0; i <predictions.length;i++ ){
				 prediction_probas[i][1]= Math.min(1.0,1.0/(1+Math.exp(-temp[i]) ) );
				 prediction_probas[i][0]= 1.0-prediction_probas[i][1];
				 
			 }
		        temp=null;
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".pred" );
		        f.delete(); 

				System.gc();        
		        
		 }else {
			 
			 for (int n=0 ; n <this.n_classes; n++){
				
				 create_libffm_suprocess(
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n +".mod",
						this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred",
						false
						);
				 
					 double temp []=io.input.Retrievecolumn(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred", "\t", 0, 0.0, false, false);
				 if (temp.length!=predictions.length){
					 throw new IllegalStateException(" The produced score in temporary file " + this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred " + " is not of correct size" );
				 }
				 for (int i =0; i <predictions.length;i++ ){
					 prediction_probas[i][n]=temp[i]; 
				 } 
			        temp=null;
			        
					File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + n + ".pred" );
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

		if ( !opt.equals("order")  && !opt.equals("no_order") ){
			
		    	String splits [] = opt.split(",");
		    	for (String ele: splits){
		    		try{
		    			Integer.parseInt(ele);
		    		}catch (Exception e){
		    			throw new IllegalStateException(" if opt is not 'order' and 'no_order', it needs to have comma separated integer indices .Here it receied: " + ele  );	
		    		}
		    	}
		}		

		
		if (this.iteration<=0){
			this.iteration=1;
		}	
	
		if (this.lambda<0){
			this.lambda=0.00001;
		}
		if (this.factor<1){
			this.factor=1;
		}	
		if (this.learn_rate<=0){
			this.learn_rate=0.01;
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
		//System.out.println(X.GetColumnDimension() + X.GetRowDimension());
		fields=null;
		if(this.opt.equals("order")){
			fields=Fieldfinder.get_fileds(X);
		} else if(this.opt.equals("no_order")){
			fields=Fieldfinder.get_fileds_noorder(X);
		} else {
			fields=Fieldfinder.get_fileds(this.opt,X.GetColumnDimension() );
		}
		
		output out = new output();
		out.verbose=false;

		if (n_classes==2){
			//System.out.println(Arrays.toString(classes));

			out = new output();
			out.verbose=false;
			out.printsmatrix(X,fstarget,fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
			X=null;			
			fstarget=null;
			
			//make subprocess
			 create_libffm_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
					 "",
					 true);
			 
	        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();			
	        f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +  ".train.bin" );
	        // tries to delete a non-existing file
	        f.delete();	
		}else {	
		
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [fstarget.length];
				for (int i=0; i < label.length; i++){
					if ( fstarget[i]==Double.parseDouble(classes[n])){
							label[i]=1.0;
						} else {
							label[i]=0.0;	
						}
				}
				
				out = new output();
				out.verbose=false;
				out.printsmatrix(X,label,fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train");
				label=null;
				
				//make subprocess
				 create_libffm_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train",
						 this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".mod",
						 "",
						 true);				

				 
		        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train" );
		        // tries to delete a non-existing file
		        f.delete();				
				
		        f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train.bin" );
		        // tries to delete a non-existing file
		        f.delete();	
	
	
				}
		}	

		
		
		X=null;
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

		if ( !opt.equals("order")  && !opt.equals("no_order") ){
			
	    	String splits [] = opt.split(",");
	    	for (String ele: splits){
	    		try{
	    			Integer.parseInt(ele);
	    		}catch (Exception e){
	    			throw new IllegalStateException(" if opt is not 'order' and 'no_order', it needs to have comma separated integer indices .Here it receied: " + ele  );	
	    		}
	    	}

	}		

	
	if (this.iteration<=0){
		this.iteration=1;
	}	

	if (this.lambda<0){
		this.lambda=0.00001;
	}
	if (this.factor<1){
		this.factor=1;
	}	
	if (this.learn_rate<=0){
		this.learn_rate=0.01;
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
		fields=null;
		if(this.opt.equals("order")){
			fields=Fieldfinder.get_fileds(X);
		} else if(this.opt.equals("no_order")){
			fields=Fieldfinder.get_fileds_noorder(X);
		} else {
			fields=Fieldfinder.get_fileds(this.opt,X.GetColumnDimension() );
		}
		
		output out = new output();
		out.verbose=false;

		if (n_classes==2){
			out = new output();
			out.verbose=false;
			//print.Print(fields, 12);
			out.printsmatrix(X,fstarget,fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
			X=null;			
			fstarget=null;
			
			//make subprocess
			 create_libffm_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
					 "",
					 true);
			 
	        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();			
	        f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  + ".train.bin" );
	        // tries to delete a non-existing file
	        f.delete();	
		}else {	
		
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [fstarget.length];
				for (int i=0; i < label.length; i++){
						if ( fstarget[i]==Double.parseDouble(classes[n])){
							label[i]=1.0;
						} else {
							label[i]=0.0;	
						}
				}
				
				out = new output();
				out.verbose=false;
				out.printsmatrix(X,label,fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train");
				label=null;

				
				//make subprocess
				 create_libffm_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train",
						 this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".mod",
						 "",
						 true);				

				 
		        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name   +n+   ".train" );
		        // tries to delete a non-existing file
		        f.delete();				
		        f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train.bin" );
		        // tries to delete a non-existing file
		        f.delete();	
				
	
	
				}
		}	
		
		
		
		X=null;
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

		if ( !opt.equals("order")  && !opt.equals("no_order") ){
			
	    	String splits [] = opt.split(",");
	    	for (String ele: splits){
	    		try{
	    			Integer.parseInt(ele);
	    		}catch (Exception e){
	    			throw new IllegalStateException(" if opt is not 'order' and 'no_order', it needs to have comma separated integer indices .Here it receied: " + ele  );	
	    		}
	    	}

	}		

	
	if (this.iteration<=0){
		this.iteration=1;
	}	

	if (this.lambda<0){
		this.lambda=0.00001;
	}
	if (this.factor<1){
		this.factor=1;
	}	
	if (this.learn_rate<=0){
		this.learn_rate=0.01;
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

		smatrix X= (smatrix) data.Copy();
		
		columndimension=data.GetColumnDimension();
		//generate config file
		
		//generate dataset

		fields=null;
		if(this.opt.equals("order")){
			fields=Fieldfinder.get_fileds(X);
		} else if(this.opt.equals("no_order")){
			fields=Fieldfinder.get_fileds_noorder(X);
		} else {
			fields=Fieldfinder.get_fileds(this.opt,X.GetColumnDimension() );
		}
		
		output out = new output();
		out.verbose=false;

		if (n_classes==2){
			out = new output();
			out.verbose=false;
			out.printsmatrix(X,fstarget,fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");//this.usedir +  File.separator +  "models"+File.separator + 
			X=null;			
			fstarget=null;
			
			//make subprocess
			 create_libffm_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name +"0.mod",
					 "",
					 true);
			 
	        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();	
	        f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +".train.bin" );
	        // tries to delete a non-existing file
	        f.delete();	

		}else {	
		
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [fstarget.length];
				for (int i=0; i < label.length; i++){
					if ( fstarget[i]==Double.parseDouble(classes[n])){
							label[i]=1.0;
						} else {
							label[i]=0.0;	
						}
				}
				
				out = new output();
				out.verbose=false;
				out.printsmatrix(X,label,fields, this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train");
				label=null;

				
				//make subprocess
				 create_libffm_suprocess(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train",
						 this.usedir +  File.separator +  "models"+File.separator + this.model_name +n+ ".mod",
						 "",
						 true);				

				 
		        File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train" );
		        // tries to delete a non-existing file
		        f.delete();				
		        f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name  +n+  ".train.bin" );
		        // tries to delete a non-existing file
		        f.delete();	
				
	
	
				}
		}	
		
		X=null;
		data=null;
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
		return "libffmClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: libffm Classifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Column dimension: " + columndimension);	
		
	    System.out.println("learn_rate=" + this.learn_rate  );
	    System.out.println("use_norm=" + this.use_norm );
	    System.out.println("iteration=" + this.iteration );	    
	    System.out.println("lambda=" + this.lambda );    
	    System.out.println("threads=" + this.threads );
	    System.out.println("factor=" +  this.factor );
	    System.out.println("seed=" +  this.seed );				    			    			    
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
		f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + "0.mod" );
        f.delete();  
        factor=100;
		iteration=10;
		learn_rate=0.01;
		use_norm=true;
		lambda=0.00002;
		opt="no_order";
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
				if (metric.toLowerCase().equals("lambda")) {this.lambda=Double.parseDouble(value);}				
				else if (metric.toLowerCase().equals("factor")) {this.factor=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("learn_rate")) {this.learn_rate=Double.parseDouble(value);}				
				else if (metric.toLowerCase().equals("iteration")) {this.iteration=Integer.parseInt(value);}										
				else if (metric.toLowerCase().equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("use_norm")) {this.use_norm=(value.toLowerCase().equals("true")?true:false);}				
				else if (metric.toLowerCase().equals("opt")) {this.opt=value;}					
				else if (metric.toLowerCase().equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.toLowerCase().equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}				
				
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
			  

