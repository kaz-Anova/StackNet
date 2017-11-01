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

package ml.LibFm;

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
import preprocess.scaling.maxscaler;
import exceptions.DimensionMismatchException;
import exceptions.LessThanMinimum;
import io.output;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;

/**
*<p>Wraps the original implementation of <a href="http://libfm.org/">libFM</a>, made from Steffen Rendle. The reason this implementation is made, is because internal results show that it has better performance (as in accuracy) than StackNet's internal implementation.  
*This particular instance is allowing only for classification results. libFM models are being trained via a subprocess based on the operating systems
*executing the class. <b>It is expected that files will be created and their size will vary based on the volumne of the training data.</b></p>
*
*<p> More information can be found in the following paper: Rendle, S. (2012). Factorization machines with libfm. <em> ACM Transactions on Intelligent Systems and Technology (TIST)</em>, 3(3), 57.
Chicago	.<a href="http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf">Link</a>  

*
*<p>Information about the tunable parameters can be found in the <a href="http://www.libfm.org/libfm-1.42.manual.pdf">libFM manuall</a> </p> 
*
*<p><b>VERY IMPORTANT NOTE</b> This implementation may not include all libFM features plus it actually uses a version of it <b>that had a bug(!) on purpose</b>. You can find more information 
*about why this was chosen in the following <a href="https://github.com/jfloff/pywFM">python wrapper for libFM</a>. It basically had this bug that was allowing you to get the parameters of the trained models for all training methods. These parameters are now extracted once a model has been trained and the scoring uses only these parameters (e.g. not the libFM executable).
*<p>The user is advised to use it directly from source to get its full potential.
*Also the version may not be the final and it is not certain whether it will be updated in the future as it required manual work to find all libraries and
*files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly . Additionally
*the descritpion of the parameters may not match the one in the website.
*
*<p>More information can be found in <a href="https://github.com/srendle/libfm">use libFM's repo on github</a>.
*
*<p>Don't forget to acknowledge libFM if you publish results produced with this software.
*
* 
*
 */
public class OriginalLibFMClassifier implements estimator,classifier {
	/**
	 * Type of algorithm to use. It has to be  sgd, als, mcmc. Default is mcmc
	 */
	public String Type="mcmc";
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;
	/**
	 * Regularization value for the latent features
	 */
	public double C2=1.0;	
	/**
	 * This will hold the latent features to encapsulate the 2d interactions among the variables
	 */
	private double[][] latent_features ;
	 /**
	  * Number of latent features to use. Defaults to 10
	  */
	 public int lfeatures=4;
	/**
	 * stdev for initialization of 2-way factors; default=0.1
	 */
	public double init_values=0.1;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * Maximum number of iterations.
	 */
	public int maxim_Iteration=3;
	/**
	 * learn_rate for SGD; default=0.1
	 */
	public double learn_rate=0.1;
	/**
	 * Scaler to use in case of usescale=true
	 */
	private preprocess.scaling.scaler Scaler;
	
	/**
	 * scale the dataset before use
	 */
	public boolean usescale=true;

	/**
	 * scale the copy the dataset
	 */
	public boolean copy=true;
    /**
     * seed to use
     */
	public int seed=1;
	/**
	 * Random number generator to use
	 */
	private Random random;
	/**
	 * weighst to used per row(sample)
	 */
	public double [] weights;
	/**
	 * if true, it prints stuff
	 */
	public boolean verbose=true;
	/**
	 * Target variable in double format
	 */
	public double target[];
	/**
	 * Target variable in String format
	 */	
	public String Starget[];
	/**
	 * where the coefficients are held
	 */
	private double betas[][];
	/**
	 * The cosntant value
	 */
	private double constant[];
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;}
	/**
	 * Number of classes
	 */
	private int n_classes=0;
	/**
	 * holds the name of the output model. All files produced (like predictions and model dumps) will use this as prefix.  
	 */
	private String model_name="";
	/**
	 * The directory where all files should be saved
	 */
	private String usedir="";
	/**
	 * Name of the unique classes
	 */
	private String classes[];
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
	 * Default constructor for Libfm with no data
	 */
	public OriginalLibFMClassifier(){
	
	}	
	public OriginalLibFMClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	

	public OriginalLibFMClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}

	public OriginalLibFMClassifier(smatrix data){
		
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
		* @param test : the test dataset to be used 
	    */
	   private void create_libfm_suprocess(String datset, String model, String test) {
		   
		   // create the subprocess
			try {
				 String operational_system=detectos.getOS();
				 if (!operational_system.equals("win") && !operational_system.equals("linux")&& !operational_system.equals("mac")){
					 throw new IllegalStateException(" The operational system is not identified as win, linux or mac which is required to run libfm" ); 
				 }
				 String libfm_path="lib" + File.separator + operational_system + File.separator + "libfm" + File.separator + "libfm";
				 List<String> list = new ArrayList<String>();
				 list.add(libfm_path);	
				 //String Str_build="-d "  + datset;


				 list.add("-train");
				 list.add(datset);				 
				 list.add("-test");
				 list.add(test);						 					 
				 list.add("-save_model");
				 list.add(model);					 				 
				 list.add("-task" );
				 list.add("c" );				 

				 list.add("-dim");
				 list.add(String.format("'1,1,%s'", this.lfeatures+""));
				 list.add("-method");
				 list.add(this.Type +"");					 	
				 list.add("-init_stdev");
				 list.add(this.init_values +"");				 	
				 list.add("-iter");
				 list.add(this.maxim_Iteration +"");				 
				 list.add("-learn_rate");
				 list.add(this.learn_rate +"");					 
				 list.add("-regular");
				 list.add(String.format("'%s,%s,%s'", this.C+"",this.C+"",this.C2+""));
				 list.add("-seed");
				 list.add(this.seed +"");					 
			     list.add("-verbosity");
				 list.add(this.verbose==true?1+ "":0+ "" );				 
				 
				 
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
				throw new IllegalStateException(" failed to create libfm subprocess");
			}
		   
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
	 * Retrieve the number of uniqye classes
	 */
	public int getnumber_of_classes(){
		return n_classes;
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
	 * 
	 * @return the betas
	 */
	public double [][] Getbetas(){
		if (betas==null || betas.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(betas);
	}

	/**
	 * @return the constant of the model should be length n_classes=-1
	 */
	public double [] Getcosntant(){

		return manipulate.copies.copies.Copy(constant);
	}	
	
	 /** 
	 * @return the HashMap of that holds the latent features
	 */
	public  double[] [] GetLatentFeatures(){
		
		if (latent_features==null || latent_features.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		double[][] lat= new double [this.latent_features.length][this.latent_features[0].length];
		
		for (int f=0; f < this.latent_features.length; f++){
			
			//HashMap<Integer, double[]> latent_featuress= new  HashMap<Integer, double[]>();
			for (int s=0; s <latent_features[f].length; s++){
				lat[f][s]=latent_features[f][s];
			}
		}

		return lat;
	}	
	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8617161535854392960L;


	@Override
	public double[][] predict_proba(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		//System.out.println(n_classes);
		double predictions[][]= new double [data.length][n_classes];
			
			for (int i=0; i < predictions.length; i++) {
				double sum=0.0;
				
		    	  for (int k=0; k<betas.length; k++) {
		    		    
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data[i][d], d);
		    			}else {
		    				current_fetaure=data[i][d];
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;
		    				
		    				
		    				for (int j=0; j <lfeatures; j++){
		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
		    				
		    				//end latent features loop
		    				}
		    			}
		    			//end latent features loop	
		    		}
		    		
		    		for (int j=0; j <lfeatures; j++){
		    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		    		}
		    		

		    		//calculate the final product
		    		// the final prediction
		    		double final_product =(linear_pred+productf/2.0);
		    		
		    		//convert to probability
		    		final_product= 1. / (1. + Math.exp(-final_product));
		    		predictions[i][k]=final_product;
		    		sum=sum+ predictions[i][k];
		    	  }
		    	  
		    	  if (this.n_classes==2){
		    		  predictions[i][0]=1-predictions[i][1];
		    		  sum+=predictions[i][0];
		    	  }
		    	  for (int k=0; k<this.n_classes; k++) {
		    		  predictions[i][k]=   predictions[i][k]/sum;
		    	  }
		    	  //System.out.println(Arrays.toString(predictions[i]));
	
				
		
		}
		return predictions;
	}

	@Override
	public double[][] predict_proba(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		
		//System.out.println(n_classes);
		double predictions[][]= new double [data.GetRowDimension()][n_classes];
			
			for (int i=0; i < predictions.length; i++) {
				double sum=0.0;
				
		    	  for (int k=0; k<betas.length; k++) {
		    		    
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data.GetElement(i, d), d);
		    			}else {
		    				current_fetaure=data.GetElement(i, d);
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;

		    				
		    				for (int j=0; j <lfeatures; j++){
		    					
		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
		    				//end latent features loop
		    				}
		    			}
		    			//end latent features loop	
		    		}
		    		
		    		for (int j=0; j <lfeatures; j++){
		    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		    		}
		    		

		    		//calculate the final product
		    		// the final prediction
		    		double final_product =(linear_pred+productf/2.0);
		    		
		    		//convert to probability
		    		final_product= 1. / (1. + Math.exp(-final_product));
		    		predictions[i][k]=final_product;
		    		sum=sum+ predictions[i][k];
		    	  }
		    	  
		    	  if (this.n_classes==2){
		    		  predictions[i][0]=1-predictions[i][1];
		    		  sum+=predictions[i][0];
		    	  }
		    	  for (int k=0; k<this.n_classes; k++) {
		    		  predictions[i][k]=   predictions[i][k]/sum;
		    	  }
		    	  //System.out.println(Arrays.toString(predictions[i]));
	
				
		
		}
		return predictions;
	}

	@Override
	public double[][] predict_proba(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		
		double predictions[][]= new double [data.GetRowDimension()][n_classes];
		for (int i=0; i < predictions.length; i++) {
			double sum=0.0;
			
	    	  for (int k=0; k<betas.length; k++) {
	    		    
	    		  if (this.n_classes==2){
	    			  k++;
	    		  }
	      		double sumone []= new double[lfeatures];
	    		double sumtwo []= new double[lfeatures];
	    		double productf=0.0;
	    		double linear_pred=0.0;
	    		

		        linear_pred=constant[k] ;
		    		
		        for (int s=data.indexpile[i]; s < data.indexpile[i+1]; s++){
	    			int d=data.mainelementpile[s] ;
	    			double current_fetaure=0.0;
	    			if(usescale && Scaler!=null) {
	    			 current_fetaure=Scaler.transform(data.valuespile[s], d);
	    			}else {
	    				current_fetaure=data.valuespile[s];
	    			}
	    				
	    				linear_pred+=betas[k][d]*current_fetaure;

	    				
	    				for (int j=0; j <lfeatures; j++){
	    					
	    					double value=latent_features[k][d*this.lfeatures + j];
	    					sumone[j]+=value*current_fetaure;
		    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;		
	    				//end latent features loop
	    				}
	    			
	    			//end latent features loop	
	    		}
	    		
	    		for (int j=0; j <lfeatures; j++){
	    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
	    		}
	    		

	    		//calculate the final product
	    		// the final prediction
	    		double final_product =(linear_pred+productf/2.0);
	    		
	    		//convert to probability
	    		final_product= 1. / (1. + Math.exp(-final_product));
	    		predictions[i][k]=final_product;
	    		sum=sum+ predictions[i][k];
	    	  }
	    	  
	    	  if (this.n_classes==2){
	    		  predictions[i][0]=1-predictions[i][1];
	    		  sum+=predictions[i][0];
	    	  }
	    	  for (int k=0; k<this.n_classes; k++) {
	    		  predictions[i][k]=   predictions[i][k]/sum;
	    	  }
	    	  //System.out.println(Arrays.toString(predictions[i]));

			
	
	}
	return predictions;
	}

	@Override
	public double[] predict_probaRow(double[] row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (row==null || row.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (row.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + row.length);	
		}
		
		double predictions[]= new double [n_classes];

				double sum=0.0;
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  
		    		    
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(row[d], d);
		    			}else {
		    				current_fetaure=row[d];
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;
		    				
		    				
		    				for (int j=0; j <lfeatures; j++){
		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
		    				//end latent features loop
		    				}
		    			}
		    			//end latent features loop	
		    		}
		    		
		    		for (int j=0; j <lfeatures; j++){
		    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		    		}
		    		

		    		//calculate the final product
		    		// the final prediction
		    		double final_product =(linear_pred+productf/2.0);
		    		
		    		//convert to probability
		    		final_product= 1. / (1. + Math.exp(-final_product));
		    		predictions[k]=final_product;
		    		sum=sum+ predictions[k];
		    	  }
		    	  
		    	  if (this.n_classes==2){
		    		  predictions[0]=1-predictions[1];
		    		  sum+=predictions[0];
		    	  }
		    	  for (int k=0; k<this.n_classes; k++) {
		    		  predictions[k]=   predictions[k]/sum;
		    	  }
		    	  
		
		    	  return predictions;
	}

	@Override
	public double[] predict_probaRow(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRow(rows).length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double predictions[]= new double [n_classes];
		double sum=0.0;
		
  	  for (int k=0; k<betas.length; k++) {
  		  
  		    
  		  if (this.n_classes==2){
  			  k++;
  		  }
    	double sumone []= new double[lfeatures];
  		double sumtwo []= new double[lfeatures];
  		double productf=0.0;
  		double linear_pred=0.0;
  		

	        linear_pred=constant[k] ;
	    		
  		
  		for (int d=0; d < columndimension; d++){
  			double current_fetaure=0.0;
  			if(usescale && Scaler!=null) {
  			 current_fetaure=Scaler.transform(data.GetElement(rows, d), d);
  			}else {
  				current_fetaure=data.GetElement(rows, d);
  			}
  			
  			if (current_fetaure!=0.0){
  				
  				linear_pred+=betas[k][d]*current_fetaure;
  				
  				
  				for (int j=0; j <lfeatures; j++){
  					
					double value=latent_features[k][d*this.lfeatures + j];
					sumone[j]+=value*current_fetaure;
    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
  				
  				//end latent features loop
  				}
  			}
  			//end latent features loop	
  		}
  		
  		for (int j=0; j <lfeatures; j++){
  			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
  		}
  		

  		//calculate the final product
  		// the final prediction
  		double final_product =(linear_pred+productf/2.0);
  		
  		//convert to probability
  		final_product= 1. / (1. + Math.exp(-final_product));
  		predictions[k]=final_product;
  		sum=sum+ predictions[k];
  	  }
  	  
  	  if (this.n_classes==2){
  		  predictions[0]=1-predictions[1];
  		  sum+=predictions[0];
  	  }
  	  for (int k=0; k<this.n_classes; k++) {
  		  predictions[k]=   predictions[k]/sum;
  	  }
  	  

  	  return predictions;
	}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null ){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double predictions[]= new double [n_classes];
		double sum=0.0;
		 for (int k=0; k<betas.length; k++) {
		  if (this.n_classes==2){
			  k++;
		  }

  		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
		

        linear_pred=constant[k] ;
    		
        for (int s=start; s <end; s++){
			int d=data.mainelementpile[s] ;
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data.valuespile[s], d);
			}else {
				current_fetaure=data.valuespile[s];
			}
				
				linear_pred+=betas[k][d]*current_fetaure;
				

				
				for (int j=0; j <lfeatures; j++){
					
					double value=latent_features[k][d*this.lfeatures + j];
					sumone[j]+=value*current_fetaure;
    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
				
				//end latent features loop
				}
			
			//end latent features loop	
		}
		
		for (int j=0; j <lfeatures; j++){
			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions[k]=final_product;
		sum=sum+ predictions[k];
	  }
	  
	  if (this.n_classes==2){
		  predictions[0]=1-predictions[1];
		  sum+=predictions[0];
	  }
	  
	  for (int k=0; k<this.n_classes; k++) {
		  predictions[k]=   predictions[k]/sum;
	  }
	  //System.out.println(Arrays.toString(predictions[i]));


return predictions;

	}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double predictions[]= new double [data.GetRowDimension()];
		for (int i=0; i < predictions.length; i++) {
			double sum=0.0;
			double temp[]= new double[n_classes];

	    	  for (int k=0; k<betas.length; k++) {
	    		  
	    		  if (this.n_classes==2){
	    			  k++;
	    		  }

		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data.GetElement(i, d), d);
		    			}else {
		    				current_fetaure=data.GetElement(i, d);
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;
		    				
		    				
		    				for (int j=0; j <lfeatures; j++){
		    					
		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
		    				//end latent features loop
		    				}
		    			}
		    			//end latent features loop	
		    		}
		    		
		    		for (int j=0; j <lfeatures; j++){
		    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		    		}
		    		
		    		double final_product =(linear_pred+productf/2.0);
		    		
	    	  temp[k]=1/(1+Math.exp( -final_product) );
	    	  sum=sum+ temp[k];
	    	  }
	    	  if (this.n_classes==2){
	    		  temp[0]=1-temp[1];
	    		  sum=sum+temp[0];
    			 
    		  }
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
	
	return predictions;
	}

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		
		double predictions[]= new double [data.GetRowDimension()];
			for (int i=0; i < predictions.length; i++) {
				double sum=0.0;
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		    
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
			        for (int s=data.indexpile[i]; s < data.indexpile[i+1]; s++){
		    			int d=data.mainelementpile[s] ;
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data.valuespile[s], d);
		    			}else {
		    				current_fetaure=data.valuespile[s];
		    			}
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;

		    				
		    				for (int j=0; j <lfeatures; j++){
		    					
		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
		    				//end latent features loop
		    				}
		    			
		    			//end latent features loop	
		    		}
		    		
		    		for (int j=0; j <lfeatures; j++){
		    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		    		}
		    		

		    		//calculate the final product
		    		// the final prediction
		    		double final_product =(linear_pred+productf/2.0);
		    		
		    		//convert to probability
		    		final_product= 1. / (1. + Math.exp(-final_product));
		    		temp[k]=final_product;
		    		sum=sum+ temp[k];
		    	  }
		    	  
		    	  if (this.n_classes==2){
		    		  temp[0]=1-temp[1];
		    		  sum+=temp[0];
		    	  }
		    	  
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
		
		return predictions;
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		
		double predictions[]= new double [data.length];
			for (int i=0; i < predictions.length; i++) {
				double sum=0.0;
				double temp[]= new double[n_classes];

		    	  for (int k=0; k<betas.length; k++) {
		    		  
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
			      		double sumone []= new double[lfeatures];
			    		double sumtwo []= new double[lfeatures];
			    		double productf=0.0;
			    		double linear_pred=0.0;
			    		

				        linear_pred=constant[k] ;
				    		
			    		
			    		for (int d=0; d < columndimension; d++){
			    			double current_fetaure=0.0;
			    			if(usescale && Scaler!=null) {
			    			 current_fetaure=Scaler.transform(data[i][d], d);
			    			}else {
			    				current_fetaure=data[i][d];
			    			}
			    			
			    			if (current_fetaure!=0.0){
			    				
			    				linear_pred+=betas[k][d]*current_fetaure;

			    				
			    				for (int j=0; j <lfeatures; j++){
			    					
			    					double value=latent_features[k][d*this.lfeatures + j];
			    					sumone[j]+=value*current_fetaure;
				    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
			    				//end latent features loop
			    				}
			    			}
			    			//end latent features loop	
			    		}
			    		
			    		for (int j=0; j <lfeatures; j++){
			    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
			    		}
			    		
			    		double final_product =(linear_pred+productf/2.0);
			    		
		    	  temp[k]=1/(1+Math.exp( -final_product) );
		    	  sum=sum+ temp[k];
		    	  }
		    	  if (this.n_classes==2){
		    		  temp[0]=1-temp[1];
		    		  sum=sum+temp[0];
	    			 
	    		  }
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
		
		return predictions;
	}

	@Override
	public double predict_Row(double[] row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (row==null || row.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (row.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + row.length);	
		}
		
		double predictions=0.0;
				double sum=0.0;
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
			      		double sumone []= new double[lfeatures];
			    		double sumtwo []= new double[lfeatures];
			    		double productf=0.0;
			    		double linear_pred=0.0;
			    		

				        linear_pred=constant[k] ;
				    		
			    		
			    		for (int d=0; d < columndimension; d++){
			    			double current_fetaure=0.0;
			    			if(usescale && Scaler!=null) {
			    			 current_fetaure=Scaler.transform(row[d], d);
			    			}else {
			    				current_fetaure=row[d];
			    			}
			    			
			    			if (current_fetaure!=0.0){
			    				
			    				linear_pred+=betas[k][d]*current_fetaure;
			    				
			    				for (int j=0; j <lfeatures; j++){
			    					
			    					double value=latent_features[k][d*this.lfeatures + j];
			    					sumone[j]+=value*current_fetaure;
				    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
			    				//end latent features loop
			    				}
			    			}
			    			//end latent features loop	
			    		}
			    		
			    		for (int j=0; j <lfeatures; j++){
			    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
			    		}
			    		
			    		double final_product =(linear_pred+productf/2.0);
			    		
		    	  temp[k]=1/(1+Math.exp( -final_product) );
		    	  sum=sum+ temp[k];
		    	  }
		    	  if (this.n_classes==2){
		    		  temp[0]=1-temp[1];
		    		  sum=sum+temp[0];
	    			 
	    		  
		    	  }
		    	  int maxi=0;
		    	  double max=temp[0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (temp[k]>max){
		    			 max=temp[k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions=maxi;
		    	  }
	
		
		return predictions;
	}

	@Override
	public double predict_Row(fsmatrix f, int row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
		
		double predictions=0.0;
				double sum=0.0;
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
			      		double sumone []= new double[lfeatures];
			    		double sumtwo []= new double[lfeatures];
			    		double productf=0.0;
			    		double linear_pred=0.0;
			    		

				        linear_pred=constant[k] ;
				    		
			    		
			    		for (int d=0; d < columndimension; d++){
			    			double current_fetaure=0.0;
			    			if(usescale && Scaler!=null) {
			    			 current_fetaure=Scaler.transform(f.GetElement(row, d), d);
			    			}else {
			    				current_fetaure=f.GetElement(row, d);
			    			}
			    			
			    			if (current_fetaure!=0.0){
			    				
			    				linear_pred+=betas[k][d]*current_fetaure;
			    				
			    				
			    				for (int j=0; j <lfeatures; j++){
			    					
			    					double value=latent_features[k][d*this.lfeatures + j];
			    					sumone[j]+=value*current_fetaure;
				    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
			    				//end latent features loop
			    				}
			    			}
			    			//end latent features loop	
			    		}
			    		
			    		for (int j=0; j <lfeatures; j++){
			    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
			    		}
			    		
			    		double final_product =(linear_pred+productf/2.0);
			    		
		    	  temp[k]=1/(1+Math.exp( -final_product) );
		    	  sum=sum+ temp[k];
		    	  }
		    	  if (this.n_classes==2){
		    		  temp[0]=1-temp[1];
		    		  sum=sum+temp[0];
	    			 
	    		  
		    	  }
		    	  int maxi=0;
		    	  double max=temp[0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (temp[k]>max){
		    			 max=temp[k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions=maxi;
		    	  }
	
		
		return predictions;
	}
	

	@Override
	public double predict_Row(smatrix f, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || betas==null || betas.length<=0 || n_classes<2) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
		
		double predictions=0.0;

				double sum=0.0;
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

		            linear_pred=constant[k] ;
		        		
		            for (int s=start; s <end; s++){
		    			int d=f.mainelementpile[s] ;
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(f.valuespile[s], d);
		    			}else {
		    				current_fetaure=f.valuespile[s];
		    			}
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;

		    				
		    				for (int j=0; j <lfeatures; j++){
		    					
		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
		    				//end latent features loop
		    				}
		    			
		    			//end latent features loop	
		    		}
		    		
		    		for (int j=0; j <lfeatures; j++){
		    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		    		}
		    		

		    		//calculate the final product
		    		// the final prediction
		    		double final_product =(linear_pred+productf/2.0);
		    		
		    		//convert to probability
		    		final_product= 1. / (1. + Math.exp(-final_product));
		    		temp[k]=final_product;
		    		sum=sum+ temp[k];
		    	  }
		    	  
		    	  if (this.n_classes==2){
		    		  temp[0]=1-temp[1];
		    		  sum+=temp[0];
		    	  }
		    	  

		    	  int maxi=0;
		    	  double max=temp[0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (temp[k]>max){
		    			 max=temp[k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions=maxi;
		    	  }
	
	
		return predictions;
	}

	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
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
		if (C<0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (C2<0){
			throw new IllegalStateException(" The regularization Value C2 for the latent features needs to be higher than zero" );
		}		
		if (this.init_values<=0){
			this.init_values=0.1; // a high value just in case id cannot converge
		}	
		
		if (this.lfeatures<=0){
			throw new IllegalStateException(" Number of Latent features has to be higher than 0. You may use linear models instead." );	
		}		
		if (  !Type.equals("sgd") && !Type.equals("mcmc")  && !Type.equals("als") ){
			throw new IllegalStateException(" Type has to be sgd, mcmc or  als" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10;
		}
		
		if (this.learn_rate<=0){
			this.learn_rate=0.1; 
		}

		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.length) && (Starget==null || Starget.length!=data.length) ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else {
			if (target!=null && (classes==null ||  classes.length<=1) ){
				
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
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    }
			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);				
				
			}
		}
		if (weights==null) {
			weights=new double [data.length];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
		} else {
			if (weights.length!=data.length){
				throw new DimensionMismatchException(weights.length,data.length);
			}
		}

		//hard copy
		if (copy){
			data= manipulate.copies.copies.Copy(data);
		}
		n_classes=classes.length;

		//Initialise column dimension
		columndimension=data[0].length;
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		latent_features = new double [this.n_classes][this.lfeatures*this.columndimension] ;

		smatrix X= new smatrix(data);
		
		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale ){
			Scaler.fit(X);
			X=Scaler.transform(X);
		}
		
		output out = new output();
		out.verbose=false;
		if (X.GetRowDimension()>100){
			int ros []= new int[100];
			double minitar []= new double [100];
			for (int s=0; s <100; s++){
				ros[s]=s;
				minitar[s]=1.;
			}
		
			out.printsmatrix(X.makesubmatrix(ros),minitar ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
		} else {
			out.printsmatrix(X,target ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
		}		
		if (n_classes==2){
			double label []= new double [data.length];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( target[i]==Double.parseDouble(classes[1]) ){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}	

			//System.out.println(X.GetColumnDimension() + X.GetRowDimension());
			out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");

			X=null;
			label=null;
			System.gc();
			
			//make subprocess
			 create_libfm_suprocess( this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
			 
			libfm_filreader lbs= new libfm_filreader();
			String fi=this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod";
			lbs.load_libfm(fi, this.lfeatures);
			double Betas[]= lbs.Getbetas();
			double Latents[]= lbs.GetLatentFeatures();		
			double Cont=lbs.Getcosntant(); 
			
			for (int g=0; g <betas[0].length; g++ ){
				betas[1][g]=Betas[g];
				betas[0][g]=-betas[1][g];							
				for (int f=0; f <this.lfeatures;f++){	
					latent_features[1][g*this.lfeatures+f]=Latents[g*this.lfeatures+f];
					latent_features[0][g*this.lfeatures+f]=-latent_features[1][g*this.lfeatures+f];
				}			
			}
			constant[1]= Cont;
			constant[0]= -constant[1];			
			Betas=null;
			Latents=null;
	        // create new file
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();
	        //f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
	        // tries to delete a non-existing file
	        //f.delete();
			f = new File( this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
	        // tries to delete a non-existing file
	        f.delete();	
			
			
		}else {	
		
		for (int n=0; n <n_classes; n++ ){
			double label []= new double [data.length];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( target[i]==Double.parseDouble(classes[n]) ){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}

			out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");

			
			label=null;
			System.gc();
			
			//make subprocess
			 create_libfm_suprocess( this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
			 
			libfm_filreader lbs= new libfm_filreader();
			String fi=this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod";
			lbs.load_libfm(fi, this.lfeatures);
			double Betas[]= lbs.Getbetas();
			double Latents[]= lbs.GetLatentFeatures();		
			double Cont=lbs.Getcosntant(); 
			
			for (int g=0; g <betas[0].length; g++ ){
				betas[n][g]=Betas[g];						
				for (int f=0; f <this.lfeatures;f++){	
					latent_features[n][g*this.lfeatures+f]=Latents[g*this.lfeatures+f];
				}			
			}
			constant[n]= Cont;		
			Betas=null;
			Latents=null;
	        // create new file
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();
	        //f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
	        // tries to delete a non-existing file
	        //f.delete();
			f = new File( this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
	        // tries to delete a non-existing file
	        f.delete();			

		}		
		
		}
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        // tries to delete a non-existing file
        f.delete();
        X=null;
		sdataset=null;
		fsdataset=null;
		dataset=null;
		System.gc();
		
	}

	@Override
	public void fit(fsmatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
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
		if (C<0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (C2<0){
			throw new IllegalStateException(" The regularization Value C2 for the latent features needs to be higher than zero" );
		}		
		if (this.init_values<=0){
			this.init_values=0.1; // a high value just in case id cannot converge
		}	
		
		if (this.lfeatures<=0){
			throw new IllegalStateException(" Number of Latent features has to be higher than 0. You may use linear models instead." );	
		}		
		if (  !Type.equals("sgd") && !Type.equals("mcmc")  && !Type.equals("als") ){
			throw new IllegalStateException(" Type has to be sgd, mcmc or  als" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10;
		}
		
		if (this.learn_rate<=0){
			this.learn_rate=0.1; 
		}

		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else {
			if (target!=null && (classes==null ||  classes.length<=1) ){
				
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
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    }
			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);				
				
			}
		}
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
		}

		//hard copy
		if (copy){
			data= (fsmatrix) data.Copy();
		}

		
		n_classes=classes.length;
		//System.out.println(Arrays.toString(classes));
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		latent_features = new double [this.n_classes][this.lfeatures*this.columndimension] ;

		smatrix X= new smatrix(data);
		
		
		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale ){
			Scaler.fit(X);
			X=Scaler.transform(X);
			
		}
		output out = new output();
		out.verbose=false;
		if (X.GetRowDimension()>100){
			int ros []= new int[100];
			double minitar []= new double [100];
			for (int s=0; s <100; s++){
				ros[s]=s;
				minitar[s]=1.;
			}
		
			out.printsmatrix(X.makesubmatrix(ros),minitar ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
		} else {
			out.printsmatrix(X,target ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
		}		
		if (n_classes==2){
			
			double label []= new double [data.GetRowDimension()];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					
					if ( target[i]==Double.parseDouble(classes[1]) ){
						label[i]=1.0;
						
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}	

			//System.out.println(X.GetColumnDimension() + X.GetRowDimension());
			out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");

			X=null;
			label=null;
			System.gc();
			
			//make subprocess
			 create_libfm_suprocess( this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
			 
			libfm_filreader lbs= new libfm_filreader();
			String fi=this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod";
			lbs.load_libfm(fi, this.lfeatures);
			double Betas[]= lbs.Getbetas();
			double Latents[]= lbs.GetLatentFeatures();		
			double Cont=lbs.Getcosntant(); 
			
			for (int g=0; g <betas[0].length; g++ ){
				betas[1][g]=Betas[g];
				betas[0][g]=-betas[1][g];							
				for (int f=0; f <this.lfeatures;f++){	
					latent_features[1][g*this.lfeatures+f]=Latents[g*this.lfeatures+f];
					latent_features[0][g*this.lfeatures+f]=-latent_features[1][g*this.lfeatures+f];
				}			
			}
			constant[1]= Cont;
			constant[0]= -constant[1];			
			Betas=null;
			Latents=null;
	        // create new file
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();
	        //f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
	        // tries to delete a non-existing file
	        //f.delete();
			f = new File( this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
	        // tries to delete a non-existing file
	        f.delete();	
			
			
		}else {	
		
		for (int n=0; n <n_classes; n++ ){
			double label []= new double [data.GetRowDimension()];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( target[i]==Double.parseDouble(classes[n]) ){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}

			out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");

			
			label=null;
			System.gc();
			
			//make subprocess
			 create_libfm_suprocess( this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
					 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
			 
			libfm_filreader lbs= new libfm_filreader();
			String fi=this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod";
			lbs.load_libfm(fi, this.lfeatures);
			double Betas[]= lbs.Getbetas();
			double Latents[]= lbs.GetLatentFeatures();		
			double Cont=lbs.Getcosntant(); 
			
			for (int g=0; g <betas[0].length; g++ ){
				betas[n][g]=Betas[g];						
				for (int f=0; f <this.lfeatures;f++){	
					latent_features[n][g*this.lfeatures+f]=Latents[g*this.lfeatures+f];
				}			
			}
			constant[n]= Cont;		
			Betas=null;
			Latents=null;
	        // create new file
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
	        // tries to delete a non-existing file
	        f.delete();
	        //f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
	        // tries to delete a non-existing file
	        //f.delete();
			f = new File( this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
	        // tries to delete a non-existing file
	        f.delete();			

		}		
		
		}
		File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
        // tries to delete a non-existing file
        f.delete();
        X=null;
		sdataset=null;
		fsdataset=null;
		dataset=null;
		System.gc();
		
		
	}

	@Override
	public void fit(smatrix data) {
		// make sensible checks

		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
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
		if (C<0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (C2<0){
			throw new IllegalStateException(" The regularization Value C2 for the latent features needs to be higher than zero" );
		}		
		if (this.init_values<=0){
			this.init_values=0.1; // a high value just in case id cannot converge
		}	
		
		if (this.lfeatures<=0){
			throw new IllegalStateException(" Number of Latent features has to be higher than 0. You may use linear models instead." );	
		}		
		if (  !Type.equals("sgd") && !Type.equals("mcmc")  && !Type.equals("als") ){
			throw new IllegalStateException(" Type has to be sgd, mcmc or  als" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10;
		}
		
		if (this.learn_rate<=0){
			this.learn_rate=0.1; 
		}

		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}		
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else {
			if (target!=null && (classes==null ||  classes.length<=1) ){
				
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
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    }
			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);				
				
			}
		}
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
		}

		//hard copy
		if (copy){
			data= (smatrix) data.Copy();
		}		
		 
		if (!data.IsSortedByRow()){
			data.convert_type();
			}

		
		if (Scaler==null){
			Scaler = new maxscaler();
			
		}
		smatrix X;
		if (usescale ){
			Scaler.fit(data);
			 X=Scaler.transform(data);
			
		} else {
			X=(smatrix) data.Copy();
		}
		
			n_classes=classes.length;
			//System.out.println(Arrays.toString(classes));
			//initialize column dimension
			columndimension=X.GetColumnDimension();
			//initialise beta and constant
			betas= new double[n_classes][columndimension];
			constant=new double[n_classes];
			latent_features = new double [this.n_classes][this.lfeatures*this.columndimension] ;

			output out = new output();
			out.verbose=false;
			if (X.GetRowDimension()>100){
				int ros []= new int[100];
				double minitar []= new double [100];
				for (int s=0; s <100; s++){
					ros[s]=s;
					minitar[s]=1.;
				}
			
				out.printsmatrix(X.makesubmatrix(ros),minitar ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
			} else {
				out.printsmatrix(X,target ,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
			}		
			if (n_classes==2){
				double label []= new double [X.GetRowDimension()];
				for (int i=0; i < label.length; i++){
					if (target!=null){
						if ( target[i]==Double.parseDouble(classes[1]) ){
							label[i]=1.0;
						} else {
							label[i]=-1.0;	
						}
					} else {
						if ( (Starget[i]).equals(classes[1])){
							label[i]=1.0;
						} else {
							label[i]=-1.0;	
						}
					}
				}	

				//System.out.println(X.GetColumnDimension() + X.GetRowDimension());
				out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");

				//X=null;
				label=null;
				System.gc();
				
				//make subprocess
				 create_libfm_suprocess( this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
						 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
						 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
				 
				libfm_filreader lbs= new libfm_filreader();
				String fi=this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod";
				lbs.load_libfm(fi, this.lfeatures);
				double Betas[]= lbs.Getbetas();
				double Latents[]= lbs.GetLatentFeatures();		
				double Cont=lbs.Getcosntant(); 
				
				for (int g=0; g <betas[0].length; g++ ){
					betas[1][g]=Betas[g];
					betas[0][g]=-betas[1][g];							
					for (int f=0; f <this.lfeatures;f++){	
						latent_features[1][g*this.lfeatures+f]=Latents[g*this.lfeatures+f];
						latent_features[0][g*this.lfeatures+f]=-latent_features[1][g*this.lfeatures+f];
					}			
				}
				constant[1]= Cont;
				constant[0]= -constant[1];			
				Betas=null;
				Latents=null;

		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
		        // tries to delete a non-existing file
		        f.delete();
		        //f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		        // tries to delete a non-existing file
		        //f.delete();
				f = new File( this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
		        // tries to delete a non-existing file
		        f.delete();	
				
				
			}else {	
			
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [X.GetRowDimension()];
				for (int i=0; i < label.length; i++){
					if (target!=null){
						if ( target[i]==Double.parseDouble(classes[n]) ){
							label[i]=1.0;
						} else {
							label[i]=-1.0;	
						}
					} else {
						if ( (Starget[i]).equals(classes[n])){
							label[i]=1.0;
						} else {
							label[i]=-1.0;	
						}
					}
				}

				out.printsmatrix(X, label,this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train");

				//X=null;
				label=null;
				System.gc();
				
				//make subprocess
				 create_libfm_suprocess( this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train",
						 this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod",
						 this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test");
				 
				libfm_filreader lbs= new libfm_filreader();
				String fi=this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod";
				lbs.load_libfm(fi, this.lfeatures);
				double Betas[]= lbs.Getbetas();
				double Latents[]= lbs.GetLatentFeatures();		
				double Cont=lbs.Getcosntant(); 
				
				for (int g=0; g <betas[0].length; g++ ){
					betas[n][g]=Betas[g];						
					for (int f=0; f <this.lfeatures;f++){	
						latent_features[n][g*this.lfeatures+f]=Latents[g*this.lfeatures+f];
					}			
				}
				constant[n]= Cont;		
				Betas=null;
				Latents=null;
		        // create new file
				File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".train" );
		        // tries to delete a non-existing file
		        f.delete();
		        //f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
		        // tries to delete a non-existing file
		        //f.delete();
				f = new File( this.usedir +  File.separator +  "models"+File.separator + this.model_name +".mod");
		        // tries to delete a non-existing file
		        f.delete();			

			}		
			
			}
			File f = new File(this.usedir +  File.separator +  "models"+File.separator + this.model_name + ".test" );
	        // tries to delete a non-existing file
	        f.delete();
	        X=null;
	        data=null;
			sdataset=null;
			fsdataset=null;
			dataset=null;
			System.gc();
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
		return "OriginalLibFMClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier:  Original LibFM Classifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Regularization L2 value for latent features : "+ this.C2);			
		System.out.println("Number of Latent features: "+ this.lfeatures);				
		System.out.println("Initial value range of the latent features: "+ this.init_values);	
		System.out.println("Regularization value: "+ this.C);				
		System.out.println("Training method: "+ this.Type);	
		System.out.println("Maximum Iterations: "+ maxim_Iteration);
		System.out.println("Learning Rate: "+ this.learn_rate);	
		System.out.println("use scaling: "+ this.usescale);	
		
		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		boolean ist=false;
		if (betas==null){
			System.out.println("Trained: False");	
		} else {
			for (int j=0; j<betas[0].length;j++ ){
				if (betas[0][1]!=0.0){
					System.out.println("Trained: True");
					ist=true;
					break;
				}
			}
			if (ist==false){
				System.out.println("Trained: False");		
			}
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
		if (betas!=null || betas.length>0){
			return true;
		} else {
		return false;
		}
	}

	@Override
	public boolean IsRegressor() {
		return false;
	}

	@Override
	public boolean IsClassifier() {
		return true;
	}

	@Override
	public void reset() {
		constant=null;
		betas=null;
		latent_features=null;
		n_classes=0;
		classes=null;
		C2=0.01;
		lfeatures=4;
		init_values=0.1;
		C=0.001;
		Type="mcmc";
		threads=1;
		maxim_Iteration=-1;
		columndimension=0;
		learn_rate=1.0;
		usescale=false;
		Scaler=null;
		copy=true;
		seed=1;
		random=null;
		target=null;
		weights=null;
		verbose=true;
		
	}

	@Override
	public estimator copy() {
		
		OriginalLibFMClassifier br = new OriginalLibFMClassifier();
		br.constant=manipulate.copies.copies.Copy(this.constant);
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		//hard copy of the latent features
		br.latent_features = new double [this.n_classes][this.lfeatures*this.columndimension] ;
		for (int f=0; f<this.n_classes;f++){
			for (int j=0; j < this.latent_features.length; j++){
				br.latent_features[f][j]=this.latent_features[f][j];
			}
			
		}

		br.classes=this.classes.clone();
		br.n_classes=this.n_classes;
		
		
		br.C2=this.C2;		
		br.lfeatures=this.lfeatures;
		br.init_values=this.init_values;
		br.C=this.C;
		br.usescale=this.usescale;
		br.columndimension=this.columndimension;
		br.Type=this.Type;
		br.threads=this.threads;
		br.maxim_Iteration=this.maxim_Iteration;
		br.learn_rate=this.learn_rate;
		br.Scaler=this.Scaler;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.Starget=this.Starget.clone();		
		br.weights=manipulate.copies.copies.Copy(this.weights.clone());
		br.verbose=this.verbose;
		return br;
	}
	
	@Override	
	public void set_params(String params){
		
		String splitted_params []=params.split(" " + "+");
		
		for (int j=0; j<splitted_params.length; j++ ){
			String mini_split []=splitted_params[j].split(":");
			if (mini_split.length>=2){
				String metric=mini_split[0];
				String value=mini_split[1];
				
				if (metric.toLowerCase().equals("c")) {this.C=Double.parseDouble(value);}
				else if (metric.toLowerCase().equals("c2")) {this.C2=Double.parseDouble(value);}
				else if (metric.toLowerCase().equals("type")) {this.Type=value;}
				else if (metric.equals("usescale")) {this.usescale=(value.toLowerCase().equals("true")?true:false);}				
				else if (metric.toLowerCase().equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("maxim_iteration")) {this.maxim_Iteration=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("lfeatures")) {this.lfeatures=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("init_values")) {this.init_values=Double.parseDouble(value);}
				else if (metric.toLowerCase().equals("learn_rate")) {this.learn_rate=Double.parseDouble(value);}
				else if (metric.toLowerCase().equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.toLowerCase().equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.toLowerCase().equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}
				
			}
			
		}
		

	}
	
	@Override
	public scaler ReturnScaler() {
		return this.Scaler;
	}
	
	@Override
	public void setScaler(scaler sc) {
		this.Scaler=sc;
		
	}
	@Override
	public void setSeed(int seed) {
		this.seed=seed;}	
	
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
}
