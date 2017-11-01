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

package ml.LSVC;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import preprocess.scaling.maxscaler;
import preprocess.scaling.scaler;
import exceptions.DimensionMismatchException;
import exceptions.LessThanMinimum;
import matrix.fsmatrix;
import matrix.smatrix;
import misc.print;
import ml.classifier;
import ml.estimator;
/**
 * 
 * Linear Support vector machine class runnable with 3 different optimisation methods:
 * <p> for multiclass problems , K binary models are built, where k= number of distinct classes,
  apart from when K=2 when only 1 binary model is built </p>
 * <ol>
 * <li> LibLinear (L2 and L1 regularization) - R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.<a href="http://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf">LIBLINEAR: A library for large linear classification</a> Journal of Machine Learning Research 9(2008), 1871-1874.</li> 
 * <li> SGD "Stochastic Gradient Descent" with adaptive learning Rate (supports L1 and L2)  </li> 
 * <li> FTRL"Follow The Regularized Leader" (supports L1 and L2), inspired by Tingru's code in Kaggel forums <a href="https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory"> link </a>  </li> 
 * </ol>
 * <p> we optimize for the hinge loss: </p>
 * <pre> max(0, 1-y<sub>i</sub> beta X) </pre>
 * <p> The Sub gradients are computed as : </p>
 * <pre> {-y<sub>i</sub>x<sub>i</sub> + lbeta<sub>i</sub>  if y<sub>i</sub> beta X < 1 }</pre> 
 * <pre> {0 + lbeta<sub>i</sub>  if y<sub>i</sub> beta X >= 1 } </pre> 
 * (Collobert et al , 2001)
 */
public class LSVC implements estimator,classifier {
	/**
	 * Type of regularization,can be any of L2, L1 or anything else for none
	 */
	public String RegularizationType="L2";
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;
	/**
	 * Regularization value for l1 "Follow The Regularized Leader"
	 */
	public double l1C=1.0;		
	/**
	 * Type of algorithm to use. It has to be one of Liblinear, SGD, FTRL =Follow The Regularized Leader
	 */
	public String Type="Liblinear";
	/**
	 * True if we want to scale with highest maximum value
	 */
	public boolean scale=false;
	/**
	 * True if we want to optimise for a quadratic error
	 */
	public boolean quadratic=false;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * If we want to use constant in the model
	 */
	public boolean UseConstant=true;
	/**
	 * Maximum number of iterations. -1 for "until convergence"
	 */
	public int maxim_Iteration=-1;
	/**
	 * scale the dataset before use
	 */
	public boolean usescale=true;
	/**
	 * for sgd only
	 */
	public boolean shuffle=true;
	/**
	 * for SGD
	 */
	public double learn_rate=1.0;
	/**
	 * Scaler to use in case of usescale=true
	 */
	private preprocess.scaling.scaler Scaler;
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
	 * minimal change in coefficients' values after nth iterations to stop the algorithm
	 */
	public double tolerance=0.0001; 
	
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
		return columndimension;
	}
	/**
	 * Number of classes
	 */
	private int n_classes=0;
	
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
	 * Default constructor 
	 */
	public LSVC(){
	
	}	
	/**
	 * Default constructor for LSVC with double[][] data
	 */
	public LSVC(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LSVC with fsmatrix data
	 */
	public LSVC(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LSVC with smatrix data
	 */
	public LSVC(smatrix data){
		
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
	 * default Serial id
	 */
	private static final long serialVersionUID = -8617161535854392960L;


	@Override
	/**
	 * Is not really probabilities but that is left as it is for consistency with other classifiers. 
	 * This is the linear score (e.g. beta * x) for each class
	 */
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
		if(usescale && Scaler!=null) {
			
			for (int i=0; i < predictions.length; i++) {
		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];  
		    		
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  predictions[i][k]+=Scaler.transform(data[i][j], j)*betas[k][j]; 
		    		  }

		    	  }

		    	  //System.out.println(Arrays.toString(predictions[i]));
	
			}
		} else {
		for (int i=0; i < predictions.length; i++) {
		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  predictions[i][k]+=data[i][j]*betas[k][j]; 
		    		  }
		    	  }
			}
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
		
		double predictions[][]= new double [data.GetRowDimension()][n_classes];
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {

		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  predictions[i][k]+=Scaler.transform(data.GetElement(i, j), j)*betas[k][j]; 
		    		  }

		    	  }
			}
		} else {
		for (int i=0; i < predictions.length; i++) {


		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  predictions[i][k]+=data.GetElement(i, j)*betas[k][j]; 
		    		  }

		    	  }
	
			}
		
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
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    		  for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
		    		  predictions[i][k]+=Scaler.transform(data.valuespile[j], data.mainelementpile[j])*betas[k][data.mainelementpile[j]]; 
		    		  }
		    	  }
	
			}
		} else {
			for (int i=0; i < predictions.length; i++) {

		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    		  for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
		    		  predictions[i][k]+=data.valuespile[j]*betas[k][data.mainelementpile[j]]; 
		    		  }

		    	  }
	
			}
		
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
		if(usescale && Scaler!=null) {

		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  predictions[k]+=Scaler.transform(row[j], j)*betas[k][j]; 
		    		  }

		    	  }

		} else {

	    	  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]= constant[k];  
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int j=0; j<columndimension; j++) {
	    		  predictions[k]+=row[j]*betas[k][j]; 
	    		  }

	    	  }
		
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
		if(usescale && Scaler!=null) {

		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  predictions[k]+=Scaler.transform(data.GetElement(rows, j), j)*betas[k][j]; 
		    		  }

		    	  }
			
		} else {
	    	  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]= constant[k];  
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int j=0; j<columndimension; j++) {
	    		  predictions[k]+=data.GetElement(rows, j)*betas[k][j]; 
	    		  }

	    	  }
		
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
		if(usescale && Scaler!=null) {
		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    		  for (int j=start; j < end; j++){
		    		  predictions[k]+=Scaler.transform(data.valuespile[j], data.mainelementpile[j])*betas[k][data.mainelementpile[j]]; 
		    		  }

		    	  }
	
			
		} else {

	    	  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]= constant[k];  
	    	 /*
	    	  * loop through all predictors
	    	  */
	    		  for (int j=start; j < end; j++){
	    		  predictions[k]+=data.mainelementpile[j]*betas[k][data.mainelementpile[j]]; 
 
	    		  }

	    	  }
		
		}
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
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=Scaler.transform(data.GetElement(i, j), j)*betas[k][j]; 
		    		  }

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
		} else {
			for (int i=0; i < predictions.length; i++) {
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=data.GetElement(i, j)*betas[k][j]; 
		    		  }
		    	  }
		    	  int maxi=0;
		    	  double max=temp[0];
		    	  for (int k=0; k<n_classes; k++) {
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
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++) {
		    		  temp[k]+=Scaler.transform(data.valuespile[j], data.mainelementpile[j])*betas[k][data.mainelementpile[j]]; 
		    		  }
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
		} else {
			for (int i=0; i < predictions.length; i++) {
				double temp[]= new double[n_classes];

		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    		  for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++) {
		    		  temp[k]+=data.valuespile[j]*betas[k][data.mainelementpile[j]]; 
		    		  }
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
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=Scaler.transform(data[i][j], j)*betas[k][j]; 
		    		  }
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
		} else {
			for (int i=0; i < predictions.length; i++) {
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=data[i][j]*betas[k][j]; 
		    		  }
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
		if(usescale && Scaler!=null) {
				double temp[]= new double[n_classes];				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=Scaler.transform(row[j], j)*betas[k][j]; 
		    		  }
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
	
		} else {

				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=row[j]*betas[k][j]; 
		    		  }
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
		if(usescale && Scaler!=null) {
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=Scaler.transform(f.GetElement(row,j), j)*betas[k][j]; 
		    		  }
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
	
		} else {

				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  temp[k]+=f.GetElement(row,j)*betas[k][j]; 
		    		  }
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
		if(usescale && Scaler!=null) {
				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=start; j<end; j++) {
		    		  temp[k]+=Scaler.transform(f.valuespile[j], f.mainelementpile[j])*betas[k][f.mainelementpile[j]]; 
		    		  }
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
	
		} else {

				double temp[]= new double[n_classes];
				
		    	  for (int k=0; k<betas.length; k++) {
		    		  temp[k]= constant[k];  
		    	 /*
		    	  * loop through all predictors
		    	  */
		    		  for (int j=start; j<end; j++) {
		    		  temp[k]+=f.valuespile[j]*betas[k][f.mainelementpile[j]]; 
		    		  }
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
	

		
		}
		return predictions;
	}

	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if ( !this.RegularizationType.equals("L2") &&  !this.RegularizationType.equals("L1") &&Type.equals("Liblinear") ){
			throw new IllegalStateException(" No regularization is supported by SGD and Routine methods" );	
		}
		if ( !Type.equals("Liblinear")   && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD,FTRL or Liblinear methods" );	
		}
		if ( !Type.equals("LibLinear")  && quadratic==true ){
			throw new IllegalStateException(" quadratic loss is only available for liblinear" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case it cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
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
		// Initialise scaler
		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		n_classes=classes.length;
		//System.out.println(Arrays.toString(classes));
		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		double [][] constants=new double[n_classes][1];
		Thread[] thread_array= new Thread[threads];
		
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
			
			binaryLSVC svc = new binaryLSVC(data);
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.RegularizationType=this.RegularizationType;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.quadratic=this.quadratic;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.l1C=this.l1C;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}
			svc.SetBetas(betas[1], constants[1]);
			svc.target=label;
			svc.run();
			for (int g=0; g <betas[0].length; g++ ){
				betas[0][g]=-betas[1][g];
			}
			constant[1]= constants[1][0];
			constant[0]= -constant[1];
		}else {
		
		int count_of_live_threads=0;
		int class_passed=0;
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

			binaryLSVC svc = new binaryLSVC(data);
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.RegularizationType=this.RegularizationType;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.quadratic=this.quadratic;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.l1C=this.l1C;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}
			svc.SetBetas(betas[n], constants[n]);
			svc.target=label;
			thread_array[count_of_live_threads]= new Thread(svc);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for class: " + classes[class_passed]);
							
						}
						thread_array[s].join();
					} catch (InterruptedException e) {
					   System.out.println(e.getMessage());
					   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
					}
					class_passed++;
				}
				class_passed-=count_of_live_threads;
			
				
				for (int j=0; j < count_of_live_threads; j++){
					constant[class_passed]= constants[class_passed][0];
					class_passed++;
				}
				count_of_live_threads=0;
			}
		}		
		
		}
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
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if ( !this.RegularizationType.equals("L2") &&  !this.RegularizationType.equals("L1") &&Type.equals("Liblinear") ){
			throw new IllegalStateException(" No regularization is supported by SGD and Routine methods" );	
		}
		if ( !Type.equals("Liblinear")   && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD,FTRL or Liblinear methods" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case it cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
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
		// Initialise scaler


		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		n_classes=classes.length;
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		double [][] constants=new double[n_classes][1];
		Thread[] thread_array= new Thread[threads];
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
			
			binaryLSVC svc = new binaryLSVC(data);
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.RegularizationType=this.RegularizationType;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.quadratic=this.quadratic;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.l1C=this.l1C;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}
			svc.SetBetas(betas[1], constants[1]);
			svc.target=label;
			svc.run();
			for (int g=0; g <betas[0].length; g++ ){
				betas[0][g]=-betas[1][g];
			}
			constant[1]= constants[1][0];
			constant[0]= -constant[1];
		}else {
			
		int count_of_live_threads=0;
		int class_passed=0;
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
			
			binaryLSVC svc = new binaryLSVC(data);
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.RegularizationType=this.RegularizationType;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.quadratic=this.quadratic;
			svc.l1C=this.l1C;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}
			svc.SetBetas(betas[n], constants[n]);
			svc.target=label;
			thread_array[count_of_live_threads]= new Thread(svc);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for class: " + classes[class_passed]);
						
						}
						thread_array[s].join();
					} catch (InterruptedException e) {
					   System.out.println(e.getMessage());
					   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
					}
					class_passed++;
				}
				class_passed-=count_of_live_threads;
			
				
				for (int j=0; j < count_of_live_threads; j++){
					constant[class_passed]= constants[class_passed][0];
					class_passed++;
				}
				count_of_live_threads=0;
			}
		}		
		}

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
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if ( !this.RegularizationType.equals("L2") &&  !this.RegularizationType.equals("L1") &&Type.equals("Liblinear") ){
			throw new IllegalStateException(" No regularization is supported by SGD and Routine methods" );	
		}
		if ( !Type.equals("Liblinear")   && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD,FTRL or Liblinear methods" );	
		}
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case it cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
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
		// Initialise scaler
		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		
		if (this.RegularizationType.equals("L1") && Type.equals("Liblinear")){
			
			if (!data.IsSortedByColumn()){
			data.convert_type();
			}		
			
		} else {
			if (!data.IsSortedByRow()){
			data.convert_type();
			}

		}
		
		
		n_classes=classes.length;
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		double [][] constants=new double[n_classes][1];
		Thread[] thread_array= new Thread[threads];
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
			
			binaryLSVC svc = new binaryLSVC(data);
			svc.set_sparse_indicator(true);
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.RegularizationType=this.RegularizationType;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.quadratic=this.quadratic;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.l1C=this.l1C;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}
			svc.SetBetas(betas[1], constants[1]);
			svc.target=label;
			svc.run();
			for (int g=0; g <betas[0].length; g++ ){
				betas[0][g]=-betas[1][g];
			}
			constant[1]= constants[1][0];
			constant[0]= -constant[1];
		}else {		
		int count_of_live_threads=0;
		int class_passed=0;
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
			binaryLSVC svc = new binaryLSVC(data);
			svc.set_sparse_indicator(true);
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.RegularizationType=this.RegularizationType;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.copy=false;
			svc.usescale=false;
			svc.quadratic=this.quadratic;
			svc.C=this.C;
			svc.l1C=this.l1C;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}
			svc.SetBetas(betas[n], constants[n]);
			svc.target=label;
			thread_array[count_of_live_threads]= new Thread(svc);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for class: " + classes[class_passed]);
							
						}
						thread_array[s].join();
					} catch (InterruptedException e) {
					   System.out.println(e.getMessage());
					   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
					}
					class_passed++;
				}
				class_passed-=count_of_live_threads;
			
				
				for (int j=0; j < count_of_live_threads; j++){
					constant[class_passed]= constants[class_passed][0];
					class_passed++;
				}
				count_of_live_threads=0;
			}
		}		
		
		}
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
		return "LSVC";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier:  Regularized Linear Support Vector Machine");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);				
		System.out.println("Constant in the model: "+ this.UseConstant);
		System.out.println("Regularization type: "+ this.RegularizationType);
		System.out.println("Regularization value: "+ this.C);		
		System.out.println("Regularization L1 for FTLR: "+ this.l1C);			
		System.out.println("Training method: "+ this.Type);	
		System.out.println("Maximum Iterations: "+ maxim_Iteration);
		System.out.println("Learning Rate: "+ this.learn_rate);	
		System.out.println("used Scaling: "+ this.usescale);			
		System.out.println("Tolerance: "+ tolerance);
		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (betas==null){
			System.out.println("Trained: False");	
		} else {
			System.out.println("Trained: True");
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
		n_classes=0;
		classes=null;
		RegularizationType="L2";
		C=1.0;
		l1C=1.0;
		Type="Liblinear";
		quadratic=false;
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
		columndimension=0;
		shuffle=true;
		learn_rate=1.0;
		Scaler=null;
		copy=true;
		seed=1;
		random=null;
		tolerance=0.0001; 
		target=null;
		weights=null;
		verbose=true;
		
	}

	@Override
	public estimator copy() {
		LSVC br = new LSVC();
		br.constant=manipulate.copies.copies.Copy(this.constant);
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		br.classes=this.classes.clone();
		br.n_classes=this.n_classes;
		br.RegularizationType=this.RegularizationType;
		br.C=this.C;
		br.l1C=this.l1C;
		br.columndimension=this.columndimension;
		br.Type=this.Type;
		br.quadratic=this.quadratic;
		br.threads=this.threads;
		br.UseConstant=this.UseConstant;
		br.maxim_Iteration=this.maxim_Iteration;
		br.usescale=this.usescale;
		br.shuffle=this.shuffle;
		br.learn_rate=this.learn_rate;
		br.Scaler=this.Scaler;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.tolerance=this.tolerance; 
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
				
				if (metric.equals("C")) {this.C=Double.parseDouble(value);}
				else if (metric.equals("l1C")) {this.l1C=Double.parseDouble(value);}				
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("RegularizationType")) {this.RegularizationType=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("UseConstant")) {this.UseConstant=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.equals("maxim_Iteration")) {this.maxim_Iteration=Integer.parseInt(value);}
				else if (metric.equals("usescale")) {this.usescale=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("shuffle")) {this.shuffle=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("learn_rate")) {this.learn_rate=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("tolerance ")) {this.tolerance =Double.parseDouble(value);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}		
				
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
