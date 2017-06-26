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

package ml.LinearRegression;

import java.util.Random;

import preprocess.scaling.scaler;
import preprocess.scaling.maxscaler;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;
/**
 * 
 * <p>This class will perform Linear Least square (or MAE or QUANTILE) regression for multiple targets. Sparse Input is also possible.</p>
 * <p> In Linear OLS Regression as an optimization problem we are trying to minimise the 
 * sum of squared difference between the real value the prediction of a Y value 
 * when we know a number of other characteristics or predictors labelled as x. 
 * This deviation (real value-prediction) is also called <em> residual </em>.
 *  The equation to minimise is :
 * <pre> Min(f)=Ó(Y<sub>i</sub>-y<sub>i</sub>)<sup>2</sup>
Where Y is the real value of the variable we are trying to predict
and y is the prediction</pre>
<p> The class involves solving the ridge problem (adding l2 regularization) as well as
the l1 one using FTRL. </p>
<p>MAE and QUANTILE solvers are also included .  </p>
 */
public class LinearRegression implements estimator,regressor {
	
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;
	/**
	 * Regularization value for l1 "Follow The Regularized Leader"
	 */
	public double l1C=1.0;		
	/**
	 * quantile value
	 */
	public double tau=0.5;
	/**
	 * Value that helps when dividing with past gradients (to avoid zero divisions)
	 */
	public double smooth=0.01;
	/**
	 * Type of algorithm to use. It has to be one of Routine, SGD, FTRL
	 */
	public String Type="Routine";
	/**
	 * The objective to optimise . It may be RMSE (which is for classic linear-least squares regression)
	 *  or MAE for Mean Absolute Error or QUANTILE 
	 */

	public String Objective="RMSE";	
	/**
	 * True if we want to scale with highest maximum value
	 */
	public boolean scale=false;
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
	 * Number of target-variable columns. The name is left as n_classes(same as classification for consistency)
	 */
	private int n_classes=0;
	
	
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
	public LinearRegression(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public LinearRegression(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public LinearRegression(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public LinearRegression(smatrix data){
		
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
	 * Retrieve the number of target variables
	 */
	public int getnumber_of_targets(){
		return n_classes;
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
	private static final long serialVersionUID = -8617161535154392960L;

	
	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					} 
		if (  n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
			
		}
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}

		
		double predictions[]= new double [data.GetRowDimension()];
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[0][j]*Scaler.transform(data.GetElement(i, j), j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[0][j]*data.GetElement(i, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=value  ;
		}
		
		}
		return predictions;
	}

	@Override
	public double[] predict(smatrix data) {
		
		if ( betas==null || betas.length<=0 || n_classes<1) {
			throw new IllegalStateException("The fit method needs to be run successfully in " +
								"order to create the logic before attempting scoring a new set");
				} 
		if ( n_classes>1) {
		System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
		
		}
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [data.GetRowDimension()];
		// check if data is sorted via row
		if (!data.IsSortedByRow()){
			data.convert_type();
		}


		
		if(usescale && Scaler!=null) {

			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
					value+=betas[0][data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value ;
			}

		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[0][data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=value ;
		}
		}

		return predictions;
	}

	@Override
	public double [] predict(double[][] data) {
		
		if ( betas==null || betas.length<=0 || n_classes<1) {
			throw new IllegalStateException("The fit method needs to be run successfully in " +
								"order to create the logic before attempting scoring a new set");
				} 
		if ( n_classes>1) {
		System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
		
		}		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double predictions[]= new double [data.length];
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[0][j]*Scaler.transform(data[i][j], j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[0][j]*data[i][j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=value  ;
		}
		
		}
		return predictions;
	}
	
	
	@Override
	public double predict_Row(double[] data) {
		
		if ( betas==null || betas.length<=0 || n_classes<1) {
			throw new IllegalStateException("The fit method needs to be run successfully in " +
								"order to create the logic before attempting scoring a new set");
				} 
//		if (  n_classes>1) {
//		//System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//		
//		}		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}
		double predictions=0.0;
		if(usescale && Scaler!=null) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[0][j]*Scaler.transform(data[j], j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions=value  ;
		} else {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[0][j]*data[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=value;
		
		
		}
		return predictions;
	}

	@Override
	public double predict_Row(fsmatrix data, int row) {
		
		if ( betas==null || betas.length<=0 || n_classes<1) {
			throw new IllegalStateException("The fit method needs to be run successfully in " +
								"order to create the logic before attempting scoring a new set");
				} 
//		if ( betas==null || betas.length<=0 || n_classes>1) {
//		//System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//		
//		}		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions=0.0;
		if(usescale && Scaler!=null) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[0][j]*Scaler.transform(data.GetElement(row, j), j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions=value;

			
		} else {

			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[0][j]*data.GetElement(row, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=value ;
		
		}
		return predictions;
	}

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		
		if ( betas==null || betas.length<=0 || n_classes<1) {
			throw new IllegalStateException("The fit method needs to be run successfully in " +
								"order to create the logic before attempting scoring a new set");
				} 
//		if ( betas==null || betas.length<=0 || n_classes>1) {
//		//System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//		
//		}		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions= 0.0;
		
		if(usescale && Scaler!=null) {

				double value=constant[0];
				for (int j=start; j < end ; j++){
					value+=betas[0][data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);;
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions=value ;		


		} else {
			
			double value=constant[0];
			for (int j=start; j < end ; j++){
				value+=betas[0][data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=value ;
		}
		return predictions;
	}
	
	
	
	@Override
	public double[][] predict2d(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
		    	  }
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  double value=Scaler.transform(data.GetElement(i, j),j);
		    		  //int column= j;
		    		  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]+=value*betas[k][j]; 
		    		  }

		    	  }
			}
		} else {
		for (int i=0; i < predictions.length; i++) {



		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];
		    	  }
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  double value=data.GetElement(i, j);
		    		  //int column= j;
		    		  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]+=value*betas[k][j]; 
		    		  }

		    	  }

			}
		
		}
		return predictions;
	}
	
	@Override
	public double[][] predict2d(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
		    	  }
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int b=data.indexpile[i]; b<data.indexpile[i+1]; b++) {
		    		  int j=data.mainelementpile[b];
		    		  double value=Scaler.transform(data.valuespile[b],j);
		    		  //int column= j;
		    		  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]+=value*betas[k][j]; 
		    		  }

		    	  }
			}
		} else {
		for (int i=0; i < predictions.length; i++) {



		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];
		    	  }
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int b=data.indexpile[i]; b<data.indexpile[i+1]; b++) {
		    		  int j=data.mainelementpile[b];
		    		  double value=data.valuespile[b];
		    		  //int column= j;
		    		  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]+=value*betas[k][j]; 
		    		  }

		    	  }

			}
		
		}
		return predictions;
	}

	
	
	
	@Override
	public double[][] predict2d(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (betas==null || betas.length<=0 || n_classes<1) {
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
		    	  }
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  double value=Scaler.transform(data[i][j],j);
		    		  //int column= j;
		    		  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]+=value*betas[k][j]; 
		    		  }

		    	  }
			}
		} else {
		for (int i=0; i < predictions.length; i++) {



		    	  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]= constant[k];
		    	  }
		    	 /*
		    	  * loop through all predictors
		    	  */
		    	  for (int j=0; j<columndimension; j++) {
		    		  double value=data[i][j];
		    		  //int column= j;
		    		  for (int k=0; k<betas.length; k++) {
		    		  predictions[i][k]+=value*betas[k][j]; 
		    		  }

		    	  }

			}
		
		}
		return predictions;
	}
	
	
	@Override
	public double[] predict_Row2d(double[] row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
	    	  }
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int j=0; j<columndimension; j++) {
	    		  double value=Scaler.transform(row[j],j);
	    		  //int column= j;
	    		  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]+=value*betas[k][j]; 
	    		  }

	    	  }

		} else {

	    	  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]= constant[k];
	    	  }
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int j=0; j<columndimension; j++) {
	    		  double value=row[j];
	    		  //int column= j;
	    		  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]+=value*betas[k][j]; 
	    		  }

	    	  }
		
		}
		return predictions;
	}
	
	@Override
	public double[] predict_Row2d(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
	    	  }
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int j=0; j<columndimension; j++) {
	    		  double value=Scaler.transform(data.GetElement(rows, j),j);
	    		  //int column= j;
	    		  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]+=value*betas[k][j]; 
	    		  }

	    	  }
			
		} else {
	    	  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]= constant[k];
	    	  }
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int j=0; j<columndimension; j++) {
	    		  double value=data.GetElement(rows, j);
	    		  //int column= j;
	    		  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]+=value*betas[k][j]; 
	    		  }

	    	  }
		
		}
		return predictions;
	}
	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (betas==null || betas.length<=0 || n_classes<1) {
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
	    	  }
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int b=start; b<end; b++) {
	    		  int j=data.mainelementpile[b];
	    		  double value=Scaler.transform(data.valuespile[b],j);
	    		  //int column= j;
	    		  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]+=value*betas[k][j]; 
	    		  }

	    	  }
	
			
		} else {

	    	  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]= constant[k];
	    	  }
	    	 /*
	    	  * loop through all predictors
	    	  */
	    	  for (int b=start; b<end; b++) {
	    		  int j=data.mainelementpile[b];
	    		  double value=data.valuespile[b];
	    		  //int column= j;
	    		  for (int k=0; k<betas.length; k++) {
	    		  predictions[k]+=value*betas[k][j]; 
	    		  }

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
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}			
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
		if ( !Type.equals("Routine")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Routine methods" );	
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
		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
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
		//System.out.println(Arrays.toString(classes));
		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		double [][] constants=new double[n_classes][1];
		Thread[] thread_array= new Thread[threads];
		
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
			

		
			singleLinearRegression svc = new singleLinearRegression(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.tau=this.tau;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.l1C=this.l1C;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			svc.weights=this.weights;

			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}			
			
			svc.SetBetas(betas[0], constants[0]);
			svc.target=label;
			svc.run();
			constant[0]= constants[0][0];

		}else {
		
		int count_of_live_threads=0;
		int class_passed=0;
		
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
					  starget.convert_type();;
				    }

			        
			        

		            for (int i=starget.indexpile[n]; i <starget.indexpile[n+1]; i++) {
		                double val = starget.valuespile[i];
		                int ind =starget.mainelementpile[i];
		                label[ind]=val;
		            }
				    
				    
			} else {
				throw new IllegalStateException(" A target array needs to be provided" );
			}

			singleLinearRegression svc = new singleLinearRegression(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.tau=this.tau;
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
			svc.weights=this.weights;
			svc.SetBetas(betas[n], constants[n]);
			svc.target=label;
			thread_array[count_of_live_threads]= new Thread(svc);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for target: " + class_passed);
							
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
		
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}	
		if ( this.Objective.equals("QUANTILE") && (this.tau<=0 || this.tau>=1) )  {
			throw new IllegalStateException("For  QUANTILE tau value needs to be in (0,1)" );	
		}			
		
		if ( !Type.equals("Routine")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Routine methods" );	
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

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
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
		//System.out.println(Arrays.toString(classes));
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		double [][] constants=new double[n_classes][1];
		Thread[] thread_array= new Thread[threads];
		
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
			

		
			singleLinearRegression svc = new singleLinearRegression(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.tau=this.tau;
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
			svc.weights=this.weights;
			svc.SetBetas(betas[0], constants[0]);
			svc.target=label;
			svc.run();
			constant[0]= constants[0][0];

		}else {
		
		int count_of_live_threads=0;
		int class_passed=0;
		
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

			singleLinearRegression svc = new singleLinearRegression(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.UseConstant=this.UseConstant;
			svc.verbose=false;
			svc.tau=this.tau;
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
			svc.weights=this.weights;
			svc.SetBetas(betas[n], constants[n]);
			svc.target=label;
			thread_array[count_of_live_threads]= new Thread(svc);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for target: " + class_passed);
							
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
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}			
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
		if ( !Type.equals("Routine")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Routine methods" );	
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

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
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
			//initialise beta and constant
			betas= new double[n_classes][columndimension];
			constant=new double[n_classes];
			double [][] constants=new double[n_classes][1];
			Thread[] thread_array= new Thread[threads];
			
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
				

			
				singleLinearRegression svc = new singleLinearRegression(data);
				svc.smooth=this.smooth;
				svc.Type=this.Type;
				svc.maxim_Iteration=this.maxim_Iteration;
				svc.Objective=this.Objective;
				svc.UseConstant=this.UseConstant;
				svc.verbose=false;
				svc.tau=this.tau;
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
				svc.weights=this.weights;
				svc.SetBetas(betas[0], constants[0]);
				svc.target=label;
				svc.run();
				constant[0]= constants[0][0];

			}else {
			
			int count_of_live_threads=0;
			int class_passed=0;
			
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

				singleLinearRegression svc = new singleLinearRegression(data);
				svc.smooth=this.smooth;
				svc.Type=this.Type;
				svc.maxim_Iteration=this.maxim_Iteration;
				svc.Objective=this.Objective;
				svc.UseConstant=this.UseConstant;
				svc.verbose=false;
				svc.tau=this.tau;
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
				svc.weights=this.weights;
				svc.SetBetas(betas[n], constants[n]);
				svc.target=label;
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(n_classes)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("fitting for target: " + class_passed);
								
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
	
		return "regressor";
	}

	@Override
	public boolean SupportsWeights() {
		return true;
	}

	@Override
	public String GetName() {
		return "LinearRegression";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor:  Regularized Linear Regression");
		System.out.println("Targets: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);				
		System.out.println("Constant in the model: "+ this.UseConstant);
		System.out.println("Objective: "+ this.Objective);
		System.out.println("tau: "+ this.tau);
		System.out.println("smooth: "+ this.smooth);
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
		return true ;
	}

	@Override
	public boolean IsClassifier() {
		return false;
	}

	@Override
	public void reset() {
		constant=null;
		betas=null;
		n_classes=0;
		tau=0.5;
		Objective="RMSE";
		C=1.0;
		l1C=1.0;
		Type="Routine";
		smooth=0.1;
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
		shuffle=true;
		learn_rate=1.0;
		Scaler=null;
		columndimension=0;
		copy=true;
		seed=1;
		random=null;
		tolerance=0.0001; 
		target=null;
		target2d=null;
		fstarget=null;
		starget=null;
		weights=null;
		verbose=true;
		
	}

	@Override
	public estimator copy() {
		LinearRegression br = new LinearRegression();
		br.constant=manipulate.copies.copies.Copy(this.constant);
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		br.n_classes=this.n_classes;
		br.C=this.C;
		br.l1C=this.l1C;
		br.tau= this.tau;
		br.smooth=this.smooth;
		br.Objective=this.Objective;
		br.Type=this.Type;
		br.threads=this.threads;
		br.UseConstant=this.UseConstant;
		br.maxim_Iteration=this.maxim_Iteration;
		br.columndimension=this.columndimension;
		br.usescale=this.usescale;
		br.shuffle=this.shuffle;
		br.learn_rate=this.learn_rate;
		br.Scaler=this.Scaler;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.tolerance=this.tolerance; 
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.target2d=manipulate.copies.copies.Copy(this.target2d.clone());	
		br.fstarget=(fsmatrix) this.fstarget.Copy();
		br.starget=(smatrix) this.starget.Copy();
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
				else if (metric.equals("tau")) {this.tau=Double.parseDouble(value);}				
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("Objective")) {this.Objective=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("UseConstant")) {this.UseConstant=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.equals("maxim_Iteration")) {this.maxim_Iteration=Integer.parseInt(value);}
				else if (metric.equals("smooth")) {this.smooth=Double.parseDouble(value);}
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
