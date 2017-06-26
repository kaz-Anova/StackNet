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

package ml.LSVR;

import java.util.HashSet;
import java.util.Random;

import preprocess.scaling.maxscaler;
import preprocess.scaling.scaler;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;


/**
<p>This class will perform Linear Support Vector Regression for single target with Simple or quadratic loss and will include the following implementations:</p>
 * <ol>
 * <li> LibLinear (L2 )</li> 
 * <li> SGD "Stochastic Gradient Descent" with adaptive learning Rate (supports L2,-loss)  </li> 
 * <li> FTRL"Follow The Regularized Leader" (supports L1 and L2), inspired by Tingru's code in Kaggle forums <a href="https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory"> link </a>  </li> 
 * </ol> 

 * The Liblinear implementation is almost equivalent with the Liblinear package with a few adjustments.
 * For simplicity, the c (regularization power) has been set the same for upper and lower bounds.
 * See Algorithm 3 of Algorithm 4 of Ho and Lin, 2012
--------------------------------------------------------------------------------------------------
   /**
     * A coordinate descent algorithm for
     * L1-loss and L2-loss epsilon-SVR dual problem
     *
     *  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
     *    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
     *
     *  where Qij = xi^T xj and
     *  D is a diagonal matrix
     *
     * In L1-SVM case:
     *         upper_bound_i = C
     *         lambda_i = 0
     * In L2-SVM case:
     *         upper_bound_i = INF
     *         lambda_i = 1/(2*C)
     *
     * Given:
     * x, y, p, C
     * eps is the stopping tolerance
     * (taken from the liblnear implementation)
     */

public class SinlgeLSVR implements estimator,regressor,Runnable {

	private static final long serialVersionUID = 83052972388893394L;
	/**
	 * Private method for when this class is used in a multinomial context to avoid re-sorting each time (for each class)
	 */
	private boolean sparse_set=false;
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;
	/**
	 * Regularization value for l1 "Follow The Regularized Leader"
	 */
	public double l1C=1.0;	
	/**
	 * Value that helps when dividing with past gradients (to avoid zero divisions)
	 */
	public double smooth=0.01;
	/**
	 * Type of algorithm to use. It has to be one of Liblinear, SGD, FTRL
	 */
	public String Type="Liblinear";
	/**
	 * The objective to optimise . It may be L2 for squared loss
	 *  or L1 for normal Linear Loss
	 */
	public String Objective="L2";	

	/**
	 * quantile value
	 */
	public double P=0.1;
	
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
	 * minimal change in coefficients' values after nth iterations to stop the algorithm
	 */
	public double tolerance=0.0001; 
	/**
	 * double target use , an array of 0 and 1
	 */
	public double  [] target;
	/**
	 * weighst to used per row(sample)
	 */
	public double [] weights;
	/**
	 * if true, it prints stuff
	 */
	public boolean verbose=true;
	/**
	 * where the coefficients are held
	 */
	private double betas[];
	/**
	 * The cosntant value
	 */
	private double[] constant;
	
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
	
	
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
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
	 * Default constructor for Binary Logistic Regression
	 */
	public SinlgeLSVR(){}
	
	/**
	 * 
	 * @param need_sort : Whether we want to avoid sorting (used internally for multinomial models)
	 */
	public void set_sparse_indicator(boolean need_sort){
		this.sparse_set=true;
	}
	/**
	 * Default constructor for Binary Logistic Regression with double data
	 */
	public SinlgeLSVR(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	/**
	 * Default constructor for Binary Logistic Regression with fsmatrix data
	 */
	public SinlgeLSVR(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for Binary Logistic Regression with smatrix data
	 */
	public SinlgeLSVR(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
	}	
	/**
	 * 
	 * @param Betas : Sets initial beta-coefficients' array
	 * @param intercept : Sets initial intercept
	 */
	public void SetBetas(double Betas [], double intercept []){
		betas= Betas;
		constant=intercept;
	}
	/**
	 * 
	 * @return the betas
	 */
	public double [] Getbetas(){
		if (betas==null || betas.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(betas);
	}

	/**
	 * @return the constant of the model
	 */
	public double  Getcosntant(){

		return constant[0];
	}	

	
	@Override
	public double[] predict(fsmatrix data) {
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
					value+=betas[j]*Scaler.transform(data.GetElement(i, j), j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=value  ;
		}
		
		}
		return predictions;
	}

	@Override
	public double[] predict(smatrix data) {
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
					value+=betas[data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value ;
			}


		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=value ;
		}
		}

		return predictions;
	}

	@Override
	public double [] predict(double[][] data) {
		
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
					value+=betas[j]*Scaler.transform(data[i][j], j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=value  ;
		}
		
		}
		return predictions;
	}
	
	@Override
	public double[][] predict2d(fsmatrix data) {
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][1];
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data.GetElement(i, j), j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][0]=value ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i][0]=value  ;
		}
		
		}
		return predictions;
	}

	@Override
	public double[][] predict2d(smatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][1];
		// check if data is sorted via row
		if (!data.IsSortedByRow()){
			data.convert_type();
		}


		
		if(usescale && Scaler!=null) {

			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
					value+=betas[data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][0]=value ;
			}


		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i][0]=value ;
		}
		}

		return predictions;
	}

	@Override
	public double[][] predict2d(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double predictions[][]= new double [data.length][1];
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data[i][j], j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][0]=value ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i][0]=value  ;
		}
		
		}
		return predictions;
	}

	@Override
	public double[] predict_Row2d(double[] data) {
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}
		double predictions[]=new double [1];
		if(usescale && Scaler!=null) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data[j], j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[0]=value  ;
		} else {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[0]=value;
		
		
		}
		return predictions;
	}

	@Override
	public double[] predict_Row2d(fsmatrix data, int row) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]=new double [1];
		if(usescale && Scaler!=null) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data.GetElement(row, j), j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[0]=value;

			
		} else {

			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(row, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[0]=value ;
		
		}
		return predictions;
	}

	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [1];
		
		if(usescale && Scaler!=null) {

				double value=constant[0];
				for (int j=start; j < end ; j++){
					value+=betas[data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);;
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions[0]=value ;		


		} else {
			
			double value=constant[0];
			for (int j=start; j < end ; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[0]=value ;
		}
		return predictions;
	}

	@Override
	public double predict_Row(double[] data) {
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
					value+=betas[j]*Scaler.transform(data[j], j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions=value  ;
		} else {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=value;
		
		
		}
		return predictions;
	}

	@Override
	public double predict_Row(fsmatrix data, int row) {
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
					value+=betas[j]*Scaler.transform(data.GetElement(row, j), j);
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions=value;

			
		} else {

			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(row, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=value ;
		
		}
		return predictions;
	}

	@Override
	public double predict_Row(smatrix data, int start, int end) {
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
					value+=betas[data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);;
				}
				//value= 1. / (1. + Math.exp(-value));
				predictions=value ;		


		} else {
			
			double value=constant[0];
			for (int j=start; j < end ; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=value ;
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
		if (P<=0){
			throw new IllegalStateException(" The Epsilon P value cannot be less equal to zero" );
		}		
		if ( !this.Objective.equals("L1") && !this.Objective.equals("L2"))  {
			throw new IllegalStateException("the objective has to be one of L2 or L1 " );	
		}			
		if ( !Type.equals("Liblinear")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD,FTRL or Liblinear methods" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		// make sensible checks on the target data
		if (target==null || target.length!=data.length){
			throw new IllegalStateException(" target array needs to be provided" );
		} else {
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more than 1 different values!" );	
			}
			has=null;
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
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			if (betas.length!=columndimension){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  columndimension);
			}
			
		
		} else { //Initialise beta if not given
			betas= new double[columndimension];
			constant= new double[]{0.0};
			
		}
		
		if (Type.equals("Liblinear")){
			
			int l = data.length;

	        double p =P ;
	        int w_size = columndimension;
	        double eps = tolerance;
	        int i, s, iter = 0;
	        int max_iter = maxim_Iteration;
	        int active_size = l;
	        int[] index = new int[l];

	        double d, G, H;
	        double Gmax_old = Double.POSITIVE_INFINITY;
	        double Gmax_new, Gnorm1_new;
	        double Gnorm1_init = 0; // initialize to 0 to get rid of Eclipse warning/error
	        double[] betass = new double[l];
	        double[] QD = new double[l];
	        double val=0.0;
	        // L2R_L2LOSS_SVR_DUAL
	        double[] lambda = new double[] {0.5 / C};
	        double[] upper_bound = new double[] {Double.POSITIVE_INFINITY};

	        if (this.Objective.equals("L1")) {
	            lambda[0] = 0;
	            upper_bound[0] = C;
	        }

	        // Initial beta can be set here. Note that
	        // -upper_bound <= beta[i] <= upper_bound
	        for (i = 0; i < l; i++)
	        	betass[i] = 0;
	        
	        for (i = 0; i < l; i++) {
	            QD[i] = 0;
				if (UseConstant){
					QD[i] +=1;
					constant[0]+= betass[i];
				}
	            for (int j=0; j < w_size; j++) {
	                val= data[i][j] ;
	                if (val==0.0){
	                	continue;
	                }
	            	if (usescale){
	            		val=Scaler.transform(val, j);
	            	}
	            			
	                QD[i] +=val *  val;
	                betas[j] += betass[i] * val;
	            }

	            index[i] = i;
	        }


	        while (iter < max_iter) {
	            Gmax_new = 0;
	            Gnorm1_new = 0;

	            for (i = 0; i < active_size; i++) {
	                int j = i + random.nextInt(active_size - i);
	                swap(index, i, j);
	            }

	            for (s = 0; s < active_size; s++) {
	                i = index[s];
	                G = -target[i] + lambda[GETI_SVR(i)] * betass[i];
	                H = QD[i] + lambda[GETI_SVR(i)];

					if (UseConstant){
						 G +=  constant[0];
					}
					
	                for (int j=0; j < w_size; j++) {
	                    int ind = j;
		                val= data[i][j] ;
		                if (val==0.0){
		                	continue;
		                }
		            	if (usescale){
		            		val=Scaler.transform(val, ind);
		            	}
	                    G += val * betas[ind];
	                }

	                double Gp = G + p;
	                double Gn = G - p;
	                double violation = 0;
	                if (betass[i] == 0) {
	                    if (Gp < 0)
	                        violation = -Gp;
	                    else if (Gn > 0)
	                        violation = Gn;
	                    else if (Gp > Gmax_old && Gn < -Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] >= upper_bound[GETI_SVR(i)]) {
	                    if (Gp > 0)
	                        violation = Gp;
	                    else if (Gp < -Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] <= -upper_bound[GETI_SVR(i)]) {
	                    if (Gn < 0)
	                        violation = -Gn;
	                    else if (Gn > Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] > 0)
	                    violation = Math.abs(Gp);
	                else
	                    violation = Math.abs(Gn);

	                Gmax_new = Math.max(Gmax_new, violation);
	                Gnorm1_new += violation;

	                // obtain Newton direction d
	                if (Gp < H * betass[i])
	                    d = -Gp / H;
	                else if (Gn > H * betass[i])
	                    d = -Gn / H;
	                else
	                    d = -betass[i];

	                if (Math.abs(d) < 1.0e-12) continue;

	                double beta_old = betass[i];
	                betass[i] = Math.min(Math.max(betass[i] + d, -upper_bound[GETI_SVR(i)]), upper_bound[GETI_SVR(i)]);
	                d = betass[i] - beta_old;

	                if (d != 0) {
						if (UseConstant){
							  constant[0]+= d;
						}
	                	for (int j=0; j < w_size; j++) {
	    	                val= data[i][j] ;
	    	                if (val==0.0){
	    	                	continue;
	    	                }
	    	            	if (usescale){
	    	            		val=Scaler.transform(val, j);
	    	            	}
	                        betas[j] += d * val;
	                    }
	                }
	            }

	            if (iter == 0) Gnorm1_init = Gnorm1_new;
	            
	            if (verbose ){
	           System.out.println("Iteration: " + iter);
	            }
	            
	            iter++;
	            
	            if (Gnorm1_new <= eps * Gnorm1_init) {
	                if (active_size == l)
	                    break;
	                else {
	                    active_size = l;

	                    Gmax_old = Double.POSITIVE_INFINITY;
	                    continue;
	                }
	            }

	            Gmax_old = Gmax_new;
	        }

	      }else if (Type.equals("SGD")){
	        	
				if (this.learn_rate<=0.00000000000001){
					throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
				}
				if (this.smooth<0.0){
					throw new IllegalStateException(" smooth value cannot be less  than 0");
				}
				double iteration_tol=Double.POSITIVE_INFINITY;
				int it=0;
				double nc=0;
				//initiali
				// random number generator
				random= new Random();
				random.setSeed(seed);
				double n []= new double[data[0].length]; // sum of squared gradients
				double feature=0.0;
				// iterative algorithms start here
				while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

			    for (int k=0; k < data.length; k++){
			    	int i=k;

			    	if (this.shuffle){
			    	 i=random.nextInt(data.length);
			    	}
			    	double pred=constant[0];
			    	double yi=target[i];
		
			    	// compute score
			    		
			    	for (int j=0; j < data[i].length; j++){
		                feature= data[i][j] ;
		                if (feature==0.0){
		                	continue;
		                }
		    				if (usescale){
		    					feature=Scaler.transform(feature, j);
		    				}    
				    		pred+=feature*betas[j];

			    		
			    	}
			    	
			    	double residual=(pred-yi);
			    	if (Math.abs(residual)>=this.P){
				    	// we update constant gradient
				    	 if (UseConstant){
				    		 double gradient=0 ;
				    		 if (this.Objective.equals("L2")){
				    			 gradient=residual;
				    		 } else if  (this.Objective.equals("L1")){
				    			 if (residual>0){
				    				 gradient=1;
				    			 } else if (residual<0){
				    				 gradient=-1;
				    			 }
				    		 }
				    	     gradient+=C*constant[0];
				    		 double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);
				    		 nc+=gradient*gradient;
				    		 constant[0]=constant[0]-move; 
				    	 }
				    	 
				    	 for (int j=0; j < data[i].length;j++){
				                feature= data[i][j] ;
				                if (feature==0.0){
				                	continue;
				                }
				    				if (usescale){
				    					feature=Scaler.transform(feature, j);
				    				}   
						    	 double gradient=0.0;
						    	 
						    	 if (this.Objective.equals("L2")){
					    			 gradient=residual*feature;
					    		 } else if  (this.Objective.equals("L1")){
					    			 if (residual>0){
					    				 gradient=feature;
					    			 } else if (residual<0){
					    				 gradient=-feature;
					    			 }
					    		 }    		 
				    		 
				    			 gradient+=C*betas[j];
	
					    		 double move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
					    		 n[j]+=gradient*gradient;
					    		 betas[j]=betas[j]-move;
	
				    		 	
				    	 }
			    
			    	}
			    	
			    }
	             
		           //end of while
		            it++; 
		    		if (verbose){
		    					System.out.println("iteration: " + it);
		    				}
				}
				
				// end of SGD			
			// end of SGD
		} else if (Type.equals("FTRL")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.l1C<=0){
				throw new IllegalStateException(" L1C regularizer cannot be less equal to zero");
			}	
			if (this.smooth<0.0){
				throw new IllegalStateException(" smooth value cannot be less  than 0");
			}	
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[columndimension]; // sum of squared gradients

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.length; k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.length);
		    	}
		    	//System.out.println(Arrays.toString(data[i]));
		    	
				double BETA [] = new double[columndimension];
		    	double pred=0.0;
		    	double Constant=0.0;
			 
		    	// if we use constant
		    	if (UseConstant){
		    		
		    		double sign=1;
		    		
		    		if (constant[0]<0){
		    			sign=-1;
		    		}
				    	 if (sign * constant[0]  <= l1C){				    		 
				    		 Constant=0 ;				    		 
				    	 } else {				    		 
				    		 Constant= (sign * l1C -constant[0]) / ((+ this.smooth + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	}
		    	
		    	// other features
			    for (int j=0; j < columndimension; j++){	
	                feature= data[i][j] ;
	                if (feature==0.0){
	                	continue;
	                }
	    				if (usescale){
	    					feature=Scaler.transform(feature, j);
	    				} 	
	    				
			    	double sign=1.0;	
			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / ((  this.smooth+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    
			    pred+= BETA[j]*feature;	
			    
			    	
			    	
			    }			     
			    
			    
		    	double yi=target[i];
			    double residual= (pred-yi);
			    
			   if ( Math.abs(residual)>=this.P){
					if (UseConstant){	
						double gradient=0.0;
						 if (this.Objective.equals("L2")){
								 gradient=residual;
							 
			    		 } else if  (this.Objective.equals("L1")){
				    			 if (residual>0){
				    				 gradient=1;
				    			 } else if (residual<0){
				    				 gradient=-1;
				    			 }
			    			 
			    		 }
						 
						double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;
	
						constant[0]+=gradient-move*Constant;
						nc+=gradient*gradient;
					 }
				    //updates
					//print.Print(betas, 5);
				    for (int j=0; j < columndimension; j++){
		                feature= data[i][j] ;
		                if (feature==0.0){
		                	continue;
		                }
		    				if (usescale){
		    					feature=Scaler.transform(feature, j);
		    				} 
				    		double gradientx=0.0;
				    		
					    	 if (this.Objective.equals("L2")){
									 gradientx=residual*feature;
								 
					    		
				    		 } else if  (this.Objective.equals("L1")){
					    			 if (residual>0){
					    				 gradientx=feature;
					    			 } else if (residual<0){
					    				 gradientx=-feature;
					    			 }
				    			 
				    		 }	 	 
				    		
				    	//System.out.println(" gradient: " + gradientx);
				    	double move=(Math.sqrt(n[j] + gradientx * gradientx) - Math.sqrt(n[j])) / this.learn_rate;
				    	betas[j] += gradientx - move * BETA[j];
	                    n[j] += gradientx * gradientx;	
	
	                    // check for highest tolerance
	      				 if (Math.abs(BETA[j])>iteration_tol){
								 iteration_tol= Math.abs(BETA[j]);
							 }
				    
				    }
			    
			   } 
			  	
		    }	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			// reset coeffs
	    	if (UseConstant){
	    		
	    		double sign=1;
	    		
	    		if (constant[0]<0){
	    			sign=-1;
	    		}
			    	 if (sign * constant[0]  <= l1C){				    		 
			    		 constant[0]=0 ;				    		 
			    	 } else {				    		 
			    		 constant[0]= (sign * l1C -constant[0]) / (( this.smooth + Math.sqrt(nc)) / this.learn_rate + C) ;
			    	 }				    	    		
	    	}
	    	
	    	// other features
		    for (int j=0; j < columndimension; j++){	
		    	double sign=1.0;			    	
		    	if (betas[j]  <0){
		    		sign=-1.0;
		    	}
		    	 if (sign * betas[j]  <= l1C){
		    		 betas[j]=0 ;
		    	 } else {
		    		 betas[j]=  (sign * l1C - betas[j]) / ((  this.smooth+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
		    	 }

		    	
		    }				
			
			
			// end of FTRL
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
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (P<=0){
			throw new IllegalStateException(" The Epsilon P value cannot be less equal to zero" );
		}			
		if ( !this.Objective.equals("L1") && !this.Objective.equals("L2"))  {
			throw new IllegalStateException("the objective has to be one of L2 or L1 " );	
		}			

		if ( !Type.equals("Liblinear")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, Liblinear or Routine methods" );	
		}			
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		// make sensible checks on the target data
		if (target==null || target.length!=data.GetRowDimension()){
			throw new IllegalStateException(" target array needs to be provided" );
		} else {
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more than 1 different values!" );	
			}
			has=null;
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
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			if (betas.length!=columndimension){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  columndimension);
			}
		} else { //Initialise beta if not given
			betas= new double[columndimension];
			constant= new double[]{0.0};
		}
		
		if (Type.equals("Liblinear")){
			
			int l = data.GetRowDimension();

	        double p =P ;
	        int w_size = columndimension;
	        double eps = tolerance;
	        int i, s, iter = 0;
	        int max_iter = maxim_Iteration;
	        int active_size = l;
	        int[] index = new int[l];

	        double d, G, H;
	        double Gmax_old = Double.POSITIVE_INFINITY;
	        double Gmax_new, Gnorm1_new;
	        double Gnorm1_init = 0; // initialize to 0 to get rid of Eclipse warning/error
	        double[] betass = new double[l];
	        double[] QD = new double[l];
	        double val=0.0;
	        // L2R_L2LOSS_SVR_DUAL
	        double[] lambda = new double[] {0.5 / C};
	        double[] upper_bound = new double[] {Double.POSITIVE_INFINITY};

	        if (this.Objective.equals("L1")) {
	            lambda[0] = 0;
	            upper_bound[0] = C;
	        }

	        // Initial beta can be set here. Note that
	        // -upper_bound <= beta[i] <= upper_bound
	        for (i = 0; i < l; i++)
	        	betass[i] = 0;
	        
	        for (i = 0; i < l; i++) {
	            QD[i] = 0;
				if (UseConstant){
					QD[i] +=1;
					constant[0]+= betass[i];
				}
	            for (int j=0; j < w_size; j++) {
	            	val= data.GetElement(i, j) ;
	            	if (val==0.0){
	            		continue;
	            	}
	            	if (usescale){
	            		val=Scaler.transform(val, j);
	            	}
	            			
	                QD[i] +=val *  val;
	                betas[j] += betass[i] * val;
	            }

	            index[i] = i;
	        }


	        while (iter < max_iter) {
	            Gmax_new = 0;
	            Gnorm1_new = 0;

	            for (i = 0; i < active_size; i++) {
	                int j = i + random.nextInt(active_size - i);
	                swap(index, i, j);
	            }

	            for (s = 0; s < active_size; s++) {
	                i = index[s];
	                G = -target[i] + lambda[GETI_SVR(i)] * betass[i];
	                H = QD[i] + lambda[GETI_SVR(i)];

					if (UseConstant){
						 G +=  constant[0];
					}
					
	                for (int j=0; j < w_size; j++) {
	                    int ind = j;
		            	val= data.GetElement(i, j) ;
		            	if (val==0.0){
		            		continue;
		            	}
		            	if (usescale){
		            		val=Scaler.transform(val, ind);
		            	}
	                    G += val * betas[ind];
	                }

	                double Gp = G + p;
	                double Gn = G - p;
	                double violation = 0;
	                if (betass[i] == 0) {
	                    if (Gp < 0)
	                        violation = -Gp;
	                    else if (Gn > 0)
	                        violation = Gn;
	                    else if (Gp > Gmax_old && Gn < -Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] >= upper_bound[GETI_SVR(i)]) {
	                    if (Gp > 0)
	                        violation = Gp;
	                    else if (Gp < -Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] <= -upper_bound[GETI_SVR(i)]) {
	                    if (Gn < 0)
	                        violation = -Gn;
	                    else if (Gn > Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] > 0)
	                    violation = Math.abs(Gp);
	                else
	                    violation = Math.abs(Gn);

	                Gmax_new = Math.max(Gmax_new, violation);
	                Gnorm1_new += violation;

	                // obtain Newton direction d
	                if (Gp < H * betass[i])
	                    d = -Gp / H;
	                else if (Gn > H * betass[i])
	                    d = -Gn / H;
	                else
	                    d = -betass[i];

	                if (Math.abs(d) < 1.0e-12) continue;

	                double beta_old = betass[i];
	                betass[i] = Math.min(Math.max(betass[i] + d, -upper_bound[GETI_SVR(i)]), upper_bound[GETI_SVR(i)]);
	                d = betass[i] - beta_old;

	                if (d != 0) {
						if (UseConstant){
							  constant[0]+= d;
						}
	                	for (int j=0; j < w_size; j++) {
	    	            	val= data.GetElement(i, j) ;
	    	            	if (val==0.0){
	    	            		continue;
	    	            	}
	    	            	if (usescale){
	    	            		val=Scaler.transform(val, j);
	    	            	}
	                        betas[j] += d * val;
	                    }
	                }
	            }

	            if (iter == 0) Gnorm1_init = Gnorm1_new;
	            
	            if (verbose ){
	           System.out.println("Iteration: " + iter);
	            }
	            
	            iter++;
	            
	            if (Gnorm1_new <= eps * Gnorm1_init) {
	                if (active_size == l)
	                    break;
	                else {
	                    active_size = l;

	                    Gmax_old = Double.POSITIVE_INFINITY;
	                    continue;
	                }
	            }

	            Gmax_old = Gmax_new;
	        }

	      }else if (Type.equals("SGD")){
	        	
				if (this.learn_rate<=0.00000000000001){
					throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
				}
				if (this.smooth<0.0){
					throw new IllegalStateException(" smooth value cannot be less  than 0");
				}
				double iteration_tol=Double.POSITIVE_INFINITY;
				int it=0;
				double nc=0;
				//initiali
				// random number generator
				random= new Random();
				random.setSeed(seed);
				double n []= new double[data.GetColumnDimension()]; // sum of squared gradients
				double feature=0.0;
				// iterative algorithms start here
				while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

			    for (int k=0; k < data.GetRowDimension(); k++){
			    	int i=k;

			    	if (this.shuffle){
			    	 i=random.nextInt(data.GetRowDimension());
			    	}
			    	double pred=constant[0];
			    	double yi=target[i];
		
			    	// compute score
			    		
			    	for (int j=0; j < data.GetColumnDimension(); j++){
			    		
				    		feature= data.GetElement(i, j) ;
			            	if (feature==0.0){
			            		continue;
			            	}
		    				if (usescale){
		    					feature=Scaler.transform(feature, j);
		    				}    
				    		pred+=feature*betas[j];
			    		
			    		
			    	}
			    	
			    	double residual=(pred-yi);
			    	if (Math.abs(residual)>=this.P){
				    	// we update constant gradient
				    	 if (UseConstant){
				    		 double gradient=0 ;
				    		 if (this.Objective.equals("L2")){
				    			 gradient=residual;
				    		 } else if  (this.Objective.equals("L1")){
				    			 if (residual>0){
				    				 gradient=1;
				    			 } else if (residual<0){
				    				 gradient=-1;
				    			 }
				    		 }
				    	     gradient+=C*constant[0];
				    		 double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);
				    		 nc+=gradient*gradient;
				    		 constant[0]=constant[0]-move; 
				    	 }
				    	 
				    	 for (int j=0; j < data.GetColumnDimension();j++){
				    		 	    		 
					    		feature= data.GetElement(i, j) ;
				            	if (feature==0.0){
				            		continue;
				            	}
				    				if (usescale){
				    					feature=Scaler.transform(feature, j);
				    				}   
						    	 double gradient=0.0;
						    	 
						    	 if (this.Objective.equals("L2")){
					    			 gradient=residual*feature;
					    		 } else if  (this.Objective.equals("L1")){
					    			 if (residual>0){
					    				 gradient=feature;
					    			 } else if (residual<0){
					    				 gradient=-feature;
					    			 }
					    		 }    		 
				    		 
				    			 gradient+=C*betas[j];
	
					    		 double move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
					    		 n[j]+=gradient*gradient;
					    		 betas[j]=betas[j]-move;
	
				    		 	
				    	 }
			    
			    	}
			    	
			    }
	             
		           //end of while
		            it++; 
		    		if (verbose){
		    					System.out.println("iteration: " + it);
		    				}
				}
				
				// end of SGD			
			// end of SGD
		} else if (Type.equals("FTRL")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.l1C<=0){
				throw new IllegalStateException(" L1C regularizer cannot be less equal to zero");
			}	
			if (this.smooth<0.0){
				throw new IllegalStateException(" smooth value cannot be less  than 0");
			}	
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[columndimension]; // sum of squared gradients

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	//System.out.println(Arrays.toString(data[i]));
		    	
				double BETA [] = new double[columndimension];
		    	double pred=0.0;
		    	double Constant=0.0;
			 
		    	// if we use constant
		    	if (UseConstant){
		    		
		    		double sign=1;
		    		
		    		if (constant[0]<0){
		    			sign=-1;
		    		}
				    	 if (sign * constant[0]  <= l1C){				    		 
				    		 Constant=0 ;				    		 
				    	 } else {				    		 
				    		 Constant= (sign * l1C -constant[0]) / ((+ this.smooth + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	}
		    	
		    	// other features
			    for (int j=0; j < columndimension; j++){	
			    		feature= data.GetElement(i, j) ;
		            	if (feature==0.0){
		            		continue;
		            	}
	    				if (usescale){
	    					feature=Scaler.transform(feature, j);
	    				} 	
	    				
			    	double sign=1.0;	
			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / ((  this.smooth+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    
			    pred+= BETA[j]*feature;	
			    
			    	
			    	
			    }			     
			    
			    
		    	double yi=target[i];
			    double residual= (pred-yi);
			    
			   if ( Math.abs(residual)>=this.P){
					if (UseConstant){	
						double gradient=0.0;
						 if (this.Objective.equals("L2")){
								 gradient=residual;
							 
			    		 } else if  (this.Objective.equals("L1")){
				    			 if (residual>0){
				    				 gradient=1;
				    			 } else if (residual<0){
				    				 gradient=-1;
				    			 }
			    			 
			    		 }
						 
						double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;
	
						constant[0]+=gradient-move*Constant;
						nc+=gradient*gradient;
					 }
				    //updates
					//print.Print(betas, 5);
				    for (int j=0; j < columndimension; j++){
				    		feature= data.GetElement(i, j) ;
			            	if (feature==0.0){
			            		continue;
			            	}
		    				if (usescale){
		    					feature=Scaler.transform(feature, j);
		    				} 
				    		double gradientx=0.0;
				    		
					    	 if (this.Objective.equals("L2")){
									 gradientx=residual*feature;
								 
					    		
				    		 } else if  (this.Objective.equals("L1")){
					    			 if (residual>0){
					    				 gradientx=feature;
					    			 } else if (residual<0){
					    				 gradientx=-feature;
					    			 }
				    			 
				    		 }	 	 
				    		
				    	//System.out.println(" gradient: " + gradientx);
				    	double move=(Math.sqrt(n[j] + gradientx * gradientx) - Math.sqrt(n[j])) / this.learn_rate;
				    	betas[j] += gradientx - move * BETA[j];
	                    n[j] += gradientx * gradientx;	
	
	                    // check for highest tolerance
	      				 if (Math.abs(BETA[j])>iteration_tol){
								 iteration_tol= Math.abs(BETA[j]);
							 }
				    
				    }
			    
			   } 
			  	
		    }	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			// reset coeffs
	    	if (UseConstant){
	    		
	    		double sign=1;
	    		
	    		if (constant[0]<0){
	    			sign=-1;
	    		}
			    	 if (sign * constant[0]  <= l1C){				    		 
			    		 constant[0]=0 ;				    		 
			    	 } else {				    		 
			    		 constant[0]= (sign * l1C -constant[0]) / (( this.smooth + Math.sqrt(nc)) / this.learn_rate + C) ;
			    	 }				    	    		
	    	}
	    	
	    	// other features
		    for (int j=0; j < columndimension; j++){	
		    	double sign=1.0;			    	
		    	if (betas[j]  <0){
		    		sign=-1.0;
		    	}
		    	 if (sign * betas[j]  <= l1C){
		    		 betas[j]=0 ;
		    	 } else {
		    		 betas[j]=  (sign * l1C - betas[j]) / ((  this.smooth+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
		    	 }

		    	
		    }				
			
			
			// end of FTRL
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
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (P<=0){
			throw new IllegalStateException(" The Epsilon P value cannot be less equal to zero" );
		}			
		if ( !this.Objective.equals("L1") && !this.Objective.equals("L2"))  {
			throw new IllegalStateException("the objective has to be one of L2 or L1 " );	
		}			

		if ( !Type.equals("Liblinear")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, Liblinear or Routine methods" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		// make sensible checks on the target data
		if (target==null || target.length!=data.GetRowDimension()){
			throw new IllegalStateException(" target array needs to be provided" );
		} else {
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more than 1 different values!" );	
			}
			has=null;
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
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			if (betas.length!=columndimension){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  columndimension);
			}
		} else { //Initialise beta if not given
			betas= new double[columndimension];
			constant= new double[]{0.0};
		}
		
		// sort the matrix if necessary
		
        if (sparse_set==false){
		    if (!data.IsSortedByRow()){
		    	data.convert_type();
		    }

	        }


	    double val=0.0;
		
		
		if (Type.equals("Liblinear")){
			
			int l = data.GetRowDimension();

	        double p =P ;
	        double eps = tolerance;
	        int i, s, iter = 0;
	        int max_iter = maxim_Iteration;
	        int active_size = l;
	        int[] index = new int[l];

	        double d, G, H;
	        double Gmax_old = Double.POSITIVE_INFINITY;
	        double Gmax_new, Gnorm1_new;
	        double Gnorm1_init = 0; // initialize to 0 to get rid of Eclipse warning/error
	        double[] betass = new double[l];
	        double[] QD = new double[l];

	        // L2R_L2LOSS_SVR_DUAL
	        double[] lambda = new double[] {0.5 / C};
	        double[] upper_bound = new double[] {Double.POSITIVE_INFINITY};

	        if (this.Objective.equals("L1")) {
	            lambda[0] = 0;
	            upper_bound[0] = C;
	        }

	        // Initial beta can be set here. Note that
	        // -upper_bound <= beta[i] <= upper_bound
	        for (i = 0; i < l; i++)
	        	betass[i] = 0;
	        
	        for (i = 0; i < l; i++) {
	            QD[i] = 0;
				if (UseConstant){
					QD[i] +=1;
					constant[0]+= betass[i];
				}
				for (int h=data.indexpile[i]; h<data.indexpile[i+1]; h++ ) {
					int j= data.mainelementpile[h];
    				val=data.valuespile[h];
	            	if (usescale){
	            		val=Scaler.transform(val, j);
	            	}
	            			
	                QD[i] +=val *  val;
	                betas[j] += betass[i] * val;
	            }

	            index[i] = i;
	        }


	        while (iter < max_iter) {
	            Gmax_new = 0;
	            Gnorm1_new = 0;

	            for (i = 0; i < active_size; i++) {
	                int j = i + random.nextInt(active_size - i);
	                swap(index, i, j);
	            }

	            for (s = 0; s < active_size; s++) {
	                i = index[s];
	                G = -target[i] + lambda[GETI_SVR(i)] * betass[i];
	                H = QD[i] + lambda[GETI_SVR(i)];

					if (UseConstant){
						 G +=  constant[0];
					}
					
					for (int h=data.indexpile[i]; h<data.indexpile[i+1]; h++ ) {
	    				val=data.valuespile[h];
	                    int ind =data.mainelementpile[h];
		            	if (usescale){
		            		val=Scaler.transform(val, ind);
		            	}
	                    G += val * betas[ind];
	                }

	                double Gp = G + p;
	                double Gn = G - p;
	                double violation = 0;
	                if (betass[i] == 0) {
	                    if (Gp < 0)
	                        violation = -Gp;
	                    else if (Gn > 0)
	                        violation = Gn;
	                    else if (Gp > Gmax_old && Gn < -Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] >= upper_bound[GETI_SVR(i)]) {
	                    if (Gp > 0)
	                        violation = Gp;
	                    else if (Gp < -Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] <= -upper_bound[GETI_SVR(i)]) {
	                    if (Gn < 0)
	                        violation = -Gn;
	                    else if (Gn > Gmax_old) {
	                        active_size--;
	                        swap(index, s, active_size);
	                        s--;
	                        continue;
	                    }
	                } else if (betass[i] > 0)
	                    violation = Math.abs(Gp);
	                else
	                    violation = Math.abs(Gn);

	                Gmax_new = Math.max(Gmax_new, violation);
	                Gnorm1_new += violation;

	                // obtain Newton direction d
	                if (Gp < H * betass[i])
	                    d = -Gp / H;
	                else if (Gn > H * betass[i])
	                    d = -Gn / H;
	                else
	                    d = -betass[i];

	                if (Math.abs(d) < 1.0e-12) continue;

	                double beta_old = betass[i];
	                betass[i] = Math.min(Math.max(betass[i] + d, -upper_bound[GETI_SVR(i)]), upper_bound[GETI_SVR(i)]);
	                d = betass[i] - beta_old;

	                if (d != 0) {
						if (UseConstant){
							  constant[0]+= d;
						}
						for (int h=data.indexpile[i]; h<data.indexpile[i+1]; h++ ) {
							int j= data.mainelementpile[h];
		    				val=data.valuespile[h];
	    	            	if (usescale){
	    	            		val=Scaler.transform(val, j);
	    	            	}
	                        betas[j] += d * val;
	                    }
	                }
	            }

	            if (iter == 0) Gnorm1_init = Gnorm1_new;
	            
	            if (verbose ){
	           System.out.println("Iteration: " + iter);
	            }
	            
	            iter++;
	            
	            if (Gnorm1_new <= eps * Gnorm1_init) {
	                if (active_size == l)
	                    break;
	                else {
	                    active_size = l;

	                    Gmax_old = Double.POSITIVE_INFINITY;
	                    continue;
	                }
	            }

	            Gmax_old = Gmax_new;
	        }
        	
        }else if (Type.equals("SGD")){
        	
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.smooth<0.0){
				throw new IllegalStateException(" smooth value cannot be less  than 0");
			}
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[data.GetColumnDimension()]; // sum of squared gradients
			
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	double pred=constant[0];
		    	double yi=target[i];
	
		    	// compute score
		    		
    			for (int s=data.indexpile[i]; s<data.indexpile[i+1]; s++ ) {
    				int j= data.mainelementpile[s];
    				feature=data.valuespile[s];

	    				if (usescale){
	    					feature=Scaler.transform(feature, j);
	    				}    
			    		pred+=feature*betas[j];
		    		
		    		
		    	}
		    	
		    	double residual=(pred-yi);
				 if ( Math.abs(residual)>=this.P){
			    	 // we update constant gradient
			    	 if (UseConstant){
			    		 double gradient=0 ;
			    		 if (this.Objective.equals("L2")){
			    			 gradient=residual;
			    		 } else if  (this.Objective.equals("L1")){
			    			 if (residual>0){
			    				 gradient=1;
			    			 } else if (residual<0){
			    				 gradient=-1;
			    			 }
			    		 }
			    	     gradient+=C*constant[0];
			    		 double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);
			    		 nc+=gradient*gradient;
			    		 constant[0]=constant[0]-move; 
			    	 }
			    	 
		    			for (int s=data.indexpile[i]; s<data.indexpile[i+1]; s++ ) {
		    				int j= data.mainelementpile[s];
		    				feature=data.valuespile[s];
			    		 	    		 
		
			    				if (usescale){
			    					feature=Scaler.transform(feature, j);
			    				}   
					    	 double gradient=0.0;
					    	 
					    	 if (this.Objective.equals("L2")){
				    			 gradient=residual*feature;
				    		 } else if  (this.Objective.equals("L1")){
				    			 if (residual>0){
				    				 gradient=feature;
				    			 } else if (residual<0){
				    				 gradient=-feature;
				    			 }
				    		 }	    		 
			    		 
			    			 gradient+=C*betas[j];
	
				    		 double move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
				    		 n[j]+=gradient*gradient;
				    		 betas[j]=betas[j]-move;
	
			    		 	
			    	 }
			    
			    }
		    } 
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			// end of SGD
		} else if (Type.equals("FTRL")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.l1C<=0){
				throw new IllegalStateException(" L1C regularizer cannot be less equal to zero");
			}	
			if (this.smooth<0.0){
				throw new IllegalStateException(" smooth value cannot be less  than 0");
			}	
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[columndimension]; // sum of squared gradients
			double feature=0.0;
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	//System.out.println(Arrays.toString(data[i]));
		    	
				double BETA [] = new double[columndimension];
		    	double pred=0.0;
		    	double Constant=0.0;
			 
		    	// if we use constant
		    	if (UseConstant){
		    		
		    		double sign=1;
		    		
		    		if (constant[0]<0){
		    			sign=-1;
		    		}
				    	 if (sign * constant[0]  <= l1C){				    		 
				    		 Constant=0 ;				    		 
				    	 } else {				    		 
				    		 Constant= (sign * l1C -constant[0]) / ((+ this.smooth + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	}
		    	
		    	// other features
    			for (int s=data.indexpile[i]; s<data.indexpile[i+1]; s++ ) {
    				int j= data.mainelementpile[s];
    				feature=data.valuespile[s];	

	    				if (usescale){
	    					feature=Scaler.transform(feature, j);
	    				} 	
	    				
			    	double sign=1.0;	
			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / ((  this.smooth+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    
			    pred+= BETA[j]*feature;	
			    
			    	
			    	
			    }			     
			    			    
		    	double yi=target[i];
			    double residual= (pred-yi);
				if ( Math.abs(residual)>=this.P){
					if (UseConstant){	
						double gradient=0.0;
						 if (this.Objective.equals("L2")){
			    			 gradient=residual;
			    		 } else if  (this.Objective.equals("L1")){
			    			 if (residual>0){
			    				 gradient=1;
			    			 } else if (residual<0){
			    				 gradient=-1;
			    			 }
			    		 }
						double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;
	
						constant[0]+=gradient-move*Constant;
						nc+=gradient*gradient;
					 }
				    //updates
					//print.Print(betas, 5);
	    			for (int s=data.indexpile[i]; s<data.indexpile[i+1]; s++ ) {
	    				int j= data.mainelementpile[s];
	    				feature=data.valuespile[s];
	
		    				if (usescale){
		    					feature=Scaler.transform(feature, j);
		    				} 
				    		double gradientx=0.0;
					    	 if (this.Objective.equals("L2")){
					    		 gradientx=residual*feature;
				    		 } else if  (this.Objective.equals("L1")){
				    			 if (residual>0){
				    				 gradientx=feature;
				    			 } else if (residual<0){
				    				 gradientx=-feature;
				    			 }
				    		 }	 
				    		
				    	//System.out.println(" gradient: " + gradientx);
				    	double move=(Math.sqrt(n[j] + gradientx * gradientx) - Math.sqrt(n[j])) / this.learn_rate;
				    	betas[j] += gradientx - move * BETA[j];
	                    n[j] += gradientx * gradientx;	
	
	                    // check for highest tolerance
	      				 if (Math.abs(BETA[j])>iteration_tol){
								 iteration_tol= Math.abs(BETA[j]);
							 }
	    			} 
				    }
			  	
		    }	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			// reset coeffs
	    	if (UseConstant){
	    		
	    		double sign=1;
	    		
	    		if (constant[0]<0){
	    			sign=-1;
	    		}
			    	 if (sign * constant[0]  <= l1C){				    		 
			    		 constant[0]=0 ;				    		 
			    	 } else {				    		 
			    		 constant[0]= (sign * l1C -constant[0]) / (( this.smooth + Math.sqrt(nc)) / this.learn_rate + C) ;
			    	 }				    	    		
	    	}
	    	
	    	// other features
		    for (int j=0; j < columndimension; j++){	
		    	double sign=1.0;			    	
		    	if (betas[j]  <0){
		    		sign=-1.0;
		    	}
		    	 if (sign * betas[j]  <= l1C){
		    		 betas[j]=0 ;
		    	 } else {
		    		 betas[j]=  (sign * l1C - betas[j]) / ((  this.smooth+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
		    	 }

		    	
		    }				
			
			
			// end of FTRL
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
		return "singleLSVR";
	}
	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor: Single Regularized "+ Objective + " Linear Regression");
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);		
		System.out.println("Constant in the model: "+ this.UseConstant);
		System.out.println("Regularization value: "+ this.C);
		System.out.println("Regularization L1 for FTLR: "+ this.l1C);		
		System.out.println("Smooth value: "+ this.smooth);	
		System.out.println("Training method: "+ this.Type);	
		System.out.println("P Epsilon value: "+ this.P);		
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
	
    static void swap(double[] array, int idxA, int idxB) {
        double temp = array[idxA];
        array[idxA] = array[idxB];
        array[idxB] = temp;
    }

    static void swap(int[] array, int idxA, int idxB) {
        int temp = array[idxA];
        array[idxA] = array[idxB];
        array[idxB] = temp;
    }
    
    private static int GETI_SVR(int i) {
        return 0;
    }
	@Override
	public boolean IsRegressor() {
		return true;
	}
	@Override
	public boolean IsClassifier() {
		return false;
	}
	@Override
	public void reset() {
		constant=new double []{0.0};
		betas=null;
		Objective="L2";
		C=1.0;
		l1C=1.0;
		P=0.1;
		smooth=0.01;
		Type="Liblinear";
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
		shuffle=true;
		learn_rate=1.0;
		columndimension=0;
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
		SinlgeLSVR br = new SinlgeLSVR();
		br.constant=this.constant;
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		br.Objective=this.Objective;
		br.C=this.C;
		br.l1C=this.l1C;
		br.P=this.P;
		br.Type=this.Type;
		br.smooth=this.smooth;
		br.threads=this.threads;
		br.columndimension=this.columndimension;
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
				else if (metric.equals("P")) {this.P=Double.parseDouble(value);}		
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("Objective")) {this.Objective=value;}
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

	@Override
	public void setScaler(scaler sc) {
		this.Scaler=sc;		
	}
	@Override
	public void setSeed(int seed) {
		this.seed=seed;}
	@Override
	public scaler ReturnScaler() {
		return this.Scaler;
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
	
	}
}
