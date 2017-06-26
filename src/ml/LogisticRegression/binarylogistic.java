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

package ml.LogisticRegression;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import preprocess.scaling.maxscaler;
import preprocess.scaling.scaler;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;

/**
 * @author marios
 *<p> class to implement binary logistic regression (that is also used in the multinomial model) <p>
   <p> It contains 4 solvers:</p>
 * <li><a href="https://www.csie.ntu.edu.tw/~cjlin/liblinear/">LibLinear</a> (L2 and L1 regularization)</li> 
 * <li> Routine matrix method with <a href="https://www.stat.cmu.edu/~cshalizi/350/lectures/26/lecture-26.pdf" >Newton-Raphson </a> method (supports L2)</li> 
 * <li> SGD "Stochastic Gradient Descent" with adaptive learning Rate (supports L1 and L2)  </li> 
 * <li> FTRL"Follow The Regularized Leader" (supports L1 and L2), inspired by Tingru's code in Kaggle.com forums <a href="https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory"> link </a>  </li> 
 * </ol>
 
 */
public class binarylogistic implements estimator,classifier,Runnable {

	private static final long serialVersionUID = 830529727388893394L;
	/**
	 * Private method for when this class is used in a multinomial context to avoid re-sorting each time (for each class)
	 */
	private boolean sparse_set=false;
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
	 * Type of algorithm to use. It has to be one of Liblinear, Routine, SGD, FTRL =Follow The Regularized Leader
	 */
	public String Type="Liblinear";
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
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
	}
	
	/**
	 * 
	 * @param need_sort : Whether we want to avoid sorting (used internally for multinomial models)
	 */
	public void set_sparse_indicator(boolean need_sort){
		this.sparse_set=true;
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
	public binarylogistic(){}
	
	/**
	 * Default constructor for Binary Logistic Regression with double data
	 */
	public binarylogistic(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	/**
	 * Default constructor for Binary Logistic Regression with fsmatrix data
	 */
	public binarylogistic(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for Binary Logistic Regression with smatrix data
	 */
	public binarylogistic(smatrix data){
		
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
	/**
	 * Default constructor for Binary Logistic Regression with fsmatrix data
	 */
	public void setdata(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for Binary Logistic Regression with smatrix data
	 */
	public void setdata(smatrix data){
		
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
	public double[][] predict_proba(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double predictions[][]= new double [data.length][2];
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data[i][j], j);
				}
				value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=value;
				predictions[i][0]=1-value;
			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i][1]=value;
			predictions[i][0]=1-value;
		}
		
		}
		return predictions;
	}
     /**
      * 
      * @param data : to be scored
      * @return the probability for the event to be 1
      */
	public double[] predict_single(double[][] data) {
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
				value= 1. / (1. + Math.exp(-value));
				predictions[i]=value;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i]=value;
		}
		
		}
		return predictions;
	}
	
	@Override
	public double[][] predict_proba(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][2];
		if(usescale && Scaler!=null) {
			for (int i=0; i < predictions.length; i++) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data.GetElement(i, j), j);
				}
				value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=value;
				predictions[i][0]=1-value;
			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i][1]=value;
			predictions[i][0]=1-value;
		}
		
		}
		return predictions;
	}

    /**
     * 
     * @param data : to be scored
     * @return the probability for the event to be 1
     */
	public double[] predict_single(fsmatrix data) {
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
				value= 1. / (1. + Math.exp(-value));
				predictions[i]=value;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i]=value;
		}
		
		}
		return predictions;
	}
	
	@Override
	public double[][] predict_proba(smatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][2];
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
				value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=value;
				predictions[i][0]=1-value;
			}

		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i][1]=value;
			predictions[i][0]=1-value;
		}
		}

		return predictions;
	}
	
	public double[] predict_single(smatrix data) {
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
					value+=betas[data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);;
				}
				value= 1. / (1. + Math.exp(-value));
				predictions[i]=value;
			}


		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i]=value;
		}
		}

		return predictions;
	}	
	

	@Override
	public double[] predict_probaRow(double[] data) {
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}
		double predictions[]= new double [2];
		if(usescale && Scaler!=null) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data[j], j);
				}
				value= 1. / (1. + Math.exp(-value));
				predictions[1]=value;
				predictions[0]=1-value;

		} else {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[1]=value;
			predictions[0]=1-value;
		
		
		}
		return predictions;
		

	}

	@Override
	public double[] predict_probaRow(fsmatrix data, int row) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [2];
		if(usescale && Scaler!=null) {
				double value=constant[0];
				for (int j=0; j < columndimension; j++){
					value+=betas[j]*Scaler.transform(data.GetElement(row, j), j);
				}
				value= 1. / (1. + Math.exp(-value));
				predictions[1]=value;
				predictions[0]=1-value;
			
		} else {

			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(row, j);
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[1]=value;
			predictions[0]=1-value;
		
		}
		return predictions;
	}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [2];
		
		if(usescale && Scaler!=null) {
				double value=constant[0];
				for (int j=start; j < end ; j++){
					value+=betas[data.mainelementpile[j]]*Scaler.transform(data.valuespile[j], data.mainelementpile[j]);;
				}
				value= 1. / (1. + Math.exp(-value));
				predictions[1]=value;
				predictions[0]=1-value;			


		} else {
			
			double value=constant[0];
			for (int j=start; j < end ; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[1]=value;
			predictions[0]=1-value;	
		
		}

		return predictions;
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
				value= 1. / (1. + Math.exp(-value));
				predictions[i]=(value >= 0.5) ? 1.0 :0.0 ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i]=(value >= 0.5) ? 1.0 :0.0 ;
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
				value= 1. / (1. + Math.exp(-value));
				predictions[i]=(value >= 0.5) ? 1.0 :0.0 ;
			}


		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i]=(value >= 0.5) ? 1.0 :0.0 ;
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
				value= 1. / (1. + Math.exp(-value));
				predictions[i]=(value >= 0.5) ? 1.0 :0.0 ;

			}

		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions[i]=(value >= 0.5) ? 1.0 :0.0 ;
		}
		
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
				value= 1. / (1. + Math.exp(-value));
				predictions=(value >= 0.5) ? 1.0 :0.0 ;
		} else {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions=(value >= 0.5) ? 1.0 :0.0 ;
		
		
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
				value= 1. / (1. + Math.exp(-value));
				predictions=(value >= 0.5) ? 1.0 :0.0 ;

			
		} else {

			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(row, j);
			}
			value= 1. / (1. + Math.exp(-value));
			predictions=(value >= 0.5) ? 1.0 :0.0 ;
		
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
				value= 1. / (1. + Math.exp(-value));
				predictions=(value >= 0.5) ? 1.0 :0.0 ;		


		} else {
			
			double value=constant[0];
			for (int j=start; j < end ; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			value= 1. / (1. + Math.exp(-value));
			predictions=(value >= 0.5) ? 1.0 :0.0 ;
		}

		return predictions;
	}

	@Override
	public void fit(double[][] data) {
		
		// make sensible checks
		if (data==null || data.length<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if ( !this.RegularizationType.equals("L2") &&  !this.RegularizationType.equals("L1") &&Type.equals("Liblinear") ){
			throw new IllegalStateException(" No regularization is supported by SGD and Routine methods" );	
		}
		if ( !Type.equals("Liblinear")  && !Type.equals("Routine") && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD,FTRL Routine or Liblinear methods" );	
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
			if (has.size()!=2){
				throw new IllegalStateException(" target array needs to have exactly 2 values: -1 and 1" );	
			}
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		        double f = it.next();
		        if (f!=-1.0 && f!=1.0){
		        	throw new IllegalStateException("target array needs to have values: -1 and 1");
		    }
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
			if (RegularizationType.equals("L2")){
				
		    	//initialise other values
				int l = data.length;
			    int w_size = data[0].length;
			    int i, s, iter = 0;
			    double xTx[] = new double[l];
			    int max_iter = maxim_Iteration;
			    int index[] = new int[l];
			    double alpha[] = new double[2 * l]; // store alpha and C - alpha
			    int max_inner_iter = 100; // for inner Newton
			    double innereps = 1e-2;
			    double innereps_min = Math.min(1e-8, tolerance);
			    double upper_bound[] = new double[] {1/C, 0,1/C};            
			    double val=0.0;
				
		    // Initial alpha can be set here. Note that
		    // 0 < alpha[i] < upper_bound[GETI(i)]
		    // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
		    for (i = 0; i < l; i++) {
		    	 index[i] = i;
		        alpha[2 * i] = Math.min(0.001 * upper_bound[GETI(target, i)], 1e-8);
		        alpha[2 * i + 1] = upper_bound[GETI(target, i)] - alpha[2 * i];
		    }

		    for (i = 0; i < l; i++) {
		    	if (UseConstant){
		    		 xTx[i]+=1;
		    		 constant[0]+=target[i] *weights[i] * alpha[2 * i];
		    	}
		        for (int j=0; j<w_size; j++ ) {
		             val = data[i][j];
		             if (val==0){
		            	 continue;
		             }
		            if (this.usescale){
		            	val=Scaler.transform(val, j);
		            }
		            xTx[i] += val*val*weights[i];
		            betas[j] += target[i]*weights[i] * alpha[2 * i] * val;
		        }
		       
		    }

		    while (iter < max_iter) {
		        for (i = 0; i < l; i++) {
		            int j = i +random.nextInt(l - i) ;
		            swap(index, i, j);
		        }
		        
		        int newton_iter = 0;
		        double Gmax = 0;
		        for (s = 0; s < l; s++) {
		            i = index[s];
		            double yi = target[i];
		            double C = upper_bound[GETI(target, i)];
		            double ywTx = 0, xisq = xTx[i];
			    	if (UseConstant){
			    		 ywTx +=constant[0];
			    	}
		            for (int j=0; j<data[0].length;j++) {
		                val = data[i][j];
			             if (val==0){
			            	 continue;
			             }
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
		                ywTx += betas[j] * val;
		            }
		            ywTx *= target[i];
		            double a = xisq, b = ywTx;

		            // Decide to minimize g_1(z) or g_2(z)
		            int ind1 = 2 * i, ind2 = 2 * i + 1, sign = 1;
		            if (0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0) {
		                ind1 = 2 * i + 1;
		                ind2 = 2 * i;
		                sign = -1;
		            }

		            //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
		            double alpha_old = alpha[ind1];
		            double z = alpha_old;
		            if (C - z < 0.5 * C) z = 0.1 * z;
		            double gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
		            Gmax = Math.max(Gmax, Math.abs(gp));

		            // Newton method on the sub-problem
		            final double eta = 0.1; // xi in the paper
		            int inner_iter = 0;
		            while (inner_iter <= max_inner_iter) {
		                if (Math.abs(gp) < innereps) break;
		                double gpp = a + C / (C - z) / z;
		                double tmpz = z - gp / gpp;
		                if (tmpz <= 0)
		                    z *= eta;
		                else
		                    // tmpz in (0, C)
		                    z = tmpz;
		                gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
		                newton_iter++;
		                inner_iter++;
		            }

		            if (inner_iter > 0) // update w
		            {
		                alpha[ind1] = z;
		                alpha[ind2] = C - z;
				    	if (UseConstant){
				    		constant[0]+= sign * (z - alpha_old) * yi;
				    	}
		                for (int j=0; j<data[0].length;j++) {
			                val = data[i][j];
				             if (val==0){
				            	 continue;
				             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
		                	
		                    betas[j] += sign * (z - alpha_old) * yi * val;
		                }
		            }
		        }

		        iter++;
		        if (Gmax < tolerance) break;
		        if (newton_iter <= l / 10) {
		            innereps = Math.max(innereps_min, 0.1 * innereps);
		        }

		    }

		    
		    if (this.verbose){
		    	 System.out.println("Training method successfully converged " + iter);
		    }


		    
		    
				// end of L2 Liblinear
			} else if (RegularizationType.equals("L1")){
				
		    	int l = data.length;
			    int w_size = data[0].length;
		        int j, s, newton_iter = 0, iter = 0;
		        int max_newton_iter = 100;
		        int max_iter = maxim_Iteration;
		        int max_num_linesearch = 20;
		        int active_size;
		        int QP_active_size;
		        double nu = 1e-12;
		        double inner_eps = 1;
		        double sigma = 0.01;
		        double w_norm, w_norm_new;
		        double z, G, H;
		        double Gnorm1_init = 0; // eclipse moans this variable might not be initialized
		        double Gmax_old = Double.POSITIVE_INFINITY;
		        double Gmax_new, Gnorm1_new;
		        double QP_Gmax_old = Double.POSITIVE_INFINITY;
		        double QP_Gmax_new, QP_Gnorm1_new;
		        double delta, negsum_xTd, cond;

		        int[] index = new int[w_size];
		        int indexc=0;
		        double[] Hdiag = new double[w_size];
		        double hdiag=0;
		        double[] Grad = new double[w_size];
		        double grad=0;
		        double[] wpd = new double[w_size];
		        double wp=0;
		        double[] xjneg_sum = new double[w_size];
		        double xjneg_s=0;
		        double[] xTd = new double[l];
		        double[] exp_wTx = new double[l];
		        double[] exp_wTx_new = new double[l];
		        double[] tau = new double[l];
		        double[] D = new double[l];
		        double val=0.0;
		        double[] C = {1/this.C, 0, 1/this.C};

		        w_norm = 0;
		        if (UseConstant){
		        	 w_norm +=constant[0];
		        	 wp=constant[0];
		        	 indexc=0;
		        	 xjneg_s=0;
		        	 for (int i=0; i <l; i++) {
		        		 exp_wTx[i] += constant[0] *weights[i];
			                if (target[i] < 0) {
			                    xjneg_s += C[GETI(target, i)];
			                }
		        	 }
		        }
		        for (j = 0; j < w_size; j++) {
		            w_norm += Math.abs(betas[j]);
		            wpd[j] = betas[j];
		            index[j] = j;
		            xjneg_sum[j] = 0;
		            for (int i=0; i <l; i++) {
		                val = data[i][j];
			             if (val==0){
			            	 continue;
			             }
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
		                exp_wTx[i] += betas[j] * val*weights[i];
		                if (target[i] < 0) {
		                    xjneg_sum[j] += C[GETI(target, i)] * val;
		                }
		            }
		        }
		        for (j = 0; j < l; j++) {
		            exp_wTx[j] = Math.exp(exp_wTx[j]);
		            double tau_tmp = 1 / (1 + exp_wTx[j]);
		            tau[j] = C[GETI(target, j)] * tau_tmp;
		            D[j] = C[GETI(target, j)] * exp_wTx[j] * tau_tmp * tau_tmp;
		        }

		        while (newton_iter < max_newton_iter) {
		            Gmax_new = 0;
		            Gnorm1_new = 0;
		            active_size = w_size;

		            if (UseConstant){
		            	j=indexc;
		                hdiag = nu;
		                grad = 0;

		                double tmp = 0;
		                for (int i=0; i <l; i++) {
		                    hdiag += D[i];
		                    tmp +=  tau[i];
		                }
		                grad = -tmp + xjneg_s;
		                double Gp = grad + 1;
		                double Gn = grad - 1;
		                double violation = 0;
		                if (constant[0] == 0) {
		                    if (Gp < 0)
		                        violation = -Gp;
		                    else if (Gn > 0)
		                        violation = Gn;

		                } else if (constant[0] > 0)
		                    violation = Math.abs(Gp);
		                else
		                    violation = Math.abs(Gn);

		                Gmax_new = Math.max(Gmax_new, violation);
		                Gnorm1_new += violation;
		            }
		            	
		            
		            for (s = 0; s < active_size; s++) {
		                j = index[s];
		                Hdiag[j] = nu;
		                Grad[j] = 0;

		                double tmp = 0;
		                for (int i=0; i <l; i++) {
			                val = data[i][j];
				             if (val==0){
				            	 continue;
				             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }	                	
		                	
		                    Hdiag[j] +=val* val * D[i];
		                    tmp += val * tau[i];
		                }
		                Grad[j] = -tmp + xjneg_sum[j];

		                double Gp = Grad[j] + 1;
		                double Gn = Grad[j] - 1;
		                double violation = 0;
		                if (betas[j] == 0) {
		                    if (Gp < 0)
		                        violation = -Gp;
		                    else if (Gn > 0)
		                        violation = Gn;
		                    //outer-level shrinking
		                    else if (Gp > Gmax_old / l && Gn < -Gmax_old / l) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    }
		                } else if (betas[j] > 0)
		                    violation = Math.abs(Gp);
		                else
		                    violation = Math.abs(Gn);

		                Gmax_new = Math.max(Gmax_new, violation);
		                Gnorm1_new += violation;
		            }

		            if (newton_iter == 0) Gnorm1_init = Gnorm1_new;

		            if (Gnorm1_new <= tolerance * Gnorm1_init) break;

		            iter = 0;
		            QP_Gmax_old = Double.POSITIVE_INFINITY;
		            QP_active_size = active_size;

		            for (int i = 0; i < l; i++)
		                xTd[i] = 0;
		            // optimize QP over wpd
		            while (iter < max_iter) {
		                QP_Gmax_new = 0;
		                QP_Gnorm1_new = 0;

		                for (j = 0; j < QP_active_size; j++) {
		                    int i = random.nextInt(QP_active_size - j);
		                    swap(index, i, j);
		                }

		                if (UseConstant){
		                    H = hdiag;

		                    G = grad + (wp - constant[0]) * nu;
		                    for (int i=0; i <l; i++) {
		                        G +=  D[i] * xTd[i];
		                    }

		                    double Gp = G + 1;
		                    double Gn = G - 1;
		                    double violation = 0;
		                    if (wp == 0) {
		                        if (Gp < 0)
		                            violation = -Gp;
		                        else if (Gn > 0)
		                            violation = Gn;
		                    } else if (wp > 0)
		                        violation = Math.abs(Gp);
		                    else
		                        violation = Math.abs(Gn);

		                    QP_Gmax_new = Math.max(QP_Gmax_new, violation);
		                    QP_Gnorm1_new += violation;

		                    // obtain solution of one-variable problem
		                    if (Gp < H * wp)
		                        z = -Gp / H;
		                    else if (Gn > H * wp)
		                        z = -Gn / H;
		                    else
		                        z = -wp;

		                    if (Math.abs(z) < 1.0e-12) continue;
		                    z = Math.min(Math.max(z, -10.0), 10.0);

		                    wp += z;

		                    for (int i=0; i <l; i++) {
		                        xTd[i] +=  z;
		                    }
		                	
		                }
		                for (s = 0; s < QP_active_size; s++) {
		                    j = index[s];
		                    H = Hdiag[j];

		                    G = Grad[j] + (wpd[j] - betas[j]) * nu;
		                    for (int i=0; i <l; i++) {
		                    	val = data[i][j];
			   		            if (val==0){
					            	 continue;
					             }
					            if (this.usescale){
					            	val=Scaler.transform(val, j);
					            }
		                    	
		                        G += val * D[i] * xTd[i];
		                    }

		                    double Gp = G + 1;
		                    double Gn = G - 1;
		                    double violation = 0;
		                    if (wpd[j] == 0) {
		                        if (Gp < 0)
		                            violation = -Gp;
		                        else if (Gn > 0)
		                            violation = Gn;
		                        //inner-level shrinking
		                        else if (Gp > QP_Gmax_old / l && Gn < -QP_Gmax_old / l) {
		                            QP_active_size--;
		                            swap(index, s, QP_active_size);
		                            s--;
		                            continue;
		                        }
		                    } else if (wpd[j] > 0)
		                        violation = Math.abs(Gp);
		                    else
		                        violation = Math.abs(Gn);

		                    QP_Gmax_new = Math.max(QP_Gmax_new, violation);
		                    QP_Gnorm1_new += violation;

		                    // obtain solution of one-variable problem
		                    if (Gp < H * wpd[j])
		                        z = -Gp / H;
		                    else if (Gn > H * wpd[j])
		                        z = -Gn / H;
		                    else
		                        z = -wpd[j];

		                    if (Math.abs(z) < 1.0e-12) continue;
		                    z = Math.min(Math.max(z, -10.0), 10.0);

		                    wpd[j] += z;

		                    for (int i=0; i <l; i++) {
		                    	val = data[i][j];
			   		            if (val==0){
					            	 continue;
					            }
					            if (this.usescale){
					            	val=Scaler.transform(val, j);
					            }
		                        xTd[i] += val * z;
		                    }
		                }

		                iter++;

		                if (QP_Gnorm1_new <= inner_eps * Gnorm1_init) {
		                    //inner stopping
		                    if (QP_active_size == active_size)
		                        break;
		                    //active set reactivation
		                    else {
		                        QP_active_size = active_size;
		                        QP_Gmax_old = Double.POSITIVE_INFINITY;
		                        continue;
		                    }
		                }

		                QP_Gmax_old = QP_Gmax_new;
		            }


		            delta = 0;
		            w_norm_new = 0;
		            if (UseConstant){
		            	delta += grad * (wp - constant[0]);
		            	if (wp != 0) w_norm_new += Math.abs(wp);
		            }
		            
		            for (j = 0; j < w_size; j++) {
		                delta += Grad[j] * (wpd[j] - betas[j]);
		                if (wpd[j] != 0) w_norm_new += Math.abs(wpd[j]);
		            }
		            delta += (w_norm_new - w_norm);

		            negsum_xTd = 0;
		            for (int i = 0; i < l; i++)
		                if (target[i] < 0) negsum_xTd += C[GETI(target, i)] * xTd[i];

		            int num_linesearch;
		            for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
		                cond = w_norm_new - w_norm + negsum_xTd - sigma * delta;

		                for (int i = 0; i < l; i++) {
		                    double exp_xTd = Math.exp(xTd[i]);
		                    exp_wTx_new[i] = exp_wTx[i] * exp_xTd;
		                    cond += C[GETI(target, i)] * Math.log((1 + exp_wTx_new[i]) / (exp_xTd + exp_wTx_new[i]));
		                }

		                if (cond <= 0) {
		                    w_norm = w_norm_new;
		                    if (UseConstant){
		                    	constant[0]=wp;
		                    }
		                    for (j = 0; j < w_size; j++)
		                    	betas[j] = wpd[j];
		                    for (int i = 0; i < l; i++) {
		                        exp_wTx[i] = exp_wTx_new[i];
		                        double tau_tmp = 1 / (1 + exp_wTx[i]);
		                        tau[i] = C[GETI(target, i)] * tau_tmp;
		                        D[i] = C[GETI(target, i)] * exp_wTx[i] * tau_tmp * tau_tmp;
		                    }
		                    break;
		                } else {
		                    w_norm_new = 0;
		                    if (UseConstant){
		                    	
		                        wp = (constant[0] + wp) * 0.5;
		                        if (wp != 0) w_norm_new += Math.abs(wp);
		                    }
		                    for (j = 0; j < w_size; j++) {
		                        wpd[j] = (betas[j] + wpd[j]) * 0.5;
		                        if (wpd[j] != 0) w_norm_new += Math.abs(wpd[j]);
		                    }
		                    delta *= 0.5;
		                    negsum_xTd *= 0.5;
		                    for (int i = 0; i < l; i++)
		                        xTd[i] *= 0.5;
		                }
		            }

		            // Recompute some info due to too many line search steps
		            if (num_linesearch >= max_num_linesearch) {
		                for (int i = 0; i < l; i++)
		                    exp_wTx[i] = 0;

		                if (UseConstant){
		                if (constant[0]==0) continue;
	                    for (int m=0; m <l; m++) {
	                        exp_wTx[m] += constant[0];
	                    }
		                }
		                for (int i = 0; i < w_size; i++) {
		                    if (betas[i] == 0) continue;
		                    for (int m=0; m <l; m++) {
		                    	val =  data[m][i];
			   		             if (val==0){
					            	 continue;
					             }
					            if (this.usescale){
					            	val=Scaler.transform(val, j);
					            }
		                        exp_wTx[m] += betas[i] * val;
		                    }
		                }
		                for (int i = 0; i < l; i++)
		                    exp_wTx[i] = Math.exp(exp_wTx[i]);
		            }

		            if (iter == 1) inner_eps *= 0.25;

		            newton_iter++;
		            Gmax_old = Gmax_new;


		        }

		         if (verbose){
			    	 System.out.println("Training method successfully converged " + iter);
			    }
		
				// end of L1 Liblinear
			}
			
			
		} else if (Type.equals("Routine")){

			// initiate one variable to hold the teration's tolerance
			
			double iteration_tol=1;
			
			// initiate the iteration count
			
			int it=0;
			
			
			// Initialize beta
			int pivot= hascostant(this.UseConstant);			
			double betas2[] = new double [data[0].length + pivot];
			double val=0.0;
			double val1=0.0;
			double val2=0.0;
			for (int i=0; i < betas2.length; i++ ) {
				betas2[i]=0;
			}
			
			// Begin the big Iteration and the Optimization Algorithm
			
			while (iteration_tol> tolerance) {
				
				// put a check in the beginning that will stop the process if solution is not identified. This problem is known as 'failed to converge'
				
	            if (it>maxim_Iteration) {
						break;				
				}
	

	            double second_part []= new double [data[0].length+ pivot];
			// calculate the Probabilities	
	        	double covariancev[]= new double [(data[0].length+ pivot)*(data[0].length+pivot)];

				
	        	// fill it in
	        	for (int i=0; i <data.length; i++ ){
	        		double value=0;
	        		if (this.UseConstant){
	        			value=betas2[0];
	        		}
	        		if (this.RegularizationType.equals("L2")){
	        		 for (int j = 0; j < data[0].length; j++) {
	        			 val = data[i][j];
			             if (val==0){
			            	 continue;
			             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
	        			 value+=betas2[j+pivot]*val;
					 }
	        		} else {
		        		 for (int j = 0; j < data[0].length; j++) {
		        			 if (Math.abs(betas2[j+pivot])>C){
		        				 val = data[i][j];
		    		             if (val==0){
		    		            	 continue;
		    		             }
		 			            if (this.usescale){
		 			            	val=Scaler.transform(val, j);
		 			            }
		        			 value+=betas2[j+pivot]*val;
		        			 
		        			 } else {
		        				 value+=0.0;
		        			 }
						 }	        			
	        			
	        		}
					 double probabilities= 1. / (1. + Math.exp(-Math.max(Math.min(value, 35.), -35.)));
					 // compute weighted residual
					 double weighted_residual=0;
					 if (target[i]>0){
						 weighted_residual=(1-probabilities)*weights[i];
					 } else {
						 weighted_residual=-probabilities*weights[i];
					 }
					if (UseConstant){
						second_part[0]+=weighted_residual;
						covariancev[0]+= probabilities * weights[i];
    					if ( this.RegularizationType.equals("L2") || ( this.RegularizationType.equals("L1") )){
    						covariancev[0]+=C;
    					} else {
    						covariancev[0]-=C;
    					}
	        			for (int d=0;d <data[0].length; d++ ){
		        			 val2=data[i][d];
				             if (val2==0){
				            	 continue;
				             }
		        			if (val2!=0.0){
		        				 if (this.usescale){
		        					 val2=Scaler.transform(val2, d);
			 			            }
	        				covariancev[d+1]+=val2 * probabilities * weights[i];
		        			}
	        			}
					}

	        		for (int j=0; j <data[0].length; j++ ){
	        			 val1=data[i][j];
			             if (val1==0){
			            	 continue;
			             }
	        			if (val1!=0.0){
	        				 if (this.usescale){
	        					 val1=Scaler.transform(val1, j);
		 			            }
	        			second_part[j+pivot]+=val1 * weighted_residual ;
						if (UseConstant){
	        				covariancev[(j+1)*(data[0].length+ 1) ]+= val1 * probabilities * weights[i];
						}
	        			for (int d=0;d <data[0].length; d++ ){
		        			 val2=data[i][d];
				             if (val2==0){
				            	 continue;
				             }
		        			if (val2!=0.0){
		        				 if (this.usescale){
		        					 val2=Scaler.transform(val2, d);
			 			            }
	        				covariancev[(j+pivot)*(data[0].length+pivot) +d+pivot]+= val1* val2 * probabilities * weights[i];
	        				if (j==d){
	        					// add regularization in diagonals
	        					if ( this.RegularizationType.equals("L2") || ( this.RegularizationType.equals("L1"))){
	        					covariancev[(j+pivot)*(data[0].length+pivot) +d+pivot]+=C;
	        				} else {
	        					covariancev[(j+pivot)*(data[0].length+pivot) +d+pivot]-=C;
	        				}
	        					
	        				}
	        			}
	        			}
	        		} 
	        			
	        	}
	        		
	        		
	        		
	        		
	        	}
	        	// betas' updates
	        	double Betas []= new double [betas2.length];
	        	//inverse that matrix with qr decomposition if threads less-equal than 1
	        	if (this.threads<=1){
	        	manipulate.matrixoperations.Inverse.GetInversethis(covariancev);
	        	} else {
	        		// else use a multi-threaded LU-decomposition based inverse
	        		manipulate.matrixoperations.Inverse.LUInversethis(covariancev, this.threads);
	        	}
	        	
	        		for (int j=0; j <betas2.length; j++ ){
	        			for (int i=0; i <betas2.length; i++ ){
	        				Betas[j]+=covariancev[i*betas2.length + j] * second_part[i];
	        				} 
       				 if (Math.abs(Betas[j])>iteration_tol){
							 iteration_tol= Math.abs(Betas[j]);
						 }
 						 betas2[j]+=Betas[j];


	                if (Double.isNaN(betas2[0])){
	    					throw new exceptions.ConvergenceException("Regression failed to converge");
	                }


		//------------------ Here ends Newtons-Raphson Algorithm and the Big While-----------------
			}
		//pass over the coefficients
			if (UseConstant){
				constant[0]=betas2[0];
			}
			for (int j=0; j < betas.length; j++){
				betas[j]=betas2[j+pivot];
			}
			it=it+1;
			if (verbose){
				System.out.println("iteration: " + it);
			}
			}
		} else if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			

			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[data[0].length]; // sum of squared gradients
			double val=0.0;
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.length; k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.length);
		    	}
		    	/*
				double BETA [] = new double [betas.length];
				double Constant=0.0;
		    	double pred=0;
		    	*/
		    	double pred=constant[0];
		    	double yi=0.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	if (!this.RegularizationType.equals("L2") && Math.abs(constant[0])<=C){
		    		pred=0.0;
		    	}
		    	// compute score
		    	if (this.RegularizationType.equals("L2")){
		    	for (int j=0; j < data[i].length; j++){
		    		
		    		val = data[i][j];
		             if (val==0){
		            	 continue;
		             }
		            if (this.usescale){
		            	val=Scaler.transform(val, j);
		            }
		    		pred+=val*betas[j];
		    	}
		    	} else {
		        	for (int j=0; j < data[i].length; j++){
		        		if (Math.abs(betas[j]) >C){
		        			val = data[i][j];
				             if (val==0){
				            	 continue;
				             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
			    		pred+=val*betas[j];
		        		}
			    	}
		    	}
		    	// pred to probability
		    	 pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));
		    	 if (UseConstant){
		    		 boolean superceeds_regularization=true;
		    		 double gradient=(pred - yi) ;
		    		 if (this.RegularizationType.equals("L2")){
		    			 gradient+=C*constant[0];
		    		 } else{
		    			 //get sign
		    			 double sign=-1;
		    			 
		    			 if (constant[0]>0){
		    				 sign=1.0;
		 		    	}
		    			 if (sign * constant[0]  <= C  && nc!=0.0){
		    				// gradient=0.0;
		    				 superceeds_regularization=false;
		    			 } else{
		    				 gradient+=C*sign*constant[0];
		    			 }
		    		 }
		    		if (superceeds_regularization){
		    		 nc+=gradient*gradient;
		    		 double move=(this.learn_rate*gradient)/Math.sqrt(nc+0.00000000000001);
		    		 constant[0]=constant[0]-move;
		    		 
		    		} else {
		    			 //nc=(pred - yi)*(pred - yi);
		    			 //constant[0]=(pred - yi);
		    		}
		    		 
		    	 }
		    	 for (int j=0; j < data[i].length;j++){
		    		 boolean superceeds_regularization=true;
		    		 val = data[i][j];
		             if (val==0){
		            	 continue;
		             }
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
		    		 double gradient=(pred - yi) * val;
		    		 if (this.RegularizationType.equals("L2")){
		    			 gradient+=C*betas[j];
		    		 } else{
		    			 //get sign
		    			 double sign=-1;
		    			 if (betas[j]>0){
		    				 sign=1.0;
		    				  
		 		    	}
		    			 if (sign * betas[j]  <= C && n[j]!=0.0){
		    				// gradient=0.0;
		    				 superceeds_regularization=false;
		    			 } else{
		    				 gradient+=C*sign*betas[j];
		    			 }
		    		 }
		    		 
			    		if (superceeds_regularization){
				    		 n[j]+=gradient*gradient;
				    		 double move=(this.learn_rate*gradient)/Math.sqrt(n[j]+0.00000000000001);
				    		 betas[j]=betas[j]-move;
				    		 if (Math.abs(move)>iteration_tol){
								 iteration_tol= Math.abs(move);
							 }
				    		} else {
				    			//n[j]=((pred - yi)*data[i][j])*((pred - yi)*data[i][j]);
				    			// betas[j]=(pred - yi)*data[i][j];
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
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[columndimension]; // sum of squared gradients
			double val=0.0;

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
				    		 Constant= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	}
		    	
		    	// other features
			    for (int j=0; j < columndimension; j++){	
			    	 val = data[i][j];
		             if (val==0){
		            	 continue;
		             }
			    	double sign=1.0;			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    	
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
			    pred+= BETA[j]*val;	
			    
			    	
			    	
			    }			     
			    pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0))); 
			    

			    
			    double gradient= 0;
			    
				 if (target[i]>0){
					 gradient=  (pred - 1.0);
					    //System.out.println("it is");
				 } else {
					 gradient=  (pred) ;
					   // System.out.println("it is NOT");
				 }
				 
				if (UseConstant){		                    
					double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

					constant[0]+=gradient-move*Constant;
					nc+=gradient*gradient;
				 }
			    //updates
				//print.Print(betas, 5);
			    for (int j=0; j < columndimension; j++){	
			    		val = data[i][j];
			             if (val==0){
			            	 continue;
			             }
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
			    	double gradientx=gradient*val ;
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
			    		 constant[0]= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
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
		    		 betas[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
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
		if ( !this.RegularizationType.equals("L2") &&  !this.RegularizationType.equals("L1") &&Type.equals("Liblinear") ){
			throw new IllegalStateException(" No regularization is supported by SGD and Routine methods" );	
		}
		if ( !Type.equals("Liblinear")  && !Type.equals("Routine") && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL,Routine or Liblinear methods" );	
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
			if (has.size()!=2){
				throw new IllegalStateException(" target array needs to have exactly 2 values: -1 and 1" );	
			}
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		        double f = it.next();
		        if (f!=-1.0 && f!=1.0){
		        	throw new IllegalStateException("target array needs to have values: -1 and 1");
		    }
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
			if (RegularizationType.equals("L2")){
				
		    	//initialise other values
				int l = data.GetRowDimension();
			    int w_size = data.GetColumnDimension();
			    int i, s, iter = 0;
			    double xTx[] = new double[l];
			    int max_iter = maxim_Iteration;
			    int index[] = new int[l];
			    double alpha[] = new double[2 * l]; // store alpha and C - alpha
			    int max_inner_iter = 100; // for inner Newton
			    double innereps = 1e-2;
			    double innereps_min = Math.min(1e-8, tolerance);
			    double upper_bound[] = new double[] {1/C, 0,1/C};            
			    double val=0.0;
				
		    // Initial alpha can be set here. Note that
		    // 0 < alpha[i] < upper_bound[GETI(i)]
		    // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
		    for (i = 0; i < l; i++) {
		    	 index[i] = i;
		        alpha[2 * i] = Math.min(0.001 * upper_bound[GETI(target, i)], 1e-8);
		        alpha[2 * i + 1] = upper_bound[GETI(target, i)] - alpha[2 * i];
		    }

		    for (i = 0; i < l; i++) {
		    	if (UseConstant){
		    		 xTx[i]+=1;
		    		 constant[0]+=target[i] *weights[i] * alpha[2 * i];
		    	}
		        for (int j=0; j<w_size; j++ ) {
		            val = data.GetElement(i, j);
		             if (val==0){
		            	 continue;
		             }
		            if (this.usescale){
		            	val=Scaler.transform(val, j);
		            }
		            xTx[i] += val*val*weights[i];
		            betas[j] += target[i]*weights[i] * alpha[2 * i] * val;
		        }
		       
		    }

		    while (iter < max_iter) {
		        for (i = 0; i < l; i++) {
		            int j = i +random.nextInt(l - i) ;
		            swap(index, i, j);
		        }
		        
		        int newton_iter = 0;
		        double Gmax = 0;
		        for (s = 0; s < l; s++) {
		            i = index[s];
		            double yi = target[i];
		            double C = upper_bound[GETI(target, i)];
		            double ywTx = 0, xisq = xTx[i];
			    	if (UseConstant){
			    		 ywTx +=constant[0];
			    	}
		            for (int j=0; j<data.GetColumnDimension();j++) {
		            	val = data.GetElement(i, j);
			             if (val==0){
			            	 continue;
			             }
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
		                ywTx += betas[j] *val;
		                
		            }
		            ywTx *= target[i];
		            double a = xisq, b = ywTx;

		            // Decide to minimize g_1(z) or g_2(z)
		            int ind1 = 2 * i, ind2 = 2 * i + 1, sign = 1;
		            if (0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0) {
		                ind1 = 2 * i + 1;
		                ind2 = 2 * i;
		                sign = -1;
		            }

		            //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
		            double alpha_old = alpha[ind1];
		            double z = alpha_old;
		            if (C - z < 0.5 * C) z = 0.1 * z;
		            double gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
		            Gmax = Math.max(Gmax, Math.abs(gp));

		            // Newton method on the sub-problem
		            final double eta = 0.1; // xi in the paper
		            int inner_iter = 0;
		            while (inner_iter <= max_inner_iter) {
		                if (Math.abs(gp) < innereps) break;
		                double gpp = a + C / (C - z) / z;
		                double tmpz = z - gp / gpp;
		                if (tmpz <= 0)
		                    z *= eta;
		                else
		                    // tmpz in (0, C)
		                    z = tmpz;
		                gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
		                newton_iter++;
		                inner_iter++;
		            }

		            if (inner_iter > 0) // update w
		            {
		                alpha[ind1] = z;
		                alpha[ind2] = C - z;
				    	if (UseConstant){
				    		constant[0]+= sign * (z - alpha_old) * yi;
				    	}
		                for (int j=0; j<data.GetColumnDimension();j++) {
		                	val = data.GetElement(i, j);
				             if (val==0){
				            	 continue;
				             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
		                	
		                    betas[j] += sign * (z - alpha_old) * yi *val;
		                }
		            }
		        }
		        if (this.verbose){
			    	 System.out.println("Logistic step " + iter);
			    }
		        iter++;
		        if (Gmax < tolerance) break;
		        if (newton_iter <= l / 10) {
		            innereps = Math.max(innereps_min, 0.1 * innereps);
		        }

		    }

		    
		    if (this.verbose){
		    	 System.out.println("Training method successfully converged " + iter);
		    }


		    
		    
				// end of L2 Liblinear
			} else if (RegularizationType.equals("L1")){
				
		    	int l = data.GetRowDimension();
			    int w_size = data.GetColumnDimension();
		        int j, s, newton_iter = 0, iter = 0;
		        int max_newton_iter = 100;
		        int max_iter = maxim_Iteration;
		        int max_num_linesearch = 20;
		        int active_size;
		        int QP_active_size;
		        double nu = 1e-12;
		        double inner_eps = 1;
		        double sigma = 0.01;
		        double w_norm, w_norm_new;
		        double z, G, H;
		        double Gnorm1_init = 0; // eclipse moans this variable might not be initialized
		        double Gmax_old = Double.POSITIVE_INFINITY;
		        double Gmax_new, Gnorm1_new;
		        double QP_Gmax_old = Double.POSITIVE_INFINITY;
		        double QP_Gmax_new, QP_Gnorm1_new;
		        double delta, negsum_xTd, cond;
		        double val=0.0;
		        int[] index = new int[w_size];
		        int indexc=0;
		        double[] Hdiag = new double[w_size];
		        double hdiag=0;
		        double[] Grad = new double[w_size];
		        double grad=0;
		        double[] wpd = new double[w_size];
		        double wp=0;
		        double[] xjneg_sum = new double[w_size];
		        double xjneg_s=0;
		        double[] xTd = new double[l];
		        double[] exp_wTx = new double[l];
		        double[] exp_wTx_new = new double[l];
		        double[] tau = new double[l];
		        double[] D = new double[l];

		        double[] C = {1/this.C, 0, 1/this.C};

		        w_norm = 0;
		        if (UseConstant){
		        	 w_norm +=constant[0];
		        	 wp=constant[0];
		        	 indexc=0;
		        	 xjneg_s=0;
		        	 for (int i=0; i <l; i++) {
		        		 exp_wTx[i] += constant[0] *weights[i];
			                if (target[i] < 0) {
			                    xjneg_s += C[GETI(target, i)];
			                }
		        	 }
		        }
		        for (j = 0; j < w_size; j++) {
		            w_norm += Math.abs(betas[j]);
		            wpd[j] = betas[j];
		            index[j] = j;
		            xjneg_sum[j] = 0;
		            for (int i=0; i <l; i++) {
		                 val = data.GetElement(i, j);
			             if (val==0){
			            	 continue;
			             }
		                 if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
		                exp_wTx[i] += betas[j] * val*weights[i];
		                if (target[i] < 0) {
		                    xjneg_sum[j] += C[GETI(target, i)] * val;
		                }
		            }
		        }
		        for (j = 0; j < l; j++) {
		            exp_wTx[j] = Math.exp(exp_wTx[j]);
		            double tau_tmp = 1 / (1 + exp_wTx[j]);
		            tau[j] = C[GETI(target, j)] * tau_tmp;
		            D[j] = C[GETI(target, j)] * exp_wTx[j] * tau_tmp * tau_tmp;
		        }

		        while (newton_iter < max_newton_iter) {
		            Gmax_new = 0;
		            Gnorm1_new = 0;
		            active_size = w_size;

		            if (UseConstant){
		            	j=indexc;
		                hdiag = nu;
		                grad = 0;

		                double tmp = 0;
		                for (int i=0; i <l; i++) {
		                    hdiag += D[i];
		                    tmp +=  tau[i];
		                }
		                grad = -tmp + xjneg_s;
		                double Gp = grad + 1;
		                double Gn = grad - 1;
		                double violation = 0;
		                if (constant[0] == 0) {
		                    if (Gp < 0)
		                        violation = -Gp;
		                    else if (Gn > 0)
		                        violation = Gn;

		                } else if (constant[0] > 0)
		                    violation = Math.abs(Gp);
		                else
		                    violation = Math.abs(Gn);

		                Gmax_new = Math.max(Gmax_new, violation);
		                Gnorm1_new += violation;
		            }
		            	
		            
		            for (s = 0; s < active_size; s++) {
		                j = index[s];
		                Hdiag[j] = nu;
		                Grad[j] = 0;

		                double tmp = 0;
		                for (int i=0; i <l; i++) {
		                	
		                	val =data.GetElement(i, j);
				             if (val==0){
				            	 continue;
				             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
		                	
		                    Hdiag[j] +=val* val* D[i];
		                    tmp += val * tau[i];
		                }
		                Grad[j] = -tmp + xjneg_sum[j];

		                double Gp = Grad[j] + 1;
		                double Gn = Grad[j] - 1;
		                double violation = 0;
		                if (betas[j] == 0) {
		                    if (Gp < 0)
		                        violation = -Gp;
		                    else if (Gn > 0)
		                        violation = Gn;
		                    //outer-level shrinking
		                    else if (Gp > Gmax_old / l && Gn < -Gmax_old / l) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    }
		                } else if (betas[j] > 0)
		                    violation = Math.abs(Gp);
		                else
		                    violation = Math.abs(Gn);

		                Gmax_new = Math.max(Gmax_new, violation);
		                Gnorm1_new += violation;
		            }

		            if (newton_iter == 0) Gnorm1_init = Gnorm1_new;

		            if (Gnorm1_new <= tolerance * Gnorm1_init) break;

		            iter = 0;
		            QP_Gmax_old = Double.POSITIVE_INFINITY;
		            QP_active_size = active_size;

		            for (int i = 0; i < l; i++)
		                xTd[i] = 0;
		            // optimize QP over wpd
		            while (iter < max_iter) {
		                QP_Gmax_new = 0;
		                QP_Gnorm1_new = 0;

		                for (j = 0; j < QP_active_size; j++) {
		                    int i = random.nextInt(QP_active_size - j);
		                    swap(index, i, j);
		                }

		                if (UseConstant){
		                    H = hdiag;

		                    G = grad + (wp - constant[0]) * nu;
		                    for (int i=0; i <l; i++) {
		                        G +=  D[i] * xTd[i];
		                    }

		                    double Gp = G + 1;
		                    double Gn = G - 1;
		                    double violation = 0;
		                    if (wp == 0) {
		                        if (Gp < 0)
		                            violation = -Gp;
		                        else if (Gn > 0)
		                            violation = Gn;
		                    } else if (wp > 0)
		                        violation = Math.abs(Gp);
		                    else
		                        violation = Math.abs(Gn);

		                    QP_Gmax_new = Math.max(QP_Gmax_new, violation);
		                    QP_Gnorm1_new += violation;

		                    // obtain solution of one-variable problem
		                    if (Gp < H * wp)
		                        z = -Gp / H;
		                    else if (Gn > H * wp)
		                        z = -Gn / H;
		                    else
		                        z = -wp;

		                    if (Math.abs(z) < 1.0e-12) continue;
		                    z = Math.min(Math.max(z, -10.0), 10.0);

		                    wp += z;

		                    for (int i=0; i <l; i++) {
		                        xTd[i] +=  z;
		                    }
		                	
		                }
		                for (s = 0; s < QP_active_size; s++) {
		                    j = index[s];
		                    H = Hdiag[j];

		                    G = Grad[j] + (wpd[j] - betas[j]) * nu;
		                    for (int i=0; i <l; i++) {
		                    	val = data.GetElement(i, j);
			   		             if (val==0){
					            	 continue;
					             }
					            if (this.usescale){
					            	val=Scaler.transform(val, j);
					            }
		                    	
		                        G += val* D[i] * xTd[i];
		                    }

		                    double Gp = G + 1;
		                    double Gn = G - 1;
		                    double violation = 0;
		                    if (wpd[j] == 0) {
		                        if (Gp < 0)
		                            violation = -Gp;
		                        else if (Gn > 0)
		                            violation = Gn;
		                        //inner-level shrinking
		                        else if (Gp > QP_Gmax_old / l && Gn < -QP_Gmax_old / l) {
		                            QP_active_size--;
		                            swap(index, s, QP_active_size);
		                            s--;
		                            continue;
		                        }
		                    } else if (wpd[j] > 0)
		                        violation = Math.abs(Gp);
		                    else
		                        violation = Math.abs(Gn);

		                    QP_Gmax_new = Math.max(QP_Gmax_new, violation);
		                    QP_Gnorm1_new += violation;

		                    // obtain solution of one-variable problem
		                    if (Gp < H * wpd[j])
		                        z = -Gp / H;
		                    else if (Gn > H * wpd[j])
		                        z = -Gn / H;
		                    else
		                        z = -wpd[j];

		                    if (Math.abs(z) < 1.0e-12) continue;
		                    z = Math.min(Math.max(z, -10.0), 10.0);

		                    wpd[j] += z;

		                    for (int i=0; i <l; i++) {
		                    	val = data.GetElement(i, j);
			   		             if (val==0){
					            	 continue;
					             }
					            if (this.usescale){
					            	val=Scaler.transform(val, j);
					            }
		                    	
		                        xTd[i] += val * z;
		                    }
		                }

		                iter++;

		                if (QP_Gnorm1_new <= inner_eps * Gnorm1_init) {
		                    //inner stopping
		                    if (QP_active_size == active_size)
		                        break;
		                    //active set reactivation
		                    else {
		                        QP_active_size = active_size;
		                        QP_Gmax_old = Double.POSITIVE_INFINITY;
		                        continue;
		                    }
		                }

		                QP_Gmax_old = QP_Gmax_new;
		            }


		            delta = 0;
		            w_norm_new = 0;
		            if (UseConstant){
		            	delta += grad * (wp - constant[0]);
		            	if (wp != 0) w_norm_new += Math.abs(wp);
		            }
		            
		            for (j = 0; j < w_size; j++) {
		                delta += Grad[j] * (wpd[j] - betas[j]);
		                if (wpd[j] != 0) w_norm_new += Math.abs(wpd[j]);
		            }
		            delta += (w_norm_new - w_norm);

		            negsum_xTd = 0;
		            for (int i = 0; i < l; i++)
		                if (target[i] < 0) negsum_xTd += C[GETI(target, i)] * xTd[i];

		            int num_linesearch;
		            for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
		                cond = w_norm_new - w_norm + negsum_xTd - sigma * delta;

		                for (int i = 0; i < l; i++) {
		                    double exp_xTd = Math.exp(xTd[i]);
		                    exp_wTx_new[i] = exp_wTx[i] * exp_xTd;
		                    cond += C[GETI(target, i)] * Math.log((1 + exp_wTx_new[i]) / (exp_xTd + exp_wTx_new[i]));
		                }

		                if (cond <= 0) {
		                    w_norm = w_norm_new;
		                    if (UseConstant){
		                    	constant[0]=wp;
		                    }
		                    for (j = 0; j < w_size; j++)
		                    	betas[j] = wpd[j];
		                    for (int i = 0; i < l; i++) {
		                        exp_wTx[i] = exp_wTx_new[i];
		                        double tau_tmp = 1 / (1 + exp_wTx[i]);
		                        tau[i] = C[GETI(target, i)] * tau_tmp;
		                        D[i] = C[GETI(target, i)] * exp_wTx[i] * tau_tmp * tau_tmp;
		                    }
		                    break;
		                } else {
		                    w_norm_new = 0;
		                    if (UseConstant){
		                    	
		                        wp = (constant[0] + wp) * 0.5;
		                        if (wp != 0) w_norm_new += Math.abs(wp);
		                    }
		                    for (j = 0; j < w_size; j++) {
		                        wpd[j] = (betas[j] + wpd[j]) * 0.5;
		                        if (wpd[j] != 0) w_norm_new += Math.abs(wpd[j]);
		                    }
		                    delta *= 0.5;
		                    negsum_xTd *= 0.5;
		                    for (int i = 0; i < l; i++)
		                        xTd[i] *= 0.5;
		                }
		            }

		            // Recompute some info due to too many line search steps
		            if (num_linesearch >= max_num_linesearch) {
		                for (int i = 0; i < l; i++)
		                    exp_wTx[i] = 0;

		                if (UseConstant){
		                if (constant[0]==0) continue;
	                    for (int m=0; m <l; m++) {
	                        exp_wTx[m] += constant[0];
	                    }
		                }
		                for (int i = 0; i < w_size; i++) {
		                    if (betas[i] == 0) continue;
		                    for (int m=0; m <l; m++) {
		                    	val =data.GetElement(m, i);
			   		             if (val==0){
					            	 continue;
					             }
					            if (this.usescale){
					            	val=Scaler.transform(val, j);
					            }
		                        exp_wTx[m] += betas[i] * val;
		                    }
		                }
		                for (int i = 0; i < l; i++)
		                    exp_wTx[i] = Math.exp(exp_wTx[i]);
		            }

		            if (iter == 1) inner_eps *= 0.25;

		            newton_iter++;
		            Gmax_old = Gmax_new;


		        }

		         if (verbose){
			    	 System.out.println("Training method successfully converged " + iter);
			    }
		
				// end of L1 Liblinear
			}
			
			
		} else if (Type.equals("Routine")){


			// initiate one variable to hold the teration's tolerance
			
			double iteration_tol=1;
			
			// initiate the iteration count
			
			int it=0;
			
			
			// Initialize beta
			int pivot= hascostant(this.UseConstant);			
			double betas2[] = new double [data.GetColumnDimension() + pivot];
			double val=0.0;
			double val1=0.0;
			double val2=0.0;
			for (int i=0; i < betas2.length; i++ ) {
				betas2[i]=0;
			}
			
			// Begin the big Iteration and the Optimization Algorithm
			
			while (iteration_tol> tolerance) {
				
				// put a check in the beginning that will stop the process if solution is not identified. This problem is known as 'failed to converge'
				
	            if (it>maxim_Iteration) {
						break;				
				}
	

	            double second_part []= new double [data.GetColumnDimension()+ pivot];
			// calculate the Probabilities	
	        	double covariancev[]= new double [(data.GetColumnDimension()+ pivot)*(data.GetColumnDimension()+ pivot)];

				
	        	// fill it in
	        	for (int i=0; i <data.GetRowDimension(); i++ ){
	        		double value=0;
	        		if (this.UseConstant){
	        			value=betas2[0];
	        		}
	        		if (this.RegularizationType.equals("L2")){
	        		 for (int j = 0; j < data.GetColumnDimension(); j++) {
	        			 val = data.GetElement(i, j);
			             if (val==0){
			            	 continue;
			             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
	        			 
	        			 value+=betas2[j+pivot]*val;
					 }
	        		} else {
		        		 for (int j = 0; j < data.GetColumnDimension(); j++) {
		        			 if (Math.abs(betas2[j+pivot])>C){
		        				 val = data.GetElement(i, j);
		    		             if (val==0){
		    		            	 continue;
		    		             }
		 			            if (this.usescale){
		 			            	val=Scaler.transform(val, j);
		 			            } 
		        				 
		        			 value+=betas2[j+pivot]*val;
		        			 } else {
		        				 betas2[j+pivot]=0.0;
		        			 }
						 }	        			
	        			
	        		}
					 double probabilities= 1. / (1. + Math.exp(-Math.max(Math.min(value, 35.), -35.)));
					 // compute weighted residual
					 double weighted_residual=0;
					 if (target[i]>0){
						 weighted_residual=(target[i]-probabilities)*weights[i];
					 } else {
						 weighted_residual=-probabilities*weights[i];
					 }
					if (UseConstant){
						second_part[0]+=weighted_residual;
						covariancev[0]+= probabilities * weights[i];
    					if ( this.RegularizationType.equals("L2") || ( this.RegularizationType.equals("L1") )){
    						covariancev[0]+=C;
    					} else {
    						covariancev[0]-=C;
    					}
	        			for (int d=0;d <data.GetColumnDimension(); d++ ){
		        			 val2=data.GetElement(i, d);
				             if (val2==0){
				            	 continue;
				             }
		        			if (val2!=0.0){
		        				if (this.usescale){
		        					val2=Scaler.transform(val2, d);
		 			            } 
	        				covariancev[d+1]+=val2 * probabilities * weights[i];
		        			}
	        			}
					}

	        		for (int j=0; j <data.GetColumnDimension(); j++ ){
	        			 val1=data.GetElement(i, j);
			             if (val1==0){
			            	 continue;
			             }
	        			if (val1!=0.0){
	        				if (this.usescale){
	        					val1=Scaler.transform(val1, j);
	 			            } 
	        			second_part[j+pivot]+=val1 * weighted_residual ;
						if (UseConstant){
	        				covariancev[(j+1)*(data.GetColumnDimension()+ 1) ]+= val1 * probabilities * weights[i];
						}
	        			for (int d=0;d <data.GetColumnDimension(); d++ ){
		        			 val2=data.GetElement(i, d);
				             if (val2==0){
				            	 continue;
				             }
		        			if (val2!=0.0){
		        				if (this.usescale){
		        					val2=Scaler.transform(val2, d);
		 			            } 
		        				
	        				covariancev[(j+pivot)*(data.GetColumnDimension()+pivot) +d+pivot]+= val1* val2 * probabilities * weights[i];
	        				if (j==d){
	        					// add regularization in diagonals
	        					if ( this.RegularizationType.equals("L2") || ( this.RegularizationType.equals("L1"))){
	        					covariancev[(j+pivot)*(data.GetColumnDimension()+pivot) +d+pivot]+=C;
	        				} else {
	        					covariancev[(j+pivot)*(data.GetColumnDimension()+pivot) +d+pivot]-=C;
	        				}
	        					
	        				}
	        			}
	        			}
	        		} 
	        			
	        	}
	        		
	        		
	        		
	        		
	        	}
	        	// betas' updates
	        	double Betas []= new double [betas2.length];
	        	//inverse that matrix with qr decomposition if threads less-equal than 1
	        	if (this.threads<=1){
	        	manipulate.matrixoperations.Inverse.GetInversethis(covariancev);
	        	} else {
	        		// else use a multi-threaded LU-decomposition based inverse
	        		manipulate.matrixoperations.Inverse.LUInversethis(covariancev, this.threads);
	        	}
	        	
	        		for (int j=0; j <betas2.length; j++ ){
	        			for (int i=0; i <betas2.length; i++ ){
	        				Betas[j]+=covariancev[i*betas2.length + j] * second_part[i];
	        				} 
       				 if (Math.abs(Betas[j])>iteration_tol){
							 iteration_tol= Math.abs(Betas[j]);
						 }
 						 betas2[j]+=Betas[j];


	                if (Double.isNaN(betas2[0])){
	    					throw new exceptions.ConvergenceException("Regression failed to converge");
	                }


		//------------------ Here ends Newtons-Raphson Algorithm and the Big While-----------------
			}
		//pass over the coefficients
			if (UseConstant){
				constant[0]=betas2[0];
			}
			for (int j=0; j < betas.length; j++){
				betas[j]=betas2[j+pivot];
			}
			it=it+1;
			if (verbose){
				System.out.println("iteration: " + it);
			}
			}
		} else if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double val=0.0;
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
		    	/*
				double BETA [] = new double [betas.length];
				double Constant=0.0;
		    	double pred=0;
		    	*/
		    	double pred=constant[0];
		    	double yi=0.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	if (!this.RegularizationType.equals("L2") && Math.abs(constant[0])<=C){
		    		pred=0.0;
		    	}
		    	// compute score
		    	if (this.RegularizationType.equals("L2")){
		    	for (int j=0; j < data.GetColumnDimension() ;j++){
		    		val = data.GetElement(i, j);
		             if (val==0){
		            	 continue;
		             }
		            if (this.usescale){
		            	val=Scaler.transform(val, j);
		            }
		    		
		    		
		    		pred+=val*betas[j];
		    	}
		    	} else {
		        	for (int j=0; j < data.GetColumnDimension(); j++){
		        		if (Math.abs(betas[j]) >C)
		        			val = data.GetElement(i, j);
				             if (val==0){
				            	 continue;
				             }
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
			    		pred+=val*betas[j];
			    	}
		    	}
		    	// pred to probability
		    	 pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));
		    	 if (UseConstant){
		    		 boolean superceeds_regularization=true;
		    		 double gradient=(pred - yi) ;
		    		 if (this.RegularizationType.equals("L2")){
		    			 gradient+=C*constant[0];
		    		 } else{
		    			 //get sign
		    			 double sign=-1;
		    			 
		    			 if (constant[0]>0){
		    				 sign=1.0;
		 		    	}
		    			 if (sign * constant[0]  <= C  && nc!=0.0){
		    				// gradient=0.0;
		    				 superceeds_regularization=false;
		    			 } else{
		    				 gradient+=C*sign*constant[0];
		    			 }
		    		 }
		    		if (superceeds_regularization){
		    		 nc+=gradient*gradient;
		    		 double move=(this.learn_rate*gradient)/Math.sqrt(nc+0.00000000000001);
		    		 constant[0]=constant[0]-move;
		    		 
		    		} else {
		    			// nc=(pred - yi)*(pred - yi);
		    			 //constant[0]=(pred - yi);
		    		}
		    		 
		    	 }
		    	 for (int j=0; j < data.GetColumnDimension();j++){
		    		 boolean superceeds_regularization=true;
	        			val = data.GetElement(i, j);
			             if (val==0){
			            	 continue;
			             }
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
		    		 double gradient=(pred - yi) *val;
		    		 if (this.RegularizationType.equals("L2")){
		    			 gradient+=C*betas[j];
		    		 } else{
		    			 //get sign
		    			 double sign=-1;
		    			 if (betas[j]>0){
		    				 sign=1.0;
		    				  
		 		    	}
		    			 if (sign * betas[j]  <= C && n[j]!=0.0){
		    				// gradient=0.0;
		    				 superceeds_regularization=false;
		    			 } else{
		    				 gradient+=C*sign*betas[j];
		    			 }
		    		 }
		    		 
			    		if (superceeds_regularization){
				    		 n[j]+=gradient*gradient;
				    		 double move=(this.learn_rate*gradient)/Math.sqrt(n[j]+0.00000000000001);
				    		 betas[j]=betas[j]-move;
				    		 if (Math.abs(move)>iteration_tol){
								 iteration_tol= Math.abs(move);
							 }
				    		} else {
				    			//n[j]=((pred - yi)*data.GetElement(i, j))*((pred - yi)*data.GetElement(i, j));
				    			// betas[j]=(pred - yi)*data.GetElement(i, j);
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
			

			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[columndimension]; // sum of squared gradients
			double val=0.0;
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
				    		 Constant= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	}
		    	
		    	// other features
			    for (int j=0; j < columndimension; j++){	
			    	 val = data.GetElement(i, j);
		             if (val==0){
		            	 continue;
		             }
			    	double sign=1.0;			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    	
			            if (this.usescale){
			            	val=Scaler.transform(val, j);
			            }
			    pred+= BETA[j]*val;	
			    

			    	
			    }			     
			    pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0))); 
			    

			    
			    double gradient= 0;
			    
				 if (target[i]>0){
					 gradient=  (pred - 1.0);
					    //System.out.println("it is");
				 } else {
					 gradient=  (pred) ;
					   // System.out.println("it is NOT");
				 }
				 
				if (UseConstant){		                    
					double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

					constant[0]+=gradient-move*Constant;
					nc+=gradient*gradient;
				 }
			    //updates
				//print.Print(betas, 5);
			    for (int j=0; j < columndimension; j++){
		
			    	 	 val = data.GetElement(i, j);
			             if (val==0){
			            	 continue;
			             }
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
			    	double gradientx=gradient*val ;
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
			    		 constant[0]= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
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
		    		 betas[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
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
		if ( !this.RegularizationType.equals("L2") &&  !this.RegularizationType.equals("L1") &&Type.equals("Liblinear") ){
			throw new IllegalStateException(" No regularization is supported by SGD and Routine methods" );	
		}
		if ( !Type.equals("Liblinear")  && !Type.equals("Routine") && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD,FTRL, Routine or Liblinear methods" );	
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
			if (has.size()!=2){
				throw new IllegalStateException(" target array needs to have exactly 2 values: -1 and 1" );	
			}
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		        double f = it.next();
		        if (f!=-1.0 && f!=1.0){
		        	throw new IllegalStateException("target array needs to have values: -1 and 1");
		    }
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
			if (RegularizationType.equals("L2")){
				
		    	//initialise other values
				int l = data.GetRowDimension();
			    int i, s, iter = 0;
			    double xTx[] = new double[l];
			    int max_iter = maxim_Iteration;
			    int index[] = new int[l];
			    double alpha[] = new double[2 * l]; // store alpha and C - alpha
			    int max_inner_iter = 100; // for inner Newton
			    double innereps = 1e-2;
			    double innereps_min = Math.min(1e-8, tolerance);
			    double upper_bound[] = new double[] {1/C, 0,1/C};            
			    double val=0.0;
				
		    // Initial alpha can be set here. Note that
		    // 0 < alpha[i] < upper_bound[GETI(i)]
		    // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
		    for (i = 0; i < l; i++) {
		    	 index[i] = i;
		        alpha[2 * i] = Math.min(0.001 * upper_bound[GETI(target, i)], 1e-8);
		        alpha[2 * i + 1] = upper_bound[GETI(target, i)] - alpha[2 * i];
		    }
		    // check if data is sorted by row and sort it if now
	        if (sparse_set==false){
		    if (!data.IsSortedByRow()){
		    	data.convert_type();
		    }
		
	        }
		   



		    for (i = 0; i < l; i++) {
		    	if (UseConstant){
		    		 xTx[i]+=1;
		    		 constant[0]+=target[i] *weights[i] * alpha[2 * i];
		    	}
		        for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		            val = data.valuespile[j];
		            if (this.usescale){
		            	val=Scaler.transform(val, data.mainelementpile[j]);
		            }
		            xTx[i] += val*val*weights[i];
		            betas[data.mainelementpile[j]] += target[i]*weights[i] * alpha[2 * i] * val;
		        }
		       
		    }

		    while (iter < max_iter) {
		        for (i = 0; i < l; i++) {
		            int j = i +random.nextInt(l - i) ;
		            swap(index, i, j);
		        }
		        
		        int newton_iter = 0;
		        double Gmax = 0;
		        for (s = 0; s < l; s++) {
		            i = index[s];
		            double yi = target[i];
		            double C = upper_bound[GETI(target, i)];
		            double ywTx = 0, xisq = xTx[i];
			    	if (UseConstant){
			    		 ywTx +=constant[0];
			    	}
			        for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
			        	val = data.valuespile[j];
			            if (this.usescale){
			            	val=Scaler.transform(val, data.mainelementpile[j]);
			            }
		                ywTx += betas[data.mainelementpile[j]] *val;
		            }
		            ywTx *= target[i];
		            double a = xisq, b = ywTx;

		            // Decide to minimize g_1(z) or g_2(z)
		            int ind1 = 2 * i, ind2 = 2 * i + 1, sign = 1;
		            if (0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0) {
		                ind1 = 2 * i + 1;
		                ind2 = 2 * i;
		                sign = -1;
		            }

		            //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
		            double alpha_old = alpha[ind1];
		            double z = alpha_old;
		            if (C - z < 0.5 * C) z = 0.1 * z;
		            double gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
		            Gmax = Math.max(Gmax, Math.abs(gp));

		            // Newton method on the sub-problem
		            final double eta = 0.1; // xi in the paper
		            int inner_iter = 0;
		            while (inner_iter <= max_inner_iter) {
		                if (Math.abs(gp) < innereps) break;
		                double gpp = a + C / (C - z) / z;
		                double tmpz = z - gp / gpp;
		                if (tmpz <= 0)
		                    z *= eta;
		                else
		                    // tmpz in (0, C)
		                    z = tmpz;
		                gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
		                newton_iter++;
		                inner_iter++;
		            }

		            if (inner_iter > 0) // update w
		            {
		                alpha[ind1] = z;
		                alpha[ind2] = C - z;
				    	if (UseConstant){
				    		constant[0]+= sign * (z - alpha_old) * yi;
				    	}
				        for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
				        	val = data.valuespile[j];
				            if (this.usescale){
				            	val=Scaler.transform(val, data.mainelementpile[j]);
				            }
		                    betas[data.mainelementpile[j]] += sign * (z - alpha_old) * yi *val;
		                }
		            }
		        }

		        iter++;
		        if (Gmax < tolerance) break;
		        if (newton_iter <= l / 10) {
		            innereps = Math.max(innereps_min, 0.1 * innereps);
		        }

		    }

		    
		    if (this.verbose){
		    	 System.out.println("Training method successfully converged " + iter);
		    }


		    
		    
				// end of L2 Liblinear
			} else if (RegularizationType.equals("L1")){
				
		    	int l = data.GetRowDimension();
			    int w_size = data.GetColumnDimension();
		        int j, s, newton_iter = 0, iter = 0;
		        int max_newton_iter = 100;
		        int max_iter = maxim_Iteration;
		        int max_num_linesearch = 20;
		        int active_size;
		        int QP_active_size;
		        double nu = 1e-12;
		        double inner_eps = 1;
		        double sigma = 0.01;
		        double w_norm, w_norm_new;
		        double z, G, H;
		        double Gnorm1_init = 0; // eclipse moans this variable might not be initialized
		        double Gmax_old = Double.POSITIVE_INFINITY;
		        double Gmax_new, Gnorm1_new;
		        double QP_Gmax_old = Double.POSITIVE_INFINITY;
		        double QP_Gmax_new, QP_Gnorm1_new;
		        double delta, negsum_xTd, cond;
		        double val=0.0;
		        int[] index = new int[w_size];
		        int indexc=0;
		        double[] Hdiag = new double[w_size];
		        double hdiag=0;
		        double[] Grad = new double[w_size];
		        double grad=0;
		        double[] wpd = new double[w_size];
		        double wp=0;
		        double[] xjneg_sum = new double[w_size];
		        double xjneg_s=0;
		        double[] xTd = new double[l];
		        double[] exp_wTx = new double[l];
		        double[] exp_wTx_new = new double[l];
		        double[] tau = new double[l];
		        double[] D = new double[l];

		        double[] C = {1/this.C, 0, 1/this.C};

		        // check if dataset is sorted by column and sort it if not
		        
		        if (sparse_set==false){
			    if (!data.IsSortedByColumn()){
			    	data.convert_type();
			    }

		        }
		        

			    
		        w_norm = 0;
		        if (UseConstant){
		        	 w_norm +=constant[0];
		        	 wp=constant[0];
		        	 indexc=0;
		        	 xjneg_s=0;
		        	 for (int i=0; i <l; i++) {
		        		 exp_wTx[i] += constant[0] *weights[i];
			                if (target[i] < 0) {
			                    xjneg_s += C[GETI(target, i)];
			                }
		        	 }
		        }
		        for (j = 0; j < w_size; j++) {
		            w_norm += Math.abs(betas[j]);
		            wpd[j] = betas[j];
		            index[j] = j;
		            xjneg_sum[j] = 0;
		            for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
		                 val = data.valuespile[i];
		                 if (this.usescale){
				            	val=Scaler.transform(val,j);
				            }
		                 
		                int ind =data.mainelementpile[i];
		                exp_wTx[ind] += betas[j] * val*weights[ind];
		                if (target[ind] < 0) {
		                    xjneg_sum[j] += C[GETI(target, ind)] * val;
		                }
		            }
		        }
		        for (j = 0; j < l; j++) {
		            exp_wTx[j] = Math.exp(exp_wTx[j]);
		            double tau_tmp = 1 / (1 + exp_wTx[j]);
		            tau[j] = C[GETI(target, j)] * tau_tmp;
		            D[j] = C[GETI(target, j)] * exp_wTx[j] * tau_tmp * tau_tmp;
		        }

		        while (newton_iter < max_newton_iter) {
		            Gmax_new = 0;
		            Gnorm1_new = 0;
		            active_size = w_size;

		            if (UseConstant){
		            	j=indexc;
		                hdiag = nu;
		                grad = 0;

		                double tmp = 0;
		                for (int i=0; i <l; i++) {
		                    hdiag += D[i];
		                    tmp +=  tau[i];
		                }
		                grad = -tmp + xjneg_s;
		                double Gp = grad + 1;
		                double Gn = grad - 1;
		                double violation = 0;
		                if (constant[0] == 0) {
		                    if (Gp < 0)
		                        violation = -Gp;
		                    else if (Gn > 0)
		                        violation = Gn;

		                } else if (constant[0] > 0)
		                    violation = Math.abs(Gp);
		                else
		                    violation = Math.abs(Gn);

		                Gmax_new = Math.max(Gmax_new, violation);
		                Gnorm1_new += violation;
		            }
		            	
		            
		            for (s = 0; s < active_size; s++) {
		                j = index[s];
		                Hdiag[j] = nu;
		                Grad[j] = 0;

		                double tmp = 0;
		                for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
		                	int ind = data.mainelementpile[i];
		                	val =data.valuespile[i];
				            if (this.usescale){
				            	val=Scaler.transform(val, j);
				            }
		                    Hdiag[j] +=val* val * D[ind];
		                    tmp += val * tau[ind];
		                }
		                Grad[j] = -tmp + xjneg_sum[j];

		                double Gp = Grad[j] + 1;
		                double Gn = Grad[j] - 1;
		                double violation = 0;
		                if (betas[j] == 0) {
		                    if (Gp < 0)
		                        violation = -Gp;
		                    else if (Gn > 0)
		                        violation = Gn;
		                    //outer-level shrinking
		                    else if (Gp > Gmax_old / l && Gn < -Gmax_old / l) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    }
		                } else if (betas[j] > 0)
		                    violation = Math.abs(Gp);
		                else
		                    violation = Math.abs(Gn);

		                Gmax_new = Math.max(Gmax_new, violation);
		                Gnorm1_new += violation;
		            }

		            if (newton_iter == 0) Gnorm1_init = Gnorm1_new;

		            if (Gnorm1_new <= tolerance * Gnorm1_init) break;

		            iter = 0;
		            QP_Gmax_old = Double.POSITIVE_INFINITY;
		            QP_active_size = active_size;

		            for (int i = 0; i < l; i++)
		                xTd[i] = 0;
		            // optimize QP over wpd
		            while (iter < max_iter) {
		                QP_Gmax_new = 0;
		                QP_Gnorm1_new = 0;

		                for (j = 0; j < QP_active_size; j++) {
		                    int i = random.nextInt(QP_active_size - j);
		                    swap(index, i, j);
		                }

		                if (UseConstant){
		                    H = hdiag;

		                    G = grad + (wp - constant[0]) * nu;
		                    for (int i = 0; i < l; i++) {
		                        G +=  D[i] * xTd[i];
		                    }

		                    double Gp = G + 1;
		                    double Gn = G - 1;
		                    double violation = 0;
		                    if (wp == 0) {
		                        if (Gp < 0)
		                            violation = -Gp;
		                        else if (Gn > 0)
		                            violation = Gn;
		                    } else if (wp > 0)
		                        violation = Math.abs(Gp);
		                    else
		                        violation = Math.abs(Gn);

		                    QP_Gmax_new = Math.max(QP_Gmax_new, violation);
		                    QP_Gnorm1_new += violation;

		                    // obtain solution of one-variable problem
		                    if (Gp < H * wp)
		                        z = -Gp / H;
		                    else if (Gn > H * wp)
		                        z = -Gn / H;
		                    else
		                        z = -wp;

		                    if (Math.abs(z) < 1.0e-12) continue;
		                    z = Math.min(Math.max(z, -10.0), 10.0);

		                    wp += z;

		                    for (int i=0; i <l; i++) {
		                        xTd[i] +=  z;
		                    }
		                	
		                }
		                for (s = 0; s < QP_active_size; s++) {
		                    j = index[s];
		                    H = Hdiag[j];

		                    G = Grad[j] + (wpd[j] - betas[j]) * nu;
		                    for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
		                    	int ind = data.mainelementpile[i];
		                    	val=data.valuespile[i];
		    		            if (this.usescale){
		    		            	val=Scaler.transform(val, j);
		    		            }
		                    	
		                        G += val * D[ind] * xTd[ind];
		                    }

		                    double Gp = G + 1;
		                    double Gn = G - 1;
		                    double violation = 0;
		                    if (wpd[j] == 0) {
		                        if (Gp < 0)
		                            violation = -Gp;
		                        else if (Gn > 0)
		                            violation = Gn;
		                        //inner-level shrinking
		                        else if (Gp > QP_Gmax_old / l && Gn < -QP_Gmax_old / l) {
		                            QP_active_size--;
		                            swap(index, s, QP_active_size);
		                            s--;
		                            continue;
		                        }
		                    } else if (wpd[j] > 0)
		                        violation = Math.abs(Gp);
		                    else
		                        violation = Math.abs(Gn);

		                    QP_Gmax_new = Math.max(QP_Gmax_new, violation);
		                    QP_Gnorm1_new += violation;

		                    // obtain solution of one-variable problem
		                    if (Gp < H * wpd[j])
		                        z = -Gp / H;
		                    else if (Gn > H * wpd[j])
		                        z = -Gn / H;
		                    else
		                        z = -wpd[j];

		                    if (Math.abs(z) < 1.0e-12) continue;
		                    z = Math.min(Math.max(z, -10.0), 10.0);

		                    wpd[j] += z;

		                    for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
		                    	int ind = data.mainelementpile[i];
		                    	val=data.valuespile[i];
		    		            if (this.usescale){
		    		            	val=Scaler.transform(val, j);
		    		            }
		                        xTd[ind] +=val * z;
		                    }
		                }

		                iter++;

		                if (QP_Gnorm1_new <= inner_eps * Gnorm1_init) {
		                    //inner stopping
		                    if (QP_active_size == active_size)
		                        break;
		                    //active set reactivation
		                    else {
		                        QP_active_size = active_size;
		                        QP_Gmax_old = Double.POSITIVE_INFINITY;
		                        continue;
		                    }
		                }

		                QP_Gmax_old = QP_Gmax_new;
		            }


		            delta = 0;
		            w_norm_new = 0;
		            if (UseConstant){
		            	delta += grad * (wp - constant[0]);
		            	if (wp != 0) w_norm_new += Math.abs(wp);
		            }
		            
		            for (j = 0; j < w_size; j++) {
		                delta += Grad[j] * (wpd[j] - betas[j]);
		                if (wpd[j] != 0) w_norm_new += Math.abs(wpd[j]);
		            }
		            delta += (w_norm_new - w_norm);

		            negsum_xTd = 0;
		            for (int i = 0; i < l; i++)
		                if (target[i] < 0) negsum_xTd += C[GETI(target, i)] * xTd[i];

		            int num_linesearch;
		            for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
		                cond = w_norm_new - w_norm + negsum_xTd - sigma * delta;

		                for (int i = 0; i < l; i++) {
		                    double exp_xTd = Math.exp(xTd[i]);
		                    exp_wTx_new[i] = exp_wTx[i] * exp_xTd;
		                    cond += C[GETI(target, i)] * Math.log((1 + exp_wTx_new[i]) / (exp_xTd + exp_wTx_new[i]));
		                }

		                if (cond <= 0) {
		                    w_norm = w_norm_new;
		                    if (UseConstant){
		                    	constant[0]=wp;
		                    }
		                    for (j = 0; j < w_size; j++)
		                    	betas[j] = wpd[j];
		                    for (int i = 0; i < l; i++) {
		                        exp_wTx[i] = exp_wTx_new[i];
		                        double tau_tmp = 1 / (1 + exp_wTx[i]);
		                        tau[i] = C[GETI(target, i)] * tau_tmp;
		                        D[i] = C[GETI(target, i)] * exp_wTx[i] * tau_tmp * tau_tmp;
		                    }
		                    break;
		                } else {
		                    w_norm_new = 0;
		                    if (UseConstant){
		                    	
		                        wp = (constant[0] + wp) * 0.5;
		                        if (wp != 0) w_norm_new += Math.abs(wp);
		                    }
		                    for (j = 0; j < w_size; j++) {
		                        wpd[j] = (betas[j] + wpd[j]) * 0.5;
		                        if (wpd[j] != 0) w_norm_new += Math.abs(wpd[j]);
		                    }
		                    delta *= 0.5;
		                    negsum_xTd *= 0.5;
		                    for (int i = 0; i < l; i++)
		                        xTd[i] *= 0.5;
		                }
		            }

		            // Recompute some info due to too many line search steps
		            if (num_linesearch >= max_num_linesearch) {
		            	for (int i = 0; i < l; i++)
		                    exp_wTx[i] = 0;

		                if (UseConstant){
		                if (constant[0]==0) continue;
	                    for (int m=0; m <l; m++) {
	                        exp_wTx[m] += constant[0];
	                    }
		                }
		                for (int i = 0; i < w_size; i++) {
		                    if (betas[i] == 0) continue;
		                    for (int m=data.indexpile[i]; m <data.indexpile[i+1]; m++) {
		                    	
		                    	val=data.valuespile[m];
		    		            if (this.usescale){
		    		            	val=Scaler.transform(val, i);
		    		            }
		                    	
		                    	
		                        exp_wTx[data.mainelementpile[m]] += betas[i] *val ;
		                    }
		                }
		                for (int i = 0; i < l; i++)
		                    exp_wTx[i] = Math.exp(exp_wTx[i]);
		            }

		            if (iter == 1) inner_eps *= 0.25;

		            newton_iter++;
		            Gmax_old = Gmax_new;


		        }

		         if (verbose){
			    	 System.out.println("Training method successfully converged " + iter);
			    }
		
				// end of L1 Liblinear
			}
			
			
		} else if (Type.equals("Routine")){

			// initiate one variable to hold the teration's tolerance
			
			double iteration_tol=1;
			
			// initiate the iteration count
			
			int it=0;
			
	        if (sparse_set==false){
		    if (!data.IsSortedByRow()){
		    	data.convert_type();
		    }
	
	        }

			double val=0.0;
			double val1=0.0;
			double val2=0.0;
			// Initialize beta
			int pivot= hascostant(this.UseConstant);			
			double betas2[] = new double [data.GetColumnDimension() + pivot];

			for (int i=0; i < betas2.length; i++ ) {
				betas2[i]=0;
			}
			
			// Begin the big Iteration and the Optimization Algorithm
			
			while (iteration_tol> tolerance) {
				
				// put a check in the beginning that will stop the process if solution is not identified. This problem is known as 'failed to converge'
				
	            if (it>maxim_Iteration) {
						break;				
				}
	

	            double second_part []= new double [data.GetColumnDimension()+ pivot];
			// calculate the Probabilities	
	        	double covariancev[]= new double [(data.GetColumnDimension()+ pivot)*(data.GetColumnDimension()+ pivot)];

				
	        	// fill it in
	        	for (int i=0; i <data.GetRowDimension(); i++ ){
	        		double value=0;
	        		if (this.UseConstant){
	        			value=betas2[0];
	        		}
	        		if (this.RegularizationType.equals("L2")){
	        			//System.out.println(data.indexpile[i] + " " + data.indexpile[i] + lengths[i]);
		        		 for (int j = data.indexpile[i]; j < data.indexpile[i+1] ; j++) {
		        			 val = data.valuespile[j];
		 		            if (this.usescale){
		 		            	val=Scaler.transform(val, data.mainelementpile[j]);
		 		            }
		        			 
		        			 value+=betas2[data.mainelementpile[j]+pivot]*val;
					 }
	        		} else {
		        		 for (int j = data.indexpile[i]; j < data.indexpile[i+1]; j++) {
		        			 if (Math.abs(betas2[data.mainelementpile[j]+pivot])>C){
		        				 val = data.valuespile[j];
				 		            if (this.usescale){
				 		            	val=Scaler.transform(val, data.mainelementpile[j]);
				 		            }
			        			 value+=betas2[data.mainelementpile[j]+pivot]*val;
		        			 } else {
		        				 betas2[data.mainelementpile[j]+pivot]=0.0;
		        			 }
						 }	        			
	        			
	        		}
					 double probabilities= 1. / (1. + Math.exp(-Math.max(Math.min(value, 35.), -35.)));
					 // compute weighted residual
					 double weighted_residual=0;
					 if (target[i]>0){
						 weighted_residual=(target[i]-probabilities)*weights[i];
					 } else {
						 weighted_residual=-probabilities*weights[i];
					 }
					if (UseConstant){
						second_part[0]+=weighted_residual;
						covariancev[0]+= probabilities * weights[i];
    					if ( this.RegularizationType.equals("L2") || ( this.RegularizationType.equals("L1") )){
    						covariancev[0]+=C;
    					} else {
    						covariancev[0]-=C;
    					}
	   	        		 for (int d = data.indexpile[i]; d < data.indexpile[i+1]; d++) {

			        		val2 = data.valuespile[d];
				            
		        			if (val2!=0.0){
		        				if (this.usescale){
					            	val2=Scaler.transform(val2, data.mainelementpile[d]);
					            }
	        				covariancev[data.mainelementpile[d]+1]+=val2 * probabilities * weights[i];
		        			}
	        			}
					}

	        		 for (int j = data.indexpile[i]; j < data.indexpile[i+1]; j++) {
	        			 val1= data.valuespile[j];
	        			if (val1!=0.0){
	    		            if (this.usescale){
	    		            	val1=Scaler.transform(val1, data.mainelementpile[j]);
	    		            }
	        			second_part[data.mainelementpile[j]+pivot]+=val1 * weighted_residual ;
						if (UseConstant){
	        				covariancev[(data.mainelementpile[j]+1)*(data.GetColumnDimension()+ 1) ]+= val1 * probabilities * weights[i];
						}
	   	        		 for (int d = data.indexpile[i]; d < data.indexpile[i+1]; d++) {
			        		 val2=data.valuespile[d];
		        			if (val2!=0.0){
		        				if (this.usescale){
					            	val2=Scaler.transform(val2, data.mainelementpile[d]);
					            }
	        				covariancev[(data.mainelementpile[j]+pivot)*(data.GetColumnDimension()+pivot) +data.mainelementpile[d]+pivot]+= val1* val2 * probabilities * weights[i];
	        				if (data.mainelementpile[j]==data.mainelementpile[d]){
	        					// add regularization in diagonals
	        					if ( this.RegularizationType.equals("L2") || ( this.RegularizationType.equals("L1"))){
	        					covariancev[(data.mainelementpile[j]+pivot)*(data.GetColumnDimension()+pivot) +data.mainelementpile[d]+pivot]+=C;
	        				} else {
	        					covariancev[(data.mainelementpile[j]+pivot)*(data.GetColumnDimension()+pivot) +data.mainelementpile[d]+pivot]-=C;
	        				}
	        					
	        				}
	        			}
	        			}
	        		} 
	        			
	        	}        		
	        		
	        	}
	        	// betas' updates
	        	double Betas []= new double [betas2.length];
	        	//inverse that matrix with qr decomposition if threads less-equal than 1
	        	if (this.threads<=1){
	        	manipulate.matrixoperations.Inverse.GetInversethis(covariancev);
	        	} else {
	        		// else use a multi-threaded LU-decomposition based inverse
	        		manipulate.matrixoperations.Inverse.LUInversethis(covariancev, this.threads);
	        	}
	        	
	        		for (int j=0; j <betas2.length; j++ ){
	        			for (int i=0; i <betas2.length; i++ ){
	        				Betas[j]+=covariancev[i*betas2.length + j] * second_part[i];
	        				} 
       				 if (Math.abs(Betas[j])>iteration_tol){
							 iteration_tol= Math.abs(Betas[j]);
						 }
 						 betas2[j]+=Betas[j];


	                if (Double.isNaN(betas2[0])){
	    					throw new exceptions.ConvergenceException("Regression failed to converge");
	                }


		//------------------ Here ends Newtons-Raphson Algorithm and the Big While-----------------
			}
		//pass over the coefficients
			if (UseConstant){
				constant[0]=betas2[0];
			}
			for (int j=0; j < betas.length; j++){
				betas[j]=betas2[j+pivot];
			}
			it=it+1;
			if (verbose){
				System.out.println("iteration: " + it);
			}
			}

		} else if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[data.GetColumnDimension()]; // sum of squared gradients
			double val=0.0;
		    // check if data is sorted by row and sort it if now
	        if (sparse_set==false){
		    if (!data.IsSortedByRow()){
		    	data.convert_type();
		    }

	        }

			
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	/*
				double BETA [] = new double [betas.length];
				double Constant=0.0;
		    	double pred=0;
		    	*/
		    	double pred=constant[0];
		    	double yi=0.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	if (!this.RegularizationType.equals("L2") && Math.abs(constant[0])<=C){
		    		pred=0.0;
		    	}
		    	// compute score
		    	if (this.RegularizationType.equals("L2")){
		    		 for (int j = data.indexpile[i]; j < data.indexpile[i+1] ; j++) {
		    			 val = data.valuespile[j];
				            if (this.usescale){
				            	val=Scaler.transform(val, data.mainelementpile[j]);
				            } 
		    			 
		    		pred+=val*betas[data.mainelementpile[j]];
		    	}
		    	} else {
		    		 for (int j = data.indexpile[i]; j < data.indexpile[i+1]; j++) {
		        		if (Math.abs(betas[data.mainelementpile[j]]) >C)
		        			val = data.valuespile[j];
			            if (this.usescale){
			            	val=Scaler.transform(val, data.mainelementpile[j]);
			            } 
			    		pred+=val*betas[data.mainelementpile[j]];
			    	}
		    	}
		    	// pred to probability
		    	 pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));
		    	 if (UseConstant){
		    		 boolean superceeds_regularization=true;
		    		 double gradient=(pred - yi) ;
		    		 if (this.RegularizationType.equals("L2")){
		    			 gradient+=C*constant[0];
		    		 } else{
		    			 //get sign
		    			 double sign=-1;
		    			 
		    			 if (constant[0]>0){
		    				 sign=1.0;
		 		    	}
		    			 if (sign * constant[0]  <= C  && nc!=0.0){
		    				// gradient=0.0;
		    				 superceeds_regularization=false;
		    			 } else{
		    				 gradient+=C*sign*constant[0];
		    			 }
		    		 }
		    		if (superceeds_regularization){
		    		 nc+=gradient*gradient;
		    		 double move=(this.learn_rate*gradient)/Math.sqrt(nc+0.00000000000001);
		    		 constant[0]=constant[0]-move;
		    		 
		    		} else {
		    			// nc=(pred - yi)*(pred - yi);
		    			// constant[0]=(pred - yi);
		    		}
		    		 
		    	 }
		    	 for (int j = data.indexpile[i]; j < data.indexpile[i+1]; j++) {
		    		 boolean superceeds_regularization=true;
		    			val = data.valuespile[j];
			            if (this.usescale){
			            	val=Scaler.transform(val, data.mainelementpile[j]);
			            } 
		    		 double gradient=(pred - yi) * val;
		    		 if (this.RegularizationType.equals("L2")){
		    			 gradient+=C*betas[data.mainelementpile[j]];
		    		 } else{
		    			 //get sign
		    			 double sign=-1;
		    			 if (betas[data.mainelementpile[j]]>0){
		    				 sign=1.0;
		    				  
		 		    	}
		    			 if (sign * betas[data.mainelementpile[j]]  <= C && n[data.mainelementpile[j]]!=0.0){
		    				// gradient=0.0;
		    				 superceeds_regularization=false;
		    			 } else{
		    				 gradient+=C*sign*betas[data.mainelementpile[j]];
		    			 }
		    		 }
		    		 
			    		if (superceeds_regularization){
				    		 n[data.mainelementpile[j]]+=gradient*gradient;
				    		 double move=(this.learn_rate*gradient)/Math.sqrt(n[data.mainelementpile[j]]+0.00000000000001);
				    		 betas[data.mainelementpile[j]]-=move;
				    		 if (Math.abs(move)>iteration_tol){
								 iteration_tol= Math.abs(move);
							 }
				    		} else {
				    			//n[data.mainelementpile[j]]=((pred - yi)*data.valuespile[j])*((pred - yi)*data.valuespile[j]);
				    			 //betas[data.mainelementpile[j]]=(pred - yi)*data.valuespile[j];
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
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double val=0.0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[columndimension]; // sum of squared gradients
	        if (sparse_set==false){
		    if (!data.IsSortedByRow()){
		    	data.convert_type();
		    }
	        }
			
	
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	//System.out.println(Arrays.toString(data[i]));
		    	
				double BETA [] = new double[data.GetColumnDimension()];
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
				    		 Constant= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	}
		    	
		    	// other features
		    	int st=data.indexpile[i];
		    	for (int j = st; j < data.indexpile[i+1]; j++) {
			    	double sign=1.0;			    	
			    	if (betas[data.mainelementpile[j] ] <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[data.mainelementpile[j]]  <= l1C){
			    		 BETA[j-st]=0 ;
			    	 } else {
			    		 BETA[j-st]=  (sign * l1C - betas[data.mainelementpile[j]]) / (( 0.1+Math.sqrt(n[data.mainelementpile[j]])) / this.learn_rate + C);
			    	
			    	 }
			    	 val = data.valuespile[j];
			            if (this.usescale){
			            	val=Scaler.transform(val, data.mainelementpile[j]);
			            }
			    pred+= BETA[j-st]*val;	
			    

			    	
			    }			     
			    pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0))); 
			    

			    
			    double gradient= 0;
			    
				 if (target[i]>0){
					 gradient=  (pred - 1.0);
					    //System.out.println("it is");
				 } else {
					 gradient=  (pred) ;
					   // System.out.println("it is NOT");
				 }
				 
				if (UseConstant){		                    
					double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

					constant[0]+=gradient-move*Constant;
					nc+=gradient*gradient;
				 }
			    //updates
				//print.Print(betas, 5);
		    	for (int j = st; j < data.indexpile[i+1]; j++) {
		    		 val = data.valuespile[j];
			            if (this.usescale){
			            	val=Scaler.transform(val, data.mainelementpile[j]);
			            }
			    	double gradientx=gradient*val ;
			    	//System.out.println(" gradient: " + gradientx);
			    	double move=(Math.sqrt(n[data.mainelementpile[j]] + gradientx * gradientx) - Math.sqrt(n[data.mainelementpile[j]])) / this.learn_rate;
			    	betas[data.mainelementpile[j]] += gradientx - move * BETA[j-st];
                    n[data.mainelementpile[j]] += gradientx * gradientx;	

                    // check for highest tolerance
      				 if (Math.abs(BETA[j-st])>iteration_tol){
							 iteration_tol= Math.abs(BETA[j-st]);
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
			    		 constant[0]= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
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
		    		 betas[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
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
		return "classifier";
	}
	@Override
	public boolean SupportsWeights() {
		return true;
	}
	@Override
	public String GetName() {
		return "binary_logistic";
	}
	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: Binary Regularized Logistic Regression");
		System.out.println("Classes: 2 (Binary)");
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
		constant=new double []{0.0};
		betas=null;
		RegularizationType="L2";
		C=1.0;
		l1C=1.0;
		Type="Liblinear";
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
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
		binarylogistic br = new binarylogistic();
		br.constant=this.constant;
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		br.RegularizationType=this.RegularizationType;
		br.C=this.C;
		br.l1C=this.l1C;
		br.Type=this.Type;
		br.threads=this.threads;
		br.UseConstant=this.UseConstant;
		br.columndimension=this.columndimension;
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
	 * this method corresponds to the following define in the C version:
	 * #define GETI(i) (y[i]+1)
	 */
	private static int GETI(double[] y, int i) {
		if (y[i]>0){
			return 2;
		} else{
			return 0;
		} 
	}
	private static void swap(int[] array, int idxA, int idxB) {
	    int temp = array[idxA];
	    array[idxA] = array[idxB];
	    array[idxB] = temp;
	}
	/**
	 * 
	 * @param usesesconstant : true if we use constant
	 * @return 1 if true
	 */
	private static int hascostant(boolean usesesconstant){
		if (usesesconstant){
			return 1;
		} else {
			return 0;
		}
	}

	@Override
	public String[] getclasses() {
		return new String []{"-1","1"};
	}

	@Override
	public int getnumber_of_classes() {
		return 2;
	}

	@Override
	public void AddClassnames(String[] names) {
		//not applicable in binary model=not needed
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
		this.seed=seed;
		
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
}
