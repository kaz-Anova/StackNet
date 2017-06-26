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
 *<p> class to implement binary linear support vector machines</p>
 * <ol>
 * <li> LibLinear (L2 and L1 regularization) - R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.<a href="http://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf">LIBLINEAR: A library for large linear classification</a> Journal of Machine Learning Research 9(2008), 1871-1874.</li> 
 * <li> SGD "Stochastic Gradient Descent" with adaptive learning Rate (supports L1 and L2)  </li> 
 * <li> FTRL"Follow The Regularized Leader" (supports L1 and L2), inspired by Tingru's code in Kaggel forums <a href="https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory"> link </a>  </li> 
 * </ol>
 * <p> we optimise for the hinge loss: </p>
 * <pre> max(0, 1-y<sub>i</sub> beta X) </pre>
 * <p> The Sub gradients are computed as : </p>
 * <pre> {-y<sub>i</sub>x<sub>i</sub> + lbeta<sub>i</sub>  if y<sub>i</sub> beta X < 1 }</pre> 
 * <pre> {0 + lbeta<sub>i</sub>  if y<sub>i</sub> beta X >= 1 } </pre> 
 * (Collobert et al , 2001)
 */
public class binaryLSVC implements estimator,classifier,Runnable {

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
	 * Type of algorithm to use. It has to be one of Liblinear, SGD, FTRL
	 */
	public String Type="Liblinear";
	/**
	 * True if we want to scale with highest maximum value
	 */
	public boolean scale=false;
	/**
	 * True if we want to optimize for squared distance or not
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
	public binaryLSVC(){}
	
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
	public binaryLSVC(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	/**
	 * Default constructor for Binary Logistic Regression with fsmatrix data
	 */
	public binaryLSVC(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for Binary Logistic Regression with smatrix data
	 */
	public binaryLSVC(smatrix data){
		
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=value;
				predictions[i][0]=-value;
			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i][1]=value;
			predictions[i][0]=-value;
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
				// 1. / (1. + Math.exp(-value));
				predictions[i]=value;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			//value= 1. / (1. + Math.exp(-value));
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=value;
				predictions[i][0]=-value;
			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i][1]=value;
			predictions[i][0]=-value;
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			//value= 1. / (1. + Math.exp(-value));
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=value;
				predictions[i][0]=-value;
			}

		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i][1]=value;
			predictions[i][0]=-value;
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=value;
			}


		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[1]=value;
				predictions[0]=-value;

		} else {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[1]=value;
			predictions[0]=-value;
		
		
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[1]=value;
				predictions[0]=-value;
			
		} else {

			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(row, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[1]=value;
			predictions[0]=-value;
		
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[1]=value;
				predictions[0]=-value;			


		} else {
			
			double value=constant[0];
			for (int j=start; j < end ; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
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
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=(value >= 0.0) ? 1.0 :0.0 ;

			}
		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(i, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=(value >= 0.0) ? 1.0 :0.0 ;
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
				predictions[i]=(value >= 0.0) ? 1.0 :0.0 ;
			}


		} else {
			
			
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];
			for (int j=data.indexpile[i]; j < data.indexpile[i+1]; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=(value >= 0.0) ? 1.0 :0.0 ;
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
				predictions[i]=(value >= 0.0) ? 1.0 :0.0 ;

			}

		} else {
		for (int i=0; i < predictions.length; i++) {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[i][j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions[i]=(value >= 0.0) ? 1.0 :0.0 ;
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
				//value= 1. / (1. + Math.exp(-value));
				predictions=(value >= 0.0) ? 1.0 :0.0 ;
		} else {
			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=(value >= 0.0) ? 1.0 :0.0 ;
		
		
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
				predictions=(value >= 0.0) ? 1.0 :0.0 ;

			
		} else {

			double value=constant[0];

			for (int j=0; j < columndimension; j++){
				value+=betas[j]*data.GetElement(row, j);
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=(value >= 0.0) ? 1.0 :0.0 ;
		
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
				predictions=(value >= 0.0) ? 1.0 :0.0 ;		


		} else {
			
			double value=constant[0];
			for (int j=start; j < end ; j++){
				value+=betas[data.mainelementpile[j]]*data.valuespile[j];
			}
			//value= 1. / (1. + Math.exp(-value));
			predictions=(value >= 0.0) ? 1.0 :0.0 ;
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
		if ( !Type.equals("Liblinear")  &&  !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Liblinear methods" );	
		}		
		if ( !Type.equals("Liblinear")  && quadratic==true ){
			throw new IllegalStateException(" quadratic loss is only available for liblinear" );	
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
				//Initialise beta
				//betas = new double[data[0].length];
				int l =data.length;
		        int w_size = data[0].length;
		        int i, s, iter = 0;
		        double C, d, G;
		        double[] QD = new double[l];
		        int max_iter = maxim_Iteration;
		        int[] index = new int[l];
		        double[] alpha = new double[l];
		        int active_size = l;
		        // PG: projected gradient, for shrinking and stopping
		        double PG;
		        double PGmax_old = Double.POSITIVE_INFINITY;
		        double PGmin_old = Double.NEGATIVE_INFINITY;
		        double PGmax_new, PGmin_new;
		        double val=0.0;
		        // default solver_type: L2R_L2LOSS_SVC_DUAL
		        double diag[] = new double[] {0.5 / this.C, 0, 0.5 / this.C};
		        double upper_bound[] = new double[] {Double.POSITIVE_INFINITY, 0, Double.POSITIVE_INFINITY};
		        if (quadratic==true) {
		            diag[0] = 0;
		            diag[2] = 0;
		            upper_bound[0] = this.C;
		            upper_bound[2] = this.C;
		        }

		        // Initial alpha can be set here. Note that
		        // 0 <= alpha[i] <= upper_bound[GETI(i)]


		        for (i = 0; i < l; i++) {
		            QD[i] = diag[GETI(target, i)];
		            //ystem.out.println(GETI(target, i));
		            if (UseConstant){
		            	QD[i] +=1;
		            	constant[0]+=target[i] * alpha[i];
			        }
		            for (int j=0; j < w_size; j++) {
		            	val=data[i][j];
		            	if (val==0.0){
		            		continue;
		            	}
		            	if (usescale){
		            		val=Scaler.transform(val, j);
		            	}
		                QD[i] += val*val;
		                betas[j] += target[i] * alpha[i] * val;
		            }
		            index[i] = i;
		        }

		        while (iter < max_iter) {
		            PGmax_new = Double.NEGATIVE_INFINITY;
		            PGmin_new = Double.POSITIVE_INFINITY;

		            for (i = 0; i < active_size; i++) {
		                int j = i + random.nextInt(active_size - i);
		                swap(index, i, j);
		            }

		            for (s = 0; s < active_size; s++) {
		                i = index[s];
		                G = 0;
		                double yi = target[i];
		                
		                if (UseConstant){
		                	 G +=constant[0] ;
		                }
		                for (int j=0; j < w_size; j++) {
			            	val=data[i][j];
			            	if (val==0.0){
			            		continue;
			            	}
			            	if (usescale){
			            		val=Scaler.transform(val, j);
			            	}
		                    G += betas[j] *val;
		                }
		                G = G * yi - 1;

		                C = upper_bound[GETI(target, i)];
		                G += alpha[i] * diag[GETI(target, i)];

		                PG = 0;
		                if (alpha[i] == 0) {
		                    if (G > PGmax_old) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    } else if (G < 0) {
		                        PG = G;
		                    }
		                } else if (alpha[i] == C) {
		                    if (G < PGmin_old) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    } else if (G > 0) {
		                        PG = G;
		                    }
		                } else {
		                    PG = G;
		                }

		                PGmax_new = Math.max(PGmax_new, PG);
		                PGmin_new = Math.min(PGmin_new, PG);

		                if (Math.abs(PG) > 1.0e-12) {
		                    double alpha_old = alpha[i];
		                    alpha[i] = Math.min(Math.max(alpha[i] - G / QD[i], 0.0), C);
		                    d = (alpha[i] - alpha_old) * yi;
		                    if (UseConstant){
			                	 constant[0]+=d ;
			                }
		                    for (int j=0; j < w_size; j++) {
				            	val=data[i][j];
				            	if (val==0.0){
				            		continue;
				            	}
				            	if (usescale){
				            		val=Scaler.transform(val, j);
				            	}
		                    	betas[j] += d * val ;
		                    }
		                }
		            }
		            if ( verbose){
		            	System.out.println("Iteration: " + iter);
		            }
		            iter++;

		            if (PGmax_new - PGmin_new <= tolerance) {
		                if (active_size == l)
		                    break;
		                else {
		                    active_size = l;
		                    PGmax_old = Double.POSITIVE_INFINITY;
		                    PGmin_old = Double.NEGATIVE_INFINITY;
		                    continue;
		                }
		            }
		            PGmax_old = PGmax_new;
		            PGmin_old = PGmin_new;
		            if (PGmax_old <= 0) PGmax_old = Double.POSITIVE_INFINITY;
		            if (PGmin_old >= 0) PGmin_old = Double.NEGATIVE_INFINITY;
		        }

		        // calculate objective value


		        
				} else {
					
					    //betas = new double[data[0].length];
					    int l =data.length;
				        int j, s, iter = 0;
				        int max_iter = maxim_Iteration;
				        int active_size = betas.length;
				        int max_num_linesearch = 20;

				        double sigma = 0.01;
				        double d, G_loss, G, H;
				        double Gmax_old = Double.POSITIVE_INFINITY;
				        double Gmax_new, Gnorm1_new;
				        double Gnorm1_init = 0; // eclipse moans this variable might not be initialized
				        double d_old, d_diff;
				        double loss_old = 0; // eclipse moans this variable might not be initialized
				        double loss_new;
				        double appxcond, cond;
				        double val=0.0;
				        int[] index = new int[betas.length];
				        //int index_constant=0;
				        double[] b = new double[l]; // b = 1-ywTx
				        double[] xj_sq = new double[betas.length];
				        double xj_sq_contant=0.0;
				        double[] C = new double[] {this.C, 0, this.C};

				        // Initial w can be set here.

				        for (j = 0; j < l; j++) 
				            b[j] = 1;

				        if (UseConstant){
				        	//index_constant=0;
				        	xj_sq_contant = 0;
				            for (int i=0; i < l; i++) {
				                int ind =i;
				                 val = target[i];
				                b[i] -= constant[0] * val;
				                xj_sq_contant += C[GETI(target, ind)] * val * val;
				            }
				        }
				        for (j = 0; j < betas.length; j++) {
				            index[j] = j;
				            xj_sq[j] = 0;
				            for (int i=0; i < l; i++) {
				                int ind =i;
				            	val=data[i][j];
				            	if (val==0.0){
				            		continue;
				            	}
				            	if (usescale){
				            		val=Scaler.transform(val, j);
				            	}
				                 val = (val* target[i]);
				                b[i] -= betas[j] * val;

				                xj_sq[j] += C[GETI(target, ind)] * val * val;
				            }
				        }

				        while (iter < max_iter) {
				            Gmax_new = 0;
				            Gnorm1_new = 0;

				            for (j = 0; j < active_size; j++) {
				                int i = j + random.nextInt(active_size - j);
				                swap(index, i, j);
				            }
				            
				            if (UseConstant){
				                G_loss = 0;
				                H = 0;

				                for (int i=0; i < l; i++) {
				                    int ind = i;
				                    if (b[ind] > 0) {
				                         val =( target[i]);
				                        double tmp = C[GETI(target, ind)] * val;
				                        G_loss -= tmp * b[ind];
				                        H += tmp * val;
				                    }
				                }
				                G_loss *= 2;

				                G = G_loss;
				                H *= 2;
				                H = Math.max(H, 1e-12);

				                double Gp = G + 1;
				                double Gn = G - 1;
				                double violation = 0;
				                if (constant[0] == 0) {
				                    if (Gp < 0)
				                        violation = -Gp;
				                    else {
				                        violation = Gn;
				                    }

				                } else if (constant[0] > 0)
				                    violation = Math.abs(Gp);
				                else
				                    violation = Math.abs(Gn);

				                Gmax_new = Math.max(Gmax_new, violation);
				                Gnorm1_new += violation;

				                // obtain Newton direction d
				                if (Gp < H *constant[0])
				                    d = -Gp / H;
				                else if (Gn > H * constant[0])
				                    d = -Gn / H;
				                else
				                    d = -constant[0];

				                if (Math.abs(d) >= 1.0e-12) {

				                double delta = Math.abs(constant[0] + d) - Math.abs(constant[0]) + G * d;
				                d_old = 0;
				                int num_linesearch;
				                for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
				                    d_diff = d_old - d;
				                    cond = Math.abs(constant[0] + d) - Math.abs(constant[0]) - sigma * delta;

				                    appxcond = xj_sq_contant * d * d + G_loss * d + cond;
				                    if (appxcond <= 0) {
				                        for (int i=0; i < l; i++) {
				                            b[i] += d_diff *( target[i]);
				                        }
				                        break;
				                    }

				                    if (num_linesearch == 0) {
				                        loss_old = 0;
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                            int ind = i;
				                            if (b[ind] > 0) {
				                                loss_old += C[GETI(target, ind)] * b[ind] * b[ind];
				                            }
				                            double b_new = b[ind] + d_diff * ( target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    } else {
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                            int ind =i;
				                            double b_new = b[ind] + d_diff *(target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    }

				                    cond = cond + loss_new - loss_old;
				                    if (cond <= 0)
				                        break;
				                    else {
				                        d_old = d;
				                        d *= 0.5;
				                        delta *= 0.5;
				                    }
				                }

				                constant[0] += d;

				                // recompute b[] if line search takes too many steps
				                if (num_linesearch >= max_num_linesearch) {

				                    for (int i = 0; i < l; i++)
				                        b[i] = 1;
				                    
				                    if (constant[0]!=0.0){
				                    	for (int k= 0; k < l; k++) {
				                            b[k] -= constant[0] * (target[k]);
				                        }
				                    }
				                    for (int i = 0; i < betas.length; i++) {
				                        if (betas[i] == 0) continue;
				                        for (int k= 0; k < l; k++) {
				    		            	val=data[k][i];
				    		            	if (val==0.0){
				    		            		continue;
				    		            	}
				    		            	if (usescale){
				    		            		val=Scaler.transform(val, i);
				    		            	}
				                            b[k] -= betas[i] * (val* target[k]);
				                        }
				                    }
				                }	
				            }
				            }
				            for (s = 0; s < active_size; s++) {
				                j = index[s];
				                G_loss = 0;
				                H = 0;

				                for (int i=0; i < l; i++) {
				                    int ind = i;
				                    if (b[ind] > 0) {
			    		            	val=data[i][j];
			    		            	if (val==0.0){
			    		            		continue;
			    		            	}
			    		            	if (usescale){
			    		            		val=Scaler.transform(val, j);
			    		            	}
				                         val =(val* target[i]);
				                        double tmp = C[GETI(target, ind)] * val;
				                        G_loss -= tmp * b[ind];
				                        H += tmp * val;
				                    }
				                }
				                G_loss *= 2;

				                G = G_loss;
				                H *= 2;
				                H = Math.max(H, 1e-12);

				                double Gp = G + 1;
				                double Gn = G - 1;
				                double violation = 0;
				                if (betas[j] == 0) {
				                    if (Gp < 0)
				                        violation = -Gp;
				                    else if (Gn > 0)
				                        violation = Gn;
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

				                // obtain Newton direction d
				                if (Gp < H * betas[j])
				                    d = -Gp / H;
				                else if (Gn > H * betas[j])
				                    d = -Gn / H;
				                else
				                    d = -betas[j];

				                if (Math.abs(d) < 1.0e-12) continue;

				                double delta = Math.abs(betas[j] + d) - Math.abs(betas[j]) + G * d;
				                d_old = 0;
				                int num_linesearch;
				                for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
				                    d_diff = d_old - d;
				                    cond = Math.abs(betas[j] + d) - Math.abs(betas[j]) - sigma * delta;

				                    appxcond = xj_sq[j] * d * d + G_loss * d + cond;
				                    if (appxcond <= 0) {
				                        for (int i=0; i < l; i++) {
				    		            	val=data[i][j];
				    		            	if (val==0.0){
				    		            		continue;
				    		            	}
				    		            	if (usescale){
				    		            		val=Scaler.transform(val, j);
				    		            	}
				                            b[i] += d_diff *(val* target[i]);
				                        }
				                        break;
				                    }

				                    if (num_linesearch == 0) {
				                        loss_old = 0;
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				    		            	val=data[i][j];
				    		            	if (val==0.0){
				    		            		continue;
				    		            	}
				    		            	if (usescale){
				    		            		val=Scaler.transform(val, j);
				    		            	}
				                            int ind = i;
				                            if (b[ind] > 0) {
				                                loss_old += C[GETI(target, ind)] * b[ind] * b[ind];
				                            }
				                            double b_new = b[ind] + d_diff * (val* target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    } else {
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				    		            	val=data[i][j];
				    		            	if (val==0.0){
				    		            		continue;
				    		            	}
				    		            	if (usescale){
				    		            		val=Scaler.transform(val, j);
				    		            	}
				                            int ind =i;
				                            double b_new = b[ind] + d_diff *(val* target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    }

				                    cond = cond + loss_new - loss_old;
				                    if (cond <= 0)
				                        break;
				                    else {
				                        d_old = d;
				                        d *= 0.5;
				                        delta *= 0.5;
				                    }
				                }

				                betas[j] += d;

				                // recompute b[] if line search takes too many steps
				                if (num_linesearch >= max_num_linesearch) {

				                    for (int i = 0; i < l; i++)
				                        b[i] = 1;

				                    for (int i = 0; i < betas.length; i++) {
				                        if (betas[i] == 0) continue;
				                        for (int k= 0; k < l; k++) {
				    		            	val=data[k][i];
				    		            	if (val==0.0){
				    		            		continue;
				    		            	}
				    		            	if (usescale){
				    		            		val=Scaler.transform(val, i);
				    		            	}
				                            b[k] -= betas[i] * (val* target[k]);
				                        }
				                    }
				                }
				            }

				            if (iter == 0) {
				                Gnorm1_init = Gnorm1_new;
				            }
				            iter++;
				            if (iter % 10 == 0);

				            if (Gmax_new <= tolerance * Gnorm1_init) {
				                if (active_size == betas.length)
				                    break;
				                else {
				                    active_size = betas.length;
				                    
				                    Gmax_old = Double.POSITIVE_INFINITY;
				                    continue;
				                }
				            }

				            Gmax_old = Gmax_new;
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
			double n []= new double[data[0].length]; // sum of squared gradients
			
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
		    	double yi=-1.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	if (!this.RegularizationType.equals("L2") && Math.abs(constant[0])<=C){
		    		pred=0.0;
		    	}
		    	// compute score
		    	if (this.RegularizationType.equals("L2")){
		    	for (int j=0; j < data[i].length; j++){
	            	val=data[i][j];
	            	if (val==0.0){
	            		continue;
	            	}
	            	if (usescale){
	            		val=Scaler.transform(val, j);
	            	}
		    		pred+=val*betas[j];
		    	}
		    	} else {
		        	for (int j=0; j < data[i].length; j++){
		        		if (Math.abs(betas[j]) >C){
    		            	val=data[i][j];
    		            	if (val==0.0){
    		            		continue;
    		            	}
    		            	if (usescale){
    		            		val=Scaler.transform(val, j);
    		            	}
			    		pred+=val*betas[j];
		        		}
			    	}
		    	}
		    	
		    	//we need to check if pred*yi is > 1
		    	//if  pred*yi < 1 then the gradient is -yixi + C*beta
		    	// if pred*yi >=1 then the gradient is C*beta
		    	

		    	// we update constant gradient
		    	 if (UseConstant){
				    	boolean is_first_gradient=true;
				    	if (pred*yi>1){
				    		is_first_gradient=false;
				    		}
		    		 boolean superceeds_regularization=true;
		    		 double gradient=0 ;
		    		 if (is_first_gradient==true){
		    			 gradient=-yi;
		    		 }
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
		    			if (is_first_gradient==true){
		    				 nc=(-yi)*(-yi);
		    				 constant[0]=-yi;
			    		 } else {
		    			
		    			 //nc=0;
		    			 //constant[0]=0;
			    		 }
		    		}
		    		 
		    	 }
		    	 for (int j=0; j < data[i].length;j++){
		    		 boolean superceeds_regularization=true;
		    		 boolean is_first_gradient=true;
				    	if (pred*yi>=1){
				    		is_first_gradient=false;
				    		}
				    	 double gradient=0.0;
				    	if 	(is_first_gradient==true){
    		            	val=data[i][j];
    		            	if (val==0.0){
    		            		continue;
    		            	}
    		            	if (usescale){
    		            		val=Scaler.transform(val, j);
    		            	}
				    		gradient=-val*yi;
				    	}
		    		 
		    		 
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
				    			
				      			if (is_first_gradient==true){
		    		            	val=data[i][j];
		    		            	if (val==0.0){
		    		            		continue;
		    		            	}
		    		            	if (usescale){
		    		            		val=Scaler.transform(val, j);
		    		            	}
				    				 n[j]=(-val*yi)*(-val*yi);
				    				 betas[j]=-val*yi;
					    		 } else {
				    			
					    			 //n[j]=0;
					    			 //betas[j]=0;
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
	            	val=data[i][j];
	            	if (val==0.0){
	            		continue;
	            	}
	            	if (usescale){
	            		val=Scaler.transform(val, j);
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

			    pred+= BETA[j]*val;	
			    

			    	
			    }			     
			    
			    
		    	double yi=-1.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
			    
			    double gradient= 0;
			    
				 if (yi*pred<1){
					 gradient=  -yi;
					    //System.out.println("it is");
				 } 
				 
				if (UseConstant){		                    
					double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

					constant[0]+=gradient-move*Constant;
					nc+=gradient*gradient;
				 }
			    //updates
				//print.Print(betas, 5);
			    for (int j=0; j < columndimension; j++){
	            	val=data[i][j];
	            	if (val==0.0){
	            		continue;
	            	}
	            	if (usescale){
	            		val=Scaler.transform(val, j);
	            	}
			    		double gradientx=0.0;
			    		if (yi*pred<1){
			    			gradientx=  -yi*val;
							    //System.out.println("it is");
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
		if ( !Type.equals("Liblinear")  &&  !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Liblinear methods" );	
		}		
		if ( !Type.equals("Liblinear")  && quadratic==true ){
			throw new IllegalStateException(" quadratic loss is only available for liblinear" );	
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
				//Initialise beta
				//betas = new double[data[0].length];
				int l =data.GetRowDimension();
		        int w_size = data.GetColumnDimension();
		        int i, s, iter = 0;
		        double C, d, G;
		        double[] QD = new double[l];
		        int max_iter = maxim_Iteration;
		        int[] index = new int[l];
		        double[] alpha = new double[l];
		        int active_size = l;
		        // PG: projected gradient, for shrinking and stopping
		        double PG;
		        double PGmax_old = Double.POSITIVE_INFINITY;
		        double PGmin_old = Double.NEGATIVE_INFINITY;
		        double PGmax_new, PGmin_new;
		        double val=0.0;
		        // default solver_type: L2R_L2LOSS_SVC_DUAL
		        double diag[] = new double[] {0.5 / this.C, 0, 0.5 / this.C};
		        double upper_bound[] = new double[] {Double.POSITIVE_INFINITY, 0, Double.POSITIVE_INFINITY};
		        if (quadratic==true) {
		            diag[0] = 0;
		            diag[2] = 0;
		            upper_bound[0] = this.C;
		            upper_bound[2] = this.C;
		        }

		        // Initial alpha can be set here. Note that
		        // 0 <= alpha[i] <= upper_bound[GETI(i)]

		       

		        for (i = 0; i < l; i++) {
		            QD[i] = diag[GETI(target, i)];
		            //ystem.out.println(GETI(target, i));
		            if (UseConstant){
		            	QD[i] +=1;
		            	constant[0]+=target[i] * alpha[i];
			        }
		            for (int j=0; j < w_size; j++) {
		            	val=data.GetElement(i, j);
		            	if (val==0.0){
		            		continue;
		            	}
		            	if (usescale){
		            		val=Scaler.transform(val, j);
		            	}
		                QD[i] += val*val;
		                betas[j] += target[i] * alpha[i] * val;
		            }
		            index[i] = i;
		        }

		        while (iter < max_iter) {
		            PGmax_new = Double.NEGATIVE_INFINITY;
		            PGmin_new = Double.POSITIVE_INFINITY;

		            for (i = 0; i < active_size; i++) {
		                int j = i + random.nextInt(active_size - i);
		                swap(index, i, j);
		            }

		            for (s = 0; s < active_size; s++) {
		                i = index[s];
		                G = 0;
		                double yi = target[i];
		                
		                if (UseConstant){
		                	 G +=constant[0] ;
		                }
		                for (int j=0; j < w_size; j++) {
		                	val=data.GetElement(i, j);
		                	if (val==0.0){
		                		continue;
		                	}
		                	if (usescale){
		                		val=Scaler.transform(val, j);
		                	}
		                    G += betas[j] * val;
		                }
		                G = G * yi - 1;

		                C = upper_bound[GETI(target, i)];
		                G += alpha[i] * diag[GETI(target, i)];

		                PG = 0;
		                if (alpha[i] == 0) {
		                    if (G > PGmax_old) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    } else if (G < 0) {
		                        PG = G;
		                    }
		                } else if (alpha[i] == C) {
		                    if (G < PGmin_old) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    } else if (G > 0) {
		                        PG = G;
		                    }
		                } else {
		                    PG = G;
		                }

		                PGmax_new = Math.max(PGmax_new, PG);
		                PGmin_new = Math.min(PGmin_new, PG);

		                if (Math.abs(PG) > 1.0e-12) {
		                    double alpha_old = alpha[i];
		                    alpha[i] = Math.min(Math.max(alpha[i] - G / QD[i], 0.0), C);
		                    d = (alpha[i] - alpha_old) * yi;
		                    if (UseConstant){
			                	 constant[0]+=d ;
			                }
		                    for (int j=0; j < w_size; j++) {
		                    	val=data.GetElement(i, j);
		                    	if (val==0.0){
		                    		continue;
		                    	}
		                    	if (usescale){
		                    		val=Scaler.transform(val, j);
		                    	}
		                    	betas[j] += d * val ;
		                    }
		                }
		            }
		            if ( verbose){
		            	System.out.println("Iteration: " + iter);
		            }
		            iter++;

		            if (PGmax_new - PGmin_new <= tolerance) {
		                if (active_size == l)
		                    break;
		                else {
		                    active_size = l;
		                    PGmax_old = Double.POSITIVE_INFINITY;
		                    PGmin_old = Double.NEGATIVE_INFINITY;
		                    continue;
		                }
		            }
		            PGmax_old = PGmax_new;
		            PGmin_old = PGmin_new;
		            if (PGmax_old <= 0) PGmax_old = Double.POSITIVE_INFINITY;
		            if (PGmin_old >= 0) PGmin_old = Double.NEGATIVE_INFINITY;
		        }

		        // calculate objective value


		        
				} else {
					
					    //betas = new double[data[0].length];
					    int l =data.GetRowDimension();
				        int j, s, iter = 0;
				        int max_iter = maxim_Iteration;
				        int active_size = betas.length;
				        int max_num_linesearch = 20;

				        double sigma = 0.01;
				        double d, G_loss, G, H;
				        double Gmax_old = Double.POSITIVE_INFINITY;
				        double Gmax_new, Gnorm1_new;
				        double Gnorm1_init = 0; // eclipse moans this variable might not be initialized
				        double d_old, d_diff;
				        double loss_old = 0; // eclipse moans this variable might not be initialized
				        double loss_new;
				        double appxcond, cond;

				        int[] index = new int[betas.length];
				        //int index_constant=0;
				        double[] b = new double[l]; // b = 1-ywTx
				        double[] xj_sq = new double[betas.length];
				        double xj_sq_contant=0.0;
				        double[] C = new double[] {this.C, 0, this.C};
				        double val=0.0;
				        // Initial w can be set here.

				        for (j = 0; j < l; j++) 
				            b[j] = 1;

				        if (UseConstant){
				        	//index_constant=0;
				        	xj_sq_contant = 0;
				            for (int i=0; i < l; i++) {
				                int ind =i;
				                 val = target[i];
				                b[i] -= constant[0] * val;
				                xj_sq_contant += C[GETI(target, ind)] * val * val;
				            }
				        }
				        for (j = 0; j < betas.length; j++) {
				            index[j] = j;
				            xj_sq[j] = 0;
				            for (int i=0; i < l; i++) {
				                int ind =i;
				            	val=data.GetElement(i, j);
				            	if (val==0.0){
				            		continue;
				            	}
				            	if (usescale){
				            		val=Scaler.transform(val, j);
				            	}
				                 val = (val* target[i]);
				                if (val!=0.0){
				                b[i] -= betas[j] * val;

				                xj_sq[j] += C[GETI(target, ind)] * val * val;
				                }
				            }
				        }

				        while (iter < max_iter) {
				            Gmax_new = 0;
				            Gnorm1_new = 0;

				            for (j = 0; j < active_size; j++) {
				                int i = j + random.nextInt(active_size - j);
				                swap(index, i, j);
				            }
				            
				            if (UseConstant){
				                G_loss = 0;
				                H = 0;

				                for (int i=0; i < l; i++) {
				                    int ind = i;
				                    if (b[ind] > 0) {
				                         val =( target[i]);
				                        double tmp = C[GETI(target, ind)] * val;
				                        G_loss -= tmp * b[ind];
				                        H += tmp * val;
				                    }
				                }
				                G_loss *= 2;

				                G = G_loss;
				                H *= 2;
				                H = Math.max(H, 1e-12);

				                double Gp = G + 1;
				                double Gn = G - 1;
				                double violation = 0;
				                if (constant[0] == 0) {
				                    if (Gp < 0)
				                        violation = -Gp;
				                    else {
				                        violation = Gn;
				                    }

				                } else if (constant[0] > 0)
				                    violation = Math.abs(Gp);
				                else
				                    violation = Math.abs(Gn);

				                Gmax_new = Math.max(Gmax_new, violation);
				                Gnorm1_new += violation;

				                // obtain Newton direction d
				                if (Gp < H *constant[0])
				                    d = -Gp / H;
				                else if (Gn > H * constant[0])
				                    d = -Gn / H;
				                else
				                    d = -constant[0];

				                if (Math.abs(d) >= 1.0e-12) {

				                double delta = Math.abs(constant[0] + d) - Math.abs(constant[0]) + G * d;
				                d_old = 0;
				                int num_linesearch;
				                for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
				                    d_diff = d_old - d;
				                    cond = Math.abs(constant[0] + d) - Math.abs(constant[0]) - sigma * delta;

				                    appxcond = xj_sq_contant * d * d + G_loss * d + cond;
				                    if (appxcond <= 0) {
				                        for (int i=0; i < l; i++) {
				                            b[i] += d_diff *( target[i]);
				                        }
				                        break;
				                    }

				                    if (num_linesearch == 0) {
				                        loss_old = 0;
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                            int ind = i;
				                            if (b[ind] > 0) {
				                                loss_old += C[GETI(target, ind)] * b[ind] * b[ind];
				                            }
				                            double b_new = b[ind] + d_diff * (target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    } else {
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                            int ind =i;
				                            double b_new = b[ind] + d_diff *(target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    }

				                    cond = cond + loss_new - loss_old;
				                    if (cond <= 0)
				                        break;
				                    else {
				                        d_old = d;
				                        d *= 0.5;
				                        delta *= 0.5;
				                    }
				                }

				                constant[0] += d;

				                // recompute b[] if line search takes too many steps
				                if (num_linesearch >= max_num_linesearch) {

				                    for (int i = 0; i < l; i++)
				                        b[i] = 1;
				                    
				                    if (constant[0]!=0.0){
				                    	for (int k= 0; k < l; k++) {
				                            b[k] -= constant[0] * (target[k]);
				                        }
				                    }
				                    for (int i = 0; i < betas.length; i++) {
				                        if (betas[i] == 0) continue;
				                        for (int k= 0; k < l; k++) {
				                        	val=data.GetElement(k, i);
				                        	if (val==0.0){
				                        		continue;
				                        	}
				                        	if (usescale){
				                        		val=Scaler.transform(val, i);
				                        	}
				                            b[k] -= betas[i] * (val* target[k]);
				                        }
				                    }
				                }	
				            }
				            }
				            for (s = 0; s < active_size; s++) {
				                j = index[s];
				                G_loss = 0;
				                H = 0;

				                for (int i=0; i < l; i++) {
				                    int ind = i;
				                    if (b[ind] > 0) {
				                    	val=data.GetElement(i, j);
				                    	if (val==0.0){
				                    		continue;
				                    	}
				                    	if (usescale){
				                    		val=Scaler.transform(val, j);
				                    	}
				                         val =(val* target[i]);
				                        double tmp = C[GETI(target, ind)] * val;
				                        G_loss -= tmp * b[ind];
				                        H += tmp * val;
				                    }
				                }
				                G_loss *= 2;

				                G = G_loss;
				                H *= 2;
				                H = Math.max(H, 1e-12);

				                double Gp = G + 1;
				                double Gn = G - 1;
				                double violation = 0;
				                if (betas[j] == 0) {
				                    if (Gp < 0)
				                        violation = -Gp;
				                    else if (Gn > 0)
				                        violation = Gn;
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

				                // obtain Newton direction d
				                if (Gp < H * betas[j])
				                    d = -Gp / H;
				                else if (Gn > H * betas[j])
				                    d = -Gn / H;
				                else
				                    d = -betas[j];

				                if (Math.abs(d) < 1.0e-12) continue;

				                double delta = Math.abs(betas[j] + d) - Math.abs(betas[j]) + G * d;
				                d_old = 0;
				                int num_linesearch;
				                for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
				                    d_diff = d_old - d;
				                    cond = Math.abs(betas[j] + d) - Math.abs(betas[j]) - sigma * delta;

				                    appxcond = xj_sq[j] * d * d + G_loss * d + cond;
				                    if (appxcond <= 0) {
				                        for (int i=0; i < l; i++) {
				                        	val=data.GetElement(i, j);
				                        	if (val==0.0){
				                        		continue;
				                        	}
				                        	if (usescale){
				                        		val=Scaler.transform(val, j);
				                        	}
				                            b[i] += d_diff *(val* target[i]);
				                        }
				                        break;
				                    }

				                    if (num_linesearch == 0) {
				                        loss_old = 0;
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                        	val=data.GetElement(i, j);
				                        	if (val==0.0){
				                        		continue;
				                        	}
				                        	if (usescale){
				                        		val=Scaler.transform(val, j);
				                        	}
				                            int ind = i;
				                            if (b[ind] > 0) {
				                                loss_old += C[GETI(target, ind)] * b[ind] * b[ind];
				                            }
				                            double b_new = b[ind] + d_diff * (val* target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    } else {
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                        	val=data.GetElement(i, j);
				                        	if (val==0.0){
				                        		continue;
				                        	}
				                        	if (usescale){
				                        		val=Scaler.transform(val, j);
				                        	}
				                            int ind =i;
				                            double b_new = b[ind] + d_diff *(val* target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    }

				                    cond = cond + loss_new - loss_old;
				                    if (cond <= 0)
				                        break;
				                    else {
				                        d_old = d;
				                        d *= 0.5;
				                        delta *= 0.5;
				                    }
				                }

				                betas[j] += d;

				                // recompute b[] if line search takes too many steps
				                if (num_linesearch >= max_num_linesearch) {

				                    for (int i = 0; i < l; i++)
				                        b[i] = 1;

				                    for (int i = 0; i < betas.length; i++) {
				                        if (betas[i] == 0) continue;
				                        for (int k= 0; k < l; k++) {
				                        	val=data.GetElement(k, i);
				                        	if (val==0.0){
				                        		continue;
				                        	}
				                        	if (usescale){
				                        		val=Scaler.transform(val, i);
				                        	}
				                            b[k] -= betas[i] * (val* target[k]);
				                        }
				                    }
				                }
				            }

				            if (iter == 0) {
				                Gnorm1_init = Gnorm1_new;
				            }
				            iter++;
				            if (iter % 10 == 0);

				            if (Gmax_new <= tolerance * Gnorm1_init) {
				                if (active_size == betas.length)
				                    break;
				                else {
				                    active_size = betas.length;
				                    
				                    Gmax_old = Double.POSITIVE_INFINITY;
				                    continue;
				                }
				            }

				            Gmax_old = Gmax_new;
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
		    	double yi=-1.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	if (!this.RegularizationType.equals("L2") && Math.abs(constant[0])<=C){
		    		pred=0.0;
		    	}
		    	// compute score
		    	if (this.RegularizationType.equals("L2")){
		    	for (int j=0; j < data.GetColumnDimension(); j++){
		        	val=data.GetElement(i, j);
		        	if (val==0.0){
		        		continue;
		        	}
		        	if (usescale){
		        		val=Scaler.transform(val, j);
		        	}
		    		pred+=val*betas[j];
		    	}
		    	} else {
		        	for (int j=0; j < data.GetColumnDimension(); j++){
		        		if (Math.abs(betas[j]) >C){
		        	    	val=data.GetElement(i, j);
		        	    	if (val==0.0){
		        	    		continue;
		        	    	}
		        	    	if (usescale){
		        	    		val=Scaler.transform(val, j);
		        	    	}
			    		pred+=val*betas[j];
		        		}
			    	}
		    	}
		    	
		    	//we need to check if pred*yi is > 1
		    	//if  pred*yi < 1 then the gradient is -yixi + C*beta
		    	// if pred*yi >=1 then the gradient is C*beta
		    	

		    	// we update constant gradient
		    	 if (UseConstant){
				    	boolean is_first_gradient=true;
				    	if (pred*yi>1){
				    		is_first_gradient=false;
				    		}
		    		 boolean superceeds_regularization=true;
		    		 double gradient=0 ;
		    		 if (is_first_gradient==true){
		    			 gradient=-yi;
		    		 }
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
		    			if (is_first_gradient==true){
		    				 nc=(-yi)*(-yi);
		    				 constant[0]=-yi;
			    		 } else {
		    			
		    			 //nc=0;
		    			 //constant[0]=0;
			    		 }
		    		}
		    		 
		    	 }
		    	 for (int j=0; j < data.GetColumnDimension();j++){
		    		 boolean superceeds_regularization=true;
		    		 boolean is_first_gradient=true;
				    	if (pred*yi>=1){
				    		is_first_gradient=false;
				    		}
				    	 double gradient=0.0;
				    	if 	(is_first_gradient==true){
				        	val=data.GetElement(i, j);
				        	if (val==0.0){
				        		continue;
				        	}
				        	if (usescale){
				        		val=Scaler.transform(val, j);
				        	}
				    		gradient=-val*yi;
				    	}
		    		 
		    		 
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
				    			
				      			if (is_first_gradient==true){
				      		    	val=data.GetElement(i, j);
				      		    	if (val==0.0){
				      		    		continue;
				      		    	}
				      		    	if (usescale){
				      		    		val=Scaler.transform(val, j);
				      		    	}
				    				 n[j]=(-val*yi)*(-val*yi);
				    				 betas[j]=-val*yi;
					    		 } else {
				    			
					    			 //n[j]=0;
					    			 //betas[j]=0;
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
			    	val=data.GetElement(i, j);
			    	if (val==0.0){
			    		continue;
			    	}
			    	if (usescale){
			    		val=Scaler.transform(val, j);
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
			    
			    pred+= BETA[j]*val;	
			    

			    	
			    }			     
			    
			    
		    	double yi=-1.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
			    
			    double gradient= 0;
			    
				 if (yi*pred<1){
					 gradient=  -yi;
					    //System.out.println("it is");
				 } 
				 
				if (UseConstant){		                    
					double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

					constant[0]+=gradient-move*Constant;
					nc+=gradient*gradient;
				 }
			    //updates
				//print.Print(betas, 5);
			    for (int j=0; j < columndimension; j++){
			    	val=data.GetElement(i, j);
			    	if (val==0.0){
			    		continue;
			    	}
			    	if (usescale){
			    		val=Scaler.transform(val, j);
			    	}
			    		double gradientx=0.0;
			    		if (yi*pred<1){
			    			gradientx=  -yi*val;
							    //System.out.println("it is");
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
		if ( !Type.equals("Liblinear")  &&  !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Liblinear methods" );	
		}		
		if ( !Type.equals("Liblinear")  && quadratic==true ){
			throw new IllegalStateException(" quadratic loss is only available for liblinear" );	
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
				//Initialise beta
				//betas = new double[data[0].length];
				int l =data.GetRowDimension();
		        int i, s, iter = 0;
		        double C, d, G;
		        double[] QD = new double[l];
		        int max_iter = maxim_Iteration;
		        int[] index = new int[l];
		        double[] alpha = new double[l];
		        int active_size = l;
		        // PG: projected gradient, for shrinking and stopping
		        double PG;
		        double PGmax_old = Double.POSITIVE_INFINITY;
		        double PGmin_old = Double.NEGATIVE_INFINITY;
		        double PGmax_new, PGmin_new;
		        double val=0.0;
		        // default solver_type: L2R_L2LOSS_SVC_DUAL
		        double diag[] = new double[] {0.5 / this.C, 0, 0.5 / this.C};
		        double upper_bound[] = new double[] {Double.POSITIVE_INFINITY, 0, Double.POSITIVE_INFINITY};
		        if (quadratic==true) {
		            diag[0] = 0;
		            diag[2] = 0;
		            upper_bound[0] = this.C;
		            upper_bound[2] = this.C;
		        }

		        // Initial alpha can be set here. Note that
		        // 0 <= alpha[i] <= upper_bound[GETI(i)]
		        
		        //sort our data!
		        if (sparse_set==false){
			    if (!data.IsSortedByRow()){
			    	data.convert_type();
			    }

		        }

		        for (i = 0; i < l; i++) {
		            QD[i] = diag[GETI(target, i)];
		            //ystem.out.println(GETI(target, i));
		            if (UseConstant){
		            	QD[i] +=1;
		            	constant[0]+=target[i] * alpha[i];
			        }
		            for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		                val= data.valuespile[j];
		                QD[i] += val*val;
		                betas[data.mainelementpile[j]] += target[i] * alpha[i] * val;
		            }
		            index[i] = i;
		        }

		        while (iter < max_iter) {
		            PGmax_new = Double.NEGATIVE_INFINITY;
		            PGmin_new = Double.POSITIVE_INFINITY;

		            for (i = 0; i < active_size; i++) {
		                int j = i + random.nextInt(active_size - i);
		                swap(index, i, j);
		            }

		            for (s = 0; s < active_size; s++) {
		                i = index[s];
		                G = 0;
		                double yi = target[i];
		                
		                if (UseConstant){
		                	 G +=constant[0] ;
		                }
		                for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		                    G += betas[data.mainelementpile[j]] *data.valuespile[j];
		                }
		                G = G * yi - 1;

		                C = upper_bound[GETI(target, i)];
		                G += alpha[i] * diag[GETI(target, i)];

		                PG = 0;
		                if (alpha[i] == 0) {
		                    if (G > PGmax_old) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    } else if (G < 0) {
		                        PG = G;
		                    }
		                } else if (alpha[i] == C) {
		                    if (G < PGmin_old) {
		                        active_size--;
		                        swap(index, s, active_size);
		                        s--;
		                        continue;
		                    } else if (G > 0) {
		                        PG = G;
		                    }
		                } else {
		                    PG = G;
		                }

		                PGmax_new = Math.max(PGmax_new, PG);
		                PGmin_new = Math.min(PGmin_new, PG);

		                if (Math.abs(PG) > 1.0e-12) {
		                    double alpha_old = alpha[i];
		                    alpha[i] = Math.min(Math.max(alpha[i] - G / QD[i], 0.0), C);
		                    d = (alpha[i] - alpha_old) * yi;
		                    if (UseConstant){
			                	 constant[0]+=d ;
			                }
		                    for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		                    	betas[data.mainelementpile[j]] += d * data.valuespile[j] ;
		                    }
		                }
		            }
		            if ( verbose){
		            	System.out.println("Iteration: " + iter);
		            }
		            iter++;

		            if (PGmax_new - PGmin_new <= tolerance) {
		                if (active_size == l)
		                    break;
		                else {
		                    active_size = l;
		                    PGmax_old = Double.POSITIVE_INFINITY;
		                    PGmin_old = Double.NEGATIVE_INFINITY;
		                    continue;
		                }
		            }
		            PGmax_old = PGmax_new;
		            PGmin_old = PGmin_new;
		            if (PGmax_old <= 0) PGmax_old = Double.POSITIVE_INFINITY;
		            if (PGmin_old >= 0) PGmin_old = Double.NEGATIVE_INFINITY;
		        }

		        // calculate objective value


		        
				} else {
					
					    //betas = new double[data[0].length];
					    int l =data.GetRowDimension();
				        int j, s, iter = 0;
				        int max_iter = maxim_Iteration;
				        int active_size = betas.length;
				        int max_num_linesearch = 20;

				        double sigma = 0.01;
				        double d, G_loss, G, H;
				        double Gmax_old = Double.POSITIVE_INFINITY;
				        double Gmax_new, Gnorm1_new;
				        double Gnorm1_init = 0; // eclipse moans this variable might not be initialized
				        double d_old, d_diff;
				        double loss_old = 0; // eclipse moans this variable might not be initialized
				        double loss_new;
				        double appxcond, cond;
				        double val=0.0;
				        int[] index = new int[betas.length];
				        //int index_constant=0;
				        double[] b = new double[l]; // b = 1-ywTx
				        double[] xj_sq = new double[betas.length];
				        double xj_sq_contant=0.0;
				        double[] C = new double[] {this.C, 0, this.C};

				        // Initial w can be set here.

				        for (j = 0; j < l; j++) 
				            b[j] = 1;
				        
				        //sort by column
				        if (sparse_set==false){
						    if (!data.IsSortedByColumn()){
						    	data.convert_type();
						    }

					        }


					    
				        if (UseConstant){
				        	//index_constant=0;
				        	xj_sq_contant = 0;
				            for (int i=0; i < l; i++) {
				                int ind =i;
				                val = target[i];
				                b[i] -= constant[0] * val;
				                xj_sq_contant += C[GETI(target, ind)] * val * val;
				            }
				        }
				        for (j = 0; j < betas.length; j++) {
				            index[j] = j;
				            xj_sq[j] = 0;
				            for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
				                int ind =data.mainelementpile[i];
				                val = (data.valuespile[i]* target[ind]);
				                if (val!=0.0){
				                b[ind] -= betas[j] * val;

				                xj_sq[j] += C[GETI(target, ind)] * val * val;
				                }
				            }
				        }

				        while (iter < max_iter) {
				            Gmax_new = 0;
				            Gnorm1_new = 0;

				            for (j = 0; j < active_size; j++) {
				                int i = j + random.nextInt(active_size - j);
				                swap(index, i, j);
				            }
				            
				            if (UseConstant){
				                G_loss = 0;
				                H = 0;

				                for (int i=0; i < l; i++) {
				                    int ind = i;
				                    if (b[ind] > 0) {
				                         val =( target[i]);
				                        double tmp = C[GETI(target, ind)] * val;
				                        G_loss -= tmp * b[ind];
				                        H += tmp * val;
				                    }
				                }
				                G_loss *= 2;

				                G = G_loss;
				                H *= 2;
				                H = Math.max(H, 1e-12);

				                double Gp = G + 1;
				                double Gn = G - 1;
				                double violation = 0;
				                if (constant[0] == 0) {
				                    if (Gp < 0)
				                        violation = -Gp;
				                    else {
				                        violation = Gn;
				                    }

				                } else if (constant[0] > 0)
				                    violation = Math.abs(Gp);
				                else
				                    violation = Math.abs(Gn);

				                Gmax_new = Math.max(Gmax_new, violation);
				                Gnorm1_new += violation;

				                // obtain Newton direction d
				                if (Gp < H *constant[0])
				                    d = -Gp / H;
				                else if (Gn > H * constant[0])
				                    d = -Gn / H;
				                else
				                    d = -constant[0];

				                if (Math.abs(d) >= 1.0e-12) {

				                double delta = Math.abs(constant[0] + d) - Math.abs(constant[0]) + G * d;
				                d_old = 0;
				                int num_linesearch;
				                for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
				                    d_diff = d_old - d;
				                    cond = Math.abs(constant[0] + d) - Math.abs(constant[0]) - sigma * delta;

				                    appxcond = xj_sq_contant * d * d + G_loss * d + cond;
				                    if (appxcond <= 0) {
				                        for (int i=0; i < l; i++) {
				                            b[i] += d_diff *( target[i]);
				                        }
				                        break;
				                    }

				                    if (num_linesearch == 0) {
				                        loss_old = 0;
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                            int ind = i;
				                            if (b[ind] > 0) {
				                                loss_old += C[GETI(target, ind)] * b[ind] * b[ind];
				                            }
				                            double b_new = b[ind] + d_diff * ( target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    } else {
				                        loss_new = 0;
				                        for (int i=0; i < l; i++) {
				                            int ind =i;
				                            double b_new = b[ind] + d_diff *(target[i]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    }

				                    cond = cond + loss_new - loss_old;
				                    if (cond <= 0)
				                        break;
				                    else {
				                        d_old = d;
				                        d *= 0.5;
				                        delta *= 0.5;
				                    }
				                }

				                constant[0] += d;

				                // recompute b[] if line search takes too many steps
				                if (num_linesearch >= max_num_linesearch) {

				                    for (int i = 0; i < l; i++)
				                        b[i] = 1;
				                    
				                    if (constant[0]!=0.0){
				                    	for (int k= 0; k < l; k++) {
				                            b[k] -= constant[0] * (target[k]);
				                        }
				                    }
				                    for (int i = 0; i < betas.length; i++) {
				                        if (betas[i] == 0) continue;
				                        for (int k=data.indexpile[i]; k <data.indexpile[i+1] ; k++) {
				                        	int index_row=data.mainelementpile[k];
				                            b[index_row] -= betas[i] * (data.valuespile[k]* target[index_row]);
				                        }
				                    }
				                }	
				            }
				            }
				            for (s = 0; s < active_size; s++) {
				                j = index[s];
				                
				                G_loss = 0;
				                H = 0;

				                for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
				                    int ind = data.mainelementpile[i];
				                    if (b[ind] > 0) {
				                         val =(data.valuespile[i]* target[ind]);
				                        double tmp = C[GETI(target, ind)] * val;
				                        G_loss -= tmp * b[ind];
				                        H += tmp * val;
				                    }
				                }
				                G_loss *= 2;

				                G = G_loss;
				                H *= 2;
				                H = Math.max(H, 1e-12);

				                double Gp = G + 1;
				                double Gn = G - 1;
				                double violation = 0;
				                if (betas[j] == 0) {
				                    if (Gp < 0)
				                        violation = -Gp;
				                    else if (Gn > 0)
				                        violation = Gn;
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

				                // obtain Newton direction d
				                if (Gp < H * betas[j])
				                    d = -Gp / H;
				                else if (Gn > H * betas[j])
				                    d = -Gn / H;
				                else
				                    d = -betas[j];

				                if (Math.abs(d) < 1.0e-12) continue;

				                double delta = Math.abs(betas[j] + d) - Math.abs(betas[j]) + G * d;
				                d_old = 0;
				                int num_linesearch;
				                for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
				                    d_diff = d_old - d;
				                    cond = Math.abs(betas[j] + d) - Math.abs(betas[j]) - sigma * delta;

				                    appxcond = xj_sq[j] * d * d + G_loss * d + cond;
				                    if (appxcond <= 0) {
				                    	for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
				                    		int in=data.mainelementpile[i];
				                            b[in] += d_diff *(data.valuespile[i]* target[in]);
				                        }
				                        break;
				                    }

				                    if (num_linesearch == 0) {
				                        loss_old = 0;
				                        loss_new = 0;
				                        for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
				                            int ind = data.mainelementpile[i];
				                            if (b[ind] > 0) {
				                                loss_old += C[GETI(target, ind)] * b[ind] * b[ind];
				                            }
				                            double b_new = b[ind] + d_diff * (data.valuespile[i]* target[ind]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    } else {
				                        loss_new = 0;
				                         for (int i=data.indexpile[j]; i <data.indexpile[j+1]; i++) {
				                            int ind =data.mainelementpile[i];
				                            double b_new = b[ind] + d_diff *(data.valuespile[i]* target[ind]);
				                            b[ind] = b_new;
				                            if (b_new > 0) {
				                                loss_new += C[GETI(target, ind)] * b_new * b_new;
				                            }
				                        }
				                    }

				                    cond = cond + loss_new - loss_old;
				                    if (cond <= 0)
				                        break;
				                    else {
				                        d_old = d;
				                        d *= 0.5;
				                        delta *= 0.5;
				                    }
				                }

				                betas[j] += d;

				                // recompute b[] if line search takes too many steps
				                if (num_linesearch >= max_num_linesearch) {

				                    for (int i = 0; i < l; i++)
				                        b[i] = 1;

				                    for (int i = 0; i < betas.length; i++) {
				                        if (betas[i] == 0) continue;
				                        for (int k=data.indexpile[i]; k <data.indexpile[i+1]; k++) {
				                        	int index_row=data.mainelementpile[k];
				                            b[index_row] -= betas[i] * (data.valuespile[k]* target[index_row]);
				                        }
				                    }
				                }
				            }

				            if (iter == 0) {
				                Gnorm1_init = Gnorm1_new;
				            }
				            iter++;
				            if (iter % 10 == 0);

				            if (Gmax_new <= tolerance * Gnorm1_init) {
				                if (active_size == betas.length)
				                    break;
				                else {
				                    active_size = betas.length;
				                    
				                    Gmax_old = Double.POSITIVE_INFINITY;
				                    continue;
				                }
				            }

				            Gmax_old = Gmax_new;
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
			
			// sort by row
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
		    	double yi=-1.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	if (!this.RegularizationType.equals("L2") && Math.abs(constant[0])<=C){
		    		pred=0.0;
		    	}
		    	// compute score
		    	if (this.RegularizationType.equals("L2")){
		    		for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		    		pred+=data.valuespile[j]*betas[data.mainelementpile[j]];
		    	}
		    	} else {
		    		for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		        		if (Math.abs(betas[data.mainelementpile[j]]) >C)
			    		pred+=data.valuespile[j]*betas[data.mainelementpile[j]];
			    	}
		    	}
		    	
		    	//we need to check if pred*yi is > 1
		    	//if  pred*yi < 1 then the gradient is -yixi + C*beta
		    	// if pred*yi >=1 then the gradient is C*beta
		    	

		    	// we update constant gradient
		    	 if (UseConstant){
				    	boolean is_first_gradient=true;
				    	if (pred*yi>1){
				    		is_first_gradient=false;
				    		}
		    		 boolean superceeds_regularization=true;
		    		 double gradient=0 ;
		    		 if (is_first_gradient==true){
		    			 gradient=-yi;
		    		 }
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
		    			if (is_first_gradient==true){
		    				 nc=(-yi)*(-yi);
		    				 constant[0]=-yi;
			    		 } else {
		    			
		    			 //nc=0;
		    			 //constant[0]=0;
			    		 }
		    		}
		    		 
		    	 }
		    	 for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		    		 int col_index=data.mainelementpile[j];
		    		 double col_val=data.valuespile[j];
		    		 boolean superceeds_regularization=true;
		    		 boolean is_first_gradient=true;
				    	if (pred*yi>=1){
				    		is_first_gradient=false;
				    		}
				    	 double gradient=0.0;
				    	if 	(is_first_gradient==true){
				    		gradient=-col_val*yi;
				    	}
		    		 
		    		 
		    		 if (this.RegularizationType.equals("L2")){
		    			 gradient+=C*betas[col_index];
		    		 } else{
		    			 //get sign
		    			 double sign=-1;
		    			 if (betas[col_index]>0){
		    				 sign=1.0;  
		 		    	}
		    			 if (sign * betas[col_index]  <= C && n[col_index]!=0.0){
		    				// gradient=0.0;
		    				 superceeds_regularization=false;
		    			 } else{
		    				 gradient+=C*sign*betas[col_index];
		    			 }
		    		 }
		    		 
			    		if (superceeds_regularization){
				    		 n[col_index]+=gradient*gradient;
				    		 double move=(this.learn_rate*gradient)/Math.sqrt(n[col_index]+0.00000000000001);
				    		 betas[col_index]=betas[col_index]-move;
				    		 if (Math.abs(move)>iteration_tol){
								 iteration_tol= Math.abs(move);
							 }
				    		} else {
				    			
				      			if (is_first_gradient==true){
				    				 n[col_index]=(-col_val*yi)*(-col_val*yi);
				    				 betas[col_index]=-col_val*yi;
					    		 } else {
				    			
					    			 //n[j]=0;
					    			 //betas[j]=0;
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

			// sort by row
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
		    	for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		    		double col_val=data.valuespile[j];
		    		int in=data.mainelementpile[j];
			    	if (col_val!=0.0){
			    	double sign=1.0;			    	
			    	if (betas[in]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[in]  <= l1C){
			    		 BETA[in]=0 ;
			    	 } else {
			    		 BETA[in]=  (sign * l1C - betas[in]) / (( 0.1+Math.sqrt(n[in])) / this.learn_rate + C);
			    	
			    	 }
			    
			    pred+= BETA[in]*col_val;	
			    
			    	}
			    	
			    }			     
			    
			    
		    	double yi=-1.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
			    
			    double gradient= 0;
			    
				 if (yi*pred<1){
					 gradient=  -yi;
					    //System.out.println("it is");
				 } 
				 
				if (UseConstant){		                    
					double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

					constant[0]+=gradient-move*Constant;
					nc+=gradient*gradient;
				 }
			    //updates
				//print.Print(betas, 5);
		    	for (int j=data.indexpile[i]; j<data.indexpile[i+1]; j++ ) {
		    		double col_val=data.valuespile[j];
		    		int in=data.mainelementpile[j];
			    	if (col_val!=0.0){
			    		double gradientx=0.0;
			    		if (yi*pred<1){
			    			gradientx=  -yi*col_val;
							    //System.out.println("it is");
						 } 
			    	//System.out.println(" gradient: " + gradientx);
			    	double move=(Math.sqrt(n[in] + gradientx * gradientx) - Math.sqrt(n[in])) / this.learn_rate;
			    	betas[in] += gradientx - move * BETA[in];
                    n[in] += gradientx * gradientx;	

                    // check for highest tolerance
      				 if (Math.abs(BETA[in])>iteration_tol){
							 iteration_tol= Math.abs(BETA[in]);
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
		return "binaryLSVC";
	}
	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: Binary Regularized Linear Support Vector classifier");
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
		binaryLSVC br = new binaryLSVC();
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
	        return  (int) (y[i] + 1);
	    }
	 
	private static void swap(int[] array, int idxA, int idxB) {
	    int temp = array[idxA];
	    array[idxA] = array[idxB];
	    array[idxB] = temp;
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
