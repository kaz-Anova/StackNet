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

package ml;
import java.io.Serializable;

import matrix.fsmatrix;
import matrix.smatrix;

/**
 * 
 * @author marios
 * Main interface for Regressors, Classifier and other ml unsupervised objects
 */
public interface estimator extends Serializable,Runnable{
  /**
   * @param f : Fized Size Matrix to be scored
   * @return : predictions
   */
	public double [] predict(fsmatrix f);
	/**
	 * @param f : Sparse Matrix to be scored
	 * @return : predictions
	 */
	public double [] predict(smatrix f);
	 /**
	  * @param data : data to be scored
	  * @return : predictions
	  */
	public double [] predict(double data [][]);	
	/** 
	 * @param row double array to predict probabilities as 1 sample
	 * @return predictions
	 */
	double predict_Row(double row []);
	/**
	 * @param f Fized size matrix to use
	 * @param row to process
	 * @return predictions 
	 */	
	double predict_Row(fsmatrix f, int row);
	/**
	 * @param f Sparse  matrix to use
	 * @param start of the loop the row appears
	 * @param end of the loop the row appears
	 * @return predictions 
	 */	
	double predict_Row(smatrix f, int start, int end);
	/**
	 * @param data : data to fit the estimator with
	 */
	public void fit(double data [][]);
	/**
	 * @param f : Fized Size Matrix to fit the estimator with
	 */
	 public void fit(fsmatrix f);
	 /**
	  * @param f : Sparse Matrix to fit the estimator with
	  */
	 public void fit(smatrix f);
	/**
	 * @return the type of the object
	 */
	public String GetType();
	/**
	 * @return True if the estimator supports weights
	 */
	public boolean SupportsWeights();
	/**
	 * @return name of the Estimator
	 */
	public String GetName();
	/**
	 * <p> Prints information for the specific algorithms </p>
	 */
	public void PrintInformation();	
	/**
	 * @param a : The estimator to compare with
	 * @return True if the estimator has the same type with the current one
	 */
	public boolean HasTheSametype(estimator a);
	/**
	 * @return True if the estimator Has been fitted
	 */
	public boolean isfitted();
	/**
	 * @return True if the estimator is a regrerssor
	 */
	public boolean IsRegressor();
	/**
	 * @return True if the estimator is a classifier
	 */
	public boolean IsClassifier();
	/**
	 * Resets current Algorithm
	 */
	public void reset();
	/**
	 * Copies current estimator
	 * @return the copied estimator
	 */
	public estimator copy();
	/**
	 * @return the scaler used for the modelling purposes
	 */
	public preprocess.scaling.scaler ReturnScaler();
	/**
	 * 
	 * @param sc : scaler to be set
	 */
	public void setScaler(preprocess.scaling.scaler sc);	
	/**
	 * 
	 * @param seed : int value for seed
	 */
	public void setSeed(int seed);
	/**
	 * 
	 *@return seed 
	 */
	public int getSeed();
	/**
	 * 
	 * @param data : data to create the constructor
	 */
	public void setdata(double data [][]);
	/**
	 * 
	 * @param data : data to create the constructor
	 */
	public void setdata(fsmatrix data);
	/**
	 * 
	 * @param data : data to create the constructor
	 */
	public void setdata(smatrix data);	
	/**
	 * Method to run in parallel (threads)
	 */
	public void run() ;
	/**
	 * 
	 * @param params : parameters as a string
	 */
	public void set_params(String params);
	/**
	 * 
	 * @param data : dataset as double array
	 * @return predicted probabilities for classifiers, standard prediction for regressors
	 */
	public double [][] predict_proba(double data [][]);
	/**
	 * 
	 * @param f : dataset as Fixed Size matrix
	 * @return predicted probabilities for classifiers, standard prediction for regressors
	 */
	public double [][] predict_proba(fsmatrix f);
	/**
	 * 
	 * @param f :  dataset as Sparse matrix
	 * @return predicted probabilities for classifiers, standard prediction for regressors
	 */	
	public double [][] predict_proba(smatrix f);
	/**
	 * 
	 * @param data : target varible as 1d
	 */
	public void set_target(double data []);
	/**
	 * @param names : Classes' names
	 * <p> method to add order and names to the classes to be classified
	 */
	public void AddClassnames(String names[]);
	
}
