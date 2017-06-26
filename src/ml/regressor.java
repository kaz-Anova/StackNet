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

import matrix.fsmatrix;
import matrix.smatrix;

/**
 * 
 * @author marios
 * <p> Main interface for Regressors
 */
public interface regressor extends estimator {
	
	  /**
	   * @param f : Fized Size Matrix to be scored
	   * @return : predictions
	   */
		public double [][] predict2d(fsmatrix f);
		/**
		 * @param f : Sparse Matrix to be scored
		 * @return : predictions
		 */
		public double [][] predict2d(smatrix f);
		 /**
		  * @param data : data to be scored
		  * @return : predictions
		  */
		public double [][] predict2d(double data [][]);	
		/** 
		 * @param row double array to predict probabilities as 1 sample
		 * @return predictions
		 */
		double []predict_Row2d(double row []);
		/**
		 * @param f Fixed size matrix to use
		 * @param row to process
		 * @return predictions 
		 */	
		double []predict_Row2d(fsmatrix f, int row);
		/**
		 * @param f Sparse  matrix to use
		 * @param start of the loop the row appears
		 * @param end of the loop the row appears
		 * @return predictions 
		 */	
		double[] predict_Row2d(smatrix f, int start, int end);
		/**
		 * 
		 * @param fstarget : target variable in fixed-size matrix
		 */
		public void set_target(fsmatrix fstarget);
}
