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
 * @author marios
 * <p> Main interface for classifiers
 */
public interface classifier extends estimator {
/**
 * 
 * @param data dataset as double array
 * @return predicted probablities
 */
	public double [][] predict_proba(double data [][]);
	/**
	 * 
	 * @param f dataset as Fixed Size matrix
	 * @return predicted probablities
	 */
	public double [][] predict_proba(fsmatrix f);
	/**
	 * 
	 * @param f dataset as Sparse matrix
	 * @return predicted probablities
	 */	
	public double [][] predict_proba(smatrix f);
	
	/** 
	 * @param row double array to predict probabilities as 1 sample
	 * @return predictions in probabilities with length equal to the number of distinct classes when fitted the model
	 */
	double [] predict_probaRow(double row []);
	/**
	 * @param f Fized size matrix to use
	 * @param row to process
	 * @return predictions in probabilities with length equal to the number of distinct classes when fitted the model
	 */	
	double [] predict_probaRow(fsmatrix f, int row);
	/**
	 * @param f Sparse  matrix to use
	 * @param start of the loop the row appears
	 * @param end of the loop the row appears
	 * @return predictions in probabilities with length equal to the number of distinct classes when fitted the model
	 */	
	double [] predict_probaRow(smatrix f, int start, int end);
	
	/**
	 * @return the name of the classes as the will appear in prediction order
	 */
	public String[] getclasses();
	
	/**
	 * @return the number of distinct classes
	 */
	public int getnumber_of_classes();	


	
	//public double [][] predict_probafromfile(); 
}
