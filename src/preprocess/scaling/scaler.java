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

/**
 * 
 */
package preprocess.scaling;

import java.io.Serializable;

import matrix.fsmatrix;
import matrix.smatrix;

/**
 * @author marios
 *<p> main interface for a 'scaler', an object used to transform data via scaling</p>
 */
public interface scaler extends Serializable {

	/**
	 * 
	 * @param data to fit the scaler
	 */
	public void fit (double data[][]);
	/**
	 * 
	 * @param data to fit the scaler
	 */
	
	public void fit (fsmatrix data);
	/**
	 * 
	 * @param data to fit the scaler
	 */
	
	public void fit (smatrix data);	
	/**
	 * 
	 * @param data to transform
	 */
	
	public void transformthis (double data[][]);
	/**
	 * 
	 * @param data to transform
	 */
	
	public void transformthis (fsmatrix data);
	/**
	 * 
	 * @param data to transform
	 */
	
	public void transformthis (smatrix data);	
	/**
	 * 
	 * @param data to transform
	 * @return double data
	 */
	public double [][] transform (double data[][]);
	/**
	 * @param data to transform
	 * @return fsmatrix data
	 */
	public fsmatrix  transform (fsmatrix data);
	/** 
	 * @param data to transform
	 * @return Sparse matrix data
	 */
	public smatrix  transform (smatrix data);
	/**
	 * 
	 * @return the maximum values as computed from the fit methods
	 */
	public double transform(double value, int column);
	
	/**
	 * 
	 * @return true if the scaler is fitted 
	 */
	public boolean IsFitted();
	
}
