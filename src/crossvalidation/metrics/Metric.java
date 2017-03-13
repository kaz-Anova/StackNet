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


package crossvalidation.metrics;
/**
 * 
 * @author marios
 *<p> main interface for metrics </p>
 */
public interface Metric {

	/**
	 * 
	 * @param predicted : array with predicted values
	 * @param actual : array with actual values
	 * @return the value of the metric
	 */
	public double GetValue(double predicted [], double actual []);
	/**
	 * 
	 * @param predicted : array with predicted values
	 * @param actual : array with actual values
	 * @param threads : Number of threads
	 * @return the values of the metric []
	 */
	public double [] GetValue(double predicted [][], double actual [][], int threads);	
	/**
	 * @return the value of the computed metric
	 */
	public double GetValue();
	/**
	 * 
	 * @param value : the value to compare with
	 * @return : True if this value is better than the previous one
	 */
	public boolean IsBetter(double value);
	/**
	 * @param value : the metric to compare with
	 * @return : True if this value is better than the previous one
	 */
	public boolean IsBetter(Metric m);
	/**
	 * @param value : to set in the metric
	 */
	public void UpdateValue(double value);
	/**
	 * @return : the type of the metric
	 */
	public String Gettype();
	
	
	
}
