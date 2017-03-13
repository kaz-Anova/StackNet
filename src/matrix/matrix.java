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

package matrix;

/**
 * 
 * @author marios
 * <p> this is the main data object interface (namely matrix ) that should have some basic functions in common </p>
 */
public interface matrix {
     /**
      * 
      * @return rows of the current matrix 
      */
	public int GetRowDimension();
    /**
     * 
     * @return rows of the current matrix 
     */
	public int GetColumnDimension();		
	/**
	 * 
	 * @param coulmn the array to "append" 
	 */
	public void AddColumn(double coulmn[]);
	/**
	 * 
	 * @param row the row-array to "append" 
	 */
	public void AddRow(double row[]);
	/**
	 * 
	 * <p>Removes the last column</p> 
	 */
	public void RemoveColumn();
	/**
	 * 
	 * <p>Removes the last row</p> 
	 */
	public void RemoveRow();
	/**
	 * @param coulmn the array-column to retrieve 
	 */
	public double [] GetColumn( int column);	
	/**
	 * @param row the array-row to retrieve 
	 */
	public double[] GetRow( int row);	
	/**
	 * <p> Makes a hard copy of all the  elements of the matrix </p>
	 * @return the copied matrix
	 */
	public matrix Copy(); 
	/**
	 * Prints information for the current object
	 */
	public void PrintInfo();
	/**
	 * 
	 * @param file : file to print the matrix
	 */
	public void ToFile(String file);
	
}
