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

import java.io.Serializable;

import io.output;
import exceptions.DimensionMismatchException;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 * <p> Fixed size matrix refers to a non-sparse double 2 dimensional matrix that "sits" on a single 1 dimension double array
 * Memory-wise and speed-wise should be very efficient or as efficient as tit gets for the cost of having fixed sizes </p>
 */
public class fsmatrix implements matrix, Serializable  {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6799103896162858749L;
	/**
	 * Number of rows in the current matrix
	 */
	private int rows;
	/**
	 * Number of columns in the current matrix
	 */
	private int columns;
	/**
	 * The one dimensional data stored in this matrix
	 */
	public double data [];
	/**
	 * 
	 * @param data : the data to feed into the matrix
	 * @param rows : how many rows (samples) are represented in the current matrix
	 * @param columns : columns in the current matrix
	 * <p> basic constructor for a fixed size matrix. The idea is if you have 1 10*10 matrix this can be represented by a 100 cases 1-dimension
	 * double array.
	 */
	public fsmatrix(double data [], int rows, int columns) {
		// sensible checking
		if (data==null){
			throw new NullObjectException(" data to feed in the fsmatric cannot be null ");
		}
		int expected_columns=data.length/rows;
		
		if (expected_columns!=columns){
			throw new DimensionMismatchException(expected_columns,columns);
		}
		this.data=data;
		this.rows=rows;
		this.columns=columns;
	}
	
	/**
	 * 
	 * @param rows : how many rows (samples) are represented in the current matrix
	 * @param columns : columns in the current matrix
	 * <p> basic constructor for a fixed size matrix. Creates an empty matrix
	 */
	public fsmatrix( int rows, int columns) {
		
		// sensible checking
		this.data=new double [rows*columns];
		this.rows=rows;
		this.columns=columns;
		
	}
		
	
	/**
	 * 
	 * @param rows : Subset of rows's indices to use to create the submatrix
	 * @return fsmatrix
	 * <p> returns a subset of rows as fsmatrix
	 */
	public fsmatrix makerowsubset( int [] rows ) {
		
		if (this.data==null){
			throw new NullObjectException(" Current matrix is empty, nothing to make a subset for");
		}
		double newdata[] = new double [rows.length * this.columns];
		int k=0;
		for (int i=0; i<rows.length;i++){
			for (int j=0; j < this.columns; j++){
				newdata[k]=this.data[ rows[i]*this.columns + j];
				k++;
			}
		}
		
		return new  fsmatrix(newdata, rows.length, this.columns) ;
	}
	/**
	 * 
	 * @param columns : Subset of columns' indices to use to create the submatrix
	 * @return fsmatrix
	 * <p> returns a subset of columns as fsmatrix
	 */
	public fsmatrix makecolumnubset( int [] columns ) {
		
		if (this.data==null){
			throw new NullObjectException(" Current matrix is empty, nothing to make a subset for");
		}
		double newdata[] = new double [this.rows * columns.length];
		int k=0;
		for (int i=0; i<this.rows;i++){
			for (int j=0; j < columns.length; j++){
				newdata[k]=this.data[ i*columns[j] + columns[j]];
				k++;
			}
		}
		
		return new  fsmatrix(newdata,  this.rows, columns.length) ;
	}	
	
	/**
	 * 
	 * @param rows : Subset of rows' indices to use to create the submatrix
	 * @param columns : Subset of columns' indices to use to create the submatrix
	 * @return fsmatrix 
	 * <p> returns a subset of rows and columns as fsmatrix
	 */
	public fsmatrix makerowsandcolumnubset(int rows [],  int [] columns ) {
		
		if (this.data==null){
			throw new NullObjectException(" Current matrix is empty, nothing to make a subset for");
		}
		double newdata[] = new double [rows.length* columns.length];
		int k=0;
		for (int i=0; i<rows.length;i++){
			for (int j=0; j < columns.length; j++){
				newdata[k]=this.data[ rows[i]*columns[j] + columns[j]];
				k++;
			}
		}
		
		return new  fsmatrix(newdata, rows.length, columns.length) ;
	}	
	
	/**
	 * 
	 * @param data : the data to feed into the matrix
	 * <p> basic constructor for a fixed size matrix. The idea is if you have 1 10*10 matrix this can be represented by a 100 cases 1-dimension
	 * double array.
	 */
	public fsmatrix(double data [][]) {
		// sensible checking
		if (data==null){
			throw new NullObjectException(" data to feed in the fsmatric cannot be null ");
		}
		double single_layer_data[]= new double [data.length* data[0].length];
		
		int n=0;
		
		for (int i=0; i <data.length; i++ ){
			for (int j=0; j<data[i].length; j++ ){
				single_layer_data[n]=data[i][j];
				n++;
			}
		}
		this.data=single_layer_data;
		this.rows=data.length;
		this.columns=data[0].length;
	}		
	
	/**
	 * 
	 * @param rowtoget : row to access
	 * @param columntoget : column to access
	 * @return the elements that sits (virtually) in [rowtoget,columntoget]
	 */
	
	public double GetElement(int rowtoget, int columntoget){
		return data[rowtoget *columns + columntoget ];
	}
	/**
	 * @param rowtoget : row to access
	 * @param columntoget : column to access
	 * @param element : element to set in [rowtoget,columntoget]
	 */
	public void SetElement(int rowtoget, int columntoget, double element){
		 data[rowtoget *columns + columntoget]=element;
	}
	
	/**
	 * 
	 * @return This data as 1-dimension double array
	 */
	
	public double [] GetDatathis(){
		return this.data;	
	}
	/**
	 * 
	 * @return A copy of the data as 1-dimension double array
	 */
	public double [] GetData(){
		
		return manipulate.copies.copies.Copy(this.data);
		
	}	
	/**
	 * 
	 * @return A copy of the data as 2-dimension double array (rows,columns)
	 */
	public double [][] GetData2d(){
		
		double array [][]= new double [rows][columns];
		int n=0;
		for (int i=0; i <array.length; i++ ){
			for (int j=0; j<array[0].length; j++ ){
				array[i][j]=data[n];
				n++;
			}
		}
		
		return array;
		
	}		
	@Override
	public int GetRowDimension() {
		return rows;
	}
	@Override
	public int GetColumnDimension() {
		return columns;
	}

	@Override
	public void AddColumn(double[] coulmn) {
		
		if (coulmn.length!=rows){
			throw new DimensionMismatchException(coulmn.length,rows);
			
		}
		double [] newdata= new double [rows*columns + rows];
		int n=0;
		int column_n=0;
		for (int i=0; i <rows; i++ ){
			for (int j=0; j<columns; j++ ){
				newdata[n]=this.data[columns*i + j];
				n++;
			}
			newdata[n]=coulmn[column_n];
			n++;
			column_n++;
			}
		
		this.data=newdata;
		this.columns++;

	}
	/**
	 * adds constant in the current fs matrix
	 */
	public void AddConstant() {
		if (data==null){
			throw new NullObjectException(" data to feed in the fsmatric cannot be null ");
		}
		double new_data []= new double [data.length + rows];
		int n=0;
		for (int i=0; i< rows; i++){
			new_data[n]=1.0;
			n++;
			for (int j=0; j < columns; j++){
				new_data[n]=data[i*columns + j];
				n++;
			}
		}
		
		this.data=new_data;
		this.columns++;
	}

	@Override
	public double [] GetColumn( int column) {

        double [] columntoget = new double [rows];
        
        for (int i=0; i <rows; i++ ){
        	columntoget[i]=this.data[i*columns + column];
        }
        
		return columntoget;
	}
	
	@Override
	public void AddRow(double[] row) {
		
		if (row.length!=columns){
			throw new DimensionMismatchException(row.length,columns);
			
		}
		double [] newdata= new double [rows*columns + columns];
		int n=0;
		int row_n=0;
		for (int i=0; i <rows; i++ ){
			for (int j=0; j<columns; j++ ){
				newdata[n]=this.data[columns*i + j];
				n++;
			}
			}
		// add the last row
		for (int j=0; j<columns; j++ ){
			newdata[n]=row[row_n];
			n++;
			row_n++;
		}
		
		
		this.data=newdata;
		this.rows++;
	}

	@Override
	public double []GetRow( int row) {
        double [] rowtoget = new double [columns];
        
        for (int i=0; i <columns; i++ ){
        	rowtoget[i]=this.data[row*columns + i];
        }
        
		return rowtoget;
	}
	
	@Override
	public void RemoveColumn() {
		double [] newdata= new double [rows*columns - rows];
		int n=0;
		for (int i=0; i <rows; i++ ){
			for (int j=0; j<columns; j++ ){
				if (j!=columns-1) {
				newdata[n]=this.data[columns*i + j];
				n++;
				}
			}

			}
		
		this.data=newdata;
		this.columns--;
		
		
	}

	@Override
	public void RemoveRow() {
		
		double [] newdata= new double [rows*columns - columns];
		int n=0;
		for (int i=0; i <rows-1; i++ ){
			for (int j=0; j<columns; j++ ){
				newdata[n]=this.data[columns*i + j];
				n++;
			}
			}
			
		this.data=newdata;
		this.rows--;
		
	}

	/**
	 * @return returns a new (hard) copy of the matrix
	 */
	public matrix Copy() {
      return new fsmatrix(this.GetData(), this.rows, this.columns);	
	}

	@Override
	public void PrintInfo() {
		System.out.println(" Fixed size Matrix [ " + this.rows + " X " + this.columns + " ] ");
		System.out.println(" Total Elements " +this.rows + " X " + this.columns + " = " + (this.rows *  this.columns));
	}

	/**
	 * 
	 * @param array : array to print
	 * @param cases : number of cases from the beginning to print
	 */
     public  void Print( int cases){
		
		if (cases <=0){
			cases=rows;
		}
		if (cases >100){
			cases=100;
		}
		if (cases >rows){
			cases=rows;
		}
		// printing
		for (int i=0; i < cases; i++) {
			System.out.print("row: " + i);
			for (int j=0; j < columns; j++) {							
			   System.out.print( "  col" + j +":"  + this.GetElement(i, j));
			}
			System.out.println("");
		}
		
	}

	@Override
	public void ToFile(String file) {
		output out = new output();
		out.printfsmatrix(this, file);
		
	}		

	public void ToFileTarget(String file, double target[]) {
		output out = new output();
		out.target=target;
		out.printfsmatrix(this, file);
		
	}

}
