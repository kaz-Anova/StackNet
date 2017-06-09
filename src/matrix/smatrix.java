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
import java.util.Arrays;

import utilis.map.intint.IntIntMapminus4a;
import io.output;
import exceptions.DimensionMismatchException;
import exceptions.IllegalStateException;
import exceptions.NullObjectException;

/**
 * 
 * @author marios
 * <p> This is the most basic sparse matrix implementation that contains 3 arrays:
 * <ol>
 * <li>values' array</li>
 * <li>Rows' array</li>
 * <li>Columns' array</li>
 * </ol>
 * </p>
 * 
 */
public class smatrix implements matrix, Serializable {
	/**
	 * a serialised id.
	 */
	private static final long serialVersionUID = -6610797892124240356L;
	/**
	 * values' array
	 */
	public double [] valuespile;
	/**
	 * columns' array
	 */
	public int [] mainelementpile;
	/**
	 * rows' array
	 */
	public int [] indexpile;
	
	/***
	 * for quick row and column retrievals
	 */
	public IntIntMapminus4a indexer;
	
	/**
	 * build map for quick accessing of columns, rows 
	 */
	public void buildmap(){
		
		 if (valuespile==null || valuespile.length==0){
			 throw new IllegalStateException(" There is nothing to convert, matrix is empty");
		 }
		indexer= new IntIntMapminus4a(valuespile.length, 0.99F);
		
		if (this.iscolumnmatrix){
			
			for (int i=0; i < this.rows;i++ ){
				for (int c=this.indexpile[i]; c < this.indexpile[i+1];c++ ){
					int j=this.mainelementpile[c];
					indexer.put(i*this.GetColumnDimension() + j, c);
				}
			}	
		} else {
			
			for (int j=0; j < this.columns;j++ ){
				for (int c=this.indexpile[j]; c < this.indexpile[j+1];c++ ){
					int i=this.mainelementpile[c];
					indexer.put(i*this.GetColumnDimension() + j, c);
				}
			
			}		
			
			
		}
	}
	/**
	 * build map for quick accessing of columns, rows with a float loader 
	 */
	public void buildmap(float value){
		
		 if (valuespile==null || valuespile.length==0){
			 throw new IllegalStateException(" There is nothing to convert, matrix is empty");
		 }
		indexer= new IntIntMapminus4a(valuespile.length,value);
		
		if (this.iscolumnmatrix){
			
			for (int i=0; i < this.rows;i++ ){
				for (int c=this.indexpile[i]; c < this.indexpile[i+1];c++ ){
					int j=this.mainelementpile[c];
					indexer.put(i*this.GetColumnDimension() + j, c);
				}
			}	
		} else {
			
			for (int j=0; j < this.columns;j++ ){
				for (int c=this.indexpile[j]; c < this.indexpile[j+1];c++ ){
					int i=this.mainelementpile[c];
					indexer.put(i*this.GetColumnDimension() + j, c);
				}
			
			}		
			
			
		}
	}
	
	/**
	 * 
	 * @param rowtoget : row to access
	 * @param columntoget : column to access
	 * @return the elements that sits (virtually) in [rowtoget,columntoget]
	 */
	
	public double GetElement(int rowtoget, int columntoget){
		int index=indexer.get(rowtoget*this.GetColumnDimension()+columntoget );
		return index>-1 ?  this.valuespile[index] : 0.0  ;
	}
	/**
	 * These will hold the the row indices in the same way as the 
	 * column indices (useful for some algos)
	 */
	public int optional_rows [];
	
	/**
	 * Number of Rows
	 */

	private int rows;
	/**
	 * Number of columns
	 */
	private int columns;
	/**
	 * current length
	 */
	 private int slength;
	 

	 /**
	  * Holds whether the matrix is row or column oriented
	  */
	private boolean iscolumnmatrix=true;
	private int[] columnspile;
	
	/**
	 * method to create a sparse matrix
	 * @param valuespile : the values array
	 * @param mainelementpile : the supporting array
	 * @param indexpile : the main index array
	 * @param rows : the row dimension
	 * @param columns : the column dimension
	 * @param slength : total non-zero elements
	 * @param iscolumnmatrix : if it is column matrix or not
	 */
	private smatrix set_values( double [] valuespile,int [] mainelementpile, int [] indexpile,int rows,int columns,int slength, boolean iscolumnmatrix){
		
		smatrix b = new smatrix();
		
		b.valuespile=valuespile;
		b.mainelementpile= mainelementpile;
		b.indexpile= indexpile;
		b.rows= rows;
		b.columns= columns;
		b.slength= slength;
		b.iscolumnmatrix= iscolumnmatrix;
		return b;
	}
	
	/**
	 * create smatrix from file
	 * @param Filename : the name of the file to load
	 * @param targets : whether it contains a target variable
	 */
	public static smatrix smatrxfromfile(String Filename, boolean targets){
		io.input ios= new io.input();
		return ios.readsmatrixdata(Filename, ":",false,targets);
		
	}
	
	/**
	 * Constructor but not recommended. Only if you know what you are doing!
	 * @param valuespile : the values array
	 * @param mainelementpile : the supporting array
	 * @param indexpile : the main index array
	 * @param rows : the row dimension
	 * @param columns : the column dimension
	 * @param slength : total non-zero elements
	 * @param iscolumnmatrix : if it is column matrix or not
	 */
	 public smatrix( double [] valuespile,int [] mainelementpile, int [] indexpile,int rows,int columns,int slength, boolean iscolumnmatrix){
		
		 optional_rows=null;
	
		this.valuespile=valuespile;
		this.mainelementpile= mainelementpile;
		this.indexpile= indexpile;
		this.rows= rows;
		this.columns= columns;
		this.slength= slength;
		this.iscolumnmatrix= iscolumnmatrix;

	}
	
	/**
	 * changes from row-oriented to column oriented and vice versa!
	 */
	 public void convert_type(){
		 optional_rows=null;
		 if (valuespile==null || valuespile.length==0){
			 throw new IllegalStateException(" There is nothing to convert, matrix is empty");
		 }

		iscolumnmatrix=!iscolumnmatrix ;
	
		int columns[]= new int [slength];
		for (int i=0; i <this.indexpile.length-1;i++){

			for (int j=this.indexpile[i]; j <this.indexpile[i+1];j++ ){
				columns[j]=i;
			}
			
		}
		manipulate.sort.mergesorts.mergesort(this.mainelementpile ,columns , valuespile, 0, slength-1);
		
		//this.mainelementpile=columns;
		if (iscolumnmatrix){
		this.indexpile= new int[this.rows+1];
		
		} else {
			this.indexpile= new int[this.columns+1];
		}
		
		int temp=0;
		
		temp=this.mainelementpile[0];
		
		for (int j=0; j <temp+1;j++ ){
			this.indexpile[j]=0;
		}
		for (int i=1; i <slength;){
			
			while(i<slength && this.mainelementpile[i]==temp  ){
				i++;
			}				
			if (i<slength && this.mainelementpile[i]!=temp ){					
				if (this.mainelementpile[i]>temp){
					for (int j=temp+1; j <this.mainelementpile[i]+1;j++ ){
						this.indexpile[j]=i;
					}
					temp=this.mainelementpile[i];
					i++;
				} else {
					throw new IllegalStateException(" The row array needs to be sorted before provided");
				}
			}
			
		}	
		
		this.indexpile[this.indexpile.length-1]= slength;
		this.mainelementpile=columns;
		 
		 
	 }
	 /**
	  * 
	  * @param cols : Columimension to set
	  * <b> WARNING </b> This is for internal use only
	  */
	 public void set_column_dimension(int cols){
		 this.columns=cols;
	 }
	 
	 public void void_update_indice(){
		 
		 if (this.valuespile==null || this.valuespile.length<=0){
			 
			  throw new IllegalStateException(" there is nothing to add elements for!");	
		 }
		 optional_rows= new int [this.valuespile.length];
		 for (int i=0; i< this.GetRowDimension(); i++){
			 for (int f=this.indexpile[i]; f< this.indexpile[i+1]; f++){
				 optional_rows[f]=i;
			 }
		 }
		 
	 }
	 
	/** <p> Basic constructor where the values, rows and columns are known </p>
	 * @param values : double values with the elements. 
	 * @param rows : int array with the rows (virtually) the elements should be
	 * @param columns : int array with the columns (virtually) the elements should be
	 * @param column_oriented : if the matrix is column oriented or not
	 * @return smatrix
	 * <b> WARNING </b> values need to be ordered from row to column.
	 */
	 
	 
	
	public smatrix(double values [], int rows [], int columns[], boolean column_oriented) {
		
		if (values.length != rows.length){
		    throw new DimensionMismatchException(values.length,rows.length);	
		}
		if (rows.length != columns.length){
		    throw new DimensionMismatchException(rows.length,columns.length);	
		}	
		int max_row=0;
		int max_column=0;
		for (int i=0; i < rows.length; i++ ){
			if (rows[i]<0 || columns[i]<0){
				throw new IllegalStateException(" Matrix indices cannot be negative");
			}
			if (rows[i]>max_row){
				max_row=rows[i];
			}
			if (columns[i]>max_column){
				max_column=columns[i];
			}
		}
		int temp=0;
		// assign the values //
		this.columns=max_column+1;
		this.rows=max_row+1;
		this.valuespile=values;
		slength=values.length;
		
		if (column_oriented){
			
			this.mainelementpile=columns.clone();
			iscolumnmatrix=true ;
			this.indexpile= new int[this.rows+1];
			
			temp=rows[0];
			
			for (int j=0; j <temp+1;j++ ){
				this.indexpile[j]=0;
			}
			for (int i=1; i <slength;){
				
				while(i<slength && rows[i]==temp  ){
					i++;
				}				
				if (i<slength && rows[i]!=temp ){					
					if (rows[i]>temp){
						for (int j=temp+1; j <rows[i]+1;j++ ){
							this.indexpile[j]=i;
						}
						temp=rows[i];
						i++;
					} else {
						throw new IllegalStateException(" The row array needs to be sorted before provided");
					}
				}
				
			}

		} else {
			
			this.mainelementpile=rows;
			iscolumnmatrix=false;
			this.indexpile= new int[this.columns+1];
			
			temp=columns[0];
			
			for (int j=0; j <temp+1;j++ ){
				this.indexpile[j]=0;
			}
			for (int i=1; i <slength;){
				
				while(i<slength  && columns[i]==temp ){
					i++;
				}
				
				if (i<slength &&  columns[i]!=temp){
					
					if (columns[i]>temp){
						
						for (int j=temp+1; j <columns[i]+1;j++ ){
							this.indexpile[j]=i;
						}
						temp=columns[i];
						i++;
					} else {
						throw new IllegalStateException(" The column array needs to be sorted before provided");
					}
				}
				
			}
		}
		
		this.indexpile[this.indexpile.length-1]= slength;
		
	}
	

	/**
	 * @param values : double values with the elements. 
	 * @param rows : int array with the rows (virtually) the elements should be
	 * @param columns : int array with the columns (virtually) the elements should be
	 * @param n_rows : The row dimension the new matrix should have
	 * @param n_columns : the column dimension the new matrix should have
	 * @return smatrix
	 * <b> WARNING </b> values need to be ordered from row to column.
	 */ 

	public smatrix(double values [], int rows [], int columns[], int n_rows, int n_columns, boolean column_oriented) {
		
		if (values.length != rows.length){
		    throw new DimensionMismatchException(values.length,rows.length);	
		}
		if (rows.length != columns.length){
		    throw new DimensionMismatchException(rows.length,columns.length);	
		}	

		for (int i=0; i < rows.length; i++ ){
			if (rows[i]<0 || columns[i]<0){
				throw new IllegalStateException(" Matrix indices cannot be negative");
			}
			if (rows[i]>n_rows-1){
				 throw new DimensionMismatchException(rows[i],n_rows-1);	
			}
			if (columns[i]>n_columns-1){
				 throw new DimensionMismatchException(columns[i],n_columns-1);	
			}
		}
		
		// assign the values //
		this.columns=n_columns;
		this.rows=n_rows;
		valuespile=values;
		slength=values.length;
		int temp=0;
		
		if (column_oriented){
			
			this.mainelementpile=columns;
			iscolumnmatrix=true ;
			this.indexpile= new int[this.rows+1];
			
			temp=rows[0];
			
			for (int j=0; j <temp+1;j++ ){
				this.indexpile[j]=0;
			}
			for (int i=1; i <slength;){
				
				while( i<slength && rows[i]==temp){
					i++;
				}
				
				if (i<slength && rows[i]!=temp  ){
					
					if (rows[i]>temp){
						
						for (int j=temp+1; j <rows[i]+1;j++ ){
							this.indexpile[j]=i;
						}
						temp=rows[i];
						i++;
					} else {
						throw new IllegalStateException(" The row array needs to be sorted before provided");
					}
				}
				
			}

		} else {
			
			this.mainelementpile=rows;
			iscolumnmatrix=false;
			this.indexpile= new int[this.columns+1];
			
			temp=columns[0];
			
			for (int j=0; j <temp+1;j++ ){
				this.indexpile[j]=0;
			}
			for (int i=1; i <slength;){
				
				while(i<slength &&  columns[i]==temp  ){
					i++;
				}
				
				if (i<slength &&  columns[i]!=temp ){
					
					if (columns[i]>temp){
					
						for (int j=temp+1; j <columns[i]+1;j++ ){
							this.indexpile[j]=i;
						}
						temp=columns[i];
						i++;
					} else {
						throw new IllegalStateException(" The row array needs to be sorted before provided");
					}
				}
				
			}
		}
		
		this.indexpile[this.indexpile.length-1]= slength;		
		
		
	}
	
	
	/**
	 * This method ensures the there are no zeros. If they do- it removes them
	 */
	public void trim(){
		
		if (this.valuespile==null ||this.valuespile.length <=0){
			throw new IllegalStateException(" Matrix is empty");
		}
		//check if there are zero
		int zero_count=0;
		// find number of zeros
		for (int i=0; i < this.valuespile.length; i++){
			if (this.valuespile[i]==0.0){
				zero_count++;
			}
		}
		// if there are zeros...
		if (zero_count>0){
			
			double values[]= new double [this.valuespile.length -zero_count];
			int cols[]= new int [this.valuespile.length -zero_count];
			int rows[]=new int [this.indexpile.length] ;	
			int case_counter=0;
			
			for (int i=0; i <this.indexpile.length-1; i++ ){
				
				rows[i]=case_counter;
				for (int j=this.indexpile[i]; j< this.indexpile[i+1];j++){
					
					if (this.valuespile[j]!=0.0){
						
						values[case_counter]=this.valuespile[j];
						cols[case_counter]=this.mainelementpile[j];
						case_counter++;
					}
					
				}
				rows[i+1]=case_counter;
				
			}
			
		}
		
	}
	
	/**
	 * @param values : 2d array to convert to sparse matrix 
	 * @return smatrix
	 * <p> create sparse matrix from double array</p>
	 */
	public smatrix(double values [][]) {
		
		if (values==null ||values.length <=0){
			throw new IllegalStateException(" Matrix is empty");
		}

		int nonzero_counter=0;
		//count number of non zeros
		for (int i=0; i < values.length; i++ ){
			for (int j=0; j < values[0].length; j++ ){
				if (values[i][j]!=0.0){
					nonzero_counter++;
				}
				
			}
		}
		int n=0;
		// assign the values //
		this.valuespile= new double [nonzero_counter];
		this.columns=values[0].length;
		this.rows=values.length;
		slength=nonzero_counter;	
		this.mainelementpile=new int [nonzero_counter];		
		iscolumnmatrix=true ; //default
		this.indexpile= new int[this.rows+1];
		
		
		for (int i=0; i < values.length; i++ ){			
			for (int j=0; j < values[0].length; j++ ){
				
				if (values[i][j]!=0.0){
					
					this.valuespile[n]=values[i][j];
					this.mainelementpile[n]=j;
					n++;
				}
					
			}
			this.indexpile[i+1]=n;
		}
		this.indexpile[this.indexpile.length-1]= slength;

	}
	
	
	/**
	 * @param values : 2d array to convert to sparse matrix 
	 * @param rows : rows to be used for the selection
	 * @return smatrix
	 * <p> create sparse matrix from double array</p>
	 */
	public smatrix(double values [][], int [] rows) {
		
		if (values==null ||values.length <=0 || rows==null || rows.length<0){
			throw new IllegalStateException(" Matrix is empty");
		}

		int nonzero_counter=0;
		//count number of non zeros
		for (int i=0; i < rows.length; i++ ){
			for (int j=0; j < values[0].length; j++ ){
				if (values[rows[i]][j]!=0.0){
					nonzero_counter++;
				}
				
			}
		}
		int n=0;
		// assign the values //
		this.valuespile= new double [nonzero_counter];
		this.columns=values[0].length;
		this.rows=rows.length;
		slength=nonzero_counter;	
		this.mainelementpile=new int [nonzero_counter];		
		iscolumnmatrix=true ; //default
		this.indexpile= new int[this.rows+1];
		
		
		for (int i=0; i < rows.length; i++ ){			
			for (int j=0; j < values[0].length; j++ ){
				
				if (values[rows[i]][j]!=0.0){
					
					this.valuespile[n]=values[rows[i]][j];
					this.mainelementpile[n]=j;
					n++;
				}
					
			}
			this.indexpile[i+1]=n;
		}
		this.indexpile[this.indexpile.length-1]= slength;

	}	
	/**
	 * @param values : fsmatrix to create sparse matrix 
	 * @return smatrix
	 * <p> create sparse matrix from fsmatrix</p>
	 */
	public  smatrix (fsmatrix values ) {
		
		if (values==null ||values.GetRowDimension() <=0){
			throw new IllegalStateException(" Matrix is empty");
		}

		int nonzero_counter=0;
		int n=0;
		//count number of non zeros
		for (int i=0; i < values.GetRowDimension(); i++ ){
			for (int j=0; j < values.GetColumnDimension(); j++ ){
				if (values.data[n]!=0.0){
					nonzero_counter++;
				}
				n++;	
			}
		}
		n=0;
		int k=0;
		this.valuespile= new double [nonzero_counter];
		this.columns=values.GetColumnDimension();
		this.rows=values.GetRowDimension();
		slength=nonzero_counter;	
		this.mainelementpile=new int [nonzero_counter];		
		iscolumnmatrix=true ; //default
		this.indexpile= new int[this.rows+1];
		
		for (int i=0; i < values.GetRowDimension(); i++ ){			
			for (int j=0; j < values.GetColumnDimension(); j++ ){
				
				if (values.data[k]!=0.0){
					this.valuespile[n]=values.data[k];
					this.mainelementpile[n]=j;
					n++;
				}
				k++;
			}
			this.indexpile[i+1]=n;			
		}
		this.indexpile[this.indexpile.length-1]= slength;
	}	

	/**
	 * @param values : fsmatrix to create sparse matrix 
	 * @param rows : rows to be used for the selection
	 * @return smatrix
	 * <p> create sparse matrix from fsmatrix </p>
	 */
	public smatrix(fsmatrix values , int [] rows) {
		
		if (values==null ||values.GetRowDimension() <=0 || rows==null || rows.length<0){
			throw new IllegalStateException(" Matrix is empty");
		}

		int nonzero_counter=0;
		int n=0;
		//count number of non zeros
		for (int i=0; i < rows.length; i++ ){
			for (int j=0; j < values.GetColumnDimension(); j++ ){
				if (values.GetElement(rows[i], j)!=0.0){
					nonzero_counter++;
				}
				n++;	
			}
		}
		n=0;
		this.valuespile= new double [nonzero_counter];
		this.columns=values.GetColumnDimension();
		this.rows=rows.length;
		slength=nonzero_counter;	
		this.mainelementpile=new int [nonzero_counter];		
		iscolumnmatrix=true ; //default
		this.indexpile= new int[this.rows+1];
		
		for (int i=0; i < rows.length; i++ ){			
			for (int j=0; j < values.GetColumnDimension(); j++ ){
				
				if (values.GetElement(rows[i], j)!=0.0){
					this.valuespile[n]=values.GetElement(rows[i], j);
					this.mainelementpile[n]=j;
					n++;
				}

			}
			this.indexpile[i+1]=n;			
		}
		this.indexpile[this.indexpile.length-1]= slength;
		
		
	}	
	
	
	/**
	 * @param values : smatrix to create a sub-selection smatrix 
	 * @param rows : rows to be selected
	 * @return smatrix
	 * <p> create sparse matrix from sparse matrix/p>
	 * <b>WARNING</b> this method changes the type of the matrix and sorts the integer array
	 */
	public smatrix makesubmatrix(int [] rows) {
		
		if ( rows==null || rows.length<0){
			throw new IllegalStateException(" Matrix is empty");
		}

		if (this.iscolumnmatrix==false){
			this.convert_type();
		}
		
		int nonzero_counter=0;

		//count number of non zeros
		for (int i=0; i < rows.length; i++ ){
			for (int j=this.indexpile[rows[i]]; j < this.indexpile[rows[i]+1]; j++ ){
					nonzero_counter++;
			}
		}
		int n=0;
		double [] valuespile= new double [nonzero_counter];
		int columns=this.GetColumnDimension();
		int row=rows.length;
		int slength=nonzero_counter;	
		int [] mainelementpile=new int [nonzero_counter];		
		boolean iscolumnmatrix=true ; //default
		int [] indexpile= new int[row+1];
		Arrays.sort(rows);
		
		for (int i=0; i < rows.length; i++ ){
			indexpile[i]=n;
			
			for (int j=this.indexpile[rows[i]]; j < this.indexpile[rows[i]+1]; j++ ){
				valuespile[n]=this.valuespile[j];
				mainelementpile[n]=this.mainelementpile[j];
				
					n++;
			}
			
		}		
		indexpile[indexpile.length-1]= n;
		return set_values(valuespile,mainelementpile,indexpile,row,columns,slength,iscolumnmatrix);
		
	}	
		
	/**
	 * @param values : smatrix to create a sub-selection smatrix 
	 * @param max_col :maximum column dimension to allow
	 * @return smatrix
	 * <p> create sparse smatrix from sparse matrix/p>
	 * <b>WARNING</b> this method changes the type of the matrix and sorts the integer array
	 */
	public smatrix makesubmatrixcols(int max_col) {
		
		if ( max_col<=0 ){
			throw new IllegalStateException(" max_col needs to be higher than zero");
		}

		if (this.iscolumnmatrix){
			int validcounter=0;
			
			for (int i=0; i < this.mainelementpile.length; i++ ){
						if (this.mainelementpile[i]<=max_col-1){
							validcounter+=1;
						}
			}
			int indexes[] = new int [this.indexpile.length];
			double [] valuespile= new double [validcounter];
			int columns=max_col;
			int [] mainelementpile=new int [validcounter];	
			int counter=0;
			
			for (int r=0; r <this.indexpile.length-1; r++ ){
				indexes[r]=counter;
				for  (int j=this.indexpile[r]; j<this.indexpile[r+1];j++ ){
					
					int col=this.mainelementpile[j];
					double val=this.valuespile[j];	
					
					if (col<=max_col-1){
						
						mainelementpile[counter]=col;
						valuespile[counter]=val;
						counter++;
					}
				}
				indexes[r+1]=counter;
			}
			
			return new smatrix( valuespile,mainelementpile, indexes,indexes.length-1, columns,validcounter,  true);	
		} else if (max_col>=this.GetColumnDimension()){
			return this;
		} else if (!this.iscolumnmatrix){
			int validcounter=0;
			
			for (int i=0; i < max_col; i++ ){
				for (int j=this.indexpile[i]; j < this.indexpile[i+1]; j++ ){
					validcounter+=1;
				}
				
			}
			double [] valuespile= new double [validcounter];
			int [] mainelementpile=new int [validcounter];	
			int [] indexpile=new int [max_col+1];
			int columns=max_col;
			int rows=this.GetRowDimension();
			for (int i=0; i < max_col; i++ ){
				indexpile[i]=this.indexpile[i];
				indexpile[i+1]=this.indexpile[i+1];
				
				for (int j=this.indexpile[i]; j < this.indexpile[i+1]; j++ ){
					valuespile[j]=this.valuespile[j];
					mainelementpile[j]=this.mainelementpile[j];
				}
				
			}
			
			return new smatrix( valuespile,mainelementpile, indexpile,rows,columns,validcounter,  false);	
			
		} else {
			return this;
		}
		
	}	
		
	private smatrix() {
		// TODO Auto-generated constructor stub
	}

	/*
	/**
	 * Sort sparse matrix by row

	public void SortByRow(){
		manipulate.sort.mergesorts.mergesort(rowspile, columnspile, valuespile, 0, slength-1);
	}
	/**
	 * Sort sparse matrix by column

	public void SortByColumn(){
		manipulate.sort.mergesorts.mergesort(columnspile ,rowspile , valuespile, 0, slength-1);
	}	
		 */
	/**
	 * checks if current matrix is sorted by row
	 */
	public boolean IsSortedByRow(){

		return iscolumnmatrix;
	}
	/**
	 * checks if current matrix is sorted by row
	 */
	
	public boolean IsSortedByColumn(){

		return !iscolumnmatrix;
	}
	
	

	@Override
	public int GetRowDimension() {
		return rows;
	}

	@Override
	public int GetColumnDimension() {
		return columns;
	}
	/**
	 * 
	 * @return  current length of all non-zero elements
	 */
	public int GeLength() {
		return slength;
	}

	@Override
	public void AddColumn(double[] coulmn) {
		
		if (coulmn==null){
			throw new IllegalStateException(" The column object to be added cannot be null ");
		}
		/* 
		 * check if column length equals to rows
		 */
		
		if (coulmn.length!=rows){
			throw new DimensionMismatchException(coulmn.length,rows);
		}
		// we count non-zero elements
		
		int number_of_non_zero=0;
		for (int i=0; i <coulmn.length;i++ ){
			if (coulmn[i]!=0.0){
				number_of_non_zero++;
			}
		}
		
		//final we check 2 cases. If iscolumnmatrix or not
		
		if (this.iscolumnmatrix){
			
			double [] newdata= new double [slength+ number_of_non_zero];
			int new_rows[]= new int[this.GetRowDimension()];
			int new_columns[]= new int[slength+ number_of_non_zero];
			
			int casecounter=0;
			
			for (int i=0; i <this.GetRowDimension(); i++ ){
				int start=this.indexpile[i];
				int end=this.indexpile[i+1];
				
				if (start<end){
					new_rows[i]=casecounter;
					
					for (int j=start; j < end; j++){
						new_columns[casecounter]=this.mainelementpile[j];
						newdata[casecounter]=this.valuespile[j];
						
					}
					if (coulmn[i]!=0.0){
						new_columns[casecounter]=this.GetColumnDimension();
						newdata[casecounter]=coulmn[i];
						casecounter+=1;
					}
					
				} else if (start==end){
					
					if (coulmn[i]!=0.0){
						new_columns[casecounter]=this.GetColumnDimension();
						newdata[casecounter]=coulmn[i];
						casecounter+=1;
					}
				} else {
					throw new IllegalStateException(" The existing sparse matrix has a row index of a previous row higher than a later row...");
				}
				
				new_rows[new_rows.length-1]=slength+ number_of_non_zero;
				
				
			}
			
			this.valuespile=newdata;
			this.mainelementpile= new_columns;
			this.indexpile= new_rows;
			this.columns= columns+1;
			this.slength= slength + number_of_non_zero;
			

		// if it is not a column matrix
		} else {
			
			double [] newdata= new double [slength+ number_of_non_zero];
			int new_columns[]= new int[this.GetColumnDimension()+1];
			int new_rows[]= new int[slength+ number_of_non_zero];
				
			int casecounter=slength;
			
			//make a copy of elements
			for (int k =0; k < this.valuespile.length; k++){
				newdata[k]=this.valuespile[k];
				new_rows[k]=this.mainelementpile[k];
			}
			
			// copy columns
			for (int k =0; k < this.indexpile.length; k++){
				new_columns[k]=this.indexpile[k];
			}
			
			new_columns[new_columns.length -1]=slength+ number_of_non_zero;
			
			for (int i=0; i < coulmn.length; i++){
				if (coulmn[i]!=0.0){
					newdata[casecounter]=coulmn[i];
					new_rows[casecounter]=i;
					casecounter+=1;
				}
				
			}
			
			this.valuespile=newdata;
			this.mainelementpile= new_rows;
			this.indexpile=  new_columns ;
			this.columns= columns+1;
			this.slength= slength + number_of_non_zero;

		
		}
		
				
	}

	@Override
	public void AddRow(double[] row) {
		
		if (!this.iscolumnmatrix){
			this.convert_type();
		}
		if (row.length!=columns){
			throw new DimensionMismatchException(row.length,columns);
		}
		int number_of_non_zero=0;
		for (int i=0; i <row.length;i++ ){
			if (row[i]!=0.0){
				number_of_non_zero++;
			}
		}
		double [] newdata= new double [slength+ number_of_non_zero];
		int new_rows[]= new int[this.indexpile.length+1];
		int new_columns[]= new int[slength+ number_of_non_zero];		
		
		int n=0;
		for (int i=0; i <slength; i++ ){
			newdata[n]=this.valuespile[i];
			new_columns[n]=this.columnspile[i];
			n++;
			}
		for (int i=0; i <this.indexpile.length; i++ ){
			new_rows[i]=this.indexpile[i];
		}
		
		
		for (int i=0; i <row.length; i++ ){
			if (row[i]!=0.0){
				newdata[n]=row[i];
				new_columns[n]=i;
				n++;
			}
		}
		new_rows[new_rows.length-1]=n;
		
		this.rows++;
		slength+= number_of_non_zero;
		valuespile=newdata;
		columnspile=new_columns;
		this.indexpile=new_rows;
		this.indexer=null;
	}

	@Override
	public void RemoveColumn() {
		
		if (this.GetColumnDimension()==1){
			throw new IllegalStateException(" Can't remove last (and only) column from this matrix.");
		}
		if (this.iscolumnmatrix){
			//count non-zero elements of the last column
			int lat_column=this.columns-1;
			int nonzero_of_last_c=0;
			for (int i=0; i <this.mainelementpile.length;i++){
				if (this.mainelementpile[i]==lat_column ){
					nonzero_of_last_c+=1;
				}
			}
			double [] newdata= new double [slength- nonzero_of_last_c];
			int new_columns[]= new int[slength- nonzero_of_last_c];
			int new_rows[]= new int[this.indexpile.length];
			
			new_rows[0]=0;
			int counter=0;
			for (int k=0; k < this.rows; k++){
				int start=this.indexpile[k];
				int end=this.indexpile[k+1];
				for (int j=start; j <end; j++){
					if (this.mainelementpile[j]!=lat_column ){
						new_columns[counter]=this.mainelementpile[j];
						newdata[counter]=this.valuespile[j];
						counter++;
					} 
				}
					new_rows[k+1]=counter;

			}
			
			this.valuespile=newdata;
			this.mainelementpile= new_columns;
			this.indexpile= new_rows;
			this.columns= columns-1;
			this.slength= slength - nonzero_of_last_c;
			
		} else {
			int nonzero_of_last_c=0;
			int last_col=this.columns-1;
			for (int j=this.indexpile[last_col]; j <this.indexpile[last_col+1]; j++){
				nonzero_of_last_c+=1;
			}
			double [] newdata= new double [slength- nonzero_of_last_c];
			int new_columns  []= new int[this.indexpile.length-1];
			int new_rows[]= new int[slength- nonzero_of_last_c ];	
			
			for (int i=0; i <this.indexpile[last_col];i++ ){
				newdata[i]=this.valuespile[i];
				new_rows[i]=this.mainelementpile[i];				
			}
			
			for (int i=0; i <last_col;i++ ){
				new_columns[i]=this.indexpile[i];
				
			}
			
			this.valuespile=newdata;
			this.mainelementpile=new_rows ;
			this.indexpile= new_columns ;
			this.columns= columns-1;
			this.slength= slength - nonzero_of_last_c;
			
		}
		
		
	}

	@Override
	public void RemoveRow() {
		
		// not available for sparse matrix
		
	}

	@Override
	public double[] GetColumn(int column) {
		if (column>=this.columns){
			throw new IllegalStateException("current column exceeds current diension: " + this.columns);
		}
		
		double column_tosend []= new double[rows];
		
		if (this.iscolumnmatrix){
			
			if (this.indexer!=null){
				for (int i=0; i < this.rows;i++ ){
					column_tosend[i]=this.GetElement(i, column);
				}
			} else {
			
				for (int i=0; i < this.rows;i++ ){
					for (int c=this.indexpile[i]; c < this.indexpile[i+1];c++ ){
						int j=this.mainelementpile[c];
						if (j< column){
							continue;
						}
						if (j==column){
							column_tosend[i]=this.valuespile[c];
							break;
						}
						if (j>column){
							break;
						}
						//double value=this.valuespile[c];
					
					}
				
			
				}	
			}
		} else {
			for (int c=this.indexpile[column]; c < this.indexpile[column+1];c++ ){
				int i=this.mainelementpile[c];
				//double value=this.valuespile[c];
				column_tosend[i] =this.valuespile[c];
			    
			}
			
		}
		
		

		return column_tosend;
	}

	@Override
	public double[] GetRow(int row) {
		if (row>=this.rows){
			throw new IllegalStateException("current row exceeds current diension: " + this.rows);
		}
		
		double row_tosend []= new double[columns];
		if (this.iscolumnmatrix){
			
			for (int c=this.indexpile[row]; c < this.indexpile[row+1];c++ ){
				int j=this.mainelementpile[c];
				//double value=this.valuespile[c];
				row_tosend[ j] =this.valuespile[c];
			}
			
		}
		else {
			if (this.indexer!=null){
				for (int j=0; j < this.columns;j++ ){
					row_tosend[j]=this.GetElement(row, j);
				}
			} else {
				
			
				for (int j=0; j < this.columns;j++ ){
					for (int c=this.indexpile[j]; c < this.indexpile[j+1];c++ ){
						int i=this.mainelementpile[c];
						//double value=this.valuespile[c];
					    if (i==row){
					    	row_tosend[ j] =this.valuespile[c];
					    }
					}
					
				
				}	
			}
		}
		

		return row_tosend;
	}

	@Override
	public matrix Copy() {
		
		double [] newdata= new double [slength];
		//System.out.println(slength + " " +this.valuespile.length );
		int mainp[]= new int[slength];
		int newindex[]= new int[this.indexpile.length];
		
		for (int i=0; i < this.valuespile.length;i++ ){
			newdata[i]=this.valuespile[i];
			mainp[i]=this.mainelementpile[i];

		}
		for (int i=0; i < this.indexpile.length;i++ ){
			newindex[i]=this.indexpile[i];


		}
		
		return  set_values( newdata, mainp,  newindex,this.rows,this.columns,this.slength, this.IsSortedByRow());
	
	}
	
	/**
	 * 
	 * @return a fixed size matrix
	 * <p> convert a  fixed-size matrix to a sparse matrix  
	 */
	public fsmatrix ConvertToFixedSizeMatrix(){
		
		if (this.valuespile==null){
			throw new NullObjectException(" The sparse matrix is empty, nothing to make a matrix from.");
		}
		double fs_data[]= new double [ this.columns * this.rows];
		
		if (this.iscolumnmatrix){
			
			for (int i=0; i < this.rows;i++ ){
				for (int c=this.indexpile[i]; c < this.indexpile[i+1];c++ ){
					int j=this.mainelementpile[c];
					//double value=this.valuespile[c];
					fs_data[i *this.columns+  j] =this.valuespile[c];
				}
				
			
			}	
		} else {
			
			for (int j=0; j < this.columns;j++ ){
				for (int c=this.indexpile[j]; c < this.indexpile[j+1];c++ ){
					int i=this.mainelementpile[c];
					//double value=this.valuespile[c];
					fs_data[i *this.columns+  j] =this.valuespile[c];
				}
				
			
			}		
			
			
		}

		
		return new fsmatrix(fs_data,  this.rows, this.columns);
	}
	/**
	 * 
	 * @return a double array
	 * <p> convert a fixed-size matrix to sparse matrix 
	 */
	public double [][] ConvertToDoubleArray(){
		
		if (this.valuespile==null){
			throw new NullObjectException(" The sparse matrix is empty, nothing to make a matrix from.");
		}
		double fs_data[][]= new double [ this.rows] [this.columns ];
		
		if (this.iscolumnmatrix){
			
			for (int i=0; i < this.rows;i++ ){
				for (int c=this.indexpile[i]; c < this.indexpile[i+1];c++ ){
					int j=this.mainelementpile[c];
					//double value=this.valuespile[c];
					fs_data[i][j] =this.valuespile[c];
				}
				
			
			}	
		} else {
			
			for (int j=0; j < this.columns;j++ ){
				for (int c=this.indexpile[j]; c < this.indexpile[j+1];c++ ){
					int i=this.mainelementpile[c];
					//double value=this.valuespile[c];
					fs_data[i][j] =this.valuespile[c];
				}
				
			
			}		
			
			
		}

		
		return fs_data;
	}
	@Override
	public void PrintInfo() {
		System.out.println(" Fixed size Matrix [ " + this.rows + " X " + this.columns + " ] ");
		System.out.println(" Total (non-zero) Elements "  + " = " + (slength));
		
	}
	
	/**
	 * 
	 * @param array : array to print
	 * @param cases : number of cases from the beginning to print
	 */
     public  void Print( int cases){
		
    	
		if (cases <=0){
			cases=this.rows;
		}
		if (cases >100){
			cases=100;
		}
		if (cases >this.rows){
			cases=this.rows;
		}
		// printing
	
	

			for (int i=0; i < cases; i++){
				
				for (int j=this.indexpile[i]; j < this.indexpile[i+1]; j++){
					
					if (this.iscolumnmatrix){
			   System.out.print( "row: " + i +  "  col: " + this.mainelementpile[j] +  "  value: " + this.valuespile[j]);
					} else {
						 System.out.print( "col: " + i +  "  row: " + this.mainelementpile[j] +  "  value: " + this.valuespile[j]);
					}
			   System.out.println("");
				
			}
				
			
			}
			
		
		
	}
     

	@Override
	public void ToFile(String file) {
		output out = new output();
		out.printsmatrix(this, file);
		
	}


}
