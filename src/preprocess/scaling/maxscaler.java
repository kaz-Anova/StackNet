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

package preprocess.scaling;


import java.io.Serializable;

import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
/**
 * @author mariosm
 *<p> class to provide max scaling via dividing each feature (or column) with the maximum value for that feature  
 */
public class maxscaler implements scaler,Serializable{

	/**
	 * serial id
	 */
	private static final long serialVersionUID = -1338386012244012368L;
	/**
	 * Holds whether the scaler is fitted or not
	 */
	private boolean is_itfitted=false;
	/**
	 * max absolute values of all columns
	 */
	public double max_values[];
	/**
	 * 
	 * @return the maximum values as computed from the fit methods
	 */
	public double []get_max_absolute_values(){
		if (max_values==null || max_values.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		return max_values;
	}
	@Override
	public void fit(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		//initialize with negative values
		max_values= new double[data[0].length] ;
		for (int j=0; j< data[0].length; j++){
			max_values[j]=Double.NEGATIVE_INFINITY;
		}
		// find highest absolute values per column
		for (int i=0; i< data.length; i++){
			for (int j=0; j< data[i].length; j++){
				if (Math.abs(data[i][j])>max_values[j]){
					max_values[j]=Math.abs(data[i][j]);
				}
				
			}
			
		}
		//System.out.println(Arrays.toString(max_values));
		
		is_itfitted=true;
	}

	@Override
	public void fit(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		//initialize with negative values
		max_values= new double[data.GetColumnDimension()] ;
		for (int j=0; j< data.GetColumnDimension(); j++){
			max_values[j]=Double.NEGATIVE_INFINITY;
		}
		// find highest absolute values per column
		for (int i=0; i< data.GetRowDimension(); i++){
			for (int j=0; j< data.GetColumnDimension(); j++){
				if (Math.abs(data.GetElement(i, j))>max_values[j]){
					max_values[j]=Math.abs(data.GetElement(i, j));
				}
				
			}
			
		}
		//System.out.println(Arrays.toString(max_values));
		is_itfitted=true;
	}

	@Override
	public void fit(smatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		//initialize with negative values
		max_values= new double[data.GetColumnDimension()] ;
		for (int j=0; j< data.GetColumnDimension(); j++){
			max_values[j]=Double.NEGATIVE_INFINITY;
		}
		if(data.IsSortedByRow()){
		// find highest absolute values per column
		for (int i=0; i< data.mainelementpile.length; i++){
				if (Math.abs(data.valuespile[i])>max_values[data.mainelementpile[i]]){
					max_values[data.mainelementpile[i]]=Math.abs(data.valuespile[i]);
				
			}
			
		}
		} else {
			
			for (int j=0; j< data.GetColumnDimension(); j++){				
				for (int i=data.indexpile[j];i < data.indexpile[j+1]; i++){
				if (Math.abs(data.valuespile[i])>max_values[j]){
					max_values[j]=Math.abs(data.valuespile[i]);
				}
			}
			
		}	
			
			
		}
		is_itfitted=true;
		
	}

	@Override
	public void transformthis(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (max_values==null || max_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data[0].length!=max_values.length){
			throw new DimensionMismatchException(data[0].length,max_values.length);
		}	
		
		for (int i=0; i< data.length; i++){
			for (int j=0; j< data[0].length; j++){
				if (max_values[j]!=0.0){
				data[i][j]/=max_values[j];
				} else {
					data[i][j]=0.0;	
				}
				
			}
			
		}
	}

	@Override
	public void transformthis(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (max_values==null || max_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=max_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),max_values.length);
		}
		int n=0;
		for (int i=0; i< data.GetRowDimension(); i++){
			for (int j=0; j< data.GetColumnDimension(); j++){
				if (max_values[j]!=0.0){
					data.data[n]/=max_values[j];
				} else {
					data.data[n]=0.0;	
				}
			n++;	
			}	
		}
		
	}

	@Override
	public void transformthis(smatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (max_values==null || max_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=max_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),max_values.length);
		}	
		if (data.IsSortedByRow()){
		for (int i=0; i< data.mainelementpile.length; i++){
				if (max_values[data.mainelementpile[i]]!=0.0){
					data.valuespile[i]/=max_values[data.mainelementpile[i]];
				} else {
					data.valuespile[i]=0.0;		
			}	
		 }
		} else {
			for (int j=0; j< data.GetColumnDimension(); j++){				
				for (int i=data.indexpile[j];i < data.indexpile[j+1]; i++){
					if (max_values[j]!=0.0){
						data.valuespile[i]/=max_values[j];
					} else {
						data.valuespile[i]=0.0;		
				}
			}
			
		}	
			
			
			
		}
	}

	@Override
	public double[][] transform(double[][] datas) {
		if (datas==null || datas.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (max_values==null || max_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (datas[0].length!=max_values.length){
			throw new DimensionMismatchException(datas[0].length,max_values.length);
		}	
		double data[][]= new double[datas.length][datas[0].length];
		for (int i=0; i< data.length; i++){
			for (int j=0; j< data[0].length; j++){
				if (max_values[j]!=0.0){
				data[i][j]=datas[i][j]/max_values[j];
				} else {
					data[i][j]=0.0;	
				}
				
			}
			
		}
		return data;
	}

	@Override
	public fsmatrix transform(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (max_values==null || max_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=max_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),max_values.length);
		}	
		
		double new_data []= new double [data.data.length];
		int n=0;
		for (int i=0; i< data.GetRowDimension(); i++){
			for (int j=0; j< data.GetColumnDimension(); j++){
				if (max_values[j]!=0.0){
					new_data[n]=data.data[n]/max_values[j];
				} else {
					new_data[n]=0.0;	
				}
			n++;	
			}	
		}
		
		return new fsmatrix(new_data,data.GetRowDimension(), data.GetColumnDimension() );
	}

	@Override
	public smatrix transform(smatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (max_values==null || max_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=max_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),max_values.length);
		}	
		
		smatrix copythat= (smatrix) data.Copy();
		transformthis(copythat);
		
		
		return copythat;
	}
	@Override
	public double transform(double value, int column) {
		if (max_values[column]!=0.0){
		return value/this.max_values[column];
	} else {
		return 0.0;
	}
	}
	@Override
	public boolean IsFitted() {
		// TODO Auto-generated method stub
		return is_itfitted;
	}
}
