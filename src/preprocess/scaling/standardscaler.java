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
 *<p> class to provide standard scaling via substracting each feature with its mean and dividing with teh standard deviation
 */
public class standardscaler implements scaler,Serializable{

	/**
	 * serial id
	 */
	private static final long serialVersionUID = -1338386012244012368L;
	/**
	 * Holds whether the scaler is fitted or not
	 */
	private boolean is_itfitted=false;
	/**
	 * mean values of all columns
	 */
	public double mean_values[];
	/**
	 * standard deviation values of all columns
	 */
	public double std_values[];
	/**
	 * 
	 * @return the maximum values as computed from the fit methods
	 */
	public double []get_mean_values(){
		if (mean_values==null || mean_values.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		return mean_values;
	}
	/**
	 * 
	 * @return the std values as computed from the fit methods
	 */
	public double []get_std_values(){
		if (std_values==null || std_values.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		return std_values;
	}
	@Override
	public void fit(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		//initialize with negative values
		mean_values= new double[data[0].length] ;
		std_values= new double[data[0].length] ;

		// find highest absolute values per column
		for (int i=0; i< data.length; i++){
			for (int j=0; j< data[i].length; j++){
				mean_values[j]+=(data[i][j]);
				std_values[j]+=(data[i][j])*(data[i][j]);				
			}
		}
		double length=data.length;
		for (int j=0; j< mean_values.length; j++){
			std_values[j]=  Math.sqrt((std_values[j] - (mean_values[j] * mean_values[j]) / length) / (length )) ;
			mean_values[j]/=length;
		}
		//System.out.println(Arrays.toString(max_values));
		
		is_itfitted=true;
	}

	@Override
	public void fit(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		
		mean_values= new double[data.GetColumnDimension()] ;
		std_values= new double[data.GetColumnDimension()] ;

		// find highest absolute values per column
		for (int i=0; i< data.GetRowDimension(); i++){
			for (int j=0; j< data.GetColumnDimension(); j++){
				mean_values[j]+=(data.GetElement(i, j));
				std_values[j]+=(data.GetElement(i, j))*(data.GetElement(i, j));				
			}
		}
		double length=data.GetRowDimension();
		for (int j=0; j< mean_values.length; j++){
			std_values[j]=  Math.sqrt((std_values[j] - (mean_values[j] * mean_values[j]) / length) / (length )) ;
			mean_values[j]/=length;
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
		mean_values= new double[data.GetColumnDimension()] ;
		std_values= new double[data.GetColumnDimension()] ;
		
		if(data.IsSortedByRow()){
			
		// find highest absolute values per column
		for (int i=0; i< data.mainelementpile.length; i++){
			mean_values[data.mainelementpile[i]]+=data.valuespile[i];
			std_values[data.mainelementpile[i]]+=(data.valuespile[i])*(data.valuespile[i]);	
		}
		} else {
			
			for (int j=0; j< data.GetColumnDimension(); j++){				
				for (int i=data.indexpile[j];i < data.indexpile[j+1]; i++){
					mean_values[j]+=data.valuespile[i];
					std_values[j]+=(data.valuespile[i])*(data.valuespile[i]);		
			}
		}			
		}
		
		double length=data.GetRowDimension();
		for (int j=0; j< mean_values.length; j++){
			std_values[j]=  Math.sqrt((std_values[j] - (mean_values[j] * mean_values[j]) / length) / (length)) ;
			mean_values[j]/=length;
		}
		is_itfitted=true;
		
	}

	@Override
	public void transformthis(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (mean_values==null || mean_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (std_values==null || std_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data[0].length!=mean_values.length){
			throw new DimensionMismatchException(data[0].length,mean_values.length);
		}	
		
		for (int i=0; i< data.length; i++){
			for (int j=0; j< data[0].length; j++){
				if (std_values[j]!=0){
					data[i][j]=(data[i][j]-mean_values[j])/std_values[j];
				}	
			}
			
		}
	}

	@Override
	public void transformthis(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" data object appears to be empty");
		}
		if (mean_values==null || mean_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (std_values==null || std_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=mean_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),mean_values.length);
		}
		int n=0;
		for (int i=0; i< data.GetRowDimension(); i++){
			for (int j=0; j< data.GetColumnDimension(); j++){
				if (std_values[j]!=0){
					data.data[n]=(data.data[n]-mean_values[j])/std_values[j];
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
		if (mean_values==null || mean_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (std_values==null || std_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=mean_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),mean_values.length);
		}
		if (data.IsSortedByRow()){
		for (int i=0; i< data.mainelementpile.length; i++){
			if (std_values[data.mainelementpile[i]]!=0){
				data.valuespile[i]=(data.valuespile[i]-mean_values[data.mainelementpile[i]])/std_values[data.mainelementpile[i]];
			}		
		 }
		} else {
			for (int j=0; j< data.GetColumnDimension(); j++){				
				for (int i=data.indexpile[j];i < data.indexpile[j+1]; i++){
					if (std_values[j]!=0){
						data.valuespile[i]=(data.valuespile[i]-mean_values[j])/std_values[j];
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
		if (mean_values==null || mean_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (std_values==null || std_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (datas[0].length!=mean_values.length){
			throw new DimensionMismatchException(datas[0].length,mean_values.length);
		}	
		double data[][]= new double[datas.length][datas[0].length];
		
		for (int i=0; i< data.length; i++){
			for (int j=0; j< data[0].length; j++){
				if (std_values[j]!=0){
					data[i][j]=(data[i][j]-mean_values[j])/std_values[j];
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
		if (mean_values==null || mean_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (std_values==null || std_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=mean_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),mean_values.length);
		}	
		
		double new_data []= new double [data.data.length];
		int n=0;
		for (int i=0; i< data.GetRowDimension(); i++){
			for (int j=0; j< data.GetColumnDimension(); j++){
				if (std_values[j]!=0){
					new_data[n]=(data.data[n]-mean_values[j])/std_values[j];
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
		if (mean_values==null || mean_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (std_values==null || std_values.length<=0){
			throw new IllegalStateException(" Method needs to be trained first by the fit method");
		}
		if (data.GetColumnDimension()!=mean_values.length){
			throw new DimensionMismatchException(data.GetColumnDimension(),mean_values.length);
		}
		
		smatrix copythat= (smatrix) data.Copy();
		transformthis(copythat);
		
		
		return copythat;
	}
	@Override
	public double transform(double value, int column) {
		
		if (std_values[column]!=0){
			return (value-mean_values[column])/std_values[column];
		
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
