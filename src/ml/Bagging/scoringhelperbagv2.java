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

package ml.Bagging;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.regressor;


/**
 * <p>This class will perform scoring for BaggingRegressor in a Runnable fashion so that it can be used for threading.</p>
 */
public class scoringhelperbagv2 implements Runnable {

	/**
	 * Whether to use scale or not
	 */
	public boolean usescale=true;
    /**
     * seed to use
     */
	public int seed=1;
	  /**
	   * This holds all the nodes in the tree
	   */
	private regressor tree_body ; 
	/**
	 * The object that holds the predictions
	 */
	private fsmatrix predictions;
	/**
	 * The object that holds the modelling data in double form in cases the user chooses this form
	 */
	private double dataset[][];
	/**
	 * The object that holds the modelling data in fsmatrix form cases the user chooses this form
	 */
	private fsmatrix fsdataset;
	/**
	 * The object that holds the modelling data in smatrix form cases the user chooses this form
	 */
	private smatrix sdataset;	
	
	
	
	public scoringhelperbagv2(double data [][], fsmatrix predictions  ,
			regressor tree_body){
		
		if ( tree_body.isfitted()==false){
			throw new IllegalStateException(" Tree is not fitted" );
		}

		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (  data.length!=predictions.GetRowDimension( )){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}

		this.tree_body=tree_body;		
		dataset=data;	
		this.predictions=predictions;
		
	}

	public scoringhelperbagv2(fsmatrix data, fsmatrix predictions,
			
			regressor tree_body){
		
		if ( tree_body.isfitted()==false){
			throw new IllegalStateException(" Tree is not fitted" );
		}

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (  data.GetRowDimension()!=predictions.GetRowDimension( )){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}
		
		this.tree_body=tree_body;
		this.predictions=predictions;
		fsdataset=data;
	}
	

	public scoringhelperbagv2(smatrix data, fsmatrix predictions , 	
			regressor tree_body){
		
		if ( tree_body.isfitted()==false){
			throw new IllegalStateException(" Tree is not fitted" );
		}

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (  data.GetRowDimension()!=predictions.GetRowDimension( )){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}
		


		this.tree_body=tree_body;
		this.predictions=predictions;
		sdataset=data;
		}
	

	private void score() {
		

		//double data
		if ( this.dataset!=null ){
			
			double [][] temp=this.tree_body.predict2d(this.dataset);
			
			for (int i=0; i < temp.length; i++){
				for (int j=0; j < temp[0].length; j++){
					predictions.SetElement(i, j, temp[i][j]);
				}
			}
			
		
		
			
		} else if ( this.fsdataset!=null){
			
			double [][] temp=this.tree_body.predict2d(this.fsdataset);
			for (int i=0; i < temp.length; i++){
				for (int j=0; j < temp[0].length; j++){
					predictions.SetElement(i, j, temp[i][j]);
				}
			}	
			
			
				// return the 1st prediction

		}else if ( this.sdataset!=null ){
			
			double [][] temp=this.tree_body.predict2d(this.sdataset);
			for (int i=0; i < temp.length; i++){
				for (int j=0; j < temp[0].length; j++){
					predictions.SetElement(i, j, temp[i][j]);
				}
			}
			
		} else {
			
			throw new IllegalStateException(" There is an issue with the data provided being null" );
			
		}
		


			// end of SGD

	}

	


	@Override
	public void run() {
		if (tree_body==null ){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		} else {
			this.score();
		}
	}



}
