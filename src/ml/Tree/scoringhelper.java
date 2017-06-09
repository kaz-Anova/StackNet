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

package ml.Tree;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.Tree.DecisionTreeRegressor.Node;


/**
 * <p>This class will perform scoring for nodes in a Runnable fashion so that it can be used for threading.</p>
 */
public class scoringhelper implements Runnable {

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
	private Node [] tree_body ; 
	/**
	 * start of the loop in the given_indices array
	 */
	private int start_array=-1;
	/**
	 * end of the loop in the given_indices array
	 */
	private int end_array=-1;
	/**
	 * The object that holds the predictions
	 */
	private double predictions[][];
	/**
	 * The object that holds the predictions (as 1 array)
	 */
	private double single_predictions[];
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
	/**
	 * rounding to apply to predictions
	 */
	private double rounding=3.0;
	
	/**
	 * 
	 * @param data : data to score
	 * @param predictions : placement of predictions
	 * @param st : start of the loop for threads
	 * @param ed : end of the loop for threads
	 * @param tree_body : the nodes
	 * @param rounding : rounding applied to that model.
	 */
	public scoringhelper(double data [][], double predictions [][],  int st, int ed ,
			Node [] tree_body, double rounding){
		
		if (tree_body==null || tree_body.length==0){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		}

		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (predictions==null || predictions.length<=0 || data.length!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}
		if (st<0 || st>=predictions.length ){
			throw new IllegalStateException(" Start indice has to be with in [0, maindata length]" );
		}	
		if (ed<0 || ed>predictions.length ){
			throw new IllegalStateException(" end indice  has to be with in [0, maindata length]" );
		}

		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	

		this.tree_body=tree_body;		
		this.start_array=st;
		this.end_array=ed;
		dataset=data;	
		this.predictions=predictions;
		this.rounding=rounding;
		;
	}
	/**
	 * 
	 * @param data : data to score
	 * @param predictions : placement of predictions
	 * @param st : start of the loop for threads
	 * @param ed : end of the loop for threads
	 * @param tree_body : the nodes
	 * @param rounding : rounding applied to that model.
	 */
	public scoringhelper(double data [][], double predictions [],  int st, int ed,
			Node [] tree_body , double rounding){
		
		if (tree_body==null || tree_body.length==0){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		}

		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (st<0 || st>=predictions.length ){
			throw new IllegalStateException(" Start indice has to be with in [0, maindata length]" );
		}	
		if (ed<0 || ed>predictions.length ){
			throw new IllegalStateException(" end indice  has to be with in [0, maindata length]" );
		}
		if (predictions==null || predictions.length<=0 || data.length!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	

		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		
		this.tree_body=tree_body;		
		this.start_array=st;
		this.end_array=ed;
		dataset=data;	
		this.single_predictions=predictions;
		this.rounding=rounding;
		;
	}
	/**
	 * 
	 * @param data : data to score
	 * @param predictions : placement of predictions
	 * @param st : start of the loop for threads
	 * @param ed : end of the loop for threads
	 * @param tree_body : the nodes
	 * @param rounding : rounding applied to that model.
	 */
	public scoringhelper(fsmatrix data, double predictions [][], int st, int ed,
			
			Node [] tree_body, double rounding){
		
		if (tree_body==null || tree_body.length==0){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		}

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (st<0 || st>=predictions.length ){
			throw new IllegalStateException(" Start indice has to be with in [0, maindata length]" );
		}	
		if (ed<0 || ed>predictions.length ){
			throw new IllegalStateException(" end indice  has to be with in [0, maindata length]" );
		}
		if (predictions==null || predictions.length<=0 || data.GetRowDimension()!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	

		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}
		
		this.tree_body=tree_body;
		this.start_array=st;
		this.end_array=ed;
		this.predictions=predictions;
		this.rounding=rounding;
		;
		fsdataset=data;
	}
	/**
	 * 
	 * @param data : data to score
	 * @param predictions : placement of predictions
	 * @param st : start of the loop for threads
	 * @param ed : end of the loop for threads
	 * @param tree_body : the nodes
	 * @param rounding : rounding applied to that model.
	 */
	public scoringhelper(fsmatrix data, double predictions [], int st, int ed,
			
			DecisionTreeRegressor.Node [] tree_body, double rounding){
		
		if (tree_body==null || tree_body.length==0){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		}

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (st<0 || st>=predictions.length ){
			throw new IllegalStateException(" Start indice has to be with in [0, maindata length]" );
		}	
		if (ed<0 || ed>predictions.length ){
			throw new IllegalStateException(" end indice  has to be with in [0, maindata length]" );
		}
		if (predictions==null || predictions.length<=0 || data.GetRowDimension()!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	

		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		
		this.tree_body=tree_body;		
		this.start_array=st;
		this.end_array=ed;
		this.single_predictions=predictions;
		this.rounding=rounding;
		;
		fsdataset=data;
	}
	/**
	 * 
	 * @param data : data to score
	 * @param predictions : placement of predictions
	 * @param st : start of the loop for threads
	 * @param ed : end of the loop for threads
	 * @param tree_body : the nodes
	 * @param rounding : rounding applied to that model.
	 */
	public scoringhelper(smatrix data, double predictions [][],  int st, int ed,	
			 DecisionTreeRegressor.Node [] tree_body, double rounding){
		
		if (tree_body==null || tree_body.length==0){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		}

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (predictions==null || predictions.length<=0 || data.GetRowDimension()!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	

		if (st<0 || st>=predictions.length ){
			throw new IllegalStateException(" Start indice has to be with in [0, maindata length]" );
		}	
		if (ed<0 || ed>predictions.length ){
			throw new IllegalStateException(" end indice  has to be with in [0, maindata length]" );
		}
		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	


		this.start_array=st;
		this.end_array=ed;
		this.tree_body=tree_body;
		this.rounding=rounding;
		this.predictions=predictions;
		;
		this.rounding=rounding;
		
		sdataset=data;
		}
	/**
	 * 
	 * @param data : data to score
	 * @param predictions : placement of predictions
	 * @param st : start of the loop for threads
	 * @param ed : end of the loop for threads
	 * @param tree_body : the nodes
	 * @param rounding : rounding applied to that model.
	 */
	public scoringhelper(smatrix data, double predictions [],  int st, int ed,	
			DecisionTreeRegressor.Node [] tree_body,double rounding){
		
		if (tree_body==null || tree_body.length==0){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		}

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (predictions==null || predictions.length<=0 || data.GetRowDimension()!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	

		if (st<0 || st>=predictions.length ){
			throw new IllegalStateException(" Start indice has to be with in [0, maindata length]" );
		}	
		if (ed<0 || ed>predictions.length ){
			throw new IllegalStateException(" end indice  has to be with in [0, maindata length]" );
		}
		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	


		this.start_array=st;
		this.end_array=ed;
		this.tree_body=tree_body;
		this.rounding=rounding;
		this.single_predictions=predictions;
		this.rounding=rounding;

		sdataset=data;
		}	


	

	private void score() {
		

		//double data
		if ( this.dataset!=null && this.predictions!=null){
			
			for (int i=this.start_array; i < this.end_array; i++ ){
				//System.out.println(i);
				
				
				int THE_id=0;
				int previous_id=0;
				while (THE_id>=0){
					
						Node new_Node=this.tree_body[THE_id];
						int split_var=new_Node.Variable;
						double cutt_off=new_Node.cutoffval;
						double value=Math.round(dataset[i][split_var]* 10.0 * rounding) / (10.0 * rounding); 
						previous_id=THE_id;
						// left split
						if (value <=cutt_off) {
							THE_id= new_Node.childless;
	
						} else if (value >cutt_off) {
							THE_id= new_Node.childmore;

						} 
		
					}// end of columns loop	
					
						
				predictions[i]=this.tree_body[previous_id].getprediction();
			
				}
				// return the 1st prediction

			
			
		} else if ( this.dataset!=null && this.single_predictions!=null){
			
			for (int i=this.start_array; i < this.end_array; i++ ){
				//System.out.println(i);
				
				
				int THE_id=0;
				int previous_id=0;
				
				while (THE_id>=0){
					
						Node new_Node=this.tree_body[THE_id];
						int split_var=new_Node.Variable;
						double cutt_off=new_Node.cutoffval;
						double value=Math.round(dataset[i][split_var]* 10.0 * rounding) / (10.0 * rounding); 
						previous_id=THE_id;
						
						// left split
						if (value <=cutt_off) {
							THE_id= new_Node.childless;

						} else if (value >cutt_off) {
							THE_id= new_Node.childmore;
	
						}
		
					}// end of columns loop	
	
				single_predictions[i]=this.tree_body[previous_id].predict(0);
			
				}
				// return the 1st prediction

			
			
		} else if ( this.fsdataset!=null && this.predictions!=null){
			
			for (int i=this.start_array; i < this.end_array; i++ ){
				//System.out.println(i);
				
				
				int THE_id=0;
				int previous_id=0;
				while (THE_id>=0){
					
						Node new_Node=this.tree_body[THE_id];
						int split_var=new_Node.Variable;
						double cutt_off=new_Node.cutoffval;
						double value=Math.round(fsdataset.GetElement(i, split_var)* 10.0 * rounding) / (10.0 * rounding); 
						previous_id=THE_id;
						// left split
						if (value <=cutt_off) {
							THE_id= new_Node.childless;
	
						} else if (value >cutt_off) {
							THE_id= new_Node.childmore;

						} 
		
					}// end of columns loop	
					
						
				predictions[i]=this.tree_body[previous_id].getprediction();
			
				}
				// return the 1st prediction

			
			
		} else if ( this.fsdataset!=null && this.single_predictions!=null){
			
			for (int i=this.start_array; i < this.end_array; i++ ){
				//System.out.println(i);
				
				
				int THE_id=0;
				int previous_id=0;
				
				while (THE_id>=0){
					
						Node new_Node=this.tree_body[THE_id];
						int split_var=new_Node.Variable;
						double cutt_off=new_Node.cutoffval;
						
						double value=0.0;
						for (int j=sdataset.indexpile[i];j<sdataset.indexpile[i+1];j++){
							int check_feature=sdataset.mainelementpile[j];
							if (check_feature<split_var){ // we found our feature
								continue;// next row - here the feature has zero value
							}
							else if (check_feature>split_var){ // we found our feature
								break;// next row - here the feature has zero value
							}					
							else { // we found our feature
								value=Math.round(sdataset.valuespile[j] * 10.0 * rounding) / (10.0 * rounding) ;
								break;//found it! no longer need to keep on looping
							}
						}
						previous_id=THE_id;
						
						// left split
						if (value <=cutt_off) {
							THE_id= new_Node.childless;

						} else if (value >cutt_off) {
							THE_id= new_Node.childmore;
	
						}
		
					}// end of columns loop	
	
				single_predictions[i]=this.tree_body[previous_id].predict(0);
			
				}
				// return the 1st prediction

			
			
		}else if ( this.sdataset!=null && this.predictions!=null){
			
			for (int i=this.start_array; i < this.end_array; i++ ){
				int THE_id=0;
				int previous_id=0;
				
				while (THE_id>=0){
					
						Node new_Node=this.tree_body[THE_id];
						int split_var=new_Node.Variable;
						double cutt_off=new_Node.cutoffval;
						
						double value=0.0;
						for (int j=sdataset.indexpile[i];j<sdataset.indexpile[i+1];j++){
							int check_feature=sdataset.mainelementpile[j];
							if (check_feature<split_var){ // we found our feature
								continue;// next row - here the feature has zero value
							}
							else if (check_feature>split_var){ // we found our feature
								break;// next row - here the feature has zero value
							}					
							else { // we found our feature
								value=Math.round(sdataset.valuespile[j] * 10.0 * rounding) / (10.0 * rounding) ;
								break;//found it! no longer need to keep on looping
							}
						}
						
						previous_id=THE_id;
						// left split
						if (value <=cutt_off) {
							THE_id= new_Node.childless;

						} else if (value >cutt_off) {
							THE_id= new_Node.childmore;

						} 
		
					}// end of columns loop	
	
				predictions[i]=this.tree_body[previous_id].getprediction();
			
				}
				// return the 1st prediction
			
			
		}else if ( this.sdataset!=null && this.single_predictions!=null){
			
			for (int i=this.start_array; i < this.end_array; i++ ){
				int THE_id=0;
				int previous_id=0;
				
				while (THE_id>=0){
					
						Node new_Node=this.tree_body[THE_id];
						int split_var=new_Node.Variable;
						double cutt_off=new_Node.cutoffval;
						double value=0.0;
						for (int j=sdataset.indexpile[i];j<sdataset.indexpile[i+1];j++){
							int check_feature=sdataset.mainelementpile[j];
							if (check_feature<split_var){ // we found our feature
								continue;// next row - here the feature has zero value
							}
							else if (check_feature>split_var){ // we found our feature
								break;// next row - here the feature has zero value
							}					
							else { // we found our feature
								value=Math.round(sdataset.valuespile[j] * 10.0 * rounding) / (10.0 * rounding) ;
								break;//found it! no longer need to keep on looping
							}
						}
						previous_id=THE_id;
						// left split
						if (value <=cutt_off) {
							THE_id= new_Node.childless;
		
						} else if (value >cutt_off) {
							THE_id= new_Node.childmore;

						} 
		
					}// end of columns loop	
					
				if (THE_id<0){
					THE_id=0;
				}
						
				single_predictions[i]=this.tree_body[previous_id].predict(0);
			
				}

		} else {
			
			throw new IllegalStateException(" There is an issue with the data provided being null" );
			
		}
		


			// end of SGD

	}

	


	@Override
	public void run() {
		// check which object was chosen to train on
		if (tree_body==null || tree_body.length==0){
			throw new IllegalStateException(" Tree body is empty (e.g depth=0)" );
		} else {
			this.score();
		}
	}



}
