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
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import preprocess.scaling.scaler;
import utilis.XorShift128PlusRandom;
import utilis.map.intint.IntIntMapminus4a;
import utilis.map.intint.StringIntMap4a;
import exceptions.DimensionMismatchException;
import exceptions.LessThanMinimum;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.classifier;

/**
 * 
 * @author mariosm
 *<p> Class to implement generic decision tree classifier. It supports binary splitting of numerical variable based on three
 *possible metrics : </p>
 *<ol>
 *<li> Information Gain (Entropy) </li>
 *<li> Gini Impurity Index </li>
 *<li> AUC (for binary target only) - experimental </li>
 *
 */

public class DecisionTreeClassifier implements estimator,classifier {

	/**
	 * 
	 * <p> This private class will be the main interface for nodes
	 *
	 */
	  public class Node  implements Serializable {
	    	 
	    	 /**
		 * 
		 */
		private static final long serialVersionUID = -8889500386115018792L;
			/*
	    	  *  Element of nodes
	    	  *    a) The number of the variable that is splitted to enter the node
	    	  *    b) the value of that variable
	    	  *    c) the prediction
	    	  *    d) the id of the father
	    	  *    e) the id of the first child
	    	  *    f) the id of the second child
	    	  */
		  

		  /**
		   * The variable that splits the next node
		   */
		  public int Variable;
		  /**
		   * The value that splits the selected variable
		   */
		  public double cutoffval;		  
		  /**
		   * The sum weighted value of the target variable in the node.
		   */
		  public double sum_prediction [];
		  /**
		   * The weighted count
		   */
		  public double weighted_count;
		  /**
		   * Id of the current node
		   */
		  public int id;

		 /**
		  * location for the child that the value of the variable is less equal than the cut-off
		  */
		  public int childless=-1;	  
		  /**
			* location for the child that the value of the variable is more than the cut-off
			*/
		  public int childmore=-1;

		/**
		   * @param predict : The prediction in the node
		   * @param count : the counts in the node
		   * @param ID : the id of the node
		   */
		  public Node(double sum [],double weighted_c, int ID ){	
			  this.sum_prediction=sum;
			  this.weighted_count=weighted_c;
	    	  id=ID;
	         // end of method
	       }
		  /**
		   * 
		   * @param a target column to retrieve prediction for
		   * @return the prediction for the "a" target column.
		   */
		  public double predict(int a){
			  return sum_prediction[a]/this.weighted_count;
		  }
	      /**
	       * 
	       * @param Var : The Variable that splits the node
	       * @param cuttoff : the cut off value for the node.
	       */
		  public void specifyvariable(int Var, double cuttoff){
	    	  Variable= Var;
	    	  cutoffval=cuttoff;
	      }
	      /**
	       * 
	       * @param a : The new node to be populated for the child less equal than the cut off.
	       */
	      public void setchildless( int a){
	    	  childless=a;
	      }
	      /**
	       * 
	       * @param a : The new node to be populated for the child more than the cut off.
	       */
	      public void setchildmore(int b){
	    	  childmore=b;
	      } 
	      /**
	       * 
	       * @return  : The child less than the cut off.
	       */
	      public int  getchildless(){
	    	  return childless;
	      }
	      /**
	       * 
	       * @return  : The  child more than the cut off.
	       */
	      public int getchildmore(){
	    	  return childmore ;
	      } 
	      /**
	       * 
	       * @return  : The prediction for the node.
	       */
	      public double [] getsum(){
	    	  return  this.sum_prediction.clone() ;
	      } 
	      /**
	       * @return  : The prediction for the node.
	       */
	      public double [] getprediction(){
	    	  double k[]= new double [this.sum_prediction.length];
	    	  for (int s=0; s < k.length;s++){
	    		  k[s]=this.predict(s);
	    	  }
	    	  return  k ;
	    	  
	      }	      
	      /**
	       * 
	       * @return a : The cuttoff value for the node.
	       */
	      public double getvalue(){
	    	  return cutoffval;
	      }  

	      /**
	       * 
	       * @return a : The id of the variable that splits for the node
	       */
	      public int getvariable(){
	    	  return Variable;
	      } 
	      /**
	       * 
	       * @return a : The id of the node
	       */
	      public int getid(){
	    	  return id;
	      } 	      

	      // end of private class 
	  }
	  
	  /**
	   * This holds all the nodes in the tree
	   */
	private ArrayList<Node> temp_tree_body ; 
	
	/**
	 * the final version of the nodes in the tree, use as array for faster access (and less memory)
	 */
	private Node [] tree_body;
	
	/**
	 * maximum number of nodes allowed
	 */
	public double max_tree_size=-1;	
	/**
	 * maximum depth of the tree
	 */
	public double max_depth=3;
	  /**
	   * digits of rounding to prevent overfitting
	   */
	 public double rounding=6;
	/**
	 * offset for divisions
	 */
	public double offset=0.0001;
	/**
	 * Minimum gain to allow for a node to split
	 */
	public double gamma=1E-30;
	/**
	 * Minimum weighted sum to split a node
	 */
	public double min_split=2.0;
	/**
	 * Minimum weighted sum to keep a splitted node
	 */
	public double min_leaf=1.0;		
	/**
	 * Proportions of columns (features) to consider
	 */
	public double max_features=1.0;
	/**
	 * Proportions of columns (features) to consider
	 */
	public double feature_subselection=1.0;
	/**
	 * Proportions of best cut offs to consider
	 */
	public double cut_off_subsample=1.0;
	/**
	 * Proportions of best cut offs to consider
	 */
	public double row_subsample=1.0;
	/**
	 * This keeps the sorted indices for each column
	 */
	private int [][] sorted_indices;

	public void set_sorted_indices (int [][] indices){

		this.sorted_indices=indices;
	}
	/**
	 * Holds the rank of the 'zero' (e.g. sparse) elements
	 */
	private int zero_rank_holder [];
	
	public void set_zero_rank (int [] indices){

		this.zero_rank_holder=indices;
	}
	/**
	 * This keeps the sorted indices for each column
	 */
	private int  maximum_ranks;

	public void set_ranked_scores (int indices){

		this.maximum_ranks=indices;
	}

	
	/**
	 * use samples with replacement
	 */
	public boolean bootsrap=false;
	/**
	 * Rows to use
	 */
	private int rows [];
	
	public  Node []  Get_tree(){
		if (this.tree_body==null || this.tree_body.length<=0){
			throw new IllegalStateException(" There is NO tree" );
		}
		return tree_body;
	}
	
	public double [] get_importances(){
		if (this.feature_importances==null || feature_importances.length<=0){
			throw new IllegalStateException(" There no importances (yet)" );
		}
		return feature_importances;
	}
	
	public void set_rows(int rows []){
		if (rows==null || rows.length<=0){
			throw new IllegalStateException(" The row indices are empty" );
		}
		this.rows= rows;
		/*
		this.rows= new int[rows.length];
		int cc=0;
		for (int k:rows ){
		this.rows[cc++]=k;
		}
		*/
	}
	/**
	 * columns to use
	 */
	private int columns [];
	
	public void set_columns(int columns []){
		if (columns==null || columns.length<=0){
			throw new IllegalStateException(" The columns indices are empty" );
		}
		this.columns=columns;
	}
	
	/**
	 * The objective to optimise in split . It may be ENTROPY 
	 *  , GINI or AUC 
	 */
	public String Objective="ENTROPY";	
	/**
	 * quantile value
	 */
	public double tau=0.5;
	/**
	 * threads to use
	 */
	public int threads=1;

	/**
	 * scale the copy the dataset
	 */
	public boolean copy=true;
	
    /**
     * seed to use
     */
	public int seed=1;
	/**
	 * Random number generator to use
	 */
	private Random random;
	/**
	 * weighst to used per row(sample)
	 */
	public double [] weights;
	/**
	 * if true, it prints stuff
	 */
	public boolean verbose=true;
	/**
	 * Target variable in double format
	 */
	public double target[];

	/**
	 * Target variable indices
	 */	
	public int [] fstarget;	
	/**
	 * Target variable in sparse matrix format
	 */	
	public smatrix starget;	
	/**
	 * Hold feature importance for the tree
	 */
	 double feature_importances [];
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
	}

	
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
	 * Default constructor for LinearRegression with no data
	 */
	/**
	 * Number of classes
	 */
	private int n_classes=0;
	
	/**
	 * Name of the unique classes
	 */
	private String classes[];
	/**
	 * Retrieve the number of uniqye classes
	 */

	/**
	 * Target variable in String format
	 */	
	public String Starget[];
	
	public int getnumber_of_classes(){
		return n_classes;
	}
	
	/**
	 * 
	 * @param classes : the classes to put
	 */
	public void setclasses(String[] classes ) {
		if (classes==null || classes.length<=0){
			throw new  IllegalStateException (" No classes are found");
		} else {
		this.classes= classes;
		}
	}
	@Override
	public String[] getclasses() {
		if (classes==null || classes.length<=0){
			throw new  IllegalStateException (" No classes are found, the model needs to be fitted first");
		} else {
		return classes;
		}
	}
	@Override
	public void AddClassnames(String[] names) {
		
		String distinctnames[]=manipulate.distinct.distinct.getstringDistinctset(names);
		if (distinctnames.length<2){
			throw new LessThanMinimum(names.length,2);
		}
		if (distinctnames.length!=names.length){
			throw new  IllegalStateException (" There are duplicate values in the names of the addClasses method, dedupe before adding them");
		}
		classes=new String[names.length];
		for (int j=0; j < names.length; j++){
			classes[j]=names[j];
		}
	}
	public DecisionTreeClassifier(){
	
	}	
	/**
	 * Default constructor for LinearRegression with double data
	 */
	public DecisionTreeClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * Default constructor for LinearRegression with fsmatrix data
	 */
	public DecisionTreeClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for LinearRegression with smatrix data
	 */
	public DecisionTreeClassifier(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
	}

	public void setdata(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}

	public void setdata(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}

	public void setdata(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
		}
		
	@Override
	public void run() {
		// check which object was chosen to train on
		if (dataset!=null){
			this.fit(dataset);
		} else if (fsdataset!=null){
			this.fit(fsdataset);	
		} else if (sdataset!=null){
			this.fit(sdataset);	
		} else {
			throw new IllegalStateException(" No data structure specifed in the constructor" );			
		}	
	}		
	
	
	private int current_id=0;	
	
	
	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8611561535854392960L;
	@Override
	public double[][] predict_proba(double[][] data) {
		 
		/*  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		
		double predictions[][]= new double [data.length][this.n_classes];
		
		double per=data.length/this.threads;
		int batch_size=(int) (per);
		if (batch_size<1){
			batch_size=1;
		}	
		int counter=0;
		int sum=0;
		while(sum<data.length){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=data.length;
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 

			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				if (this.verbose==true){
					System.out.println("Scorint batch from : " + loop_list[n] + " to " + loop_list[n+1]);
					
				}
				
				//Initialise an svc helper model
				scoringhelpercat svc = new scoringhelpercat(data , predictions, loop_list[n], loop_list[n+1],this.tree_body, this.rounding);
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Fitting batch: " + class_passed);
								
							}
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
						class_passed++;
					}
				
					

					count_of_live_threads=0;
				}
			}
		

			// return the 1st prediction
			return predictions;
			
			}

	@Override
	public double[][] predict_proba(fsmatrix data) {
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		
		double per=data.GetRowDimension()/this.threads;
		int batch_size=(int) (per);
		if (batch_size<1){
			batch_size=1;
		}	
		int counter=0;
		int sum=0;
		while(sum<data.GetRowDimension()){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=data.GetRowDimension();
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 

			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				if (this.verbose==true){
					System.out.println("Scorint batch from : " + loop_list[n] + " to " + loop_list[n+1]);
					
				}
				
				//Initialise an svc helper model
				scoringhelpercat svc = new scoringhelpercat(data , predictions, loop_list[n], loop_list[n+1],this.tree_body, this.rounding);
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Fitting batch: " + class_passed);
								
							}
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
						class_passed++;
					}
				
					

					count_of_live_threads=0;
				}
			}
		

			// return the 1st prediction
			return predictions;
			
			}
	
	public fsmatrix predict_probafs(fsmatrix data) {
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}	

		
		fsmatrix predictions= new fsmatrix(data.GetRowDimension(),this.n_classes ) ;
		
		int split_var=0;
		double cutt_off=0.0;
		double value=0.0;	
		int THE_id=0;
		int previous_id=0;
		
		for (int i=0; i < data.GetRowDimension(); i++ ){
			//System.out.println(i);
			
			
		THE_id=0;
		previous_id=0;
		
		while (THE_id>-1){
					
					Node new_Node=this.tree_body[THE_id];
					if (new_Node.childless==-1 &&new_Node.childmore==-1){
						previous_id=THE_id;
						break;
					}
					 split_var=new_Node.Variable;
					 cutt_off=new_Node.cutoffval;
					 value=Math.round(data.GetElement(i,split_var)* 10.0 * rounding) / (10.0 * rounding) ;	
					previous_id=THE_id;
					// left split
					if (value <=cutt_off) {
						THE_id= new_Node.childless;

					} else  {
						THE_id= new_Node.childmore;
					} 
				}// end of columns loop	
				

			double preds []=this.tree_body[previous_id].getprediction();	
			for (int j=0; j< this.n_classes; j++){
				predictions.SetElement(i, j, preds[j]);
			}
			}
			// return the 1st prediction
			return predictions;	
			}
	
	public fsmatrix predict_probafs(smatrix data) {
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}	
		
		fsmatrix predictions= new fsmatrix(data.GetRowDimension(),this.n_classes ) ;
		
		int split_var=0;
		double cutt_off=0.0;
		double value=0.0;	
		int THE_id=0;
		int previous_id=0;
		
		for (int i=0; i < data.GetRowDimension(); i++ ){
			//System.out.println(i);
			
			
		THE_id=0;
		previous_id=0;
		
		while (THE_id>-1){
					
					Node new_Node=this.tree_body[THE_id];
					if (new_Node.childless==-1 &&new_Node.childmore==-1){
						previous_id=THE_id;
						break;
					}
					 split_var=new_Node.Variable;
					 cutt_off=new_Node.cutoffval;
					 value=0.0;
						for (int j=data.indexpile[i];j<data.indexpile[i+1];j++){
							int check_feature=data.mainelementpile[j];
							if (check_feature<split_var){ // we found our feature
								continue;// next row - here the feature has zero value
							}
							else if (check_feature>split_var){ // we found our feature
								break;// next row - here the feature has zero value
							}					
							else { // we found our feature
								value=Math.round(data.valuespile[j] * 10.0 * rounding) / (10.0 * rounding) ;
								break;//found it! no longer need to keep on looping
							}
						}	
					previous_id=THE_id;
					// left split
					if (value <=cutt_off) {
						THE_id= new_Node.childless;

					} else  {
						THE_id= new_Node.childmore;
					} 
				}// end of columns loop	
				

			double preds []=this.tree_body[previous_id].getprediction();	
			for (int j=0; j< this.n_classes; j++){
				predictions.SetElement(i, j, preds[j]);
			}
			}
			// return the 1st prediction
			return predictions;	
			}
	
	
	public fsmatrix predict_probafs(double data[][]) {
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
	
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}

		
		fsmatrix predictions= new fsmatrix(data.length,this.n_classes ) ;
		
		int split_var=0;
		double cutt_off=0.0;
		double value=0.0;	
		int THE_id=0;
		int previous_id=0;
		
		for (int i=0; i < data.length; i++ ){
			//System.out.println(i);	
			
		THE_id=0;
		previous_id=0;
		
		while (THE_id>-1){
					
					Node new_Node=this.tree_body[THE_id];
					if (new_Node.childless==-1 &&new_Node.childmore==-1){
						previous_id=THE_id;
						break;
					}
					 split_var=new_Node.Variable;
					 cutt_off=new_Node.cutoffval;
					 value= Math.round( data[i][split_var]* 10.0 * rounding) / (10.0 * rounding) ;	
					previous_id=THE_id;
					// left split
					if (value <=cutt_off) {
						THE_id= new_Node.childless;

					} else  {
						THE_id= new_Node.childmore;
					} 
				}// end of columns loop	
				

			double preds []=this.tree_body[previous_id].getprediction();	
			for (int j=0; j< this.n_classes; j++){
				predictions.SetElement(i, j, preds[j]);
			}
			}	
			// return the 1st prediction
			return predictions;	
			}	

	@Override
	public double[][] predict_proba(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}
		double predictions[][]= new double [data.GetRowDimension()][this.n_classes];
		
		double per=data.GetRowDimension()/this.threads;
		int batch_size=(int) (per);
		if (batch_size<1){
			batch_size=1;
		}	
		int counter=0;
		int sum=0;
		while(sum<data.GetRowDimension()){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=data.GetRowDimension();
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 

			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				if (this.verbose==true){
					System.out.println("Scorint batch from : " + loop_list[n] + " to " + loop_list[n+1]);
					
				}
				
				//Initialise an svc helper model
				scoringhelpercat svc = new scoringhelpercat(data , predictions, loop_list[n], loop_list[n+1],this.tree_body, this.rounding);
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Fitting batch: " + class_passed);
								
							}
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
						class_passed++;
					}
				
					

					count_of_live_threads=0;
				}
			}
		

			// return the 1st prediction
			return predictions;
	}

	@Override
	public double[] predict_probaRow(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	

		int THE_id=0;

		while (THE_id>=0){
			
			Node new_Node=this.tree_body[THE_id];
			int split_var=new_Node.Variable;
			double cutt_off=new_Node.cutoffval;
			double value=Math.round( data[split_var]* 10.0 * rounding) / (10.0 * rounding) ;

			// left split
			if (value <=cutt_off) {
				int less_child= new_Node.childless;
				if (less_child>0){
					THE_id=less_child;
				}else {
					break;
				}
			} else if (value >cutt_off) {
				int more_child= new_Node.childmore;
				if (more_child>0){
					THE_id=more_child;
				}else {
					break;
				}
			} else {
				break;
			}

			}// end of columns loop	
			
		if (THE_id<0){
			THE_id=0;
		}
				

		// return the 1st prediction
		return this.tree_body[THE_id].getprediction();
			}


	@Override
	public double[] predict_probaRow(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	


		int THE_id=0;

		while (THE_id>=0){
			
			Node new_Node=this.tree_body[THE_id];
			int split_var=new_Node.Variable;
			double cutt_off=new_Node.cutoffval;
			double value=  Math.round(data.GetElement(rows, split_var)* 10.0 * rounding) / (10.0 * rounding) ;

			// left split
			if (value <=cutt_off) {
				int less_child= new_Node.childless;
				if (less_child>0){
					THE_id=less_child;
				}else {
					break;
				}
			} else if (value >cutt_off) {
				int more_child= new_Node.childmore;
				if (more_child>0){
					THE_id=more_child;
				}else {
					break;
				}
			} else {
				break;
			}

			}// end of columns loop	
			
		if (THE_id<0){
			THE_id=0;
		}
				

		// return the 1st prediction
		return this.tree_body[THE_id].getprediction();
			
			
	}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}
		
		if (data.indexer==null){
			data.buildmap();
		}


		int THE_id=0;
		IntIntMapminus4a column_Values=new IntIntMapminus4a(data.GetColumnDimension(), 0.5F);
		for (int b=start; b < end ;b++ ){
			column_Values.put(data.mainelementpile[b], b);
		}
		
		while (THE_id>=0){
			
			Node new_Node=this.tree_body[THE_id];
			int split_var=new_Node.Variable;
			double cutt_off=new_Node.cutoffval;
			double value=0.0;
			Integer existed_var_index_in_sparse_array =column_Values.get(split_var);
			if (existed_var_index_in_sparse_array!=-1){
				 value= Math.round(data.valuespile[existed_var_index_in_sparse_array]* 10.0 * rounding) / (10.0 * rounding)  ;
			}
			// left split
			if (value <=cutt_off) {
				int less_child= new_Node.childless;
				if (less_child>0){
					THE_id=less_child;
				}else {
					break;
				}
			} else if (value >cutt_off) {
				int more_child= new_Node.childmore;
				if (more_child>0){
					THE_id=more_child;
				}else {
					break;
				}
			} else {
				break;
			}

			}// end of columns loop	
			
		if (THE_id<0){
			THE_id=0;
		}
				

		// return the 1st prediction
		return this.tree_body[THE_id].getprediction();
			}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		
		double predictions[]= new double [data.GetRowDimension()];
		
		double per=data.GetRowDimension()/this.threads;
		int batch_size=(int) (per);
		if (batch_size<1){
			batch_size=1;
		}	
		int counter=0;
		int sum=0;
		while(sum<data.GetRowDimension()){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=data.GetRowDimension();
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 

			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				if (this.verbose==true){
					System.out.println("Scorint batch from : " + loop_list[n] + " to " + loop_list[n+1]);
					
				}
				
				//Initialise an svc helper model
				scoringhelpercat svc = new scoringhelpercat(data , predictions, loop_list[n], loop_list[n+1],this.tree_body, this.classes, this.rounding);
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Fitting batch: " + class_passed);
								
							}
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
						class_passed++;
					}
				
					

					count_of_live_threads=0;
				}
			}
		

			// return the 1st prediction
			return predictions;
			
			}
			

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		double predictions[]= new double [data.GetRowDimension()];
		
		double per=data.GetRowDimension()/this.threads;
		int batch_size=(int) (per);
		if (batch_size<1){
			batch_size=1;
		}	
		int counter=0;
		int sum=0;
		while(sum<data.GetRowDimension()){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=data.GetRowDimension();
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 

			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				if (this.verbose==true){
					System.out.println("Scorint batch from : " + loop_list[n] + " to " + loop_list[n+1]);
					
				}
				
				//Initialise an svc helper model
				scoringhelpercat svc = new scoringhelpercat(data , predictions, loop_list[n], loop_list[n+1],this.tree_body, this.classes, this.rounding);
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Fitting batch: " + class_passed);
								
							}
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
						class_passed++;
					}
				
					

					count_of_live_threads=0;
				}
			}
		

			// return the 1st prediction
			return predictions;
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 ||this.tree_body==null || this.tree_body.length<=0 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		
		double predictions[]= new double [data.length];
		
		double per=data.length/this.threads;
		int batch_size=(int) (per);
		if (batch_size<1){
			batch_size=1;
		}	
		int counter=0;
		int sum=0;
		while(sum<data.length){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=data.length;
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 

			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				if (this.verbose==true){
					System.out.println("Scorint batch from : " + loop_list[n] + " to " + loop_list[n+1]);
					
				}
				
				//Initialise an svc helper model
				scoringhelpercat svc = new scoringhelpercat(data , predictions, loop_list[n], loop_list[n+1],this.tree_body, this.classes, this.rounding);
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Fitting batch: " + class_passed);
								
							}
							thread_array[s].join();
						} catch (InterruptedException e) {
						   System.out.println(e.getMessage());
						   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
						}
						class_passed++;
					}
				
					

					count_of_live_threads=0;
				}
			}
		

			// return the 1st prediction
			return predictions;
			
			}

	@Override
	public double predict_Row(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	


		int THE_id=0;

		while (THE_id>=0){
			
			Node new_Node=this.tree_body[THE_id];
			int split_var=new_Node.Variable;
			double cutt_off=new_Node.cutoffval;
			double value=Math.round( data[split_var]* 10.0 * rounding) / (10.0 * rounding);

			// left split
			if (value <=cutt_off) {
				int less_child= new_Node.childless;
				if (less_child>0){
					THE_id=less_child;
				}else {
					break;
				}
			} else if (value >cutt_off) {
				int more_child= new_Node.childmore;
				if (more_child>0){
					THE_id=more_child;
				}else {
					break;
				}
			} else {
				break;
			}

			}// end of columns loop	
			
		if (THE_id<0){
			THE_id=0;
		}
				

		// return the 1st prediction
	  double temp []=  this.tree_body[THE_id].getprediction();
  	  int maxi=0;
  	  double max=temp[0];
  	  for (int k=1; k<n_classes; k++) {
  		 if (temp[k]>max){
  			 max=temp[k];
  			 maxi=k;	 
  		 }
  	  }
  	  try{
  		return Double.parseDouble(classes[maxi]);
  	  } catch (Exception e){
  		return maxi;
  	  }

		
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	


		int THE_id=0;

		while (THE_id>=0){
			
			Node new_Node=this.tree_body[THE_id];
			int split_var=new_Node.Variable;
			double cutt_off=new_Node.cutoffval;
			double value=Math.round( data.GetElement(rows, split_var)* 10.0 * rounding) / (10.0 * rounding);

			// left split
			if (value <=cutt_off) {
				int less_child= new_Node.childless;
				if (less_child>0){
					THE_id=less_child;
				}else {
					break;
				}
			} else if (value >cutt_off) {
				int more_child= new_Node.childmore;
				if (more_child>0){
					THE_id=more_child;
				}else {
					break;
				}
			} else {
				break;
			}

			}// end of columns loop	
			
		if (THE_id<0){
			THE_id=0;
		}
				

		  double temp []=  this.tree_body[THE_id].getprediction();
	  	  int maxi=0;
	  	  double max=temp[0];
	  	  for (int k=1; k<n_classes; k++) {
	  		 if (temp[k]>max){
	  			 max=temp[k];
	  			 maxi=k;	 
	  		 }
	  	  }
	  	  try{
	  		return Double.parseDouble(classes[maxi]);
	  	  } catch (Exception e){
	  		return maxi;
	  	  }

			}
			
	

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.tree_body==null || this.tree_body.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}		
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	

		if (data.indexer==null){
			data.buildmap();
		}


		int THE_id=0;
		IntIntMapminus4a column_Values=new IntIntMapminus4a(data.GetColumnDimension(), 0.5F);
		for (int b=start; b < end ;b++ ){
			column_Values.put(data.mainelementpile[b], b);
		}
		

		while (THE_id>=0){
			
			Node new_Node=this.tree_body[THE_id];
			int split_var=new_Node.Variable;
			double cutt_off=new_Node.cutoffval;
			double value=0.0;
			Integer existed_var_index_in_sparse_array =column_Values.get(split_var);
			if (existed_var_index_in_sparse_array!=-1){
				 value=Math.round( data.valuespile[existed_var_index_in_sparse_array]* 10.0 * rounding) / (10.0 * rounding) ;
			}
			// left split
			if (value <=cutt_off) {
				int less_child= new_Node.childless;
				if (less_child>0){
					THE_id=less_child;
				}else {
					break;
				}
			} else if (value >cutt_off) {
				int more_child= new_Node.childmore;
				if (more_child>0){
					THE_id=more_child;
				}else {
					break;
				}
			} else {
				break;
			}

			}// end of columns loop	
			
		if (THE_id<0){
			THE_id=0;
		}
				
		  double temp []=  this.tree_body[THE_id].getprediction();
	  	  int maxi=0;
	  	  double max=temp[0];
	  	  for (int k=1; k<n_classes; k++) {
	  		 if (temp[k]>max){
	  			 max=temp[k];
	  			 maxi=k;	 
	  		 }
	  	  }
	  	  try{
	  		return Double.parseDouble(classes[maxi]);
	  	  } catch (Exception e){
	  		return maxi;
	  	  }

			}

	
	
	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		dataset=data;
		
		if (max_tree_size<=0){
			max_tree_size=Double.MAX_VALUE;
		}
	
		if (min_split<2){
			min_split=2;
		}
		if (min_leaf<1){
			min_leaf=1;
		}	
		if (max_features<=0){
			max_features=1;
		}			
		if (feature_subselection<=0){
			feature_subselection=1;
		}		
		if (cut_off_subsample<=0){
			cut_off_subsample=1;
		}		
		
		if (row_subsample<=0){
			row_subsample=1;
		}	
		if (this.offset<=0){
			this.offset=0.0000001;
		}

		if ( !this.Objective.equals("ENTROPY")&& !this.Objective.equals("GINI") && !this.Objective.equals("AUC"))  {
			throw new IllegalStateException("the objective has to be one of ENTROPY,GINI or AUC" );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.length) && (Starget==null || Starget.length!=data.length)&& this.fstarget==null ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else if (target!=null  ){
				
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
			}
			double uniquevalues[]= new double[has.size()];
			
			int k=0;
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		    	uniquevalues[k]= it.next();
		    	k++;
		    	}
		    // sort values
		    Arrays.sort(uniquevalues);
		    
		    classes= new String[uniquevalues.length];
		    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
		    int index=0;
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    	mapper.put(classes[j], index);
		    	index++;
		    }
		    fstarget=new int[target.length];
		    for (int i=0; i < fstarget.length; i++){
		    	fstarget[i]=mapper.get(target[i] + "");
		    }
		    

			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);
			    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
			    int index=0;
			    for (int j=0; j < classes.length; j++){
			    	mapper.put(classes[j], index);
			    	index++;
			    }
			    
			    fstarget=new int[Starget.length];
			    for (int i=0; i < fstarget.length; i++){
			    	fstarget[i]=mapper.get(Starget[i]);
			    }    
			    
			}
			
			
			
		
		if (weights==null) {
			/*
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
			*/
		} else {
			if (weights.length!=data.length){
				throw new DimensionMismatchException(weights.length,data.length);
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			for (int i=0; i < weights.length; i++){
				weights[i]*= weights.length;
			}
		}
		this.n_classes=classes.length;			
		if (this.Objective.equals("AUC") && (this.n_classes!=2)){
			throw new IllegalStateException("The 'AUC' Metric can only be used when n_classes=2" );	
		}
		//hard copy
		if (copy){
			data= manipulate.copies.copies.Copy( data);
		}
		// Initialise randomizer
		fsdataset=new fsmatrix(data);
		
		
		this.random = new XorShift128PlusRandom(this.seed);

		
		

		/**
		 *  generate rows required by the algorithm
		 */
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		columndimension=data[0].length;
		feature_importances= new double [columndimension];
		//initialise beta and constant
		 int subsampleinteger = get_random_integer(this.row_subsample);
		 //System.out.println(" subsampleinteger " + subsampleinteger);
		//check if initial rows are given
		 
			if (rows==null){
				ArrayList<Integer> subset_of_rows= new ArrayList<Integer>();
				
				for (int i=0; i <data.length; i++ ){
					if (random.nextInt()<subsampleinteger){
						subset_of_rows.add(i);
					}
				}			
				if (subset_of_rows.size()<=0){
					subset_of_rows.add(random.nextInt(data.length));	
				}
				rows= new int [subset_of_rows.size()];
				for (int i=0; i <rows.length; i++ ){
					rows[i]=subset_of_rows.get(i);
				}
				subset_of_rows=null;
			}else if (this.row_subsample!=1.0){
				
				ArrayList<Integer> subset_of_rows= new ArrayList<Integer>();
				
				for (int i : rows ){
					if (random.nextInt()<subsampleinteger){
						subset_of_rows.add(i);
					}
				}			
				if (subset_of_rows.size()<=0){
					subset_of_rows.add(rows[random.nextInt(rows.length)]);	
				}
				rows= new int [subset_of_rows.size()];
				for (int i=0; i <rows.length; i++ ){
					rows[i]=subset_of_rows.get(i);
				}
				subset_of_rows=null;
				
			}
	
			//check if initial cols are given
			if (columns==null){
				subsampleinteger = get_random_integer(this.feature_subselection);
				
				ArrayList<Integer> subset_of_cols= new ArrayList<Integer>();
				
				for (int i=0; i <data[0].length; i++ ){
					if (random.nextInt()<subsampleinteger){
						subset_of_cols.add(i);
					}
				}			
				if (subset_of_cols.size()<=0){
					subset_of_cols.add(random.nextInt(data[0].length));	
				}
				columns= new int [subset_of_cols.size()];
				for (int i=0; i <columns.length; i++ ){
					columns[i]=subset_of_cols.get(i);
				}
				subset_of_cols=null;
			}
			
			
			if (this.sorted_indices==null){
					this.sorted_indices=new int  [3] [];
					this.maximum_ranks= 0;
					this.zero_rank_holder= new int [this.columndimension];

				// create indices

					sortcolumnsnomap sorty= new sortcolumnsnomap (this.fsdataset,
																this.rows,
																this.sorted_indices,
																this.zero_rank_holder,
																this.rounding );
					sorty.verbose=this.verbose;
					sorty.merge_thresold=this.gamma;
					sorty.target_vales=fstarget;					
					sorty.run();
					this.maximum_ranks=	sorty.getmaxrank();
			
		}
		double current_weighted_count=0.0;
		double current_sum_value [] =new double [this.n_classes];
	
		if (this.bootsrap){
			int newrows[]= new int[this.rows.length];
			for (int i=0; i < newrows.length; i++){
				newrows[i]=this.rows[random.nextInt(this.rows.length)];
			}
			this.rows=newrows;
		}	
		
		if (this.weights!=null){
		for (int i :rows  ){
			current_weighted_count+=weights[i];
		    current_sum_value[fstarget[i]]+=this.weights[i];
			
		}
		} else {
			for (int i :rows){
				current_weighted_count+=1.0;
				current_sum_value[fstarget[i]]+=1;
				
			}			
		}

		if (this.verbose){
			System.out.println(" weighted count Value(s) of the zero level with counts " + current_weighted_count);
			System.out.println(" weighted sum Value(s): " + Arrays.toString(current_sum_value));			
		}
		// Initialise the tree structure
		temp_tree_body = new ArrayList< Node>(10000);

		Node initial_node = new Node(current_sum_value,current_weighted_count, current_id);
		
		temp_tree_body.add(initial_node);
		current_id+=1;
		int current_level=1;
		
		// Start expanding the trees
		expand_node(initial_node , this.rows, current_sum_value, current_weighted_count, current_level ) ;
		
		rows=null;
		columns=null;
		double sum_importances=get_sum(this.feature_importances);
		for (int i=0; i <feature_importances.length; i++ ){
			feature_importances[i]/=sum_importances;
			
		}
		
		this.rows=null;
		this.weights=null;
		columns=null;
		this.sorted_indices=null;
		this.zero_rank_holder=null;
		this.fstarget=null;
		data=null;
		target=null;
		this.starget=null;
		fsdataset=null;
		this.sdataset=null;	
		tree_body = new Node [temp_tree_body.size()];
		
 
		for (   Node a: this.temp_tree_body) {
			   int key=a.id;
	           tree_body[key]=a;
		}
		temp_tree_body=null;
		
		System.gc();
		
	}
	@Override
	public void fit(fsmatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		fsdataset=data;
		if (this.offset<=0){
			this.offset=0.0000001;
		}
		if (max_tree_size<=0){
			max_tree_size=Double.MAX_VALUE;
		}
				
		if (min_split<2){
			min_split=2;
		}
		if (min_leaf<1){
			min_leaf=1;
		}	
		if (max_features<=0){
			max_features=1;
		}			
		if (feature_subselection<=0){
			feature_subselection=1;
		}		
		if (cut_off_subsample<=0){
			cut_off_subsample=1;
		}	
		
		if (row_subsample<=0){
			row_subsample=1;
		}				
		if ( !this.Objective.equals("ENTROPY")&& !this.Objective.equals("GINI") && !this.Objective.equals("AUC"))  {
			throw new IllegalStateException("the objective has to be one of ENTROPY,GINI or AUC" );	
		}			

		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		// make sensible checks on the target data
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) && this.fstarget==null){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else if (target!=null  ){
				
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
			}
			double uniquevalues[]= new double[has.size()];
			
			int k=0;
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		    	uniquevalues[k]= it.next();
		    	k++;
		    	}
		    // sort values
		    Arrays.sort(uniquevalues);
		    
		    classes= new String[uniquevalues.length];
		    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
		    int index=0;
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    	mapper.put(classes[j], index);
		    	index++;
		    }
		    fstarget=new int[target.length];
		    for (int i=0; i < fstarget.length; i++){
		    	fstarget[i]=mapper.get(target[i] + "");
		    }
		    

			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);
			    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
			    int index=0;
			    for (int j=0; j < classes.length; j++){
			    	mapper.put(classes[j], index);
			    	index++;
			    }
			    
			    fstarget=new int[Starget.length];
			    for (int i=0; i < fstarget.length; i++){
			    	fstarget[i]=mapper.get(Starget[i]);
			    }    
			    
			
			
			
			
		}
		if (weights==null) {
			/*
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
			*/
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			for (int i=0; i < weights.length; i++){
				weights[i]*= weights.length;
			}
		}

		this.n_classes=classes.length;			
		if (this.Objective.equals("AUC") && (this.n_classes!=2)){
			throw new IllegalStateException("The 'AUC' Metric can only be used when n_classes=2" );	
		}
		//hard copy
		if (copy){
			data= (fsmatrix) (data.Copy());
		}
		// Initialise randomizer

		
		this.random = new XorShift128PlusRandom(this.seed);


		/**
		 *  generate rows required by the algorithm
		 */
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		columndimension=data.GetColumnDimension();
		feature_importances= new double [columndimension];
		//initialise beta and constant
		 int subsampleinteger = get_random_integer(this.row_subsample);
		 //System.out.println(" subsampleinteger " + subsampleinteger);
		//check if initial rows are given
			if (rows==null){
				
				ArrayList<Integer> subset_of_rows= new ArrayList<Integer>();
				
				for (int i=0; i <data.GetRowDimension(); i++ ){
					if (random.nextInt()<subsampleinteger){
						subset_of_rows.add(i);
					}
				}			
				if (subset_of_rows.size()<=0){
					subset_of_rows.add(random.nextInt(data.GetRowDimension()));	
				}
				rows= new int [subset_of_rows.size()];
				for (int i=0; i <rows.length; i++ ){
					rows[i]=subset_of_rows.get(i);
				}
				subset_of_rows=null;
			} else if (this.row_subsample!=1.0){
				
				ArrayList<Integer> subset_of_rows= new ArrayList<Integer>();
				
				for (int i : rows ){
					if (random.nextInt()<subsampleinteger){
						subset_of_rows.add(i);
					}
				}			
				if (subset_of_rows.size()<=0){
					subset_of_rows.add(rows[random.nextInt(rows.length)]);	
				}
				rows= new int [subset_of_rows.size()];
				for (int i=0; i <rows.length; i++ ){
					rows[i]=subset_of_rows.get(i);
				}
				subset_of_rows=null;
				
			}

			//check if initial cols are given
			if (columns==null){
				subsampleinteger = get_random_integer(this.feature_subselection);
				
				ArrayList<Integer> subset_of_cols= new ArrayList<Integer>();
				
				for (int i=0; i <data.GetColumnDimension(); i++ ){
					if (random.nextInt()<subsampleinteger){
						subset_of_cols.add(i);
					}
				}			
				if (subset_of_cols.size()<=0){
					subset_of_cols.add(random.nextInt(data.GetColumnDimension()));	
				}
				columns= new int [subset_of_cols.size()];
				for (int i=0; i <columns.length; i++ ){
					columns[i]=subset_of_cols.get(i);
				}
				subset_of_cols=null;
			}
		
		
		if (this.sorted_indices==null){
				this.sorted_indices=new int  [3] [];
				this.maximum_ranks= 0;
				this.zero_rank_holder= new int [this.columndimension];

			// create indices

				sortcolumnsnomap sorty= new sortcolumnsnomap (this.fsdataset,
															this.rows,
															this.sorted_indices,
															this.zero_rank_holder,
															this.rounding );
				sorty.verbose=this.verbose;
				sorty.merge_thresold=this.gamma;
				sorty.target_vales=fstarget;				
				
				sorty.run();
				this.maximum_ranks=	sorty.getmaxrank();
		}
		
		if (this.bootsrap){
			int newrows[]= new int[this.rows.length];
			for (int i=0; i < newrows.length; i++){
				newrows[i]=this.rows[random.nextInt(this.rows.length)];
			}
			this.rows=newrows;
		}		
		double current_weighted_count=0.0;
		double current_sum_value [] =new double [this.n_classes];
		
		if (this.weights!=null){
		for (int i :rows  ){
			current_weighted_count+=weights[i];
			current_sum_value[fstarget[i]]+=this.weights[i];
			
		}
		} else {
			for (int i :rows){
				current_weighted_count+=1.0;
				current_sum_value[fstarget[i]]+=1.0;
				
			}			
		}
		
		if (this.verbose){
			System.out.println(" weighted count Value(s) of the zero level with counts " + current_weighted_count);
			System.out.println(" weighted sum Value(s): " + Arrays.toString(current_sum_value));			
		}
		// Initialise the tree structure
		temp_tree_body = new ArrayList<Node>(10000);

		Node initial_node = new Node(current_sum_value,current_weighted_count, current_id);
		
		temp_tree_body.add( initial_node);
		current_id+=1;
		int current_level=1;
		
		// Start expanding the trees
		expand_node(initial_node ,  this.rows, current_sum_value, current_weighted_count, current_level ) ;
		

		double sum_importances=get_sum(this.feature_importances);
		for (int i=0; i <feature_importances.length; i++ ){
			feature_importances[i]/=sum_importances;
			
		}
		
		this.rows=null;
		this.weights=null;
		columns=null;
		this.sorted_indices=null;
		this.zero_rank_holder=null;
		this.fstarget=null;
		data=null;
		target=null;
		fsdataset=null;
		this.starget=null;	
		this.sdataset=null;
		
		tree_body = new Node [temp_tree_body.size()];
		
		 
		for (   Node a: this.temp_tree_body) {
			   int key=a.id;
	           tree_body[key]=a;
		}
		System.gc();

		
	}
	
	@Override
	public void fit(smatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		sdataset=data;
		
		if (max_tree_size<=0){
			max_tree_size=Double.MAX_VALUE;
		}

		if (this.offset<=0){
			this.offset=0.0000001;
		}	
		
		if (min_split<2){
			min_split=2;
		}
		if (min_leaf<1){
			min_leaf=1;
		}	
		if (max_features<=0){
			max_features=1;
		}			
		if (feature_subselection<=0){
			feature_subselection=1;
		}		
		if (cut_off_subsample<=0){
			cut_off_subsample=1;
		}
		
		if (row_subsample<=0){
			row_subsample=1;
		}				

		if ( !this.Objective.equals("ENTROPY")&& !this.Objective.equals("GINI") && !this.Objective.equals("AUC"))  {
			throw new IllegalStateException("the objective has to be one of ENTROPY,GINI or AUC" );	
		}		

		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
			if (this.threads<1){
				this.threads=1;
			}
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) && this.fstarget==null){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else if (target!=null  ){
				
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
			}
			double uniquevalues[]= new double[has.size()];
			
			int k=0;
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		    	uniquevalues[k]= it.next();
		    	k++;
		    	}
		    // sort values
		    Arrays.sort(uniquevalues);
		    
		    classes= new String[uniquevalues.length];
		    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
		    int index=0;
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    	mapper.put(classes[j], index);
		    	index++;
		    }
		    fstarget=new int[target.length];
		    for (int i=0; i < fstarget.length; i++){
		    	fstarget[i]=mapper.get(target[i] + "");
		    }
		    

			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);
			    StringIntMap4a mapper = new StringIntMap4a(classes.length,0.5F);
			    int index=0;
			    for (int j=0; j < classes.length; j++){
			    	mapper.put(classes[j], index);
			    	index++;
			    }
			    
			    fstarget=new int[Starget.length];
			    for (int i=0; i < fstarget.length; i++){
			    	fstarget[i]=mapper.get(Starget[i]);
			    }    
			    
		}
		if (weights==null) {
			/*
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
			*/
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			for (int i=0; i < weights.length; i++){
				weights[i]*= weights.length;
			}
		}
		this.n_classes=classes.length;			
		if (this.Objective.equals("AUC") && (this.n_classes!=2)){
			throw new IllegalStateException("The 'AUC' Metric can only be used when n_classes=2" );	
		}
		//hard copy
		if (copy){
			data= (smatrix) (data.Copy());
		}
		
		
		// Initialise randomizer

		
		this.random = new XorShift128PlusRandom(this.seed);
		// trim data
		
		if (this.sorted_indices==null){
		sdataset.trim();
	
		if (!this.sdataset.IsSortedByRow()){
			this.sdataset.convert_type();
			//System.out.println("built sort");
			}
		
		}
		/**
		 *  generate rows required by the algorithm
		 */
		/*
		if (data.optional_rows==null){
			data.void_update_indice();
		}
		*/
		columndimension=this.sdataset.GetColumnDimension();
		feature_importances= new double [columndimension];
		//initialise beta and constant
		 int subsampleinteger = get_random_integer(this.row_subsample);
		 //System.out.println(" subsampleinteger " + subsampleinteger);
		//check if initial rows are given
			if (rows==null){
				ArrayList<Integer> subset_of_rows= new ArrayList<Integer>();
				
				for (int i=0; i <data.GetRowDimension(); i++ ){
					if (random.nextInt()<subsampleinteger){
						subset_of_rows.add(i);
					}
				}			
				if (subset_of_rows.size()<=0){
					subset_of_rows.add(random.nextInt(data.GetRowDimension()));	
				}
				rows= new int [subset_of_rows.size()];
				for (int i=0; i <rows.length; i++ ){
					rows[i]=subset_of_rows.get(i);
				}
				subset_of_rows=null;
			}else if (this.row_subsample!=1.0){
				
				ArrayList<Integer> subset_of_rows= new ArrayList<Integer>();
				
				for (int i : rows ){
					if (random.nextInt()<subsampleinteger){
						subset_of_rows.add(i);
					}
				}			
				if (subset_of_rows.size()<=0){
					subset_of_rows.add(rows[random.nextInt(rows.length)]);	
				}
				rows= new int [subset_of_rows.size()];
				for (int i=0; i <rows.length; i++ ){
					rows[i]=subset_of_rows.get(i);
				}
				subset_of_rows=null;
				
			}
	
			//check if initial cols are given
			if (columns==null){
				
				subsampleinteger = get_random_integer(this.feature_subselection);
				
				ArrayList<Integer> subset_of_cols= new ArrayList<Integer>();
				
				for (int i=0; i <data.GetColumnDimension(); i++ ){
					if (random.nextInt()<subsampleinteger){
						subset_of_cols.add(i);
					}
				}			
				if (subset_of_cols.size()<=0){
					subset_of_cols.add(random.nextInt(data.GetColumnDimension()));	
				}
				columns= new int [subset_of_cols.size()];
				for (int i=0; i <columns.length; i++ ){
					columns[i]=subset_of_cols.get(i);
				}
				subset_of_cols=null;
			}
		
		if (this.sorted_indices==null){
			this.sorted_indices=new int  [3] [];
			this.maximum_ranks= 0;
			this.zero_rank_holder= new int [this.columndimension];

		// create indices

			sortcolumnsnomap sorty= new sortcolumnsnomap (this.sdataset,
														this.rows,
														this.sorted_indices,
														this.zero_rank_holder,
														this.rounding );
			sorty.verbose=this.verbose;
			sorty.merge_thresold=this.gamma;
			sorty.target_vales=fstarget;			
			
			sorty.run();
			this.maximum_ranks=	sorty.getmaxrank();
		}
		
		if (this.bootsrap){
			int newrows[]= new int[this.rows.length];
			for (int i=0; i < newrows.length; i++){
				newrows[i]=this.rows[random.nextInt(this.rows.length)];
			}
			this.rows=newrows;
		}	
		double current_weighted_count=0.0;
		double current_sum_value [] =new double [this.n_classes];
	
		
		if (this.weights!=null){
		for (int i :rows  ){
			current_weighted_count+=weights[i];
			current_sum_value[fstarget[i]]+=this.weights[i];
			
		}
		} else {
			for (int i :rows){
				current_weighted_count+=1.0;
				current_sum_value[fstarget[i]]+=1.0;
				
			}			
		}
		;
		if (this.verbose){
			System.out.println(" weighted count Value(s) of the zero level with counts " + current_weighted_count);
			System.out.println(" weighted sum Value(s): " + Arrays.toString(current_sum_value));			
		}
		// Initialise the tree structure
		temp_tree_body = new ArrayList<Node>(10000);

		Node initial_node = new Node(current_sum_value,current_weighted_count, current_id);
		
		temp_tree_body.add( initial_node);
		current_id+=1;
		int current_level=1;
		
		// Start expanding the trees
		expand_node(initial_node , this.rows, current_sum_value, current_weighted_count, current_level ) ;
		
	
		
		double sum_importances=get_sum(this.feature_importances);
		for (int i=0; i <feature_importances.length; i++ ){
			feature_importances[i]/=sum_importances;
			
		}
		
		this.rows=null;
		this.weights=null;
		columns=null;
		this.sorted_indices=null;
		this.zero_rank_holder=null;
		this.fstarget=null;
		data=null;
		target=null;
		this.sdataset=null;
		this.starget=null;	
		this.sdataset=null;
		tree_body = new Node [temp_tree_body.size()];

		for (   Node a: this.temp_tree_body) {
			   int key=a.id;
	           tree_body[key]=a;
		}
		
		System.gc();

		
		// calculate first node
			
	}
   /**
    * 
    * @param A_node : The node for expansion
    * @param rowsubset : Rows to use
    * @param current_average_value : the target values
    * @param current_weighted_count : the current weighted count
    * @param current_level : level of the node to expand;
    * <p> main method for expanding tree.
    */
	private void expand_node(Node A_node,int [] rowsubset,  double[] current_sum_value,
			double current_weighted_count, int current_level) {
		
		//Sanity checks to see if we move on from here 
		
		if (current_level<=this.max_depth
				&& current_weighted_count>=this.min_split 
				&& rowsubset.length>=2
				&& this.max_tree_size>this.temp_tree_body.size()
				) {
			
			// create a subset of columns in this iterations
			int columns_to_use [] = new int [this.columndimension];
			int valid_cols[]=new int [this.columndimension];			
			int z=0;
			int subsampleinteger = get_random_integer(this.max_features);
			
			double sum_target []=new double [this.maximum_ranks * this.n_classes];
			double weighted_count []=new double [this.maximum_ranks ];
			int count []=new int [this.maximum_ranks ];
			
			if (this.fsdataset!=null){
				
				for (int col : columns) {
					if (random.nextInt()<=subsampleinteger){
						columns_to_use[z]=col;
						z++;
					} 
				}
				// if we don't get any match...lets have at least one
				if (z==0){
					columns_to_use[z]=columns[random.nextInt(this.columns.length)];

					z++;
					
				}
				
				
			} else if (this.sdataset!=null){
			
				for (int col : columns) {
					if (random.nextInt()<=subsampleinteger){
						columns_to_use[z]=col;
						valid_cols[col]=1;
						z++;
						
						int zerorank=this.zero_rank_holder[col];
						
						if (zerorank==-1){
							continue;
						}	
						count[zerorank]=rowsubset.length;
						weighted_count[zerorank]=current_weighted_count;
						for (int c=0; c <this.n_classes;c++){
							sum_target[zerorank*this.n_classes + c]=current_sum_value[c];
						}
						
					} 
				}
				// if we don't get any match...lets have at least one
				if (z==0){
					columns_to_use[z]=columns[random.nextInt(this.columns.length)];
					valid_cols[columns_to_use[z]]=1;
					
					
					int zerorank=this.zero_rank_holder[columns_to_use[z]];
					
					if (zerorank!=-1){
							
						count[zerorank]=rowsubset.length;
						weighted_count[zerorank]=current_weighted_count;
						for (int c=0; c <this.n_classes;c++){
							sum_target[zerorank*this.n_classes + c]=current_sum_value[c];
						}
					}
					z++;
					
				}
			}
			int best_variable =-1; // the one that determines the split
			int countless_rows=-1;
			double weighted_countless=-1.0;
			double weighted_sumless[]= new double [this.n_classes];
			double best_cuttof=Double.MIN_VALUE; // the cut-off for the best_variable
			boolean better_one_is_found=false; // use this to keep track of whether we found better features or not
			int best_rank=-1;	
			double best_gamma =Double.NEGATIVE_INFINITY; // the metric gain for the best variable
			
			//trigerred columns for this round
			int active_coounter=0;
			if (this.fsdataset!=null){
				
				if (this.weights==null ){
					for (int i=0; i <rowsubset.length;i++){
						//get the row
					    int row=rowsubset[i];
					    //loop through all non zero elements
					    for (int b=0; b<z; b++ ){
					    	//retrieve the column
					    	int column=columns_to_use[ b];
					    	int rank=this.sorted_indices[0][row*this.fsdataset.GetColumnDimension() + column];
					    	active_coounter++; 	
					    	
					    	count[rank]+=1;
					    	weighted_count[rank]+=1.0;
					    	sum_target[(rank*this.n_classes+this.fstarget[row])]+=1.0;
		

					    }
					}
				}else {
					for (int i=0; i <rowsubset.length;i++){
						//get the row
					    int row=rowsubset[i];
					    //loop through all non zero elements
					    for (int b=0; b<z; b++ ){
					    	//retrieve the column
					    	int column=columns_to_use[ b];
					    	int rank=this.sorted_indices[0][row*this.fsdataset.GetColumnDimension() + column];
					    	active_coounter++; 	
					    	count[rank]+=1;
					    	weighted_count[rank]+=weights[row];			    	
					    	sum_target[(rank*this.n_classes+fstarget[row])]+=weights[row];

					    }
					}
				}
				
				
			} else if (this.sdataset!=null){
			
				if (this.weights==null ){
					for (int i=0; i <rowsubset.length;i++){
						//get the row
					    int row=rowsubset[i];
					    //loop through all non zero elements
					    for (int el=sdataset.indexpile[row];el<sdataset.indexpile[row+1];el++ ){
					    	//retrieve the column
					    	int column=sdataset.mainelementpile[el];
					    	// check if column is subset
					    	if (valid_cols[column]!=1){
					    		continue;
					    	}
					    	int rank=this.sorted_indices[0][el];
					    	
					    	count[rank]+=1;
					    	weighted_count[rank]+=1.0;
					    	sum_target[(rank*this.n_classes+this.fstarget[row])]+=1.0;
		
							int zerorank=this.zero_rank_holder[column];
					    	active_coounter++;
							if (zerorank==-1){
								continue;
							}	
							count[zerorank]-=1;
							weighted_count[zerorank]-=1.0;
							sum_target[(zerorank*this.n_classes+this.fstarget[row])]-=1.0;	    	
		
					    }
					}
				}else {
					for (int i=0; i <rowsubset.length;i++){
						//get the row
					    int row=rowsubset[i];
					    //loop through all non zero elements
					    for (int el=sdataset.indexpile[row];el<sdataset.indexpile[row+1];el++ ){
					    	//retrieve the column
					    	int column=sdataset.mainelementpile[el];
					    	// check if column is subset
					    	if (valid_cols[column]!=1){
					    		continue;
					    	}
					    	int rank=this.sorted_indices[0][el];
	
					    	count[rank]+=1;
					    	weighted_count[rank]+=weights[row];			    	
					    	sum_target[(rank*this.n_classes+fstarget[row])]+=weights[row];
					    	active_coounter++;
							int zerorank=this.zero_rank_holder[column];
							
							if (zerorank==-1){
								continue;
							}	
							count[zerorank]-=1;
							weighted_count[zerorank]-=weights[row];	
							sum_target[(zerorank*this.n_classes+fstarget[row])]-=weights[row];    	
		
					    }

					}
				}
			}
			columns_to_use=null;
			valid_cols=null;
			//retrieve all the values

			//subsample ranks
				if (active_coounter>0){
					int valid_threads=(weighted_count.length<this.threads)?weighted_count.length:this.threads;
					int length_of_each_threaded_pass =(int)(((double)weighted_count.length/(double)valid_threads));
					Thread [] thread_array= new Thread[valid_threads];
					int locations [][]= new int[valid_threads][2];
					int points=0;
					int m=0;
					
					if (this.fsdataset!=null){
						
						for (int n=0; n <valid_threads-1; n++ ){
							locations[n][0]=points;
							//System.out.println((points + length_of_each_threaded_pass) + " " + this.maximum_ranks + " " + weighted_count.length);
						
							if (points + length_of_each_threaded_pass>=weighted_count.length){
								break;
							}
							int end=points + length_of_each_threaded_pass;
							int el=this.sorted_indices[1][end];
							int col=el%fsdataset.GetColumnDimension();
							int newcol=col;
							while (col==newcol && end <weighted_count.length-1){
								end++;
								el=this.sorted_indices[1][end];
								newcol=el%fsdataset.GetColumnDimension();
							}
							if (end>weighted_count.length-1){
								end=weighted_count.length;
							}
							locations[n][1]=end;
							m++;
							points=end;
							if (end>=weighted_count.length){
								break;
							}
						}

						} else if (this.sdataset!=null){

						// the threads of operations
						for (int n=0; n <valid_threads-1; n++ ){
							locations[n][0]=points;
							if (points + length_of_each_threaded_pass>=weighted_count.length){
								break;
							}
							int end=points + length_of_each_threaded_pass;
							int col=this.sorted_indices[2][end];
							int newcol=col;
							while (col==newcol && end <weighted_count.length-1){
								end++;
								newcol=this.sorted_indices[2][end];
							}
							if (end>weighted_count.length-1){
								end=weighted_count.length;
							}
							locations[n][1]=end;
							m++;
							points=end;
							if (end>=weighted_count.length){
								break;
							}
						}

						}
					
					if (points<weighted_count.length){
						locations[m][0]=points;
						locations[m][1]=weighted_count.length;
						m++;
					}
					int best_ranks[]= new int [m];
					double best_gama[]=new double [m];
					
					subsampleinteger = get_random_integer(this.cut_off_subsample);
					if (this.fsdataset!=null){
						
						for (int t=0; t<m;t++){
							thread_array[t]=new Thread(new categoricalmetric(
									current_weighted_count,
									current_sum_value,
									best_gama,
									best_ranks,
									subsampleinteger,
									this.sorted_indices[1],
									count,
									sum_target,
									weighted_count,
									this.Objective,
									t,locations[t][0],
									locations[t][1],
									this.fsdataset.GetColumnDimension()));
							//System.out.println(t + " " + locations[t][0]+ " " + locations[t][1] + " " + this.sorted_indices[2][locations[t][0]] + " " + this.sorted_indices[2][locations[t][1]-1]);
							thread_array[t].start();
						}
					} else if (this.sdataset!=null){
						for (int t=0; t<m;t++){
							thread_array[t]=new Thread(new categoricalmetric(
									current_weighted_count,
									current_sum_value,
									best_gama,
									best_ranks,
									subsampleinteger,
									this.sorted_indices[2],
									count,
									sum_target,
									weighted_count,
									this.Objective,
									t,locations[t][0],locations[t][1]));
							//System.out.println(t + " " + locations[t][0]+ " " + locations[t][1] + " " + this.sorted_indices[2][locations[t][0]] + " " + this.sorted_indices[2][locations[t][1]-1]);
							thread_array[t].start();
						}
					}
					for (int t=0; t<m;t++){
						try {
							thread_array[t].join();
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
					
					
		
					for (int j=0;j < best_ranks.length; j++){
						//System.out.println(best_gama[j]  + " " + best_ranks[j]);
						if (best_gama[j]>best_gamma){
							best_gamma=best_gama[j];
							best_rank=best_ranks[j];
							better_one_is_found=true;
						}
					}
			
				}
			
			if (better_one_is_found){
				if (this.fsdataset!=null){
					int ind=this.sorted_indices[1][best_rank];
					best_variable=ind%this.fsdataset.GetColumnDimension();
				} else if (this.sdataset!=null){
					best_variable =this.sorted_indices[2][best_rank]; // the one that determines the split
				}
				countless_rows=count[best_rank];
				weighted_countless=weighted_count[best_rank];
				for (int j=0; j < this.n_classes; j++){
					weighted_sumless[j]=sum_target[best_rank*this.n_classes + j];
				}
				int best_index=this.sorted_indices[1][best_rank];
				if (this.fsdataset!=null){
					best_cuttof=Math.round(fsdataset.data[best_index]* 10.0 * rounding) / (10.0 * rounding);
				} else if (this.sdataset!=null){
					if (best_index==-1){
						best_cuttof=0.0;
					} else {
						best_cuttof=Math.round(sdataset.valuespile[best_index]* 10.0 * rounding) / (10.0 * rounding);
								
					}
				}

			}
	
				//System.gc();
		
				// there is  a better split - thats good!
			if (better_one_is_found && best_gamma>= Double.NEGATIVE_INFINITY) {
				//System.out.println("--------------------------------------");

				feature_importances[best_variable]+=best_gamma;
				
			// calculate the rest of the stats for the "higher - than " of teh best split
			int countmore_rows=rowsubset.length-countless_rows;
			double weighted_countmore= current_weighted_count-weighted_countless;
			
			double weighted_summore[]= new double [this.n_classes];
			for (int s=0; s < this.n_classes ; s++){
				weighted_summore[s]=current_sum_value[s]-weighted_sumless[s];
			}
			if (this.verbose){
			System.out.println("level " + current_level +  "  best_variable " +  " "+ best_variable + " best_cuttof " + best_cuttof + " " + this.Objective +  " "+ best_gamma);
			}
			/*	
			System.out.println("level " + current_level);
			System.out.println("best_variable " + best_variable);
			System.out.println("best_cuttof " + best_cuttof);
			System.out.println("best_gamma " + best_gamma);
			System.out.println("countless_rows " + countless_rows);
			System.out.println("weighted_sumless " + Arrays.toString(weighted_sumless));
			System.out.println("weighted_countless " + weighted_countless);
			System.out.println("countmore_rows " +countmore_rows);
			System.out.println("weighted_summore " +  Arrays.toString(weighted_summore));
			System.out.println("weighted_countmore " + weighted_countmore);	
			System.out.println("rows.length " +rowsubset.length);
			System.out.println("weighted_summ" +  Arrays.toString(current_sum_value));
			System.out.println("current_weighted_count " + current_weighted_count);	
			
			
			if (countmore_rows<0 ) {
				throw new exceptions.IllegalStateException(" count_more rows should not be less thab zero: " + countmore_rows);
			}
			if (countless_rows<0 ) {
				throw new exceptions.IllegalStateException("  countless_rowsrows should not be less thab zero: " + countless_rows);
			}
			if (weighted_countmore<0 ) {
				throw new exceptions.IllegalStateException("  weighted_countmore rows should not be less thab zero: " + weighted_countmore);
			}
			if (weighted_countless<0 ) {
				throw new exceptions.IllegalStateException(" weighted_countless rows should not be less thab zero: " + weighted_countless);

			}			
			 */
			
			if (this.fsdataset!=null){
				
				// both are to be splitted
				if (countmore_rows>0 && countless_rows >0 && weighted_countmore>=this.min_leaf && weighted_countless>=this.min_leaf){
					
					// add details to the current nod
					A_node.specifyvariable(best_variable, best_cuttof);
					
					int valid_rows_for_less [] =new int[countless_rows];
					int valid_rows_for_more [] =new int[countmore_rows];
					int less=0;
					int more=0;
					
					//populate with the correct indices
	
						for(int ro: rowsubset){
							
							int index=ro*this.fsdataset.GetColumnDimension() + best_variable;
							int ranking=this.sorted_indices[0][ index ];

									if ( ranking<=best_rank){
										valid_rows_for_less[less++]=ro;	
									}else {
										valid_rows_for_more[more++]=ro;	
									}
								} // end of check_feature if statement

	
					 // aren't we good?! We have our rows, lets create the nodes
					 //before this code becomes any longer
					 Node Less= new Node(weighted_sumless ,weighted_countless,current_id);
					 A_node.setchildless(current_id);
					 temp_tree_body.add( Less);
					 current_id+=1;
					 Node Right= new Node(weighted_summore,weighted_countmore,current_id);	
					 A_node.setchildmore(current_id);
					 temp_tree_body.add( Right);
					 current_id+=1;
					 
					 rowsubset=null;
					 
					 // expand Left Node nodes
					 expand_node(Less,valid_rows_for_less, weighted_sumless,
							 weighted_countless,current_level + 1);
					
					// expand Right Node nodes
					 expand_node(Right,valid_rows_for_more, weighted_summore,
							 weighted_countmore,current_level + 1);				
					
					
					// end of "all is good, everything is to be splitted"
				} else if (countless_rows >0 &&  weighted_countless>=this.min_leaf) {
					
					
					// add details to the current nod
					A_node.specifyvariable(best_variable, best_cuttof);
	
					//populate with the correct indices
					
						int valid_rows_for_less [] =new int[countless_rows];
	
						int less=0;
							
							for(int ro: rowsubset){
								
								int index=ro*this.fsdataset.GetColumnDimension() + best_variable;
								int ranking=this.sorted_indices[0][ index ];
								
										if ( ranking<=best_rank){
											valid_rows_for_less[less++]=ro;	
										}

							}					
						
	
					 // aren't we good?! We have our rows, lets create the nodes
					 //before this code becomes any longer
					 Node Less= new Node(weighted_sumless ,weighted_countless,current_id);
					 A_node.setchildless(current_id);
					 temp_tree_body.add( Less);
					 current_id+=1;
					 rowsubset=null;
					 // expand Left Node nodes
					 expand_node(Less,valid_rows_for_less, weighted_sumless,
							 weighted_countless,current_level + 1);
						
	
					// end of Left splitting only"
				}  else if (countmore_rows>0 && weighted_countmore>=this.min_leaf ){
					
					// add details to the current nod
					A_node.specifyvariable(best_variable, best_cuttof);
					int valid_rows_for_more [] =new int[countmore_rows];
					int more=0;
					
					//populate with the correct indices

						for(int ro: rowsubset){
							
							int index=ro*this.fsdataset.GetColumnDimension() + best_variable;
							int ranking=this.sorted_indices[0][ index ];
									if ( ranking>best_rank){
										valid_rows_for_more[more++]=ro;	
									}

						}					
					
					 // aren't we good?! We have our rows, lets create the nodes
					 //before this code becomes any longer
					 Node Right= new Node(weighted_summore,weighted_countmore,current_id);	
					 A_node.setchildmore(current_id);
					 temp_tree_body.add( Right);
					 current_id+=1;
					 rowsubset=null;
					// expand Right Node nodes
					 expand_node(Right,valid_rows_for_more, weighted_summore,
							 weighted_countmore,current_level + 1);				
					
					
					// end of right splitting only
				
	
				} else {
					
					 rowsubset=null;
					 //children-less I am afraid
					 A_node.childless=-1;
					 A_node.childmore=-1;
					
					 // end of no match.
				}
				
				
			} else if(this.sdataset!=null){
			
			
				// both are to be splitted
				if (countmore_rows>0 && countless_rows >0 && weighted_countmore>=this.min_leaf && weighted_countless>=this.min_leaf){
					
					// add details to the current nod
					A_node.specifyvariable(best_variable, best_cuttof);
					
					int valid_rows_for_less [] =new int[countless_rows];
					int valid_rows_for_more [] =new int[countmore_rows];
					int less=0;
					int more=0;
					
					//populate with the correct indices
	
						for(int ro: rowsubset){
							int count_rows=0;
							for (int j=sdataset.indexpile[ro]; j < sdataset.indexpile[ro+1];j++ ){
								
								int check_feature=sdataset.mainelementpile[j];
								
								if (check_feature<best_variable){ // we found our feature
									continue;// next row - here the feature has zero value
								}
								if (check_feature>best_variable){ // we found our feature
									count_rows+=1;
									int ranking=this.zero_rank_holder[best_variable];
									if ( ranking<=best_rank){
										valid_rows_for_less[less++]=ro;	
									}else {
										valid_rows_for_more[more++]=ro;	
									}
									break;// next row - here the feature has zero value
								}					
								if (check_feature==best_variable ){ // we found our feature
									count_rows+=1;
									int ranking=this.sorted_indices[0][ j ];
									if ( ranking<=best_rank){
										valid_rows_for_less[less++]=ro;	
									}else {
										valid_rows_for_more[more++]=ro;	
									}
									break;//found it! no longer need to keep on looping
								} // end of check_feature if statement
								
								
							}// end of columns loop	
							if (count_rows==0){
								int ranking=this.zero_rank_holder[best_variable];
								if ( ranking<=best_rank){
									valid_rows_for_less[less++]=ro;	
								}else {
									valid_rows_for_more[more++]=ro;	
								}								
							}
						}					
					
	
					 // aren't we good?! We have our rows, lets create the nodes
					 //before this code becomes any longer
					 Node Less= new Node(weighted_sumless ,weighted_countless,current_id);
					 A_node.setchildless(current_id);
					 temp_tree_body.add( Less);
					 current_id+=1;
					 Node Right= new Node(weighted_summore,weighted_countmore,current_id);	
					 A_node.setchildmore(current_id);
					 temp_tree_body.add( Right);
					 current_id+=1;
					 
					 rowsubset=null;
					 
					 // expand Left Node nodes
					 expand_node(Less,valid_rows_for_less, weighted_sumless,
							 weighted_countless,current_level + 1);
					
					// expand Right Node nodes
					 expand_node(Right,valid_rows_for_more, weighted_summore,
							 weighted_countmore,current_level + 1);				
					
					
					// end of "all is good, everything is to be splitted"
				} else if (countless_rows >0 &&  weighted_countless>=this.min_leaf) {
					
					
					// add details to the current nod
					A_node.specifyvariable(best_variable, best_cuttof);
	
					//populate with the correct indices
					
						int valid_rows_for_less [] =new int[countless_rows];
	
						int less=0;
							
							for(int ro: rowsubset){
								int count_rows=0;
								
								for (int j=sdataset.indexpile[ro]; j < sdataset.indexpile[ro+1];j++ ){
									int check_feature=sdataset.mainelementpile[j];
								
									if (check_feature<best_variable){ // we found our feature
										continue;// next row - here the feature has zero value
									}
									if (check_feature>best_variable){ // we found our feature
										count_rows+=1;
										int ranking=this.zero_rank_holder[best_variable];
										if ( ranking<=best_rank){
											valid_rows_for_less[less++]=ro;	
										}
										break;// next row - here the feature has zero value
									}					
									if (check_feature==best_variable ){ // we found our feature
										count_rows+=1;
										int ranking=this.sorted_indices[0][ j ];
										if ( ranking<=best_rank){
											valid_rows_for_less[less++]=ro;	
										}
										break;//found it! no longer need to keep on looping
									} // end of check_feature if statement
									
									
								}// end of columns loop		
								if (count_rows==0){
									int ranking=this.zero_rank_holder[best_variable];
									if ( ranking<=best_rank){
										valid_rows_for_less[less++]=ro;	
									}						
								}
							}					
						
	
					 // aren't we good?! We have our rows, lets create the nodes
					 //before this code becomes any longer
					 Node Less= new Node(weighted_sumless ,weighted_countless,current_id);
					 A_node.setchildless(current_id);
					 temp_tree_body.add( Less);
					 current_id+=1;
					 rowsubset=null;
					 // expand Left Node nodes
					 expand_node(Less,valid_rows_for_less, weighted_sumless,
							 weighted_countless,current_level + 1);
						
	
					// end of Left splitting only"
				}  else if (countmore_rows>0 && weighted_countmore>=this.min_leaf ){
					
					// add details to the current nod
					A_node.specifyvariable(best_variable, best_cuttof);
					int valid_rows_for_more [] =new int[countmore_rows];
					int more=0;
					
					//populate with the correct indices
						
	
						for(int ro: rowsubset){
							
							int count_rows=0;
							
							for (int j=sdataset.indexpile[ro]; j < sdataset.indexpile[ro+1];j++ ){
								int check_feature=sdataset.mainelementpile[j];

								if (check_feature<best_variable){ // we found our feature
									continue;// next row - here the feature has zero value
								}
								if (check_feature>best_variable){ // we found our feature
									count_rows+=1;
									int ranking=this.zero_rank_holder[best_variable];
									if ( ranking>best_rank){
									
										valid_rows_for_more[more++]=ro;	
									}
									break;// next row - here the feature has zero value
								}					
								if (check_feature==best_variable ){ // we found our feature
									count_rows+=1;
									int ranking=this.sorted_indices[0][ j ];
									if ( ranking>best_rank){
										valid_rows_for_more[more++]=ro;	
									}
									break;//found it! no longer need to keep on looping
								} // end of check_feature if statement
								
								
							}// end of columns loop				
							if (count_rows==0){
								int ranking=this.zero_rank_holder[best_variable];
								if ( ranking>best_rank){
								
									valid_rows_for_more[more++]=ro;	
								}				
							}
						}					
					
					 // aren't we good?! We have our rows, lets create the nodes
					 //before this code becomes any longer
					 Node Right= new Node(weighted_summore,weighted_countmore,current_id);	
					 A_node.setchildmore(current_id);
					 temp_tree_body.add( Right);
					 current_id+=1;
					 rowsubset=null;
					// expand Right Node nodes
					 expand_node(Right,valid_rows_for_more, weighted_summore,
							 weighted_countmore,current_level + 1);				
					
					
					// end of right splitting only
				
	
				} else {
					
					 rowsubset=null;
					 //children-less I am afraid
					 A_node.childless=-1;
					 A_node.childmore=-1;
					
					 // end of no match.
				}
			
			}
			
			// end of better one is found
			}
				
			
			
			
			
			
			
			
			
			// end of requirements' section
		}

		
		
	}
	/**
	 * Retrieve the number of target variables
	 */
	public int getnumber_of_targets(){
		return n_classes;
	}
	
	
	public double get_sum(double array []){
		double a=0.0;
		for (int i=0; i <array.length; i++ ){
			a+=array[i];
		}
		return a;
	}
	
	/**
	 * 
	 * @returns the closest integer that reflects this percentage!
	 * <p> it may sound strange, random.nextint can be significantly faster than nextdouble()
	 */
	public int get_random_integer(double percentage){
		
		double per= Math.min(Math.max(0, percentage),1.0);
		double difference=2147483647.0+(2147483648.0);
		int point=(int)(-2147483648.0 +  (per*difference ));
		
		
		return point;
		
	}

	@Override
	public String GetType() {
	
		return "classifier";
	}

	@Override
	public boolean SupportsWeights() {
		return true;
	}

	@Override
	public String GetName() {
		return "DecisionTreeClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: Decision Tree Classifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);				
		System.out.println("cut_off_subsample: "+ this.cut_off_subsample);
		System.out.println("Objective: "+ this.Objective);
		System.out.println("tau: "+ this.tau);
		System.out.println("feature_subselection: "+ this.feature_subselection);
		System.out.println("gamma: "+ this.gamma);		
		System.out.println("max_depth: "+ this.max_depth);			
		System.out.println("max_features : "+ this.max_features);	
		System.out.println("max_tree_size : "+ this.max_tree_size);
		System.out.println("min_leaf : "+ this.min_leaf);	
		System.out.println("min_leaf : "+ this.min_split);	
		System.out.println("row_subsample : "+ this.row_subsample);			
		System.out.println("threads : "+ this.threads);			
		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (this.tree_body==null){
			System.out.println("Trained: False");	
		} else {
			System.out.println("Trained: True");
		}
		
	}

	@Override
	public boolean HasTheSametype(estimator a) {
		if (a.GetType().equals(this.GetType())){
			return true;
		} else {
		return false;
		}
	}

	@Override
	public boolean isfitted() {
		if (tree_body!=null || tree_body.length>0){
			return true;
		} else {
		return false;
		}
	}

	@Override
	public boolean IsRegressor() {
		return false ;
	}

	@Override
	public boolean IsClassifier() {
		return true ;
	}

	@Override
	public void reset() {
		this.tree_body= null;
		n_classes=0;
		tau=0.5;
		Objective="ENTROPY";
		threads=1;
		this.columns=null;
		this.random=null;
		this.cut_off_subsample=1.0;
		this.feature_importances.clone();
		this.feature_subselection=1.0;
		this.gamma=1E-30;
		this.max_depth=3;
		this.max_features=1.0;
		this.max_tree_size=-1;
		this.min_leaf=1.0;
		this.min_split=2.0;
		this.row_subsample=1.0;
		columndimension=0;
		copy=true;
		seed=1;
		random=null;
		target=null;
		fstarget=null;
		starget=null;
		weights=null;
		verbose=true;
		
	}

	@Override
	public estimator copy() {
		DecisionTreeClassifier br = new DecisionTreeClassifier();
		
		Node[] ntree = new Node[tree_body.length];
		
		for (int i=0; i <ntree.length; i++ ){
			   Node a =tree_body[i];
	           Node b= new Node(a.sum_prediction.clone(),a.weighted_count,i);
	           b.Variable=a.Variable;
	           b.cutoffval=a.cutoffval;
	           b.childless=a.childless;
	           b.childmore=a.childmore;
	           ntree[i]= b;
		}
		

		
		br.n_classes=this.n_classes;
		br.columns=this.columns.clone();
		br.cut_off_subsample=this.cut_off_subsample;
		br.tau= this.tau;
		br.feature_importances=this.feature_importances.clone();
		br.feature_subselection=this.feature_subselection;
		br.gamma=this.gamma;
		br.max_depth=this.max_depth;
		br.max_features=this.max_features;
		br.max_tree_size=this.max_tree_size;
		br.Objective=this.Objective;
		br.min_leaf=this.min_leaf;
		br.min_split=this.min_split;
		br.row_subsample=this.row_subsample;
		br.threads=this.threads;
		br.columndimension=this.columndimension;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.target=manipulate.copies.copies.Copy(this.target.clone());	
		br.fstarget= this.fstarget.clone();
		br.starget=(smatrix) this.starget.Copy();
		br.weights=manipulate.copies.copies.Copy(this.weights.clone());
		br.verbose=this.verbose;
		return br;
	}
	
	@Override	
	public void set_params(String params){
		
		String splitted_params []=params.split(" " + "+");
		
		for (int j=0; j<splitted_params.length; j++ ){
			String mini_split []=splitted_params[j].split(":");
			if (mini_split.length>=2){
				String metric=mini_split[0];
				String value=mini_split[1];
				
				if (metric.equals("cut_off_subsample")) {this.cut_off_subsample=Double.parseDouble(value);}
				else if (metric.equals("feature_subselection")) {this.feature_subselection=Double.parseDouble(value);}
				else if (metric.equals("row_subsample")) {this.row_subsample=Double.parseDouble(value);}	
				else if (metric.equals("min_leaf")) {this.min_leaf=Double.parseDouble(value);}						
				else if (metric.equals("max_depth")) {this.max_depth=Integer.parseInt(value);}
				else if (metric.equals("Objective")) {this.Objective=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("rounding")) {this.rounding=Double.parseDouble(value);}				
				else if (metric.equals("offset")) {this.offset=Double.parseDouble(value);}						
				else if (metric.equals("max_tree_size")) {this.max_tree_size=Integer.parseInt(value);}
				else if (metric.equals("gamma")) {this.gamma=Double.parseDouble(value);}
				else if (metric.equals("max_features")) {this.max_features=Double.parseDouble(value);}
				else if (metric.equals("bootsrap")) {this.bootsrap=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("min_split")) {this.min_split=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}				
				
			}
			
		}
		

	}


	@Override
	public scaler ReturnScaler() {
		return null;
	}
	@Override
	public void setScaler(scaler sc) {

	}
	@Override
	public void setSeed(int seed) {
		this.seed=seed;}	
	
	@Override	
	public void set_target(double data []){
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		this.target=data;
	}
	@Override
	public int getSeed() {
		return this.seed;}
}

	  

