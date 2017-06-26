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

package ml.nn;


import java.util.Random;

import preprocess.scaling.maxscaler;
import preprocess.scaling.scaler;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;

/**
 * @author marios
 *<p> class to implement binary  2-hidden layer ff neural network for regression with single-layer output <p>
 *The implementation is heavily based on the equivalent one in the <a href="https://pypi.python.org/pypi/Kaggler">kaggler</a> package
 <p> There has been some changes to make it quicker that mostly have to do with the adjustement of the leanring rate</p>
 <p> it also supports more functions </p>
 */
public class singleVanilla2hnn implements estimator,regressor,Runnable {

	private static final long serialVersionUID = 830529727388893394L;
	
	/**
	 * Private method for when this class is used in a multinomial context to avoid re-sorting each time (for each class)
	 */
	private boolean sparse_set=false;
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;	
	/**
	 * Type of algorithm to use. It has to be  SGD
	 */
	public String Type="SGD";
	/**
	 * Type of nonlinearity between layers. It has to be one of  Relu,Linear,Sigmoid,Tanh
	 */
	public String connection_nonlinearity="Relu";	

	/**
	 * True if we want to optimize for squared distance or not
	 */
	public boolean quadratic=false;	
	/**
	 * threads to use
	 */
	public int threads=1;

	/**
	 * Maximum number of iterations. -1 for "until convergence"
	 */
	public int maxim_Iteration=-1;
	/**
	 * scale the dataset before use
	 */
	public boolean usescale=true;
	/**
	 * for sgd only
	 */
	public boolean shuffle=true;
	/**
	 * for SGD
	 */
	public double learn_rate=1.0;
	/**
	 * Initialise values of the latent features with values between[0,init_values)
	 */
	public double init_values=0.1;
	/**
	 * Scaler to use in case of usescale=true
	 */
	private preprocess.scaling.scaler Scaler;
	/**
	 * scale the copy the dataset
	 */
	public boolean copy=false;
    /**
     * seed to use
     */
	public int seed=1;
	/**
	 * Random number generator to use
	 */
	private Random random;
	/**
	 * minimal change in coefficients' values after nth iterations to stop the algorithm
	 */
	public double tolerance=0.0001; 
	
	/***
	 * Smooth valued for better converges
	 */
	public double smooth=0.1;
	/**
	 * double target use , an array of 0 and 1
	 */
	public double  [] target;
	/**
	 * weighst to used per row(sample)
	 */
	public double [] weights;
	/**
	 * if true, it prints stuff
	 */
	public boolean verbose=true;
	/**
	 * weights between the input and 1st hidden layers
	 */
	private double w0[];
	/**
	 *  weights between the 1st and 2nd hidden layers
	 */
	private double w1[];
	/**
	 *  weights between the 2nd hidden and output layers
	 */
	private double w2[];
	
    /**number of the 1st level hidden units**/
	
    public int h1 =256;
    
    /**number of the 2nd level hidden units**/
	
    public int h2 =128;    
    
    /**
	 * The objective to optimise . It may be RMSE (which is for classic linear-least squares regression)
	 *  or MAE for Mean Absolute Error or QUANTILE 
	 */

	public String Objective="RMSE";	
	/**
	 * quantile value
	 */
	public double tau=0.5;
	
	
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
	}
	public int get_1st_layerhunits(){
		return h1;
	}
	public int get_2ndst_layerhunits(){
		return h2;
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
	/**
	 * Default constructor for singleVanilla2hnn regressor
	 */
	public singleVanilla2hnn(){}
	/**
	 * 
	 * @param need_sort : Whether we want to avoid sorting (used internally for multinomial models)
	 */
	public void set_sparse_indicator(boolean need_sort){
		this.sparse_set=true;
	}
	/**
	 * 
	 * @param data : data to create a constructor
	 */
	public singleVanilla2hnn(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	/**
	 * 
	 * @param data : data to create a constructor
	 */
	public singleVanilla2hnn(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * 
	 * @param data : data to create a constructor
	 */
	public singleVanilla2hnn(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
	}	
	/**
	 * @param w_0 : array for input-to-hidden weights
	 * @param w_1 : array for hidden-to-hidden2 weights
	 * @param w_2 : array for hidden2-to-output weights
	 * <p>Sets initial values for the weights between the input, hidden layers and output
	 */
	public void set_w0_w1_w2(double w_0 [],double w_1 [],double w_2 []){
		w0=w_0;
		w1=w_1;
		w2=w_2;
	}

	/**
	 * 
	 * @return the weights between the input and 1st hidden layer
	 */
	public double [] get_w0(){
		if (w0==null || w0.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(w0);
	}
	/**
	 * 
	 * @return the weights between the   1st hidden layer and 2nd  hidden layer
	 */
	public double [] get_w1(){
		if (w1==null || w1.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(w1);
	}
	/**
	 * 
	 * @return the weights between the  2nd  hidden layer and the output
	 */
	public double [] get_w2(){
		if (w2==null || w2.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(w2);
	}

	
     /**
      * 
      * @param data : to be scored
      * @return the probability for the event to be 1
      */
	public double[] predict(double[][] data) {
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data[0].length);	
		}
		
		double predictions[]= new double [data.length];
        //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
			
			for (int i=0; i < predictions.length; i++) {
				double pred=w2[this.h2];

		    	
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[this.columndimension * this.h1 + j];
	             }
	             
		    	
               for (int d=0; d <this.columndimension; d++){
              	 double v=data[i][d];
              	 	if ( v == 0){
	                    continue;
              	 		}
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[d * this.h1 + j] * v;
		                 }
	               	 
               }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[this.h1 * this.h2 + s];
		             
		             for (int j=0; j < this.h1; j++){

		                 z2[s] += w1[j * this.h2 + s] * z1[j];
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }		             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         

		         predictions[i]=pred;
			}
		return predictions;
	}
	

    /**
     * 
     * @param data : to be scored
     * @return the continuous prediction
     */
	public double[] predict(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [data.GetRowDimension()];
		
	       //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
			
			for (int i=0; i < predictions.length; i++) {
				double pred=w2[this.h2];

		    	
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[this.columndimension * this.h1 + j];
	             }
	             
		    	
               for (int d=0; d <this.columndimension; d++){
              	 double v=data.GetElement(i, d);
              	 	if ( v == 0){
	                    continue;
              	 		}
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[d * this.h1 + j] * v;
		                 }
	               	 
               }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[this.h1 * this.h2 + s];
		             
		             for (int j=0; j < this.h1; j++){

		                 z2[s] += w1[j * this.h2 + s] * z1[j];
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }		             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         

		         predictions[i]=pred;
			}
		return predictions;
	}
	
    /**
     * 
     * @param data : to be scored
     * @return the continuous prediction
     */
	
	public double[] predict(smatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [data.GetRowDimension()];
		// check if data is sorted via row
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

	       //hidden units in the 2nd hidden layer
		double z2 [] = new double [this.h2];
		//hidden units in the 1st hidden layer
		double z1 [] = new double [this.h1];
			
			for (int i=0; i < predictions.length; i++) {
				double pred=w2[this.h2];

		    	
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[this.columndimension * this.h1 + j];
	             }
	             
		    	
		            for (int y=data.indexpile[i]; y<data.indexpile[i+1]; y++ ) {
		            int d=data.mainelementpile[y];
		            double v=data.valuespile[y];
           	 	if ( v == 0){
	                    continue;
           	 		}
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[d * this.h1 + j] * v;
		                 }
	               	 
            }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[this.h1 * this.h2 + s];
		             
		             for (int j=0; j < this.h1; j++){

		                 z2[s] += w1[j * this.h2 + s] * z1[j];
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }		             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         

		         predictions[i]=pred;
			}

		return predictions;
	}	
	


	
	

	@Override
	public double predict_Row(double[] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}
	       //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
			
		double pred=w2[this.h2];

    	
         for (int j=0; j < this.h1; j++){
             // starting with the bias in the input layer
             z1[j] = this.w0[this.columndimension * this.h1 + j];
         }
         
    	
       for (int d=0; d <this.columndimension; d++){
      	 double v=data[d];
      	 	if ( v == 0){
                continue;
      	 		}
               	 if (this.usescale){
               		 v=Scaler.transform(v, d);
               	 }
                 for (int j=0; j < this.h1; j++){
                    z1[j] += this.w0[d * this.h1 + j] * v;
                 }
           	 
       }
    	
    	
         for (int j=0; j < this.h1; j++){
             // apply the ReLU activation function to the first level hidden unit
             if (this.connection_nonlinearity.equals(("Relu"))){
             z1[j] = Relu(z1[j]);
             }else if (this.connection_nonlinearity.equals(("Linear"))){
            	 z1[j] = Linear(z1[j]); 
             }else if (this.connection_nonlinearity.equals(("Tanh"))){
            	 z1[j] = Tanh(z1[j]); 
             } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
            	 z1[j] = sigmoid(z1[j]); 
             }
         }		    	
    	
    	 // calculating and adding values of 2nd level hidden units
         for (int s=0; s <this.h2; s++ ){
             // staring with the bias in the 1st hidden layer
             z2[s]= w1[this.h1 * this.h2 + s];
             
             for (int j=0; j < this.h1; j++){

                 z2[s] += w1[j * this.h2 + s] * z1[j];
             }
             
             // apply the ReLU activation function to the 2nd level hidden unit
             if (this.connection_nonlinearity.equals(("Relu"))){
            	 z2[s] = Relu( z2[s]);
             } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
            	 z2[s] = sigmoid( z2[s]); 
             }else if (this.connection_nonlinearity.equals(("Tanh"))){
            	 z2[s] = Tanh( z2[s]); 
             }else if (this.connection_nonlinearity.equals(("Linear"))){
            	 z2[s] = Linear( z2[s]); 
             }		             
             
             pred += this.w2[s] * z2[s];    
         }

		         
		return pred;
	}

	@Override
	public double predict_Row(fsmatrix data, int row) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
	 //hidden units in the 2nd hidden layer
     double z2 [] = new double [this.h2];
     //hidden units in the 1st hidden layer
     double z1 [] = new double [this.h1];
			
		double pred=w2[this.h2];

 	
      for (int j=0; j < this.h1; j++){
          // starting with the bias in the input layer
          z1[j] = this.w0[this.columndimension * this.h1 + j];
      }
      
 	
    for (int d=0; d <this.columndimension; d++){
   	 double v=data.GetElement(row, d);
   	 	if ( v == 0){
             continue;
   	 		}
            	 if (this.usescale){
            		 v=Scaler.transform(v, d);
            	 }
              for (int j=0; j < this.h1; j++){
                 z1[j] += this.w0[d * this.h1 + j] * v;
              }
        	 
    }
 	
 	
      for (int j=0; j < this.h1; j++){
          // apply the ReLU activation function to the first level hidden unit
          if (this.connection_nonlinearity.equals(("Relu"))){
          z1[j] = Relu(z1[j]);
          }else if (this.connection_nonlinearity.equals(("Linear"))){
         	 z1[j] = Linear(z1[j]); 
          }else if (this.connection_nonlinearity.equals(("Tanh"))){
         	 z1[j] = Tanh(z1[j]); 
          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
         	 z1[j] = sigmoid(z1[j]); 
          }
      }		    	
 	
 	 // calculating and adding values of 2nd level hidden units
      for (int s=0; s <this.h2; s++ ){
          // staring with the bias in the 1st hidden layer
          z2[s]= w1[this.h1 * this.h2 + s];
          
          for (int j=0; j < this.h1; j++){

              z2[s] += w1[j * this.h2 + s] * z1[j];
          }
          
          // apply the ReLU activation function to the 2nd level hidden unit
          if (this.connection_nonlinearity.equals(("Relu"))){
         	 z2[s] = Relu( z2[s]);
          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
         	 z2[s] = sigmoid( z2[s]); 
          }else if (this.connection_nonlinearity.equals(("Tanh"))){
         	 z2[s] = Tanh( z2[s]); 
          }else if (this.connection_nonlinearity.equals(("Linear"))){
         	 z2[s] = Linear( z2[s]); 
          }		             
          
          pred += this.w2[s] * z2[s];    
      }
      
		         
	return pred;
	}

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		 //hidden units in the 2nd hidden layer
	     double z2 [] = new double [this.h2];
	     //hidden units in the 1st hidden layer
	     double z1 [] = new double [this.h1];
				
			double pred=w2[this.h2];

	 	
	      for (int j=0; j < this.h1; j++){
	          // starting with the bias in the input layer
	          z1[j] = this.w0[this.columndimension * this.h1 + j];
	      }
	      
	 	
	    for (int y=start; y <end; y++){
          int d=data.mainelementpile[y];
          double v=data.valuespile[y];	    	

	   	 	if ( v == 0){
	             continue;
	   	 		}
	            	 if (this.usescale){
	            		 v=Scaler.transform(v, d);
	            	 }
	              for (int j=0; j < this.h1; j++){
	                 z1[j] += this.w0[d * this.h1 + j] * v;
	              }
	        	 
	    }
	 	
	 	
	      for (int j=0; j < this.h1; j++){
	          // apply the ReLU activation function to the first level hidden unit
	          if (this.connection_nonlinearity.equals(("Relu"))){
	          z1[j] = Relu(z1[j]);
	          }else if (this.connection_nonlinearity.equals(("Linear"))){
	         	 z1[j] = Linear(z1[j]); 
	          }else if (this.connection_nonlinearity.equals(("Tanh"))){
	         	 z1[j] = Tanh(z1[j]); 
	          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	         	 z1[j] = sigmoid(z1[j]); 
	          }
	      }		    	
	 	
	 	 // calculating and adding values of 2nd level hidden units
	      for (int s=0; s <this.h2; s++ ){
	          // staring with the bias in the 1st hidden layer
	          z2[s]= w1[this.h1 * this.h2 + s];
	          
	          for (int j=0; j < this.h1; j++){

	              z2[s] += w1[j * this.h2 + s] * z1[j];
	          }
	          
	          // apply the ReLU activation function to the 2nd level hidden unit
	          if (this.connection_nonlinearity.equals(("Relu"))){
	         	 z2[s] = Relu( z2[s]);
	          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	         	 z2[s] = sigmoid( z2[s]); 
	          }else if (this.connection_nonlinearity.equals(("Tanh"))){
	         	 z2[s] = Tanh( z2[s]); 
	          }else if (this.connection_nonlinearity.equals(("Linear"))){
	         	 z2[s] = Linear( z2[s]); 
	          }		             
	          
	          pred += this.w2[s] * z2[s];    
	      }
	      
			         
		return pred;
	}

	
	
	


	@Override
	public double[][] predict2d(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data[0].length);	
		}
		
		double predictions[][]= new double [data.length][1];
        //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
			
			for (int i=0; i < predictions.length; i++) {
				double pred=w2[this.h2];

		    	
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[this.columndimension * this.h1 + j];
	             }
	             
		    	
               for (int d=0; d <this.columndimension; d++){
              	 double v=data[i][d];
              	 	if ( v == 0){
	                    continue;
              	 		}
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[d * this.h1 + j] * v;
		                 }
	               	 
               }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[this.h1 * this.h2 + s];
		             
		             for (int j=0; j < this.h1; j++){

		                 z2[s] += w1[j * this.h2 + s] * z1[j];
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }		             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         

		         predictions[i][0]=pred;
			}
		return predictions;
	}
	@Override
	public double[][] predict2d(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][1];
		
	       //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
			
			for (int i=0; i < predictions.length; i++) {
				double pred=w2[this.h2];

		    	
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[this.columndimension * this.h1 + j];
	             }
	             
		    	
               for (int d=0; d <this.columndimension; d++){
              	 double v=data.GetElement(i, d);
              	 	if ( v == 0){
	                    continue;
              	 		}
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[d * this.h1 + j] * v;
		                 }
	               	 
               }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[this.h1 * this.h2 + s];
		             
		             for (int j=0; j < this.h1; j++){

		                 z2[s] += w1[j * this.h2 + s] * z1[j];
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }		             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         
		         predictions[i][0]=pred;
			}
		return predictions;
		
		
	}
	@Override
	public double[][] predict2d(smatrix data) {
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][1];
		// check if data is sorted via row
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

	       //hidden units in the 2nd hidden layer
		double z2 [] = new double [this.h2];
		//hidden units in the 1st hidden layer
		double z1 [] = new double [this.h1];
			
			for (int i=0; i < predictions.length; i++) {
				double pred=w2[this.h2];

		    	
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[this.columndimension * this.h1 + j];
	             }
	             
		    	
		            for (int y=data.indexpile[i]; y<data.indexpile[i+1]; y++ ) {
		            int d=data.mainelementpile[y];
		            double v=data.valuespile[y];
           	 	if ( v == 0){
	                    continue;
           	 		}
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[d * this.h1 + j] * v;
		                 }
	               	 
            }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[this.h1 * this.h2 + s];
		             
		             for (int j=0; j < this.h1; j++){

		                 z2[s] += w1[j * this.h2 + s] * z1[j];
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }		             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         

		         predictions[i][0]=pred;
			}

		return predictions;
		
	}
	@Override
	public double[] predict_Row2d(double[] data) {
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}
	       //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
			
		double pred=w2[this.h2];

    	
         for (int j=0; j < this.h1; j++){
             // starting with the bias in the input layer
             z1[j] = this.w0[this.columndimension * this.h1 + j];
         }
         
    	
       for (int d=0; d <this.columndimension; d++){
      	 double v=data[d];
      	 	if ( v == 0){
                continue;
      	 		}
               	 if (this.usescale){
               		 v=Scaler.transform(v, d);
               	 }
                 for (int j=0; j < this.h1; j++){
                    z1[j] += this.w0[d * this.h1 + j] * v;
                 }
           	 
       }
    	
    	
         for (int j=0; j < this.h1; j++){
             // apply the ReLU activation function to the first level hidden unit
             if (this.connection_nonlinearity.equals(("Relu"))){
             z1[j] = Relu(z1[j]);
             }else if (this.connection_nonlinearity.equals(("Linear"))){
            	 z1[j] = Linear(z1[j]); 
             }else if (this.connection_nonlinearity.equals(("Tanh"))){
            	 z1[j] = Tanh(z1[j]); 
             } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
            	 z1[j] = sigmoid(z1[j]); 
             }
         }		    	
    	
    	 // calculating and adding values of 2nd level hidden units
         for (int s=0; s <this.h2; s++ ){
             // staring with the bias in the 1st hidden layer
             z2[s]= w1[this.h1 * this.h2 + s];
             
             for (int j=0; j < this.h1; j++){

                 z2[s] += w1[j * this.h2 + s] * z1[j];
             }
             
             // apply the ReLU activation function to the 2nd level hidden unit
             if (this.connection_nonlinearity.equals(("Relu"))){
            	 z2[s] = Relu( z2[s]);
             } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
            	 z2[s] = sigmoid( z2[s]); 
             }else if (this.connection_nonlinearity.equals(("Tanh"))){
            	 z2[s] = Tanh( z2[s]); 
             }else if (this.connection_nonlinearity.equals(("Linear"))){
            	 z2[s] = Linear( z2[s]); 
             }		             
             
             pred += this.w2[s] * z2[s];    
         }

		         
		return new double []{pred};
	}
	@Override
	public double[] predict_Row2d(fsmatrix data, int row) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
	 //hidden units in the 2nd hidden layer
     double z2 [] = new double [this.h2];
     //hidden units in the 1st hidden layer
     double z1 [] = new double [this.h1];
			
		double pred=w2[this.h2];

 	
      for (int j=0; j < this.h1; j++){
          // starting with the bias in the input layer
          z1[j] = this.w0[this.columndimension * this.h1 + j];
      }
      
 	
    for (int d=0; d <this.columndimension; d++){
   	 double v=data.GetElement(row, d);
   	 	if ( v == 0){
             continue;
   	 		}
            	 if (this.usescale){
            		 v=Scaler.transform(v, d);
            	 }
              for (int j=0; j < this.h1; j++){
                 z1[j] += this.w0[d * this.h1 + j] * v;
              }
        	 
    }
 	
 	
      for (int j=0; j < this.h1; j++){
          // apply the ReLU activation function to the first level hidden unit
          if (this.connection_nonlinearity.equals(("Relu"))){
          z1[j] = Relu(z1[j]);
          }else if (this.connection_nonlinearity.equals(("Linear"))){
         	 z1[j] = Linear(z1[j]); 
          }else if (this.connection_nonlinearity.equals(("Tanh"))){
         	 z1[j] = Tanh(z1[j]); 
          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
         	 z1[j] = sigmoid(z1[j]); 
          }
      }		    	
 	
 	 // calculating and adding values of 2nd level hidden units
      for (int s=0; s <this.h2; s++ ){
          // staring with the bias in the 1st hidden layer
          z2[s]= w1[this.h1 * this.h2 + s];
          
          for (int j=0; j < this.h1; j++){

              z2[s] += w1[j * this.h2 + s] * z1[j];
          }
          
          // apply the ReLU activation function to the 2nd level hidden unit
          if (this.connection_nonlinearity.equals(("Relu"))){
         	 z2[s] = Relu( z2[s]);
          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
         	 z2[s] = sigmoid( z2[s]); 
          }else if (this.connection_nonlinearity.equals(("Tanh"))){
         	 z2[s] = Tanh( z2[s]); 
          }else if (this.connection_nonlinearity.equals(("Linear"))){
         	 z2[s] = Linear( z2[s]); 
          }		             
          
          pred += this.w2[s] * z2[s];    
      }
      
		         
	return new double []{pred};
	}
	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		 //hidden units in the 2nd hidden layer
	     double z2 [] = new double [this.h2];
	     //hidden units in the 1st hidden layer
	     double z1 [] = new double [this.h1];
				
			double pred=w2[this.h2];

	 	
	      for (int j=0; j < this.h1; j++){
	          // starting with the bias in the input layer
	          z1[j] = this.w0[this.columndimension * this.h1 + j];
	      }
	      
	 	
	    for (int y=start; y <end; y++){
          int d=data.mainelementpile[y];
          double v=data.valuespile[y];	    	

	   	 	if ( v == 0){
	             continue;
	   	 		}
	            	 if (this.usescale){
	            		 v=Scaler.transform(v, d);
	            	 }
	              for (int j=0; j < this.h1; j++){
	                 z1[j] += this.w0[d * this.h1 + j] * v;
	              }
	        	 
	    }
	 	
	 	
	      for (int j=0; j < this.h1; j++){
	          // apply the ReLU activation function to the first level hidden unit
	          if (this.connection_nonlinearity.equals(("Relu"))){
	          z1[j] = Relu(z1[j]);
	          }else if (this.connection_nonlinearity.equals(("Linear"))){
	         	 z1[j] = Linear(z1[j]); 
	          }else if (this.connection_nonlinearity.equals(("Tanh"))){
	         	 z1[j] = Tanh(z1[j]); 
	          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	         	 z1[j] = sigmoid(z1[j]); 
	          }
	      }		    	
	 	
	 	 // calculating and adding values of 2nd level hidden units
	      for (int s=0; s <this.h2; s++ ){
	          // staring with the bias in the 1st hidden layer
	          z2[s]= w1[this.h1 * this.h2 + s];
	          
	          for (int j=0; j < this.h1; j++){

	              z2[s] += w1[j * this.h2 + s] * z1[j];
	          }
	          
	          // apply the ReLU activation function to the 2nd level hidden unit
	          if (this.connection_nonlinearity.equals(("Relu"))){
	         	 z2[s] = Relu( z2[s]);
	          } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	         	 z2[s] = sigmoid( z2[s]); 
	          }else if (this.connection_nonlinearity.equals(("Tanh"))){
	         	 z2[s] = Tanh( z2[s]); 
	          }else if (this.connection_nonlinearity.equals(("Linear"))){
	         	 z2[s] = Linear( z2[s]); 
	          }		             
	          
	          pred += this.w2[s] * z2[s];    
	      }
	      
			         
		return new double []{pred};
	}
	
	
	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (h1<=0){
			throw new IllegalStateException(" The Number of hidden units between input and hidden layer cannot be less/equal to zero" );
		}	
		if (h2<=0){
			throw new IllegalStateException(" The Number of hidden units between the 1st and 2nd hidden layer cannot be less/equal to zero" );
		}	
		if (this.init_values<=0){
			throw new IllegalStateException(" Initial randomization value cannot be less equal to zero as the weights need to have some some initial values " );
		}		
		if ( !this.connection_nonlinearity.equals("Relu") &&  !this.connection_nonlinearity.equals("Sigmoid") && !this.connection_nonlinearity.equals("Tanh")  && !this.connection_nonlinearity.equals("Linear")  ){
			throw new IllegalStateException(" connection_nonlinearity has to be one of Relu,Sigmoid,Tanh or Linear " );	
		}
		if (  !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be one of SGD" );	
		}	
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}	
		if ( this.Objective.equals("QUANTILE") && (this.tau<=0 || this.tau>=1) )  {
			throw new IllegalStateException("For  QUANTILE tau value needs to be in (0,1)" );	
		}			
		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		// make sensible checks on the target data
		if (target==null || target.length!=data.length){
			throw new IllegalStateException(" target array needs to be provided" );
		} 
		if (weights==null) {
			weights=new double [data.length];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
		} else {
			if (weights.length!=data.length){
				throw new DimensionMismatchException(weights.length,data.length);
			}
		}

		//hard copy
		if (copy){
			data= manipulate.copies.copies.Copy(data);
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta
		if (w0!=null && w0.length>=1 ){ // check if a set of betas is already given e.g. threads
			
			if (w0.length!=(this.columndimension + 1) * this.h1){
				throw new IllegalStateException(" The pre-given w0 do not have the same dimension with the current data. e.g " + w0.length + "<> " +  ((this.columndimension + 1) * this.h1));
			}
			if (w1.length!=((this.h1 + 1) * this.h2)){
				throw new IllegalStateException(" The pre-given w1 do not have the same dimension with the hidden units of the first leayer. e.g " + w1.length + "<> " +  (this.h1 + 1) * this.h2);
			}			
			if (w2.length!=this.h2+1){
				throw new IllegalStateException(" The pre-given w2 do not have the same dimension with the hidden units of the second leayer. e.g " + w2.length + "<> " +  (this.h2+1));
			}
			
		} else { //Initialise beta if not given
			w0= new double[(this.columndimension + 1) * this.h1];
			w1= new double[( this.h1 + 1) * this.h2];
			w2= new double[this.h2+1];
		}
		
		/* set initial values */
		for (int j=0; j <w0.length; j++ ){
			w0[j]=(random.nextDouble()-0.5)*this.init_values;
		}
		/* set initial values */
		for (int j=0; j <w1.length; j++ ){
			w1[j]=(random.nextDouble()-0.5)*this.init_values;
		}
		/* set initial values */
		for (int j=0; j <w2.length; j++ ){
			w2[j]=(random.nextDouble()-0.5)*this.init_values;
		}		
		
		/* initialize some additional variables (counters and units) */
		
        //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
        
        // counters for the hidden units and inputs
        double c = 0.;
        double c2 [] = new double [this.h2];        
        double c1 [] = new double [this.h1];  
        double c0 [] = new double [this.columndimension];         
        double dl_dz1_array[]= new double[this.h1];
        int col_product=this.h1 * this.h2;
        int col_h1=this.columndimension * this.h1;
        int cc=0;
        int cc2=0;
        double dl_dy=0.0;
        double dl_dw2=0.0;
        double dl_dz2=0.0;
        double dl_dw1=0.0;
        double dl_dz1 =0.0;
        double dl_dw0 =0.0;
        double e=0.0;
        double v=0.0;
        int h1_col=0;
        
	 if (Type.equals("SGD")){
		 
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
				
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.length; k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.length);
		    	}
	
		    	
		    	double yi=target[i];

		    	
		    	double pred=w2[this.h2];

		    	 cc=col_h1;
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[cc++];
	             }
	             
		    	
                for (int d=0; d <this.columndimension; d++){
               	 v=data[i][d];
            	 if ( v == 0){
	                    continue;
	                }
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		               	 h1_col=d * this.h1;
		               	 
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[h1_col++] * v;
		                 }
	               	 
                }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[col_product + s];
		             cc=s;
		             for (int j=0; j < this.h1; j++){
		                 z2[s] += w1[cc] * z1[j];
		                 cc+=this.h2;
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }	             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         
		         
		       
		         
		        // move to updates
		         
		        e=pred-yi; //error
		        
		        /*
		        if (this.Objective.equals("RMSE")){
	    			 gradient=residual*feature;
	    		 } else if  (this.Objective.equals("MAE")){
	    			 if (residual>0){
	    				 gradient=feature;
	    			 } else if (residual<0){
	    				 gradient=-feature;
	    			 }
	    		 }else if  (this.Objective.equals("QUANTILE")){
	    			 if (residual>0){
	    				 gradient=feature*this.tau;
	    			 } else if (residual<0){
	    				 gradient=-feature*this.tau;
	    			 }
	    		 }
		        */
		        dl_dy=0.0;
		        if  (this.Objective.equals("RMSE")){
		        	  dl_dy = e ;  // dl/dy * (initial learning rate)	
		        }
		        else if  (this.Objective.equals("MAE")){
	    			 if (e>0){
	    				 dl_dy=1;
	    			 } else if (e<0){
	    				 dl_dy=-1;
	    			 }
	    		 }else if  (this.Objective.equals("QUANTILE")){
	    			 if (e>0){
	    				 dl_dy=1*this.tau;
	    			 } else if (e<0){
	    				 dl_dy=-1*this.tau;
	    			 }
	    		 }
    	        // starting with the bias in the 2nd hidden layer
    	        w2[this.h2] -= (dl_dy + this.C * w2[this.h2]) * this.learn_rate / (Math.sqrt(c) + this.smooth);
    	        // update overall counter
    	        c += dl_dy * dl_dy;  
    	        
    	        dl_dz1_array= new double[this.h1];
    	        
    	        for (int s=0; s <this.h2; s++){
    	            // update weights related to non-zero 2nd level hidden units
    	            if (z2[s] == 0){
    	                continue;
    	            }
    	            // update weights between the 2nd hidden units and output
    	            // dl/dw2 = dl/dy * dy/dw2 = dl/dy * z2
    	          
    	            if  (this.Objective.equals("RMSE")){
    	            	  dl_dw2 = dl_dy * z2[s];
    	            }
    	            else if  (this.Objective.equals("MAE")){
	   	    			 if (dl_dy>0){
	   	    				dl_dw2=z2[s];
	   	    			 } else if (dl_dy<0){
	   	    				dl_dw2=-z2[s];
	   	    			 }
   	    		 }else if  (this.Objective.equals("QUANTILE")){
   	    			 if (dl_dy>0){
   	    				dl_dw2= z2[s]*this.tau;
   	    			 } else if (dl_dy<0){
   	    				dl_dw2=- z2[s]*this.tau;
   	    			 }
   	    		 }
    	            
    	            w2[s] -= (dl_dw2 + this.C * this.w2[s]) * this.learn_rate / (Math.sqrt(c2[s]) + this.smooth);

    	            // starting with the bias in the 1st hidden layer
    	            // dl/dz2 = dl/dy * dy/dz2 = dl/dy * w2
    	            //dl_dz2 = dl_dy * this.w2[s];
	    	        if  (this.Objective.equals("RMSE")){
	    	        	dl_dz2 = dl_dy * this.w2[s];
	  	            }
	  	            else if  (this.Objective.equals("MAE")){
		   	    			 if (dl_dy>0){
		   	    				dl_dz2=this.w2[s];
		   	    			 } else if (dl_dy<0){
		   	    				dl_dz2=-this.w2[s];
		   	    			 }
	 	    	    }
	  	            else if  (this.Objective.equals("QUANTILE")){
	 	    			 if (dl_dy>0){
	 	    				dl_dz2= this.w2[s]*this.tau;
	 	    			 } else if (dl_dy<0){
	 	    				dl_dz2=-this.w2[s]*this.tau;
	 	    			 }
 	    		 }    	            
    	            w1[col_product + s] -= (dl_dz2 +this.C * this.w1[col_product + s]) * this.learn_rate /
    	                                               (Math.sqrt(c2[s]) +this.smooth);
    	            
    	            cc=s;
    	            cc2=col_h1;
    	            for (int j=0; j < this.h1; j++){
    	                // update weights realted to non-zero hidden units
    	                if ( z1[j] == 0){ 	                   
    	                    cc+=this.h2;
    	                    cc2++;
    	                    continue;
    	                }
    	                // update weights between the hidden units and output
    	                // dl/dw1 = dl/dz2 * dz2/dw1 = dl/dz2 * z1
    	                //dl_dw1 = dl_dz2 * z1[j];
    	                
    	    	        if  (this.Objective.equals("RMSE")){
    	    	        	dl_dw1 = dl_dz2 * z1[j];
    	  	            }
    	  	            else if  (this.Objective.equals("MAE")){
    		   	    			 if (dl_dz2>0){
    		   	    				dl_dw1=z1[j];
    		   	    			 } else if (dl_dz2<0){
    		   	    				dl_dw1=-z1[j];
    		   	    			 }
    	 	    	    }
    	  	            else if  (this.Objective.equals("QUANTILE")){
    	 	    			 if (dl_dz2>0){
    	 	    				dl_dw1= z1[j]*this.tau;
    	 	    			 } else if (dl_dz2<0){
    	 	    				dl_dw1=-z1[j]*this.tau;
    	 	    			 }
    	  	            }       	                
    	                
    	                
    	                w1[cc] -= (dl_dw1 + C * this.w1[j]) * this.learn_rate / (Math.sqrt(c1[j]) + this.smooth);

    	                // starting with the bias in the input layer
    	                // dl/dz1 = dl/dz2 * dz2/dz1 = dl/dz2 * w1
    	                //dl_dz1 = dl_dz2 * this.w1[cc];
 
     	    	        if  (this.Objective.equals("RMSE")){
     	    	        	dl_dz1 = dl_dz2 *this.w1[cc];
    	  	            }
    	  	            else if  (this.Objective.equals("MAE")){
    		   	    			 if (dl_dz2>0){
    		   	    				dl_dz1=this.w1[cc];
    		   	    			 } else if (dl_dz2<0){
    		   	    				dl_dz1=-this.w1[cc];
    		   	    			 }
    	 	    	    }
    	  	            else if  (this.Objective.equals("QUANTILE")){
    	 	    			 if (dl_dz2>0){
    	 	    				dl_dz1= this.w1[cc]*this.tau;
    	 	    			 } else if (dl_dz2<0){
    	 	    				dl_dz1=-this.w1[cc]*this.tau;
    	 	    			 }
    	  	            }     	                 
    	                 
    	                 
    	                this.w0[cc2] -= (dl_dz1 + this.C * this.w0[cc2++]) * this.learn_rate /
    	                		(Math.sqrt(c1[j]) + this.smooth);
    	                dl_dz1_array[j]+=dl_dz1;
    	                // update counter for the 1st level hidden unit j
    	                c1[j] += dl_dw1 * dl_dw1;
    	                cc+=this.h2;
    	            }
    	            // update counter for the 2nd level hidden unit k
    	            c2[s] += dl_dw2 * dl_dw2;
				}
    	         
		        
    	        
                // update weights related to non-zero input units
                for (int d=0; d <this.columndimension; d++){
                    // update weights between the hidden unit j and input i
                    // dl/dw0 = dl/dz1 * dz/dw0 = dl/dz1 * v
                	v=data[i][d];
                	 if ( v == 0){
 	                    continue;
 	                }
	                	if (this.usescale){
	                		v=Scaler.transform(v, d);
	                	}
	                	h1_col=d * this.h1;
	                	 for (int j=0; j < this.h1; j++){
	                		 
		                	//dl_dw0 = dl_dz1_array[j] * v;
		                	
	    	    	        if  (this.Objective.equals("RMSE")){
	    	    	        	dl_dw0 = dl_dz1_array[j] *v;
	    	  	            }
	    	  	            else if  (this.Objective.equals("MAE")){
	    		   	    			 if (dl_dz1_array[j]>0){
	    		   	    				dl_dw0=v;
	    		   	    			 } else if (dl_dz1_array[j]<0){
	    		   	    				dl_dw0=-v;
	    		   	    			 }
	    	 	    	    }
	    	  	            else if  (this.Objective.equals("QUANTILE")){
	    	 	    			 if (dl_dz1_array[j]>0){
	    	 	    				dl_dw0= v*this.tau;
	    	 	    			 } else if (dl_dz1_array[j]<0){
	    	 	    				dl_dw0=-v*this.tau;
	    	 	    			 }
	    	  	            }     	   
		                	
		                	
		                	this.w0[h1_col] -=
		                			this.h2*(dl_dw0 +this.C * this.w0[h1_col++]) * this.learn_rate / (Math.sqrt(c0[d]) + this.smooth);
		                	 
	                	// update counter for the input i
	                	c0[d] += dl_dw0 * dl_dw0;
	                	 }
                }
		    	
	
		    
		    }
             
		    	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			// end of SGD
		}
		sdataset=null;
		fsdataset=null;
		dataset=null;
		System.gc();
	}
	@Override
	public void fit(fsmatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (h1<=0){
			throw new IllegalStateException(" The Number of hidden units between input and hidden layer cannot be less/equal to zero" );
		}	
		if (h2<=0){
			throw new IllegalStateException(" The Number of hidden units between the 1st and 2nd hidden layer cannot be less/equal to zero" );
		}	
		if (this.init_values<=0){
			throw new IllegalStateException(" Initial randomization value cannot be less equal to zero as the weights need to have some some initial values " );
		}		
		if ( !this.connection_nonlinearity.equals("Relu") &&  !this.connection_nonlinearity.equals("Sigmoid") && !this.connection_nonlinearity.equals("Tanh")  && !this.connection_nonlinearity.equals("Linear")  ){
			throw new IllegalStateException(" connection_nonlinearity has to be one of Relu,Sigmoid,Tanh or Linear " );	
		}
		if (  !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be one of SGD" );	
		}	
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}	
		if ( this.Objective.equals("QUANTILE") && (this.tau<=0 || this.tau>=1) )  {
			throw new IllegalStateException("For  QUANTILE tau value needs to be in (0,1)" );	
		}			
			
		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		// make sensible checks on the target data
		if (target==null || target.length!=data.GetRowDimension()){
			throw new IllegalStateException(" target array needs to be provided" );
		}
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
		}

		//hard copy
		if (copy){
			data= (fsmatrix)(data.Copy());
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta
		if (w0!=null && w0.length>=1 ){ // check if a set of betas is already given e.g. threads
			
			if (w0.length!=(this.columndimension + 1) * this.h1){
				throw new IllegalStateException(" The pre-given w0 do not have the same dimension with the current data. e.g " + w0.length + "<> " +  ((this.columndimension + 1) * this.h1));
			}
			if (w1.length!=((this.h1 + 1) * this.h2)){
				throw new IllegalStateException(" The pre-given w1 do not have the same dimension with the hidden units of the first leayer. e.g " + w1.length + "<> " +  (this.h1 + 1) * this.h2);
			}			
			if (w2.length!=this.h2+1){
				throw new IllegalStateException(" The pre-given w2 do not have the same dimension with the hidden units of the second leayer. e.g " + w2.length + "<> " +  (this.h2+1));
			}
			
		} else { //Initialise beta if not given
			w0= new double[(this.columndimension + 1) * this.h1];
			w1= new double[( this.h1 + 1) * this.h2];
			w2= new double[this.h2+1];
		}
		
		/* set initial values */
		for (int j=0; j <w0.length; j++ ){
			w0[j]=(random.nextDouble()-0.5)*this.init_values;
		}
		/* set initial values */
		for (int j=0; j <w1.length; j++ ){
			w1[j]=(random.nextDouble()-0.5)*this.init_values;
		}
		/* set initial values */
		for (int j=0; j <w2.length; j++ ){
			w2[j]=(random.nextDouble()-0.5)*this.init_values;
		}		
		
		/* initialize some additional variables (counters and units) */
		
        //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
        
        // counters for the hidden units and inputs
        double c = 0.;
        double c2 [] = new double [this.h2];        
        double c1 [] = new double [this.h1];  
        double c0 [] = new double [this.columndimension];         
        double dl_dz1_array[]= new double[this.h1];
        int col_product=this.h1 * this.h2;
        int col_h1=this.columndimension * this.h1;
        int cc=0;
        int cc2=0;
        double dl_dy=0.0;
        double dl_dw2=0.0;
        double dl_dz2=0.0;
        double dl_dw1=0.0;
        double dl_dz1 =0.0;
        double dl_dw0 =0.0;
        double e=0.0;
        double v=0.0;
        int h1_col=0;
	 if (Type.equals("SGD")){
		 
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
				
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
	
		    	
		    	double yi=target[i];
		    	
		    	double pred=w2[this.h2];

		    	 cc=col_h1;
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[cc++];
	             }
	             
		    	
                for (int d=0; d <this.columndimension; d++){
               	 v=data.GetElement(i, d);
            	 if ( v == 0){
	                    continue;
	                }
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		               	 h1_col=d * this.h1;
		               	 
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[h1_col++] * v;
		                 }
	               	 
                }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[col_product + s];
		             cc=s;
		             for (int j=0; j < this.h1; j++){
		                 z2[s] += w1[cc] * z1[j];
		                 cc+=this.h2;
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }	             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         
		         
		         
		         
		        // move to updates
		         e=pred-yi; //error
		        
			        dl_dy=0.0;
			        if  (this.Objective.equals("RMSE")){
			        	  dl_dy = e ;  // dl/dy * (initial learning rate)	
			        }
			        else if  (this.Objective.equals("MAE")){
		    			 if (e>0){
		    				 dl_dy=1;
		    			 } else if (e<0){
		    				 dl_dy=-1;
		    			 }
		    		 }else if  (this.Objective.equals("QUANTILE")){
		    			 if (e>0){
		    				 dl_dy=1*this.tau;
		    			 } else if (e<0){
		    				 dl_dy=-1*this.tau;
		    			 }
		    		 }
	    	        // starting with the bias in the 2nd hidden layer
	    	        w2[this.h2] -= (dl_dy + this.C * w2[this.h2]) * this.learn_rate / (Math.sqrt(c) + this.smooth);
	    	        // update overall counter
	    	        c += dl_dy * dl_dy;  
	    	        
	    	        dl_dz1_array= new double[this.h1];
	    	        
	    	        for (int s=0; s <this.h2; s++){
	    	            // update weights related to non-zero 2nd level hidden units
	    	            if (z2[s] == 0){
	    	                continue;
	    	            }
	    	            // update weights between the 2nd hidden units and output
	    	            // dl/dw2 = dl/dy * dy/dw2 = dl/dy * z2
	    	          
	    	            if  (this.Objective.equals("RMSE")){
	    	            	  dl_dw2 = dl_dy * z2[s];
	    	            }
	    	            else if  (this.Objective.equals("MAE")){
		   	    			 if (dl_dy>0){
		   	    				dl_dw2=z2[s];
		   	    			 } else if (dl_dy<0){
		   	    				dl_dw2=-z2[s];
		   	    			 }
	   	    		 }else if  (this.Objective.equals("QUANTILE")){
	   	    			 if (dl_dy>0){
	   	    				dl_dw2= z2[s]*this.tau;
	   	    			 } else if (dl_dy<0){
	   	    				dl_dw2=- z2[s]*this.tau;
	   	    			 }
	   	    		 }
	    	            
	    	            w2[s] -= (dl_dw2 + this.C * this.w2[s]) * this.learn_rate / (Math.sqrt(c2[s]) + this.smooth);

	    	            // starting with the bias in the 1st hidden layer
	    	            // dl/dz2 = dl/dy * dy/dz2 = dl/dy * w2
	    	            //dl_dz2 = dl_dy * this.w2[s];
		    	        if  (this.Objective.equals("RMSE")){
		    	        	dl_dz2 = dl_dy * this.w2[s];
		  	            }
		  	            else if  (this.Objective.equals("MAE")){
			   	    			 if (dl_dy>0){
			   	    				dl_dz2=this.w2[s];
			   	    			 } else if (dl_dy<0){
			   	    				dl_dz2=-this.w2[s];
			   	    			 }
		 	    	    }
		  	            else if  (this.Objective.equals("QUANTILE")){
		 	    			 if (dl_dy>0){
		 	    				dl_dz2= this.w2[s]*this.tau;
		 	    			 } else if (dl_dy<0){
		 	    				dl_dz2=-this.w2[s]*this.tau;
		 	    			 }
	 	    		 }    	            
	    	            w1[col_product + s] -= (dl_dz2 +this.C * this.w1[col_product + s]) * this.learn_rate /
	    	                                               (Math.sqrt(c2[s]) +this.smooth);
	    	            
	    	            cc=s;
	    	            cc2=col_h1;
	    	            for (int j=0; j < this.h1; j++){
	    	                // update weights realted to non-zero hidden units
	    	                if ( z1[j] == 0){ 	                   
	    	                    cc+=this.h2;
	    	                    cc2++;
	    	                    continue;
	    	                }
	    	                // update weights between the hidden units and output
	    	                // dl/dw1 = dl/dz2 * dz2/dw1 = dl/dz2 * z1
	    	                //dl_dw1 = dl_dz2 * z1[j];
	    	                
	    	    	        if  (this.Objective.equals("RMSE")){
	    	    	        	dl_dw1 = dl_dz2 * z1[j];
	    	  	            }
	    	  	            else if  (this.Objective.equals("MAE")){
	    		   	    			 if (dl_dz2>0){
	    		   	    				dl_dw1=z1[j];
	    		   	    			 } else if (dl_dz2<0){
	    		   	    				dl_dw1=-z1[j];
	    		   	    			 }
	    	 	    	    }
	    	  	            else if  (this.Objective.equals("QUANTILE")){
	    	 	    			 if (dl_dz2>0){
	    	 	    				dl_dw1= z1[j]*this.tau;
	    	 	    			 } else if (dl_dz2<0){
	    	 	    				dl_dw1=-z1[j]*this.tau;
	    	 	    			 }
	    	  	            }       	                
	    	                
	    	                
	    	                w1[cc] -= (dl_dw1 + C * this.w1[j]) * this.learn_rate / (Math.sqrt(c1[j]) + this.smooth);

	    	                // starting with the bias in the input layer
	    	                // dl/dz1 = dl/dz2 * dz2/dz1 = dl/dz2 * w1
	    	                //dl_dz1 = dl_dz2 * this.w1[cc];
	 
	     	    	        if  (this.Objective.equals("RMSE")){
	     	    	        	dl_dz1 = dl_dz2 *this.w1[cc];
	    	  	            }
	    	  	            else if  (this.Objective.equals("MAE")){
	    		   	    			 if (dl_dz2>0){
	    		   	    				dl_dz1=this.w1[cc];
	    		   	    			 } else if (dl_dz2<0){
	    		   	    				dl_dz1=-this.w1[cc];
	    		   	    			 }
	    	 	    	    }
	    	  	            else if  (this.Objective.equals("QUANTILE")){
	    	 	    			 if (dl_dz2>0){
	    	 	    				dl_dz1= this.w1[cc]*this.tau;
	    	 	    			 } else if (dl_dz2<0){
	    	 	    				dl_dz1=-this.w1[cc]*this.tau;
	    	 	    			 }
	    	  	            }     	                 
	    	                 
	    	                 
	    	                this.w0[cc2] -= (dl_dz1 + this.C * this.w0[cc2++]) * this.learn_rate /
	    	                		(Math.sqrt(c1[j]) + this.smooth);
	    	                dl_dz1_array[j]+=dl_dz1;
	    	                // update counter for the 1st level hidden unit j
	    	                c1[j] += dl_dw1 * dl_dw1;
	    	                cc+=this.h2;
	    	            }
	    	            // update counter for the 2nd level hidden unit k
	    	            c2[s] += dl_dw2 * dl_dw2;
					}
	    	         
			        
	    	        
	                // update weights related to non-zero input units
	                for (int d=0; d <this.columndimension; d++){
	                    // update weights between the hidden unit j and input i
	                    // dl/dw0 = dl/dz1 * dz/dw0 = dl/dz1 * v
	                	v=data.GetElement(i, d);
	                	 if ( v == 0){
	 	                    continue;
	 	                }
		                	if (this.usescale){
		                		v=Scaler.transform(v, d);
		                	}
		                	h1_col=d * this.h1;
		                	 for (int j=0; j < this.h1; j++){
		                		 
			                	//dl_dw0 = dl_dz1_array[j] * v;
			                	
		    	    	        if  (this.Objective.equals("RMSE")){
		    	    	        	dl_dw0 = dl_dz1_array[j] *v;
		    	  	            }
		    	  	            else if  (this.Objective.equals("MAE")){
		    		   	    			 if (dl_dz1_array[j]>0){
		    		   	    				dl_dw0=v;
		    		   	    			 } else if (dl_dz1_array[j]<0){
		    		   	    				dl_dw0=-v;
		    		   	    			 }
		    	 	    	    }
		    	  	            else if  (this.Objective.equals("QUANTILE")){
		    	 	    			 if (dl_dz1_array[j]>0){
		    	 	    				dl_dw0= v*this.tau;
		    	 	    			 } else if (dl_dz1_array[j]<0){
		    	 	    				dl_dw0=-v*this.tau;
		    	 	    			 }
		    	  	            }     	   
			                	
			                	
			                	this.w0[h1_col] -=
			                			this.h2*(dl_dw0 +this.C * this.w0[h1_col++]) * this.learn_rate / (Math.sqrt(c0[d]) + this.smooth);
			                	 
		                	// update counter for the input i
		                	c0[d] += dl_dw0 * dl_dw0;
		                	 }
	                }
			    	
		
			    
			    }
	             
			    	
		           //end of while
		            it++; 
		    		if (verbose){
		    					System.out.println("iteration: " + it);
		    				}
				}
				
				// end of SGD
			}
		sdataset=null;
		fsdataset=null;
		dataset=null;
		System.gc();
		}
	
	@Override
	public void fit(smatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}
		if (h1<=0){
			throw new IllegalStateException(" The Number of hidden units between input and hidden layer cannot be less/equal to zero" );
		}	
		if (h2<=0){
			throw new IllegalStateException(" The Number of hidden units between the 1st and 2nd hidden layer cannot be less/equal to zero" );
		}	
		if (this.init_values<=0){
			throw new IllegalStateException(" Initial randomization value cannot be less equal to zero as the weights need to have some some initial values " );
		}		
		if ( !this.connection_nonlinearity.equals("Relu") &&  !this.connection_nonlinearity.equals("Sigmoid") && !this.connection_nonlinearity.equals("Tanh")  && !this.connection_nonlinearity.equals("Linear")  ){
			throw new IllegalStateException(" connection_nonlinearity has to be one of Relu,Sigmoid,Tanh or Linear " );	
		}
		if (  !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be one of SGD" );	
		}
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}	
		if ( this.Objective.equals("QUANTILE") && (this.tau<=0 || this.tau>=1) )  {
			throw new IllegalStateException("For  QUANTILE tau value needs to be in (0,1)" );	
		}			
				
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		// make sensible checks on the target data
		if (target==null || target.length!=data.GetRowDimension()){
			throw new IllegalStateException(" target array needs to be provided" );
		}
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
		}

		//hard copy
		if (copy){
			data= (smatrix)(data.Copy());
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		
		// get sparse data ready
        if (sparse_set==false){
	    if (!data.IsSortedByRow()){
	    	data.convert_type();
	    }

        }
	   
		
		
		//initialise beta
		if (w0!=null && w0.length>=1 ){ // check if a set of betas is already given e.g. threads
			
			if (w0.length!=(this.columndimension + 1) * this.h1){
				throw new IllegalStateException(" The pre-given w0 do not have the same dimension with the current data. e.g " + w0.length + "<> " +  ((this.columndimension + 1) * this.h1));
			}
			if (w1.length!=((this.h1 + 1) * this.h2)){
				throw new IllegalStateException(" The pre-given w1 do not have the same dimension with the hidden units of the first leayer. e.g " + w1.length + "<> " +  (this.h1 + 1) * this.h2);
			}			
			if (w2.length!=this.h2+1){
				throw new IllegalStateException(" The pre-given w2 do not have the same dimension with the hidden units of the second leayer. e.g " + w2.length + "<> " +  (this.h2+1));
			}
			
		} else { //Initialise beta if not given
			w0= new double[(this.columndimension + 1) * this.h1];
			w1= new double[( this.h1 + 1) * this.h2];
			w2= new double[this.h2+1];
		}
		
		/* set initial values */
		for (int j=0; j <w0.length; j++ ){
			w0[j]=(random.nextDouble()-0.5)*this.init_values;
		}
		/* set initial values */
		for (int j=0; j <w1.length; j++ ){
			w1[j]=(random.nextDouble()-0.5)*this.init_values;
		}
		/* set initial values */
		for (int j=0; j <w2.length; j++ ){
			w2[j]=(random.nextDouble()-0.5)*this.init_values;
		}		
		
		/* initialize some additional variables (counters and units) */
		
        //hidden units in the 2nd hidden layer
        double z2 [] = new double [this.h2];
        //hidden units in the 1st hidden layer
        double z1 [] = new double [this.h1];
        
        // counters for the hidden units and inputs
        double c = 0.;
        double c2 [] = new double [this.h2];        
        double c1 [] = new double [this.h1];  
        double c0 [] = new double [this.columndimension];         
        double dl_dz1_array[]= new double[this.h1];
        int col_product=this.h1 * this.h2;
        int col_h1=this.columndimension * this.h1;
        int cc=0;
        int cc2=0;
        double dl_dy=0.0;
        double dl_dw2=0.0;
        double dl_dz2=0.0;
        double dl_dw1=0.0;
        double dl_dz1 =0.0;
        double dl_dw0 =0.0;
        double e=0.0;
        double v=0.0;
        int d=0;
        int h1_col=0;
        
	 if (Type.equals("SGD")){
		 
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
				
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
	
		    	
		    	double yi=target[i];
		    	
		    	double pred=w2[this.h2];

		    	 cc=col_h1;
	             for (int j=0; j < this.h1; j++){
	                 // starting with the bias in the input layer
	                 z1[j] = this.w0[cc++];
	             }
	             
	            for (int y=data.indexpile[i]; y<data.indexpile[i+1]; y++ ) {
	            	d=data.mainelementpile[y];
	            	v=data.valuespile[y];

            	 if ( v == 0){
	                    continue;
	                }
		               	 if (this.usescale){
		               		 v=Scaler.transform(v, d);
		               	 }
		               	 h1_col=d * this.h1;
		               	 
		                 for (int j=0; j < this.h1; j++){
		                    z1[j] += this.w0[h1_col++] * v;
		                 }
	               	 
                }
		    	
		    	
	             for (int j=0; j < this.h1; j++){
	                 // apply the ReLU activation function to the first level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                 z1[j] = Relu(z1[j]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z1[j] = Linear(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z1[j] = Tanh(z1[j]); 
	                 }else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z1[j] = sigmoid(z1[j]); 
	                 }
	             }		    	
		    	
		    	 // calculating and adding values of 2nd level hidden units
		         for (int s=0; s <this.h2; s++ ){
		             // staring with the bias in the 1st hidden layer
		             z2[s]= w1[col_product + s];
		             cc=s;
		             for (int j=0; j < this.h1; j++){
		                 z2[s] += w1[cc] * z1[j];
		                 cc+=this.h2;
		             }
		             
		             // apply the ReLU activation function to the 2nd level hidden unit
	                 if (this.connection_nonlinearity.equals(("Relu"))){
	                	 z2[s] = Relu( z2[s]);
	                 }else if (this.connection_nonlinearity.equals(("Linear"))){
	                	 z2[s] = Linear( z2[s]); 
	                 }else if (this.connection_nonlinearity.equals(("Tanh"))){
	                	 z2[s] = Tanh( z2[s]); 
	                 } else if (this.connection_nonlinearity.equals(("Sigmoid"))){
	                	 z2[s] = sigmoid( z2[s]); 
	                 }	             
		             
		             pred += this.w2[s] * z2[s];    
		         }
		         
		         
		         
		        // move to updates
		        e=pred-yi; //error
		        
		        dl_dy=0.0;
		        if  (this.Objective.equals("RMSE")){
		        	  dl_dy = e ;  // dl/dy * (initial learning rate)	
		        }
		        else if  (this.Objective.equals("MAE")){
	    			 if (e>0){
	    				 dl_dy=1;
	    			 } else if (e<0){
	    				 dl_dy=-1;
	    			 }
	    		 }else if  (this.Objective.equals("QUANTILE")){
	    			 if (e>0){
	    				 dl_dy=1*this.tau;
	    			 } else if (e<0){
	    				 dl_dy=-1*this.tau;
	    			 }
	    		 }
    	        // starting with the bias in the 2nd hidden layer
    	        w2[this.h2] -= (dl_dy + this.C * w2[this.h2]) * this.learn_rate / (Math.sqrt(c) + this.smooth);
    	        // update overall counter
    	        c += dl_dy * dl_dy;  
    	        
    	        dl_dz1_array= new double[this.h1];
    	        
    	        for (int s=0; s <this.h2; s++){
    	            // update weights related to non-zero 2nd level hidden units
    	            if (z2[s] == 0){
    	                continue;
    	            }
    	            // update weights between the 2nd hidden units and output
    	            // dl/dw2 = dl/dy * dy/dw2 = dl/dy * z2
    	          
    	            if  (this.Objective.equals("RMSE")){
    	            	  dl_dw2 = dl_dy * z2[s];
    	            }
    	            else if  (this.Objective.equals("MAE")){
	   	    			 if (dl_dy>0){
	   	    				dl_dw2=z2[s];
	   	    			 } else if (dl_dy<0){
	   	    				dl_dw2=-z2[s];
	   	    			 }
   	    		 }else if  (this.Objective.equals("QUANTILE")){
   	    			 if (dl_dy>0){
   	    				dl_dw2= z2[s]*this.tau;
   	    			 } else if (dl_dy<0){
   	    				dl_dw2=- z2[s]*this.tau;
   	    			 }
   	    		 }
    	            
    	            w2[s] -= (dl_dw2 + this.C * this.w2[s]) * this.learn_rate / (Math.sqrt(c2[s]) + this.smooth);

    	            // starting with the bias in the 1st hidden layer
    	            // dl/dz2 = dl/dy * dy/dz2 = dl/dy * w2
    	            //dl_dz2 = dl_dy * this.w2[s];
	    	        if  (this.Objective.equals("RMSE")){
	    	        	dl_dz2 = dl_dy * this.w2[s];
	  	            }
	  	            else if  (this.Objective.equals("MAE")){
		   	    			 if (dl_dy>0){
		   	    				dl_dz2=this.w2[s];
		   	    			 } else if (dl_dy<0){
		   	    				dl_dz2=-this.w2[s];
		   	    			 }
	 	    	    }
	  	            else if  (this.Objective.equals("QUANTILE")){
	 	    			 if (dl_dy>0){
	 	    				dl_dz2= this.w2[s]*this.tau;
	 	    			 } else if (dl_dy<0){
	 	    				dl_dz2=-this.w2[s]*this.tau;
	 	    			 }
 	    		 }    	            
    	            w1[col_product + s] -= (dl_dz2 +this.C * this.w1[col_product + s]) * this.learn_rate /
    	                                               (Math.sqrt(c2[s]) +this.smooth);
    	            
    	            cc=s;
    	            cc2=col_h1;
    	            for (int j=0; j < this.h1; j++){
    	                // update weights realted to non-zero hidden units
    	                if ( z1[j] == 0){ 	                   
    	                    cc+=this.h2;
    	                    cc2++;
    	                    continue;
    	                }
    	                // update weights between the hidden units and output
    	                // dl/dw1 = dl/dz2 * dz2/dw1 = dl/dz2 * z1
    	                //dl_dw1 = dl_dz2 * z1[j];
    	                
    	    	        if  (this.Objective.equals("RMSE")){
    	    	        	dl_dw1 = dl_dz2 * z1[j];
    	  	            }
    	  	            else if  (this.Objective.equals("MAE")){
    		   	    			 if (dl_dz2>0){
    		   	    				dl_dw1=z1[j];
    		   	    			 } else if (dl_dz2<0){
    		   	    				dl_dw1=-z1[j];
    		   	    			 }
    	 	    	    }
    	  	            else if  (this.Objective.equals("QUANTILE")){
    	 	    			 if (dl_dz2>0){
    	 	    				dl_dw1= z1[j]*this.tau;
    	 	    			 } else if (dl_dz2<0){
    	 	    				dl_dw1=-z1[j]*this.tau;
    	 	    			 }
    	  	            }       	                
    	                
    	                
    	                w1[cc] -= (dl_dw1 + C * this.w1[j]) * this.learn_rate / (Math.sqrt(c1[j]) + this.smooth);

    	                // starting with the bias in the input layer
    	                // dl/dz1 = dl/dz2 * dz2/dz1 = dl/dz2 * w1
    	                //dl_dz1 = dl_dz2 * this.w1[cc];
 
     	    	        if  (this.Objective.equals("RMSE")){
     	    	        	dl_dz1 = dl_dz2 *this.w1[cc];
    	  	            }
    	  	            else if  (this.Objective.equals("MAE")){
    		   	    			 if (dl_dz2>0){
    		   	    				dl_dz1=this.w1[cc];
    		   	    			 } else if (dl_dz2<0){
    		   	    				dl_dz1=-this.w1[cc];
    		   	    			 }
    	 	    	    }
    	  	            else if  (this.Objective.equals("QUANTILE")){
    	 	    			 if (dl_dz2>0){
    	 	    				dl_dz1= this.w1[cc]*this.tau;
    	 	    			 } else if (dl_dz2<0){
    	 	    				dl_dz1=-this.w1[cc]*this.tau;
    	 	    			 }
    	  	            }     	                 
    	                 
    	                 
    	                this.w0[cc2] -= (dl_dz1 + this.C * this.w0[cc2++]) * this.learn_rate /
    	                		(Math.sqrt(c1[j]) + this.smooth);
    	                dl_dz1_array[j]+=dl_dz1;
    	                // update counter for the 1st level hidden unit j
    	                c1[j] += dl_dw1 * dl_dw1;
    	                cc+=this.h2;
    	            }
    	            // update counter for the 2nd level hidden unit k
    	            c2[s] += dl_dw2 * dl_dw2;
				}
    	         
		        
    	        
                // update weights related to non-zero input units
                for (int b=data.indexpile[i]; b <data.indexpile[i+1]; b++){
                    // update weights between the hidden unit j and input i
                    // dl/dw0 = dl/dz1 * dz/dw0 = dl/dz1 * v
                	d=data.mainelementpile[b];
                	v=data.valuespile[b];
                	 if ( v == 0){
 	                    continue;
 	                }
	                	if (this.usescale){
	                		v=Scaler.transform(v, d);
	                	}
	                	h1_col=d * this.h1;
	                	 for (int j=0; j < this.h1; j++){
	                		 
		                	//dl_dw0 = dl_dz1_array[j] * v;
		                	
	    	    	        if  (this.Objective.equals("RMSE")){
	    	    	        	dl_dw0 = dl_dz1_array[j] *v;
	    	  	            }
	    	  	            else if  (this.Objective.equals("MAE")){
	    		   	    			 if (dl_dz1_array[j]>0){
	    		   	    				dl_dw0=v;
	    		   	    			 } else if (dl_dz1_array[j]<0){
	    		   	    				dl_dw0=-v;
	    		   	    			 }
	    	 	    	    }
	    	  	            else if  (this.Objective.equals("QUANTILE")){
	    	 	    			 if (dl_dz1_array[j]>0){
	    	 	    				dl_dw0= v*this.tau;
	    	 	    			 } else if (dl_dz1_array[j]<0){
	    	 	    				dl_dw0=-v*this.tau;
	    	 	    			 }
	    	  	            }     	   
		                	
		                	
		                	this.w0[h1_col] -=
		                			this.h2*(dl_dw0 +this.C * this.w0[h1_col++]) * this.learn_rate / (Math.sqrt(c0[d]) + this.smooth);
		                	 
	                	// update counter for the input i
	                	c0[d] += dl_dw0 * dl_dw0;
	                	 }
                }
		    	
	
		    
		    }
             
		    	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			// end of SGD
		}
		sdataset=null;
		fsdataset=null;
		dataset=null;
		System.gc();
	}
	@Override
	public String GetType() {
		return "regressor";
	}
	@Override
	public boolean SupportsWeights() {
		return true;
	}
	@Override
	public String GetName() {
		return "singleVanilla2hnn";
	}
	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier:  Regularized 2-hidden-layers feedforward neural network regressor");
		System.out.println("Classes: 2 (Binary)");
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);	
		System.out.println("First layer hidden units: " + this.h1);	
		System.out.println("2nd layer hidden units: " + this.h2);		
		System.out.println("Initial randomizer thresold: " + this.init_values);			
		System.out.println("Connection Nonlinearity: "+ this.connection_nonlinearity);
		System.out.println("Objective is: "+ this.Objective);		
		System.out.println("Tau value: "+ this.tau);		
		System.out.println("smooth: "+ this.smooth);				
		System.out.println("Regularization value: "+ this.C);	
		System.out.println("Training method: "+ this.Type);	
		System.out.println("Maximum Iterations: "+ maxim_Iteration);
		System.out.println("Learning Rate: "+ this.learn_rate);	
		System.out.println("used Scaling: "+ this.usescale);			
		System.out.println("Tolerance: "+ tolerance);
		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (w0==null){
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
		if (w0!=null || w0.length>0){
			return true;
		} else {
		return false;
		}
	}
	@Override
	public boolean IsRegressor() {
		return false;
	}
	@Override
	public boolean IsClassifier() {
		return true;
	}
	@Override
	public void reset() {
		w0=null;
		w1=null;
		w2=null;
		init_values=0.1;
		connection_nonlinearity="Relu";
		C=1.0;
	    h1 =256;
	    h2 =128;  
		Objective="RMSE";
		tau=0.5;
		Type="SGD";
		threads=1;
		maxim_Iteration=-1;
		usescale=true;
		shuffle=true;
		learn_rate=1.0;
		Scaler=null;
		copy=true;
		smooth=0.1;
		seed=1;
		random=null;
		tolerance=0.0001; 
		target=null;
		weights=null;
		verbose=true;
		
	}
	@Override
	public estimator copy() {
		singleVanilla2hnn br = new singleVanilla2hnn();
		br.w0=manipulate.copies.copies.Copy(this.w0.clone());		
		br.w1=manipulate.copies.copies.Copy(this.w1.clone());
		br.w2=manipulate.copies.copies.Copy(this.w2.clone());		
		br.connection_nonlinearity=this.connection_nonlinearity;
		br.h1=this.h1;
		br.h2=this.h2;
		br.init_values=this.init_values;
		br.Objective="RMSE";
		br.tau=0.5;
		br.columndimension=this.columndimension;
		br.C=this.C;
		br.Type=this.Type;
		br.threads=this.threads;
		br.maxim_Iteration=this.maxim_Iteration;
		br.usescale=this.usescale;
		br.shuffle=this.shuffle;
		br.learn_rate=this.learn_rate;
		br.Scaler=this.Scaler;
		br.smooth=this.smooth;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.tolerance=this.tolerance; 
		br.target=manipulate.copies.copies.Copy(this.target.clone());
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
				
				if (metric.equals("C")) {this.C=Double.parseDouble(value);}
				else if (metric.equals("tau")) {this.tau=Double.parseDouble(value);}				
				
				else if (metric.equals("h1")) {this.h1=Integer.parseInt(value);}
				else if (metric.equals("h2")) {this.h2=Integer.parseInt(value);}
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("Objective")) {this.Objective=value;}				
				else if (metric.equals("connection_nonlinearity")) {this.connection_nonlinearity=value;}	
				
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("maxim_Iteration")) {this.maxim_Iteration=Integer.parseInt(value);}
				else if (metric.equals("init_values")) {this.init_values=Double.parseDouble(value);}
				else if (metric.equals("smooth")) {this.smooth=Double.parseDouble(value);}
				else if (metric.equals("usescale")) {this.usescale=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("shuffle")) {this.shuffle=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("learn_rate")) {this.learn_rate=Double.parseDouble(value);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("tolerance ")) {this.tolerance =Double.parseDouble(value);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}		
				
			}
			
		}
		

	}
	
	@Override
	public double [][] predict_proba(double data [][]){
		return predict2d(data);
	}
	@Override
	public double [][] predict_proba(fsmatrix f){
		return predict2d( f);
	}
	@Override
	public double [][] predict_proba(smatrix f){
		return predict2d( f);
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

	 


	
	 
	/* some helper functions */
	/**
	 * 
	 * @param value double value to be converted through a sigmoid transformation 
	 * @return the transformed value
	 */
	public static double sigmoid(double value){
		return 1.0/ (1.0 + Math.exp(-Math.max(Math.min(value, 35.0), -35.0))); 
		//return 1.0/ (1.0 + Math.exp(-value)); 
	}
	/**
	 * 
	 * @param value double value to be converted through a Relu transformation 
	 * @return the transformed value
	 */
	public static double Relu(double value){
		return Math.max(0.0, value); 
	}	
	/**
	 * 
	 * @param value double value to be converted through a Linear or no transformation 
	 * @return the untransformed value
	 */
	public static double Linear(double value){
		return value; 
	}	
	/**
	 * 
	 * @param value double value to be converted through a Hyperbolic Tangent transformation 
	 * @return the transformed value
	 */
	public static double Tanh(double value){
		return (Math.exp(Math.max(Math.min(value, 35.0), -35.0)) -Math.exp(-Math.max(Math.min(value, 35.0), -35.0)))/
				(Math.exp(Math.max(Math.min(value, 35.0), -35.0)) +Math.exp(-Math.max(Math.min(value, 35.0), -35.0))); 
	}

	
	@Override
	public scaler ReturnScaler() {
		return this.Scaler;
	}
	
	@Override
	public void setScaler(scaler sc) {
		this.Scaler=sc;
		
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
	
	@Override
	public void AddClassnames(String names[]){
		//none
	}
	
	@Override
	public void set_target(fsmatrix fstarget){
		
	}
	
}
