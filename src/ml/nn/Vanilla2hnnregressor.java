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
 *<p> class to implement binary  2-hidden layer ff neural network for regression with multiple-layer output , where each layer is solved as a single problem (similar to a linear regression with multiple output) <p>
 *The implementation is heavily based on the equivalent one in the <a href="https://pypi.python.org/pypi/Kaggler">kaggler</a> package
 <p> There has been some changes to make it quicker that mostly have to do with the adjustement of the leanring rate</p>
 <p> it also supports more functions </p>

 */
public class Vanilla2hnnregressor implements estimator,regressor {

	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;	
	/**
	 * Type of algorithm to use. It has to be SGD
	 */
	public String Type="SGD";

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
	 * True if we want to scale with highest maximum value
	 */
	public boolean scale=false;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 * If we want to use constant in the model
	 */
	public boolean UseConstant=true;
	/**
	 * Maximum number of iterations. -1 for "until convergence"
	 */
	public int maxim_Iteration=10;
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
	 * Scaler to use in case of usescale=true
	 */
	private preprocess.scaling.scaler Scaler;
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
	 * minimal change in the objective function after nth iterations to stop the algorithm
	 */
	public double tolerance=0.0001; 
	
	/**
	 * Type of nonlinearity between layers. It has to be one of  Relu,Linear,Sigmoid,Tanh
	 */
	public String connection_nonlinearity="Relu";	
	/**
	 * Initialise values of the latent features with values between[0,init_values)
	 */
	public double init_values=0.1;
	
    /**number of the 1st level hidden units**/
	
    public int h1 =20;
    
    /**number of the 2nd level hidden units**/
	
    public int h2 =20;  
    
	/***
	 * Smooth valued for better converges
	 */
	public double smooth=0.1;
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
	 * Target variable in 2d double format
	 */	
	public double target2d[][];
	/**
	 * Target variable in fixed-size matrix format
	 */	
	public fsmatrix fstarget;	
	/**
	 * Target variable in sparse matrix format
	 */	
	public smatrix starget;	

	
	/**
	 * int put weights
	 */
	private double W0s[][];
	/**
	 *  weights between 1st and 2nd layers
	 */
	private double W1s[][];
	/**
	 *  weights between 2nd  and output layers
	 */
	private double W2s[][];

	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
	}
	/**
	 * Number of classes
	 */
	private int n_classes=0;
	

	
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
	 * Default constructor for Vanilla2hnnclassifier with no data
	 */
	public Vanilla2hnnregressor(){
	
	}	
	/**
	 * 
	 * @param data : data to create a constructor
	 */
	public Vanilla2hnnregressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * 
	 * @param data : data to create a constructor
	 */
	public Vanilla2hnnregressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * 
	 * @param data : data to create a constructor
	 */
	public Vanilla2hnnregressor(smatrix data){
		
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

	/**
	 * @return Retrieve the number of unique classes
	 */
	public int getnumber_of_classes(){
		return n_classes;
	}

	/**
	 * 
	 * @return the weights between the input and the first hidden layer
	 */
	public double [][] w0s(){
		if (W0s==null || W0s.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(W0s);
	}
	/**
	 * 
	 * @return the weights between  the first and second hidden layer
	 */
	public double [][] w1s(){
		if (W1s==null || W1s.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(W1s);
	}
	/**
	 * 
	 * @return the weights between the second hidden and the output  layer
	 */
	public double [][] w2s(){
		if (W2s==null || W2s.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(W2s);
	}	

	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8617161535854392960L;



	@Override
	public double[][] predict2d(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		//System.out.println(n_classes);
		double predictions[][]= new double [data.length][n_classes];
			
			for (int i=0; i < predictions.length; i++) {

				
		    	  for (int k=0; k<W0s.length; k++) {
		    		    
		   	       //hidden units in the 2nd hidden layer
		    	        double z2 [] = new double [this.h2];
		    	        //hidden units in the 1st hidden layer
		    	        double z1 [] = new double [this.h1];
		    				
		    			double pred=W2s[k][this.h2];

		    			    	
		    		             for (int j=0; j < this.h1; j++){
		    		                 // starting with the bias in the input layer
		    		                 z1[j] = this.W0s[k][this.columndimension * this.h1 + j];
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
		    			                    z1[j] += this.W0s[k][d * this.h1 + j] * v;
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
		    			             z2[s]= W1s[k][this.h1 * this.h2 + s];
		    			             
		    			             for (int j=0; j < this.h1; j++){

		    			                 z2[s] += W1s[k][j * this.h2 + s] * z1[j];
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
		    			             
		    			             pred += this.W2s[k][s] * z2[s];    
		    			         }
		    			         
		    			         pred=sigmoid(pred);

		    		
		    		//convert to probability

		    		predictions[i][k]=pred;

		    	  }
		    	  
		    	  //System.out.println(Arrays.toString(predictions[i]));
	
				
		
		}
		return predictions;
	}

	@Override
	public double[][] predict2d(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		
		//System.out.println(n_classes);
		double predictions[][]= new double [data.GetRowDimension()][n_classes];
			
		
		for (int i=0; i < predictions.length; i++) {

			
	    	  for (int k=0; k<W0s.length; k++) {
	    		    
	   	       //hidden units in the 2nd hidden layer
	    	        double z2 [] = new double [this.h2];
	    	        //hidden units in the 1st hidden layer
	    	        double z1 [] = new double [this.h1];
	    				
	    			double pred=W2s[k][this.h2];

	    			    	
	    		             for (int j=0; j < this.h1; j++){
	    		                 // starting with the bias in the input layer
	    		                 z1[j] = this.W0s[k][this.columndimension * this.h1 + j];
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
	    			                    z1[j] += this.W0s[k][d * this.h1 + j] * v;
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
	    			             z2[s]= W1s[k][this.h1 * this.h2 + s];
	    			             
	    			             for (int j=0; j < this.h1; j++){

	    			                 z2[s] += W1s[k][j * this.h2 + s] * z1[j];
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
	    			             
	    			             pred += this.W2s[k][s] * z2[s];    
	    			         }
	    			         
	    			         pred=sigmoid(pred);

	    		
	    		//convert to probability

	    		predictions[i][k]=pred;
	
	    	  }
	    	  
	    	  //System.out.println(Arrays.toString(predictions[i]));

			
	
	}
	return predictions;
	}

	@Override
	public double[][] predict2d(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (  W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}
		
		double predictions[][]= new double [data.GetRowDimension()][n_classes];		
		for (int i=0; i < predictions.length; i++) {

			
	    	  for (int k=0; k<W0s.length; k++) {
	    		    
	   	       //hidden units in the 2nd hidden layer
	    	        double z2 [] = new double [this.h2];
	    	        //hidden units in the 1st hidden layer
	    	        double z1 [] = new double [this.h1];
	    				
	    			double pred=W2s[k][this.h2];

	    			    	
	    		             for (int j=0; j < this.h1; j++){
	    		                 // starting with the bias in the input layer
	    		                 z1[j] = this.W0s[k][this.columndimension * this.h1 + j];
	    		             }
	    		             
	    			    	
	    	               for (int b=data.indexpile[i]; b <data.indexpile[i+1];b++){
	    	            	   int d=data.mainelementpile[b];
	    	               	   double v=data.valuespile[b];
	    	              	 	if ( v == 0){
	    		                    continue;
	    	              	 		}
	    			               	 if (this.usescale){
	    			               		 v=Scaler.transform(v, d);
	    			               	 }
	    			                 for (int j=0; j < this.h1; j++){
	    			                    z1[j] += this.W0s[k][d * this.h1 + j] * v;
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
	    			             z2[s]= W1s[k][this.h1 * this.h2 + s];
	    			             
	    			             for (int j=0; j < this.h1; j++){

	    			                 z2[s] += W1s[k][j * this.h2 + s] * z1[j];
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
	    			             
	    			             pred += this.W2s[k][s] * z2[s];    
	    			         }
	    			         
	    			         pred=sigmoid(pred);

	    		
	    		//convert to probability

	    		predictions[i][k]=pred;

	    	  }
	    	  

	    	  //System.out.println(Arrays.toString(predictions[i]));

			
	
	}
	return predictions;
	}

	@Override
	public double[] predict_Row2d(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (  W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}
		
		double predictions[]= new double [n_classes];

		
  	  for (int k=0; k<W0s.length; k++) {
  		    
 	       //hidden units in the 2nd hidden layer
  	        double z2 [] = new double [this.h2];
  	        //hidden units in the 1st hidden layer
  	        double z1 [] = new double [this.h1];
  				
  			double pred=W2s[k][this.h2];

  			    	
  		             for (int j=0; j < this.h1; j++){
  		                 // starting with the bias in the input layer
  		                 z1[j] = this.W0s[k][this.columndimension * this.h1 + j];
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
  			                    z1[j] += this.W0s[k][d * this.h1 + j] * v;
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
  			             z2[s]= W1s[k][this.h1 * this.h2 + s];
  			             
  			             for (int j=0; j < this.h1; j++){

  			                 z2[s] += W1s[k][j * this.h2 + s] * z1[j];
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
  			             
  			             pred += this.W2s[k][s] * z2[s];    
  			         }
  			         

  		//convert to probability
  		predictions[k]=pred;

  	  }

		    	  
		
  	  return predictions;
	}

	@Override
	public double[] predict_Row2d(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null || data.GetRow(rows).length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double predictions[]= new double [n_classes];


		
  	    for (int k=0; k<W0s.length; k++) {
  		    

 	       //hidden units in the 2nd hidden layer
  	        double z2 [] = new double [this.h2];
  	        //hidden units in the 1st hidden layer
  	        double z1 [] = new double [this.h1];
  				
  			double pred=W2s[k][this.h2];

  			    	
  		             for (int j=0; j < this.h1; j++){
  		                 // starting with the bias in the input layer
  		                 z1[j] = this.W0s[k][this.columndimension * this.h1 + j];
  		             }
  		             
  			    	
  	               for (int d=0; d <this.columndimension; d++){
  	              	 double v=data.GetElement(rows, d);
  	              	 	if ( v == 0){
  		                    continue;
  	              	 		}
  			               	 if (this.usescale){
  			               		 v=Scaler.transform(v, d);
  			               	 }
  			                 for (int j=0; j < this.h1; j++){
  			                    z1[j] += this.W0s[k][d * this.h1 + j] * v;
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
  			             z2[s]= W1s[k][this.h1 * this.h2 + s];
  			             
  			             for (int j=0; j < this.h1; j++){

  			                 z2[s] += W1s[k][j * this.h2 + s] * z1[j];
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
  			             
  			             pred += this.W2s[k][s] * z2[s];    
  			         }

  		predictions[k]=pred;

  	  }
  	  
		    	  
		
  	  return predictions;

	}

	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (data==null ){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double predictions[]= new double [n_classes];


  	  for (int k=0; k<W0s.length; k++) {
  		    

 	       //hidden units in the 2nd hidden layer
  	        double z2 [] = new double [this.h2];
  	        //hidden units in the 1st hidden layer
  	        double z1 [] = new double [this.h1];
  				
  			double pred=W2s[k][this.h2];

  			    	
  		             for (int j=0; j < this.h1; j++){
  		                 // starting with the bias in the input layer
  		                 z1[j] = this.W0s[k][this.columndimension * this.h1 + j];
  		             }
  		             
  			    	
  	               for (int b=start; b <end; b++){
  	            	   int d=data.mainelementpile[b];
  	            	   double v=data.valuespile[b];
  	              	 	if ( v == 0){
  		                    continue;
  	              	 		}
  			               	 if (this.usescale){
  			               		 v=Scaler.transform(v, d);
  			               	 }
  			                 for (int j=0; j < this.h1; j++){
  			                    z1[j] += this.W0s[k][d * this.h1 + j] * v;
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
  			             z2[s]= W1s[k][this.h1 * this.h2 + s];
  			             
  			             for (int j=0; j < this.h1; j++){

  			                 z2[s] += W1s[k][j * this.h2 + s] * z1[j];
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
  			             
  			             pred += this.W2s[k][s] * z2[s];    
  			         }
  			         

  		predictions[k]=pred;

  	  }
  	   	  
		
  	  return predictions;

	}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (  W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
		}

		if (   n_classes>1) {
		System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
				
		}				

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double predictions[]= new double [data.GetRowDimension()];
		for (int i=0; i < predictions.length; i++) {


		   	       //hidden units in the 2nd hidden layer
	    	        double z2 [] = new double [this.h2];
	    	        //hidden units in the 1st hidden layer
	    	        double z1 [] = new double [this.h1];
	    				
	    			double pred=W2s[0][this.h2];

	    			    	
	    		             for (int j=0; j < this.h1; j++){
	    		                 // starting with the bias in the input layer
	    		                 z1[j] = this.W0s[0][this.columndimension * this.h1 + j];
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
	    			                    z1[j] += this.W0s[0][d * this.h1 + j] * v;
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
	    			             z2[s]= W1s[0][this.h1 * this.h2 + s];
	    			             
	    			             for (int j=0; j < this.h1; j++){

	    			                 z2[s] += W1s[0][j * this.h2 + s] * z1[j];
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
	    			             
	    			             pred += this.W2s[0][s] * z2[s];    
	    			         }
	    			         



	    	   predictions[i]=pred;


		}
	
	return predictions;
	}

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  	
		if (n_classes>1) {
		System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
				
		}			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		
		double predictions[]= new double [data.GetRowDimension()];
		for (int i=0; i < predictions.length; i++) {

		   	       //hidden units in the 2nd hidden layer
	    	        double z2 [] = new double [this.h2];
	    	        //hidden units in the 1st hidden layer
	    	        double z1 [] = new double [this.h1];
	    				
	    			double pred=W2s[0][this.h2];

	    			    	
	    		             for (int j=0; j < this.h1; j++){
	    		                 // starting with the bias in the input layer
	    		                 z1[j] = this.W0s[0][this.columndimension * this.h1 + j];
	    		             }
	    		             
	    			    	
	    	               for (int b=data.indexpile[i]; b <data.indexpile[i+1]; b++){
	    	            	   int d= data.mainelementpile[b];
	    	              	   double v=data.valuespile[b];
	    	              	 	if ( v == 0){
	    		                    continue;
	    	              	 		}
	    			               	 if (this.usescale){
	    			               		 v=Scaler.transform(v, d);
	    			               	 }
	    			                 for (int j=0; j < this.h1; j++){
	    			                    z1[j] += this.W0s[0][d * this.h1 + j] * v;
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
	    			             z2[s]= W1s[0][this.h1 * this.h2 + s];
	    			             
	    			             for (int j=0; j < this.h1; j++){

	    			                 z2[s] += W1s[0][j * this.h2 + s] * z1[j];
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
	    			             
	    			             pred += this.W2s[0][s] * z2[s];    
	    			         }

	    		  predictions[i]=pred;


		}
	
	return predictions;
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  
		if (n_classes>1) {
		System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
				
		}		
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		
		double predictions[]= new double [data.length];
		for (int i=0; i < predictions.length; i++) {

		   	       //hidden units in the 2nd hidden layer
	    	        double z2 [] = new double [this.h2];
	    	        //hidden units in the 1st hidden layer
	    	        double z1 [] = new double [this.h1];
	    				
	    			double pred=W2s[0][this.h2];

	    			    	
	    		             for (int j=0; j < this.h1; j++){
	    		                 // starting with the bias in the input layer
	    		                 z1[j] = this.W0s[0][this.columndimension * this.h1 + j];
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
	    			                    z1[j] += this.W0s[0][d * this.h1 + j] * v;
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
	    			             z2[s]= W1s[0][this.h1 * this.h2 + s];
	    			             
	    			             for (int j=0; j < this.h1; j++){

	    			                 z2[s] += W1s[0][j * this.h2 + s] * z1[j];
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
	    			             
	    			             pred += this.W2s[0][s] * z2[s];    
	    			         }
	    			         


	    		  predictions[i]=pred;


		}
	
	return predictions;
	}

	@Override
	public double predict_Row(double[] row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  	
//		if (n_classes>1) {
//		System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//				
//		}		
	
		if (row==null || row.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (row.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + row.length);	
		}
		
		double predictions=0.0;

			   	       //hidden units in the 2nd hidden layer
		    	        double z2 [] = new double [this.h2];
		    	        //hidden units in the 1st hidden layer
		    	        double z1 [] = new double [this.h1];
		    				
		    			double pred=W2s[0][this.h2];

		    			    	
		    		             for (int j=0; j < this.h1; j++){
		    		                 // starting with the bias in the input layer
		    		                 z1[j] = this.W0s[0][this.columndimension * this.h1 + j];
		    		             }
		    		             
		    			    	
		    	               for (int d=0; d <this.columndimension; d++){
		    	              	 double v=row[d];
		    	              	 	if ( v == 0){
		    		                    continue;
		    	              	 		}
		    			               	 if (this.usescale){
		    			               		 v=Scaler.transform(v, d);
		    			               	 }
		    			                 for (int j=0; j < this.h1; j++){
		    			                    z1[j] += this.W0s[0][d * this.h1 + j] * v;
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
		    			             z2[s]= W1s[0][this.h1 * this.h2 + s];
		    			             
		    			             for (int j=0; j < this.h1; j++){

		    			                 z2[s] += W1s[0][j * this.h2 + s] * z1[j];
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
		    			             
		    			             pred += this.W2s[0][s] * z2[s];    
		    			         }

		    		  predictions=pred;

	
		
		return predictions;
	}

	@Override
	public double predict_Row(fsmatrix f, int row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (  W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  	
//		if (n_classes>1) {
//		System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//				
//		}			
		
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
		
		double predictions=0.0;
	
			   	       //hidden units in the 2nd hidden layer
		    	        double z2 [] = new double [this.h2];
		    	        //hidden units in the 1st hidden layer
		    	        double z1 [] = new double [this.h1];
		    				
		    			double pred=W2s[0][this.h2];

		    			    	
		    		             for (int j=0; j < this.h1; j++){
		    		                 // starting with the bias in the input layer
		    		                 z1[j] = this.W0s[0][this.columndimension * this.h1 + j];
		    		             }
		    		             
		    			    	
		    	               for (int d=0; d <this.columndimension; d++){
		    	              	 double v=f.GetElement(row, d);
		    	              	 	if ( v == 0){
		    		                    continue;
		    	              	 		}
		    			               	 if (this.usescale){
		    			               		 v=Scaler.transform(v, d);
		    			               	 }
		    			                 for (int j=0; j < this.h1; j++){
		    			                    z1[j] += this.W0s[0][d * this.h1 + j] * v;
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
		    			             z2[s]= W1s[0][this.h1 * this.h2 + s];
		    			             
		    			             for (int j=0; j < this.h1; j++){

		    			                 z2[s] += W1s[0][j * this.h2 + s] * z1[j];
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
		    			             
		    			             pred += this.W2s[0][s] * z2[s];    
		    			         }
		    			         
	


		    		  predictions=pred;

	
		
		return predictions;
	}
	

	@Override
	public double predict_Row(smatrix f, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (W0s==null || W0s.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  	
 	
	//if (n_classes>1) {
	//System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
	//
	//}			
		
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
		
		double predictions=0.0;


			   	       //hidden units in the 2nd hidden layer
		    	        double z2 [] = new double [this.h2];
		    	        //hidden units in the 1st hidden layer
		    	        double z1 [] = new double [this.h1];
		    				
		    			double pred=W2s[0][this.h2];

		    			    	
		    		             for (int j=0; j < this.h1; j++){
		    		                 // starting with the bias in the input layer
		    		                 z1[j] = this.W0s[0][this.columndimension * this.h1 + j];
		    		             }
		    		             
		    			    	
		    	               for (int b=start; b <end; b++){
		    	            	   
		    	            	   int d=f.mainelementpile[b];
		    	              	   double v=f.valuespile[b];
		    	              	   
		    	              	 	if ( v == 0){
		    		                    continue;
		    	              	 		}
		    			               	 if (this.usescale){
		    			               		 v=Scaler.transform(v, d);
		    			               	 }
		    			                 for (int j=0; j < this.h1; j++){
		    			                    z1[j] += this.W0s[0][d * this.h1 + j] * v;
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
		    			             z2[s]= W1s[0][this.h1 * this.h2 + s];
		    			             
		    			             for (int j=0; j < this.h1; j++){

		    			                 z2[s] += W1s[0][j * this.h2 + s] * z1[j];
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
		    			             
		    			             pred += this.W2s[0][s] * z2[s];    
		    			         }
		    			         
	
		    		  predictions=pred;
	
		return predictions;
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
		if ( (this.Objective.equals("MAE")|| this.Objective.equals("QUANTILE")) &&  this.Type.equals("Routine") )  {
			throw new IllegalStateException(" MAE is not available for the Routine optimization method" );	
		}		
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
	
		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case it cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.length) && (target2d==null || target2d.length!=data.length) && (fstarget==null || fstarget.GetRowDimension()!=data.length)  && (starget==null || starget.GetRowDimension()!=data.length)  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
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
		n_classes=0;
		if (target!=null){
			n_classes=1;
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		//System.out.println(Arrays.toString(classes));
		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta and constant
		W0s= new double[n_classes][(this.columndimension + 1) * this.h1];
		W1s= new double[n_classes][( this.h1 + 1) * this.h2];
		W2s= new double[n_classes][this.h2+1];
		Thread[] thread_array= new Thread[threads];
		
		if (n_classes==1){
			
			double label []= new double [data.length];
			if (target!=null){
				for (int i=0; i < label.length; i++){
					label[i]=target[i];
				}
			}else if  (target2d!=null){
				for (int i=0; i < label.length; i++){
					label[i]=target2d[i][0];
				}
			}else if  (fstarget!=null){
				for (int i=0; i < label.length; i++){
					label[i]=fstarget.GetElement(i, 0);
				}
			}else if  (starget!=null){
				
				  if (!starget.IsSortedByColumn()){
					  starget.convert_type();
				    }

		            for (int i=starget.indexpile[0]; i <starget.indexpile[1]; i++) {
		                double val = starget.valuespile[i];
		                int ind =starget.mainelementpile[i];
		                label[ind]=val;
		            }
				    
				    
			} else {
				throw new IllegalStateException(" A target array needs to be provided" );
			}
			

		
			singleVanilla2hnn svc = new singleVanilla2hnn(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.verbose=false;
			svc.tau=this.tau;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.h1=this.h1;
			svc.h2=this.h2;		
			svc.connection_nonlinearity=this.connection_nonlinearity;
			svc.init_values=this.init_values;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;

			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}			
			
			svc.set_w0_w1_w2(W0s[0], W1s[0], W2s[0]);
			svc.target=label;
			svc.run();


		}else {
		
		int count_of_live_threads=0;
		int class_passed=0;
		
		for (int n=0; n <n_classes; n++ ){
			double label []= new double [data.length];
			
			
			if  (target2d!=null){
				for (int i=0; i < label.length; i++){
					label[i]=target2d[i][n];
				}
			}else if  (fstarget!=null){
				for (int i=0; i < label.length; i++){
					label[i]=fstarget.GetElement(i,n);
				}
			}else if  (starget!=null){
				
				  if (!starget.IsSortedByColumn()){
					  starget.convert_type();;
				    }

			        
			        

		            for (int i=starget.indexpile[n]; i <starget.indexpile[n+1]; i++) {
		                double val = starget.valuespile[i];
		                int ind =starget.mainelementpile[i];
		                label[ind]=val;
		            }
				    
				    
			} else {
				throw new IllegalStateException(" A target array needs to be provided" );
			}

			singleVanilla2hnn svc = new singleVanilla2hnn(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.verbose=false;
			svc.tau=this.tau;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;	
			svc.h1=this.h1;
			svc.h2=this.h2;		
			svc.connection_nonlinearity=this.connection_nonlinearity;
			svc.init_values=this.init_values;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}				
			svc.set_w0_w1_w2(W0s[n], W1s[n], W2s[n]);
			svc.target=label;
			thread_array[count_of_live_threads]= new Thread(svc);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for target: " + class_passed);
							
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
		if ( (this.Objective.equals("MAE")|| this.Objective.equals("QUANTILE")) &&  this.Type.equals("Routine") )  {
			throw new IllegalStateException(" MAE is not available for the Routine optimization method" );	
		}			
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
		if (h1<=0){
			throw new IllegalStateException(" The Number of hidden units between input and hidden layer cannot be less/equal to zero" );
		}	
		if (h2<=0){
			throw new IllegalStateException(" The Number of hidden units between the 1st and 2nd hidden layer cannot be less/equal to zero" );
		}
		if ( !Type.equals("Routine")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Routine methods" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case it cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (target2d==null || target2d.length!=data.GetRowDimension()) && (fstarget==null || fstarget.GetRowDimension()!=data.GetRowDimension())  && (starget==null || starget.GetRowDimension()!=data.GetRowDimension())  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
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
			data= (fsmatrix) (data.Copy());
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		n_classes=0;
		if (target!=null){
			n_classes=1;
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		//System.out.println(Arrays.toString(classes));
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		W0s= new double[n_classes][(this.columndimension + 1) * this.h1];
		W1s= new double[n_classes][( this.h1 + 1) * this.h2];
		W2s= new double[n_classes][this.h2+1];

		Thread[] thread_array= new Thread[threads];
		
		if (n_classes==1){
			
			double label []= new double [data.GetRowDimension()];
			if (target!=null){
				for (int i=0; i < label.length; i++){
					label[i]=target[i];
				}
			}else if  (target2d!=null){
				for (int i=0; i < label.length; i++){
					label[i]=target2d[i][0];
				}
			}else if  (fstarget!=null){
				for (int i=0; i < label.length; i++){
					label[i]=fstarget.GetElement(i, 0);
				}
			}else if  (starget!=null){
				
				  if (!starget.IsSortedByColumn()){
					  starget.convert_type();
				    }


				    
		            for (int i=starget.indexpile[0]; i <starget.indexpile[1]; i++) {
		                double val = starget.valuespile[i];
		                int ind =starget.mainelementpile[i];
		                label[ind]=val;
		            }
				    
				    
			} else {
				throw new IllegalStateException(" A target array needs to be provided" );
			}
			

		
			singleVanilla2hnn svc = new singleVanilla2hnn(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.verbose=false;
			svc.tau=this.tau;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;	
			svc.h1=this.h1;
			svc.h2=this.h2;		
			svc.connection_nonlinearity=this.connection_nonlinearity;
			svc.init_values=this.init_values;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;

			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}			
			
			svc.set_w0_w1_w2(W0s[0], W1s[0], W2s[0]);
			svc.target=label;
			svc.run();


		}else {
		
		int count_of_live_threads=0;
		int class_passed=0;
		
		for (int n=0; n <n_classes; n++ ){
			double label []= new double [data.GetRowDimension()];
			
			
			if  (target2d!=null){
				for (int i=0; i < label.length; i++){
					label[i]=target2d[i][n];
				}
			}else if  (fstarget!=null){
				for (int i=0; i < label.length; i++){
					label[i]=fstarget.GetElement(i,n);
				}
			}else if  (starget!=null){
				
				  if (!starget.IsSortedByColumn()){
					  starget.convert_type();;
				    }

			        
			        

		            for (int i=starget.indexpile[n]; i <starget.indexpile[n+1]; i++) {
		                double val = starget.valuespile[i];
		                int ind =starget.mainelementpile[i];
		                label[ind]=val;
		            }
				    
				    
			} else {
				throw new IllegalStateException(" A target array needs to be provided" );
			}

			singleVanilla2hnn svc = new singleVanilla2hnn(data);
			svc.smooth=this.smooth;
			svc.Type=this.Type;
			svc.maxim_Iteration=this.maxim_Iteration;
			svc.Objective=this.Objective;
			svc.verbose=false;
			svc.tau=this.tau;
			svc.copy=false;
			svc.usescale=false;
			svc.C=this.C;
			svc.connection_nonlinearity=this.connection_nonlinearity;	
			svc.h1=this.h1;
			svc.h2=this.h2;			
			svc.init_values=this.init_values;	
			svc.seed=this.seed;
			svc.shuffle=this.shuffle;
			svc.learn_rate=this.learn_rate;
			svc.tolerance=this.tolerance;
			if (usescale){
				svc.setScaler(this.Scaler);
				svc.usescale=true;
			}				
			svc.set_w0_w1_w2(W0s[n], W1s[n], W2s[n]);
			svc.target=label;
			thread_array[count_of_live_threads]= new Thread(svc);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for target: " + class_passed);
							
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
		if ( (this.Objective.equals("MAE")|| this.Objective.equals("QUANTILE")) &&  this.Type.equals("Routine") )  {
			throw new IllegalStateException(" MAE is not available for the Routine optimization method" );	
		}		
		if (this.Objective.equals("QUANTILE") && (this.tau<=0. || this.tau>=1.0)){
			throw new IllegalStateException("The 'tau' value in the QUANTILE regression has to be between 0 and 1" );	
		}
		if (h1<=0){
			throw new IllegalStateException(" The Number of hidden units between input and hidden layer cannot be less/equal to zero" );
		}	
		if (h2<=0){
			throw new IllegalStateException(" The Number of hidden units between the 1st and 2nd hidden layer cannot be less/equal to zero" );
		}
		if ( !Type.equals("Routine")  && !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Routine methods" );	
		}		
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case it cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		// make sensible checks on the target data
		if ( (target==null || target.length!=data.GetRowDimension()) && (target2d==null || target2d.length!=data.GetRowDimension()) && (fstarget==null || fstarget.GetRowDimension()!=data.GetRowDimension())  && (starget==null || starget.GetRowDimension()!=data.GetRowDimension())  ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
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
			data= (smatrix) (data.Copy());
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		n_classes=0;
		if (target!=null){
			n_classes=1;
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}
		
		
		if (!data.IsSortedByRow()){
			data.convert_type();
			}

		
		
		
			columndimension=data.GetColumnDimension();
			//initialise beta and constant
			W0s= new double[n_classes][(this.columndimension + 1) * this.h1];
			W1s= new double[n_classes][( this.h1 + 1) * this.h2];
			W2s= new double[n_classes][this.h2+1];

			Thread[] thread_array= new Thread[threads];
			
			if (n_classes==1){
				
				double label []= new double [data.GetRowDimension()];
				if (target!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target[i];
					}
				}else if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][0];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i, 0);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();
					    }


					    
			            for (int i=starget.indexpile[0]; i <starget.indexpile[1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}
				

			
				singleVanilla2hnn svc = new singleVanilla2hnn(data);
				svc.smooth=this.smooth;
				svc.Type=this.Type;
				svc.maxim_Iteration=this.maxim_Iteration;
				svc.Objective=this.Objective;
				svc.verbose=false;
				svc.tau=this.tau;
				svc.copy=false;
				svc.usescale=false;
				svc.C=this.C;
				svc.connection_nonlinearity=this.connection_nonlinearity;	
				svc.h1=this.h1;
				svc.h2=this.h2;			
				svc.init_values=this.init_values;	
				svc.seed=this.seed;
				svc.shuffle=this.shuffle;
				svc.learn_rate=this.learn_rate;
				svc.tolerance=this.tolerance;

				if (usescale){
					svc.setScaler(this.Scaler);
					svc.usescale=true;
				}			
				
				svc.set_w0_w1_w2(W0s[0], W1s[0], W2s[0]);
				svc.target=label;
				svc.run();


			}else {
			
			int count_of_live_threads=0;
			int class_passed=0;
			
			for (int n=0; n <n_classes; n++ ){
				double label []= new double [data.GetRowDimension()];
				
				
				if  (target2d!=null){
					for (int i=0; i < label.length; i++){
						label[i]=target2d[i][n];
					}
				}else if  (fstarget!=null){
					for (int i=0; i < label.length; i++){
						label[i]=fstarget.GetElement(i,n);
					}
				}else if  (starget!=null){
					
					  if (!starget.IsSortedByColumn()){
						  starget.convert_type();;
					    }

				        
				        

			            for (int i=starget.indexpile[n]; i <starget.indexpile[n+1]; i++) {
			                double val = starget.valuespile[i];
			                int ind =starget.mainelementpile[i];
			                label[ind]=val;
			            }
					    
					    
				} else {
					throw new IllegalStateException(" A target array needs to be provided" );
				}

				singleVanilla2hnn svc = new singleVanilla2hnn(data);
				svc.smooth=this.smooth;
				svc.Type=this.Type;
				svc.maxim_Iteration=this.maxim_Iteration;
				svc.Objective=this.Objective;
				svc.verbose=false;
				svc.tau=this.tau;
				svc.copy=false;
				svc.usescale=false;
				svc.C=this.C;	
				svc.h1=this.h1;
				svc.connection_nonlinearity=this.connection_nonlinearity;
				svc.h2=this.h2;			
				svc.init_values=this.init_values;	
				svc.seed=this.seed;
				svc.shuffle=this.shuffle;
				svc.learn_rate=this.learn_rate;
				svc.tolerance=this.tolerance;
				if (usescale){
					svc.setScaler(this.Scaler);
					svc.usescale=true;
				}				
				svc.set_w0_w1_w2(W0s[n], W1s[n], W2s[n]);
				svc.target=label;
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(n_classes)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("fitting for target: " + class_passed);
								
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
		return "Vanilla2hnnregressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor: Regularized 2-hidden-layers feedforward neural network");
		System.out.println("Targets: " + n_classes);
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
		if (W0s==null){
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
		if (W0s!=null || W0s.length>0){
			return true;
		} else {
		return false;
		}
	}

	@Override
	public boolean IsRegressor() {
		return true ;
	}

	@Override
	public boolean IsClassifier() {
		return false;
	}

	@Override
	public void reset() {
		W0s=null;
		W1s=null;
		W2s=null;
		n_classes=0;
		connection_nonlinearity="Relu";
		h1=20;
		h2=20;
		init_values=0.1;
		tau=0.5;
		Objective="RMSE";	
		smooth=0.1;
		C=1.0;
		Type="SGD";
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
		shuffle=true;
		learn_rate=1.0;
		columndimension=0;
		Scaler=null;
		copy=true;
		seed=1;
		random=null;
		tolerance=0.0001; 
		target=null;
		target2d=null;
		fstarget=null;
		starget=null;
		weights=null;
		verbose=true;
		
	}

	@Override
	public estimator copy() {
		Vanilla2hnnregressor br = new Vanilla2hnnregressor();
		br.W0s=manipulate.copies.copies.Copy(this.W0s.clone());
		br.W1s=manipulate.copies.copies.Copy(this.W1s.clone());
		br.W2s=manipulate.copies.copies.Copy(this.W2s.clone());
		br.n_classes=this.n_classes;
		br.connection_nonlinearity=this.connection_nonlinearity;
		br.h1=this.h1;
		br.h2=this.h2;
		br.init_values=this.init_values;
		br.Objective=this.Objective;
		br.tau=this.tau;
		br.C=this.C;
		br.smooth=this.smooth;
		br.Type=this.Type;
		br.columndimension=this.columndimension;
		br.threads=this.threads;
		br.UseConstant=this.UseConstant;
		br.maxim_Iteration=this.maxim_Iteration;
		br.usescale=this.usescale;
		br.shuffle=this.shuffle;
		br.learn_rate=this.learn_rate;
		br.Scaler=this.Scaler;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.tolerance=this.tolerance; 
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.target2d=manipulate.copies.copies.Copy(this.target2d.clone());	
		br.fstarget=(fsmatrix) this.fstarget.Copy();
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
				
				if (metric.equals("C")) {this.C=Double.parseDouble(value);}
				else if (metric.equals("tau")) {this.tau=Double.parseDouble(value);}
				else if (metric.equals("connection_nonlinearity")) {this.connection_nonlinearity=value;}	
				else if (metric.equals("h1")) {this.h1=Integer.parseInt(value);}
				else if (metric.equals("h2")) {this.h2=Integer.parseInt(value);}
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("Objective")) {this.Objective=value;}
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

	/**
	 * 
	 * @param value double value to be converted through a sigmoid transformation 
	 * @return the transformed value
	 */
	public static double sigmoid(double value){
		return 1.0/ (1.0 + Math.exp(-Math.max(Math.min(value, 35.0), -35.0))); 
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
	public void set_target(double data []){
		if (data==null || data.length<=0){
			throw new IllegalStateException("There is nothing to train on" );
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
		this.fstarget=fstarget;
	}
}
