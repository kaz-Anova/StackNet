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

package ml.LibFm;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import preprocess.scaling.scaler;
import preprocess.scaling.maxscaler;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;

/**
 * @author marios
 *<p> class to implement LibFm , assuming the target is a binary variable<p>
 */
public class binaryLibFm implements estimator,classifier,Runnable {

	private static final long serialVersionUID = 830529127388893394L;
	/**
	 * Private method for when this class is used in a multinomial context to avoid re-sorting each time (for each class)
	 */
	private boolean sparse_set=false;
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;
	/**
	 * Regularization value for the latent features
	 */
	public double C2=1.0;
	/**
	 * Type of algorithm to use. It has to be SGD for now.
	 */
	public String Type="SGD";

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
	/**
	 * Initialise values of the latent features with values between[0,init_values)
	 */
	public double init_values=0.1;
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
	 * where the coefficients are held
	 */
	private double betas[];
	/**
	 * This will hold the latent features to encapsulate the 2d interactions among the variables
	 * This will also hold the sum of past gradients to control for the learning rate
	 */
	private double[] latent_features;
	 /**
	  * Initial Learning smooth value Rate
	  */
	 public double smooth=1.0;	 
	 /**
	  * Number of latent features to use. Defaults to 10
	  */
	 public int lfeatures=4;
	/**
	 * The constant value
	 */
	private double[] constant;
	/**
	 * How many predictors the model has
	 */
	private int columndimension=0;
	//return number of predictors in the model
	public int get_predictors(){
		return columndimension;
	}
	
	/**
	 * 
	 * @param need_sort : Whether we want to avoid sorting (used internally for multinomial models)
	 */
	public void set_sparse_indicator(boolean need_sort){
		this.sparse_set=true;
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
	 * Default constructor for Binary Logistic Regression
	 */
	public binaryLibFm(){}
	
	/**
	 * Default constructor for Binary Logistic Regression with double data
	 */
	public binaryLibFm(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	/**
	 * Default constructor for Binary Logistic Regression with fsmatrix data
	 */
	public binaryLibFm(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for Binary Logistic Regression with smatrix data
	 */
	public binaryLibFm(smatrix data){
		
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
	
	/**
	 * 
	 * @param Betas : Sets initial beta-coefficients' array
	 * @param intercept : Sets initial intercept
	 * @param latent_features:  array of latent features
	 * 
	 */
	
	public void SetBetas(double Betas [], double intercept [], double[] latent_features){
		this.betas= Betas;
		this.constant=intercept;
		this.latent_features=latent_features;
	}
	
	/**
	 * 
	 * @return the HashMap of that holds the latent features
	 */
	public  double[] GetLatentFeatures(){
		
		if (latent_features==null || latent_features.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}

		return manipulate.copies.copies.Copy(this.latent_features.clone());
	}	
	
	/**
	 * 
	 * @return the betas
	 */
	public double [] Getbetas(){
		if (betas==null || betas.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(betas);
	}

	/**
	 * @return the constant of the model
	 */
	public double  Getcosntant(){

		return constant[0];
	}	
	
	

	@Override
	public double[][] predict_proba(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double predictions[][]= new double [data.length][2];
		for (int s=0; s < predictions.length; s++) {
			
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=0.0;
    		

	        linear_pred=constant[0] ;
	    		
    		
    		for (int i=0; i < columndimension; i++){
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data[s][i], i);
    			}else {
    				current_fetaure=data[s][i];
    			}
    			
    			
    			if (current_fetaure!=0.0){
    				
    				linear_pred+=betas[i]*current_fetaure;
    				
    				
    				for (int j=0; j <lfeatures; j++){
    					double val=latent_features[i*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
    				
    				//end latent features loop
    				}
    			}
    			//end latent features loop	
    		}
    		
    		for (int j=0; j <lfeatures; j++){
    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s][1]=final_product;
    		predictions[s][0]=1-final_product;    		
			}
		return predictions;
	}
     /**
      * 
      * @param data : to be scored
      * @return the probability for the event to be 1
      */
	public double[] predict_single(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double predictions[]= new double [data.length];
			
			for (int s=0; s < predictions.length; s++) {
								
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=0.0;
    		

	        linear_pred=constant[0] ;
	    		
    		
    		for (int i=0; i < columndimension; i++){
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data[s][i], i);
    			}else {
    				current_fetaure=data[s][i];
    			}
    			
    			
    			if (current_fetaure!=0.0){
    				
    				linear_pred+=betas[i]*current_fetaure;

    				
    				for (int j=0; j <lfeatures; j++){
    					double val=latent_features[i*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
    				
    				//end latent features loop
    				}
    			}
    			//end latent features loop	
    		}
    		
    		for (int j=0; j <lfeatures; j++){
    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s]=final_product;
    		
			}
			
		
		return predictions;
	}
	
	@Override
	public double[][] predict_proba(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][2];
		for (int s=0; s < predictions.length; s++) {
			
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=0.0;
    		

	        linear_pred=constant[0] ;
	    		
    		
    		for (int i=0; i < columndimension; i++){
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data.GetElement(s, i), i);
    			}else {
    				current_fetaure=data.GetElement(s, i);
    			}
    			
    			
    			if (current_fetaure!=0.0){
    				
    				linear_pred+=betas[i]*current_fetaure;

    				
    				for (int j=0; j <lfeatures; j++){
    					double val=latent_features[i*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
    				//end latent features loop
    				}
    			}
    			//end latent features loop	
    		}
    		
    		for (int j=0; j <lfeatures; j++){
    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s][1]=final_product;
    		predictions[s][0]=1-final_product;
    		
			}
		return predictions;
	}

    /**
     * 
     * @param data : to be scored
     * @return the probability for the event to be 1
     */
	public double[] predict_single(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [data.GetRowDimension()];
		for (int s=0; s < predictions.length; s++) {
			
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=0.0;
    		

	        linear_pred=constant[0] ;
	    		
    		
    		for (int i=0; i < columndimension; i++){
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data.GetElement(s, i), i);
    			}else {
    				current_fetaure=data.GetElement(s, i);
    			}
    			
    			
    			if (current_fetaure!=0.0){
    				
    				linear_pred+=betas[i]*current_fetaure;
    				
    				
    				for (int j=0; j <lfeatures; j++){
    					
    					double val=latent_features[i*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
    				
    				//end latent features loop
    				}
    			}
    			//end latent features loop	
    		}
    		
    		for (int j=0; j <lfeatures; j++){
    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s]=final_product;
    		
			}
		return predictions;
	}
	
	@Override
	public double[][] predict_proba(smatrix data) {
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[][]= new double [data.GetRowDimension()][2];
		// check if data is sorted via row
		if (!data.IsSortedByRow()){
			data.convert_type();
		}
		


		
		for (int s=0; s < predictions.length; s++) {
			
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=constant[0];
	    			
    		for (int j=data.indexpile[s]; j <data.indexpile[s+1]; j++){
				int column=data.mainelementpile[j];
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data.valuespile[j], column);
    			}else {
    				current_fetaure=data.valuespile[j];
    			} 				
    				linear_pred+=betas[column]*current_fetaure;

    				
    				for (int f=0; f <lfeatures; f++){
    					double val=latent_features[column*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
    				
    				//end latent features loop
    				}
    			
    			//end latent features loop	
    		}
    		
    		for (int f=0; f <lfeatures; f++){
    			productf+=((sumone[f]*sumone[f])-sumtwo[f]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s][1]=final_product;
    		predictions[s][0]=1-final_product;
    		
			}

		return predictions;
	}
	
	public double[] predict_single(smatrix data) {
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

	
		
		for (int s=0; s < predictions.length; s++) {
			
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=constant[0];
	    			
    		for (int j=data.indexpile[s]; j <data.indexpile[s+1]; j++){
				int column=data.mainelementpile[j];
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data.valuespile[j], column);
    			}else {
    				current_fetaure=data.valuespile[j];
    			} 				
    				linear_pred+=betas[column]*current_fetaure;
    				
    				
    				for (int f=0; f <lfeatures; f++){
    					double val=latent_features[column*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
    				
    				//end latent features loop
    				}
    			
    			//end latent features loop	
    		}
    		
    		for (int f=0; f <lfeatures; f++){
    			productf+=((sumone[f]*sumone[f])-sumtwo[f]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s]=final_product;

    		
			}

		return predictions;
	}	
	

	@Override
	public double[] predict_probaRow(double[] data) {
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.length);	
		}
		double predictions[]= new double [2];
		
		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
		

        linear_pred=constant[0] ;
    			
		for (int i=0; i < columndimension; i++){
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data[i], i);
			}else {
				current_fetaure=data[i];
			}
			
			
			if (current_fetaure!=0.0){
				
				linear_pred+=betas[i]*current_fetaure;

				
				for (int j=0; j <lfeatures; j++){
					double val=latent_features[i*this.lfeatures + j];    					
					sumone[j]+=val*current_fetaure;
    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
				
				//end latent features loop
				}
			}
			//end latent features loop	
		}
		
		for (int j=0; j <lfeatures; j++){
			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions[1]=final_product;
		predictions[0]=1-final_product;
		return predictions;
		

	}

	@Override
	public double[] predict_probaRow(fsmatrix data, int row) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [2];
		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
		

        linear_pred=constant[0] ;
    			
		for (int i=0; i < columndimension; i++){
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data.GetElement(row, i), i);
			}else {
				current_fetaure=data.GetElement(row, i);
			}
			
			
			if (current_fetaure!=0.0){
				
				linear_pred+=betas[i]*current_fetaure;

				
				for (int j=0; j <lfeatures; j++){
					double val=latent_features[i*this.lfeatures + j];    					
					sumone[j]+=val*current_fetaure;
    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
				
				//end latent features loop
				}
			}
			//end latent features loop	
		}
		
		for (int j=0; j <lfeatures; j++){
			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions[1]=final_product;
		predictions[0]=1-final_product;
		return predictions;
	}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [2];
		
		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
        linear_pred=constant[0] ;
    			
		for (int j=start; j < end; j++){
			int column=data.mainelementpile[j];
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data.valuespile[j], column);
			}else {
				current_fetaure=data.valuespile[j];
			}
				
				linear_pred+=betas[column]*current_fetaure;

				
				for (int f=0; f <lfeatures; f++){
					double val=latent_features[column*this.lfeatures + j];    					
					sumone[j]+=val*current_fetaure;
    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
				
				//end latent features loop
				}
			
			//end latent features loop	
		}
		
		for (int f=0; f <lfeatures; f++){
			productf+=((sumone[f]*sumone[f])-sumtwo[f]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions[1]=final_product;
		predictions[0]=1-final_product;
		
		return predictions;
	}

	@Override
	public double[] predict(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions[]= new double [data.GetRowDimension()];
		for (int s=0; s < predictions.length; s++) {
			
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=0.0;
    		

	        linear_pred=constant[0] ;
	    		
    		
    		for (int i=0; i < columndimension; i++){
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data.GetElement(s, i), i);
    			}else {
    				current_fetaure=data.GetElement(s, i);
    			}
    			
    			
    			if (current_fetaure!=0.0){
    				
    				linear_pred+=betas[i]*current_fetaure;
    				
    				
    				for (int j=0; j <lfeatures; j++){
    					double val=latent_features[i*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
    				//end latent features loop
    				}
    			}
    			//end latent features loop	
    		}
    		
    		for (int j=0; j <lfeatures; j++){
    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s]=(final_product >= 0.5) ? 1.0 :0.0 ;
    		
			}
		return predictions;
	}

	@Override
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


		
	for (int s=0; s < predictions.length; s++) {
			
    		double sumone []= new double[lfeatures];
    		double sumtwo []= new double[lfeatures];
    		double productf=0.0;
    		double linear_pred=constant[0];
	    			
    		for (int j=data.indexpile[s]; j <data.indexpile[s+1]; j++){
				int column=data.mainelementpile[j];
    			double current_fetaure=0.0;
    			if(usescale && Scaler!=null) {
    			 current_fetaure=Scaler.transform(data.valuespile[j], column);
    			}else {
    				current_fetaure=data.valuespile[j];
    			} 				
    				linear_pred+=betas[column]*current_fetaure;
    				
    				
    				for (int f=0; f <lfeatures; f++){
    					double val=latent_features[column*this.lfeatures + j];    					
    					sumone[j]+=val*current_fetaure;
	    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;	
    				
    				//end latent features loop
    				}
    			
    			//end latent features loop	
    		}
    		
    		for (int f=0; f <lfeatures; f++){
    			productf+=((sumone[f]*sumone[f])-sumtwo[f]);
    		}
    		

    		//calculate the final product
    		// the final prediction
    		double final_product =(linear_pred+productf/2.0);
    		
    		//convert to probability
    		final_product= 1. / (1. + Math.exp(-final_product));
    		predictions[s]=(final_product >= 0.5) ? 1.0 :0.0 ;

    		
			}

		return predictions;
	}

	@Override
	public double [] predict(double[][] data) {
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double predictions[]= new double [data.length];
		
		for (int s=0; s < predictions.length; s++) {
							
		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
		

        linear_pred=constant[0] ;
    		
		
		for (int i=0; i < columndimension; i++){
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data[s][i], i);
			}else {
				current_fetaure=data[s][i];
			}
			
			
			if (current_fetaure!=0.0){
				
				linear_pred+=betas[i]*current_fetaure;

				
				for (int j=0; j <lfeatures; j++){
					double val=latent_features[i*this.lfeatures + j];    					
					sumone[j]+=val*current_fetaure;
    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
				
				//end latent features loop
				}
			}
			//end latent features loop	
		}
		
		for (int j=0; j <lfeatures; j++){
			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions[s]=(final_product >= 0.5) ? 1.0 :0.0 ;
		
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
		double predictions=0.0;
		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
		

        linear_pred=constant[0] ;
    		
		
		for (int i=0; i < columndimension; i++){
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data[i], i);
			}else {
				current_fetaure=data[i];
			}
			
			
			if (current_fetaure!=0.0){
				
				linear_pred+=betas[i]*current_fetaure;

				
				for (int j=0; j <lfeatures; j++){
					double val=latent_features[i*this.lfeatures + j];    					
					sumone[j]+=val*current_fetaure;
    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
				
				//end latent features loop
				}
			}
			//end latent features loop	
		}
		
		for (int j=0; j <lfeatures; j++){
			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions=(final_product >= 0.5) ? 1.0 :0.0 ;
		
		return predictions;
	}

	@Override
	public double predict_Row(fsmatrix data, int row) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions=0.0;
		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
		

        linear_pred=constant[0] ;
    			
		for (int i=0; i < columndimension; i++){
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data.GetElement(row, i), i);
			}else {
				current_fetaure=data.GetElement(row, i);
			}
			
			
			if (current_fetaure!=0.0){
				
				linear_pred+=betas[i]*current_fetaure;
				
				
				for (int j=0; j <lfeatures; j++){
					
					double val=latent_features[i*this.lfeatures + j];    					
					sumone[j]+=val*current_fetaure;
    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
				
				//end latent features loop
				}
			}
			//end latent features loop	
		}
		
		for (int j=0; j <lfeatures; j++){
			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions=(final_product >= 0.5) ? 1.0 :0.0 ;

		return predictions;
	}

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double predictions= 0.0;
		
		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
        linear_pred=constant[0] ;
    			
		for (int j=start; j < end; j++){
			int column=data.mainelementpile[j];
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data.valuespile[j], column);
			}else {
				current_fetaure=data.valuespile[j];
			}
				
				linear_pred+=betas[column]*current_fetaure;

				
				for (int f=0; f <lfeatures; f++){
					
					double val=latent_features[column*this.lfeatures + j];    					
					sumone[j]+=val*current_fetaure;
    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
				
				//end latent features loop
				}
			
			//end latent features loop	
		}
		
		for (int f=0; f <lfeatures; f++){
			productf+=((sumone[f]*sumone[f])-sumtwo[f]);
		}
		

		//calculate the final product
		// the final prediction
		double final_product =(linear_pred+productf/2.0);
		
		//convert to probability
		final_product= 1. / (1. + Math.exp(-final_product));
		predictions=(final_product >= 0.5) ? 1.0 :0.0 ;

		return predictions;
	}
	
	
	
	

	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}

		if ( !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be one of SGD" );	
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
		if (this.smooth<=0){
			this.smooth=0.1; // a high value just in case id cannot converge
		}	
		if (this.init_values<=0){
			this.init_values=0.1; // a high value just in case id cannot converge
		}	
		if (this.lfeatures<=0){
			throw new IllegalStateException(" Number of Latent features has to be higher than 0. You may use linear models instead." );	
		}		
		
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value of the latent features C2 needs to be higher than zero" );
		}		
		
		// make sensible checks on the target data
		if (target==null || target.length!=data.length){
			throw new IllegalStateException(" target array needs to be provided" );
		} else {
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()!=2){
				throw new IllegalStateException(" target array needs to have exactly 2 values: -1 and 1" );	
			}
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		        double f = it.next();
		        if (f!=-1.0 && f!=1.0){
		        	throw new IllegalStateException("target array needs to have values: -1 and 1");
		    }
		}
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
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			
			if (betas.length!=columndimension){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  columndimension);
			}			
			
		} else { //Initialise beta if not given
			betas= new double[columndimension];
			constant= new double[]{0.0};
			latent_features=new  double[lfeatures*this.columndimension];
		}	
		double[] past_gradients= new double [lfeatures*this.columndimension];
		double n []= new double[data[0].length]; // sum of squared gradients		double nc=0;
		double nc=0.0;
		// Initialise latent features
		for (int f=0; f <columndimension;f++ ){
			
			for (int j=0; j < lfeatures; j++){
				latent_features[f*lfeatures + j]=random.nextDouble()+this.init_values;
			} 	
			
		}
		double current_fetaure=0.0;
		 if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;

			//initiali
			// random number generator

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {
				
		    for (int k=0; k < data.length; k++){
		    	int s=k;

		    	if (this.shuffle){
		    	 s=random.nextInt(data.length);
		    	}

		    	double y=0.0;
		    	if (target[s]>0){
		    		y=1.0;
		    	}
	    		double sumone []= new double[lfeatures];
	    		double sumtwo []= new double[lfeatures];
	    		double productf=0.0;

	    		double linear_pred=0.0;
	    		
	    		if (UseConstant){
		    		 linear_pred=constant[0] ;
		    		}
	    		
	    		for (int i=0; i < columndimension; i++){
	    			
	    			current_fetaure=data[s][i];
	    			if (current_fetaure==0){
	    				continue;
	    			}
	    			if (usescale){
	    				current_fetaure=Scaler.transform(current_fetaure,i);
	    			}
	    				
	    				linear_pred+=betas[i]*current_fetaure;
	    				
	    				
	    				
	    				for (int j=0; j <lfeatures; j++){
	    					double val=latent_features[i*this.lfeatures + j];
	    					sumone[j]+=val*current_fetaure;
		    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;	
	    				
	    				//end latent features loop
	    				}

	    			//end latent features loop	
	    		}
	    		
	    		for (int j=0; j <lfeatures; j++){
	    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
	    		}
	    		
	
	    		//calculate the final product
	    		// the final prediction
	    		double final_product =(linear_pred+productf/2.0);
	    		
	    		//convert to probability
	    		final_product= 1. / (1. + Math.exp(-Math.max(Math.min(final_product, 35.), -35.)));
	    		
	    		// compute the residual
	    		double residual=final_product-y;
	    		
		    		
	            //update constant	
	    		//compute gradient for constant
	    		if (UseConstant){
	    		double constant_gradient=residual+ C*constant[0];
	            // update sum of squared gradients for constant		
	    		nc+=constant_gradient*constant_gradient;
	    		constant[0] -= this.learn_rate * ( constant_gradient ) /Math.sqrt(nc+smooth);
	    		}		    				    			
    			for (int i=0; i < columndimension; i++){
    					//beta updates
    					 current_fetaure=data[s][i];
		    			if (current_fetaure==0){
		    				continue;
		    			}
    	    			if (usescale){
    	    				current_fetaure=Scaler.transform(current_fetaure,i);
    	    			}
	    					double beta_gradient=residual*current_fetaure+ C*betas[i];
	    					n[i]+=beta_gradient*beta_gradient;
	    					betas[i]-=this.learn_rate* (beta_gradient)/Math.sqrt(n[i]+smooth);
	    					
	         				 if (Math.abs(betas[i])>iteration_tol){
								 iteration_tol= Math.abs(betas[i]);
							 }

	    					
	    					for (int f=0; f < lfeatures; f++){
	    						double value=latent_features[i*this.lfeatures + f];
	    						double factorgradient=residual*(current_fetaure*(sumone[f] -(value*current_fetaure*current_fetaure))) + C2*value  ;
	    						past_gradients[i*this.lfeatures + f]+=factorgradient*factorgradient;
	    						latent_features[i*this.lfeatures + f]-=this.learn_rate* (factorgradient)/Math.sqrt(past_gradients[i*this.lfeatures + f]+smooth);
	    					}
	    					

    			}
		    		        
		    }
		    	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			past_gradients=null;
			n=null;
			
			// end of SGD
		
		}
			sdataset=null;
			fsdataset=null;
			dataset=null;
			System.gc();
		 
	}
	@Override
	public void fit(fsmatrix data) {
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}

		if ( !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be SGD " );	
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
		if (this.smooth<=0){
			this.smooth=0.1; // a high value just in case id cannot converge
		}	
		if (this.init_values<=0){
			this.init_values=0.1; // a high value just in case id cannot converge
		}	
		if (this.lfeatures<=0){
			throw new IllegalStateException(" Number of Latent features has to be higher than 0. You may use linear models instead." );	
		}		
		
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value of the latent features C2 needs to be higher than zero" );
		}		
		
		// make sensible checks on the target data
		if (target==null || target.length!=data.GetRowDimension()){
			throw new IllegalStateException(" target array needs to be provided" );
		} else {
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()!=2){
				throw new IllegalStateException(" target array needs to have exactly 2 values: -1 and 1" );	
			}
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		        double f = it.next();
		        if (f!=-1.0 && f!=1.0){
		        	throw new IllegalStateException("target array needs to have values: -1 and 1");
		    }
		}
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
			data= (fsmatrix) data.Copy();
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
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			
			if (betas.length!=columndimension){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  columndimension);
			}

			
		} else { //Initialise beta if not given

			betas= new double[columndimension];
			constant= new double[]{0.0};
			latent_features=new  double[lfeatures*this.columndimension];
		}	
		double[] past_gradients= new double [lfeatures*this.columndimension];
		double n []= new double[data.GetColumnDimension()]; // sum of squared gradients		double nc=0;
		double nc=0;
		double current_fetaure=0.0;
		// Initialise latent features
		for (int f=0; f <columndimension;f++ ){
			for (int j=0; j < lfeatures; j++){
				latent_features[f*lfeatures + j]=random.nextDouble()+this.init_values;
			} 		
		}
		
		 if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;

			//initiali
			// random number generator

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {
				
		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int s=k;

		    	if (this.shuffle){
		    	 s=random.nextInt(data.GetRowDimension());
		    	}

		    	double y=0.0;
		    	if (target[s]>0){
		    		y=1.0;
		    	}
	    		double sumone []= new double[lfeatures];
	    		double sumtwo []= new double[lfeatures];
	    		double productf=0.0;

	    		double linear_pred=0.0;
	    		
	    		if (UseConstant){
		    		 linear_pred=constant[0] ;
		    		}
	    		
	    		for (int i=0; i < columndimension; i++){
	    			
	    			current_fetaure=data.GetElement(s, i);
	    			if (current_fetaure==0){
	    				continue;
	    			}
	    			if (usescale){
	    				current_fetaure=Scaler.transform(current_fetaure,i);
	    			}
	    				
	    				linear_pred+=betas[i]*current_fetaure;
	    				
	    				for (int j=0; j <lfeatures; j++){
	    					double val=latent_features[i*this.lfeatures + j];
	    					sumone[j]+=val*current_fetaure;
		    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;
	    				
	    				//end latent features loop
	    				}
	    			//end latent features loop	
	    		}
	    		
	    		for (int j=0; j <lfeatures; j++){
	    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
	    		}
	    		
	
	    		//calculate the final product
	    		// the final prediction
	    		double final_product =(linear_pred+productf/2.0);
	    		
	    		//convert to probability
	    		final_product= 1. / (1. + Math.exp(-Math.max(Math.min(final_product, 35.), -35.)));
	    		
	    		// compute the residual
	    		double residual=final_product-y;
	    		
		    		
	            //update constant	
	    		//compute gradient for constant
	    		if (UseConstant){
	    		double constant_gradient=residual+ C*constant[0];
	            // update sum of squared gradients for constant		
	    		nc+=constant_gradient*constant_gradient;
	    		constant[0] -= this.learn_rate * ( constant_gradient ) /Math.sqrt(nc+smooth);
	    		}		    				    			
    			for (int i=0; i < columndimension; i++){
    					//beta updates
    					current_fetaure=data.GetElement(s, i);
    	    			if (current_fetaure==0){
    	    				continue;
    	    			}
    	    			if (usescale){
    	    				current_fetaure=Scaler.transform(current_fetaure,i);
    	    			}
	    					double beta_gradient=residual*current_fetaure+ C*betas[i];
	    					n[i]+=beta_gradient*beta_gradient;
	    					betas[i]-=this.learn_rate* (beta_gradient)/Math.sqrt(n[i]+smooth);
	    					
	         				 if (Math.abs(betas[i])>iteration_tol){
								 iteration_tol= Math.abs(betas[i]);
							 }
	         				 
	    					
		    					for (int f=0; f < lfeatures; f++){
		    						double value=latent_features[i*this.lfeatures + f];
		    						double factorgradient=residual*(current_fetaure*(sumone[f] -(value*current_fetaure*current_fetaure))) + C2*value  ;
		    						past_gradients[i*this.lfeatures + f]+=factorgradient*factorgradient;
		    						latent_features[i*this.lfeatures + f]-=this.learn_rate* (factorgradient)/Math.sqrt(past_gradients[i*this.lfeatures + f]+smooth);
		    					}
    			}
		    		        
		    }
		    	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			past_gradients=null;
			n=null;
			
			// end of SGD
		
		}
			sdataset=null;
			fsdataset=null;
			dataset=null;
			System.gc();
	}
	@Override
	public void fit(smatrix data) {
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
		if (C<=0){
			throw new IllegalStateException(" The regularization Value C needs to be higher than zero" );
		}

		if ( !Type.equals("SGD")){
			throw new IllegalStateException(" Type has to be SGD" );	
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
		if (this.smooth<=0){
			this.smooth=0.1; // a high value just in case id cannot converge
		}	
		if (this.init_values<=0){
			this.init_values=0.1; // a high value just in case id cannot converge
		}	
		if (this.lfeatures<=0){
			throw new IllegalStateException(" Number of Latent features has to be higher than 0. You may use linear models instead." );	
		}		
		
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value of the latent features C2 needs to be higher than zero" );
		}		
		
		// make sensible checks on the target data
		if (target==null || target.length!=data.GetRowDimension()){
			throw new IllegalStateException(" target array needs to be provided" );
		} else {
			// check if values only 1 and zero
			HashSet<Double> has= new HashSet<Double> ();
			for (int i=0; i < target.length; i++){
				has.add(target[i]);
			}
			if (has.size()!=2){
				throw new IllegalStateException(" target array needs to have exactly 2 values: -1 and 1" );	
			}
		    for (Iterator<Double> it = has.iterator(); it.hasNext(); ) {
		        double f = it.next();
		        if (f!=-1.0 && f!=1.0){
		        	throw new IllegalStateException("target array needs to have values: -1 and 1");
		    }
		}
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
			data= (smatrix) data.Copy();
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
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			
			if (betas.length!=columndimension){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  columndimension);
			}
			
		} else { //Initialise beta if not given
			betas= new double[columndimension];
			constant= new double[]{0.0};
			latent_features=new  double[lfeatures*this.columndimension];
		}	
		double[] past_gradients= new double [lfeatures*this.columndimension];
		
		double n []= new double[data.GetColumnDimension()]; // sum of squared gradients
		double nc=0;
		double current_fetaure=0.0;
		// Initialise latent features
		for (int f=0; f <columndimension;f++ ){

			for (int j=0; j < lfeatures; j++){

				latent_features[f*lfeatures + j]=random.nextDouble()+this.init_values;
			} 		
		}
		
		   if (sparse_set==false){
			    if (!data.IsSortedByRow()){
			    	data.convert_type();
			    }

		        }
	 		   

		
		 if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;

			// random number generator

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {
				
		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int s=k;

		    	if (this.shuffle){
		    	 s=random.nextInt(data.GetRowDimension());
		    	}

		    	double y=0.0;
		    	if (target[s]>0){
		    		y=1.0;
		    	}
	    		double sumone []= new double[lfeatures];
	    		double sumtwo []= new double[lfeatures];
	    		double productf=0.0;

	    		double linear_pred=0.0;
	    		
	    		if (UseConstant){
		    		 linear_pred=constant[0] ;
		    		}
	    		
	    			for (int d =data.indexpile[s]; d <data.indexpile[s+1]; d++) {
	    			int i=data.mainelementpile[d];	    			
	    			current_fetaure=data.valuespile[d];
	    			if (usescale){
	    				current_fetaure=Scaler.transform(current_fetaure,i);
	    			}
	    				
	    				linear_pred+=betas[i]*current_fetaure;
	    				
	    				for (int j=0; j <lfeatures; j++){
	    					double val=latent_features[i*this.lfeatures + j];	    					
	    					sumone[j]+=val*current_fetaure;
		    				sumtwo[j]+=val*val*current_fetaure*current_fetaure;	
	    				
	    				//end latent features loop
	    				}
	    			//end latent features loop	
	    		}
	    		
	    		for (int j=0; j <lfeatures; j++){
	    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
	    		}
	    		
	
	    		//calculate the final product
	    		// the final prediction
	    		double final_product =(linear_pred+productf/2.0);
	    		
	    		//convert to probability
	    		final_product= 1. / (1. + Math.exp(-Math.max(Math.min(final_product, 35.), -35.)));
	    		
	    		// compute the residual
	    		double residual=final_product-y;
	    		
		    		
	            //update constant	
	    		//compute gradient for constant
	    		if (UseConstant){
	    		double constant_gradient=residual+ C*constant[0];
	            // update sum of squared gradients for constant		
	    		nc+=constant_gradient*constant_gradient;
	    		constant[0] -= this.learn_rate * ( constant_gradient ) /Math.sqrt(nc+smooth);
	    		}		    				    			
    			for (int d =data.indexpile[s]; d <data.indexpile[s+1]; d++) {
    					int i=data.mainelementpile[d];	    			
    					current_fetaure=data.valuespile[d];
    					//beta updates
    	    			if (usescale){
    	    				current_fetaure=Scaler.transform(current_fetaure,i);
    	    			}
	    					double beta_gradient=residual*current_fetaure+ C*betas[i];
	    					n[i]+=beta_gradient*beta_gradient;
	    					betas[i]-=this.learn_rate* (beta_gradient)/Math.sqrt(n[i]+smooth);
	    					
	         				 if (Math.abs(betas[i])>iteration_tol){
								 iteration_tol= Math.abs(betas[i]);
							 }
	    					
	    					for (int f=0; f < lfeatures; f++){
	    						double value=latent_features[i*this.lfeatures + f];
	    						double factorgradient=residual*(current_fetaure*(sumone[f] -(value*current_fetaure*current_fetaure))) + C2*value  ;
	    						past_gradients[i*this.lfeatures + f]+=factorgradient*factorgradient;
	    						latent_features[i*this.lfeatures + f]-=this.learn_rate* (factorgradient)/Math.sqrt(past_gradients[i*this.lfeatures + f]+smooth);
	    					}
	    					
    					
    			}
		    		        
		    }
		    	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			past_gradients=null;
			n=null;
			
			// end of SGD
		
		}
			sdataset=null;
			fsdataset=null;
			dataset=null;
			System.gc();
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
		return "binaryLibFm";
	}
	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: Binary Regularized LibFm");
		System.out.println("Classes: 2 (Binary)");
		System.out.println("Supports Weights:  false");
		System.out.println("Column dimension: " + columndimension);		
		System.out.println("Constant in the model: "+ this.UseConstant);
		System.out.println("Regularization L2 value: "+ this.C);	
		System.out.println("Regularization L2 value for latent features : "+ this.C2);
		System.out.println("Smooth value for FTLR: "+ this.smooth);			
		System.out.println("Number of Latent features: "+ this.lfeatures);				
		System.out.println("Initial value range of the latent features: "+ this.init_values);	
		System.out.println("Training method: "+ this.Type);	
		System.out.println("Maximum Iterations: "+ maxim_Iteration);
		System.out.println("Learning Rate: "+ this.learn_rate);	
		System.out.println("used Scaling: "+ this.usescale);			
		System.out.println("Tolerance: "+ tolerance);
		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (betas==null){
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
		if (betas!=null || betas.length>0){
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
		constant=new double []{0.0};
		betas=null;
		latent_features=null;		
				
		C2=1;
		smooth=0.1;
		lfeatures=4;
		init_values=0.1;
		C=1.0;
		Type="SGD";
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
		shuffle=true;
		learn_rate=1.0;
		Scaler=null;
		copy=true;
		columndimension=0;
		seed=1;
		random=null;
		tolerance=0.0001; 
		target=null;
		weights=null;
		verbose=true;
		
	}
	@Override
	public estimator copy() {
		binaryLibFm br = new binaryLibFm();
		br.constant=this.constant;
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		br.latent_features=manipulate.copies.copies.Copy(this.latent_features.clone());
		//HashMap<Integer, double[]> latent_featuress= new  HashMap<Integer, double[]>();

		br.C=this.C;
		br.C2=this.C2;		
		br.smooth=this.smooth;	
		br.lfeatures=this.lfeatures;
		br.init_values=this.init_values;
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
				else if (metric.equals("C2")) {this.C2=Double.parseDouble(value);}
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("UseConstant")) {this.UseConstant=(value.toLowerCase().equals("true")?true:false)   ;}
				else if (metric.equals("maxim_Iteration")) {this.maxim_Iteration=Integer.parseInt(value);}
				else if (metric.equals("lfeatures")) {this.lfeatures=Integer.parseInt(value);}
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


	@Override
	public String[] getclasses() {
		return new String []{"-1","1"};
	}

	@Override
	public int getnumber_of_classes() {
		return 2;
	}

	@Override
	public void AddClassnames(String[] names) {
		//not applicable in binary model=not needed
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

}
