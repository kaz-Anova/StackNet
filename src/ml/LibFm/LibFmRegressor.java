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
import java.util.Random;

import preprocess.scaling.scaler;
import preprocess.scaling.maxscaler;
import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.estimator;
import ml.regressor;

/**
 * 
 * LibFM Regressor class for multiple targets trained with:
 * <ol>
 * <li> SGD "Stochastic Gradient Descent" with adaptive learning Rate (supports L1 and L2)  </li> 
 * </ol>
 */
public class LibFmRegressor implements estimator,regressor {
	
	/**
	 * Type of algorithm to use. It has to be SGD
	 */
	public String Type="SGD";
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=0.01;
	/**
	 * Regularization value for the latent features
	 */
	public double C2=0.01;	
	/**
	 * This will hold the latent features to encapsulate the 2d interactions among the variables
	 * This will also hold the sum of past gradients to control for the learning rate
	 */
	private double[] [] latent_features ;
	 /**
	  * Initial Learning smooth value Rate
	  */
	 public double smooth=1.0;	 
	 /**
	  * Number of latent features to use. Defaults to 4
	  */
	 public int lfeatures=4;
	/**
	 * Initialise values of the latent features with values between[0,init_values)
	 */
	public double init_values=0.1;
	/**
	 * True if we want to scale with highest maximum value
	 */
	public boolean scale=false;
	
	public String Objective="RMSE";	
	/**
	 * quantile value
	 */
	public double tau=0.5;
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
	 * minimal change in coefficients' values after nth iterations to stop the algorithm
	 */
	public double tolerance=0.0001; 
	
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
	 * where the coefficients are held
	 */
	private double betas[][];
	/**
	 * The cosntant value
	 */
	private double constant[];
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
	 * Default constructor for Libfm with no data
	 */
	public LibFmRegressor(){
	
	}	
	public LibFmRegressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	

	public LibFmRegressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}

	public LibFmRegressor(smatrix data){
		
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
	 * Retrieve the number of uniqye classes
	 */
	public int getnumber_of_classes(){
		return n_classes;
	}

	/**
	 * 
	 * @return the betas
	 */
	public double [][] Getbetas(){
		if (betas==null || betas.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return manipulate.copies.copies.Copy(betas);
	}

	/**
	 * @return the constant of the model should be length n_classes=-1
	 */
	public double [] Getcosntant(){

		return manipulate.copies.copies.Copy(constant);
	}	
	
	 /** 
	 * @return the HashMap of that holds the latent features
	 */

	public double[] [] GetLatentFeatures(){
		
		if (latent_features==null || latent_features.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		double[][] lat= new double [this.latent_features.length][latent_features[0].length];
		
		for (int f=0; f < this.latent_features.length; f++){
			
			for (int s=0; s <latent_features[f].length; s++){
				lat[f][s]=latent_features[f][s];
			}
		}

		return lat;
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
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
				
		    	  for (int k=0; k<betas.length; k++) {
		    		    
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data[i][d], d);
		    			}else {
		    				current_fetaure=data[i][d];
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;
		    				
		    				
		    				for (int j=0; j <lfeatures; j++){
		    					

		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;		
		    				
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
		    		
		    	   predictions[i][k]= final_product;
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
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
				
		    	  for (int k=0; k<betas.length; k++) {
		    		    
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data.GetElement(i, d), d);
		    			}else {
		    				current_fetaure=data.GetElement(i, d);
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;
		    				
		    				for (int j=0; j <lfeatures; j++){
		    					
		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
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
		    		

		    		  predictions[i][k]=  final_product;
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
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
			
	    	  for (int k=0; k<betas.length; k++) {
	    		    
	      		double sumone []= new double[lfeatures];
	    		double sumtwo []= new double[lfeatures];
	    		double productf=0.0;
	    		double linear_pred=0.0;
	    		

		        linear_pred=constant[k] ;
		    		
		        for (int s=data.indexpile[i]; s < data.indexpile[i+1]; s++){
	    			int d=data.mainelementpile[s] ;
	    			double current_fetaure=0.0;
	    			if(usescale && Scaler!=null) {
	    			 current_fetaure=Scaler.transform(data.valuespile[s], d);
	    			}else {
	    				current_fetaure=data.valuespile[s];
	    			}
	    				
	    				linear_pred+=betas[k][d]*current_fetaure;
	    				
	    				for (int j=0; j <lfeatures; j++){
	    					

	    					double value=latent_features[k][d*this.lfeatures + j];
	    					sumone[j]+=value*current_fetaure;
		    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
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
	    		

	    		  predictions[i][k]=  final_product;
	    	  }
	    	  //System.out.println(Arrays.toString(predictions[i]));

			
	
	}
	return predictions;
	}

	@Override
	public double[] predict_Row2d(double[] row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  			
		if (row==null || row.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (row.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + row.length);	
		}
		
		double predictions[]= new double [n_classes];

				
		    	  for (int k=0; k<betas.length; k++) {
		    		  
		    		    
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[k] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(row[d], d);
		    			}else {
		    				current_fetaure=row[d];
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[k][d]*current_fetaure;

		    				
		    				for (int j=0; j <lfeatures; j++){

		    					double value=latent_features[k][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
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
		    		

		    		  predictions[k]=  final_product;
		    	  }
		    	  
		
		    	  return predictions;
	}

	@Override
	public double[] predict_Row2d(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
		
  	  for (int k=0; k<betas.length; k++) {

    	double sumone []= new double[lfeatures];
  		double sumtwo []= new double[lfeatures];
  		double productf=0.0;
  		double linear_pred=0.0;
  		

	        linear_pred=constant[k] ;
	    		
  		
  		for (int d=0; d < columndimension; d++){
  			double current_fetaure=0.0;
  			if(usescale && Scaler!=null) {
  			 current_fetaure=Scaler.transform(data.GetElement(rows, d), d);
  			}else {
  				current_fetaure=data.GetElement(rows, d);
  			}
  			
  			if (current_fetaure!=0.0){
  				
  				linear_pred+=betas[k][d]*current_fetaure;
  				
  				for (int j=0; j <lfeatures; j++){

					double value=latent_features[k][d*this.lfeatures + j];
					sumone[j]+=value*current_fetaure;
    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
  				
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
  		

  		  predictions[k]= final_product;
  	  }
  	  

  	  return predictions;
	}

	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
		 for (int k=0; k<betas.length; k++) {

  		double sumone []= new double[lfeatures];
		double sumtwo []= new double[lfeatures];
		double productf=0.0;
		double linear_pred=0.0;
		

        linear_pred=constant[k] ;
    		
        for (int s=start; s <end; s++){
        	
			int d=data.mainelementpile[s] ;
			double current_fetaure=0.0;
			if(usescale && Scaler!=null) {
			 current_fetaure=Scaler.transform(data.valuespile[s], d);
			}else {
				current_fetaure=data.valuespile[s];
			}
				
				linear_pred+=betas[k][d]*current_fetaure;
				
				for (int j=0; j <lfeatures; j++){

					double value=latent_features[k][d*this.lfeatures + j];
					sumone[j]+=value*current_fetaure;
    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
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
		

		  predictions[k]= final_product;
	  }
	  //System.out.println(Arrays.toString(predictions[i]));


return predictions;

	}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
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
		
		double predictions[]= new double [data.GetRowDimension()];
		for (int i=0; i < predictions.length; i++) {

		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[0] ;
			    		
		    		
		    		for (int d=0; d < columndimension; d++){
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data.GetElement(i, d), d);
		    			}else {
		    				current_fetaure=data.GetElement(i, d);
		    			}
		    			
		    			if (current_fetaure!=0.0){
		    				
		    				linear_pred+=betas[0][d]*current_fetaure;
		    				
		    				for (int j=0; j <lfeatures; j++){
	
		    					double value=latent_features[0][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
		    				//end latent features loop
		    				}
		    			}
		    			//end latent features loop	
		    		}
		    		
		    		for (int j=0; j <lfeatures; j++){
		    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
		    		}
		    		
		    		double final_product =(linear_pred+productf/2.0);

	    		  predictions[i]=final_product;

	    	  

		}
	
	return predictions;
	}

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}  	
		if (  n_classes>1) {
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

		    		    

		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

			        linear_pred=constant[0] ;
			    		
			        for (int s=data.indexpile[i]; s < data.indexpile[i+1]; s++){
		    			int d=data.mainelementpile[s] ;
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(data.valuespile[s], d);
		    			}else {
		    				current_fetaure=data.valuespile[s];
		    			}
		    				
		    				linear_pred+=betas[0][d]*current_fetaure;
		    				
		    				
		    				for (int j=0; j <lfeatures; j++){

		    					double value=latent_features[0][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
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
		    		
	
		    		  predictions[i]=final_product;
		    	  
	
			}
		
		return predictions;
	}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					} 
		if (  n_classes>1) {
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

			      		double sumone []= new double[lfeatures];
			    		double sumtwo []= new double[lfeatures];
			    		double productf=0.0;
			    		double linear_pred=0.0;
			    		

				        linear_pred=constant[0] ;
				    		
			    		
			    		for (int d=0; d < columndimension; d++){
			    			double current_fetaure=0.0;
			    			if(usescale && Scaler!=null) {
			    			 current_fetaure=Scaler.transform(data[i][d], d);
			    			}else {
			    				current_fetaure=data[i][d];
			    			}
			    			
			    			if (current_fetaure!=0.0){
			    				
			    				linear_pred+=betas[0][d]*current_fetaure;
			    				
			    				
			    				for (int j=0; j <lfeatures; j++){

			    					double value=latent_features[0][d*this.lfeatures + j];
			    					sumone[j]+=value*current_fetaure;
				    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
			    				//end latent features loop
			    				}
			    			}
			    			//end latent features loop	
			    		}
			    		
			    		for (int j=0; j <lfeatures; j++){
			    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
			    		}
			    		
			    		double final_product =(linear_pred+productf/2.0);
	
		    		  predictions[i]=final_product;
		    	  
	
			}
		
		return predictions;
	}

	@Override
	public double predict_Row(double[] row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					} 
//		if (  n_classes>1) {
//			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//			
//		}		
		if (row==null || row.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (row.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + row.length);	
		}
		

			      		double sumone []= new double[lfeatures];
			    		double sumtwo []= new double[lfeatures];
			    		double productf=0.0;
			    		double linear_pred=0.0;

				        linear_pred=constant[0] ;

			    		for (int d=0; d < columndimension; d++){
			    			double current_fetaure=0.0;
			    			if(usescale && Scaler!=null) {
			    			 current_fetaure=Scaler.transform(row[d], d);
			    			}else {
			    				current_fetaure=row[d];
			    			}
			    			
			    			if (current_fetaure!=0.0){
			    				
			    				linear_pred+=betas[0][d]*current_fetaure;
			    				
			    				for (int j=0; j <lfeatures; j++){

			    					double value=latent_features[0][d*this.lfeatures + j];
			    					sumone[j]+=value*current_fetaure;
				    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
			    				//end latent features loop
			    				}
			    			}
			    			//end latent features loop	
			    		}
			    		
			    		for (int j=0; j <lfeatures; j++){
			    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
			    		}
			    		
			    		double final_product =(linear_pred+productf/2.0);

		    	  	
		return final_product;
	}

	@Override
	public double predict_Row(fsmatrix f, int row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					} 
//		if (  n_classes>1) {
//			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//			
//		}		
		
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
			      		double sumone []= new double[lfeatures];
			    		double sumtwo []= new double[lfeatures];
			    		double productf=0.0;
			    		double linear_pred=0.0;
			    		

				        linear_pred=constant[0] ;
				    		
			    		
			    		for (int d=0; d < columndimension; d++){
			    			double current_fetaure=0.0;
			    			if(usescale && Scaler!=null) {
			    			 current_fetaure=Scaler.transform(f.GetElement(row, d), d);
			    			}else {
			    				current_fetaure=f.GetElement(row, d);
			    			}
			    			
			    			if (current_fetaure!=0.0){
			    				
			    				linear_pred+=betas[0][d]*current_fetaure;
			    				
			    				for (int j=0; j <lfeatures; j++){

			    					double value=latent_features[0][d*this.lfeatures + j];
			    					sumone[j]+=value*current_fetaure;
				    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
			    				//end latent features loop
			    				}
			    			}
			    			//end latent features loop	
			    		}
			    		
			    		for (int j=0; j <lfeatures; j++){
			    			productf+=((sumone[j]*sumone[j])-sumtwo[j]);
			    		}
			    		
			    		double final_product =(linear_pred+productf/2.0);
			    	
		
		return final_product;
	}
	

	@Override
	public double predict_Row(smatrix f, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if ( betas==null || betas.length<=0 || n_classes<1) {
				throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");
					}
//		if (  n_classes>1) {
//			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");
//			
//		}		
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
		
		      		double sumone []= new double[lfeatures];
		    		double sumtwo []= new double[lfeatures];
		    		double productf=0.0;
		    		double linear_pred=0.0;
		    		

		            linear_pred=constant[0] ;
		        		
		            for (int s=start; s <end; s++){
		    			int d=f.mainelementpile[s] ;
		    			double current_fetaure=0.0;
		    			if(usescale && Scaler!=null) {
		    			 current_fetaure=Scaler.transform(f.valuespile[s], d);
		    			}else {
		    				current_fetaure=f.valuespile[s];
		    			}
		    				
		    				linear_pred+=betas[0][d]*current_fetaure;
		    				
		    				for (int j=0; j <lfeatures; j++){

		    					double value=latent_features[0][d*this.lfeatures + j];
		    					sumone[j]+=value*current_fetaure;
			    				sumtwo[j]+=value*value*current_fetaure*current_fetaure;	
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
		    	
		return final_product;
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
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value C2 for the latent features needs to be higher than zero" );
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
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}	
		if ( this.Objective.equals("QUANTILE") && (this.tau<=0 || this.tau>=1) )  {
			throw new IllegalStateException("For  QUANTILE tau value needs to be in (0,1)" );	
		}				
		
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value of the latent features C2 needs to be higher than zero" );
		}		
		if (  !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be SGD" );	
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

		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta and constant
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		latent_features = new double [this.n_classes][this.lfeatures *this.columndimension] ;

		
		double [][] constants=new double[n_classes][1];
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
		

	
		singleLibFm svc = new singleLibFm(data);
		svc.smooth=this.smooth;
		svc.Type=this.Type;
		svc.maxim_Iteration=this.maxim_Iteration;
		svc.Objective=this.Objective;
		svc.verbose=false;
		svc.tau=this.tau;
		svc.copy=false;
		svc.usescale=false;
		svc.C=this.C;
		svc.C2=this.C2;
		svc.lfeatures=this.lfeatures;
		svc.tau=this.tau;			
		svc.init_values=this.init_values;	
		svc.seed=this.seed;
		svc.shuffle=this.shuffle;
		svc.learn_rate=this.learn_rate;
		svc.tolerance=this.tolerance;

		if (usescale){
			svc.setScaler(this.Scaler);
			svc.usescale=true;
		}			
		
		svc.SetBetas( betas[0],constants[0], latent_features[0]);
		svc.target=label;
		svc.run();
		constant[0]=constants[0][0];


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

		singleLibFm svc = new singleLibFm(data);
		svc.smooth=this.smooth;
		svc.Type=this.Type;
		svc.maxim_Iteration=this.maxim_Iteration;
		svc.Objective=this.Objective;
		svc.verbose=false;
		svc.tau=this.tau;
		svc.copy=false;
		svc.usescale=false;
		svc.C=this.C;
		svc.C2=this.C2;
		svc.lfeatures=this.lfeatures;
		svc.tau=this.tau;			
		svc.init_values=this.init_values;	
		svc.seed=this.seed;
		svc.shuffle=this.shuffle;
		svc.learn_rate=this.learn_rate;
		svc.tolerance=this.tolerance;
		if (usescale){
			svc.setScaler(this.Scaler);
			svc.usescale=true;
		}				
		svc.SetBetas( betas[n],constants[n], latent_features[n]);
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
	
	for (int h=0; h <constants.length;h++){
		constant[h]=constants[h][0];
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
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value C2 for the latent features needs to be higher than zero" );
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
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}	
		if ( this.Objective.equals("QUANTILE") && (this.tau<=0 || this.tau>=1) )  {
			throw new IllegalStateException("For  QUANTILE tau value needs to be in (0,1)" );	
		}		
		
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value of the latent features C2 needs to be higher than zero" );
		}		
		if (  !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be SGD" );	
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
			data= (fsmatrix) data.Copy();
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
		betas= new double[n_classes][columndimension];
		constant=new double[n_classes];
		latent_features = new double[this.n_classes] [this.lfeatures*columndimension];

		double [][] constants=new double[n_classes][1];
		Thread[] thread_array= new Thread[threads];
		

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
		

	
		singleLibFm svc = new singleLibFm(data);
		svc.smooth=this.smooth;
		svc.Type=this.Type;
		svc.maxim_Iteration=this.maxim_Iteration;
		svc.Objective=this.Objective;
		svc.verbose=false;
		svc.tau=this.tau;
		svc.copy=false;
		svc.usescale=false;
		svc.C=this.C;
		svc.C2=this.C2;
		svc.lfeatures=this.lfeatures;
		svc.tau=this.tau;			
		svc.init_values=this.init_values;	
		svc.seed=this.seed;
		svc.shuffle=this.shuffle;
		svc.learn_rate=this.learn_rate;
		svc.tolerance=this.tolerance;

		if (usescale){
			svc.setScaler(this.Scaler);
			svc.usescale=true;
		}			
		//System.out.println(Arrays.toString(latent_features[0].get(0)));
		
		svc.SetBetas( betas[0],constants[0], latent_features[0]);
		svc.target=label;
		svc.run();
		constant[0]=constants[0][0];
		
		
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

		singleLibFm svc = new singleLibFm(data);
		svc.smooth=this.smooth;
		svc.Type=this.Type;
		svc.maxim_Iteration=this.maxim_Iteration;
		svc.Objective=this.Objective;
		svc.verbose=false;
		svc.tau=this.tau;
		svc.copy=false;
		svc.usescale=false;
		svc.C=this.C;
		svc.C2=this.C2;
		svc.lfeatures=this.lfeatures;
		svc.tau=this.tau;			
		svc.init_values=this.init_values;	
		svc.seed=this.seed;
		svc.shuffle=this.shuffle;
		svc.learn_rate=this.learn_rate;
		svc.tolerance=this.tolerance;
		if (usescale){
			svc.setScaler(this.Scaler);
			svc.usescale=true;
		}				
		svc.SetBetas( betas[n],constants[n], latent_features[n]);
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
	
	for (int h=0; h <constants.length;h++){
		constant[h]=constants[h][0];
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
		if (this.smooth<=0){
			this.smooth=0.1; // a high value just in case id cannot converge
		}	
		if (this.init_values<=0){
			this.init_values=0.1; // a high value just in case id cannot converge
		}	
		if (this.lfeatures<=0){
			throw new IllegalStateException(" Number of Latent features has to be higher than 0. You may use linear models instead." );	
		}		
		if ( !this.Objective.equals("MAE")&& !this.Objective.equals("QUANTILE") && !this.Objective.equals("RMSE"))  {
			throw new IllegalStateException("the objective has to be one of RMSE,MAE or QUANTILE" );	
		}	
		if ( this.Objective.equals("QUANTILE") && (this.tau<=0 || this.tau>=1) )  {
			throw new IllegalStateException("For  QUANTILE tau value needs to be in (0,1)" );	
		}				
		if (C2<=0){
			throw new IllegalStateException(" The regularization Value of the latent features C2 needs to be higher than zero" );
		}		
		if (  !Type.equals("SGD") ){
			throw new IllegalStateException(" Type has to be SGD" );	
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
			data= (smatrix) data.Copy();
		}
		// Initialise scaler
		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (usescale && ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		
		 
		if (!data.IsSortedByRow()){
			data.convert_type();
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
			betas= new double[n_classes][columndimension];
			constant=new double[n_classes];
			latent_features = new double[this.n_classes] [this.lfeatures*columndimension];

			double [][] constants=new double[n_classes][1];
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
				

			
				singleLibFm svc = new singleLibFm(data);
				svc.smooth=this.smooth;
				svc.Type=this.Type;
				svc.maxim_Iteration=this.maxim_Iteration;
				svc.Objective=this.Objective;
				svc.verbose=false;
				svc.tau=this.tau;
				svc.copy=false;
				svc.usescale=false;
				svc.C=this.C;
				svc.C2=this.C2;
				svc.lfeatures=this.lfeatures;
				svc.tau=this.tau;			
				svc.init_values=this.init_values;	
				svc.seed=this.seed;
				svc.shuffle=this.shuffle;
				svc.learn_rate=this.learn_rate;
				svc.tolerance=this.tolerance;

				if (usescale){
					svc.setScaler(this.Scaler);
					svc.usescale=true;
				}			
				
				svc.SetBetas( betas[0],constants[0], latent_features[0]);
				svc.target=label;
				svc.run();
				constant[0]=constants[0][0];

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

				singleLibFm svc = new singleLibFm(data);
				svc.smooth=this.smooth;
				svc.Type=this.Type;
				svc.maxim_Iteration=this.maxim_Iteration;
				svc.Objective=this.Objective;
				svc.verbose=false;
				svc.tau=this.tau;
				svc.copy=false;
				svc.usescale=false;
				svc.C=this.C;
				svc.C2=this.C2;
				svc.lfeatures=this.lfeatures;
				svc.tau=this.tau;			
				svc.init_values=this.init_values;	
				svc.seed=this.seed;
				svc.shuffle=this.shuffle;
				svc.learn_rate=this.learn_rate;
				svc.tolerance=this.tolerance;
				if (usescale){
					svc.setScaler(this.Scaler);
					svc.usescale=true;
				}				
				svc.SetBetas( betas[n],constants[n], latent_features[n]);
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
			
			for (int h=0; h <constants.length;h++){
				constant[h]=constants[h][0];
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
		return "LibFmRegressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier:  Regularized LibFm Regressor");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);				
		System.out.println("Constant in the model: "+ this.UseConstant);
		System.out.println("Regularization L2 value for latent features : "+ this.C2);
		System.out.println("Smooth value for FTLR: "+ this.smooth);			
		System.out.println("Objective : "+ this.Objective);	
		System.out.println("Tau value of QUANTILE obkective : "+ this.tau);			
		System.out.println("Number of Latent features: "+ this.lfeatures);				
		System.out.println("Initial value range of the latent features: "+ this.init_values);	
		System.out.println("Regularization value: "+ this.C);				
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
		return true;
	}

	@Override
	public boolean IsClassifier() {
		return false ;
	}

	@Override
	public void reset() {
		constant=null;
		betas=null;
		latent_features=null;
		n_classes=0;
		Objective="RMSE";
		tau=0.5;
		C2=0.01;
		smooth=0.1;
		lfeatures=4;
		init_values=0.1;
		C=0.01;
		Type="SGD";
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
		columndimension=0;
		shuffle=true;
		learn_rate=1.0;
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
		LibFmRegressor br = new LibFmRegressor();
		br.constant=manipulate.copies.copies.Copy(this.constant);
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		//hard copy of the latent features
		br.latent_features = new double [this.n_classes][this.lfeatures*this.columndimension] ;
		for (int f=0; f<this.n_classes;f++){
			for (int j=0; j < this.latent_features.length; j++){
				br.latent_features[f][j]=this.latent_features[f][j];
			}
			
		}
		
		
		br.n_classes=this.n_classes;
		
		
		br.C2=this.C2;		
		br.smooth=this.smooth;	
		br.lfeatures=this.lfeatures;
		br.init_values=this.init_values;
		br.C=this.C;
		br.columndimension=this.columndimension;
		br.Objective=this.Objective;
		br.tau=this.tau;
		br.Type=this.Type;
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
				else if (metric.equals("C2")) {this.C2=Double.parseDouble(value);}
				else if (metric.equals("tau")) {this.tau=Double.parseDouble(value);}				
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("Objective")) {this.Objective=value;}
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
		this.fstarget=fstarget;
	}
}
