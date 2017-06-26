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

package ml.NaiveBayes;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import preprocess.scaling.maxscaler;
import preprocess.scaling.scaler;
import exceptions.DimensionMismatchException;
import exceptions.LessThanMinimum;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;

/**
 * 
 * <p> The Naive Bayes classification theorem is a mathematical logic that given a set of predictors and a target variable
 * of distinct classes can classify a new set in one of the categories of the target based on its closenesses with the averages
 * and variances with the latter.
 *<p> Generally this classification method is quite fast as it does not require complex calculations and it still provides good
 *classification-prediction results with big data sets.
 *<p> About the Algorithm, Naive takes into account the mean and variance of all predictors in the given set and for another
 *set -for each observation- checks which has higher probability to be the correct prediction. The formula that calculates this
 *probability density P(x) number for a given observation is :
 *<pre>P(x)=Ð<sub>i</sub>1/sqrt(2ps<sub>i</sub><sup>2</sup>) * exp(-(T<sub>i</sub>-m<sub>i</sub>)<sup>2</sup>/2s<sub>i</sub><sup>2</sup>) 
 *
 *where x represents the class of the target
 *
 *i represents the chosen predictor
 *
 *T the actual value of the predictor for a given observation
 *
 *m and s<sup>2</sup> are the mean and variance of each predictor for a given class
 *</pre>
 */

public class NaiveBayesClassifier implements estimator,classifier {
	

	/**
	* The array that will hold the means for each predictor, for each class
	*/
	double means_per_predictor [][];
	/**
	* The array that will hold the variances for each predictor, for each class
	*/
	double variances_per_predictor [][];
	/**
	 *whether to use scale or not 
	 */
	public boolean usescale=true;

	/** 
	* Shrinkage to penalise the multiplications.
	*/
	public double Shrinkage=0.0;
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
	 * Target variable in String format
	 */	
	public String Starget[];
	
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
	 * Name of the unique classes
	 */
	private String classes[];
	/**
	 * Default constructor for Libfm with no data
	 */
	public NaiveBayesClassifier(){
	
	}	
	public NaiveBayesClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	

	public NaiveBayesClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}

	public NaiveBayesClassifier(smatrix data){
		
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
	 * Retrieve the number of unique targets
	 */
	public int getnumber_of_classes(){
		return n_classes;
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
	/**
	 * 	
	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8611561535854392960L;
	
	@Override
	public double[][] predict_proba(double[][] data) {
		if (n_classes<2 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double [][]  probabilities= new double [data.length][n_classes];

		// loop through all the observations of the given set to populate the probabilities matrix with the density probability of each observation to belong in each class.
		for (int i=0; i < data.length; i++) {
		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int j=0; j < this.columndimension;j++) {
					if (variances_per_predictor[g][j]!=0){
						double value=data[i][j];
						if (this.usescale){
							value=Scaler.transform(value, j);
						}
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[i][g]=product;
				if (Double.isNaN(probabilities[i][g])) {
					probabilities[i][g]=0.0;
				}
			// end of loop for all classes
			}
			
			// end of loop of all observations
		}
	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		
		for (int i=0; i < data.length; i++) {
			double sum_of_probabilities=0;
			for (int g=0; g<n_classes; g++){
				sum_of_probabilities+=probabilities[i][g];
			}
			for (int g=0; g<n_classes; g++){
				probabilities[i][g]=probabilities[i][g]/sum_of_probabilities;
			}
		}

		// return the probabilities array
		return probabilities;

			}

	@Override
	public double[][] predict_proba (fsmatrix data) {
		if (n_classes<2 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double [][]  probabilities= new double [data.GetRowDimension()][n_classes];

		// loop through all the observations of the given set to populate the probabilities matrix with the density probability of each observation to belong in each class.
		for (int i=0; i < data.GetRowDimension(); i++) {
		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int j=0; j < this.columndimension;j++) {
					if (variances_per_predictor[g][j]!=0){
						double value=data.GetElement(i, j);
						if (this.usescale){
							value=Scaler.transform(value, j);
						}						
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[i][g]=product;
				if (Double.isNaN(probabilities[i][g])) {
					probabilities[i][g]=0.0;
				}
			// end of loop for all classes
			}
			
			// end of loop of all observations
		}
	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		
		for (int i=0; i < data.GetRowDimension(); i++) {
			double sum_of_probabilities=0;
			for (int g=0; g<n_classes; g++){
				sum_of_probabilities+=probabilities[i][g];
			}
			for (int g=0; g<n_classes; g++){
				probabilities[i][g]=probabilities[i][g]/sum_of_probabilities;
			}
		}

		// return the probabilities array
		return probabilities;
	}

	@Override
	public double[][] predict_proba(smatrix data) {
		
		if (n_classes<2 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		double [][]  probabilities= new double [data.GetRowDimension()][n_classes];

		// loop through all the observations of the given set to populate the probabilities matrix with the density probability of each observation to belong in each class.
		for (int i=0; i < data.GetRowDimension(); i++) {
		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int h=data.indexpile[i]; h < data.indexpile[i+1];h++) {
					int j=data.mainelementpile[h];
					if (variances_per_predictor[g][j]!=0){
						double value=data.valuespile[h];
						if (this.usescale){
							value=Scaler.transform(value, j);
						}						
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[i][g]=product;
				if (Double.isNaN(probabilities[i][g])) {
					probabilities[i][g]=0.0;
				}
			// end of loop for all classes
			}
			
			// end of loop of all observations
		}
	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		
		for (int i=0; i < data.GetRowDimension(); i++) {
			double sum_of_probabilities=0;
			for (int g=0; g<n_classes; g++){
				sum_of_probabilities+=probabilities[i][g];
			}
			for (int g=0; g<n_classes; g++){
				probabilities[i][g]=probabilities[i][g]/sum_of_probabilities;
			}
		}

		// return the probabilities array
		return probabilities;
	}


	@Override
	public double[] predict_probaRow(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}
		
		double probabilities[]= new double [n_classes];

		for (int g=0; g<n_classes; g++){
			double product=1;
			//start loop for all predictors
			for (int j=0; j < this.columndimension;j++) {
				if (variances_per_predictor[g][j]!=0){
					double value=data[j];
					if (this.usescale){
						value=Scaler.transform(value, j);
					}
				product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
				// end of loop for all predictors
				}
			}
			probabilities[g]=product;
			if (Double.isNaN(probabilities[g])) {
				probabilities[g]=0.0;
			}
		}
		// end of loop for all classes
		
		double sum_of_probabilities=0;
		for (int g=0; g<n_classes; g++){
			sum_of_probabilities+=probabilities[g];
		}
		for (int g=0; g<n_classes; g++){
			probabilities[g]=probabilities[g]/sum_of_probabilities;
		}	
		
			
		return probabilities;
	}


	@Override
	public double[] predict_probaRow(fsmatrix data, int rows) {
		if (n_classes<2  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double probabilities[]= new double [n_classes];

		for (int g=0; g<n_classes; g++){
			double product=1;
			//start loop for all predictors
			for (int j=0; j < this.columndimension;j++) {
				if (variances_per_predictor[g][j]!=0){
					double value=data.GetElement(rows, j);
					if (this.usescale){
						value=Scaler.transform(value, j);
					}
				product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
				// end of loop for all predictors
				}
			}
			probabilities[g]=product;
			if (Double.isNaN(probabilities[g])) {
				probabilities[g]=0.0;
			}
		}
		// end of loop for all classes
		
		double sum_of_probabilities=0;
		for (int g=0; g<n_classes; g++){
			sum_of_probabilities+=probabilities[g];
		}
		for (int g=0; g<n_classes; g++){
			probabilities[g]=probabilities[g]/sum_of_probabilities;
		}	
		
			
		return probabilities;
	}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		if (n_classes<2  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double probabilities[]= new double [n_classes];

		for (int g=0; g<n_classes; g++){
			double product=1;
			//start loop for all predictors
			for (int h=start; h < end;h++) {
				int j=data.mainelementpile[h];
				
				if (variances_per_predictor[g][j]!=0){
					double value=data.valuespile[h];
					if (this.usescale){
						value=Scaler.transform(value, j);
					}
				product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
				// end of loop for all predictors
				}
			}
			probabilities[g]=product;
			if (Double.isNaN(probabilities[g])) {
				probabilities[g]=0.0;
			}
		}
		// end of loop for all classes
		
		double sum_of_probabilities=0;
		for (int g=0; g<n_classes; g++){
			sum_of_probabilities+=probabilities[g];
		}
		for (int g=0; g<n_classes; g++){
			probabilities[g]=probabilities[g]/sum_of_probabilities;
		}	
		
		return probabilities;
			}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<2 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double [][]  probabilities= new double [data.GetRowDimension()][n_classes];
		double predictions[]= new double [data.GetRowDimension()];
		// loop through all the observations of the given set to populate the probabilities matrix with the density probability of each observation to belong in each class.
		for (int i=0; i < data.GetRowDimension(); i++) {
		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int j=0; j < this.columndimension;j++) {
					if (variances_per_predictor[g][j]!=0){
						double value=data.GetElement(i, j);
						if (this.usescale){
							value=Scaler.transform(value, j);
						}						
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[i][g]=product;
				if (Double.isNaN(probabilities[i][g])) {
					probabilities[i][g]=0.0;
				}
			// end of loop for all classes
			}
			
			// end of loop of all observations
		}
	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		

			
			for (int i=0; i < probabilities.length; i++){
				
		    	  int maxi=0;
		    	  double max=probabilities[i][0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (probabilities[i][k]>max){
		    			 max=probabilities[i][k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions[i]=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions[i]=maxi;
		    	  }
			}
			
			probabilities=null;
			System.gc();
			return predictions;
			}

	@Override
	public double[] predict(smatrix data) {
		if (n_classes<2 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		double [][]  probabilities= new double [data.GetRowDimension()][n_classes];
		double predictions[]= new double [data.GetRowDimension()];
		// loop through all the observations of the given set to populate the probabilities matrix with the density probability of each observation to belong in each class.
		for (int i=0; i < data.GetRowDimension(); i++) {
		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int h=0; h < this.columndimension;h++) {
					int j=data.mainelementpile[h];
					if (variances_per_predictor[g][j]!=0){
						double value=data.valuespile[h];
						if (this.usescale){
							value=Scaler.transform(value, j);
						}						
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[i][g]=product;
				if (Double.isNaN(probabilities[i][g])) {
					probabilities[i][g]=0.0;
				}
			// end of loop for all classes
			}
			
			// end of loop of all observations
		}
	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		

			
			for (int i=0; i < probabilities.length; i++){
				
		    	  int maxi=0;
		    	  double max=probabilities[i][0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (probabilities[i][k]>max){
		    			 max=probabilities[i][k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions[i]=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions[i]=maxi;
		    	  }
			}
			
			probabilities=null;
			System.gc();
			return predictions;
			}

	@Override
	public double[] predict(double[][] data) {
		if (n_classes<2 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double [][]  probabilities= new double [data.length][n_classes];
		double predictions[]= new double [data.length];
		// loop through all the observations of the given set to populate the probabilities matrix with the density probability of each observation to belong in each class.
		for (int i=0; i < data.length; i++) {
		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int j=0; j < this.columndimension;j++) {
					if (variances_per_predictor[g][j]!=0){
						double value=data[i][j];
						if (this.usescale){
							value=Scaler.transform(value, j);
						}
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[i][g]=product;
				if (Double.isNaN(probabilities[i][g])) {
					probabilities[i][g]=0.0;
				}
			// end of loop for all classes
			}
			
			// end of loop of all observations
		}
	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		
		for (int i=0; i < data.length; i++) {
			double sum_of_probabilities=0;
			for (int g=0; g<n_classes; g++){
				sum_of_probabilities+=probabilities[i][g];
			}
			for (int g=0; g<n_classes; g++){
				probabilities[i][g]=probabilities[i][g]/sum_of_probabilities;
			}
		}
			
			for (int i=0; i < probabilities.length; i++){
				
		    	  int maxi=0;
		    	  double max=probabilities[i][0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (probabilities[i][k]>max){
		    			 max=probabilities[i][k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions[i]=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions[i]=maxi;
		    	  }
			}
			
			probabilities=null;
			System.gc();
			return predictions;
			}
	

	@Override
	public double predict_Row(double[] data) {
		if (n_classes<2 ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  

		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}
		
		double []  probabilities= new double [n_classes];
		double predictions=0.0;

		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int j=0; j < this.columndimension;j++) {
					if (variances_per_predictor[g][j]!=0){
						double value=data[j];
						if (this.usescale){
							value=Scaler.transform(value, j);
						}
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[g]=product;
				if (Double.isNaN(probabilities[g])) {
					probabilities[g]=0.0;
				}
			// end of loop for all classes
			}
			

	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		

			
			for (int i=0; i < probabilities.length; i++){
				
		    	  int maxi=0;
		    	  double max=probabilities[0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (probabilities[k]>max){
		    			 max=probabilities[k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions=maxi;
		    	  }
			}
			
			probabilities=null;
			System.gc();
			return predictions;
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {

		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double []  probabilities= new double [n_classes];
		double predictions=0.0;

		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int j=0; j < this.columndimension;j++) {
					if (variances_per_predictor[g][j]!=0){
						double value=data.GetElement(rows, j);
						if (this.usescale){
							value=Scaler.transform(value, j);
						}
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[g]=product;
				if (Double.isNaN(probabilities[g])) {
					probabilities[g]=0.0;
				}
			// end of loop for all classes
			}
			

	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		

			
			for (int i=0; i < probabilities.length; i++){
				
		    	  int maxi=0;
		    	  double max=probabilities[0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (probabilities[k]>max){
		    			 max=probabilities[k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions=maxi;
		    	  }
			}
			
			probabilities=null;
			System.gc();
			return predictions;
			}
	

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		
		double []  probabilities= new double [n_classes];
		double predictions=0.0;

		//loop through all class to find the one with the highest probability
			for (int g=0; g<n_classes; g++){
				double product=1;
				//start loop for all predictors
				for (int h=start; h < end;h++) {
					int j=data.mainelementpile[h];
					if (variances_per_predictor[g][j]!=0){
						double value=data.valuespile[h];
						if (this.usescale){
							value=Scaler.transform(value, j);
						}
					product=product*((1/Math.sqrt(2*Math.PI*variances_per_predictor[g][j])) * (Math.exp((-((value-means_per_predictor[g][j])*(value-means_per_predictor[g][j])))/(2*variances_per_predictor[g][j]))) + Shrinkage);
					// end of loop for all predictors
					}
				}
				probabilities[g]=product;
				if (Double.isNaN(probabilities[g])) {
					probabilities[g]=0.0;
				}
			// end of loop for all classes
			}
			

	
		// loop again to convert everything to real probabilities and make certain that sum of each observation in probabilities in 1.
		

			
			for (int i=0; i < probabilities.length; i++){
				
		    	  int maxi=0;
		    	  double max=probabilities[0];
		    	  for (int k=1; k<n_classes; k++) {
		    		 if (probabilities[k]>max){
		    			 max=probabilities[k];
		    			 maxi=k;	 
		    		 }
		    	  }
		    	  try{
		    		  predictions=Double.parseDouble(classes[maxi]);
		    	  } catch (Exception e){
		    		  predictions=maxi;
		    	  }
			}
			
			probabilities=null;
			System.gc();
			return predictions;
			}



	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}		
			

		if ( (target==null || target.length!=data.length) && (Starget==null || Starget.length!=data.length) ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else {
			if (target!=null && (classes==null ||  classes.length<=1) ){
				
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
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    }
			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);				
				
			}
		}
		// make sensible checks on the target data
		if (weights==null) {
			weights=new double [data.length];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/(double) weights.length;
			}
		} else {
			if (weights.length!=data.length){
				throw new DimensionMismatchException(weights.length,data.length);
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			
		}

		//hard copy
		if (copy){
			data= manipulate.copies.copies.Copy(data);
		}
		// Initialise scaler
		if ( this.usescale && Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if ( (this.usescale &&  Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		
		n_classes=classes.length;
		//initialize column dimension
		columndimension=data[0].length;
		
	       means_per_predictor= new double [n_classes][columndimension];   
 	       variances_per_predictor= new double [n_classes][columndimension];   
 	       
 	       for (int g=0; g <n_classes; g++){
 	    	   
 	 	    	 
 	    	  double weighted_count=0.0;
 	    	  double squared_weighted_count=0.0;
 	    	  //populate the descriptives' stats array list
 	    	  if (target!=null){
 	    	  for (int i=0; i <target.length; i++ ) {
 	    		  
 	    		  if (target[i]==Double.parseDouble(classes[g])){ 
 	    			  for (int j=0; j < data[0].length; j++){
 	    				  if (this.usescale){
 	    					 means_per_predictor[g][j]+=Scaler.transform(data[i][j],j)*weights[i];
 	    				  } else {
 	    					 means_per_predictor[g][j]+=data[i][j]*weights[i];
 	    				  }
 	    			  }
 	    			weighted_count+=weights[i];
 	    			squared_weighted_count+=weights[i]*weights[i];
 	    		  }
 	    		  //end of the total observations' array
 	    	  }
 	    	  //Starget
 	    	  } else {
 	 	    	  for (int i=0; i <Starget.length; i++ ) {
 	 	    		  if (Starget[i].equals(classes[g])){ 
 	 	    			  for (int j=0; j < data[0].length; j++){
 	 	    				 if (this.usescale){
 	 	    					means_per_predictor[g][j]+=Scaler.transform(data[i][j],j)*weights[i];
 	 	    				 }else {
 	 	    			 means_per_predictor[g][j]+=data[i][j]*weights[i];
 	 	    				 }
 	 	    			  }
 	 	    			weighted_count+=weights[i];
 	 	    			squared_weighted_count+=weights[i]*weights[i];
 	 	    		  }
 	 	    		  //end of the total observations' array
 	 	    	  }  
 	    	  }
 	    	  
 	    	  double discounted_counts []= new double [columndimension];
 	    	  for(int j=0; j < columndimension; j++){
 	    		 discounted_counts[j]=weighted_count;
 	    	  }
 	    	  if (weighted_count==0.0) {
 	    		  for (int j=0; j < columndimension; j++){
 	    			 variances_per_predictor[g][j]=0;
 	    			means_per_predictor[g][j]=0;
 	    		  }
 	    		  
 	    		  
 	    	  } else {
 	    		 if (target!=null){
 	 	     	 for (int i=0; i <data[0].length; i++ ) { 
 	 	     		if (target[i]==Double.parseDouble(classes[g])){  
 	 	    			  for (int j=0; j < data[0].length; j++){
 	 	    				if (this.usescale){
 	 	 	    				 variances_per_predictor[g][j]+=((Scaler.transform(data[i][j],j)- means_per_predictor[g][j]/weighted_count) 
 	 	 	 	 	    		*(Scaler.transform(data[i][j],j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;	 	    					
 	 	    				}else {
 	 	    				 variances_per_predictor[g][j]+=((data[i][j]- means_per_predictor[g][j]/weighted_count) 
 	 	    				*(data[i][j]- means_per_predictor[g][j]/weighted_count))*weights[i] ;
 	 	    				}
 	 	    				}
 	 	    		  }
 	 	    		  //end of the total observations' array
 	 	    	  }
 	    		 } else {
 	 	 	    	  for (int i=0; i <Starget.length; i++ ) {
 	 	 	    		 if (Starget[i].equals(classes[g])){ 
 	 	 	    			  for (int j=0; j < data[0].length; j++){
 	  	 	    				if (this.usescale){
 	  	 	 	    				 variances_per_predictor[g][j]+=((Scaler.transform(data[i][j],j)- means_per_predictor[g][j]/weighted_count) 
 	  	 	 	 	 	    		*(Scaler.transform(data[i][j],j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;	 	    					
 	  	 	    				}else {
 	  	 	    				 variances_per_predictor[g][j]+=((data[i][j]- means_per_predictor[g][j]/weighted_count) 
 	  	 	    				*(data[i][j]- means_per_predictor[g][j]/weighted_count))*weights[i] ;
 	  	 	    				}
 	  	 	    				}
 	 	 	    		  }
 	 	 	    		  //end of the total observations' array
 	 	 	    	  }
 	 	    		 
 	 	 	    	}
 	 	    	  // get the mean and variance for each predictor for this given class
 	 	    	 for (int j=0; j <data[0].length; j++ ){
 	 	    		means_per_predictor[g][j]/=weighted_count;
 	 	    		variances_per_predictor[g][j]*=(weighted_count/(weighted_count*weighted_count-squared_weighted_count )  );
 	 	    	   }
 	 	    	   // end of all the classes'loop
 	 	       }
 	    	  
 	    	  
 	       }
 	       
 	       
 	       if (verbose ){
 	    	   System.out.println("Naive Bayes Classifier trained with : " );
 	    	   System.out.println("Distinct classes : " + classes.length);
 	    	   System.out.println("Predictors : " +  columndimension);
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

		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else {
			if (target!=null && (classes==null ||  classes.length<=1) ){
				
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
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    }
			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);				
				
			}
		}
		// make sensible checks on the target data
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/(double) weights.length;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			
		}

		//hard copy
		if (copy){
			data= (fsmatrix) (data.Copy());
		}
		// Initialise scaler
		if ( this.usescale && Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if ( (this.usescale &&  Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		
		n_classes=classes.length;
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		
	       means_per_predictor= new double [n_classes][columndimension];   
 	       variances_per_predictor= new double [n_classes][columndimension];   
 	       
 	       for (int g=0; g <n_classes; g++){
 	    	   
 	    	 
 	    	  double weighted_count=0.0;
 	    	  double squared_weighted_count=0.0;
 	    	  //populate the descriptives' stats array list
 	    	  if (target!=null){
 	    	  for (int i=0; i <target.length; i++ ) {
 	    		  
 	    		  if (target[i]==Double.parseDouble(classes[g])){ 
 	    			  for (int j=0; j < data.GetColumnDimension(); j++){
 	    				  if (this.usescale){
 	    					 means_per_predictor[g][j]+=Scaler.transform(data.GetElement(i, j),j)*weights[i];
 	    				  } else {
 	    					 means_per_predictor[g][j]+=data.GetElement(i, j)*weights[i];
 	    				  }
 	    			  }
 	    			weighted_count+=weights[i];
 	    			squared_weighted_count+=weights[i]*weights[i];
 	    		  }
 	    		  //end of the total observations' array
 	    	  }
 	    	  //Starget
 	    	  } else {
 	 	    	  for (int i=0; i <Starget.length; i++ ) {
 	 	    		  if (Starget[i].equals(classes[g])){ 
 	 	    			  for (int j=0; j < data.GetColumnDimension(); j++){
 	 	    				 if (this.usescale){
 	 	    					means_per_predictor[g][j]+=Scaler.transform(data.GetElement(i, j),j)*weights[i];
 	 	    				 }else {
 	 	    			 means_per_predictor[g][j]+=data.GetElement(i, j)*weights[i];
 	 	    				 }
 	 	    			  }
 	 	    			weighted_count+=weights[i];
 	 	    			squared_weighted_count+=weights[i]*weights[i];
 	 	    		  }
 	 	    		  //end of the total observations' array
 	 	    	  }  
 	    	  }
 	    	  
 	    	  double discounted_counts []= new double [columndimension];
 	    	  for(int j=0; j < columndimension; j++){
 	    		 discounted_counts[j]=weighted_count;
 	    	  }
 	    	  if (weighted_count==0.0) {
 	    		  for (int j=0; j < columndimension; j++){
 	    			 variances_per_predictor[g][j]=0;
 	    			means_per_predictor[g][j]=0;
 	    		  }
 	    		  
 	    		  
 	    	  } else {
 	    		 if (target!=null){
 	 	     	 for (int i=0; i <data.GetRowDimension(); i++ ) { 
 	 	     		if (target[i]==Double.parseDouble(classes[g])){  
 	 	    			  for (int j=0; j < data.GetColumnDimension(); j++){
 	 	    				if (this.usescale){
 	 	 	    				 variances_per_predictor[g][j]+=((Scaler.transform(data.GetElement(i, j),j)- means_per_predictor[g][j]/weighted_count) 
 	 	 	 	 	    		*(Scaler.transform(data.GetElement(i, j),j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;	 	    					
 	 	    				}else {
 	 	    				 variances_per_predictor[g][j]+=((data.GetElement(i, j)- means_per_predictor[g][j]/weighted_count) 
 	 	    				*(data.GetElement(i, j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;
 	 	    				}
 	 	    				}
 	 	    		  }
 	 	    		  //end of the total observations' array
 	 	    	  }
 	    		 } else {
 	 	 	    	  for (int i=0; i <Starget.length; i++ ) {
 	 	 	    		 if (Starget[i].equals(classes[g])){ 
 	 	 	    			  for (int j=0; j < data.GetColumnDimension(); j++){
 	  	 	    				if (this.usescale){
 	  	 	 	    				 variances_per_predictor[g][j]+=((Scaler.transform(data.GetElement(i, j),j)- means_per_predictor[g][j]/weighted_count) 
 	  	 	 	 	 	    		*(Scaler.transform(data.GetElement(i, j),j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;	 	    					
 	  	 	    				}else {
 	  	 	    				 variances_per_predictor[g][j]+=((data.GetElement(i, j)- means_per_predictor[g][j]/weighted_count) 
 	  	 	    				*(data.GetElement(i, j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;
 	  	 	    				}
 	  	 	    				}
 	 	 	    		  }
 	 	 	    		  //end of the total observations' array
 	 	 	    	  }
 	 	    		 
 	 	 	    	}
 	 	    	  // get the mean and variance for each predictor for this given class
 	 	    	 for (int j=0; j <data.GetColumnDimension(); j++ ){
 	 	    		means_per_predictor[g][j]/=weighted_count;
 	 	    		variances_per_predictor[g][j]*=(weighted_count/(weighted_count*weighted_count-squared_weighted_count )  );
 	 	    	   }
 	 	    	   // end of all the classes'loop
 	 	       }
 	    	  
 	    	  
 	       }
 	       
 	       
 	       if (verbose ){
 	    	   System.out.println("Naive Bayes Classifier trained with : " );
 	    	   System.out.println("Distinct classes : " + classes.length);
 	    	   System.out.println("Predictors : " +  columndimension);
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
		if ( (target==null || target.length!=data.GetRowDimension()) && (Starget==null || Starget.length!=data.GetRowDimension()) ){
			throw new IllegalStateException(" target array needs to be provided with the same length as the data" );
		} else {
			if (target!=null && (classes==null ||  classes.length<=1) ){
				
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
		    for (int j=0; j < uniquevalues.length; j++){
		    	classes[j]=uniquevalues[j]+"";
		    }
			} else 	if (Starget!=null && (classes==null ||  classes.length<=1)){
				classes=manipulate.distinct.distinct.getstringDistinctset(Starget);
				if (classes.length<=1){
					throw new IllegalStateException(" target array needs to have more 2 or more classes" );	
				}
			    Arrays.sort(classes);				
				
			}
		}
		// make sensible checks on the target data
		if (weights==null) {
			weights=new double [data.GetRowDimension()];
			for (int i=0; i < weights.length; i++){
				weights[i]=1.0/(double) weights.length;
			}
		} else {
			if (weights.length!=data.GetRowDimension()){
				throw new DimensionMismatchException(weights.length,data.GetRowDimension());
			}
			weights=manipulate.transforms.transforms.scaleweight(weights);
			
		}

		//hard copy
		if (copy){
			data= (smatrix) (data.Copy());
		}
		// Initialise scaler
		if ( this.usescale && Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if ( (this.usescale &&  Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		
	    if (!data.IsSortedByRow()){
	    	data.convert_type();
	    }
		
		n_classes=classes.length;
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		
	       means_per_predictor= new double [n_classes][columndimension];   
 	       variances_per_predictor= new double [n_classes][columndimension];   
 	       
 	       for (int g=0; g <n_classes; g++){
 	    	   
 	 	    	 
 	    	  double weighted_count=0.0;
 	    	  double squared_weighted_count=0.0;
 	    	  //populate the descriptives' stats array list
 	    	  if (target!=null){
 	    	  for (int i=0; i <target.length; i++ ) {
 	    		  
 	    		  if (target[i]==Double.parseDouble(classes[g])){ 
 	    			  for (int h=data.indexpile[i]; h <data.indexpile[i+1]; h++){
 	    				  int j=data.mainelementpile[h];
 	    				  double value=data.valuespile[h];
 	    				  if (this.usescale){
 	    					 means_per_predictor[g][j]+=Scaler.transform(value,j)*weights[i];
 	    				  } else {
 	    					 means_per_predictor[g][j]+=value*weights[i];
 	    				  }
 	    			  }
 	    			weighted_count+=weights[i];
 	    			squared_weighted_count+=weights[i]*weights[i];
 	    		  }
 	    		  //end of the total observations' array
 	    	  }
 	    	  //Starget
 	    	  } else {
 	 	    	  for (int i=0; i <Starget.length; i++ ) {
 	 	    		  if (Starget[i].equals(classes[g])){ 
 	 	    			  for (int h=data.indexpile[i]; h <data.indexpile[i+1]; h++){
 	 	    				  int j=data.mainelementpile[h];
 	 	    				  double value=data.valuespile[h];
 	 	    				 if (this.usescale){
 	 	    					means_per_predictor[g][j]+=Scaler.transform(value,j)*weights[i];
 	 	    				 }else {
 	 	    			 means_per_predictor[g][j]+=value*weights[i];
 	 	    				 }
 	 	    			  }
 	 	    			weighted_count+=weights[i];
 	 	    			squared_weighted_count+=weights[i]*weights[i];
 	 	    		  }
 	 	    		  //end of the total observations' array
 	 	    	  }  
 	    	  }
 	    	  
 	    	  double discounted_counts []= new double [columndimension];
 	    	  for(int j=0; j < columndimension; j++){
 	    		 discounted_counts[j]=weighted_count;
 	    	  }
 	    	  if (weighted_count==0.0) {
 	    		  for (int j=0; j < columndimension; j++){
 	    			 variances_per_predictor[g][j]=0;
 	    			means_per_predictor[g][j]=0;
 	    		  }
 	    		  
 	    		  
 	    	  } else {
 	    		 if (target!=null){
 	 	     	 for (int i=0; i <data.GetRowDimension(); i++ ) { 
 	 	     		if (target[i]==Double.parseDouble(classes[g])){  
	 	    			  for (int h=data.indexpile[i]; h <data.indexpile[i+1]; h++){
 	 	    				  int j=data.mainelementpile[h];
 	 	    				  double value=data.valuespile[h];
 	 	    				if (this.usescale){
 	 	 	    				 variances_per_predictor[g][j]+=((Scaler.transform(value,j)- means_per_predictor[g][j]/weighted_count) 
 	 	 	 	 	    		*(Scaler.transform(value,j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;	 	    					
 	 	    				}else {
 	 	    				 variances_per_predictor[g][j]+=((value- means_per_predictor[g][j]/weighted_count) 
 	 	    				*(value- means_per_predictor[g][j]/weighted_count))*weights[i] ;
 	 	    				}
 	 	    				}
 	 	    		  }
 	 	    		  //end of the total observations' array
 	 	    	  }
 	    		 } else {
 	 	 	    	  for (int i=0; i <Starget.length; i++ ) {
 	 	 	    		 if (Starget[i].equals(classes[g])){ 
 		 	    			  for (int h=data.indexpile[i]; h <data.indexpile[i+1]; h++){
 	 	 	    				  int j=data.mainelementpile[h];
 	 	 	    				  double value=data.valuespile[h];
 	  	 	    				if (this.usescale){
 	  	 	 	    				 variances_per_predictor[g][j]+=((Scaler.transform(value,j)- means_per_predictor[g][j]/weighted_count) 
 	  	 	 	 	 	    		*(Scaler.transform(value,j)- means_per_predictor[g][j]/weighted_count))*weights[i] ;	 	    					
 	  	 	    				}else {
 	  	 	    				 variances_per_predictor[g][j]+=((value- means_per_predictor[g][j]/weighted_count) 
 	  	 	    				*(value- means_per_predictor[g][j]/weighted_count))*weights[i] ;
 	  	 	    				}
 	  	 	    				}
 	 	 	    		  }
 	 	 	    		  //end of the total observations' array
 	 	 	    	  }
 	 	    		 
 	 	 	    	}
 	 	    	  // get the mean and variance for each predictor for this given class
 	 	    	 for (int j=0; j <data.GetColumnDimension(); j++ ){
 	 	    		means_per_predictor[g][j]/=weighted_count;
 	 	    		variances_per_predictor[g][j]*=(weighted_count/(weighted_count*weighted_count-squared_weighted_count )  );
 	 	    	   }
 	 	    	   // end of all the classes'loop
 	 	       }
 	    	  
 	    	  
 	       }
 	       
 	       
 	       if (verbose ){
 	    	   System.out.println("Naive Bayes Classifier trained with : " );
 	    	   System.out.println("Distinct classes : " + classes.length);
 	    	   System.out.println("Predictors : " +  columndimension);
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
		return "knnClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: Naive Bayes");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);				
		System.out.println("Shrinkage: " + this.Shrinkage);		
		System.out.println("Usescale: " + this.usescale);	

		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (this.means_per_predictor==null){
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
		if (this.means_per_predictor!=null ){
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
		return true ;
	}

	@Override
	public void reset() {

	    means_per_predictor=null;
 	    variances_per_predictor=null;
		n_classes=0;
		this.usescale=true;
		columndimension=0;
		Shrinkage=0.0;
		Scaler=null;
		copy=true;
		seed=1;
		random=null;
		target=null;
		Starget=null;
		weights=null;
		verbose=true;
		n_classes=0;
		classes=null;
		
	}


	@Override
	public estimator copy() {
		NaiveBayesClassifier br = new NaiveBayesClassifier();
		//hard copy of the latent features

		br.means_per_predictor=this.means_per_predictor.clone();
		br.variances_per_predictor=this.variances_per_predictor.clone();		
		br.Shrinkage=this.Shrinkage;
		br.n_classes=this.n_classes;
		br.columndimension=this.columndimension;
		br.usescale=this.usescale;
		br.Scaler=this.Scaler;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.classes=this.classes.clone();
		br.n_classes=this.n_classes;
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.Starget=this.Starget.clone();		
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
				
				if (metric.equals("Shrinkage")) {this.Shrinkage=Double.parseDouble(value);}		
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
				else if (metric.equals("usescale")) {this.usescale=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("verbose")) {this.verbose=(value.toLowerCase().equals("true")?true:false)   ;}

			}
			
		}
		

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
