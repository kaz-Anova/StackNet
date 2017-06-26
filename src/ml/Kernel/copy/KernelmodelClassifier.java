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

package ml.Kernel.copy;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import preprocess.scaling.scaler;
import preprocess.scaling.maxscaler;
import exceptions.DimensionMismatchException;
import exceptions.LessThanMinimum;
import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;

/**
 * 
 * Kernel-based regularized regression model   class runnable with 2 different optimisation methods:
 * <ol>
 * <li> SGD "Stochastic Gradient Descent" with adaptive learning Rate (supports L1 and L2)  </li> 
 * <li> FTRL"Follow The Regularized Leader" (supports L1 and L2), inspired by Tingru's code in Kaggle forums <a href="https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory"> link </a>  </li> 
 * </ol>
 */
public class KernelmodelClassifier implements estimator,classifier {
	
	/**
	 * Type of algorithm to use. It has to be one of  SGD, FTRL =Follow The Regularized Leader
	 */
	public String Type="SGD";
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;
	/**
	 * Regularization value for l1 "Follow The Regularized Leader"
	 */
	public double l1C=1.0;	

	/**
	 * This will hold the latent features to encapsulate the 2d interactions among the variables
	 * This will also hold the sum of past gradients to control for the learning rate
	 */
	private smatrix [] vectorset ;
 
	/**
	 * Internal passes
	 */
	public int intpasses=1;
	/**
	 * The Final percentage of support osbervations required based on the initial given dataset
	 */
	public double pinter=0.1;
	/**
	 * The percentage of subsections of each submodel
	 */	
	public double submodelcutsper=0.01;
	/**
	 * Value that smoothes SGD steps
	 */	
	public double smooth=0.01;
	/**
	 * minimum coefficients' values for it to be considered
	 */
	public double intcoeffthres=0.1; 
	/**
	 * minimum percentage to keep after each batch
	 */
	public double intpertokeep=0.2; 
	/**
	 *  can be logistic or svm, 
	 */
	public String Objective="logistic";
	/**
	 * type of distance, can be RBF, POLY or SIGMOID
	 */
	public String distance="RBF";
	/**
	 * std for an RBF kernel
	 */	
	public double gammabfs=0.01;
	/**
	 * degree of polynomial
	 */
	public int degree=2;
	/**
	 * Coefficient for poly
	 */
	public double coeff=1.0;	
	/**
	 * threads inside the step algorithm
	 */
	public int intthreads=1;
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
	 * Target variable in String format
	 */	
	public String Starget[];
	
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
	 * Name of the unique classes
	 */
	private String classes[];
	

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
	public KernelmodelClassifier(){
	
	}	
	public KernelmodelClassifier(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	

	public KernelmodelClassifier(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}

	public KernelmodelClassifier(smatrix data){
		
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
	 * Retrieve the number of unique classes
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
	public smatrix [] GetupportFeatures(){
		
		if (vectorset==null || vectorset.length<=0){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		
		smatrix lat []= new smatrix [this.vectorset.length];
		
		for (int f=0; f < this.vectorset.length; f++){
			
			if (this.vectorset[f]!=null){
			smatrix templatent= (smatrix)  this.vectorset[f].Copy();	 
			 
	
			lat[f]=templatent;
			}
		}

		return lat;
	}	
	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8617161535854392960L;

	
	
	
	@Override
	public double[][] predict_proba(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
		 throw new IllegalStateException("The fit method needs to be run successfully in " +
									"order to create the logic before attempting scoring a new set");}  			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		//System.out.println(n_classes);
		double predictions[][]= new double [data.length][n_classes];
			
			for (int i=0; i < predictions.length; i++) {
				double sum=0.0;
				
		    	  for (int k=0; k<this.n_classes; k++) {
		    		    
		    		  if (this.n_classes==2){
		    			  k++;
		    		  }
		   	       //hidden units in the 2nd hidden layer
		    		  
		    		  	smatrix vectorset=this.vectorset[k];
		    		  	
						double pred=constant[k];

				    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
				    		double feature=0;
				    		if (distance.equals("RBF")) {
				    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
				    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data[i][vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
				    				feature+=euc*euc;
				    			}
				    			feature=Math.exp(-this.gammabfs * feature); 

				    			
				    		} else if (distance.equals("POLY")) {
				    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
				    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
				    			}
				    			feature=(this.gammabfs*feature)+this.coeff;
				    			for (int h=0; h <this.degree-1; h++){
				    				feature*=feature;
				    			}

				    		} else if (distance.equals("SIGMOID") ){
				    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
				    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
				    			}
				    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

				    		}
				    		
					    pred+=feature*betas[k][j];
				    			
				    	}	
				    	if (this.Objective.equals("logistic")){
		    		//convert to probability
		    			         
		    		 pred= 1. / (1. + Math.exp(-pred));
				    	}
		    		
		    		predictions[i][k]=pred;
		    		sum=sum+ predictions[i][k];
		    	  }
		    	  
		    	  if (this.n_classes==2){
		    		  if (this.Objective.equals("logistic")){
		    		  predictions[i][0]=1-predictions[i][1];
		    		  sum+=predictions[i][0];
		    		  }else {
		    			  predictions[i][0]=-predictions[i][1];
		    		  }
		    		  
		    		 
		    	  }
		    	  if (this.Objective.equals("logistic")){
		    	  for (int k=0; k<this.n_classes; k++) {
		    		  predictions[i][k]= predictions[i][k]/sum;
		    	  
	    		  }
		    	  }
		    	
		    	  //System.out.println(Arrays.toString(predictions[i]));

		}
			
		return predictions;
	}

	@Override
	public double[][] predict_proba(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		
		//System.out.println(n_classes);
		double predictions[][]= new double [data.GetRowDimension()][n_classes];
		//smatrix vectorsets=this.vectorset[1];
	  	//vectorsets.Print(100);
	  	//System.out.println(constant[1]);
	  	//System.out.println(Arrays.toString(betas[1]));

		for (int i=0; i < predictions.length; i++) {
			double sum=0.0;
			
	    	  for (int k=0; k<this.n_classes; k++) {
	    		    
	    		  if (this.n_classes==2){
	    			  k++;
	    		  }
	   	       //hidden units in the 2nd hidden layer
	    		  
	    		  	smatrix vectorset=this.vectorset[k];
	    		  	//vectorset.Print(100);
	    		  	//System.out.println(constant[k]);
					double pred=constant[k];

			    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
			    		double feature=0;
			    		if (distance.equals("RBF")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    				feature+=euc*euc;
			    			}
			    			feature=Math.exp(-this.gammabfs * feature); 

			    			
			    		} else if (distance.equals("POLY")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=(this.gammabfs*feature)+this.coeff;
			    			for (int h=0; h <this.degree-1; h++){
			    				feature*=feature;
			    			}

			    		} else if (distance.equals("SIGMOID") ){
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

			    		}
			    	//System.out.println(feature + " and " +betas[k][j] );
				    pred+=feature*betas[k][j];
			    			
			    	}	
	    		
			    	if (this.Objective.equals("logistic")){
			    		//convert to probability
			    			         
			    		 pred= 1. / (1. + Math.exp(-pred));
					    	}
			    		
			    		predictions[i][k]=pred;
			    		sum=sum+ predictions[i][k];
			    	  }
			    	  
			    	  if (this.n_classes==2){
			    		  if (this.Objective.equals("logistic")){
			    		  predictions[i][0]=1-predictions[i][1];
			    		  sum+=predictions[i][0];
			    		  }else {
			    			  predictions[i][0]=-predictions[i][1];
			    		  }
			    		  
			    		 
			    	  }
			    	  if (this.Objective.equals("logistic")){
			    	  for (int k=0; k<this.n_classes; k++) {
			    		  predictions[i][k]= predictions[i][k]/sum;
			    	  
		    		  }
			    	  }
			    	
			    	  //System.out.println(Arrays.toString(predictions[i]));

			}
				
			return predictions;
		}

	@Override
	public double[][] predict_proba(smatrix data) {
		
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");} 
					 			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		if (!data.IsSortedByRow()){
			data.convert_type();
		}
		
		HashMap<Integer,Integer> has_index=null;
		//System.out.println(n_classes);
		double predictions[][]= new double [data.GetRowDimension()][n_classes];
		
		for (int i=0; i < predictions.length; i++) {
			double sum=0.0;
			
	    	  for (int k=0; k<this.n_classes; k++) {
	    		    
	    		  if (this.n_classes==2){
	    			  k++;
	    		  }
	   	       //hidden units in the 2nd hidden layer
	    		  
	    		  	smatrix vectorset=this.vectorset[k];
	    		  	
					double pred=constant[k];
					 has_index=new HashMap<Integer,Integer>();
				    	for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
				    		has_index.put(data.mainelementpile[v],v);
						}
			    	for (int j=0; j <vectorset.GetRowDimension() ; j++){

			    		double feature=0;

					    if (distance.equals("RBF")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				int cc=vectorset.mainelementpile[h];
			    				Integer ks=has_index.get(cc);
			    				if (ks!=null){
			    					double euc=Scaler.transform(vectorset.valuespile[h], cc)-Scaler.transform(data.valuespile[ks], cc);
			    					feature+=euc*euc;
			    					}else {
				    					double euc=Scaler.transform(vectorset.valuespile[h], cc);
				    					feature+=euc*euc;
				    				}
			    			}
			    			feature=Math.exp(-this.gammabfs * feature); 
			    			
			    		} else if (distance.equals("POLY")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				int cc=vectorset.mainelementpile[h];
			    				Integer ks=has_index.get(cc);
			    				if (ks!=null){
			    					feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
			    				}
			    				}
			    			feature=(this.gammabfs*feature)+this.coeff;
			    			for (int h=0; h <this.degree-1; h++){
			    				feature*=feature;
			    			}
			    			
			    		} else if (distance.equals("SIGMOID") ){
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				int cc=vectorset.mainelementpile[h];
			    				Integer ks=has_index.get(cc);
			    				if (ks!=null){
				    				feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
			    				}
			    			}
			    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
			    		}
			    		
				    pred+=feature*betas[k][j];
			    			
			    	}	

			    	if (this.Objective.equals("logistic")){
			    		//convert to probability
			    			         
			    		 pred= 1. / (1. + Math.exp(-pred));
					    	}
			    		
			    		predictions[i][k]=pred;
			    		sum=sum+ predictions[i][k];
			    	  }
			    	  
			    	  if (this.n_classes==2){
			    		  if (this.Objective.equals("logistic")){
			    		  predictions[i][0]=1-predictions[i][1];
			    		  sum+=predictions[i][0];
			    		  }else {
			    			  predictions[i][0]=-predictions[i][1];
			    		  }
			    		  
			    		 
			    	  }
			    	  if (this.Objective.equals("logistic")){
			    	  for (int k=0; k<this.n_classes; k++) {
			    		  predictions[i][k]= predictions[i][k]/sum;
			    	  
		    		  }
			    	  }
			    	
			    	  //System.out.println(Arrays.toString(predictions[i]));

			}
				
			return predictions;
		}

	@Override
	public double[] predict_probaRow(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		double predictions[]= new double [n_classes];
		double sum=0.0;
		
		
  	  for (int k=0; k<this.n_classes; k++) {
  		    
  		  if (this.n_classes==2){
  			  k++;
  		  }
 	       //hidden units in the 2nd hidden layer
  		  
  		  	smatrix vectorset=this.vectorset[k];
  		  	
				double pred=constant[k];

		    	for (int j=0; j <vectorset.GetRowDimension() ; j++){

		    		double feature=0;
		    		if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data[vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    				feature+=euc*euc;
		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 

		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    			}
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}

		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

		    		}
		    		
			    pred+=feature*betas[k][j];
		    			
		    	}	

		    	if (this.Objective.equals("logistic")){
		    		//convert to probability
		    			         
		    		 pred= 1. / (1. + Math.exp(-pred));
				    	}
		    		
		    		predictions[k]=pred;
		    		sum=sum+ predictions[k];
		    	  }
		    	  
		    	  if (this.n_classes==2){
		    		  if (this.Objective.equals("logistic")){
		    		  predictions[0]=1-predictions[1];
		    		  sum+=predictions[0];
		    		  }else {
		    			  predictions[0]=-predictions[1];
		    		  }
		    		  
		    		 
		    	  }
		    	  if (this.Objective.equals("logistic")){
		    	  for (int k=0; k<this.n_classes; k++) {
		    		  predictions[k]= predictions[k]/sum;
		    	  
	    		  }
		    	  }
		    	
		    	  //System.out.println(Arrays.toString(predictions[i]));

		
			
		return predictions;
	}


	@Override
	public double[] predict_probaRow(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (data==null || data.GetRow(rows).length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		double predictions[]= new double [n_classes];


		double sum=0.0;
		
		  for (int k=0; k<this.n_classes; k++) {
	  		    
	  		  if (this.n_classes==2){
	  			  k++;
	  		  }
	 	       //hidden units in the 2nd hidden layer
	  		  
	  		  	smatrix vectorset=this.vectorset[k];
	  		  	
					double pred=constant[k];

			    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
			    		double feature=0;
			    		if (distance.equals("RBF")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data.GetElement(rows, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    				feature+=euc*euc;
			    			}
			    			feature=Math.exp(-this.gammabfs * feature); 

			    			
			    		} else if (distance.equals("POLY")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(rows, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=(this.gammabfs*feature)+this.coeff;
			    			for (int h=0; h <this.degree-1; h++){
			    				feature*=feature;
			    			}

			    		} else if (distance.equals("SIGMOID") ){
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(rows, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

			    		}
			    		
				    pred+=feature*betas[k][j];
			    			
			    	}	
  			         
  			     

  		
			       	if (this.Objective.equals("logistic")){
			    		//convert to probability
			    			         
			    		 pred= 1. / (1. + Math.exp(-pred));
					    	}
			    		
			    		predictions[k]=pred;
			    		sum=sum+ predictions[k];
			    	  }
			    	  
			    	  if (this.n_classes==2){
			    		  if (this.Objective.equals("logistic")){
			    		  predictions[0]=1-predictions[1];
			    		  sum+=predictions[0];
			    		  }else {
			    			  predictions[0]=-predictions[1];
			    		  }
			    		  
			    		 
			    	  }
			    	  if (this.Objective.equals("logistic")){
			    	  for (int k=0; k<this.n_classes; k++) {
			    		  predictions[k]= predictions[k]/sum;
			    	  
		    		  }
			    	  }
			    	
			    	  //System.out.println(Arrays.toString(predictions[i]));

			
				
			return predictions;
		}

	@Override
	public double[] predict_probaRow(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");} 
		
		if (data==null ){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		double predictions[]= new double [n_classes];


		double sum=0.0;
		
		  for (int k=0; k<this.n_classes; k++) {
	  		    
	  		  if (this.n_classes==2){
	  			  k++;
	  		  }
	 	       //hidden units in the 2nd hidden layer
	  		  
	  		  	smatrix vectorset=this.vectorset[k];
	  		  	
					double pred=constant[k];

			    		HashMap<Integer,Integer> has_index=new HashMap<Integer,Integer>();
				    	for (int v=start; v<end;v++ ){
				    		has_index.put(data.mainelementpile[v],v);
						}
				      	for (int j=0; j <vectorset.GetRowDimension() ; j++){
				      		double feature=0;

					    if (distance.equals("RBF")) {
			   			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			   				int cc=vectorset.mainelementpile[h];
			   				Integer ks=has_index.get(cc);
			   				if (ks!=null){
			   					double euc=Scaler.transform(vectorset.valuespile[h], cc)-Scaler.transform(data.valuespile[ks], cc);
			   					feature+=euc*euc;
			   					}
			   			}
			   			feature=Math.exp(-this.gammabfs * feature); 
			   			
			   		} else if (distance.equals("POLY")) {
			   			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			   				int cc=vectorset.mainelementpile[h];
			   				Integer ks=has_index.get(cc);
			   				if (ks!=null){
			   					feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
			   				}
			   				}
			   			feature=(this.gammabfs*feature)+this.coeff;
			   			for (int h=0; h <this.degree-1; h++){
			   				feature*=feature;
			   			}
			   			
			   		} else if (distance.equals("SIGMOID") ){
			   			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			   				int cc=vectorset.mainelementpile[h];
			   				Integer ks=has_index.get(cc);
			   				if (ks!=null){
				    				feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
			   				}
			   			}
			   			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
			   		}
			   		
				    pred+=feature*betas[k][j];
			   			
			   	}	


				  		
				       	if (this.Objective.equals("logistic")){
				    		//convert to probability
				    			         
				    		 pred= 1. / (1. + Math.exp(-pred));
						    	}
				    		
				    		predictions[k]=pred;
				    		sum=sum+ predictions[k];
				    	  }
				    	  
				    	  if (this.n_classes==2){
				    		  if (this.Objective.equals("logistic")){
				    		  predictions[0]=1-predictions[1];
				    		  sum+=predictions[0];
				    		  }else {
				    			  predictions[0]=-predictions[1];
				    		  }
				    		  
				    		 
				    	  }
				    	  if (this.Objective.equals("logistic")){
				    	  for (int k=0; k<this.n_classes; k++) {
				    		  predictions[k]= predictions[k]/sum;
				    	  
			    		  }
				    	  }
				    	
				    	  //System.out.println(Arrays.toString(predictions[i]));

				
					
				return predictions;
			}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		double predictions[]= new double [data.GetRowDimension()];
		
		double temp[]= new double[this.n_classes];
		for (int i=0; i < predictions.length; i++) {
			double sum=0.0;
			
	    	  for (int k=0; k<this.n_classes; k++) {
	    		  temp[k]=0;
	    		  if (this.n_classes==2){
	    			  k++;
	    			  temp[k]=0;
	    		  }
	   	       //hidden units in the 2nd hidden layer
	    		  
	    		  	smatrix vectorset=this.vectorset[k];
	    		  	
					double pred=constant[k];

			    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
			    		double feature=0;
			    		if (distance.equals("RBF")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data.GetElement(i,vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    				feature+=euc*euc;
			    			}
			    			feature=Math.exp(-this.gammabfs * feature); 

			    			
			    		} else if (distance.equals("POLY")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=(this.gammabfs*feature)+this.coeff;
			    			for (int h=0; h <this.degree-1; h++){
			    				feature*=feature;
			    			}

			    		} else if (distance.equals("SIGMOID") ){
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

			    		}
			    		
				    pred+=feature*betas[k][j];
			    			
			    	}	
			  temp[k]=pred;		         
			  if (this.Objective.equals("logistic")){	
	    	  temp[k]=1/(1+Math.exp( -pred) );
			  }
	    	  sum=sum+ temp[k];
	    	  
	    	  
	    	  }
	    	  if (this.n_classes==2){
	    		  if (this.Objective.equals("logistic")){	
	    		  temp[0]=1-temp[1];
	    		  } else {
	    			  temp[0]=-temp[1];
	    		  }
	    
    		  }
	    	  int maxi=0;
	    	  double max=temp[0];
	    	  for (int k=1; k<n_classes; k++) {
	    		 if (temp[k]>max){
	    			 max=temp[k];
	    			 maxi=k;	 
	    		 }
	    	  }
	    	  try{
	    		  predictions[i]=Double.parseDouble(classes[maxi]);
	    	  } catch (Exception e){
	    		  predictions[i]=maxi;
	    	  }

		}
	
	return predictions;
	}

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetRowDimension());	
		}
		
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		
		double predictions[]= new double [data.GetRowDimension()];
		
		HashMap <Integer,Integer> has_index=null;
		double temp[]= new double[this.n_classes];
			for (int i=0; i < predictions.length; i++) {
				double sum=0.0;
				
		    	  for (int k=0; k<this.n_classes; k++) {
		    		  temp[k]=0;
		    		  if (this.n_classes==2){
		    			  k++;
		    			  temp[k]=0;
		    		  }
		   	       //hidden units in the 2nd hidden layer
		    		  
		    		  	smatrix vectorset=this.vectorset[k];
		    		  	
						double pred=constant[k];

				    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
				    		double feature=0;

							 has_index=new HashMap<Integer,Integer>();
						    	for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
						    		has_index.put(data.mainelementpile[v],v);
								}

							    if (distance.equals("RBF")) {
					    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
					    				int cc=vectorset.mainelementpile[h];
					    				Integer ks=has_index.get(cc);
					    				if (ks!=null){
					    					double euc=Scaler.transform(vectorset.valuespile[h], cc)-Scaler.transform(data.valuespile[ks], cc);
					    					feature+=euc*euc;
					    					}else {
						    					double euc=Scaler.transform(vectorset.valuespile[h], cc);
						    					feature+=euc*euc;
						    				}
					    			}
					    			feature=Math.exp(-this.gammabfs * feature); 
					    			
					    		} else if (distance.equals("POLY")) {
					    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
					    				int cc=vectorset.mainelementpile[h];
					    				Integer ks=has_index.get(cc);
					    				if (ks!=null){
					    					feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
					    				}
					    				}
					    			feature=(this.gammabfs*feature)+this.coeff;
					    			for (int h=0; h <this.degree-1; h++){
					    				feature*=feature;
					    			}
					    			
					    		} else if (distance.equals("SIGMOID") ){
					    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
					    				int cc=vectorset.mainelementpile[h];
					    				Integer ks=has_index.get(cc);
					    				if (ks!=null){
						    				feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
					    				}
					    			}
					    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
					    		}
					    		
						    pred+=feature*betas[k][j];
					    			
					    	}	         
	    			       
						  temp[k]=pred;		         
						  if (this.Objective.equals("logistic")){	
				    	  temp[k]=1/(1+Math.exp( -pred) );
						  }
				    	  sum=sum+ temp[k];
				    	  
				    	  
				    	  }
				    	  if (this.n_classes==2){
				    		  if (this.Objective.equals("logistic")){	
				    		  temp[0]=1-temp[1];
				    		  } else {
				    			  temp[0]=-temp[1];
				    		  }
				    
			    		  }
				    	  int maxi=0;
				    	  double max=temp[0];
				    	  for (int k=1; k<n_classes; k++) {
				    		 if (temp[k]>max){
				    			 max=temp[k];
				    			 maxi=k;	 
				    		 }
				    	  }
				    	  try{
				    		  predictions[i]=Double.parseDouble(classes[maxi]);
				    	  } catch (Exception e){
				    		  predictions[i]=maxi;
				    	  }

					}
				
				return predictions;
				}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		
		double predictions[]= new double [data.length];
		
		double temp[]= new double[this.n_classes];
		for (int i=0; i < predictions.length; i++) {
			double sum=0.0;
			
	    	  for (int k=0; k<this.n_classes; k++) {
	    		  temp[k]=0;
	    		  if (this.n_classes==2){
	    			  k++;
	    			  temp[k]=0;
	    		  }
	   	       //hidden units in the 2nd hidden layer
	    		  
	    		  	smatrix vectorset=this.vectorset[k];
	    		  	
					double pred=constant[k];

			    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
			    		double feature=0;
			    		if (distance.equals("RBF")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data[i][vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
			    				feature+=euc*euc;
			    			}
			    			feature=Math.exp(-this.gammabfs * feature); 

			    			
			    		} else if (distance.equals("POLY")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
			    			}
			    			feature=(this.gammabfs*feature)+this.coeff;
			    			for (int h=0; h <this.degree-1; h++){
			    				feature*=feature;
			    			}

			    		} else if (distance.equals("SIGMOID") ){
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
			    			}
			    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

			    		}
			    		
				    pred+=feature*betas[k][j];
			    			
			    	}	
	    			        

	    		
					  temp[k]=pred;		         
					  if (this.Objective.equals("logistic")){	
			    	  temp[k]=1/(1+Math.exp( -pred) );
					  }
			    	  sum=sum+ temp[k];
			    	  
			    	  
			    	  }
			    	  if (this.n_classes==2){
			    		  if (this.Objective.equals("logistic")){	
			    		  temp[0]=1-temp[1];
			    		  } else {
			    			  temp[0]=-temp[1];
			    		  }
			    
		    		  }
			    	  int maxi=0;
			    	  double max=temp[0];
			    	  for (int k=1; k<n_classes; k++) {
			    		 if (temp[k]>max){
			    			 max=temp[k];
			    			 maxi=k;	 
			    		 }
			    	  }
			    	  try{
			    		  predictions[i]=Double.parseDouble(classes[maxi]);
			    	  } catch (Exception e){
			    		  predictions[i]=maxi;
			    	  }

				}
			
			return predictions;
			}
	

	@Override
	public double predict_Row(double[] row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (row==null || row.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (row.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + row.length);	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		double predictions=0.0;
		double sum=0.0;
		double temp[]= new double[this.n_classes];

	    	  for (int k=0; k<this.n_classes; k++) {
	    		  temp[k]=0;
	    		  if (this.n_classes==2){
	    			  k++;
	    			  temp[k]=0;
	    		  }
	   	       //hidden units in the 2nd hidden layer
	    		  
	    		  	smatrix vectorset=this.vectorset[k];
	    		  	
					double pred=constant[k];

			    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
			    		double feature=0;
			    		if (distance.equals("RBF")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(row[vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
			    				feature+=euc*euc;
			    			}
			    			feature=Math.exp(-this.gammabfs * feature); 

			    			
			    		} else if (distance.equals("POLY")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(row[vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
			    			}
			    			feature=(this.gammabfs*feature)+this.coeff;
			    			for (int h=0; h <this.degree-1; h++){
			    				feature*=feature;
			    			}

			    		} else if (distance.equals("SIGMOID") ){
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(row[vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
			    			}
			    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

			    		}
			    		
				    pred+=feature*betas[k][j];
			    			
			    	}	
		 
			    		
					
					  temp[k]=pred;		         
					  if (this.Objective.equals("logistic")){	
			    	  temp[k]=1/(1+Math.exp( -pred) );
					  }
			    	  sum=sum+ temp[k];
			    	  
			    	  
			    	  }
			    	  if (this.n_classes==2){
			    		  if (this.Objective.equals("logistic")){	
			    		  temp[0]=1-temp[1];
			    		  } else {
			    			  temp[0]=-temp[1];
			    		  }
			    
		    		  }
			    	  int maxi=0;
			    	  double max=temp[0];
			    	  for (int k=1; k<n_classes; k++) {
			    		 if (temp[k]>max){
			    			 max=temp[k];
			    			 maxi=k;	 
			    		 }
			    	  }
			    	  try{
			    		  predictions=Double.parseDouble(classes[maxi]);
			    	  } catch (Exception e){
			    		  predictions=maxi;
			    	  }

				
			
			return predictions;
			}
	

	@Override
	public double predict_Row(fsmatrix f, int row) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		double predictions=0.0;
		double sum=0.0;
		double temp[]= new double[this.n_classes];

	    	  for (int k=0; k<this.n_classes; k++) {
	    		  temp[k]=0;
	    		  if (this.n_classes==2){
	    			  k++;
	    			  temp[k]=0;
	    		  }
	   	       //hidden units in the 2nd hidden layer
	    		  
	    		  	smatrix vectorset=this.vectorset[k];
	    		  	
					double pred=constant[k];

			    	for (int j=0; j <vectorset.GetRowDimension() ; j++){
			    		double feature=0;
			    		if (distance.equals("RBF")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(f.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    				feature+=euc*euc;
			    			}
			    			feature=Math.exp(-this.gammabfs * feature); 

			    			
			    		} else if (distance.equals("POLY")) {
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(f.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=(this.gammabfs*feature)+this.coeff;
			    			for (int h=0; h <this.degree-1; h++){
			    				feature*=feature;
			    			}

			    		} else if (distance.equals("SIGMOID") ){
			    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
			    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(f.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
			    			}
			    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

			    		}
			    		
				    pred+=feature*betas[k][j];
			    			
			    	}	
  			         
  			     		    			         
		    			   

		    	
					  temp[k]=pred;		         
					  if (this.Objective.equals("logistic")){	
			    	  temp[k]=1/(1+Math.exp( -pred) );
					  }
			    	  sum=sum+ temp[k];
			    	  
			    	  
			    	  }
			    	  if (this.n_classes==2){
			    		  if (this.Objective.equals("logistic")){	
			    		  temp[0]=1-temp[1];
			    		  } else {
			    			  temp[0]=-temp[1];
			    		  }
			    
		    		  }
			    	  int maxi=0;
			    	  double max=temp[0];
			    	  for (int k=1; k<n_classes; k++) {
			    		 if (temp[k]>max){
			    			 max=temp[k];
			    			 maxi=k;	 
			    		 }
			    	  }
			    	  try{
			    		  predictions=Double.parseDouble(classes[maxi]);
			    	  } catch (Exception e){
			    		  predictions=maxi;
			    	  }

				
			
			return predictions;
			}
	

	@Override
	public double predict_Row(smatrix f, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (classes==null || classes.length<2 || this.vectorset==null || this.vectorset.length!=classes.length || betas==null || betas.length<=0  ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}    			
		if (f==null || f.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (f.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + f.GetColumnDimension());	
		}
		if (this.gammabfs<=0){
			this.gammabfs=1.0/this.columndimension;
		}
		double predictions=0.0;
		double sum=0.0;
		double temp[]= new double[this.n_classes];

	    	  for (int k=0; k<this.n_classes; k++) {
	    		  temp[k]=0;
	    		  if (this.n_classes==2){
	    			  k++;
	    			  temp[k]=0;
	    		  }
	   	       //hidden units in the 2nd hidden layer
	    		  
	    		  	smatrix vectorset=this.vectorset[k];
	    		  	
					double pred=constant[k];

		    		HashMap<Integer,Integer> has_index=new HashMap<Integer,Integer>();
			    	for (int v=start; v<end;v++ ){
			    		has_index.put(f.mainelementpile[v],v);
					}
			      	for (int j=0; j <vectorset.GetRowDimension() ; j++){
			      		double feature=0;

				    if (distance.equals("RBF")) {
		   			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		   				int cc=vectorset.mainelementpile[h];
		   				Integer ks=has_index.get(cc);
		   				if (ks!=null){
		   					double euc=Scaler.transform(vectorset.valuespile[h], cc)-Scaler.transform(f.valuespile[ks], cc);
		   					feature+=euc*euc;
		   					}
		   			}
		   			feature=Math.exp(-this.gammabfs * feature); 
		   			
		   		} else if (distance.equals("POLY")) {
		   			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		   				int cc=vectorset.mainelementpile[h];
		   				Integer ks=has_index.get(cc);
		   				if (ks!=null){
		   					feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(f.valuespile[ks], cc);
		   				}
		   				}
		   			feature=(this.gammabfs*feature)+this.coeff;
		   			for (int h=0; h <this.degree-1; h++){
		   				feature*=feature;
		   			}
		   			
		   		} else if (distance.equals("SIGMOID") ){
		   			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		   				int cc=vectorset.mainelementpile[h];
		   				Integer ks=has_index.get(cc);
		   				if (ks!=null){
			    				feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(f.valuespile[ks], cc);
		   				}
		   			}
		   			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		   		}
		   		
			    pred+=feature*betas[k][j];
		   			
		   	}	
		    			         

					  temp[k]=pred;		         
					  if (this.Objective.equals("logistic")){	
			    	  temp[k]=1/(1+Math.exp( -pred) );
					  }
			    	  sum=sum+ temp[k];
			    	  
			    	  
			    	  }
			    	  if (this.n_classes==2){
			    		  if (this.Objective.equals("logistic")){	
			    		  temp[0]=1-temp[1];
			    		  } else {
			    			  temp[0]=-temp[1];
			    		  }
			    
		    		  }
			    	  int maxi=0;
			    	  double max=temp[0];
			    	  for (int k=1; k<n_classes; k++) {
			    		 if (temp[k]>max){
			    			 max=temp[k];
			    			 maxi=k;	 
			    		 }
			    	  }
			    	  try{
			    		  predictions=Double.parseDouble(classes[maxi]);
			    	  } catch (Exception e){
			    		  predictions=maxi;
			    	  }

				
			
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

		if (  !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD or FTRL " );	
		}		
	
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.intpasses<=0){
			this.intpasses=1; // 1 pass should be enough to find the most influential observations, still might need more at the cost of more time
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		if (this.pinter> 1 ||this.pinter<=0 ){
			throw new IllegalStateException(" Final percentage of support vectors has to be (0,1]" );
		}
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if (this.intpertokeep>=1 ||this.intpertokeep<=0 ){
			throw new IllegalStateException(" percentage_thresol has to be (0,1)" );
		}		
		if (this.intcoeffthres>=1 || this.intcoeffthres<=0 ){
			throw new IllegalStateException(" coeff_thresol has to be (0,1)" );
		}	
		if ( !distance.equals("RBF")  &&  !distance.equals("POLY") && !distance.equals("SIGMOID")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Liblinear methods" );	
		}			
		if (  this.degree<1 && this.distance.equals("POLY")  ){
			throw new IllegalStateException(" The degree level in POLY cannot be less than 1" );	
		}			
		if (  this.coeff<=0.0 && this.distance.equals("POLY")  ){
			throw new IllegalStateException(" Coefficient cannot be less than 0 in POLY" );	
		}	
		if ( !Objective.equals("logistic")  &&  !Objective.equals("svm") ){
			throw new IllegalStateException(" Objective has to be Logistic or svm." );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		// make sensible checks on the target data
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
		if ( ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
		}
		
		n_classes=classes.length;

		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta and constant
		betas= new double[n_classes][];
		constant=new double[n_classes];
		vectorset = new smatrix [this.n_classes] ;

		
		Thread[] thread_array= new Thread[threads];
		binarykernelmodel [] minimodels= new binarykernelmodel[threads];
		int count_of_live_threads=0;
		int class_passed=0;
		if (n_classes==2){
			double label []= new double [data.length];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( (target[i]+"").equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}	
			
			
			
			
			binarykernelmodel logit= new binarykernelmodel(data);
			logit.Type=this.Type;
			logit.maxim_Iteration=this.maxim_Iteration;
			logit.UseConstant=this.UseConstant;
			logit.verbose=false;
			logit.copy=false;
			logit.C=this.C;
			logit.l1C=this.l1C;	
			logit.smooth=this.smooth;	
			logit.threads=this.intthreads;
			logit.intpasses=this.intpasses;
			logit.pinter=this.pinter;
			logit.submodelcutsper=this.submodelcutsper;
			logit.intcoeffthres=this.intcoeffthres; 
			logit.intpertokeep=this.intpertokeep; 
			logit.Objective=this.Objective;
			logit.distance=this.distance;
			logit.gammabfs=this.gammabfs;
			logit.degree=this.degree;
			logit.coeff=this.coeff;
			logit.seed=this.seed;
			logit.shuffle=this.shuffle;
			logit.learn_rate=this.learn_rate;
			logit.tolerance=this.tolerance;
			logit.setScaler(this.Scaler);
			logit.usescale=true;
						
			//logit.SetBetas(betas[1], constants[1]);
			logit.target=label;
			logit.run();
			//System.out.println(Arrays.toString((vectorset[1].get(0))));
			constant[1]= logit.Getcosntant();
			this.vectorset[1]=logit.getsupportvectorset();
			double beta[]= new double [this.vectorset[1].GetRowDimension()];
			double betainv[]= new double [this.vectorset[1].GetRowDimension()];
			double bigbets []=logit.Getbetas();
			for(int b=0; b <  beta.length;b++){
				beta[b]=bigbets[b];
				betainv[b]=-beta[b];
				System.out.println(bigbets[b]);
			}
			this.betas[1]=beta;
			this.betas[0]=betainv;
			bigbets=null;			
			
		}else {	
		
		for (int n=0; n <n_classes; n++ ){
			double label []= new double [data.length];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( (target[i]+"").equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}

			binarykernelmodel logit= new binarykernelmodel(data);
			

			
			logit.Type=this.Type;
			logit.maxim_Iteration=this.maxim_Iteration;
			logit.UseConstant=this.UseConstant;
			logit.verbose=false;
			logit.copy=false;
			logit.C=this.C;
			logit.l1C=this.l1C;	
			logit.smooth=this.smooth;	
			logit.threads=this.intthreads;
			logit.intpasses=this.intpasses;
			logit.pinter=this.pinter;
			logit.submodelcutsper=this.submodelcutsper;
			logit.intcoeffthres=this.intcoeffthres; 
			logit.intpertokeep=this.intpertokeep; 
			logit.Objective=this.Objective;
			logit.distance=this.distance;
			logit.gammabfs=this.gammabfs;
			logit.degree=this.degree;
			logit.coeff=this.coeff;
			logit.seed=this.seed;
			logit.shuffle=this.shuffle;
			logit.learn_rate=this.learn_rate;
			logit.tolerance=this.tolerance;
			logit.setScaler(this.Scaler);
			logit.usescale=true;
		
			//logit.SetBetas(betas[n], constants[n]);
			logit.target=label;
			minimodels[count_of_live_threads]=logit;
			
			thread_array[count_of_live_threads]= new Thread(logit);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for class: " + classes[class_passed]);
							
						}
						thread_array[s].join();
					} catch (InterruptedException e) {
					   System.out.println(e.getMessage());
					   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
					}
					class_passed++;
				}
				class_passed-=count_of_live_threads;
			

				for (int j=0; j < count_of_live_threads; j++){
								constant[class_passed]= minimodels[j].Getcosntant();
								vectorset[class_passed]=minimodels[j].getsupportvectorset();
								double beta[]= new double [this.vectorset[class_passed].GetRowDimension()];
								double bigbets []= minimodels[j].Getbetas();
								for(int b=0; b <  beta.length;b++){
									beta[b]=bigbets[b];
								}
								this.betas[class_passed]=beta;
								bigbets=null;
								
								class_passed++;
							}
				count_of_live_threads=0;
			}
		}		
		
		}
		thread_array=null;
		minimodels=null;

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

		if (  !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD or FTRL " );	
		}		
	
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.intpasses<=0){
			this.intpasses=1; // 1 pass should be enough to find the most influential observations, still might need more at the cost of more time
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		if (this.pinter> 1 ||this.pinter<=0 ){
			throw new IllegalStateException(" Final percentage of support vectors has to be (0,1]" );
		}
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if (this.intpertokeep>=1 ||this.intpertokeep<=0 ){
			throw new IllegalStateException(" percentage_thresol has to be (0,1)" );
		}		
		if (this.intcoeffthres>=1 || this.intcoeffthres<=0 ){
			throw new IllegalStateException(" coeff_thresol has to be (0,1)" );
		}	
		if ( !distance.equals("RBF")  &&  !distance.equals("POLY") && !distance.equals("SIGMOID")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Liblinear methods" );	
		}			
		if (  this.degree<1 && this.distance.equals("POLY")  ){
			throw new IllegalStateException(" The degree level in POLY cannot be less than 1" );	
		}			
		if (  this.coeff<=0.0 && this.distance.equals("POLY")  ){
			throw new IllegalStateException(" Coefficient cannot be less than 0 in POLY" );	
		}	
		if ( !Objective.equals("logistic")  &&  !Objective.equals("svm") ){
			throw new IllegalStateException(" Objective has to be Logistic or svm." );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		// make sensible checks on the target data
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
		if ( ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
		}
		
		n_classes=classes.length;

		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		betas= new double[n_classes][];
		constant=new double[n_classes];
		vectorset = new smatrix [this.n_classes] ;
		
		
		Thread[] thread_array= new Thread[threads];
		binarykernelmodel [] minimodels= new binarykernelmodel[threads];
		int count_of_live_threads=0;
		int class_passed=0;
		if (n_classes==2){
			double label []= new double [data.GetRowDimension()];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( (target[i]+"").equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}	
			
			
			
			
			binarykernelmodel logit= new binarykernelmodel(data);
			logit.Type=this.Type;
			logit.maxim_Iteration=this.maxim_Iteration;
			logit.UseConstant=this.UseConstant;
			logit.verbose=false;
			logit.copy=false;
			logit.C=this.C;
			logit.l1C=this.l1C;	
			logit.smooth=this.smooth;	
			logit.threads=this.intthreads;
			logit.intpasses=this.intpasses;
			logit.pinter=this.pinter;
			logit.submodelcutsper=this.submodelcutsper;
			logit.intcoeffthres=this.intcoeffthres; 
			logit.intpertokeep=this.intpertokeep; 
			logit.Objective=this.Objective;
			logit.distance=this.distance;
			logit.gammabfs=this.gammabfs;
			logit.degree=this.degree;
			logit.coeff=this.coeff;
			logit.seed=this.seed;
			logit.shuffle=this.shuffle;
			logit.learn_rate=this.learn_rate;
			logit.tolerance=this.tolerance;
			logit.setScaler(this.Scaler);
			logit.usescale=true;
						
			//logit.SetBetas(betas[1], constants[1]);
			logit.target=label;
			logit.run();
			//System.out.println(Arrays.toString((vectorset[1].get(0))));
			constant[1]= logit.Getcosntant();
			System.out.println("constant: " + constant[1]);
			this.vectorset[1]=logit.getsupportvectorset();
			//System.out.println("vector set length : " + this.vectorset[1].GetRowDimension());
			//System.out.println("vector set columns : " + this.vectorset[1].GetColumnDimension());
			double beta[]= new double [this.vectorset[1].GetRowDimension()];
			double betainv[]= new double [this.vectorset[1].GetRowDimension()];
			double bigbets []=logit.Getbetas();
			for(int b=0; b <  beta.length;b++){
				beta[b]=bigbets[b];
				betainv[b]=-beta[b];
				//System.out.println(bigbets[b]);
			}
			this.betas[1]=beta;
			this.betas[0]=betainv;
			//System.out.println("the betas : " + Arrays.toString(this.betas[1]));
			bigbets=null;
			
			
		}else {	
		
		for (int n=0; n <n_classes; n++ ){
			double label []= new double [data.GetRowDimension()];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( (target[i]+"").equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}

			binarykernelmodel logit= new binarykernelmodel(data);
			

			
			logit.Type=this.Type;
			logit.maxim_Iteration=this.maxim_Iteration;
			logit.UseConstant=this.UseConstant;
			logit.verbose=false;
			logit.copy=false;
			logit.C=this.C;
			logit.l1C=this.l1C;	
			logit.smooth=this.smooth;	
			logit.threads=this.intthreads;
			logit.intpasses=this.intpasses;
			logit.pinter=this.pinter;
			logit.submodelcutsper=this.submodelcutsper;
			logit.intcoeffthres=this.intcoeffthres; 
			logit.intpertokeep=this.intpertokeep; 
			logit.Objective=this.Objective;
			logit.distance=this.distance;
			logit.gammabfs=this.gammabfs;
			logit.degree=this.degree;
			logit.coeff=this.coeff;
			logit.seed=this.seed;
			logit.shuffle=this.shuffle;
			logit.learn_rate=this.learn_rate;
			logit.tolerance=this.tolerance;
			logit.setScaler(this.Scaler);
			logit.usescale=true;
		
			//logit.SetBetas(betas[n], constants[n]);
			logit.target=label;
			minimodels[count_of_live_threads]=logit;
			
			thread_array[count_of_live_threads]= new Thread(logit);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for class: " + classes[class_passed]);
							
						}
						thread_array[s].join();
					} catch (InterruptedException e) {
					   System.out.println(e.getMessage());
					   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
					}
					class_passed++;
				}
				class_passed-=count_of_live_threads;
			
				for (int j=0; j < count_of_live_threads; j++){
					constant[class_passed]= minimodels[j].Getcosntant();
					vectorset[class_passed]=minimodels[j].getsupportvectorset();
					double beta[]= new double [this.vectorset[class_passed].GetRowDimension()];
					double bigbets []= minimodels[j].Getbetas();
					for(int b=0; b <  beta.length;b++){
						beta[b]=bigbets[b];
					}
					this.betas[class_passed]=beta;
					bigbets=null;
					
					class_passed++;
				}
				count_of_live_threads=0;
			}
		}		
		
		}
		thread_array=null;
		minimodels=null;
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

		if (  !Type.equals("SGD") && !Type.equals("FTRL")){
			throw new IllegalStateException(" Type has to be one of SGD or FTRL " );	
		}		
	
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.intpasses<=0){
			this.intpasses=1; // 1 pass should be enough to find the most influential observations, still might need more at the cost of more time
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		if (this.pinter> 1 ||this.pinter<=0 ){
			throw new IllegalStateException(" Final percentage of support vectors has to be (0,1]" );
		}
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if (this.intpertokeep>=1 ||this.intpertokeep<=0 ){
			throw new IllegalStateException(" percentage_thresol has to be (0,1)" );
		}		
		if (this.intcoeffthres>=1 || this.intcoeffthres<=0 ){
			throw new IllegalStateException(" coeff_thresol has to be (0,1)" );
		}	
		if ( !distance.equals("RBF")  &&  !distance.equals("POLY") && !distance.equals("SIGMOID")){
			throw new IllegalStateException(" Type has to be one of SGD, FTRL or Liblinear methods" );	
		}			
		if (  this.degree<1 && this.distance.equals("POLY")  ){
			throw new IllegalStateException(" The degree level in POLY cannot be less than 1" );	
		}			
		if (  this.coeff<=0.0 && this.distance.equals("POLY")  ){
			throw new IllegalStateException(" Coefficient cannot be less than 0 in POLY" );	
		}	
		if ( !Objective.equals("logistic")  &&  !Objective.equals("svm") ){
			throw new IllegalStateException(" Objective has to be Logistic or svm." );	
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		// make sensible checks on the target data
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
		if ( ( Scaler.IsFitted()==false)){
			Scaler.fit(data);
		}
		
		n_classes=classes.length;

		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		betas= new double[n_classes][];
		constant=new double[n_classes];
		vectorset = new smatrix [this.n_classes] ;

		
		Thread[] thread_array= new Thread[threads];
		binarykernelmodel [] minimodels= new binarykernelmodel[threads];
		int count_of_live_threads=0;
		int class_passed=0;
		if (n_classes==2){
			double label []= new double [data.GetRowDimension()];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( (target[i]+"").equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[1])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}	
			
			
			
			
			binarykernelmodel logit= new binarykernelmodel(data);
			logit.Type=this.Type;
			logit.maxim_Iteration=this.maxim_Iteration;
			logit.UseConstant=this.UseConstant;
			logit.verbose=false;
			logit.copy=false;
			logit.C=this.C;
			logit.l1C=this.l1C;	
			logit.smooth=this.smooth;	
			logit.threads=this.intthreads;
			logit.intpasses=this.intpasses;
			logit.pinter=this.pinter;
			logit.submodelcutsper=this.submodelcutsper;
			logit.intcoeffthres=this.intcoeffthres; 
			logit.intpertokeep=this.intpertokeep; 
			logit.Objective=this.Objective;
			logit.distance=this.distance;
			logit.gammabfs=this.gammabfs;
			logit.degree=this.degree;
			logit.coeff=this.coeff;
			logit.seed=this.seed;
			logit.shuffle=this.shuffle;
			logit.learn_rate=this.learn_rate;
			logit.tolerance=this.tolerance;
			logit.setScaler(this.Scaler);
			logit.usescale=true;
						
			//logit.SetBetas(betas[1], constants[1]);
			logit.target=label;
			logit.run();
			//System.out.println(Arrays.toString((vectorset[1].get(0))));
			constant[1]= logit.Getcosntant();
			this.vectorset[1]=logit.getsupportvectorset();
			double beta[]= new double [this.vectorset[1].GetRowDimension()];
			double betainv[]= new double [this.vectorset[1].GetRowDimension()];
			double bigbets []=logit.Getbetas();
			for(int b=0; b <  beta.length;b++){
				beta[b]=bigbets[b];
				betainv[b]=-beta[b];
				System.out.println(bigbets[b]);
			}
			this.betas[1]=beta;
			this.betas[0]=betainv;
			bigbets=null;
			
		}else {	
		
		for (int n=0; n <n_classes; n++ ){
			double label []= new double [data.GetRowDimension()];
			for (int i=0; i < label.length; i++){
				if (target!=null){
					if ( (target[i]+"").equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				} else {
					if ( (Starget[i]).equals(classes[n])){
						label[i]=1.0;
					} else {
						label[i]=-1.0;	
					}
				}
			}

			binarykernelmodel logit= new binarykernelmodel(data);
			

			
			logit.Type=this.Type;
			logit.maxim_Iteration=this.maxim_Iteration;
			logit.UseConstant=this.UseConstant;
			logit.verbose=false;
			logit.copy=false;
			logit.C=this.C;
			logit.l1C=this.l1C;	
			logit.smooth=this.smooth;	
			logit.threads=this.intthreads;
			logit.intpasses=this.intpasses;
			logit.pinter=this.pinter;
			logit.submodelcutsper=this.submodelcutsper;
			logit.intcoeffthres=this.intcoeffthres; 
			logit.intpertokeep=this.intpertokeep; 
			logit.Objective=this.Objective;
			logit.distance=this.distance;
			logit.gammabfs=this.gammabfs;
			logit.degree=this.degree;
			logit.coeff=this.coeff;
			logit.seed=this.seed;
			logit.shuffle=this.shuffle;
			logit.learn_rate=this.learn_rate;
			logit.tolerance=this.tolerance;
			logit.setScaler(this.Scaler);
			logit.usescale=true;
		
			//logit.SetBetas(betas[n], constants[n]);
			logit.target=label;
			minimodels[count_of_live_threads]=logit;
			
			thread_array[count_of_live_threads]= new Thread(logit);
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==(n_classes)-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						if (this.verbose==true){
							System.out.println("fitting for class: " + classes[class_passed]);
							
						}
						thread_array[s].join();
					} catch (InterruptedException e) {
					   System.out.println(e.getMessage());
					   throw new IllegalStateException(" algorithm was terminated due to multithreading error");
					}
					class_passed++;
				}
				class_passed-=count_of_live_threads;
			
				for (int j=0; j < count_of_live_threads; j++){
					constant[class_passed]= minimodels[j].Getcosntant();
					vectorset[class_passed]=minimodels[j].getsupportvectorset();
					double beta[]= new double [this.vectorset[class_passed].GetRowDimension()];
					double bigbets []= minimodels[j].Getbetas();
					for(int b=0; b <  beta.length;b++){
						beta[b]=bigbets[b];
					}
					this.betas[class_passed]=beta;
					bigbets=null;
					
					class_passed++;
				}
				count_of_live_threads=0;
			}
		}		
		
		}
		thread_array=null;
		minimodels=null;
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
		return "KernelmodelClassifier";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: Regularized kernel model Classifier");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);				
		System.out.println("Constant in the model: "+ this.UseConstant);
		System.out.println("Smooth value for FTLR: "+ this.smooth);			
		System.out.println("Number of passes in the try-to-find-the-support-vectors submodels: "+ this.intpasses);
		System.out.println("Final percentage(%) of best cases/observations to keep to regard as support vectors  "+ this.pinter);
		System.out.println("percentage(%) of best cases/observations to include in each submodel  "+ this.submodelcutsper);
		System.out.println("Minimum value of ridge submodel coefficients to consider for a case to be considered as support vector  "+ this.intcoeffthres);
		System.out.println("percentage(%) of best cases/observations to include after each submodel   "+ this.intcoeffthres);
		System.out.println("Objective function   "+ this.Objective);
		System.out.println("Internal threads   "+ this.intthreads);
		System.out.println("kernel type   "+ this.distance);
		System.out.println("Std for RBF   "+ this.gammabfs);
		System.out.println("degrees for POLY  "+ this.degree);
		System.out.println("Coefficient for POLY  "+ this.coeff);			
		System.out.println("Regularization value: "+ this.C);		
		System.out.println("Regularization L1 for FTLR: "+ this.l1C);			
		System.out.println("Training method: "+ this.Type);	
		System.out.println("Maximum Iterations: "+ maxim_Iteration);
		System.out.println("Learning Rate: "+ this.learn_rate);	
		System.out.println("used Scaling: True (bydefault)");			
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
		constant=null;
		betas=null;
		vectorset=null;
		n_classes=0;
		classes=null;
		smooth=0.1;
		intthreads=1;
		intpasses=1;
		pinter=0.1;
		submodelcutsper=0.01;
		smooth=0.01;
		intcoeffthres=0.1; 
		intpertokeep=0.2; 
		Objective="logistic";
		distance="RBF";
		gammabfs=0.01;
		degree=2;
		coeff=1.0;
		C=1.0;
		l1C=1.0;
		Type="SGD";
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		columndimension=0;
		shuffle=true;
		learn_rate=1.0;
		Scaler=null;
		copy=true;
		seed=1;
		random=null;
		tolerance=0.0001; 
		target=null;
		weights=null;
		verbose=true;
		
	}


	@Override
	public estimator copy() {
		KernelmodelClassifier br = new KernelmodelClassifier();
		br.constant=manipulate.copies.copies.Copy(this.constant);
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		//hard copy of the latent features
		br.vectorset = new smatrix [this.n_classes] ;
		for (int f=0; f<this.n_classes;f++){

			br.vectorset[f]=(smatrix)this.vectorset[f].Copy();
		}
		


		br.smooth=this.smooth;	
		br.intpasses=this.intpasses;
		br.pinter=this.pinter;
		br.submodelcutsper=this.submodelcutsper;
		br.smooth=this.smooth;
		br.intthreads=this.intthreads;
		br.intcoeffthres=this.intcoeffthres; 
		br.intpertokeep=this.intpertokeep; 
		br.Objective=this.Objective;
		br.distance=this.distance;
		br.gammabfs=this.gammabfs;
		br.degree=this.degree;
		br.coeff=this.coeff;
		br.C=this.C;
		br.columndimension=this.columndimension;
		br.l1C=this.l1C;
		br.Type=this.Type;
		br.threads=this.threads;
		br.UseConstant=this.UseConstant;
		br.maxim_Iteration=this.maxim_Iteration;
		br.shuffle=this.shuffle;
		br.learn_rate=this.learn_rate;
		br.Scaler=this.Scaler;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
		br.tolerance=this.tolerance; 
		br.classes=this.classes.clone();
		br.n_classes=this.n_classes;
		br.target=manipulate.copies.copies.Copy(this.target.clone());
		br.Starget=this.Starget.clone();		
		br.weights=manipulate.copies.copies.Copy(this.weights.clone());
		br.verbose=this.verbose;
		return br;
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
	public void set_params(String params){
		
		String splitted_params []=params.split(" " + "+");
		
		for (int j=0; j<splitted_params.length; j++ ){
			String mini_split []=splitted_params[j].split(":");
			if (mini_split.length>=2){
				String metric=mini_split[0];
				String value=mini_split[1];
				
				 if (metric.equals("C")) {this.C=Double.parseDouble(value);}
				else if (metric.equals("l1C")) {this.l1C=Double.parseDouble(value);}
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("UseConstant")) {this.UseConstant=(value.toLowerCase().equals("true")?true:false) ;}
				else if (metric.equals("maxim_Iteration")) {this.maxim_Iteration=Integer.parseInt(value);}
				else if (metric.equals("intpasses")) {this.intpasses=Integer.parseInt(value);}
				else if (metric.equals("pinter")) {this.pinter=Integer.parseInt(value);}
				else if (metric.equals("submodelcutsper")) {this.submodelcutsper=Double.parseDouble(value);}
				else if (metric.equals("smooth")) {this.smooth=Double.parseDouble(value);}
				else if (metric.equals("ntcoeffthres ")) {this.intcoeffthres =Double.parseDouble(value);}
				else if (metric.equals("ntpertokeep ")) {this.intpertokeep =Double.parseDouble(value);}
				else if (metric.equals("Objective")) {this.Objective=value;}
				else if (metric.equals("distance")) {this.distance=value;}
				else if (metric.equals("gammabfs")) {this.gammabfs=Double.parseDouble(value);}
				else if (metric.equals("degree")) {this.degree=Integer.parseInt(value);}
				else if (metric.equals("coeff")) {this.coeff=Double.parseDouble(value);}
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
