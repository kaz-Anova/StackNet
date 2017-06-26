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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import matrix.fsmatrix;
import matrix.smatrix;
import ml.classifier;
import ml.estimator;
import preprocess.scaling.scaler;
import preprocess.scaling.maxscaler;
import exceptions.DimensionMismatchException;

/**
 * 
 * @author mariosm
 * 
 * This model will technically try to encapsulate what a support vector machine model with kernel does, but ran in a way
 * that the kernel matrix is broken down to small sub-parts (as defined by the user in parallel) and feature selection (L2) is
 * applied to reduce the dimensionality and find the key support vectors step-wise . 
 */

public class binarykernelmodel implements estimator,classifier,Runnable {

	private static final long serialVersionUID = 830529727388893394L;
	
	

	/**
	 * Type of regularization,can be any of L2, L1 or anything else for none
	 */
	public String RegularizationType="L2";
	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=0.01;
	/**
	 * Regularization value for l1 "Follow The Regularized Leader"
	 */
	public double l1C=0.01;		
	/**
	 * Type of algorithm to use. It has to be one of  SGD, FTRL
	 */
	public String Type="SGD";
	/**
	 * True if we want to scale with highest maximum value
	 */
	public boolean scale=false;
	/**
	 * True if we want to optimise for squared distance or not
	 */
	public boolean quadratic=false;	
	/**
	 * Internal passes
	 */
	public int intpasses=1;
	/**
	 * The Final percentage of support observations required based on the initial given dataset
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
	 * The object that will hold only the crucial observations
	 */
	private matrix.smatrix vectorset;	
	/**
	 * List of rows that make it to the final model
	 */
	private int support_cases[];
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
	 * The cosntant value
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
	public binarykernelmodel(){}
	
	
	/**
	 * Default constructor for Binary Logistic Regression with double data
	 */
	public binarykernelmodel(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	/**
	 * Default constructor for Binary Logistic Regression with fsmatrix data
	 */
	public binarykernelmodel(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * Default constructor for Binary Logistic Regression with smatrix data
	 */
	public binarykernelmodel(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
	}	
	/**
	 * 
	 * @param Betas : Sets initial beta-coefficients' array
	 * @param intercept : Sets initial intercept
	 */
	public void SetBetas(double Betas [], double intercept []){
		betas= Betas;
		constant=intercept;
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
	
	/**
	 * 
	 * @return the sparse matrix of support vectors
	 */
	public smatrix getsupportvectorset(){
		if (betas==null || betas.length<=0 || this.vectorset==null){
			throw new IllegalStateException(" estimator needs to be fitted first" );
		}
		return this.vectorset;
	}

	@Override
	public double[][] predict_proba(double[][] data) {
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as the trained one: " +  columndimension + " <> " + data[0].length);	
		}
		
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}		
		double feature=0.0;
		double predictions[][]= new double [data.length][2];

			for (int i=0; i < predictions.length; i++) {
				double pred=constant[0];

		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
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
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
		    	predictions[i][1]=pred;
		    	predictions[i][0]=1-pred;
		    	}  else {
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=pred;				
				predictions[i][0]=-pred;
		    	}
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
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}		
		double feature=0.0;

		double predictions[]= new double [data.length];

			for (int i=0; i < predictions.length; i++) {
				double pred=constant[0];

		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
		    		if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data[i][vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    				feature+=euc*euc;
		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 

		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i] [vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
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
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
		    	}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=pred;

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
		
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}		
		double feature=0.0;
		double predictions[][]= new double [data.GetRowDimension()][2];

			for (int i=0; i < predictions.length; i++) {
				double pred=constant[0];

		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
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
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
		    	predictions[i][1]=pred;
		    	predictions[i][0]=1-pred;
		    	}  else {
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=pred;				
				predictions[i][0]=-pred;
		    	}
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
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}		
		double feature=0.0;

		double predictions[]= new double [data.GetRowDimension()];

			for (int i=0; i < predictions.length; i++) {
				double pred=constant[0];

		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
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
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
		    	}
				//value= 1. / (1. + Math.exp(-value));
				predictions[i]=pred;

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
		
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}		
		double feature=0.0;
		double predictions[][]= new double [data.GetRowDimension()][2];
		HashMap<Integer,Integer>has_index=null;
	
			for (int i=0; i < predictions.length; i++) {
				double pred=constant[0];
				 has_index=new HashMap<Integer,Integer>();
			    	for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
			    		has_index.put(data.mainelementpile[v],v);
					}
		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;

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
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
		    	predictions[i][1]=pred;
		    	predictions[i][0]=1-pred;
		    	}  else {
				//value= 1. / (1. + Math.exp(-value));
				predictions[i][1]=pred;				
				predictions[i][0]=-pred;
		    	}
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
		double feature=0.0;
		double pred=constant[0];

	    	for (int j=0; j <support_cases.length ; j++){
	    		feature=0;
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
	    		
		    pred+=feature*betas[j];
	    			
	    	}	
	    	if (this.Objective.equals("logistic")){
	    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
	    	predictions[1]=pred;
	    	predictions[0]=1-pred;
	    	}  else {
			//value= 1. / (1. + Math.exp(-value));
			predictions[1]=pred;				
			predictions[0]=-pred;
	    	}
		
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
		double feature=0.0;
		double pred=constant[0];

	    	for (int j=0; j <support_cases.length ; j++){
	    		feature=0;
	    		if (distance.equals("RBF")) {
	    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
	    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
	    				feature+=euc*euc;
	    			}
	    			feature=Math.exp(-this.gammabfs * feature); 

	    			
	    		} else if (distance.equals("POLY")) {
	    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
	    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
	    			}
	    			feature=(this.gammabfs*feature)+this.coeff;
	    			for (int h=0; h <this.degree-1; h++){
	    				feature*=feature;
	    			}

	    		} else if (distance.equals("SIGMOID") ){
	    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
	    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
	    			}
	    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

	    		}
	    		
		    pred+=feature*betas[j];
	    			
	    	}	
	    	if (this.Objective.equals("logistic")){
	    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
	    	predictions[1]=pred;
	    	predictions[0]=1-pred;
	    	}  else {
			//value= 1. / (1. + Math.exp(-value));
			predictions[1]=pred;				
			predictions[0]=-pred;
	    	}
		
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
		
		double pred=constant[0];
		HashMap<Integer,Integer> has_index=new HashMap<Integer,Integer>();
	    	for (int v=start; v<end;v++ ){
	    		has_index.put(data.mainelementpile[v],v);
			}
	    	for (int j=0; j <support_cases.length ; j++){
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
   		
	    pred+=feature*betas[j];
   			
   	}	
   	if (this.Objective.equals("logistic")){
   	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
   	predictions[1]=pred;
   	predictions[0]=1-pred;
   	}  else {
		//value= 1. / (1. + Math.exp(-value));
		predictions[1]=pred;				
		predictions[0]=-pred;
   	}
	

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
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}		
		double feature=0.0;

		double predictions[]= new double [data.GetRowDimension()];

			for (int i=0; i < predictions.length; i++) {
				double pred=constant[0];

		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
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
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
		    	predictions[i]=(pred >= 0.5) ? 1.0 :0.0 ;
		    	} else {
		    		predictions[i]=(pred >= 0.0) ? 1.0 :0.0 ;
		    	}

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
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}	
		double predictions[]= new double [data.GetRowDimension()];
		// check if data is sorted via row
		if (!data.IsSortedByRow()){
			data.convert_type();
		}

		double feature=0.0;
		HashMap<Integer,Integer>has_index=null;
	
			for (int i=0; i < predictions.length; i++) {
				double pred=constant[0];
				 has_index=new HashMap<Integer,Integer>();
			    	for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
			    		has_index.put(data.mainelementpile[v],v);
					}
		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;

				    if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				int cc=vectorset.mainelementpile[h];
		    				Integer ks=has_index.get(cc);
		    				if (ks!=null){
		    					double euc=Scaler.transform(vectorset.valuespile[h], cc)-Scaler.transform(data.valuespile[ks], cc);
		    					feature+=euc*euc;
		    					}
		    					else {
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
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
		    	predictions[i]=(pred >= 0.5) ? 1.0 :0.0 ;
		    	} else {
		    		predictions[i]=(pred >= 0.0) ? 1.0 :0.0 ;
		    	}
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
		if (this.vectorset==null || this.vectorset.GetRowDimension()<=0 ||this.support_cases==null || this.support_cases.length<=0 || betas==null || betas.length<=0  ){
			throw new IllegalStateException(" Model needs to be 'fit' first");	
		}	
		double predictions[]= new double [data.length];
		for (int i=0; i < predictions.length; i++) {
			double pred=constant[0];

	    	for (int j=0; j <support_cases.length ; j++){
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
	    		
		    pred+=feature*betas[j];
	    			
	    	}	
	    	if (this.Objective.equals("logistic")){
	    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
	    	predictions[i]=(pred >= 0.5) ? 1.0 :0.0 ;
	    	} else {
	    		predictions[i]=(pred >= 0.0) ? 1.0 :0.0 ;
	    	}

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
		double pred=constant[0];

    	for (int j=0; j <support_cases.length ; j++){
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
    		
	    pred+=feature*betas[j];
    			
    	}	
    	if (this.Objective.equals("logistic")){
    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
    	predictions=(pred >= 0.5) ? 1.0 :0.0 ;
    	} else {
    		predictions=(pred >= 0.0) ? 1.0 :0.0 ;
    	}
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
		double pred=constant[0];

    	for (int j=0; j <support_cases.length ; j++){
    		double feature=0;
    		if (distance.equals("RBF")) {
    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
    				feature+=euc*euc;
    			}
    			feature=Math.exp(-this.gammabfs * feature); 

    			
    		} else if (distance.equals("POLY")) {
    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
    			}
    			feature=(this.gammabfs*feature)+this.coeff;
    			for (int h=0; h <this.degree-1; h++){
    				feature*=feature;
    			}

    		} else if (distance.equals("SIGMOID") ){
    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(row, vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
    			}
    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 

    		}
    		
	    pred+=feature*betas[j];
    			
    	}	
    	if (this.Objective.equals("logistic")){
    	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
    	predictions=(pred >= 0.5) ? 1.0 :0.0 ;
    	} else {
    		predictions=(pred >= 0.0) ? 1.0 :0.0 ;
    	}
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
		double predictions=0.0;
		
		double pred=constant[0];
		HashMap<Integer,Integer> has_index=new HashMap<Integer,Integer>();
	    	for (int v=start; v<end;v++ ){
	    		has_index.put(data.mainelementpile[v],v);
			}
	    	for (int j=0; j <support_cases.length ; j++){
   		double feature=0;

		    if (distance.equals("RBF")) {
   			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
   				int cc=vectorset.mainelementpile[h];
   				Integer ks=has_index.get(cc);
   				if (ks!=null){
   					double euc=Scaler.transform(vectorset.valuespile[h], cc)-Scaler.transform(data.valuespile[ks], cc);
   					feature+=euc*euc;
   					}
   				else
   					{
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
   		
	    pred+=feature*betas[j];
	    
	    	}	
   	
    	if (this.Objective.equals("logistic")){
        	pred= 1.0/ (1.0 + Math.exp(-pred));  // logit transform
        	predictions=(pred >= 0.5) ? 1.0 :0.0 ;
        	} else {
        		predictions=(pred >= 0.0) ? 1.0 :0.0 ;
        	}
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
			Scaler.fit(data);
			
		

		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			if (betas.length!=data.length){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  data.length);
			}
		} else { //Initialise beta if not given
			betas= new double[data.length]; //notice that these are rows and not columns
			constant= new double[]{0.0};
		}
		
		/*
		 * Find the "support vectors" based on the given parameters
		 */
		
		// create list of row indices
		support_cases = new int[data.length];
		for (int g=0; g < support_cases.length;g++){
			support_cases[g]=g;
		}
		
		if (this.shuffle){
			shuffleArray(support_cases,  random);
		}
		
		int final_row_zise=(int) (pinter*support_cases.length);
		if (final_row_zise<1){
			final_row_zise=1;
		}
		int batch_size=(int) (submodelcutsper*support_cases.length);
		if (batch_size<1){
			batch_size=1;
		}	
		ArrayList<Integer> temps_ints= new ArrayList<Integer>();
		int counter=0;
		int sum=0;
		while(sum<support_cases.length){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data for pre-selecting the support vectors");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=support_cases.length;
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 
		int iters=0;
		
		while(support_cases.length>final_row_zise){
			
			
			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				
				//Initialise an svc helper model
				ridgehelper svc = new ridgehelper(data, support_cases ,  loop_list[n] ,  loop_list[n+1] , this.intpertokeep, this.intcoeffthres,Scaler,temps_ints);
				svc.C=this.C;
				svc.learn_rate=this.learn_rate;
				svc.gammabfs=this.gammabfs;
				svc.coeff=this.coeff;
				svc.distance=this.distance;
				svc.maxim_Iteration=this.intpasses;
				svc.shuffle=this.shuffle;
				svc.seed=this.seed;
				svc.target=this.target;
	
				
				
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Interaction: " + iters +  " submodel " + class_passed);
								
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
			
			iters+=1;
			
			if (this.verbose==true){
				System.out.println(" Completed Iteration : " + iters + " cases so far: " + temps_ints.size());	
			}
			
			// retrieve the array that hold the chosen so far most influential observations
			support_cases =new int [temps_ints.size()];
			

			for (int g=0; g < support_cases.length;g++){
				support_cases[g]=temps_ints.get(g);
			}
			
			if (support_cases.length<=final_row_zise){
				break;
			} else {
			
				if (this.shuffle){
					shuffleArray(support_cases,  random);
				}	
				
				temps_ints= new ArrayList<Integer>();
				counter=0;
				sum=0;
				while(sum<support_cases.length){
					sum+=batch_size;
					counter++;
				}
				if (counter<1){
					throw new IllegalStateException(" some error occured in regards to dichotomizing the data for pre-selecting the support vectors");
				}
				
				loop_list =new int [counter+1];
				loop_list[0]=0;
				sum=0;
				for (int g=0; g < counter-1;g++){
					sum+=batch_size;
					loop_list[g+1]=sum;
				}	
				loop_list[loop_list.length-1]=support_cases.length;
				thread_array= new Thread[this.threads];
				// start the loop to find the support vectors 			
			
			
			}
			// end of loop that finds the support vectors
		}
		
		
		Arrays.sort(support_cases);
		
		
		//System.out.println(Arrays.toString(support_cases));
		
		
		//<Integer> unique_cases= new HashSet<Integer> (this.columndimension);
		// We need to create the model
		vectorset=new smatrix(data,support_cases);
		if (this.verbose){
		System.out.println("Stored in memory set has : " + vectorset.GetRowDimension() + " Support Vectors and  " + vectorset.GetColumnDimension() + " columns");
		}
		//vectorset.Print(10);
		 
		if (this.gammabfs<=0  ){
			this.gammabfs=1.0/((double)this.columndimension); //gamma
		}
		
		if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			double residual=0.0;
			double features []= new double [support_cases.length];
			

		    	
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[support_cases.length]; // sum of squared gradients
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.length; k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.length);
		    	}
		    	/*
				double BETA [] = new double [betas.length];
				double Constant=0.0;
		    	double pred=0;
		    	*/
		    	double pred=constant[0];
		    	double yi=0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	
		    	// compute score
		    	// run the coefficients' loop
		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
		    		//unique_cases= new HashSet<Integer> (this.columndimension);
		    		if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				//unique_cases.add(vectorset.mainelementpile[h]);
		    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data[i][ vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    				feature+=euc*euc;
		    			}
//		    			for (int f=0; f <this.columndimension ; f++){
//		    				if (!unique_cases.contains(f)){
//		    					double euc=Scaler.transform(data.GetElement(i, f), f);
//		    					feature+=euc*euc;
//		    				}
//		    				
//		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j]=feature;
		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][ vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    			}
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}
		    			
		    			features[j]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][ vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j]=feature;
		    		}
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));  // logit transform
		    	}
		    	//System.out.println(Arrays.toString(features));
		    	//we need to check if pred*yi is > 1
		    	//if  pred*yi < 1 then the gradient is -yixi + C*beta
		    	// if pred*yi >=1 then the gradient is C*beta


		    	residual=(pred-yi);
		    	
		    	if (target[i]<0){
		    		yi=-1;
		    	}

		    	// we update constant gradient
		    	double gradient=residual;
		    	
		    	if (this.Objective.equals("svm")){
		       		if ( pred*yi<1) {
		       		gradient=-yi;
		       		} else {
		       			gradient=0;
		       		}
		       	}

		    	gradient+=C*constant[0];
		    	nc+=gradient*gradient;
		    	
		        double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);

		    	constant[0]-=move; 
		    	 
		    	 
		    	for (int j=0; j < features.length;j++){
		    		 	    		 
			    		feature=features[j];
			    		if (feature==0.0){
			    			continue;	
			    		} 
			    		gradient=residual*feature;
			    		
				       	if (this.Objective.equals("svm")){
				       		if ( pred*yi<1) {
				       		gradient=-yi*feature;
				       		} else {
				       			gradient=0;
				       		}
				       	}
		    			gradient+=C*betas[j];
		    			 n[j]+=gradient*gradient;
			    		 move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
			    		 
			    		 betas[j]-=move;
		    		 	
		    	 }
		    
		    }


	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			// end of SGD
		} else if (Type.equals("FTRL")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.l1C<=0){
				throw new IllegalStateException(" L1C regularizer cannot be less equal to zero");
			}			
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[support_cases.length]; // sum of squared gradients
			double features []= new double [support_cases.length];
			double BETA [] = new double[support_cases.length];
	    	double pred=0.0;
	    	double feature=0.0;
	    	double Constant=0.0;
	    	double residual=0.0;
	    	
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.length; k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.length);
		    	}
		    	//System.out.println(Arrays.toString(data[i]));
		    	

		    	 pred=0.0;
		    	 Constant=0.0;
		    	
		    		double sign=1;
		    		
		    		if (constant[0]<0){
		    			sign=-1;
		    		}
				    	 if (sign * constant[0]  <= l1C){				    		 
				    		 Constant=0 ;				    		 
				    	 } else {				    		 
				    		 Constant= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	
		    	
		    	// other features
			    for (int j=0; j < support_cases.length; j++){	
			    	
		    		feature=0;
		    		//unique_cases= new HashSet<Integer> (this.columndimension);
		    		if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				//unique_cases.add(vectorset.mainelementpile[h]);
		    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data[i][ vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    				feature+=euc*euc;
		    			}
//		    			for (int f=0; f <this.columndimension ; f++){
//		    				if (!unique_cases.contains(f)){
//		    					double euc=Scaler.transform(data.GetElement(i, f), f);
//		    					feature+=euc*euc;
//		    				}
//		    				
//		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j]=feature;
		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][ vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    			}
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}
		    			
		    			features[j]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data[i][ vectorset.mainelementpile[h]], vectorset.mainelementpile[h]);
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j]=feature;
		    		}
			    	sign=1.0;			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    
			    pred+= BETA[j]*feature;	
			    	
			    }			     
			    
			    if (this.Objective.equals("logistic")){
			    	pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));  // logit transform
			    	}
			    
		    	double yi=0.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	residual=(pred-yi);
		    	
		    	if (target[i]<0){
		    		yi=-1;
		    	}
			    
		    	double gradient=residual;
		    	
		    	if (this.Objective.equals("svm")){
		       		if ( pred*yi<1) {
		       		gradient=-yi;
		       		} else {
		       			gradient=0;
		       		}
		       	}
				 
					                    
				double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

				constant[0]+=gradient-move*Constant;
				nc+=gradient*gradient;
				 
			    //updates
				//print.Print(betas, 5);
				for (int j=0; j < features.length;j++){
	 	    		 
		    		feature=features[j];
		    		if (feature==0.0){
		    			continue;	
		    		} 
		    		double gradientx=residual*feature;
		    		
			       	if (this.Objective.equals("svm")){
			       		if ( pred*yi<1) {
			       			gradientx=-yi*feature;
			       		} else {
			       			gradientx=0;
			       		}
			       	}
			       	gradientx+=C*betas[j];
			    	//System.out.println(" gradient: " + gradientx);
			        move=(Math.sqrt(n[j] + gradientx * gradientx) - Math.sqrt(n[j])) / this.learn_rate;
			    	betas[j] += gradientx - move * BETA[j];
                    n[j] += gradientx * gradientx;	

                    // check for highest tolerance
      				 if (Math.abs(BETA[j])>iteration_tol){
							 iteration_tol= Math.abs(BETA[j]);
						 }

			    }
			  	
		    }	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			// reset coeffs
	    		
	    		double sign=1;
	    		
	    		if (constant[0]<0){
	    			sign=-1;
	    		}
			    	 if (sign * constant[0]  <= l1C){				    		 
			    		 constant[0]=0 ;				    		 
			    	 } else {				    		 
			    		 constant[0]= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
			    	 }				    	    		
	    	
	    	// other features
		    for (int j=0; j < features.length; j++){	
		    	 sign=1.0;			    	
		    	if (betas[j]  <0){
		    		sign=-1.0;
		    	}
		    	 if (sign * betas[j]  <= l1C){
		    		 betas[j]=0 ;
		    	 } else {
		    		 betas[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
		    	 }

		    	
		    }				
			
			
			// end of FTRL
		}

	}
	@Override
	public void fit(fsmatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
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
			Scaler.fit(data);
			
		

		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			if (betas.length!=data.GetRowDimension()){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  data.GetRowDimension());
			}
		} else { //Initialise beta if not given
			betas= new double[data.GetRowDimension()]; //notice that these are rows and not columns
			constant= new double[]{0.0};
		}
		
		/*
		 * Find the "support vectors" based on the given parameters
		 */
		
		// create list of row indices
		support_cases = new int[data.GetRowDimension()];
		for (int g=0; g < support_cases.length;g++){
			support_cases[g]=g;
		}
		
		if (this.shuffle){
			shuffleArray(support_cases,  random);
		}
		
		int final_row_zise=(int) (pinter*support_cases.length);
		if (final_row_zise<1){
			final_row_zise=1;
		}
		int batch_size=(int) (submodelcutsper*support_cases.length);
		if (batch_size<1){
			batch_size=1;
		}	
		ArrayList<Integer> temps_ints= new ArrayList<Integer>();
		int counter=0;
		int sum=0;
		while(sum<support_cases.length){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data for pre-selecting the support vectors");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=support_cases.length;
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 
		int iters=0;
		
		while(support_cases.length>final_row_zise){
			
			
			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				
				//Initialise an svc helper model
				ridgehelper svc = new ridgehelper(data, support_cases ,  loop_list[n] ,  loop_list[n+1] , this.intpertokeep, this.intcoeffthres,Scaler,temps_ints);
				svc.C=this.C;
				svc.learn_rate=this.learn_rate;
				svc.gammabfs=this.gammabfs;
				svc.coeff=this.coeff;
				svc.distance=this.distance;
				svc.maxim_Iteration=this.intpasses;
				svc.shuffle=this.shuffle;
				svc.seed=this.seed;
				svc.target=this.target;
	
				
				
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Interaction: " + iters +  " submodel " + class_passed);
								
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
			
			iters+=1;
			
			if (this.verbose==true){
				System.out.println(" Completed Iteration : " + iters + " cases so far: " + temps_ints.size());	
			}
			
			// retrieve the array that hold the chosen so far most influential observations
			support_cases =new int [temps_ints.size()];
			

			for (int g=0; g < support_cases.length;g++){
				support_cases[g]=temps_ints.get(g);
			}
			
			if (support_cases.length<=final_row_zise){
				break;
			} else {
			
				if (this.shuffle){
					shuffleArray(support_cases,  random);
				}	
				
				temps_ints= new ArrayList<Integer>();
				counter=0;
				sum=0;
				while(sum<support_cases.length){
					sum+=batch_size;
					counter++;
				}
				if (counter<1){
					throw new IllegalStateException(" some error occured in regards to dichotomizing the data for pre-selecting the support vectors");
				}
				
				loop_list =new int [counter+1];
				loop_list[0]=0;
				sum=0;
				for (int g=0; g < counter-1;g++){
					sum+=batch_size;
					loop_list[g+1]=sum;
				}	
				loop_list[loop_list.length-1]=support_cases.length;
				thread_array= new Thread[this.threads];
				// start the loop to find the support vectors 			
			
			
			}
			// end of loop that finds the support vectors
		}
		
		
		Arrays.sort(support_cases);
		
		
		//System.out.println(Arrays.toString(support_cases));
		
		
		//<Integer> unique_cases= new HashSet<Integer> (this.columndimension);
		// We need to create the model
		vectorset=new smatrix(data,support_cases);
		if (this.verbose){
		System.out.println("Stored in memory set has : " + vectorset.GetRowDimension() + " Support Vectors and  " + vectorset.GetColumnDimension() + " columns");
		}
		//vectorset.Print(10);
		 
		if (this.gammabfs<=0  ){
			this.gammabfs=1.0/((double)this.columndimension); //gamma
		}
		
		if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			double residual=0.0;
			double features []= new double [support_cases.length];
			

		    	
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[support_cases.length]; // sum of squared gradients
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	/*
				double BETA [] = new double [betas.length];
				double Constant=0.0;
		    	double pred=0;
		    	*/
		    	double pred=constant[0];
		    	double yi=0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	
		    	// compute score
		    	// run the coefficients' loop
		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
		    		//unique_cases= new HashSet<Integer> (this.columndimension);
		    		if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				//unique_cases.add(vectorset.mainelementpile[h]);
		    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
		    				feature+=euc*euc;
		    			}
//		    			for (int f=0; f <this.columndimension ; f++){
//		    				if (!unique_cases.contains(f)){
//		    					double euc=Scaler.transform(data.GetElement(i, f), f);
//		    					feature+=euc*euc;
//		    				}
//		    				
//		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j]=feature;
		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
		    			}
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}
		    			
		    			features[j]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j]=feature;
		    		}
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));  // logit transform
		    	}
		    	//System.out.println(Arrays.toString(features));
		    	//we need to check if pred*yi is > 1
		    	//if  pred*yi < 1 then the gradient is -yixi + C*beta
		    	// if pred*yi >=1 then the gradient is C*beta


		    	residual=(pred-yi);
		    	
		    	if (target[i]<0){
		    		yi=-1;
		    	}

		    	// we update constant gradient
		    	double gradient=residual;
		    	
		    	if (this.Objective.equals("svm")){
		       		if ( pred*yi<1) {
		       		gradient=-yi;
		       		} else {
		       			gradient=0;
		       		}
		       	}

		    	gradient+=C*constant[0];
		    	nc+=gradient*gradient;
		    	
		        double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);

		    	constant[0]-=move; 
		    	 
		    	 
		    	for (int j=0; j < features.length;j++){
		    		 	    		 
			    		feature=features[j];
			    		if (feature==0.0){
			    			continue;	
			    		} 
			    		gradient=residual*feature;
			    		
				       	if (this.Objective.equals("svm")){
				       		if ( pred*yi<1) {
				       		gradient=-yi*feature;
				       		} else {
				       			gradient=0;
				       		}
				       	}
		    			gradient+=C*betas[j];
		    			 n[j]+=gradient*gradient;
			    		 move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
			    		 
			    		 betas[j]-=move;
		    		 	
		    	 }
		    
		    }


	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			// end of SGD
		} else if (Type.equals("FTRL")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.l1C<=0){
				throw new IllegalStateException(" L1C regularizer cannot be less equal to zero");
			}			
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[support_cases.length]; // sum of squared gradients
			double features []= new double [support_cases.length];
			double BETA [] = new double[support_cases.length];
	    	double pred=0.0;
	    	double feature=0.0;
	    	double Constant=0.0;
	    	double residual=0.0;
	    	
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	//System.out.println(Arrays.toString(data[i]));
		    	

		    	 pred=0.0;
		    	 Constant=0.0;
		    	
		    		double sign=1;
		    		
		    		if (constant[0]<0){
		    			sign=-1;
		    		}
				    	 if (sign * constant[0]  <= l1C){				    		 
				    		 Constant=0 ;				    		 
				    	 } else {				    		 
				    		 Constant= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	
		    	
		    	// other features
			    for (int j=0; j < support_cases.length; j++){	
			    	
		    		feature=0;
		    		//unique_cases= new HashSet<Integer> (this.columndimension);
		    		if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				//unique_cases.add(vectorset.mainelementpile[h]);
		    				double euc=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])-Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
		    				feature+=euc*euc;
		    			}
//		    			for (int f=0; f <this.columndimension ; f++){
//		    				if (!unique_cases.contains(f)){
//		    					double euc=Scaler.transform(data.GetElement(i, f), f);
//		    					feature+=euc*euc;
//		    				}
//		    				
//		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j]=feature;
		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
		    			}
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}
		    			
		    			features[j]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				feature+=Scaler.transform(vectorset.valuespile[h], vectorset.mainelementpile[h])*Scaler.transform(data.GetElement(i,  vectorset.mainelementpile[h]), vectorset.mainelementpile[h]);
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j]=feature;
		    		}
			    	sign=1.0;			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    
			    pred+= BETA[j]*feature;	
			    	
			    }			     
			    
			    if (this.Objective.equals("logistic")){
			    	pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));  // logit transform
			    	}
			    
		    	double yi=0.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	residual=(pred-yi);
		    	
		    	if (target[i]<0){
		    		yi=-1;
		    	}
			    
		    	double gradient=residual;
		    	
		    	if (this.Objective.equals("svm")){
		       		if ( pred*yi<1) {
		       		gradient=-yi;
		       		} else {
		       			gradient=0;
		       		}
		       	}
				 
					                    
				double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

				constant[0]+=gradient-move*Constant;
				nc+=gradient*gradient;
				 
			    //updates
				//print.Print(betas, 5);
				for (int j=0; j < features.length;j++){
	 	    		 
		    		feature=features[j];
		    		if (feature==0.0){
		    			continue;	
		    		} 
		    		double gradientx=residual*feature;
		    		
			       	if (this.Objective.equals("svm")){
			       		if ( pred*yi<1) {
			       			gradientx=-yi*feature;
			       		} else {
			       			gradientx=0;
			       		}
			       	}
			       	gradientx+=C*betas[j];
			    	//System.out.println(" gradient: " + gradientx);
			        move=(Math.sqrt(n[j] + gradientx * gradientx) - Math.sqrt(n[j])) / this.learn_rate;
			    	betas[j] += gradientx - move * BETA[j];
                    n[j] += gradientx * gradientx;	

                    // check for highest tolerance
      				 if (Math.abs(BETA[j])>iteration_tol){
							 iteration_tol= Math.abs(BETA[j]);
						 }

			    }
			  	
		    }	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			// reset coeffs
	    		
	    		double sign=1;
	    		
	    		if (constant[0]<0){
	    			sign=-1;
	    		}
			    	 if (sign * constant[0]  <= l1C){				    		 
			    		 constant[0]=0 ;				    		 
			    	 } else {				    		 
			    		 constant[0]= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
			    	 }				    	    		
	    	
	    	// other features
		    for (int j=0; j < features.length; j++){	
		    	 sign=1.0;			    	
		    	if (betas[j]  <0){
		    		sign=-1.0;
		    	}
		    	 if (sign * betas[j]  <= l1C){
		    		 betas[j]=0 ;
		    	 } else {
		    		 betas[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
		    	 }

		    	
		    }				
			
			
			// end of FTRL
		}

	}
	

	@Override
	public void fit(smatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
//		if (Type.equals("Routine") && !this.RegularizationType.equals("L2") ){
//			throw new IllegalStateException(" Routine Optimization method supports only L2 regularization" );
//		}
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
			Scaler.fit(data);
			
		

		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta
		if (betas!=null && betas.length>=1 ){ // check if a set of betas is already given e.g. threads
			if (betas.length!=data.GetRowDimension()){
				throw new IllegalStateException(" The pre-given betas do not have the same dimension with the current data. e.g " + betas.length + "<> " +  data.GetRowDimension());
			}
		} else { //Initialise beta if not given
			betas= new double[data.GetRowDimension()]; //notice that these are rows and not columns
			constant= new double[]{0.0};
		}
		
		/*
		 * Find the "support vectors" based on the given parameters
		 */
		
		// create list of row indices
		support_cases = new int[data.GetRowDimension()];
		for (int g=0; g < support_cases.length;g++){
			support_cases[g]=g;
		}
		
		if (this.shuffle){
			shuffleArray(support_cases,  random);
		}
		
		int final_row_zise=(int) (pinter*support_cases.length);
		if (final_row_zise<1){
			final_row_zise=1;
		}
		int batch_size=(int) (submodelcutsper*support_cases.length);
		if (batch_size<1){
			batch_size=1;
		}	
		ArrayList<Integer> temps_ints= new ArrayList<Integer>();
		int counter=0;
		int sum=0;
		while(sum<support_cases.length){
			sum+=batch_size;
			counter++;
		}
		if (counter<1){
			throw new IllegalStateException(" some error occured in regards to dichotomizing the data for pre-selecting the support vectors");
		}
		
		int loop_list []=new int [counter+1];
		loop_list[0]=0;
		sum=0;
		for (int g=0; g < counter-1;g++){
			sum+=batch_size;
			loop_list[g+1]=sum;
		}	
		loop_list[loop_list.length-1]=support_cases.length;
		Thread[] thread_array= new Thread[this.threads];
		// start the loop to find the support vectors 
		int iters=0;
		
		while(support_cases.length>final_row_zise){
			
			
			int count_of_live_threads=0;
			int class_passed=0;
			for (int n=0; n <loop_list.length-1; n++ ){
				
				
				//Initialise an svc helper model
				ridgehelper svc = new ridgehelper(data, support_cases ,  loop_list[n] ,  loop_list[n+1] , this.intpertokeep, this.intcoeffthres,Scaler,temps_ints);
				svc.C=this.C;
				svc.learn_rate=this.learn_rate;
				svc.gammabfs=this.gammabfs;
				svc.coeff=this.coeff;
				svc.distance=this.distance;
				svc.maxim_Iteration=this.intpasses;
				svc.shuffle=this.shuffle;
				svc.seed=this.seed;
				svc.target=this.target;
	
				
				
				thread_array[count_of_live_threads]= new Thread(svc);
				thread_array[count_of_live_threads].start();
				count_of_live_threads++;
				if (count_of_live_threads==threads || n==(loop_list.length-1)-1){
					for (int s=0; s <count_of_live_threads;s++ ){
						try {
							if (this.verbose==true){
								System.out.println("Interaction: " + iters +  " submodel " + class_passed);
								
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
			
			iters+=1;
			
			if (this.verbose==true){
				System.out.println(" Completed Iteration : " + iters + " cases so far: " + temps_ints.size());	
			}
			
			// retrieve the array that hold the chosen so far most influential observations
			support_cases =new int [temps_ints.size()];
			

			for (int g=0; g < support_cases.length;g++){
				support_cases[g]=temps_ints.get(g);
			}
			
			if (support_cases.length<=final_row_zise){
				break;
			} else {
			
				if (this.shuffle){
					shuffleArray(support_cases,  random);
				}	
				
				temps_ints= new ArrayList<Integer>();
				counter=0;
				sum=0;
				while(sum<support_cases.length){
					sum+=batch_size;
					counter++;
				}
				if (counter<1){
					throw new IllegalStateException(" some error occured in regards to dichotomizing the data for pre-selecting the support vectors");
				}
				
				loop_list =new int [counter+1];
				loop_list[0]=0;
				sum=0;
				for (int g=0; g < counter-1;g++){
					sum+=batch_size;
					loop_list[g+1]=sum;
				}	
				loop_list[loop_list.length-1]=support_cases.length;
				thread_array= new Thread[this.threads];
				// start the loop to find the support vectors 			
			
			
			}
			// end of loop that finds the support vectors
		}
		
		
		Arrays.sort(support_cases);
		
		
		System.out.println(Arrays.toString(support_cases));
	
		
		//<Integer> unique_cases= new HashSet<Integer> (this.columndimension);
		// We need to create the model
		vectorset=data.makesubmatrix(support_cases);
		if (this.verbose){
		System.out.println("Stored in memory set has : " + vectorset.GetRowDimension() + " Support Vectors and  " + vectorset.GetColumnDimension() + " columns");
		}
		//vectorset.Print(10);
		 
		if (this.gammabfs<=0  ){
			this.gammabfs=1.0/((double)this.columndimension); //gamma
		}
		
		if (Type.equals("SGD")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			double residual=0.0;
			double features []= new double [support_cases.length];

			HashMap<Integer, Integer> has_index=null;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[support_cases.length]; // sum of squared gradients
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	/*
				double BETA [] = new double [betas.length];
				double Constant=0.0;
		    	double pred=0;
		    	*/
		    	double pred=constant[0];
		    	double yi=0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	has_index=new HashMap<Integer,Integer>();
		    	for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
		    		has_index.put(data.mainelementpile[v],v);
				}	
		    	
		    	// compute score
		    	// run the coefficients' loop
		    	for (int j=0; j <support_cases.length ; j++){
		    		feature=0;
		    		//unique_cases= new HashSet<Integer> (this.columndimension);
		    		if (distance.equals("RBF")) {
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				//unique_cases.add(vectorset.mainelementpile[h]);
		    				int cc=vectorset.mainelementpile[h];
		    				Integer ks=has_index.get(cc);
		    				if (ks!=null){
		    					double euc=Scaler.transform(vectorset.valuespile[h], cc)-Scaler.transform(data.valuespile[ks], cc);
		    					feature+=euc*euc;
		    					}
		    				else {
		    					double euc=Scaler.transform(vectorset.valuespile[h], cc);
		    					feature+=euc*euc;
		    				}
		    			}
//		    			for (int f=0; f <this.columndimension ; f++){
//		    				if (!unique_cases.contains(f)){
//		    					double euc=Scaler.transform(data.GetElement(i, f), f);
//		    					feature+=euc*euc;
//		    				}
//		    				
//		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j]=feature;
		    			
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
		    			
		    			features[j]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				int cc=vectorset.mainelementpile[h];
		    				Integer ks=has_index.get(cc);
		    				if (ks!=null){
			    				feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
		    				}
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j]=feature;
		    		}
		    		
			    pred+=feature*betas[j];
		    			
		    	}	
		    	if (this.Objective.equals("logistic")){
		    	pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));  // logit transform
		    	}
		    	//System.out.println(Arrays.toString(features));
		    	//we need to check if pred*yi is > 1
		    	//if  pred*yi < 1 then the gradient is -yixi + C*beta
		    	// if pred*yi >=1 then the gradient is C*beta


		    	residual=(pred-yi);
		    	
		    	if (target[i]<0){
		    		yi=-1;
		    	}
	
		    	// we update constant gradient
		    	double gradient=residual;
		    	
		    	if (this.Objective.equals("svm")){
		       		if ( pred*yi<1) {
		       		gradient=-yi;
		       		} else {
		       			gradient=0;
		       		}
		       	}

		    	gradient+=C*constant[0];
		    	nc+=gradient*gradient;
		    	
		        double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);

		    	constant[0]-=move; 
		    	 
		    	 
		    	for (int j=0; j < features.length;j++){
		    		 	    		 
			    		feature=features[j];
			    		if (feature==0.0){
			    			continue;	
			    		} 
			    		gradient=residual*feature;
			    		
				       	if (this.Objective.equals("svm")){
				       		if ( pred*yi<1) {
				       		gradient=-yi*feature;
				       		} else {
				       			gradient=0;
				       		}
				       	}
		    			gradient+=C*betas[j];
		    			 n[j]+=gradient*gradient;
			    		 move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
			    		 
			    		 betas[j]-=move;
		    		 	
		    	 }
		    
		    }


	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			
			// end of SGD
		} else if (Type.equals("FTRL")){
			
			// based on Tingru's code
			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.l1C<=0){
				throw new IllegalStateException(" L1C regularizer cannot be less equal to zero");
			}			
			
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			//initiali
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double n []= new double[support_cases.length]; // sum of squared gradients
			double features []= new double [support_cases.length];
			double BETA [] = new double[support_cases.length];
	    	double pred=0.0;
	    	double feature=0.0;
	    	double Constant=0.0;
	    	double residual=0.0;
	    	HashMap<Integer, Integer> has_index=null;
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt(data.GetRowDimension());
		    	}
		    	//System.out.println(Arrays.toString(data[i]));
		    	has_index=new HashMap<Integer,Integer>();
		    	for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
		    		has_index.put(data.mainelementpile[v],v);
				}

		    	 pred=0.0;
		    	 Constant=0.0;
		    	
		    		double sign=1;
		    		
		    		if (constant[0]<0){
		    			sign=-1;
		    		}
				    	 if (sign * constant[0]  <= l1C){				    		 
				    		 Constant=0 ;				    		 
				    	 } else {				    		 
				    		 Constant= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
				    	 }				    
				    pred+=Constant;		    		
		    	
		    	
		    	// other features
			    for (int j=0; j < support_cases.length; j++){	
			    	
		    		feature=0;
		    		//unique_cases= new HashSet<Integer> (this.columndimension);
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
//		    			for (int f=0; f <this.columndimension ; f++){
//		    				if (!unique_cases.contains(f)){
//		    					double euc=Scaler.transform(data.GetElement(i, f), f);
//		    					feature+=euc*euc;
//		    				}
//		    				
//		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j]=feature;
		    			
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
		    			
		    			features[j]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=vectorset.indexpile[j]; h <vectorset.indexpile[j+1]; h++){
		    				int cc=vectorset.mainelementpile[h];
		    				Integer ks=has_index.get(cc);
		    				if (ks!=null){
			    				feature+=Scaler.transform(vectorset.valuespile[h], cc)*Scaler.transform(data.valuespile[ks], cc);
		    				}
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j]=feature;
		    		}
			    	sign=1.0;			    	
			    	if (betas[j]  <0){
			    		sign=-1.0;
			    	}
			    	 if (sign * betas[j]  <= l1C){
			    		 BETA[j]=0 ;
			    	 } else {
			    		 BETA[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
			    	
			    	 }
			    
			    pred+= BETA[j]*feature;	
			    	
			    }			     
			    
			    if (this.Objective.equals("logistic")){
			    	pred= 1.0/ (1.0 + Math.exp(-Math.max(Math.min(pred, 35.0), -35.0)));  // logit transform
			    	}
			    
		    	double yi=0.0;
		    	if (target[i]>0){
		    		yi=1.0;
		    	}
		    	residual=(pred-yi);
		    	
		    	if (target[i]<0){
		    		yi=-1;
		    	}
			    
		    	double gradient=residual;
		    	
		    	if (this.Objective.equals("svm")){
		       		if ( pred*yi<1) {
		       		gradient=-yi;
		       		} else {
		       			gradient=0;
		       		}
		       	}
				 
					                    
				double move=(Math.sqrt(nc + gradient * gradient) - Math.sqrt(nc)) / this.learn_rate;

				constant[0]+=gradient-move*Constant;
				nc+=gradient*gradient;
				 
			    //updates
				//print.Print(betas, 5);
				for (int j=0; j < features.length;j++){
	 	    		 
		    		feature=features[j];
		    		if (feature==0.0){
		    			continue;	
		    		} 
		    		double gradientx=residual*feature;
		    		
			       	if (this.Objective.equals("svm")){
			       		if ( pred*yi<1) {
			       			gradientx=-yi*feature;
			       		} else {
			       			gradientx=0;
			       		}
			       	}
			       	gradientx+=C*betas[j];
			    	//System.out.println(" gradient: " + gradientx);
			        move=(Math.sqrt(n[j] + gradientx * gradientx) - Math.sqrt(n[j])) / this.learn_rate;
			    	betas[j] += gradientx - move * BETA[j];
                    n[j] += gradientx * gradientx;	

                    // check for highest tolerance
      				 if (Math.abs(BETA[j])>iteration_tol){
							 iteration_tol= Math.abs(BETA[j]);
						 }

			    }
			  	
		    }	
	           //end of while
	            it++; 
	    		if (verbose){
	    					System.out.println("iteration: " + it);
	    				}
			}
			// reset coeffs
	    		
	    		double sign=1;
	    		
	    		if (constant[0]<0){
	    			sign=-1;
	    		}
			    	 if (sign * constant[0]  <= l1C){				    		 
			    		 constant[0]=0 ;				    		 
			    	 } else {				    		 
			    		 constant[0]= (sign * l1C -constant[0]) / ((0.1 + Math.sqrt(nc)) / this.learn_rate + C) ;
			    	 }				    	    		
	    	
	    	// other features
		    for (int j=0; j < features.length; j++){	
		    	 sign=1.0;			    	
		    	if (betas[j]  <0){
		    		sign=-1.0;
		    	}
		    	 if (sign * betas[j]  <= l1C){
		    		 betas[j]=0 ;
		    	 } else {
		    		 betas[j]=  (sign * l1C - betas[j]) / (( 0.1+Math.sqrt(n[j])) / this.learn_rate + C);
		    	
		    	 }

		    	
		    }				
			
			
			// end of FTRL
		}

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
		return "binarykernelmodel";
	}
	@Override
	public void PrintInformation() {
		
		System.out.println("Classifier: Binary Regularized kernelregression model");
		System.out.println("Classes: 2 (Binary)");
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);		
		System.out.println("Constant in the model: "+ this.UseConstant);
		System.out.println("Number of passes in the try-to-find-the-support-vectors submodels: "+ this.intpasses);
		System.out.println("Final percentage(%) of best cases/observations to keep to regard as support vectors  "+ this.pinter);
		System.out.println("percentage(%) of best cases/observations to include in each submodel  "+ this.submodelcutsper);
		System.out.println("Minimum value of ridge submodel coefficients to consider for a case to be considered as support vector  "+ this.intcoeffthres);
		System.out.println("percentage(%) of best cases/observations to include after each submodel   "+ this.intcoeffthres);
		System.out.println("Objective function   "+ this.Objective);
		System.out.println("kernel type   "+ this.distance);
		System.out.println("Std for RBF   "+ this.gammabfs);
		System.out.println("degrees for POLY  "+ this.degree);
		System.out.println("Coefficient for POLY  "+ this.coeff);	
		System.out.println("Regularization value: "+ this.C);
		System.out.println("Regularization L1 for FTLR: "+ this.l1C);		
		System.out.println("Training method: "+ this.Type);	
		System.out.println("Maximum Iterations: "+ maxim_Iteration);
		System.out.println("Learning Rate: "+ this.learn_rate);	
		System.out.println("smooth Raparameter for sgd : "+ this.smooth);	
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
		RegularizationType="L2";
		C=1.0;
		l1C=1.0;
		Type="SGD";
		threads=1;
		UseConstant=true;
		maxim_Iteration=-1;
		usescale=true;
		shuffle=true;
		learn_rate=0.1;
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
		support_cases=null;
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
		binarykernelmodel br = new binarykernelmodel();
		br.constant=this.constant;
		br.betas=manipulate.copies.copies.Copy(this.betas.clone());
		br.support_cases=support_cases.clone();
		br.RegularizationType=this.RegularizationType;
		br.C=this.C;
		br.l1C=this.l1C;
		br.Type=this.Type;
		br.threads=this.threads;
		br.UseConstant=this.UseConstant;
		br.columndimension=this.columndimension;
		br.maxim_Iteration=this.maxim_Iteration;
		br.intpasses=this.intpasses;
		br.pinter=this.pinter;
		br.submodelcutsper=this.submodelcutsper;
		br.smooth=this.smooth;
		br.intcoeffthres=this.intcoeffthres; 
		br.intpertokeep=this.intpertokeep; 
		br.Objective=this.Objective;
		br.distance=this.distance;
		br.gammabfs=this.gammabfs;
		br.degree=this.degree;
		br.coeff=this.coeff;
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

	//Fisher–Yates shuffle
	/**
	 * 
	 * @param ar : array to shuffke
	 * @param rand : Random number generator object
	 * <p> reshufles the given int array randomly
	 */
	static void shuffleArray(int[] ar, Random rand)
	{ 
	  for (int i = ar.length - 1; i > 0; i--)
	  {
	    int index = rand.nextInt(i + 1);
	    // Simple swap
	    int a = ar[index];
	    ar[index] = ar[i];
	    ar[i] = a;
	  }
	}
	
	

	
	@Override	
	public void set_params(String params){
		
		String splitted_params []=params.split(" " + "+");
		
		for (int j=0; j<splitted_params.length; j++ ){
			String mini_split []=splitted_params[j].split(":");
			if (mini_split.length>=2){
				String metric=mini_split[0];
				String value=mini_split[1];
				
				if (metric.equals("RegularizationType")) {this.RegularizationType=value;}
				else if (metric.equals("C")) {this.C=Double.parseDouble(value);}
				else if (metric.equals("l1C")) {this.l1C=Double.parseDouble(value);}
				else if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("UseConstant")) {this.UseConstant=(value.toLowerCase().equals("true")?true:false)   ;}
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
