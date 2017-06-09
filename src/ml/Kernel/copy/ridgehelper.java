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
import java.util.Random;
import preprocess.scaling.scaler;
import preprocess.scaling.maxscaler;
import matrix.fsmatrix;
import matrix.smatrix;


/**
 * <p>This class will perform a Ridge regression variant as a helper to the kernel-based model to help find the influential rows (predictors in kernel form) .</p>
 */
public class ridgehelper implements Runnable {

	/**
	 * Regularization value, the more, the stronger the regularization
	 */
	public double C=1.0;
	
	/**
	 * Value that helps when dividing with past gradients (to avoid zero divisions)
	 */
	public double smooth=0.01;
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
	 * type of distance, can be RBF, POLY(default degree=2)  or SIGMOID
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
	 * Scaler to use in case of usescale=true
	 */
	private preprocess.scaling.scaler Scaler;
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
//	/**
//	 * where the coefficients are held
//	 */
//	private double betas[];
	/**
	 * This needs to be provided in construction time
	 */
	private int given_indices [];
	/**
	 * start of the loop in the given_indices array
	 */
	private int start_array=-1;
	/**
	 * end of the loop in the given_indices array
	 */
	private int end_array=-1;
	
	/**
	 * % thresold of cases to keep
	 */
	private double percentage_thresol=0.0;
	/**
	 * % coefficient thresold of cases to keep
	 */
	private double coeff_thresol=0.0;
	
	/**
	 * The arrayList the holds the best cases
	 */
	private ArrayList<Integer> indices_tokeep;
	
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
	 * 
	 * @param data : Data to be fit
	 * @param indies : indices to use while training
	 * @param st : start of the loop
	 * @param ed : end of the loop
	 * @param per_thre: thresold parameter for percentage of cases that need to remain
	 * @param coeff_thre : another threshold parameter for the coefficients' value
	 * @param scs : a scaler object 
	 * @param indkeep : arraylist to hold the indices worth keeping to proceed to the next elimination phases
	 */
	public ridgehelper(double data [][],int indies[], int st, int ed, double per_thre,double coeff_thre, scaler scs,ArrayList<Integer> indkeep ){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (indies==null || indies.length<1){
			throw new IllegalStateException(" The indices cannot be empty" );
		}
		if (st<0 ){
			throw new IllegalStateException(" Start indice cannot be negative" );
		}	
		if (ed<0 ){
			throw new IllegalStateException(" end indice cannot be negative" );
		}
		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		if (per_thre>=1 || per_thre<=0 ){
			throw new IllegalStateException(" percentage_thresol has to be (0,1)" );
		}
		if (coeff_thre>=1 || coeff_thre<=0 ){
			throw new IllegalStateException(" coeff_thresol has to be (0,1)" );
		}		

		this.given_indices=indies;
		this.start_array=st;
		this.end_array=ed;
		this.percentage_thresol=per_thre;
		this.coeff_thresol=coeff_thre;
		this.Scaler=scs;
		this.indices_tokeep=indkeep;
		
		dataset=data;	
		
	}

	/**
	 * 
	 * @param data : Data to be fit
	 * @param indies : indices to use while training
	 * @param st : start of the loop
	 * @param ed : end of the loop
	 * @param per_thre: thresold parameter for percentage of cases that need to remain
	 * @param coeff_thre : another threshold parameter for the coefficients' value
	 * @param scs : a scaler object 
	 * @param indkeep : arraylist to hold the indices worth keeping to proceed to the next elimination phases
	 */
	
	public ridgehelper(fsmatrix data,int indies[], int st, int ed, double per_thre,double coeff_thre, scaler scs,ArrayList<Integer> indkeep ){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (indies==null || indies.length<1){
			throw new IllegalStateException(" The indices cannot be empty" );
		}
		if (st<0 ){
			throw new IllegalStateException(" Start indice cannot be negative" );
		}	
		if (ed<0 ){
			throw new IllegalStateException(" end indice cannot be negative" );
		}
		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		if (per_thre>=1 || per_thre<=0 ){
			throw new IllegalStateException(" percentage_thresol has to be (0,1)" );
		}
		if (coeff_thre>=1 || coeff_thre<=0 ){
			throw new IllegalStateException(" coeff_thresol has to be (0,1)" );
		}		
	

		this.given_indices=indies;
		this.start_array=st;
		this.end_array=ed;
		this.percentage_thresol=per_thre;
		this.coeff_thresol=coeff_thre;
		this.Scaler=scs;
		this.indices_tokeep=indkeep;
		
		fsdataset=data;
	}
	/**
	 * 
	 * @param data : Data to be fit
	 * @param indies : indices to use while training
	 * @param st : start of the loop
	 * @param ed : end of the loop
	 * @param per_thre: thresold parameter for percentage of cases that need to remain
	 * @param coeff_thre : another threshold parameter for the coefficients' value
	 * @param scs : a scaler object 
	 * @param indkeep : arraylist to hold the indices worth keeping to proceed to the next elimination phases
	 */
	public ridgehelper(smatrix data,int indies[], int st, int ed, double per_thre,double coeff_thre, scaler scs,ArrayList<Integer> indkeep ){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (indies==null || indies.length<1){
			throw new IllegalStateException(" The indices cannot be empty" );
		}
		if (st<0 ){
			throw new IllegalStateException(" Start indice cannot be negative" );
		}	
		if (ed<0 ){
			throw new IllegalStateException(" end indice cannot be negative" );
		}
		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		if (per_thre>=1 || per_thre<=0 ){
			throw new IllegalStateException(" percentage_thresol has to be (0,1)" );
		}
		if (coeff_thre>=1 || coeff_thre<=0 ){
			throw new IllegalStateException(" coeff_thresol has to be (0,1)" );
		}		
		

		this.given_indices=indies;
		this.start_array=st;
		this.end_array=ed;
		this.percentage_thresol=per_thre;
		this.coeff_thresol=coeff_thre;
		this.Scaler=scs;
		this.indices_tokeep=indkeep;
		
		sdataset=data;
		}
	
	

//	/**
//	 * 
//	 * @return the betas
//	 */
//	public double [] Getbetas(){
//		if (betas==null || betas.length<=0){
//			throw new IllegalStateException(" estimator needs to be fitted first" );
//		}
//		return manipulate.copies.copies.Copy(betas);
//	}
	
	/**
	 * 
	 * @param data : data to be fit
	 * <p> fit the ridge model
	 */
	private void fit(double[][] data) {
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
	
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if ( !distance.equals("RBF")   && !distance.equals("POLY") && !distance.equals("SIGMOID")){
			throw new IllegalStateException(" distance has to be one of RBF,POLY or Liblinear SIGMOID" );	
		}		
		if (this.degree<=0){
			this.degree=1; // degree for PLY
		}
		if (this.coeff<0){
			this.coeff=0.0; // a small value to make it converge...at some point
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
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more than 1 different values!" );	
			}
			has=null;
		}

		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		int columndimension=this.end_array-this.start_array;
		double [] betas= new double[columndimension];
		if (this.gammabfs<=0){
			this.gammabfs=1.0/((double)columndimension); //gamma
		}				
		//initialise beta

			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.smooth<0.0){
				throw new IllegalStateException(" smooth value cannot be less  than 0");
			}
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double constant=0.0;
			double n []= new double[columndimension]; // sum of squared gradients
			double features []= new double[columndimension]; // sum of squared gradients
			
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.length; k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt( data.length);
		    	}
		    	double pred=constant;
		    	double yi=target[i];
	
		    	// compute score
		    	// run the coefficients' loop
		    	for (int j=this.start_array; j < this.end_array; j++){
		    		feature=0;
		    		if (distance.equals("RBF")) {
		    			for (int h=0; h <data[0].length; h++){
		    				double euc=Scaler.transform(data[i][h], h)-Scaler.transform(data[given_indices[j]][ h], h);
		    				feature+=euc*euc;
		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j-this.start_array]=feature;
		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=0; h <data[0].length; h++){
		    				feature+=Scaler.transform(data[i][h], h)*Scaler.transform(data[given_indices[j]][ h], h);
		    			}
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}
		    			features[j-this.start_array]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=0; h <data[0].length; h++){
		    				feature+=Scaler.transform(data[i][h], h)*Scaler.transform(data[given_indices[j]][ h], h);
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j-this.start_array]=feature;
		    		}
		    		
			    pred+=feature*betas[j-this.start_array];
		    			
		    	}
		    	
		    	double residual=(pred-yi);
		    	
		    	// we update constant gradient
		    	double gradient=residual;

		    	gradient+=C*constant;
		        double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);
		    	nc+=gradient*gradient;
		    	constant=constant-move; 
		    	 
		    	 
		    	for (int j=0; j < features.length;j++){
		    		 	    		 
			    		feature=features[j];
			    		if (feature==0.0){
			    			continue;	
			    		} 
			    		gradient=residual*feature;
		    			gradient+=C*betas[j];

			    		 move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
			    		 n[j]+=gradient*gradient;
			    		 betas[j]=betas[j]-move;
		    		 	
		    	 }
		    
		    }
             
	           //end of while
	            it++; 
			}			
			//we will do something cheeky. Will copy the beta array to the sum-of-passed-gradients array
			//in order to avoid reallocating memory objects
			for (int j=0; j < betas.length;j++){
				n[j]=Math.abs(betas[j]);
			}
			
			/* sort*/
			Arrays.sort(n);
			int location=(int) ((1.0-percentage_thresol)*n.length);
			double thress=n[location];
			
			/* find which cases to bring */
			
			for (int j=0; j < betas.length;j++){ 
				feature=Math.abs(betas[j]);
				if (feature>=thress && feature>=this.coeff_thresol){
					this.indices_tokeep.add(this.given_indices[this.start_array+j]);
				}
			}
			
			
			n=null;
			betas=null;
			System.gc();
			
			

			// end of SGD

	}
	/**
	 * 
	 * @param data : data to be fit
	 * <p> fit the ridge model
	 */
	private void fit(fsmatrix data) {
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
	
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if ( !distance.equals("RBF")   && !distance.equals("POLY") && !distance.equals("SIGMOID")){
			throw new IllegalStateException(" distance has to be one of RBF,POLY or Liblinear SIGMOID" );	
		}		
		if (this.degree<=0){
			this.degree=1; // degree for PLY
		}
		if (this.coeff<0){
			this.coeff=0.0; // a small value to make it converge...at some point
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
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more than 1 different values!" );	
			}
			has=null;
		}

		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		int columndimension=this.end_array-this.start_array;
		double [] betas= new double[columndimension];
		if (this.gammabfs<=0){
			this.gammabfs=1.0/((double)columndimension); //gamma
		}				
		//initialise beta

			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.smooth<0.0){
				throw new IllegalStateException(" smooth value cannot be less  than 0");
			}
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double constant=0.0;
			double n []= new double[columndimension]; // sum of squared gradients
			double features []= new double[columndimension]; // sum of squared gradients
			
			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt( data.GetRowDimension());
		    	}
		    	double pred=constant;
		    	double yi=target[i];
		    	
		    	// compute score
		    	// run the coefficients' loop
		    	for (int j=this.start_array; j < this.end_array; j++){
		    		feature=0;
		    		if (distance.equals("RBF")) {
		    			for (int h=0; h <data.GetColumnDimension(); h++){
		    				double euc=Scaler.transform(data.GetElement(i, h), h)-Scaler.transform(data.GetElement(given_indices[j], h), h);
		    				feature+=euc*euc;
		    			}
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j-this.start_array]=feature;
		    			
		    		} else if (distance.equals("POLY")) {
		    			for (int h=0; h <data.GetColumnDimension(); h++){
		    				feature+=Scaler.transform(data.GetElement(i, h), h)*Scaler.transform(data.GetElement(given_indices[j], h), h);
		    			}
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}
		    			features[j-this.start_array]=feature;
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int h=0; h <data.GetColumnDimension(); h++){
		    				feature+=Scaler.transform(data.GetElement(i, h), h)*Scaler.transform(data.GetElement(given_indices[j], h), h);
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j-this.start_array]=feature;
		    		}
		    		
			    pred+=feature*betas[j-this.start_array];
		    			
		    	}
		    	
		    	double residual=(pred-yi);
		    	
		    	// we update constant gradient
		    	double gradient=residual;

		    	gradient+=C*constant;
		        double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);
		    	nc+=gradient*gradient;
		    	constant=constant-move; 
		    	 
		    	 
		    	for (int j=0; j < features.length;j++){
		    		 	    		 
			    		feature=features[j];
			    		if (feature==0.0){
			    			continue;	
			    		} 
			    		gradient=residual*feature;
		    			gradient+=C*betas[j];

			    		 move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
			    		 n[j]+=gradient*gradient;
			    		 betas[j]=betas[j]-move;
		    		 	
		    	 }
		    
		    }
             
	           //end of while
	            it++; 
			}			
			//we will do something cheeky. Will copy the beta array to the sum-of-passed-gradients array
			//in order to avoid reallocating memory objects
			for (int j=0; j < betas.length;j++){
				n[j]=Math.abs(betas[j]);
			}
			
			/* sort*/
			Arrays.sort(n);
			int location=(int) ((1.0-percentage_thresol)*n.length);
			double thress=n[location];

			/* find which cases to bring */
			
			for (int j=0; j < betas.length;j++){ 
				feature=Math.abs(betas[j]);
				if (feature>=thress && feature>=this.coeff_thresol){
					this.indices_tokeep.add(this.given_indices[this.start_array+j]);
				}
			}
			
			//System.out.println(Arrays.toString(betas));
			n=null;
			betas=null;
			System.gc();
			
			

			// end of SGD

	}
	/**
	 * 
	 * @param data : data to be fit
	 * <p> fit the ridge model
	 */
	private void fit(smatrix data) {
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
	
		if (this.maxim_Iteration<=0){
			this.maxim_Iteration=10000; // a high value just in case id cannot converge
		}
		if (this.tolerance<=0){
			this.tolerance=0.0000000000000000001; // a small value to make it converge...at some point
		}
		if ( !distance.equals("RBF")   && !distance.equals("POLY") && !distance.equals("SIGMOID")){
			throw new IllegalStateException(" distance has to be one of RBF,POLY or Liblinear SIGMOID" );	
		}		
		if (this.degree<=0){
			this.degree=1; // degree for PLY
		}
		if (this.coeff<0){
			this.coeff=0.0; // a small value to make it converge...at some point
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
			if (has.size()<=1){
				throw new IllegalStateException(" target array needs to have more than 1 different values!" );	
			}
			has=null;
		}

		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
			
		}				
		if (( Scaler.IsFitted()==false)){
			Scaler.fit(data);
			
		}
		// set random number generator 
		random = new Random();
		random.setSeed(seed);
		
		//initialize column dimension
		int columndimension=this.end_array-this.start_array;
		double []betas= new double[columndimension];
		if (this.gammabfs<=0){
			this.gammabfs=1.0/((double)columndimension); //gamma
		}				
		//initialise beta

			if (this.learn_rate<=0.00000000000001){
				throw new IllegalStateException(" Learning rate cannot be less than 0.00000000000001");
			}
			if (this.smooth<0.0){
				throw new IllegalStateException(" smooth value cannot be less  than 0");
			}
			double iteration_tol=Double.POSITIVE_INFINITY;
			int it=0;
			double nc=0;
			double feature=0.0;
			// random number generator
			random= new Random();
			random.setSeed(seed);
			double constant=0.0;
			double n []= new double[columndimension]; // sum of squared gradients
			double features []= new double[columndimension]; // sum of squared gradients
			HashMap<Integer, Integer> has_index=null;

			// iterative algorithms start here
			while (it <this.maxim_Iteration && iteration_tol> this.tolerance) {

		    for (int k=0; k < data.GetRowDimension(); k++){
		    	int i=k;

		    	if (this.shuffle){
		    	 i=random.nextInt( data.GetRowDimension());
		    	}
		    	double pred=constant;
		    	double yi=target[i];
		    	
				has_index=new HashMap<Integer, Integer>();
				for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
					has_index.put(data.mainelementpile[v],v);
				}	
		    	// compute score
		    	// run the coefficients' loop
		    	for (int j=this.start_array; j < this.end_array; j++){
		    		feature=0;

		    		if (distance.equals("RBF")) {
		    			for (int b=data.indexpile[given_indices[j]]; b <data.indexpile[given_indices[j]+1]; b++){
		    				int h=data.mainelementpile[b];
		    				double balue=data.valuespile[b];
		    				Integer colinteger=has_index.get(h);
		    				if (colinteger!=null){
			    				double euc=Scaler.transform(data.valuespile[colinteger], h)-Scaler.transform(balue, h);
			    				feature+=euc*euc;
		    				} else {
		    					double euc=Scaler.transform(balue, h);
			    				feature+=euc*euc;
		    				}
		    			}
		    			
		    			feature=Math.exp(-this.gammabfs * feature); 
		    			features[j-this.start_array]=feature;
		    			
		    		} else if (distance.equals("POLY")) {
		    			
		    			for (int b=data.indexpile[given_indices[j]]; b <data.indexpile[given_indices[j]+1]; b++){
		    				int h=data.mainelementpile[b];
		    				double balue=data.valuespile[b];
		    				Integer colinteger=has_index.get(h);
		    				if (colinteger!=null){
			    				feature+=Scaler.transform(data.valuespile[colinteger], h)*Scaler.transform(balue, h);
		    				} 
		    			}
			    			
		    			feature=(this.gammabfs*feature)+this.coeff;
		    			for (int h=0; h <this.degree-1; h++){
		    				feature*=feature;
		    			}
		    			features[j-this.start_array]=feature;
		    			
		    		} else if (distance.equals("SIGMOID") ){
		    			for (int b=data.indexpile[given_indices[j]]; b <data.indexpile[given_indices[j]+1]; b++){
		    				int h=data.mainelementpile[b];
		    				double balue=data.valuespile[b];
		    				Integer colinteger=has_index.get(h);
		    				if (colinteger!=null){
			    				feature+=Scaler.transform(data.valuespile[colinteger], h)*Scaler.transform(balue, h);
		    				} 
		    			}
		    			feature=Math.tanh(this.gammabfs * feature + this.coeff); 
		    			features[j-this.start_array]=feature;
		    		}
		    		
			    pred+=feature*betas[j-this.start_array];
		    			
		    	}
		    	
		    	double residual=(pred-yi);
		    	
		    	// we update constant gradient
		    	double gradient=residual;

		    	gradient+=C*constant;
		        double move=(this.learn_rate*gradient)/Math.sqrt(nc+this.smooth);
		    	nc+=gradient*gradient;
		    	constant=constant-move; 
		    	 
		    	 
		    	for (int j=0; j < features.length;j++){
		    		 	    		 
			    		feature=features[j];
			    		if (feature==0.0){
			    			continue;	
			    		} 
			    		gradient=residual*feature;
		    			gradient+=C*betas[j];

			    		 move=(this.learn_rate*gradient)/Math.sqrt(n[j]+this.smooth);
			    		 n[j]+=gradient*gradient;
			    		 betas[j]=betas[j]-move;
		    		 	
		    	 }
		    
		    }
             
	           //end of while
	            it++; 
			}			
			//we will do something cheeky. Will copy the beta array to the sum-of-passed-gradients array
			//in order to avoid reallocating memory objects
			for (int j=0; j < betas.length;j++){
				n[j]=Math.abs(betas[j]);
			}
			
			/* sort*/
			Arrays.sort(n);
			int location=(int) ((1.0-percentage_thresol)*n.length);
			double thress=n[location];
			
			/* find which cases to bring */
			
			for (int j=0; j < betas.length;j++){ 
				feature=Math.abs(betas[j]);
				if (feature>=thress && feature>=this.coeff_thresol){
					this.indices_tokeep.add(this.given_indices[this.start_array+j]);
				}
			}
			
			
			n=null;
			betas=null;
			System.gc();
			
			

			// end of SGD

	}



//	public boolean isfitted() {
//		if (betas!=null || betas.length>0){
//			return true;
//		} else {
//		return false;
//		}
//	}


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
	 * 
	 * @return current scaler
	 */
	public scaler ReturnScaler() {
		return this.Scaler;
	}
	/**
	 * 
	 * @param sc set scaler
	 */
	public void setScaler(scaler sc) {
		this.Scaler=sc;
		
	}

}
