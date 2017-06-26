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

package ml.knn;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
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
 * K nearest neighbour regression ( brute Force):
 * Supports various metrics and data formats (sparse/non-sparse)
 * <ol>
 * <li>  cityblock  </li> 
 * <li>  euclidean  </li> 
 * <li>  cosine     </li> 
 * </ol>
 */
public class knnRegressor implements estimator,regressor {
	
	/**
	 * The distance to use. It has to be one of  cityblock,cosine,euclidean
	 */
	public String distance="euclidean";
	/**
	 *  Weights' computation type of either standard or weights
	 */
	public String Type="standard";
	/**
	 * This will hold the support vector type of features 
	 */
	private smatrix vectorset ;
	/**
	 * This will hold the target variable
	 */
	private fsmatrix targetset ;
	/**
	 * percentage of training data use when creating distances;
	 */
	public double sub_sample_percent=1.0;
	/**
	 * The percentage of subsections of each submodel
	 */	
	public double submodelcutsper=0.01;
	/**
	 * Number of k nearest neighbours
	 */
	public int neibours=1 ;
	/**
	 * threads to use
	 */
	public int threads=1;
	/**
	 *whether to use scale or not 
	 */
	public boolean usescale=true;
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
	public knnRegressor(){
	
	}	
	/**
	 * @param data : initial data to create the reserved space for neighbour search.
	 */
	public knnRegressor(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	
	/**
	 * @param data : initial data to create the reserved space for neighbour search.
	 */
	public knnRegressor(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * @param data : initial data to create the reserved space for neighbour search.
	 */
	public knnRegressor(smatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		sdataset=data;
	}
	
	/**
	 * @param data : data for training
	 */
	public void setdata(double data [][]){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		dataset=data;		
	}
	/**
	 * @param data : data for training
	 */
	public void setdata(fsmatrix data){
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		fsdataset=data;
	}
	/**
	 * @param data : data for training
	 */
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
	 * @return number of classes
	 */
	public int getnumber_of_classes(){
		return n_classes;
	}
	
	/**
	 * default Serial id
	 */
	private static final long serialVersionUID = -8611561535854392960L;
	@Override
	public double[][] predict2d(double[][] data) {
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		double globalpreds [][]= new double[data.length][this.n_classes];
		
		int batch_size=(int) (submodelcutsper*data.length);
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
				knnhelper svc = new knnhelper(data , globalpreds,  this.vectorset, this.targetset, 
						this.weights, loop_list[n], loop_list[n+1], this.neibours, this.Scaler , this.distance,  this.Type,  this.usescale, this.sub_sample_percent);
				svc.seed=n;
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
			
			
			System.gc();
			return globalpreds;
			}

	@Override
	public double[][] predict2d(fsmatrix data) {
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		double globalpreds [][]= new double[data.GetRowDimension()][this.n_classes];
		
		int batch_size=(int) (submodelcutsper*data.GetRowDimension());
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
				knnhelper svc = new knnhelper(data , globalpreds,  this.vectorset, this.targetset, 
						this.weights, loop_list[n], loop_list[n+1], this.neibours, this.Scaler , this.distance,  this.Type,  this.usescale, this.sub_sample_percent);
				svc.seed=n;
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
			
			
			System.gc();
			return globalpreds;
	}

	@Override
	public double[][] predict2d(smatrix data) {
		
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
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
		
		double globalpreds [][]= new double[data.GetRowDimension()][this.n_classes];
		
		int batch_size=(int) (submodelcutsper*data.GetRowDimension());
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
				knnhelper svc = new knnhelper(data , globalpreds,  this.vectorset, this.targetset, 
						this.weights, loop_list[n], loop_list[n+1], this.neibours, this.Scaler , this.distance,  this.Type,  this.usescale, this.sub_sample_percent);
				svc.seed=n;
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
			
			
			System.gc();
			return globalpreds;
	}


	@Override
	public double[] predict_Row2d(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		double predictions[]= new double [n_classes];



		ArrayList<double[]> hold_distances = new ArrayList<double[]>(this.neibours);					
		
		for (int s=0; s< this.vectorset.GetRowDimension();s++){

			HashMap<Integer, Integer> has_index=new HashMap<Integer, Integer>();
				for (int v=vectorset.indexpile[s]; v<vectorset.indexpile[s+1];v++ ){
					has_index.put(vectorset.mainelementpile[v],v);
			}
				double distance[]= new double [2 + predictions.length];
				//distance[0]=y_memory[i];
				
				if (this.distance.equals("cityblock")){
				for(int j=0; j <data.length; j++ ) {
					double x1=data[j];
					if (this.usescale){
						x1=Scaler.transform(x1, j);
					}
					double x2=0.0;
					//get feature from sparse array
					Integer colinteger=has_index.get(j);
    				if (colinteger!=null){
    					x2=vectorset.valuespile[colinteger];
    					if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
    					distance[0]+=Math.abs(x1-x2);
    				} 
				}
				}
				else if(this.distance.equals("euclidean")){
					for(int j=0; j <data.length; j++ ) {
						double x1=data[j];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					distance[0]+=(x1-x2)*(x1-x2);
	    				} 
					}
					distance[0]=Math.sqrt(distance[0]);
					}	
					
				else if(this.distance.equals("cosine")){
					double product=0.0;
					double abssumx1=0.0;
					double abssumx2=0.0;
					
					for(int j=0; j <data.length; j++ ) {
						double x1=data[j];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					product+=x1*x2;
	    					abssumx1+=x1*x1;
	    					abssumx2+=x2*x2;		
	    				} 
					}
					distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)));
					}		
					
				// adjust for weights
				
				//distance[0]*=1/this.weight[s];
				
				//check if the new case is small enough to be isnerted
				
				if(hold_distances.size()<this.neibours){
					distance[1]=this.weights[s];
					if (this.Type.equals("weights")){
						if (distance[0]!=0.0){
						distance[1]*=1/distance[0];
						} else {
							distance[1]=99999999999.99;
						}
					}
					for (int j=0; j <predictions.length; j++){
						distance[2+j]=this.targetset.GetElement(s, j);
					}
					boolean foundin=false;
					for (int k=0; k <hold_distances.size(); k++){
						if (hold_distances.get(k)[0]>distance[0]){
							hold_distances.add(k,distance);
							foundin=true;
							break;
						}
						
					}
					if (foundin==false){
					hold_distances.add(distance);
					}
				}else if (distance[0]<hold_distances.get(hold_distances.size()-1)[0]){
					for (int k=hold_distances.size()-2; k >=0;k-- ){
						if (hold_distances.get(k)[0]<distance[0] || k==0){
							distance[1]=this.weights[s];
							if (this.Type.equals("weights")){
								if (distance[0]!=0.0){
									distance[1]*=1/distance[0];
									} else {
										distance[1]=99999999999.99;
									}
							}
							for (int j=0; j <predictions.length; j++){
								distance[2+j]=this.targetset.GetElement(s, j);
							}
							if(k==0){
								k=-1;
							}
							hold_distances.add(k+1,distance);
							hold_distances.remove(hold_distances.size()-1);
							break;
						}
					}
				}
				// loop for training set ends here
		
		}
			// calculate probabilities based on the chosen method
				//System.out.println(" predictions length: " + predictions[i][0]);
				// simple counts
				for (int c=0; c<predictions.length;c++){
					 predictions[c]=0.0;
					double sumweight=0.0;
				for (int kn=0; kn< hold_distances.size(); kn++){
							//System.out.println(Arrays.toString(hold_distances.get(kn)));
							//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
							predictions[c]+=hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1];
							//System.out.println(predictions[i][c]);
							sumweight+=hold_distances.get(kn)[1];
					}
				//System.out.println(predictions[i][c] + " "+ sumweight);
				predictions[c]/=sumweight;
				//System.out.println(predictions[i][c]);
				}
		
		

		
			
		return predictions;
	}


	@Override
	public double[] predict_Row2d(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		double predictions[]= new double [n_classes];



		ArrayList<double[]> hold_distances = new ArrayList<double[]>(this.neibours);					
		
		for (int s=0; s< this.vectorset.GetRowDimension();s++){

			HashMap<Integer, Integer> has_index=new HashMap<Integer, Integer>();
				for (int v=vectorset.indexpile[s]; v<vectorset.indexpile[s+1];v++ ){
					has_index.put(vectorset.mainelementpile[v],v);
			}
				double distance[]= new double [2 + predictions.length];
				//distance[0]=y_memory[i];
				
				if (this.distance.equals("cityblock")){
				for(int j=0; j <data.GetColumnDimension(); j++ ) {
					double x1=data.GetElement(rows, j);
					if (this.usescale){
						x1=Scaler.transform(x1, j);
					}
					double x2=0.0;
					//get feature from sparse array
					Integer colinteger=has_index.get(j);
    				if (colinteger!=null){
    					x2=vectorset.valuespile[colinteger];
    					if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
    					distance[0]+=Math.abs(x1-x2);
    				} 
				}
				}
				else if(this.distance.equals("euclidean")){
					for(int j=0; j <data.GetColumnDimension(); j++ ) {
						double x1=data.GetElement(rows, j);
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					distance[0]+=(x1-x2)*(x1-x2);
	    				} 
					}
					distance[0]=Math.sqrt(distance[0]);
					}	
					
				else if(this.distance.equals("cosine")){
					double product=0.0;
					double abssumx1=0.0;
					double abssumx2=0.0;
					
					for(int j=0; j <data.GetColumnDimension(); j++ ) {
						double x1=data.GetElement(rows, j);
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					product+=x1*x2;
	    					abssumx1+=x1*x1;
	    					abssumx2+=x2*x2;		
	    				} 
					}
					distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)));
					}		
					
				// adjust for weights
				
				//distance[0]*=1/this.weight[s];
				
				//check if the new case is small enough to be isnerted
				
				if(hold_distances.size()<this.neibours){
					distance[1]=this.weights[s];
					if (this.Type.equals("weights")){
						if (distance[0]!=0.0){
						distance[1]*=1/distance[0];
						} else {
							distance[1]=99999999999.99;
						}
					}
					for (int j=0; j <predictions.length; j++){
						distance[2+j]=this.targetset.GetElement(s, j);
					}
					boolean foundin=false;
					for (int k=0; k <hold_distances.size(); k++){
						if (hold_distances.get(k)[0]>distance[0]){
							hold_distances.add(k,distance);
							foundin=true;
							break;
						}
						
					}
					if (foundin==false){
					hold_distances.add(distance);
					}
				}else if (distance[0]<hold_distances.get(hold_distances.size()-1)[0]){
					for (int k=hold_distances.size()-2; k >=0;k-- ){
						if (hold_distances.get(k)[0]<distance[0] || k==0){
							distance[1]=this.weights[s];
							if (this.Type.equals("weights")){
								if (distance[0]!=0.0){
									distance[1]*=1/distance[0];
									} else {
										distance[1]=99999999999.99;
									}
							}
							for (int j=0; j <predictions.length; j++){
								distance[2+j]=this.targetset.GetElement(s, j);
							}
							if(k==0){
								k=-1;
							}
							hold_distances.add(k+1,distance);
							hold_distances.remove(hold_distances.size()-1);
							break;
						}
					}
				}
				// loop for training set ends here
		
		}
			// calculate probabilities based on the chosen method
				//System.out.println(" predictions length: " + predictions[i][0]);
				// simple counts
				for (int c=0; c<predictions.length;c++){
					 predictions[c]=0.0;
					double sumweight=0.0;
				for (int kn=0; kn< hold_distances.size(); kn++){
							//System.out.println(Arrays.toString(hold_distances.get(kn)));
							//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
							predictions[c]+=hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1];
							//System.out.println(predictions[i][c]);
							sumweight+=hold_distances.get(kn)[1];
					}
				//System.out.println(predictions[i][c] + " "+ sumweight);
				predictions[c]/=sumweight;
				//System.out.println(predictions[i][c]);
				}
		
		

		
			
		return predictions;
	}

	@Override
	public double[] predict_Row2d(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		double predictions[]= new double [n_classes];



		HashMap<Integer, Integer> has_index_main=new HashMap<Integer, Integer>();
		for (int v=start; v<end;v++ ){
			has_index_main.put(data.mainelementpile[v],v);
		}
		ArrayList<double[]> hold_distances = new ArrayList<double[]>(this.neibours);					
		
		for (int s=0; s< this.vectorset.GetRowDimension();s++){
			
			HashMap<Integer, Double> has_index_not_in=new HashMap<Integer, Double>();
			HashMap<Integer, Integer> has_index=new HashMap<Integer, Integer>();
				for (int v=vectorset.indexpile[s]; v<vectorset.indexpile[s+1];v++ ){
				Integer column=vectorset.mainelementpile[v];
				Integer colinteger=has_index_main.get(column);
				if (colinteger!=null){
					has_index.put(column,v);
				} else {
					has_index_not_in.put(column,vectorset.valuespile[v]);
				}
				
			
			}
				double distance[]= new double [2 + predictions.length];
				//distance[0]=y_memory[i];
				
				if (this.distance.equals("cityblock")){
					for(int jj=start; jj <end; jj++ ) {
					int j=data.mainelementpile[jj];
					double x1=data.valuespile[jj];
					if (this.usescale){
						x1=Scaler.transform(x1, j);
					}
					double x2=0.0;
					//get feature from sparse array
					Integer colinteger=has_index.get(j);
    				if (colinteger!=null){
    					x2=vectorset.valuespile[colinteger];
    					if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
    					distance[0]+=Math.abs(x1-x2);
    				} 
				}
				for (Entry<Integer, Double> entry : has_index_not_in.entrySet()) {
					distance[0]+=Math.abs( entry.getValue());
				}
				
				}
				else if(this.distance.equals("euclidean")){
					for(int jj=start; jj <end; jj++ ) {
						int j=data.mainelementpile[jj];
						double x1=data.valuespile[jj];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					distance[0]+=(x1-x2)*(x1-x2);
	    				} 
					}
					for (Entry<Integer, Double> entry : has_index_not_in.entrySet()) {
						double val=entry.getValue();
						distance[0]+=val*val;
					}
					distance[0]=Math.sqrt(distance[0]);
					}	
					
				else if(this.distance.equals("cosine")){
					double product=0.0;
					double abssumx1=0.0;
					double abssumx2=0.0;
					
					for(int jj=start; jj <end; jj++ ) {
						int j=data.mainelementpile[jj];
						double x1=data.valuespile[jj];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					product+=x1*x2;
	    					abssumx1+=x1*x1;
	    					abssumx2+=x2*x2;		
	    				} 
					}
					for (Entry<Integer, Double> entry : has_index_not_in.entrySet()) {
						entry.getValue();
						distance[0]+=1;
					}
					distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)+0.0000001));
					}	
					
				// adjust for weights
				
				//distance[0]*=1/this.weight[s];
				
				//check if the new case is small enough to be isnerted
				
				if(hold_distances.size()<this.neibours){
					distance[1]=this.weights[s];
					if (this.Type.equals("weights")){
						if (distance[0]!=0.0){
						distance[1]*=1/distance[0];
						} else {
							distance[1]=99999999999.99;
						}
					}
					for (int j=0; j <predictions.length; j++){
						distance[2+j]=this.targetset.GetElement(s, j);
					}
					boolean foundin=false;
					for (int k=0; k <hold_distances.size(); k++){
						if (hold_distances.get(k)[0]>distance[0]){
							hold_distances.add(k,distance);
							foundin=true;
							break;
						}
						
					}
					if (foundin==false){
					hold_distances.add(distance);
					}
				}else if (distance[0]<hold_distances.get(hold_distances.size()-1)[0]){
					for (int k=hold_distances.size()-2; k >=0;k-- ){
						if (hold_distances.get(k)[0]<distance[0] || k==0){
							distance[1]=this.weights[s];
							if (this.Type.equals("weights")){
								if (distance[0]!=0.0){
									distance[1]*=1/distance[0];
									} else {
										distance[1]=99999999999.99;
									}
							}
							for (int j=0; j <predictions.length; j++){
								distance[2+j]=this.targetset.GetElement(s, j);
							}
							if(k==0){
								k=-1;
							}
							hold_distances.add(k+1,distance);
							hold_distances.remove(hold_distances.size()-1);
							break;
						}
					}
				}
				// loop for training set ends here
		}
			
			// calculate probabilities based on the chosen method
				//System.out.println(" predictions length: " + predictions[i][0]);
				// simple counts
				for (int c=0; c<predictions.length;c++){
					 predictions[c]=0.0;
					double sumweight=0.0;
				for (int kn=0; kn< hold_distances.size(); kn++){
							//System.out.println(Arrays.toString(hold_distances.get(kn)));
							//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
							predictions[c]+=hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1];
							//System.out.println(predictions[i][c]);
							sumweight+=hold_distances.get(kn)[1];
					}
				//System.out.println(predictions[i][c] + " "+ sumweight);
				predictions[c]/=sumweight;
				//System.out.println(predictions[i][c]);
				}
		
		

		
			
		return predictions;
			}

	@Override
	public double[] predict(fsmatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}	
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		

		
		double predictions[]= new double [data.GetRowDimension()];
		double globalpreds [][]= new double[data.GetRowDimension()][this.n_classes];
		
		int batch_size=(int) (submodelcutsper*data.GetRowDimension());
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
				knnhelper svc = new knnhelper(data , globalpreds,  this.vectorset, this.targetset, 
						this.weights, loop_list[n], loop_list[n+1], this.neibours, this.Scaler , this.distance,  this.Type,  this.usescale, this.sub_sample_percent);
				svc.seed=n;
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
			for (int i=0; i < predictions.length; i++){
				predictions[i]=globalpreds[i][0];
			}
			
			globalpreds=null;
			System.gc();
			return predictions;
			}

	@Override
	public double[] predict(smatrix data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}	
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
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
		double globalpreds [][]= new double[data.GetRowDimension()][this.n_classes];
		
		int batch_size=(int) (submodelcutsper*data.GetRowDimension());
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
				knnhelper svc = new knnhelper(data , globalpreds,  this.vectorset, this.targetset, 
						this.weights, loop_list[n], loop_list[n+1], this.neibours, this.Scaler , this.distance,  this.Type,  this.usescale, this.sub_sample_percent);
				svc.seed=n;
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
			for (int i=0; i < predictions.length; i++){
				predictions[i]=globalpreds[i][0];
			}
			
			globalpreds=null;
			System.gc();
			return predictions;
			}

	@Override
	public double[] predict(double[][] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}  
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}	
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data[0].length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data[0].length);	
		}
		
		double predictions[]= new double [data.length];
		double globalpreds [][]= new double[data.length][this.n_classes];
		
		int batch_size=(int) (submodelcutsper*data.length);
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
				knnhelper svc = new knnhelper(data , globalpreds,  this.vectorset, this.targetset, 
						this.weights, loop_list[n], loop_list[n+1], this.neibours, this.Scaler , this.distance,  this.Type,  this.usescale, this.sub_sample_percent);
				svc.seed=n;
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
			for (int i=0; i < predictions.length; i++){
				predictions[i]=globalpreds[i][0];
			}
			
			globalpreds=null;
			System.gc();
			return predictions;
			}
	

	@Override
	public double predict_Row(double[] data) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.length!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.length);	
		}
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		double predictions= 0.0;



		ArrayList<double[]> hold_distances = new ArrayList<double[]>(this.neibours);					
		
		for (int s=0; s< this.vectorset.GetRowDimension();s++){

			HashMap<Integer, Integer> has_index=new HashMap<Integer, Integer>();
				for (int v=vectorset.indexpile[s]; v<vectorset.indexpile[s+1];v++ ){
					has_index.put(vectorset.mainelementpile[v],v);
			}
				double distance[]= new double [3];
				//distance[0]=y_memory[i];
				
				if (this.distance.equals("cityblock")){
				for(int j=0; j <data.length; j++ ) {
					double x1=data[j];
					if (this.usescale){
						x1=Scaler.transform(x1, j);
					}
					double x2=0.0;
					//get feature from sparse array
					Integer colinteger=has_index.get(j);
    				if (colinteger!=null){
    					x2=vectorset.valuespile[colinteger];
    					if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
    					distance[0]+=Math.abs(x1-x2);
    				} 
				}
				}
				else if(this.distance.equals("euclidean")){
					for(int j=0; j <data.length; j++ ) {
						double x1=data[j];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					distance[0]+=(x1-x2)*(x1-x2);
	    				} 
					}
					distance[0]=Math.sqrt(distance[0]);
					}	
					
				else if(this.distance.equals("cosine")){
					double product=0.0;
					double abssumx1=0.0;
					double abssumx2=0.0;
					
					for(int j=0; j <data.length; j++ ) {
						double x1=data[j];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					product+=x1*x2;
	    					abssumx1+=x1*x1;
	    					abssumx2+=x2*x2;		
	    				} 
					}
					distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)));
					}		
					
				// adjust for weights
				
				//distance[0]*=1/this.weight[s];
				
				//check if the new case is small enough to be isnerted
				
				if(hold_distances.size()<this.neibours){
					distance[1]=this.weights[s];
					if (this.Type.equals("weights")){
						if (distance[0]!=0.0){
						distance[1]*=1/distance[0];
						} else {
							distance[1]=99999999999.99;
						}
					}
					for (int j=0; j <1; j++){
						distance[2+j]=this.targetset.GetElement(s, j);
					}
					boolean foundin=false;
					for (int k=0; k <hold_distances.size(); k++){
						if (hold_distances.get(k)[0]>distance[0]){
							hold_distances.add(k,distance);
							foundin=true;
							break;
						}
						
					}
					if (foundin==false){
					hold_distances.add(distance);
					}
				}else if (distance[0]<hold_distances.get(hold_distances.size()-1)[0]){
					for (int k=hold_distances.size()-2; k >=0;k-- ){
						if (hold_distances.get(k)[0]<distance[0] || k==0){
							distance[1]=this.weights[s];
							if (this.Type.equals("weights")){
								if (distance[0]!=0.0){
									distance[1]*=1/distance[0];
									} else {
										distance[1]=99999999999.99;
									}
							}
							for (int j=0; j <1; j++){
								distance[2+j]=this.targetset.GetElement(s, j);
							}
							if(k==0){
								k=-1;
							}
							hold_distances.add(k+1,distance);
							hold_distances.remove(hold_distances.size()-1);
							break;
						}
					}
				}
				// loop for training set ends here
		}
			
			// calculate probabilities based on the chosen method
				//System.out.println(" predictions length: " + predictions[i][0]);
				// simple counts
				for (int c=0; c<1;c++){
					 predictions=0.0;
					double sumweight=0.0;
				for (int kn=0; kn< hold_distances.size(); kn++){
							//System.out.println(Arrays.toString(hold_distances.get(kn)));
							//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
							predictions+=hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1];
							//System.out.println(predictions[i][c]);
							sumweight+=hold_distances.get(kn)[1];
					}
				//System.out.println(predictions[i][c] + " "+ sumweight);
				predictions/=sumweight;
				//System.out.println(predictions[i][c]);
				}
		
		

		
			
		return predictions;
			}
	
	@Override
	public double predict_Row(fsmatrix data, int rows) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		double predictions= 0.0;

		ArrayList<double[]> hold_distances = new ArrayList<double[]>(this.neibours);					
		
		for (int s=0; s< this.vectorset.GetRowDimension();s++){

			HashMap<Integer, Integer> has_index=new HashMap<Integer, Integer>();
				for (int v=vectorset.indexpile[s]; v<vectorset.indexpile[s+1];v++ ){
					has_index.put(vectorset.mainelementpile[v],v);
			}
				double distance[]= new double [3];
				//distance[0]=y_memory[i];
				
				if (this.distance.equals("cityblock")){
				for(int j=0; j <data.GetColumnDimension(); j++ ) {
					double x1=data.GetElement(rows, j);
					if (this.usescale){
						x1=Scaler.transform(x1, j);
					}
					double x2=0.0;
					//get feature from sparse array
					Integer colinteger=has_index.get(j);
    				if (colinteger!=null){
    					x2=vectorset.valuespile[colinteger];
    					if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
    					distance[0]+=Math.abs(x1-x2);
    				} 
				}
				}
				else if(this.distance.equals("euclidean")){
					for(int j=0; j <data.GetColumnDimension(); j++ ) {
						double x1=data.GetElement(rows, j);
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					distance[0]+=(x1-x2)*(x1-x2);
	    				} 
					}
					distance[0]=Math.sqrt(distance[0]);
					}	
					
				else if(this.distance.equals("cosine")){
					double product=0.0;
					double abssumx1=0.0;
					double abssumx2=0.0;
					
					for(int j=0; j <data.GetColumnDimension(); j++ ) {
						double x1=data.GetElement(rows, j);
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=this.vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					product+=x1*x2;
	    					abssumx1+=x1*x1;
	    					abssumx2+=x2*x2;		
	    				} 
					}
					distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)));
					}		
					
				// adjust for weights
				
				//distance[0]*=1/this.weight[s];
				
				//check if the new case is small enough to be isnerted
				
				if(hold_distances.size()<this.neibours){
					distance[1]=this.weights[s];
					if (this.Type.equals("weights")){
						if (distance[0]!=0.0){
						distance[1]*=1/distance[0];
						} else {
							distance[1]=99999999999.99;
						}
					}
					for (int j=0; j <1; j++){
						distance[2+j]=this.targetset.GetElement(s, j);
					}
					boolean foundin=false;
					for (int k=0; k <hold_distances.size(); k++){
						if (hold_distances.get(k)[0]>distance[0]){
							hold_distances.add(k,distance);
							foundin=true;
							break;
						}
						
					}
					if (foundin==false){
					hold_distances.add(distance);
					}
				}else if (distance[0]<hold_distances.get(hold_distances.size()-1)[0]){
					for (int k=hold_distances.size()-2; k >=0;k-- ){
						if (hold_distances.get(k)[0]<distance[0] || k==0){
							distance[1]=this.weights[s];
							if (this.Type.equals("weights")){
								if (distance[0]!=0.0){
									distance[1]*=1/distance[0];
									} else {
										distance[1]=99999999999.99;
									}
							}
							for (int j=0; j <1; j++){
								distance[2+j]=this.targetset.GetElement(s, j);
							}
							if(k==0){
								k=-1;
							}
							hold_distances.add(k+1,distance);
							hold_distances.remove(hold_distances.size()-1);
							break;
						}
					}
				}
				// loop for training set ends here
		}
			
			// calculate probabilities based on the chosen method
				//System.out.println(" predictions length: " + predictions[i][0]);
				// simple counts
				for (int c=0; c<1;c++){
					 predictions=0.0;
					double sumweight=0.0;
				for (int kn=0; kn< hold_distances.size(); kn++){
							//System.out.println(Arrays.toString(hold_distances.get(kn)));
							//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
							predictions+=hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1];
							//System.out.println(predictions[i][c]);
							sumweight+=hold_distances.get(kn)[1];
					}
				//System.out.println(predictions[i][c] + " "+ sumweight);
				predictions/=sumweight;
				//System.out.println(predictions[i][c]);
				}
		
		

		
			
		return predictions;
			}
	

	@Override
	public double predict_Row(smatrix data, int start, int end) {
		/*
		 *  check if the Create_Logic method is run properly
		 */
		if (n_classes<1 || this.vectorset==null || this.targetset==null || this.weights==null || this.weights.length!= this.targetset.GetRowDimension() || this.targetset.GetRowDimension()!= this.vectorset.GetRowDimension() ){
			 throw new IllegalStateException("The fit method needs to be run successfully in " +
										"order to create the logic before attempting scoring a new set");}   
		if (n_classes>1) {
			System.err.println(" There were more than 1 target variables in the training dataset, Only the 1st will be returned");	
		}			
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (data.GetColumnDimension()!=columndimension){
			throw new IllegalStateException(" Number of predictors is not the same as th4 trained one: " +  columndimension + " <> " + data.GetColumnDimension());	
		}
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>vectorset.GetRowDimension()){
			neibours=vectorset.GetRowDimension();
		}
		
		double predictions= 0.0;



		HashMap<Integer, Integer> has_index_main=new HashMap<Integer, Integer>();
		for (int v=start; v<end;v++ ){
			has_index_main.put(data.mainelementpile[v],v);
		}
		ArrayList<double[]> hold_distances = new ArrayList<double[]>(this.neibours);					
		
		for (int s=0; s< this.vectorset.GetRowDimension();s++){
			
			HashMap<Integer, Double> has_index_not_in=new HashMap<Integer, Double>();
			HashMap<Integer, Integer> has_index=new HashMap<Integer, Integer>();
				for (int v=vectorset.indexpile[s]; v<vectorset.indexpile[s+1];v++ ){
				Integer column=vectorset.mainelementpile[v];
				Integer colinteger=has_index_main.get(column);
				if (colinteger!=null){
					has_index.put(column,v);
				} else {
					has_index_not_in.put(column,vectorset.valuespile[v]);
				}
				
			
			}
				double distance[]= new double [3];
				//distance[0]=y_memory[i];
				
				if (this.distance.equals("cityblock")){
					for(int jj=start; jj <end; jj++ ) {
					int j=data.mainelementpile[jj];
					double x1=data.valuespile[jj];
					if (this.usescale){
						x1=Scaler.transform(x1, j);
					}
					double x2=0.0;
					//get feature from sparse array
					Integer colinteger=has_index.get(j);
    				if (colinteger!=null){
    					x2=vectorset.valuespile[colinteger];
    					if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
    					distance[0]+=Math.abs(x1-x2);
    				} 
				}
				for (Entry<Integer, Double> entry : has_index_not_in.entrySet()) {
					distance[0]+=Math.abs( entry.getValue());
				}
				
				}
				else if(this.distance.equals("euclidean")){
					for(int jj=start; jj <end; jj++ ) {
						int j=data.mainelementpile[jj];
						double x1=data.valuespile[jj];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					distance[0]+=(x1-x2)*(x1-x2);
	    				} 
					}
					for (Entry<Integer, Double> entry : has_index_not_in.entrySet()) {
						double val=entry.getValue();
						distance[0]+=val*val;
					}
					distance[0]=Math.sqrt(distance[0]);
					}	
					
				else if(this.distance.equals("cosine")){
					double product=0.0;
					double abssumx1=0.0;
					double abssumx2=0.0;
					
					for(int jj=start; jj <end; jj++ ) {
						int j=data.mainelementpile[jj];
						double x1=data.valuespile[jj];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						Integer colinteger=has_index.get(j);
	    				if (colinteger!=null){
	    					x2=vectorset.valuespile[colinteger];
	    					if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
	    					product+=x1*x2;
	    					abssumx1+=x1*x1;
	    					abssumx2+=x2*x2;		
	    				} 
					}
					for (Entry<Integer, Double> entry : has_index_not_in.entrySet()) {
						entry.getValue();
						distance[0]+=1;
					}
					distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)+0.0000001));
					}	
					
				// adjust for weights
				
				//distance[0]*=1/this.weight[s];
				
				//check if the new case is small enough to be isnerted
				
				if(hold_distances.size()<this.neibours){
					distance[1]=this.weights[s];
					if (this.Type.equals("weights")){
						if (distance[0]!=0.0){
						distance[1]*=1/distance[0];
						} else {
							distance[1]=99999999999.99;
						}
					}
					for (int j=0; j <1; j++){
						distance[2+j]=this.targetset.GetElement(s, j);
					}
					boolean foundin=false;
					for (int k=0; k <hold_distances.size(); k++){
						if (hold_distances.get(k)[0]>distance[0]){
							hold_distances.add(k,distance);
							foundin=true;
							break;
						}
						
					}
					if (foundin==false){
					hold_distances.add(distance);
					}
				}else if (distance[0]<hold_distances.get(hold_distances.size()-1)[0]){
					for (int k=hold_distances.size()-2; k >=0;k-- ){
						if (hold_distances.get(k)[0]<distance[0] || k==0){
							distance[1]=this.weights[s];
							if (this.Type.equals("weights")){
								if (distance[0]!=0.0){
									distance[1]*=1/distance[0];
									} else {
										distance[1]=99999999999.99;
									}
							}
							for (int j=0; j <1; j++){
								distance[2+j]=this.targetset.GetElement(s, j);
							}
							if(k==0){
								k=-1;
							}
							hold_distances.add(k+1,distance);
							hold_distances.remove(hold_distances.size()-1);
							break;
						}
					}
				}
				// loop for training set ends here
		
		}
			// calculate probabilities based on the chosen method
				//System.out.println(" predictions length: " + predictions[i][0]);
				// simple counts
				for (int c=0; c<1;c++){
					 predictions=0.0;
					double sumweight=0.0;
				for (int kn=0; kn< hold_distances.size(); kn++){
							//System.out.println(Arrays.toString(hold_distances.get(kn)));
							//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
							predictions+=hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1];
							//System.out.println(predictions[i][c]);
							sumweight+=hold_distances.get(kn)[1];
					}
				//System.out.println(predictions[i][c] + " "+ sumweight);
				predictions/=sumweight;
				//System.out.println(predictions[i][c]);
				}
		
		
		return predictions;
			}


	@Override
	public void fit(double[][] data) {
		// make sensible checks
		if (data==null || data.length<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}		
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>data.length){
			neibours=data.length;
		}
		if (sub_sample_percent<=0.0000001  ){
			throw new IllegalStateException(" sub_sample_percent has to be with in [0.0000001,)" );
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
		
		n_classes=0;
		if (target!=null){
			n_classes=1;
			targetset=new fsmatrix(target,target.length,1);
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
			targetset=new fsmatrix(target2d);
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
			targetset=(fsmatrix) fstarget.Copy();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
			targetset=starget.ConvertToFixedSizeMatrix();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}

		//initialize column dimension
		columndimension=data[0].length;
		//initialise beta and constant
		vectorset = new smatrix (data) ;
		if (this.vectorset.indexer==null){
			this.vectorset.buildmap();
		}



	System.gc();
	
}


	@Override
	public void fit(fsmatrix data) {
		// make sensible checks
		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}	
		// make sensible checks on the target data
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (sub_sample_percent<=0.0000001  ){
			throw new IllegalStateException(" sub_sample_percent has to be with in [0.0000001,)" );
		}
		if (neibours>data.GetRowDimension()){
			neibours=data.GetRowDimension();
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
		
		n_classes=0;
		if (target!=null){
			n_classes=1;
			targetset=new fsmatrix(target,target.length,1);
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
			targetset=new fsmatrix(target2d);
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
			targetset=(fsmatrix) fstarget.Copy();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
			targetset=starget.ConvertToFixedSizeMatrix();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}

		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		vectorset = new smatrix (data) ;
		if (this.vectorset.indexer==null){
			this.vectorset.buildmap();
		}

		


	System.gc();
	
	
}

	@Override
	public void fit(smatrix data) {
		// make sensible checks

		if (data==null || data.GetRowDimension()<=3){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors(); // a high value just in case id cannot converge
		}	
		if (this.submodelcutsper> 1 ||this.submodelcutsper<=0 ){
			throw new IllegalStateException("Submodel percentage of data to be used to find the support vectors has to be (0,1]" );
		}				
		if ( !distance.equals("cityblock")  &&  !distance.equals("euclidean") && !distance.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if (this.threads<=0){
			this.threads=Runtime.getRuntime().availableProcessors();
		}	
		if ( !this.Type.equals("standard") &&  !this.Type.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}	
		// make sensible checks on the target data
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>data.GetRowDimension()){
			neibours=data.GetRowDimension();
		}
		if (sub_sample_percent<=0.0000001  ){
			throw new IllegalStateException(" sub_sample_percent has to be with in [0.0000001,)" );
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
		
		n_classes=0;
		if (target!=null){
			n_classes=1;
			targetset=new fsmatrix(target,target.length,1);
		} else if  (target2d!=null){
			n_classes=target2d[0].length;
			targetset=new fsmatrix(target2d);
		}else if  (fstarget!=null){
			n_classes=fstarget.GetColumnDimension();
			targetset=(fsmatrix) fstarget.Copy();
		}else if  (starget!=null){
			n_classes=starget.GetColumnDimension();
			targetset=starget.ConvertToFixedSizeMatrix();
		} else {
			throw new IllegalStateException(" A target array needs to be provided" );
		}

		//initialize column dimension
		columndimension=data.GetColumnDimension();
		//initialise beta and constant
		vectorset =data ;
		if (this.vectorset.indexer==null){
			this.vectorset.buildmap();
		}		
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
		return "knnRegressor";
	}

	@Override
	public void PrintInformation() {
		
		System.out.println("Regressor:  knn Regressor");
		System.out.println("Classes: " + n_classes);
		System.out.println("Supports Weights:  True");
		System.out.println("Column dimension: " + columndimension);				
		System.out.println("Neighbours: " + this.neibours);		
		System.out.println("Usescale: " + this.usescale);	
		System.out.println("percentage(%) of best cases/observations to include in each submodel  "+ this.submodelcutsper);
		System.out.println("Distance    "+ this.distance);		
		
		System.out.println("Weighting method: "+ this.Type);	

		System.out.println("Seed: "+ seed);		
		System.out.println("Verbality: "+ verbose);		
		if (this.vectorset==null){
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
		if (this.vectorset!=null || this.vectorset.GetRowDimension()>0){
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
		targetset=null;
		vectorset=null;
		n_classes=0;
		submodelcutsper=0.01;
		distance="cityblock";
		Type="standard";
		threads=1;
		this.usescale=true;
		this.neibours=1;
		columndimension=0;
		Scaler=null;
		copy=true;
		seed=1;
		random=null;
		target=null;
		target2d=null;
		fstarget=null;
		starget=null;
		weights=null;
		verbose=true;
		
	}


	@Override
	public estimator copy() {
		knnRegressor br = new knnRegressor();
		//hard copy of the latent features
		br.vectorset =(smatrix) (vectorset.Copy()) ;
		br.targetset=(fsmatrix) this.targetset.Copy();	
		
		
		br.n_classes=this.n_classes;
		br.columndimension=this.columndimension;
		br.submodelcutsper=this.submodelcutsper;
		br.distance=this.distance;
		br.Type=this.Type;
		br.threads=this.threads;
		br.usescale=this.usescale;
		br.neibours=this.neibours;
		br.Scaler=this.Scaler;
		br.copy=this.copy;
		br.seed=this.seed;
		br.random=this.random;
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
				
				if (metric.equals("Type")) {this.Type=value;}
				else if (metric.equals("threads")) {this.threads=Integer.parseInt(value);}
				else if (metric.equals("neibours")) {this.neibours=Integer.parseInt(value);}
				else if (metric.equals("submodelcutsper")) {this.submodelcutsper=Double.parseDouble(value);}			
				else if (metric.equals("distance")) {this.distance=value;}
				else if (metric.equals("copy")) {this.copy=(value.toLowerCase().equals("true")?true:false);}
				else if (metric.equals("seed")) {this.seed=Integer.parseInt(value);}
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
