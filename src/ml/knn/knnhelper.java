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
import java.util.Random;
import utilis.XorShift128PlusRandom;
import preprocess.scaling.scaler;
import preprocess.scaling.maxscaler;
import utilis.map.intint.IntIntMapminus4a;
import matrix.fsmatrix;
import matrix.smatrix;


/**
 * <p> Helps parallelism when searching for neighbours .</p>
 */
public class knnhelper implements Runnable {

	/**
	 * Whether to use scale or not
	 */
	public boolean usescale=true;
	/****
	 * whether this is used to support regression or classification
	 * in classification, the results are scaled to add up to 1.0
	 */
	public boolean isClassification=false;
	/**
	 * The distance to use. It has to be one of  cityblock,cosine,euclidean
	 */
	public String distance="euclidean";
	/**
	 *  Weights' computation type of either standard or weights
	 */
	public String Type="standard";
	/**
	 * Scaler to use in case of usescale=true
	 */
	private preprocess.scaling.scaler Scaler;
    /**
     * seed to use
     */
	public int seed=1;
	
	/**
	 * Number of neighbours to look at
	 */
	public int neighbours=1;
	/**
	 * Random number generator to use
	 */
	private Random random;
	/**
	 * double target use , an array of 0 and 1
	 */
	public fsmatrix target;
	/**
	 * The core dataset for scoring
	 */
	private smatrix helddataset;	
	
	/**
	 * Where the predictions will be stored
	 */
	public double predictions [][];
	
	/**
	 * percentage of training data use when creating distances;
	 */
	public double sub_sample_percent=1.0;
	/**
	 * Where the weight is stored 
	 */
	public double weight[];
	/**
	 * start of the loop in the given_indices array
	 */
	private int start_array=-1;
	/**
	 * end of the loop in the given_indices array
	 */
	private int end_array=-1;
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
	 * @param data : data to be scored
	 * @param predictions : place holder for the predictions
	 * @param maindata : base data to calculate neighbours with.
	 * @param maintarget : target variable for the base data
	 * @param weight : weight on the data
	 * @param st : start of the (sub)loop
	 * @param ed : end of the (sub)loop
	 * @param neibours : number of neighbours
	 * @param scs : scaler object
	 * @param dist :distance matrix, could be 'cityblock',  'euclidean' or 'cosine'
	 * @param wtype : prediction method Could be 'standard' or 'weights'
	 * @param usescale: true to use scaling
	 * @param sub_sample_percent: percentage of samples to examine.
	 * 
	 */
	public knnhelper(double data [][], double predictions [][],  smatrix maindata, fsmatrix maintarget, 
			double weight [], int st, int ed, int neibours, scaler scs , String dist, String wtype, boolean usescale, double sub_sample_percent){
		
		if (data==null || data.length<=0){
			throw new IllegalStateException(" There is nothing to score" );
		}
		if (predictions==null || predictions.length<=0 || data.length!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	
		if (maindata==null || maindata.GetRowDimension()<=0){
			throw new IllegalStateException(" There is no base array" );
		}	
		if (maintarget==null || maintarget.GetRowDimension()!= maindata.GetRowDimension()){
			throw new IllegalStateException(" Labels need to have the same size with main data " );
		}			
		if (weight==null || weight.length!=maindata.GetRowDimension()){
			throw new IllegalStateException(" Weight needs to have the same length with the data" );
		}

		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		if ( !dist.equals("cityblock")  &&  !dist.equals("euclidean") && !dist.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if ( !wtype.equals("standard") &&  !wtype.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>maindata.GetRowDimension()){
			neibours=maindata.GetRowDimension();
		}
		if (sub_sample_percent<=0.0000001  ){
			throw new IllegalStateException(" sub_sample_percent has to be with in [0.0000001,)" );
		}		
		this.sub_sample_percent=sub_sample_percent;
		this.start_array=st;
		this.end_array=ed;
		this.Scaler=scs;
		dataset=data;	
		this.weight=weight;
		this.predictions=predictions;
		this.helddataset=maindata;
		this.target=maintarget;
		this.Type=wtype;
		this.distance=dist;
		this.usescale=usescale;
		this.neighbours=neibours;
	
	}

	/**
	 * 
	 * @param data : data to be scored
	 * @param predictions : place holder for the predictions
	 * @param maindata : base data to calculate neighbours with.
	 * @param maintarget : target variable for the base data
	 * @param weight : weight on the data
	 * @param st : start of the (sub)loop
	 * @param ed : end of the (sub)loop
	 * @param neibours : number of neighbours
	 * @param scs : scaler object
	 * @param dist :distance matrix, could be 'cityblock',  'euclidean' or 'cosine'
	 * @param wtype : prediction method Could be 'standard' or 'weights'
	 * @param usescale: true to use scaling
	 * @param sub_sample_percent: percentage of samples to examine.
	 */
	public knnhelper(fsmatrix data, double predictions [][],  smatrix maindata, fsmatrix maintarget, 
			double weight [], int st, int ed, int neibours, scaler scs , String dist, String wtype, boolean usescale, double sub_sample_percent){
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (predictions==null || predictions.length<=0 || data.GetRowDimension()!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	
		if (maindata==null || maindata.GetRowDimension()<=0){
			throw new IllegalStateException(" There is no base array" );
		}	
		if (maintarget==null || maintarget.GetRowDimension()!= maindata.GetRowDimension()){
			throw new IllegalStateException(" Labels need to have the same size with main data " );
		}			
		if (weight==null || weight.length!=maindata.GetRowDimension()){
			throw new IllegalStateException(" Weight needs to have the same length with the data" );
		}	

		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		if ( !dist.equals("cityblock")  &&  !dist.equals("euclidean") && !dist.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if ( !wtype.equals("standard") &&  !wtype.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}			
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}
		if (neibours>maindata.GetRowDimension()){
			neibours=maindata.GetRowDimension();
		}
		if (sub_sample_percent<=0.0000001  ){
			throw new IllegalStateException(" sub_sample_percent has to be with in [0.0000001,)" );
		}		
		this.sub_sample_percent=sub_sample_percent;		
		this.start_array=st;
		this.end_array=ed;
		this.Scaler=scs;	
		this.weight=weight;
		this.predictions=predictions;
		this.helddataset=maindata;
		this.target=maintarget;
		this.Type=wtype;
		this.distance=dist;
		this.usescale=usescale;
		this.neighbours=neibours;
		fsdataset=data;
	}
	/**
	 * 
	 * @param data : data to be scored
	 * @param predictions : place holder for the predictions
	 * @param maindata : base data to calculate neighbours with.
	 * @param maintarget : target variable for the base data
	 * @param weight : weight on the data
	 * @param st : start of the (sub)loop
	 * @param ed : end of the (sub)loop
	 * @param neibours : number of neighbours
	 * @param scs : scaler object
	 * @param dist :distance matrix, could be 'cityblock',  'euclidean' or 'cosine'
	 * @param wtype : prediction method Could be 'standard' or 'weights'
	 * @param usescale: true to use scaling
	 * @param sub_sample_percent: percentage of samples to examine.
	 * 
	 */
	public knnhelper(smatrix data, double predictions [][],  smatrix maindata, fsmatrix maintarget, 
			double weight [], int st, int ed, int neibours, scaler scs , String dist, String wtype, boolean usescale, double sub_sample_percent){
		if (data==null || data.GetRowDimension()<=0){
			throw new IllegalStateException(" There is nothing to train on" );
		}
		if (predictions==null || predictions.length<=0 || data.GetRowDimension()!=predictions.length ){
			throw new IllegalStateException(" There is no place to set the scores or there is a size miss-match" );
		}	
		if (maindata==null || maindata.GetRowDimension()<=0){
			throw new IllegalStateException(" There is no base array" );
		}	
		if (maintarget==null || maintarget.GetRowDimension()!= maindata.GetRowDimension()){
			throw new IllegalStateException(" Labels need to have the same size with main data " );
		}			
		if (weight==null || weight.length!=maindata.GetRowDimension()){
			throw new IllegalStateException(" Weight needs to have the same length with the data" );
		}	

		if (ed-st<=0 ){
			throw new IllegalStateException(" end indice needs to be bigger than the start indice" );
		}	
		if ( !dist.equals("cityblock")  &&  !dist.equals("euclidean") && !dist.equals("cosine") ){
			throw new IllegalStateException("  distance has to be in {cityblock, euclidean, cosine}" );	
		}			
		if ( !wtype.equals("standard") &&  !wtype.equals("weights"))  {
			throw new IllegalStateException("Type has to be in (standard,weights)" );	
		}			
		if (neibours<1){
			throw new IllegalStateException("Neighbours cannot be less than 1" );	
		}		
		if (neibours>maindata.GetRowDimension()){
			neibours=maindata.GetRowDimension();
		}
		if (sub_sample_percent<=0.0000001  ){
			throw new IllegalStateException(" sub_sample_percent has to be with in [0.0000001,)" );
		}		
		this.sub_sample_percent=sub_sample_percent;	
		this.start_array=st;
		this.end_array=ed;
		this.Scaler=scs;	
		this.weight=weight;
		this.predictions=predictions;
		this.helddataset=maindata;
		this.target=maintarget;
		this.Type=wtype;
		this.distance=dist;
		this.usescale=usescale;
		this.neighbours=neibours;
		
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
	 * @param distances_holder : the array of distances
	 * @return the location of the highest distance
	 * 
	 */
private int find_largest_distance(double distances_holder[][]){
	int max_loc=0;
	double max=distances_holder[max_loc][0];
	
	for (int i=1; i < distances_holder.length; i++){
		if (distances_holder[i][0]>=max){
			max=distances_holder[i][0];
			max_loc=i;
		}
	}
	return max_loc;
	
}
	/**
	 * 
	 * @param data : data to use for predictions
	 * <p> finds nearest neighbours based on the stored data and the given data
	 */
	private void fit(double[][] data) {
		// make sensible checks
		if (data==null){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
		}				
		if ( this.usescale && Scaler.IsFitted()==false){
			Scaler.fit(helddataset);
		}
		

		random = new  XorShift128PlusRandom(seed);
		int thresoldcut=utilis.util.get_random_integer(this.sub_sample_percent);
		
		for (int i=this.start_array; i <this.end_array; i++){
			// set random number generator 
			
			
			double[][] hold_distances  = new double[this.neighbours][2 + predictions[0].length];
			int check_neighbours=0;
			int largest_distance_case=-1;
			double largest_distance=Double.MIN_VALUE;
			
			for (int s=0; s< this.helddataset.GetRowDimension();s++){
				if (random.nextInt()>thresoldcut) {
					continue;
				}
				
				/*
				HashMap<Integer, Integer> has_index=new HashMap<Integer, Integer>();
					for (int v=helddataset.indexpile[s]; v<helddataset.indexpile[s+1];v++ ){
						has_index.put(helddataset.mainelementpile[v],v);
				}
				*/
					double distance[]= new double [2 + predictions[0].length];
					//distance[0]=y_memory[i];
					
					if (this.distance.equals("cityblock")){
					for(int j=0; j <data[0].length; j++ ) {
						double x1=data[i][j];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						
						x2=helddataset.GetElement(s, j);
						if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
						distance[0]+=Math.abs(x1-x2);

					}
					}
					else if(this.distance.equals("euclidean")){
						for(int j=0; j <data[0].length; j++ ) {
							double x1=data[i][j];
							if (this.usescale){
								x1=Scaler.transform(x1, j);
							}
							double x2=0.0;
							//get feature from sparse array
							x2=helddataset.GetElement(s, j);
							if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
							distance[0]+=(x1-x2)*(x1-x2);
						
						}	
					}
					else if(this.distance.equals("cosine")){
						double product=0.0;
						double abssumx1=0.0;
						double abssumx2=0.0;
						for(int j=0; j <data[0].length; j++ ) {
							double x1=data[i][j];
							if (this.usescale){
								x1=Scaler.transform(x1, j);
							}
							double x2=0.0;
							//get feature from sparse array

							x2=helddataset.GetElement(s, j);
							if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
		    					product+=x1*x2;
		    					abssumx1+=x1*x1;
		    					abssumx2+=x2*x2;		
		    				 
						}
						distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)));
						}		
						
					// adjust for weights
					
					//distance[0]*=1/this.weight[s];
					
					//check if the new case is small enough to be isnerted
					
					
					if(check_neighbours<neighbours){
						distance[1]=this.weight[s];
						if (this.Type.equals("weights")){
							if (distance[0]!=0.0){
							distance[1]*=1/distance[0];
							} else {
								distance[1]=99999999999.99;
							}
						}
						for (int j=0; j <predictions[0].length; j++){
							distance[2+j]=this.target.GetElement(s, j);
						}
						if (distance[0]>=largest_distance){
							largest_distance=distance[0];
							largest_distance_case=check_neighbours;
						}
						hold_distances[check_neighbours]=distance;
						check_neighbours++;
						}
					else if (distance[0]<hold_distances[largest_distance_case][0]){

								distance[1]=this.weight[s];
								if (this.Type.equals("weights")){
									if (distance[0]!=0.0){
										distance[1]*=1/distance[0];
										} else {
											distance[1]=99999999999.99;
										}
								}
								for (int j=0; j <predictions[0].length; j++){
									distance[2+j]=this.target.GetElement(s, j);
								}
								hold_distances[largest_distance_case]=distance;

								//find new highest distance
								largest_distance_case=find_largest_distance(hold_distances);
								largest_distance=hold_distances[largest_distance_case][0];
						
					}
					// loop for training set ends here
			
			}	
				// calculate probabilities based on the chosen method
					//System.out.println(" predictions length: " + predictions[i][0]);
					// simple counts
					double sumforclassification=0.0;
					for (int c=0; c<this.predictions[0].length;c++){
						 predictions[i][c]=0.0;
						double sumweight=0.0;
						
					for (int kn=0; kn< hold_distances.length; kn++){
								//System.out.println(Arrays.toString(hold_distances.get(kn)));
								//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
								predictions[i][c]+=hold_distances[kn][2+c]*hold_distances[kn][1];
								//System.out.println(predictions[i][c]);
								sumweight+=hold_distances[kn][1];
						}
					//System.out.println(predictions[i][c] + " "+ sumweight);
					predictions[i][c]/=sumweight;
					sumforclassification+=predictions[i][c];
					//System.out.println(predictions[i][c]);
					}
					if (isClassification){
						for (int c=0; c<this.predictions[0].length;c++){
							predictions[i][c]/=sumforclassification;
						}
						
					}
			
			
			
			
			
			// end of loop
		}
		
			

			// end of SGD

	}
	/**
	 * 
	 * @param data : data to use for predictions
	 * <p> finds nearest neighbours based on the stored data and the given data
	 */
	private void fit(fsmatrix data) {
		
		// make sensible checks
		if (data==null){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
		}				
		if ( this.usescale && Scaler.IsFitted()==false){
			Scaler.fit(helddataset);
		}
		
		// set random number generator 
		random = new  XorShift128PlusRandom(seed);
		int thresoldcut=utilis.util.get_random_integer(this.sub_sample_percent);
		
		for (int i=this.start_array; i <this.end_array; i++){
			

			
			double[][] hold_distances  = new double[this.neighbours][2 + predictions[0].length];
			int check_neighbours=0;
			int largest_distance_case=-1;
			double largest_distance=Double.MIN_VALUE;			
			
			for (int s=0; s< this.helddataset.GetRowDimension();s++){

					if (random.nextInt()>thresoldcut) {
						continue;
					}
					double distance[]= new double [2 + predictions[0].length];
					//distance[0]=y_memory[i];
					
					if (this.distance.equals("cityblock")){
					for(int j=0; j <data.GetColumnDimension(); j++ ) {
						double x1=data.GetElement(i, j);
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						
						x2=helddataset.GetElement(s, j);
						if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
						distance[0]+=Math.abs(x1-x2);

					}
					}
					else if(this.distance.equals("euclidean")){
						for(int j=0; j <data.GetColumnDimension(); j++ ) {
							double x1=data.GetElement(i, j);
							if (this.usescale){
								x1=Scaler.transform(x1, j);
							}
							double x2=0.0;
							//get feature from sparse array
							x2=helddataset.GetElement(s, j);
							if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
							distance[0]+=(x1-x2)*(x1-x2);
						}	
					}
						
					else if(this.distance.equals("cosine")){
						double product=0.0;
						double abssumx1=0.0;
						double abssumx2=0.0;
						
						for(int j=0; j <data.GetColumnDimension(); j++ ) {
							double x1=data.GetElement(i, j);
							if (this.usescale){
								x1=Scaler.transform(x1, j);
							}
							double x2=0.0;
							//get feature from sparse array

							x2=helddataset.GetElement(s, j);
							if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
		    					product+=x1*x2;
		    					abssumx1+=x1*x1;
		    					abssumx2+=x2*x2;		
		    				 
						}
						distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)));
						}	
						
					// adjust for weights
					
					//distance[0]*=1/this.weight[s];
					
					//check if the new case is small enough to be isnerted
					
					if(check_neighbours<neighbours){
						distance[1]=this.weight[s];
						if (this.Type.equals("weights")){
							if (distance[0]!=0.0){
							distance[1]*=1/distance[0];
							} else {
								distance[1]=99999999999.99;
							}
						}
						for (int j=0; j <predictions[0].length; j++){
							distance[2+j]=this.target.GetElement(s, j);
						}
						if (distance[0]>=largest_distance){
							largest_distance=distance[0];
							largest_distance_case=check_neighbours;
						}
						hold_distances[check_neighbours]=distance;
						check_neighbours++;
						}
					else if (distance[0]<hold_distances[largest_distance_case][0]){

								distance[1]=this.weight[s];
								if (this.Type.equals("weights")){
									if (distance[0]!=0.0){
										distance[1]*=1/distance[0];
										} else {
											distance[1]=99999999999.99;
										}
								}
								for (int j=0; j <predictions[0].length; j++){
									distance[2+j]=this.target.GetElement(s, j);
								}
								hold_distances[largest_distance_case]=distance;

								//find new highest distance
								largest_distance_case=find_largest_distance(hold_distances);
								largest_distance=hold_distances[largest_distance_case][0];
						
					}
					// loop for training set ends here
			
			}	
				// calculate probabilities based on the chosen method
					//System.out.println(" predictions length: " + predictions[i][0]);
					// simple counts
					double sumforclassification=0.0;
					for (int c=0; c<this.predictions[0].length;c++){
						 predictions[i][c]=0.0;
						double sumweight=0.0;
						
					for (int kn=0; kn< hold_distances.length; kn++){
								//System.out.println(Arrays.toString(hold_distances.get(kn)));
								//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
								predictions[i][c]+=hold_distances[kn][2+c]*hold_distances[kn][1];
								//System.out.println(predictions[i][c]);
								sumweight+=hold_distances[kn][1];
						}
					//System.out.println(predictions[i][c] + " "+ sumweight);
					predictions[i][c]/=sumweight;
					sumforclassification+=predictions[i][c];
					//System.out.println(predictions[i][c]);
					}
					if (isClassification){
						for (int c=0; c<this.predictions[0].length;c++){
							predictions[i][c]/=sumforclassification;
						}
						
					}
			
			
			
			
			
			// end of loop
		}
		
			

			// end of SGD

	}
	/**
	 * 
	 * @param data : data to use for predictions
	 * <p> finds nearest neighbours based on the stored data and the given data
	 */
	private void fit(smatrix data) {
		// make sensible checks
		if (data==null){
			throw new IllegalStateException(" Main data object is null or has too few cases" );
		}
		// Initialise scaler

		if (Scaler==null){
			Scaler = new maxscaler();
		}				
		if ( this.usescale && Scaler.IsFitted()==false){
			Scaler.fit(helddataset);
		}
		
		// set random number generator 
		random = new  XorShift128PlusRandom(seed);
		int thresoldcut=utilis.util.get_random_integer(this.sub_sample_percent);
		
		for (int i=this.start_array; i <this.end_array; i++){

			IntIntMapminus4a has_index_main=new IntIntMapminus4a(data.indexpile[i+1]-data.indexpile[i]  , 0.99F);
			for (int v=data.indexpile[i]; v<data.indexpile[i+1];v++ ){
				has_index_main.put(data.mainelementpile[v],v);
			}
			double[][] hold_distances  = new double[this.neighbours][2 + predictions[0].length];
			int check_neighbours=0;
			int largest_distance_case=-1;
			double largest_distance=Double.MIN_VALUE;				
			
			for (int s=0; s< this.helddataset.GetRowDimension();s++){
				if (random.nextInt()>thresoldcut) {
					continue;
				}
				double extra_sum=0.0;
				double temp=0.0;

				 if (this.distance.equals("cityblock")){
					 for (int v=helddataset.indexpile[s]; v<helddataset.indexpile[s+1];v++ ){
						 Integer column=helddataset.mainelementpile[v];
						 if (has_index_main.get(column)==-1){
							 extra_sum+=Math.abs(helddataset.valuespile[v]);
		    				}
				 }
					
    				
				}else if(this.distance.equals("euclidean")){
					 for (int v=helddataset.indexpile[s]; v<helddataset.indexpile[s+1];v++ ){
						 Integer column=helddataset.mainelementpile[v];
						 if (has_index_main.get(column)==-1){
							 temp=helddataset.valuespile[v];
							 extra_sum+=temp*temp;
		    				}
				 }
				}else if(this.distance.equals("cosine")){
					 for (int v=helddataset.indexpile[s]; v<helddataset.indexpile[s+1];v++ ){
						 Integer column=helddataset.mainelementpile[v];
						 if (has_index_main.get(column)==-1){
							 temp=helddataset.valuespile[v];
							 extra_sum+=1.0;
		    				}
				 }
					
				}
					double distance[]= new double [2 + predictions[0].length];
					//distance[0]=y_memory[i];
					
					if (this.distance.equals("cityblock")){
					for(int jj=data.indexpile[i]; jj <data.indexpile[i+1]; jj++ ) {
						int j=data.mainelementpile[jj];
						double x1=data.valuespile[jj];
						if (this.usescale){
							x1=Scaler.transform(x1, j);
						}
						double x2=0.0;
						//get feature from sparse array
						
						x2=helddataset.GetElement(s, j);
						if (this.usescale){
    						x2=Scaler.transform(x2, j);
						}
						distance[0]+=Math.abs(x1-x2);

					 
					}
					distance[0]+=extra_sum;
					
					}
					else if(this.distance.equals("euclidean")){
						for(int jj=data.indexpile[i]; jj <data.indexpile[i+1]; jj++ ) {
							int j=data.mainelementpile[jj];
							double x1=data.valuespile[jj];
							if (this.usescale){
								x1=Scaler.transform(x1, j);
							}
							double x2=0.0;
							//get feature from sparse array
							x2=helddataset.GetElement(s, j);
							if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
							distance[0]+=(x1-x2)*(x1-x2);
						
						}
						distance[0]+=extra_sum;
					
						}	
						
					else if(this.distance.equals("cosine")){
						double product=0.0;
						double abssumx1=0.0;
						double abssumx2=0.0;
						
						for(int jj=data.indexpile[i]; jj <data.indexpile[i+1]; jj++ ) {
							int j=data.mainelementpile[jj];
							double x1=data.valuespile[jj];
							if (this.usescale){
								x1=Scaler.transform(x1, j);
							}
							double x2=0.0;
							//get feature from sparse array

							x2=helddataset.GetElement(s, j);
							if (this.usescale){
	    						x2=Scaler.transform(x2, j);
							}
		    					product+=x1*x2;
		    					abssumx1+=x1*x1;
		    					abssumx2+=x2*x2;		
		    				 
						}
						distance[0]+=extra_sum;
						distance[0]=1- ( product/(Math.sqrt(abssumx1)*Math.sqrt(abssumx2)+0.0000001));
						}		
						
					// adjust for weights
					
					//distance[0]*=1/this.weight[s];
					
					//check if the new case is small enough to be isnerted
					
					if(check_neighbours<neighbours){
						distance[1]=this.weight[s];
						if (this.Type.equals("weights")){
							if (distance[0]!=0.0){
							distance[1]*=1/distance[0];
							} else {
								distance[1]=99999999999.99;
							}
						}
						for (int j=0; j <predictions[0].length; j++){
							distance[2+j]=this.target.GetElement(s, j);
						}
						if (distance[0]>=largest_distance){
							largest_distance=distance[0];
							largest_distance_case=check_neighbours;
						}
						hold_distances[check_neighbours]=distance;
						check_neighbours++;
						}
					else if (distance[0]<hold_distances[largest_distance_case][0]){

								distance[1]=this.weight[s];
								if (this.Type.equals("weights")){
									if (distance[0]!=0.0){
										distance[1]*=1/distance[0];
										} else {
											distance[1]=99999999999.99;
										}
								}
								for (int j=0; j <predictions[0].length; j++){
									distance[2+j]=this.target.GetElement(s, j);
								}
								hold_distances[largest_distance_case]=distance;

								//find new highest distance
								largest_distance_case=find_largest_distance(hold_distances);
								largest_distance=hold_distances[largest_distance_case][0];
						
					}
					// loop for training set ends here
			
			}	
				// calculate probabilities based on the chosen method
					//System.out.println(" predictions length: " + predictions[i][0]);
					// simple counts
					double sumforclassification=0.0;
					for (int c=0; c<this.predictions[0].length;c++){
						 predictions[i][c]=0.0;
						double sumweight=0.0;
						
					for (int kn=0; kn< hold_distances.length; kn++){
								//System.out.println(Arrays.toString(hold_distances.get(kn)));
								//System.out.println(hold_distances.get(kn)[2+c]*hold_distances.get(kn)[1]);
								predictions[i][c]+=hold_distances[kn][2+c]*hold_distances[kn][1];
								//System.out.println(predictions[i][c]);
								sumweight+=hold_distances[kn][1];
						}
					//System.out.println(predictions[i][c] + " "+ sumweight);
					predictions[i][c]/=sumweight;
					sumforclassification+=predictions[i][c];
					//System.out.println(predictions[i][c]);
					}
					if (isClassification){
						for (int c=0; c<this.predictions[0].length;c++){
							predictions[i][c]/=sumforclassification;
						}
						
					}
			
			
			
			
			
			// end of loop
		}
		
			

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
	 * @param sc : scaler object to be set
	 */
	public void setScaler(scaler sc) {
		this.Scaler=sc;
		
	}
	/**
	 * @param percentage : a double value between [0,1]
	 * @return the closest integer that reflects this percentage
	 * <p> random.nextint can be significantly faster than nextdouble()
	 */
	public int get_random_integer(double percentage){
		
		double per= Math.min(Math.max(0, percentage),1.0);
		double difference=2147483647.0+(2147483648.0);
		int point=(int)(-2147483648.0 +  (per*difference ));
		
		
		return point;
		
	}
}
