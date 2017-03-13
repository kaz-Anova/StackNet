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


package crossvalidation.splits;
import java.util.Random;
import matrix.fsmatrix;
import matrix.smatrix;

/**
 * 
 * @author marios
 *<p> Purpose of the package is to provide functionality for splitting data sets and variables Randmomly </p>
 */
public class RandomSplit {

	/**
	 *  string id column
	 */
	public String Stringidcolumn [];
	/**
	 *  string id column for training set
	 */
	private String Stringidcolumn_train [];
	/**
	 *  string id column for test set
	 */
	private String Stringidcolumn_test [];
	/**
	*  int id column
	*/
	public int intidcolumn [];	
	/**
	*  int id column for training
	*/
	private int intidcolumn_train [];	
	/**
	*  int id column for test
	*/
	private int intidcolumn_test [];	
	/**
	 * double target variable
	 */
	public  double target [];
	/**
	 * double target variable for training
	 */
	private  double target_train [];	
	/**
	 * double target variable for test
	 */
	private  double target_test [];
	 /**
	  * double target in 2d for training
	  */
	private  double target2d_train[][];
	 /**
	  * double target in 2d for test
	  */
	private  double target2d_test[][];
	 /**
	  * double target in 2d
	  */
	public  double target2d[][];
	 /**
	  * String target 
	  */
	public String targets [];
	 /**
	  * String target for training 
	  */
	private String targets_train [];
	 /**
	  * String target for test 
	  */
	private String targets_test [];
	 /**
	  * String target in 2d
	  */	 
	public String target2ds [][]; 
	 /**
	  * String target in 2d for training 
	  */	 
	private String target2ds_train [][]; 
	 /**
	  * String target in 2d for test 
	  */	 
	private String target2ds_test [][]; 
	/**
	 * the weights
	 */
	public double weights [];
	/**
	 * the weights for training 
	 */
	private double weights_train [];
	/**
	 * the weights for test 
	 */
	private double weights_test [];
	/**
	 * the fsmatrix for training 
	 */
	private fsmatrix fsmatrix_train;
	/**
	 * the fsmatrix fo test 
	 */
	private fsmatrix fsmatrix_test ;
	/**
	 * the smatrix for training 
	 */
	private smatrix smatrix_train;
	/**
	 * the smatrix fo test 
	 */
	private smatrix smatrix_test ;
	 /**
	  * double data for training
	  */
	private  double data_train[][];
	 /**
	  * double data for test
	  */
	private  double data_test[][];
	 /**
	  * String data for training
	  */
	private  String datas_train[][];
	 /**
	  * String data for test
	  */
	private  String datas_test[][];
	
	/**
	 * if specifies the percentage of the initial set that goes to the test set,default to 0.5
	 */
	public double cutoff =0.5;
	/**
	 * seed to replicate randomized procedures.
	 */
	public long seed=10;
	/**
	 * if true it prints stuff
	 */
	public boolean verbose =false;
	/**
	 * maximum number of consecutive iterations until both train and test have cases allicated to them
	 */
	public int max_iters=10;
	
	//Getter Methods for the  private objects
	/**
	 * 
	 * @return String ID for training if String id was not null during the splitting process
	 */
	public String [] GetsStringIdTrain(){
		
		if (Stringidcolumn_train==null || Stringidcolumn_train.length<=0){
			throw new exceptions.IllegalStateException(" String Id for training is empty");
		}
	
		return Stringidcolumn_train;
	}
	/**
	 * 
	 * @return String ID for test if String id was not null during the splitting process
	 */
	public String [] GetsStringIdTest(){
		
		if (Stringidcolumn_test==null || Stringidcolumn_test.length<=0){
			throw new exceptions.IllegalStateException(" String Id for test is empty");
		}
		return Stringidcolumn_test;
	}
	/**
	 * 
	 * @return int ID for training if int id was not null during the splitting process
	 */
	public int [] GetsintIdTrain(){
		if (intidcolumn_train==null || intidcolumn_train.length<=0){
			throw new exceptions.IllegalStateException(" int Id for training is empty");
		}
		return intidcolumn_train;
	}
	/**
	 * 
	 * @return int ID for test if int id was not null during the splitting process
	 */
	public int [] GetsSintIdTest(){
		if (intidcolumn_test==null || intidcolumn_test.length<=0){
			throw new exceptions.IllegalStateException(" int Id for test is empty");
		}
		return intidcolumn_test;
	}	
	/**
	 * 
	 * @return double target for training if target was not null during the splitting process
	 */
	public double [] GettargetTrain(){
		if (target_train==null || target_train.length<=0){
			throw new exceptions.IllegalStateException(" target for training is empty");
		}		
		return target_train;
	}
	/**
	 * 
	 * @return double target for test if  target was not null during the splitting process
	 */
	public double [] GetstargetTest(){
		if (target_test==null || target_test.length<=0){
			throw new exceptions.IllegalStateException(" target for test is empty");
		}
		return target_test;
	}
	/**
	 * 
	 * @return double target 2D for training if target2d was not null during the splitting process
	 */
	public double [][] Gettarget2DTrain(){
		if ( target2d_train==null ||  target2d_train.length<=0){
			throw new exceptions.IllegalStateException(" target for training2d is empty");
		}	
		return target2d_train;
	}
	/**
	 * 
	 * @return double target 2D for test if  target2d was not null during the splitting process
	 */
	public double [][] Getstarget2DTest(){
		if (target2d_test==null || target2d_test.length<=0){
			throw new exceptions.IllegalStateException(" target for test 2d is empty");
		}	
		return target2d_test;
	}	
		
	/**
	 * 
	 * @return String target for training if String target was not null during the splitting process
	 */
	public String [] GettargetStringTrain(){
		if ( targets_train==null ||  targets_train.length<=0){
			throw new exceptions.IllegalStateException(" target string for training is empty");
		}		
		return targets_train;
	}
	/**
	 * 
	 * @return String target for test if String target was not null during the splitting process
	 */
	public String [] GetstargetStringTest(){
		if (targets_test==null || targets_test.length<=0){
			throw new exceptions.IllegalStateException(" target for test string is empty");
		}		
		return targets_test;
	}
	/**
	 * 
	 * @return String target 2D for training if String target 2D was not null during the splitting process
	 */
	public String [][] GettargetString2DTrain(){
		if ( target2ds_train==null ||  target2ds_train.length<=0){
			throw new exceptions.IllegalStateException(" target string2d for training is empty");
		}		
		return target2ds_train;
	}
	/**
	 * 
	 * @return String target 2D for test if String target 2D was not null during the splitting process
	 */
	public String [][] GetstargetString2DTest(){
		if (targets_test==null || targets_test.length<=0){
			throw new exceptions.IllegalStateException(" target for test2d string is empty");
		}		
		return target2ds_test;
	}	
	
	/**
	 * 
	 * @return double weights for training if weights was not null during the splitting process
	 */
	public double [] GetWeightsTrain(){
		if (weights_train==null || weights_train.length<=0){
			throw new exceptions.IllegalStateException(" weights for training is empty");
		}		
		return weights_train;
	}
	/**
	 * 
	 * @return double weights for test if weights was not null during the splitting process
	 */
	public double [] GetWeightsTest(){
		if (weights_test==null || weights_test.length<=0){
			throw new exceptions.IllegalStateException(" weights for test is empty");
		}		
	
		return weights_test;
	}
	
	/**
	 * 
	 * @return fsmatrix  for training if splitfsmatrix was run properly
	 */
	public fsmatrix  GetfsmatrixTrain(){
		if (fsmatrix_train==null || fsmatrix_train.GetRowDimension()<=0){
			throw new exceptions.IllegalStateException(" fsmatrix for training is empty");
		}		
		return fsmatrix_train;
	}
	/**
	 * 
	 * @return fsmatrix  for test if splitfsmatrix was run properly
	 */
	public fsmatrix GetfsmatrixTest(){
		if (fsmatrix_test==null || fsmatrix_test.GetRowDimension()<=0){
			throw new exceptions.IllegalStateException(" fsmatrix for test is empty");
		}		
	
		return fsmatrix_test;
	}
	/**
	 * 
	 * @return smatrix  for training if splitsmatrix was run properly
	 */
	public smatrix  GetsmatrixTrain(){
		if (smatrix_train==null || smatrix_train.GetRowDimension()<=0){
			throw new exceptions.IllegalStateException(" smatrix for training is empty");
		}		
		return smatrix_train;
	}
	/**
	 * 
	 * @return smatrix  for test if splitsmatrix was run properly
	 */
	public smatrix GetsmatrixTest(){
		if (smatrix_test==null || smatrix_test.GetRowDimension()<=0){
			throw new exceptions.IllegalStateException(" smatrix for test is empty");
		}		
	
		return smatrix_test;
	}
	/**
	 * 
	 * @return double data  for training if splitdata was run properly
	 */
	public double [][] GetDataTrain(){
		if ( data_train==null ||  data_train.length<=0){
			throw new exceptions.IllegalStateException(" data for training is empty");
		}	
		return data_train;
	}
	/**
	 * 
	 * @return double data for test if splitdata was run properly
	 */
	public double [][] GetDataTest(){
		if (data_test==null || data_test.length<=0){
			throw new exceptions.IllegalStateException(" data for test 2d is empty");
		}	
		return data_test;
	}	
	
	/**
	 * 
	 * @return String data  for training if splitStringdata was run properly
	 */
	public String [][] GetStringDataTrain(){
		if ( datas_train==null ||  datas_train.length<=0){
			throw new exceptions.IllegalStateException(" data String for training is empty");
		}	
		return datas_train;
	}
	/**
	 * 
	 * @return String data  for test if splitStringdata was run properly
	 */
	public String [][] GetStringDataTest(){
		if (datas_test==null || datas_test.length<=0){
			throw new exceptions.IllegalStateException(" data String for test is empty");
		}	
		return datas_test;
	}	
	
	/**
	 * 
	 * @param f : fixed size matrix to split randomly
	 * <p> splits a fs matrix along with the other objects provided as public (targets, weights, ids etc). Also the rest of the parameters remain public,,ic too.
	 * Additionally the objects do haves specific names (for example target2d is meant for 2dimensional arrays as multiple labels), however the user may split anything
	 * that fits to a 2dimensional double and so one for the rest of the objects.
	 */
	
	public void splitfsmatrix(fsmatrix f){
		
		//check that cuttoff value is sensible
		if (cutoff>0.995 || cutoff<0.005 ){
			throw new exceptions.IllegalStateException(" cuttof cannot be less than 0.005 or more than 0.995");
		}
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.GetRowDimension();
		int fscols=f.GetColumnDimension();
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

        
        if (fscols<=0 || fsrows<=1){
        	throw new IllegalStateException(" It seemes there is no pointing splitting given the minimal size of the matrix");	        
        }
		if (Stringidcolumn!=null) {
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match the one in fsmatrix");	        
			} else {
				HasStringid=true;
			}
			
		} else if (intidcolumn!=null) {
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}else {
				Hasintgid=true;
			}
		}
		if (target!=null) {
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}else {
				Hasdoubletarget=true;
			}
	    } if (target2d!=null) {
	    	if (target2d.length!=fsrows){
	    		throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
	    	}else {
	    		Hasdoubletarget2d=true;
			}
	    	
	    } if (targets!=null) {
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}else {
				HasStringtarget=true;
			}
			
		  } if (target2ds!=null) {
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}else {
					HasStringtarget2d=true;
				}
			  }
		if (weights!=null) {
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }else {
            	 Hasweights=true;
				}
          }
         //Initialize a generator
		 Random rand = new Random();
		 rand.setSeed(seed); // set the seed 
		 // determine number of cases to for validation and training
         int sum_train=0;
         int sum_test=0;
         //object to hold with 1 the case to test and 0 the cases for training 
         int splitter [] = null;
         int current_iters=0;
         while ((sum_train==0 || sum_test==0) && current_iters<max_iters){
        	 sum_train=0;
        	 sum_test=0;
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        		 }
        	 }
        	 current_iters++; 
         }
         
         // if after max_iters we have not managed to get cases for both training and test,
         //then we run one more time and assign the last case to the weak side, whichever it is
         if (sum_train==0 || sum_test==0) {
           	 sum_train=0;
        	 sum_test=0;
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows-1 ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        		 }
        	 } 
        	 //assign the last element to the weak side
        	 if (sum_train==0) {
        		 splitter[fsrows-1]=0;
        		 sum_train++;
        	 } else if (sum_test==0){
        		 splitter[fsrows-1]=1;
        		 sum_test++;
        	 }
        	 
         }
         
         //initialise fs matrix objects
         
         fsmatrix_train = new fsmatrix(new double [sum_train * fscols], sum_train, fscols);
         fsmatrix_test = new fsmatrix(new double [sum_test * fscols], sum_test, fscols); 
         
         // initialise rest of the objects if valid
         
 		if (HasStringid) {
 			Stringidcolumn_train= new String[sum_train];
 			Stringidcolumn_test= new String[sum_test];
			}
 		if (Hasintgid) {
 			intidcolumn_train= new int[sum_train];
 			intidcolumn_test= new int[sum_test];
			}	
 		if (Hasdoubletarget) {
 			target_train= new double[sum_train];
 			target_test= new double[sum_test];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_train= new double[sum_train][];
 			target2d_test= new double[sum_test][];
			}				
 		if (HasStringtarget) {
 			targets_train= new String[sum_train];
 			targets_test= new String[sum_test];
			}				
 		if (HasStringtarget2d) {
 			target2ds_train= new String[sum_train][];
 			target2ds_test= new String[sum_test][];
			}	
 		if (Hasweights) {
 			weights_train= new double[sum_train];
 			weights_test= new double[sum_test];
			}
 		//do the allocation of objects
 		int count_train=0;
 		int count_test=0;
 		for (int i=0; i < splitter.length; i++){
 			int value=splitter[i];
 			if (value==1) { // case is allocated to test set
 		 		if (HasStringid) {

 		 			Stringidcolumn_test[count_test]=Stringidcolumn[i];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[count_test]= intidcolumn[i];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[count_test]= target[i];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[count_test]= target2d[i];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[count_test]= targets[i];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[count_test]= target2ds[i];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[count_test]= weights[i];
 					}
 		 		// the fsmatrix
 		 		 for (int j=0; j < fscols; j++){
 		 			fsmatrix_test.SetElement(count_test, j, f.GetElement(i, j));
 		 		 }
 		 		
 		 		count_test++;				
 			} else { //case is allocated to training
 		 		if (HasStringid) {
 		 			Stringidcolumn_train[count_train]= Stringidcolumn[i];

 					}
 		 		if (Hasintgid) {
 		 			intidcolumn_train[count_train]= intidcolumn[i];

 					}	
 		 		if (Hasdoubletarget) {
 		 			target_train[count_train]= target[i];

 					}	
 		 		if (Hasdoubletarget2d) {
 		 			target2d_train[count_train]= target2d[i];

 					}				
 		 		if (HasStringtarget) {
 		 			targets_train[count_train]=targets[i];

 					}				
 		 		if (HasStringtarget2d) {
 		 			target2ds_train[count_train]= target2ds[i];

 					}	
 		 		if (Hasweights) {
 		 			weights_train[count_train]= weights[i];

 					}
 				// fs,atrix_train
		 		 for (int j=0; j < fscols; j++){
		 			fsmatrix_train.SetElement(count_train, j, f.GetElement(i, j));
		 		 }
 		 		count_train++;
 			}
 			
 			
 		}
        if (this.verbose){
        	System.out.println(" fs matrix was splitted with " + sum_train + " cases in train and " + sum_test + " in test");
        	System.out.println(" its counterparts can be accessed via GetfsmatrixTrain/GetfsmatrixTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTrain/GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTrain/GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTrain/GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTrain/Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTrain/GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTrain/GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTrain/GetWeightsTest");
        		}          		
        	}
        }
   
         
	
		//END OF METHOD
	}
	/**
	 * 
	 * @param data : 2dimensional Array to split
	 * <p> splits a double [][] along with the other objects provided as public (targets, weights, ids etc). Also the rest of the parameters remain public,,ic too.
	 * Additionally the objects do haves specific names (for example target2d is meant for 2dimensional arrays as multiple labels), however the user may split anything
	 * that fits to a 2dimensional double and so one for the rest of the objects.
	 */
	
	public void splitData(double data [][]){
		
		//check that cuttoff value is sensible
		if (cutoff>0.995 || cutoff<0.005 ){
			throw new exceptions.IllegalStateException(" cuttof cannot be less than 0.005 or more than 0.995");
		}
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=data.length;
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

        
        if (fsrows<=1){
        	throw new IllegalStateException(" It seemes there is no pointing splitting given the minimal size of the matrix");	        
        }
		if (Stringidcolumn!=null) {
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match the one in fsmatrix");	        
			} else {
				HasStringid=true;
			}
			
		} else if (intidcolumn!=null) {
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}else {
				Hasintgid=true;
			}
		}
		if (target!=null) {
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}else {
				Hasdoubletarget=true;
			}
	    } if (target2d!=null) {
	    	if (target2d.length!=fsrows){
	    		throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
	    	}else {
	    		Hasdoubletarget2d=true;
			}
	    	
	    } if (targets!=null) {
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}else {
				HasStringtarget=true;
			}
			
		  } if (target2ds!=null) {
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}else {
					HasStringtarget2d=true;
				}
			  }
		if (weights!=null) {
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }else {
            	 Hasweights=true;
				}
          }
         //Initialize a generator
		 Random rand = new Random();
		 rand.setSeed(seed); // set the seed 
		 // determine number of cases to for validation and training
         int sum_train=0;
         int sum_test=0;
         //object to hold with 1 the case to test and 0 the cases for training 
         int splitter [] = null;
         int current_iters=0;
         while ((sum_train==0 || sum_test==0) && current_iters<max_iters){
        	 sum_train=0;
        	 sum_test=0;
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        		 }
        	 }
        	 current_iters++; 
         }
         
         // if after max_iters we have not managed to get cases for both training and test,
         //then we run one more time and assign the last case to the weak side, whichever it is
         if (sum_train==0 || sum_test==0) {
           	 sum_train=0;
        	 sum_test=0;
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows-1 ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        		 }
        	 } 
        	 //assign the last element to the weak side
        	 if (sum_train==0) {
        		 splitter[fsrows-1]=0;
        		 sum_train++;
        	 } else if (sum_test==0){
        		 splitter[fsrows-1]=1;
        		 sum_test++;
        	 }
        	 
         }
         
         //initialise fs matrix objects
         
         data_train = new double [sum_train ][];
         data_test = new double [sum_test ][];
         
         // initialise rest of the objects if valid
         
 		if (HasStringid) {
 			Stringidcolumn_train= new String[sum_train];
 			Stringidcolumn_test= new String[sum_test];
			}
 		if (Hasintgid) {
 			intidcolumn_train= new int[sum_train];
 			intidcolumn_test= new int[sum_test];
			}	
 		if (Hasdoubletarget) {
 			target_train= new double[sum_train];
 			target_test= new double[sum_test];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_train= new double[sum_train][];
 			target2d_test= new double[sum_test][];
			}				
 		if (HasStringtarget) {
 			targets_train= new String[sum_train];
 			targets_test= new String[sum_test];
			}				
 		if (HasStringtarget2d) {
 			target2ds_train= new String[sum_train][];
 			target2ds_test= new String[sum_test][];
			}	
 		if (Hasweights) {
 			weights_train= new double[sum_train];
 			weights_test= new double[sum_test];
			}
 		//do the allocation of objects
 		int count_train=0;
 		int count_test=0;
 		for (int i=0; i < splitter.length; i++){
 			int value=splitter[i];
 			if (value==1) { // case is allocated to test set
 		 		if (HasStringid) {

 		 			Stringidcolumn_test[count_test]=Stringidcolumn[i];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[count_test]= intidcolumn[i];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[count_test]= target[i];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[count_test]= target2d[i];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[count_test]= targets[i];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[count_test]= target2ds[i];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[count_test]= weights[i];
 					}
 		 		// the fsmatrix
 		 		data_test[count_test]=data[i];

 		 		
 		 		count_test++;				
 			} else { //case is allocated to training
 		 		if (HasStringid) {
 		 			Stringidcolumn_train[count_train]= Stringidcolumn[i];

 					}
 		 		if (Hasintgid) {
 		 			intidcolumn_train[count_train]= intidcolumn[i];

 					}	
 		 		if (Hasdoubletarget) {
 		 			target_train[count_train]= target[i];

 					}	
 		 		if (Hasdoubletarget2d) {
 		 			target2d_train[count_train]= target2d[i];

 					}				
 		 		if (HasStringtarget) {
 		 			targets_train[count_train]=targets[i];

 					}				
 		 		if (HasStringtarget2d) {
 		 			target2ds_train[count_train]= target2ds[i];

 					}	
 		 		if (Hasweights) {
 		 			weights_train[count_train]= weights[i];

 					}

	 		 		data_train[count_train]=data[i];
 		 		count_train++;
 			}
 			
 			
 		}
        if (this.verbose){
        	System.out.println(" data matrix was splitted with " + sum_train + " cases in train and " + sum_test + " in test");
        	System.out.println(" its counterparts can be accessed via GetDataTrain/GetDataTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTrain/GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTrain/GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTrain/GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTrain/Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTrain/GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTrain/GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTrain/GetWeightsTest");
        		}          		
        	}
        }
   
         
	
		//END OF METHOD
	}	
	/**
	 * 
	 * @param data : 2dimensional String Array to split
	 * <p> splits a a String [][]  along with the other objects provided as public (targets, weights, ids etc). Also the rest of the parameters remain public,,ic too.
	 * Additionally the objects do haves specific names (for example target2d is meant for 2dimensional arrays as multiple labels), however the user may split anything
	 * that fits to a 2dimensional double and so one for the rest of the objects.
	 */
	
	public void spliStringtData(String data [][]){
		
		//check that cuttoff value is sensible
		if (cutoff>0.995 || cutoff<0.005 ){
			throw new exceptions.IllegalStateException(" cuttof cannot be less than 0.005 or more than 0.995");
		}
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=data.length;
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

        
        if (fsrows<=1){
        	throw new IllegalStateException(" It seemes there is no pointing splitting given the minimal size of the matrix");	        
        }
		if (Stringidcolumn!=null) {
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match the one in fsmatrix");	        
			} else {
				HasStringid=true;
			}
			
		} else if (intidcolumn!=null) {
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}else {
				Hasintgid=true;
			}
		}
		if (target!=null) {
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}else {
				Hasdoubletarget=true;
			}
	    } if (target2d!=null) {
	    	if (target2d.length!=fsrows){
	    		throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
	    	}else {
	    		Hasdoubletarget2d=true;
			}
	    	
	    } if (targets!=null) {
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}else {
				HasStringtarget=true;
			}
			
		  } if (target2ds!=null) {
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}else {
					HasStringtarget2d=true;
				}
			  }
		if (weights!=null) {
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }else {
            	 Hasweights=true;
				}
          }
         //Initialize a generator
		 Random rand = new Random();
		 rand.setSeed(seed); // set the seed 
		 // determine number of cases to for validation and training
         int sum_train=0;
         int sum_test=0;
         //object to hold with 1 the case to test and 0 the cases for training 
         int splitter [] = null;
         int current_iters=0;
         while ((sum_train==0 || sum_test==0) && current_iters<max_iters){
        	 sum_train=0;
        	 sum_test=0;
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        		 }
        	 }
        	 current_iters++; 
         }
         
         // if after max_iters we have not managed to get cases for both training and test,
         //then we run one more time and assign the last case to the weak side, whichever it is
         if (sum_train==0 || sum_test==0) {
           	 sum_train=0;
        	 sum_test=0;
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows-1 ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        		 }
        	 } 
        	 //assign the last element to the weak side
        	 if (sum_train==0) {
        		 splitter[fsrows-1]=0;
        		 sum_train++;
        	 } else if (sum_test==0){
        		 splitter[fsrows-1]=1;
        		 sum_test++;
        	 }
        	 
         }
         
         //initialise fs matrix objects
         
         datas_train = new String [sum_train ][];
         datas_test = new String [sum_test ][];
         
         // initialise rest of the objects if valid
         
 		if (HasStringid) {
 			Stringidcolumn_train= new String[sum_train];
 			Stringidcolumn_test= new String[sum_test];
			}
 		if (Hasintgid) {
 			intidcolumn_train= new int[sum_train];
 			intidcolumn_test= new int[sum_test];
			}	
 		if (Hasdoubletarget) {
 			target_train= new double[sum_train];
 			target_test= new double[sum_test];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_train= new double[sum_train][];
 			target2d_test= new double[sum_test][];
			}				
 		if (HasStringtarget) {
 			targets_train= new String[sum_train];
 			targets_test= new String[sum_test];
			}				
 		if (HasStringtarget2d) {
 			target2ds_train= new String[sum_train][];
 			target2ds_test= new String[sum_test][];
			}	
 		if (Hasweights) {
 			weights_train= new double[sum_train];
 			weights_test= new double[sum_test];
			}
 		//do the allocation of objects
 		int count_train=0;
 		int count_test=0;
 		for (int i=0; i < splitter.length; i++){
 			int value=splitter[i];
 			if (value==1) { // case is allocated to test set
 		 		if (HasStringid) {

 		 			Stringidcolumn_test[count_test]=Stringidcolumn[i];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[count_test]= intidcolumn[i];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[count_test]= target[i];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[count_test]= target2d[i];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[count_test]= targets[i];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[count_test]= target2ds[i];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[count_test]= weights[i];
 					}
 		 		// the fsmatrix
 		 		datas_test[count_test]=data[i];

 		 		
 		 		count_test++;				
 			} else { //case is allocated to training
 		 		if (HasStringid) {
 		 			Stringidcolumn_train[count_train]= Stringidcolumn[i];

 					}
 		 		if (Hasintgid) {
 		 			intidcolumn_train[count_train]= intidcolumn[i];

 					}	
 		 		if (Hasdoubletarget) {
 		 			target_train[count_train]= target[i];

 					}	
 		 		if (Hasdoubletarget2d) {
 		 			target2d_train[count_train]= target2d[i];

 					}				
 		 		if (HasStringtarget) {
 		 			targets_train[count_train]=targets[i];

 					}				
 		 		if (HasStringtarget2d) {
 		 			target2ds_train[count_train]= target2ds[i];

 					}	
 		 		if (Hasweights) {
 		 			weights_train[count_train]= weights[i];

 					}

	 		 		datas_train[count_train]=data[i];
 		 		count_train++;
 			}
 			
 			
 		}
        if (this.verbose){
        	System.out.println(" data String matrix was splitted with " + sum_train + " cases in train and " + sum_test + " in test");
        	System.out.println(" its counterparts can be accessed via GetStringDataTrain/GetStringDataTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTrain/GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTrain/GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTrain/GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTrain/Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTrain/GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTrain/GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTrain/GetWeightsTest");
        		}          		
        	}
        }
   
         
	
		//END OF METHOD
	}	
	
	
	/**
	 * 
	 * @param f : Sparse matrix to split randomly
	 * <p> splits an smatrix along with the other objects provided as public (targets, weights, ids etc). Also the rest of the parameters remain public,,ic too.
	 * Additionally the objects do haves specific names (for example target2d is meant for 2dimensional arrays as multiple labels), however the user may split anything
	 * that fits to a 2dimensional double and so one for the rest of the objects.
	 */
	
	public void splitsmatrix(smatrix f){
		
		//check that cuttoff value is sensible
		if (cutoff>0.995 || cutoff<0.005 ){
			throw new exceptions.IllegalStateException(" cuttof cannot be less than 0.005 or more than 0.995");
		}
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.GetRowDimension();
		int fscols=f.GetColumnDimension();
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

        
        if (fscols<=0 || fsrows<=1){
        	throw new IllegalStateException(" It seemes there is no pointing splitting given the minimal size of the matrix");	        
        }
		if (Stringidcolumn!=null) {
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match the one in fsmatrix");	        
			} else {
				HasStringid=true;
			}
			
		} else if (intidcolumn!=null) {
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}else {
				Hasintgid=true;
			}
		}
		if (target!=null) {
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}else {
				Hasdoubletarget=true;
			}
	    } if (target2d!=null) {
	    	if (target2d.length!=fsrows){
	    		throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
	    	}else {
	    		Hasdoubletarget2d=true;
			}
	    	
	    } if (targets!=null) {
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}else {
				HasStringtarget=true;
			}
			
		  } if (target2ds!=null) {
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}else {
					HasStringtarget2d=true;
				}
			  }
		if (weights!=null) {
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }else {
            	 Hasweights=true;
				}
          }
		  // get row elements
	    if (!f.IsSortedByRow()){
	    	f.convert_type();
	    }

		
         //Initialize a generator
		 Random rand = new Random();
		 rand.setSeed(seed); // set the seed 
		 // determine number of cases to for validation and training
         int sum_train=0;
         int sum_test=0;
         int sum_trainln=0;
         int sum_testln=0;         
         //object to hold with 1 the case to test and 0 the cases for training 
         int splitter [] = null;
         int current_iters=0;
         while ((sum_train==0 || sum_test==0) && current_iters<max_iters){
        	 sum_train=0;
        	 sum_test=0;
        	 sum_trainln=0;
        	 sum_testln=0;        	 
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        			 sum_testln+=f.indexpile[i+1]-f.indexpile[i];
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        			 sum_trainln+=f.indexpile[i+1]-f.indexpile[i];
        		 }
        	 }
        	 current_iters++; 
         }
         
         // if after max_iters we have not managed to get cases for both training and test,
         //then we run one more time and assign the last case to the weak side, whichever it is
         if (sum_train==0 || sum_test==0) {
           	 sum_train=0;
        	 sum_test=0;
        	 sum_trainln=0;
        	 sum_testln=0;         	 
        	 splitter=new int [fsrows];
        	 for (int i=0; i < fsrows-1 ; i++){
        		 if (rand.nextDouble()<=cutoff){
        			 splitter[i]=1;
        			 sum_test++;
        			 sum_testln+=f.indexpile[i+1]-f.indexpile[i];
        		 } else {
        			 splitter[i]=0;
        			 sum_train++;
        			 sum_trainln+=f.indexpile[i+1]-f.indexpile[i];
        			 
        		 }
        	 } 
        	 //assign the last element to the weak side
        	 if (sum_train==0) {
        		 splitter[fsrows-1]=0;
        		 sum_train++;
        		 sum_trainln+=f.indexpile[fsrows]-f.indexpile[fsrows-1];
        		 
        	 } else if (sum_test==0){
        		 splitter[fsrows-1]=1;
        		 sum_test++;
        		 sum_testln+=f.indexpile[fsrows]-f.indexpile[fsrows-1];
        	 }
        	 
         }
         
         //initialise smatrix objects
         int train_rows[]= new int[sum_train+1];
         int train_cols[]= new int[sum_trainln];      
         double train_vals[]= new double[sum_trainln];   
         
         int test_rows[]= new int[sum_test+1];
         int test_cols[]= new int[sum_testln];      
         double test_vals[]= new double[sum_testln]; 
         
         int train_current_elemnt_counter=0;
         int test_current_elemnt_counter=0; 
         int train_row_holder=0;
         int test_row_holder=0;

         // initialise rest of the objects if valid
         
 		if (HasStringid) {
 			Stringidcolumn_train= new String[sum_train];
 			Stringidcolumn_test= new String[sum_test];
			}
 		if (Hasintgid) {
 			intidcolumn_train= new int[sum_train];
 			intidcolumn_test= new int[sum_test];
			}	
 		if (Hasdoubletarget) {
 			target_train= new double[sum_train];
 			target_test= new double[sum_test];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_train= new double[sum_train][];
 			target2d_test= new double[sum_test][];
			}				
 		if (HasStringtarget) {
 			targets_train= new String[sum_train];
 			targets_test= new String[sum_test];
			}				
 		if (HasStringtarget2d) {
 			target2ds_train= new String[sum_train][];
 			target2ds_test= new String[sum_test][];
			}	
 		if (Hasweights) {
 			weights_train= new double[sum_train];
 			weights_test= new double[sum_test];
			}
 		//do the allocation of objects
 		int count_train=0;
 		int count_test=0;
 		for (int i=0; i < splitter.length; i++){
 			int value=splitter[i];
 			if (value==1) { // case is allocated to test set
 		 		if (HasStringid) {

 		 			Stringidcolumn_test[count_test]=Stringidcolumn[i];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[count_test]= intidcolumn[i];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[count_test]= target[i];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[count_test]= target2d[i];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[count_test]= targets[i];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[count_test]= target2ds[i];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[count_test]= weights[i];
 					}
 		 		// the fsmatrix
 		 		
 		 		 for (int j=f.indexpile[i]; j <f.indexpile[i+1]; j++){	
 		 			test_cols[test_current_elemnt_counter]=f.mainelementpile[j];	
 		 			test_vals[test_current_elemnt_counter]=f.valuespile[j];		
 		 			test_current_elemnt_counter++;
 		 		 }
 		 		
 		 		test_row_holder++;
 		 		test_rows[test_row_holder]=test_current_elemnt_counter;
 		 		count_test++;				
 			} else { //case is allocated to training
 		 		if (HasStringid) {
 		 			Stringidcolumn_train[count_train]= Stringidcolumn[i];

 					}
 		 		if (Hasintgid) {
 		 			intidcolumn_train[count_train]= intidcolumn[i];

 					}	
 		 		if (Hasdoubletarget) {
 		 			target_train[count_train]= target[i];

 					}	
 		 		if (Hasdoubletarget2d) {
 		 			target2d_train[count_train]= target2d[i];

 					}				
 		 		if (HasStringtarget) {
 		 			targets_train[count_train]=targets[i];

 					}				
 		 		if (HasStringtarget2d) {
 		 			target2ds_train[count_train]= target2ds[i];

 					}	
 		 		if (Hasweights) {
 		 			weights_train[count_train]= weights[i];

 					}
 		 		for (int j=f.indexpile[i]; j <f.indexpile[i+1]; j++){	
		 			
		 			train_cols[train_current_elemnt_counter]=f.mainelementpile[j];	
		 			train_vals[train_current_elemnt_counter]=f.valuespile[j];		
		 			train_current_elemnt_counter++;
		 		 }

		 		train_row_holder++;
		 		train_rows[train_row_holder]=train_current_elemnt_counter;
 		 		count_train++;
 			}
 			
 			
 		}
 		//make the matrices

 			
 		smatrix_train=new smatrix(train_vals,train_cols ,train_rows, train_row_holder, f.GetColumnDimension() , train_current_elemnt_counter,true);
 		smatrix_test=new smatrix(test_vals,   test_cols ,test_rows ,  test_row_holder, f.GetColumnDimension()  ,test_current_elemnt_counter,true);
 		
        if (this.verbose){
        	System.out.println(" s matrix was splitted with " + sum_train + " rows in train and " + sum_test + " in test");
        	System.out.println(" It also has " + train_current_elemnt_counter + " elements in train and " + test_current_elemnt_counter + " in test");        	
        	System.out.println(" its counterparts can be accessed via GetsmatrixTrain/GetsmatrixTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTrain/GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTrain/GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTrain/GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTrain/Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTrain/GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTrain/GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTrain/GetWeightsTest");
        		}          		
        	}
        }
   
         
	
		//END OF METHOD
	}
	
}