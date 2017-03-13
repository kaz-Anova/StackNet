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
 *<p> Purpose of the package is to provide functionality for Boostsrapping the data , 
 * That is to create data sets on the selected main data method with replacement </p>
 */
public class boostsrapping {
	/**
	 *  string id column
	 */
	public String Stringidcolumn [];
	/**
	 *  string id column for test set
	 */
	private String Stringidcolumn_test [];
	/**
	*  int id column
	*/
	public int intidcolumn [];	
	/**
	*  int id column for test
	*/
	private int intidcolumn_test [];	
	/**
	 * double target variable
	 */
	public  double target [];
	/**
	 * double target variable for test
	 */
	private  double target_test [];
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
	  * String target for test 
	  */
	private String targets_test [];
	 /**
	  * String target in 2d
	  */	 
	public String target2ds [][]; 
	 /**
	  * String target in 2d for test 
	  */	 
	private String target2ds_test [][]; 
	/**
	 * the weights
	 */
	public double weights [];
	/**
	 * the weights for test 
	 */
	private double weights_test [];
	/**
	 * the fsmatrix for test 
	 */
	private fsmatrix fsmatrix_test ;
	/**
	 * the smatrix for test 
	 */
	private smatrix smatrix_test ;
	 /**
	  * double data for test
	  */
	private  double data_test[][];
	 /**
	  * String data for test
	  */
	private  String datas_test[][];
	/**
	 * seed to replicate randomized procedures.
	 */
	public long seed=10;
	/**
	 * if true it prints stuff
	 */
	public boolean verbose =false;
	/**
	 * sample size to boostsrap
	 */
	public int sample_size=-1;
	//Getter Methods for the  private objects
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
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.GetRowDimension();
		int fscols=f.GetColumnDimension();
        if (fscols<=0 || fsrows<1){
        	throw new IllegalStateException(" It seems there is no pointing boostsrapping given the minimal size of the matrix");	        
        }
		//check the sample size- if negative or zero, it becomes the same as the data's rows
        if (this.sample_size<=0){
        	this.sample_size=fsrows;
        }
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

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
         //initialise fs matrix object
         fsmatrix_test = new fsmatrix(new double [this.sample_size * fscols], this.sample_size, fscols);                
         // initialise rest of the objects if valid         
 		if (HasStringid) {
 			Stringidcolumn_test= new String[this.sample_size];
			}
 		if (Hasintgid) {
 			intidcolumn_test= new int[this.sample_size];
			}	
 		if (Hasdoubletarget) {
 			target_test= new double[this.sample_size];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_test= new double[this.sample_size][];
			}				
 		if (HasStringtarget) {
 			targets_test= new String[this.sample_size];
			}				
 		if (HasStringtarget2d) {
 			target2ds_test= new String[this.sample_size][];
			}	
 		if (Hasweights) {
 			weights_test= new double[this.sample_size];
			}
 		//do the allocation of objects
 		for (int i=0; i < this.sample_size; i++){
 			int value=rand.nextInt(fsrows);
 		 		if (HasStringid) {
 		 			Stringidcolumn_test[i]=Stringidcolumn[value];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[i]= intidcolumn[value];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[i]= target[value];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[i]= target2d[value];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[i]= targets[value];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[i]= target2ds[value];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[i]= weights[value];
 					}
 		 		// the fsmatrix
 		 		 for (int j=0; j < fscols; j++){
 		 			fsmatrix_test.SetElement(i, j, f.GetElement(value, j));
 		 		 }
 		}
        if (this.verbose){
        	System.out.println(" fs matrix was boostsrapped with " + this.sample_size + " cases");
        	System.out.println(" its counterparts can be accessed via GetfsmatrixTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTest");
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
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=data.length;
		int fscols=data[0].length;
        if (fscols<=0 || fsrows<1){
        	throw new IllegalStateException(" It seems there is no pointing boostsrapping given the minimal size of the matrix");	        
        }
		//check the sample size- if negative or zero, it becomes the same as the data's rows
        if (this.sample_size<=0){
        	this.sample_size=fsrows;
        }
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

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
         //initialise fs matrix object
		 data_test = new double[this.sample_size][];                
         // initialise rest of the objects if valid         
 		if (HasStringid) {
 			Stringidcolumn_test= new String[this.sample_size];
			}
 		if (Hasintgid) {
 			intidcolumn_test= new int[this.sample_size];
			}	
 		if (Hasdoubletarget) {
 			target_test= new double[this.sample_size];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_test= new double[this.sample_size][];
			}				
 		if (HasStringtarget) {
 			targets_test= new String[this.sample_size];
			}				
 		if (HasStringtarget2d) {
 			target2ds_test= new String[this.sample_size][];
			}	
 		if (Hasweights) {
 			weights_test= new double[this.sample_size];
			}
 		//do the allocation of objects
 		for (int i=0; i < this.sample_size; i++){
 			int value=rand.nextInt(fsrows);
 		 		if (HasStringid) {
 		 			Stringidcolumn_test[i]=Stringidcolumn[value];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[i]= intidcolumn[value];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[i]= target[value];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[i]= target2d[value];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[i]= targets[value];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[i]= target2ds[value];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[i]= weights[value];
 					}
 		 		// the fsmatrix
 		 		data_test[i]=data[value];
 
 		}
        if (this.verbose){
        	System.out.println(" fs matrix was boostsrapped with " + this.sample_size + " cases");
        	System.out.println(" its counterparts can be accessed via GetDataTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTest");
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
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=data.length;
		int fscols=data[0].length;
        if (fscols<=0 || fsrows<1){
        	throw new IllegalStateException(" It seems there is no pointing boostsrapping given the minimal size of the matrix");	        
        }
		//check the sample size- if negative or zero, it becomes the same as the data's rows
        if (this.sample_size<=0){
        	this.sample_size=fsrows;
        }
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

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
         //initialise fs matrix object
		 datas_test = new String[this.sample_size][];                
         // initialise rest of the objects if valid         
 		if (HasStringid) {
 			Stringidcolumn_test= new String[this.sample_size];
			}
 		if (Hasintgid) {
 			intidcolumn_test= new int[this.sample_size];
			}	
 		if (Hasdoubletarget) {
 			target_test= new double[this.sample_size];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_test= new double[this.sample_size][];
			}				
 		if (HasStringtarget) {
 			targets_test= new String[this.sample_size];
			}				
 		if (HasStringtarget2d) {
 			target2ds_test= new String[this.sample_size][];
			}	
 		if (Hasweights) {
 			weights_test= new double[this.sample_size];
			}
 		//do the allocation of objects
 		for (int i=0; i < this.sample_size; i++){
 			int value=rand.nextInt(fsrows);
 		 		if (HasStringid) {
 		 			Stringidcolumn_test[i]=Stringidcolumn[value];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[i]= intidcolumn[value];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[i]= target[value];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[i]= target2d[value];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[i]= targets[value];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[i]= target2ds[value];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[i]= weights[value];
 					}
 		 		// the fsmatrix
 		 		datas_test[i]=data[value];
 
 		}
        if (this.verbose){
        	System.out.println(" fs matrix was boostsrapped with " + this.sample_size + " cases");
        	System.out.println(" its counterparts can be accessed via GetStringDataTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTest");
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
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.GetRowDimension();
		int fscols=f.GetColumnDimension();
		
        if (fscols<=0 || fsrows<1){
        	throw new IllegalStateException(" It seems there is no pointing boostsrapping given the minimal size of the matrix");	        
        }
		//check the sample size- if negative or zero, it becomes the same as the data's rows
        if (this.sample_size<=0){
        	this.sample_size=fsrows;
        }
		//boolean methods to clarify the objects that are not null (and therefore can be slitted) 
		boolean HasStringid=false;
		boolean Hasintgid=false;
		boolean Hasdoubletarget=false;
		boolean HasStringtarget=false;
		boolean Hasdoubletarget2d=false;
		boolean HasStringtarget2d=false;		
		boolean Hasweights=false;

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
		  // get row elements
	    if (!f.IsSortedByRow()){
	    	f.convert_type();
	    }
		
         //Initialize a generator
		 Random rand = new Random();
		 rand.setSeed(seed); // set the seed 
		 // determine number of cases to boostsrapped

         int sum_testln=0;  
         int splitter []= new int[this.sample_size];

         for (int i=0; i < this.sample_size ; i++){
        	 splitter[i]=rand.nextInt(fsrows);
        	 sum_testln+= f.indexpile[splitter[i]+1]- f.indexpile[splitter[i]];
        	 }
  
         
         int test_rows[]= new int[ this.sample_size+1];
         int test_cols[]= new int[sum_testln];      
         double test_vals[]= new double[sum_testln]; 
         

         int test_current_elemnt_counter=0; 

         // initialise rest of the objects if valid
         
 		if (HasStringid) {
 			Stringidcolumn_test= new String[this.sample_size];
			}
 		if (Hasintgid) {
 			intidcolumn_test= new int[this.sample_size];
			}	
 		if (Hasdoubletarget) {
 			target_test= new double[this.sample_size];
			}	
 		if (Hasdoubletarget2d) {
 			target2d_test= new double[this.sample_size][];
			}				
 		if (HasStringtarget) {
 			targets_test= new String[this.sample_size];
			}				
 		if (HasStringtarget2d) {
 			target2ds_test= new String[this.sample_size][];
			}	
 		if (Hasweights) {
 			weights_test= new double[this.sample_size];
			}
 		//do the allocation of objects
 		int count_test=0;	
 		for (int i=0; i < splitter.length; i++){
 			int value=splitter[i];
 		 		if (HasStringid) {

 		 			Stringidcolumn_test[count_test]=Stringidcolumn[value];
 					}
 		 		if (Hasintgid) {

 		 			intidcolumn_test[count_test]= intidcolumn[value];
 					}	
 		 		if (Hasdoubletarget) {

 		 			target_test[count_test]= target[value];
 					}	
 		 		if (Hasdoubletarget2d) {

 		 			target2d_test[count_test]= target2d[value];
 					}				
 		 		if (HasStringtarget) {
 	
 		 			targets_test[count_test]= targets[value];
 					}				
 		 		if (HasStringtarget2d) {

 		 			target2ds_test[count_test]= target2ds[value];
 					}	
 		 		if (Hasweights) {

 		 			weights_test[count_test]= weights[value];
 					}
 		 		// the fsmatrix
 		 		 for (int j=f.indexpile[value]; j <f.indexpile[value+1]; j++){
 		 			
 		 			test_cols[test_current_elemnt_counter]=f.mainelementpile[j];	
 		 			test_vals[test_current_elemnt_counter]=f.valuespile[j];		
 		 			test_current_elemnt_counter++;
 		 		 }
 		 		
 		 		count_test++;				
 		 		test_rows[count_test]= test_rows[test_current_elemnt_counter]=count_test;
 			
 			
 		}
 		//make the matrices
 		smatrix_test=new smatrix(test_vals,test_cols ,test_rows, count_test, f.GetColumnDimension() , test_current_elemnt_counter,true);


 		
        if (this.verbose){
        	System.out.println(" sparse matrix was boostsrapped with sample size:  " + this.sample_size);
        	System.out.println(" It also has " + test_current_elemnt_counter + " elements ");        	
        	System.out.println(" its counterparts can be accessed via GetsmatrixTest");
        	if (HasStringid || Hasintgid || Hasdoubletarget || Hasdoubletarget2d || HasStringtarget ||  HasStringtarget2d || Hasweights){
        		System.out.println("Additionally the following were splitted: ");
        		if (HasStringid){
        			System.out.println("A string Id that its counterparts can be accessed via GetsStringIdTest");
        		}
        		if (Hasintgid){
        			System.out.println("An int Id that its counterparts can be accessed via GetsintIdTest");
        		}  
        		if (Hasdoubletarget){
        			System.out.println("A double target 1D that its counterparts can be accessed via GettargetTest");
        		}  
        		if (Hasdoubletarget2d){
        			System.out.println("A double target that its counterparts can be accessed via Gettarget2DTest");
        		}     
        		if (HasStringtarget){
        			System.out.println("A String target 1D that its counterparts can be accessed via GettargetStringTest");
        		}  
        		if (HasStringtarget2d){
        			System.out.println("A String target 2D that its counterparts can be accessed via GettargetString2DTest");
        		}  
        		if (Hasweights){
        			System.out.println("A double weighst 1D that its counterparts can be accessed via GetWeightsTest");
        		}          		
        	}
        }
   
        splitter=null;
	
		//END OF METHOD
	}
	
}