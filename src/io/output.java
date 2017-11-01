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
package io;

import java.io.FileWriter;

import exceptions.DimensionMismatchException;
import matrix.fsmatrix;
import matrix.smatrix;

/**
 * 
 * @author marios
 *<p> Purpose of the package is to provide functionality for outputing files </p>
 */
public class output {

	/**
	 *  string id column
	 */
	public String Stringidcolumn [];
	/**
	*  int id column
	*/
	public int intidcolumn [];	
	/**
	 * double target variable
	 */
	public  double target [];
	 /**
	  * double target in 2d
	  */
	public  double target2d[][];
	 /**
	  * String target 
	  */
	public String targets [];
	 /**
	  * String target in 2d
	  */	 
	public String target2ds [][]; 
	/**
	 * the weights
	 */
	public double weights [];
	/**
	 *  string  column names
	 */
	public String colnames [];
	/**
	 * if true it prints stuff
	 */
	public boolean verbose =true;
	/**
	 * Delimiter to use
	 */
	public String delimeter=",";
	
	
	/**
	 * 
	 * @param f : fixed size matrix to print
	 * @param name : the name of the file
	 */
	
	public void printfsmatrix(fsmatrix f, String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.GetRowDimension();
		int fscols=f.GetColumnDimension();
        int idcolumns=0;
        int targetcolumns=0;
        int weightcolumns=0;
        
        if (fscols<=0 || fsrows<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
		if (Stringidcolumn!=null) {
			idcolumns=1;
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match  the one in fsmatrix");	        
			}
		} else if (intidcolumn!=null) {
			idcolumns=1;
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}
		}
		if (target!=null) {
			targetcolumns=1;
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}
	    }else if (target2d!=null) {
	    	targetcolumns=target2d[0].length;
		if (target2d.length!=fsrows){
			throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
		}
	    }else if (targets!=null) {
	    	targetcolumns=1;
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}
		  }else if (target2ds!=null) {
			  targetcolumns=target2ds[0].length;
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}
			  }
		if (weights!=null) {
			weightcolumns=1;
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }
          }
		
		int total_columns=fscols+idcolumns+targetcolumns+weightcolumns;
		//check columns length
		if (colnames!=null && colnames.length!=total_columns){
			throw new IllegalStateException(" Colum names do not match teh expected columns as \n"
					+ " column names are: " +colnames.length + " actual: " +total_columns +" \n" +
					" break down \n " + " id columns: " +  idcolumns + "\n target columns: " + targetcolumns 
					+ "\n weights: " + weightcolumns + "\n matrix columns: " + fscols);
			
		}
		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			
		    int current_element_count=0;
		    int row_count=0;
		    
			FileWriter writer = new FileWriter(saveFile);
			if (colnames!=null){
				if (colnames.length>=1){
					 writer.append(colnames[0]);
					 for (int i= 1; i<colnames.length;i++){
						 writer.append(this.delimeter);
						 writer.append(colnames[i]);
						
					 }
					 writer.append("\n");
				}
			}
			while (row_count<fsrows){
			int start=0;
			if (Stringidcolumn!=null) {
				 writer.append(Stringidcolumn[row_count]);
				 start=1;
				 
			} else if (intidcolumn!=null) {
				writer.append(intidcolumn[row_count] + "");

				start=1;
			}
			if (target!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(target[row_count] + "");
				} else {
					 writer.append(target[row_count] + "");
					 start=1;
				}
	
		    }else if (target2ds!=null) {
		    	
				if (start==1){
					
					for (int j=0; j <target2ds[row_count].length;j++ ){
						 writer.append(this.delimeter);
						 writer.append(target2ds[row_count][j] + "");
					}
					
				} else {
					 writer.append(target2ds[row_count][0] + "");
						for (int j=1; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] + "");
						}
						start=1;
				}
				
		    }else if (targets!=null) {
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(targets[row_count] + "");
				} else {
					 writer.append(targets[row_count] + "");
					 start=1;
				}
			  }else if (target2ds!=null) {
				  
					if (start==1){
						
						for (int j=0; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] );
						}
						
					} else {
						 writer.append(target2ds[row_count][0] );
							for (int j=1; j <target2ds[row_count].length;j++ ){
								 writer.append(this.delimeter);
								 writer.append(target2ds[row_count][j] );
							}
							start=1;
					}
				  
			  }
			if (weights!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(weights[row_count] + "");
				} else {
					 writer.append(weights[row_count] + "");
					 start=1;
				}
	          }
			// fs matrix print
			
			if (start==1){
				
				for (int j=0; j <fscols;j++ ){
					 writer.append(this.delimeter);
					 writer.append(f.data[current_element_count] + "");
					 current_element_count++;
				}
				
			} else {
				 writer.append(f.data[current_element_count] + "" );
				 current_element_count++;
					for (int j=1; j <fscols;j++ ){
						 writer.append(this.delimeter);
						 writer.append(f.data[current_element_count] + "");
						 current_element_count++;
					}
					start=1;
			}
			
			//break line
			writer.append("\n");
			//increment row
			row_count++;
   		 if (this.verbose){
			 if (row_count%(fsrows/20)==0.0) {
				 System.out.printf(" Completed: %.2f %% \n",  ((double)row_count/(double)fsrows)*100.0 );
        	 }
			 }
   		 
   		 
			}
			writer.close();
			
    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage");
    	}
		
	}
	
	/** 
	 * @param f : double 2d array to print
	 * @param name : the name of the file
	 */
	
	public void printdouble2d(double [][] f, String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.length;
		int fscols=f[0].length;
        int idcolumns=0;
        int targetcolumns=0;
        int weightcolumns=0;
        
        if (fscols<=0 || fsrows<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
		if (Stringidcolumn!=null) {
			idcolumns=1;
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match  the one in fsmatrix");	        
			}
		} else if (intidcolumn!=null) {
			idcolumns=1;
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}
		}
		if (target!=null) {
			targetcolumns=1;
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}
	    }else if (target2d!=null) {
	    	targetcolumns=target2d[0].length;
		if (target2d.length!=fsrows){
			throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
		}
	    }else if (targets!=null) {
	    	targetcolumns=1;
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}
		  }else if (target2ds!=null) {
			  targetcolumns=target2ds[0].length;
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}
			  }
		if (weights!=null) {
			weightcolumns=1;
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }
          }
		
		int total_columns=fscols+idcolumns+targetcolumns+weightcolumns;
		//check columns length
		if (colnames!=null && colnames.length!=total_columns){
			throw new IllegalStateException(" Colum names do not match teh expected columns as \n"
					+ " column names are: " +colnames.length + " actual: " +total_columns +" \n" +
					" break down \n " + " id columns: " +  idcolumns + "\n target columns: " + targetcolumns 
					+ "\n weights: " + weightcolumns + "\n matrix columns: " + fscols);
			
		}
		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			

		    int row_count=0;
		    
			FileWriter writer = new FileWriter(saveFile);
			if (colnames!=null){
				if (colnames.length>=1){
					 writer.append(colnames[0]);
					 for (int i= 1; i<colnames.length;i++){
						 writer.append(this.delimeter);
						 writer.append(colnames[i]);
						
					 }
					 writer.append("\n");
				}
			}
			while (row_count<fsrows){
			int start=0;
			if (Stringidcolumn!=null) {
				 writer.append(Stringidcolumn[row_count]);
				 start=1;
				 
			} else if (intidcolumn!=null) {
				writer.append(intidcolumn[row_count] + "");

				start=1;
			}
			if (target!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(target[row_count] + "");
				} else {
					 writer.append(target[row_count] + "");
					 start=1;
				}
	
		    }else if (target2ds!=null) {
		    	
				if (start==1){
					
					for (int j=0; j <target2ds[row_count].length;j++ ){
						 writer.append(this.delimeter);
						 writer.append(target2ds[row_count][j] + "");
					}
					
				} else {
					 writer.append(target2ds[row_count][0] + "");
						for (int j=1; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] + "");
						}
						start=1;
				}
				
		    }else if (targets!=null) {
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(targets[row_count] + "");
				} else {
					 writer.append(targets[row_count] + "");
					 start=1;
				}
			  }else if (target2ds!=null) {
				  
					if (start==1){
						
						for (int j=0; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] );
						}
						
					} else {
						 writer.append(target2ds[row_count][0] );
							for (int j=1; j <target2ds[row_count].length;j++ ){
								 writer.append(this.delimeter);
								 writer.append(target2ds[row_count][j] );
							}
							start=1;
					}
				  
			  }
			if (weights!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(weights[row_count] + "");
				} else {
					 writer.append(weights[row_count] + "");
					 start=1;
				}
	          }
			// fs matrix print
			
			if (start==1){
				
				for (int j=0; j <fscols;j++ ){
					 writer.append(this.delimeter);
					 writer.append(f[row_count][j] + "");
				}
				
			} else {
				 writer.append(f[row_count][0] + "" );
					for (int j=1; j <fscols;j++ ){
						 writer.append(this.delimeter);
						 writer.append(f[row_count][j] + "");
					}
					start=1;
			}
			
			//break line
			writer.append("\n");
			//increment row
			row_count++;
   		 if (this.verbose){
			 if (row_count%(fsrows/20)==0.0) {
				 System.out.printf(" Completed: %.2f %% \n",  ((double)row_count/(double)fsrows)*100.0 );
        	 }
			 }
   		 
   		 
			}
			writer.close();
			
    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage");
    	}
		
	}
	
	/** 
	 * @param f : String 2d array to print
	 * @param name : the name of the file
	 */
	
	public void printString2d(String [][] f, String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.length;
		int fscols=f[0].length;
        int idcolumns=0;
        int targetcolumns=0;
        int weightcolumns=0;
        
        if (fscols<=0 || fsrows<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
		if (Stringidcolumn!=null) {
			idcolumns=1;
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match  the one in fsmatrix");	        
			}
		} else if (intidcolumn!=null) {
			idcolumns=1;
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}
		}
		if (target!=null) {
			targetcolumns=1;
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}
	    }else if (target2d!=null) {
	    	targetcolumns=target2d[0].length;
		if (target2d.length!=fsrows){
			throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
		}
	    }else if (targets!=null) {
	    	targetcolumns=1;
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}
		  }else if (target2ds!=null) {
			  targetcolumns=target2ds[0].length;
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}
			  }
		if (weights!=null) {
			weightcolumns=1;
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }
          }
		
		int total_columns=fscols+idcolumns+targetcolumns+weightcolumns;
		//check columns length
		if (colnames!=null && colnames.length!=total_columns){
			throw new IllegalStateException(" Colum names do not match teh expected columns as \n"
					+ " column names are: " +colnames.length + " actual: " +total_columns +" \n" +
					" break down \n " + " id columns: " +  idcolumns + "\n target columns: " + targetcolumns 
					+ "\n weights: " + weightcolumns + "\n matrix columns: " + fscols);
			
		}
		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			

		    int row_count=0;
		    
			FileWriter writer = new FileWriter(saveFile);
			if (colnames!=null){
				if (colnames.length>=1){
					 writer.append(colnames[0]);
					 for (int i= 1; i<colnames.length;i++){
						 writer.append(this.delimeter);
						 writer.append(colnames[i]);
						
					 }
					 writer.append("\n");
				}
			}
			while (row_count<fsrows){
			int start=0;
			if (Stringidcolumn!=null) {
				 writer.append(Stringidcolumn[row_count]);
				 start=1;
				 
			} else if (intidcolumn!=null) {
				writer.append(intidcolumn[row_count] + "");

				start=1;
			}
			if (target!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(target[row_count] + "");
				} else {
					 writer.append(target[row_count] + "");
					 start=1;
				}
	
		    }else if (target2ds!=null) {
		    	
				if (start==1){
					
					for (int j=0; j <target2ds[row_count].length;j++ ){
						 writer.append(this.delimeter);
						 writer.append(target2ds[row_count][j] + "");
					}
					
				} else {
					 writer.append(target2ds[row_count][0] + "");
						for (int j=1; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] + "");
						}
						start=1;
				}
				
		    }else if (targets!=null) {
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(targets[row_count] + "");
				} else {
					 writer.append(targets[row_count] + "");
					 start=1;
				}
			  }else if (target2ds!=null) {
				  
					if (start==1){
						
						for (int j=0; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] );
						}
						
					} else {
						 writer.append(target2ds[row_count][0] );
							for (int j=1; j <target2ds[row_count].length;j++ ){
								 writer.append(this.delimeter);
								 writer.append(target2ds[row_count][j] );
							}
							start=1;
					}
				  
			  }
			if (weights!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(weights[row_count] + "");
				} else {
					 writer.append(weights[row_count] + "");
					 start=1;
				}
	          }
			// fs matrix print
			
			if (start==1){
				
				for (int j=0; j <fscols;j++ ){
					 writer.append(this.delimeter);
					 writer.append(f[row_count][j] + "");
				}
				
			} else {
				 writer.append(f[row_count][0] + "" );
					for (int j=1; j <fscols;j++ ){
						 writer.append(this.delimeter);
						 writer.append(f[row_count][j] + "");
					}
					start=1;
			}
			
			//break line
			writer.append("\n");
			//increment row
			row_count++;
   		 if (this.verbose){
			 if (row_count%(fsrows/20)==0.0) {
				 System.out.printf(" Completed: %.2f %% \n",  ((double)row_count/(double)fsrows)*100.0 );
        	 }
			 }
   		 
   		 
			}
			writer.close();
			
    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage");
    	}
		
	}	
	/** 
	 * @param f : String array to print
	 * @param name : the name of the file
	 */
	
	public void printSingleString(String []f, String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.length;
		int fscols=1;
        int idcolumns=0;
        int targetcolumns=0;
        int weightcolumns=0;
        
        if (fscols<=0 || fsrows<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
		if (Stringidcolumn!=null) {
			idcolumns=1;
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match  the one in fsmatrix");	        
			}
		} else if (intidcolumn!=null) {
			idcolumns=1;
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}
		}
		if (target!=null) {
			targetcolumns=1;
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}
	    }else if (target2d!=null) {
	    	targetcolumns=target2d[0].length;
		if (target2d.length!=fsrows){
			throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
		}
	    }else if (targets!=null) {
	    	targetcolumns=1;
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}
		  }else if (target2ds!=null) {
			  targetcolumns=target2ds[0].length;
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}
			  }
		if (weights!=null) {
			weightcolumns=1;
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }
          }
		
		int total_columns=fscols+idcolumns+targetcolumns+weightcolumns;
		//check columns length
		if (colnames!=null && colnames.length!=total_columns){
			throw new IllegalStateException(" Colum names do not match teh expected columns as \n"
					+ " column names are: " +colnames.length + " actual: " +total_columns +" \n" +
					" break down \n " + " id columns: " +  idcolumns + "\n target columns: " + targetcolumns 
					+ "\n weights: " + weightcolumns + "\n matrix columns: " + fscols);
			
		}
		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			

		    int row_count=0;
		    
			FileWriter writer = new FileWriter(saveFile);
			if (colnames!=null){
				if (colnames.length>=1){
					 writer.append(colnames[0]);
					 for (int i= 1; i<colnames.length;i++){
						 writer.append(this.delimeter);
						 writer.append(colnames[i]);
						
					 }
					 writer.append("\n");
				}
			}
			while (row_count<fsrows){
			int start=0;
			if (Stringidcolumn!=null) {
				 writer.append(Stringidcolumn[row_count]);
				 start=1;
				 
			} else if (intidcolumn!=null) {
				writer.append(intidcolumn[row_count] + "");

				start=1;
			}
			if (target!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(target[row_count] + "");
				} else {
					 writer.append(target[row_count] + "");
					 start=1;
				}
	
		    }else if (target2ds!=null) {
		    	
				if (start==1){
					
					for (int j=0; j <target2ds[row_count].length;j++ ){
						 writer.append(this.delimeter);
						 writer.append(target2ds[row_count][j] + "");
					}
					
				} else {
					 writer.append(target2ds[row_count][0] + "");
						for (int j=1; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] + "");
						}
						start=1;
				}
				
		    }else if (targets!=null) {
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(targets[row_count] + "");
				} else {
					 writer.append(targets[row_count] + "");
					 start=1;
				}
			  }else if (target2ds!=null) {
				  
					if (start==1){
						
						for (int j=0; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] );
						}
						
					} else {
						 writer.append(target2ds[row_count][0] );
							for (int j=1; j <target2ds[row_count].length;j++ ){
								 writer.append(this.delimeter);
								 writer.append(target2ds[row_count][j] );
							}
							start=1;
					}
				  
			  }
			if (weights!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(weights[row_count] + "");
				} else {
					 writer.append(weights[row_count] + "");
					 start=1;
				}
	          }
			// fs matrix print
			
			if (start==1){
				
					 writer.append(this.delimeter);
					 writer.append(f[row_count] + "");
				
				
			} else {

				writer.append(f[row_count] + "");
					
					start=1;
			}
			
			//break line
			writer.append("\n");
			//increment row
			row_count++;
   		 if (this.verbose){
			 if (row_count%(fsrows/20)==0.0) {
				 System.out.printf(" Completed: %.2f %% \n",  ((double)row_count/(double)fsrows)*100.0 );
        	 }
			 }
   		 
   		 
			}
			writer.close();
			
    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage");
    	}
		
	}	
	/** 
	 * @param f : double array to print
	 * @param name : the name of the file
	 */
	
	public void printSingledouble(double []f, String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		int fsrows=f.length;
		int fscols=1;
        int idcolumns=0;
        int targetcolumns=0;
        int weightcolumns=0;
        
        if (fscols<=0 || fsrows<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
		if (Stringidcolumn!=null) {
			idcolumns=1;
			if (Stringidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in String id does not match  the one in fsmatrix");	        
			}
		} else if (intidcolumn!=null) {
			idcolumns=1;
			if (intidcolumn.length!=fsrows){
				throw new IllegalStateException(" Rows in int id does not match  the one in fsmatrix");	        
			}
		}
		if (target!=null) {
			targetcolumns=1;
			if (target.length!=fsrows){
				throw new IllegalStateException(" Rows in target does not match  the one in fsmatrix");	        
			}
	    }else if (target2d!=null) {
	    	targetcolumns=target2d[0].length;
		if (target2d.length!=fsrows){
			throw new IllegalStateException(" Rows target 2d does not match  the one in fsmatrix");	        
		}
	    }else if (targets!=null) {
	    	targetcolumns=1;
			if (targets.length!=fsrows){
				throw new IllegalStateException(" Rows target string does not match  the one in fsmatrix");	        
			}
		  }else if (target2ds!=null) {
			  targetcolumns=target2ds[0].length;
				if (target2ds.length!=fsrows){
					throw new IllegalStateException(" Rows target string 2d does not match  the one in fsmatrix");	        
				}
			  }
		if (weights!=null) {
			weightcolumns=1;
			if (weights.length!=fsrows){
				throw new IllegalStateException(" Rows in weight array does not match  the one in fsmatrix");	        
             }
          }
		
		int total_columns=fscols+idcolumns+targetcolumns+weightcolumns;
		//check columns length
		if (colnames!=null && colnames.length!=total_columns){
			throw new IllegalStateException(" Colum names do not match teh expected columns as \n"
					+ " column names are: " +colnames.length + " actual: " +total_columns +" \n" +
					" break down \n " + " id columns: " +  idcolumns + "\n target columns: " + targetcolumns 
					+ "\n weights: " + weightcolumns + "\n matrix columns: " + fscols);
			
		}
		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			

		    int row_count=0;
		    
			FileWriter writer = new FileWriter(saveFile);
			if (colnames!=null){
				if (colnames.length>=1){
					 writer.append(colnames[0]);
					 for (int i= 1; i<colnames.length;i++){
						 writer.append(this.delimeter);
						 writer.append(colnames[i]);
						
					 }
					 writer.append("\n");
				}
			}
			while (row_count<fsrows){
			int start=0;
			if (Stringidcolumn!=null) {
				 writer.append(Stringidcolumn[row_count]);
				 start=1;
				 
			} else if (intidcolumn!=null) {
				writer.append(intidcolumn[row_count] + "");

				start=1;
			}
			if (target!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(target[row_count] + "");
				} else {
					 writer.append(target[row_count] + "");
					 start=1;
				}
	
		    }else if (target2ds!=null) {
		    	
				if (start==1){
					
					for (int j=0; j <target2ds[row_count].length;j++ ){
						 writer.append(this.delimeter);
						 writer.append(target2ds[row_count][j] + "");
					}
					
				} else {
					 writer.append(target2ds[row_count][0] + "");
						for (int j=1; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] + "");
						}
						start=1;
				}
				
		    }else if (targets!=null) {
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(targets[row_count] + "");
				} else {
					 writer.append(targets[row_count] + "");
					 start=1;
				}
			  }else if (target2ds!=null) {
				  
					if (start==1){
						
						for (int j=0; j <target2ds[row_count].length;j++ ){
							 writer.append(this.delimeter);
							 writer.append(target2ds[row_count][j] );
						}
						
					} else {
						 writer.append(target2ds[row_count][0] );
							for (int j=1; j <target2ds[row_count].length;j++ ){
								 writer.append(this.delimeter);
								 writer.append(target2ds[row_count][j] );
							}
							start=1;
					}
				  
			  }
			if (weights!=null) {
				
				if (start==1){
					 writer.append(this.delimeter);
					 writer.append(weights[row_count] + "");
				} else {
					 writer.append(weights[row_count] + "");
					 start=1;
				}
	          }
			// fs matrix print
			
			if (start==1){
				
					 writer.append(this.delimeter);
					 writer.append(f[row_count] + "");
				
				
			} else {

				writer.append(f[row_count] + "");
					
					start=1;
			}
			
			//break line
			writer.append("\n");
			//increment row
			row_count++;
   		 if (this.verbose){
			 if (row_count%(fsrows/20)==0.0) {
				 System.out.printf(" Completed: %.2f %% \n",  ((double)row_count/(double)fsrows)*100.0 );
        	 }
			 }
   		 
			}
			writer.close();
			
    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage");
    	}
	}	

	
	/**
	 * 
	 * @param f : Sparse matrix to print
	 * @param name : the name of the file
	 * @param second_delimiter : Delimiter to use to seperate columns and values
	 */
	
	public void printsmatrix(smatrix f, String name, String second_delimiter){
		
		// first check all the objects that are available such as id-columns, targets etc

        
        
        if (f.GeLength()<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
        //check if matrix sorted by row
        if(!f.IsSortedByRow()){
        	//if not sort it
        	f.convert_type();
        }

		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			


		    
			FileWriter writer = new FileWriter(saveFile);
			
			for (int i=0; i < f.GetRowDimension(); i++){
				for (int d=f.indexpile[i]; d < f.indexpile[i+1]; d++){
					int col=f.mainelementpile[d];
					double val=f.valuespile[d];
					if (d==f.indexpile[i]){
						writer.append(col + second_delimiter + val);
					} else {
						writer.append(this.delimeter + col + second_delimiter + val);
					}
					
				}
				writer.append("\n");
		   		 if (this.verbose){
					 if (i%(f.GetRowDimension()/20)==0.0) {
						 System.out.printf(" Completed: %.2f %% \n",  ((double)i/(double)f.GetRowDimension())*100.0 );
		        	 }
					 }
			}
			
			
			writer.close();

    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage");
    	}
		
	}
	
	/**
	 * 
	 * @param f : Sparse matrix to print
	 * @param name : the name of the file
	 */
	
	public void printsmatrix(smatrix f, String name){
		
		// first check all the objects that are available such as id-columns, targets etc

		 String second_delimiter=":";
        
        if (f.GeLength()<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
        //check if matrix sorted by row
        if(!f.IsSortedByRow()){
        	//if not sort it
        	f.convert_type();
        }

		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			


		    
			FileWriter writer = new FileWriter(saveFile);
			
			for (int i=0; i < f.GetRowDimension(); i++){
				for (int d=f.indexpile[i]; d < f.indexpile[i+1]; d++){
					int col=f.mainelementpile[d];
					double val=f.valuespile[d];
					if (d==f.indexpile[i]){
						writer.append(col + second_delimiter + val);
					} else {
						writer.append(" " + col + second_delimiter + val);
					}
					
				}
				writer.append("\n");
		   		 if (this.verbose){
					 if (i%(f.GetRowDimension()/20)==0.0) {
						 System.out.printf(" Completed: %.2f %% \n",  ((double)i/(double)f.GetRowDimension())*100.0 );
		        	 }
					 }
			}
			
			
			writer.close();

    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage");
    	}
		
	}
	
	/**
	 * 
	 * @param f : Sparse matrix to print
	 * @param y : a target variable
	 * @param name : the name of the file
	 */
	
	public void printsmatrix(smatrix f, double y [],String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		//System.out.println(name);
        
        String second_delimiter=":";
        if (f.GeLength()<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
        if (f.GetRowDimension()!=y.length){
        	throw new DimensionMismatchException(f.GetRowDimension(),y.length);
        }
        //check if matrix sorted by row
        if(!f.IsSortedByRow()){
        	//if not sort it
        	f.convert_type();
        }

		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			
		    
			FileWriter writer = new FileWriter(saveFile);
			
			for (int i=0; i < f.GetRowDimension(); i++){
				writer.append(y[i] + ""); 
				for (int d=f.indexpile[i]; d < f.indexpile[i+1]; d++){
					int col=f.mainelementpile[d];
					double val=f.valuespile[d];
					writer.append(" "+ col + second_delimiter + val);
				}
				writer.append("\n");
		   		 if (this.verbose){
					 if (i%(f.GetRowDimension()/20)==0.0) {
						 System.out.printf(" Completed: %.2f %% \n",  ((double)i/(double)f.GetRowDimension())*100.0 );
		        	 }
					 }
			}
			writer.close();

    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage " + e.getMessage());
    	}
	}
	
	/**
	 * 
	 * @param f : Sparse matrix to print
	 * @param y : a target variable
	 * @param fields : an int vector with size equal to the column size of f , pointing the field for each column 
	 * @param name : the name of the file
	 */
	
	public void printsmatrix(smatrix f, double y [],int fields [], String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		//System.out.println(name);
        
        String second_delimiter=":";
        if (f.GeLength()<=0){
        	throw new IllegalStateException("It seemes there is nothing to print");	        
        }
        if (f.GetRowDimension()!=y.length){
        	throw new DimensionMismatchException(f.GetRowDimension(),y.length);
        }
        if (f.GetColumnDimension()!=fields.length){
        	throw new DimensionMismatchException(f.GetColumnDimension(),fields.length);
        }       
        //check if matrix sorted by row
        if(!f.IsSortedByRow()){
        	//if not sort it
        	f.convert_type();
        }

		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			
		    
			FileWriter writer = new FileWriter(saveFile);
			
			for (int i=0; i < f.GetRowDimension(); i++){
				writer.append(y[i] + ""); 
				for (int d=f.indexpile[i]; d < f.indexpile[i+1]; d++){
					int col=f.mainelementpile[d];
					int field=fields[col];
					double val=f.valuespile[d];
					writer.append(" "+ field + second_delimiter+ col + second_delimiter + val);
					
					
				}
				writer.append("\n");
		   		 if (this.verbose){
					 if (i%(f.GetRowDimension()/20)==0.0) {
						 System.out.printf(" Completed: %.2f %% \n",  ((double)i/(double)f.GetRowDimension())*100.0 );
		        	 }
					 }
			}
			
			
			writer.close();

    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage " + e.getMessage());
    	}
		
	}
	
	/**
	 * 
	 * @param f : Sparse matrix to print in vowpal wabbit format
	 * @param y : a target variable
	 * @param name : the name of the file
	 */
	
	public void printsmatrixvw(smatrix f, double y [],String name){
		
		// first check all the objects that are available such as id-columns, targets etc
		//System.out.println(name);
        
        String second_delimiter=":";
        if (f.GeLength()<=0){
        	throw new IllegalStateException(" It seemes there is nothing to print");	        
        }
        if (f.GetRowDimension()!=y.length){
        	throw new DimensionMismatchException(f.GetRowDimension(),y.length);
        }
        //check if matrix sorted by row
        if(!f.IsSortedByRow()){
        	//if not sort it
        	f.convert_type();
        }

		// printing time!
		
		try{  // Catch errors in I/O if necessary.
		  // Open a file to write to.
			String saveFile = name;
			
		    
			FileWriter writer = new FileWriter(saveFile);
			
			for (int i=0; i < f.GetRowDimension(); i++){
				writer.append(y[i] + " |f"); 
				for (int d=f.indexpile[i]; d < f.indexpile[i+1]; d++){
					int col=f.mainelementpile[d];
					double val=f.valuespile[d];
					writer.append(" "+ col + second_delimiter + val);
					
					
				}
				writer.append("\n");
		   		 if (this.verbose){
					 if (i%(f.GetRowDimension()/20)==0.0) {
						 System.out.printf(" Completed: %.2f %% \n",  ((double)i/(double)f.GetRowDimension())*100.0 );
		        	 }
					 }
			}
			
			
			writer.close();

    	} catch (Exception e) {
    		throw new IllegalStateException(" failed to write at the writting stage " + e.getMessage());
    	}
		
	}	
}
