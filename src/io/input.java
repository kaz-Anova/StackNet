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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

import exceptions.NullObjectException;
import matrix.fsmatrix;
import matrix.smatrix;


/**
 * 
 * @author marios
 * <p> load txt and similar files in delimited format using a number of parameters that due to their significant number remain static (and publicly available) to be 
 * manipulated at will. 
 * Those parameters are:
 * <ol>
 * <li><b>delimiter</b>: the delimiter to use, default is "," </li>
 * <li><b>columns_to_import</b>: the non target columns to import  </li>
 * <li><b>targets_columns</b>: the target columns to import  </li>
 * <li><b>wcolumn</b>: the weight variable location </li>
 * <li><b>start</b>: the start location of the variables to import (column location) </li>
 * <li><b>end</b>: the end location of the variables to import (column location) </li>
 * <li><b>HasHeader</b>: True if the file has labels </li>
 * <li><b>skiprows</b>: Number of rows to skip from the beginning (excluding headers) </li>
 * <li><b>to_replace_null</b>: Replace Null values with this </li>
 * <li><b>replaceable_values</b>: Values to be considered nulls </li> 
 * <li><b>idint</b>:  column to consider as id in int [] format </li> 
 * <li><b>idstring</b>: Int column to consider as id in string [] format </li> 
 * <li><b>charset</b>: String Charset to use (default at UTF-8) </li> 
 * 
 * </ol> 
 */
public class input {
	
	/**
	 * Variable to hold 2-dimensional target variable
	 */
	private  double target2d [][];
	/**
	 * Variable to hold 1 dimensional target variable
	 */
	private  double target [];	
	/**
	 * Variable to hold 1 dimensional weights variable
	 */
	private  double weights [];	
	/**
	 * Variable to hold 2-dimensional target variable
	 */
	private  String target2dstring [][];
	/**
	 * Variable to hold 1 dimensional target variable
	 */
	private  String targetstring [];		
	/**
	 * a fixed size fmatrix
	 */
	private  fsmatrix fm;
	/**
	 * a Sparse smatrix
	 */
	private  smatrix sm;
	/**
	 * 2-dimensional double data
	 */
	private  double data[][] ;
	/**
	 * 2-dimensional String data
	 */
	private  String dataString[][] ;
	/**
	 * 1 dimensional double data of 1 column
	 */
	private double single_column[] ;
	/**
	 * Delimiter to use
	 */
	public String delimeter=" ";
	/**
	 * columns to import for the main data files
	 */
	public int [] columns_to_import ={};
	/**
	 * targets' columns
	 */
	public int [] targets_columns = {};
	/**
	 * weight column
	 */
	public int wcolumn=-1;
	/**
	 * the start location of the import for the main data
	 */
	public int start=-1;
	/**
	 * the end location of the import for the main data
	 */
	public int end=-1;	
	/**
	 * true if there are headers in the file
	 */
	public boolean HasHeader=true;
	/**
	 * skip x rows
	 */
	public  int skiprows=0;
	/**
	 * to replace null values with
	 */
	public double to_replace_null=0.0;
	/**
	 * what can be considered a null or a replaceable value with "to_replace_null"
	 */
	public String replaceable_values[]={""};
	/**
	 * Charset to use
	 */
	public static String charset="UTF-8";
	/**
	 * columns that are numbers based on the getfileinfo method
	 */
	private int [] numericcols;
	/**
	 * columns that are NOT numbers based on the getfileinfo method
	 */
	private int [] nonnumericcols;	
	/**
	 *  string id column
	 */
	 String Stringidcolumn [];
	/**
	*  int id column
	*/
	int intidcolumn [];	
	/**
	 * int column to represent the id int
	 */
	public int idint=-1;
	/**
	 * String column to represent the id as string
	 */
	public int idstring=-1;	
	/**
	 * if true it prints stuff
	 */
	public boolean verbose =true;
	
	/**
	 * column names
	 */
	private String coulmnames [];
	
	/**
	 * 
	 * @return int [] Index array representing the id
	 */
	public int [] GetIntid(){
		
		if (intidcolumn==null){
			throw new NullObjectException(" int Id array is null");
		}
		return intidcolumn;
	}
	
	/**
	 * 
	 * @return String [] Index array representing the id
	 */
	public String [] GetStringid(){
		
		if (Stringidcolumn==null){
			throw new NullObjectException(" String Id array is null");
		}
		return Stringidcolumn;
	}	
	
	/**
	 * 
	 * @return the file's column names
	 */
	
	public String [] GetColumNames(){
		if (coulmnames==null){
			throw new NullObjectException(" Column Names are null");
		}
		return coulmnames;
	}		
	/**
	 * 
	 * @return int [] columns that are numbers based on the getfileinfo method
	 */
	public int [] GetTnumericcols(){
		if (numericcols==null){
			throw new NullObjectException(" numerical columns are null, you need to run the the getfile info method first");
		}
		return numericcols;
	}	
	/**
	 * 
	 * @return int [] columns that are numbers based on the getfileinfo method
	 */
	public int [] GetTnonnumericcols(){
		if (nonnumericcols==null){
			throw new NullObjectException(" nonnumerical columns are null, you need to run the the getfile info method first");
		}
		return nonnumericcols;
	}		
	
	/**
	 * 
	 * @return 2d target data (for e.g. multl-label problems)
	 */
	public double [][] GetTarget2D(){
		if (target2d==null){
			throw new NullObjectException(" double 2d target data is null");
		}
		return target2d;
	}
	/**
	 * 
	 * @return  target data (for e.g. single-label problems)
	 */
	
	public double [] GetTarget(){
		if (target==null){
			throw new NullObjectException(" double target data is null");
		}
		return target;
	}
	/**
	 * 
	 * @return 2d target data (for e.g. multl-label problems) as String
	 */
	public String [][] GetTarget2DString(){
		if (target2dstring==null){
			throw new NullObjectException(" double 2d string data is null");
		}
		return target2dstring;
	}
	/**
	 * 
	 * @return  target data (for e.g. single-label problems)
	 */
	
	public String [] GetTargetString(){
		if (targetstring==null){
			throw new NullObjectException(" String target data is null");
		}
		return targetstring;
	}	
	/**
	 * 
	 * @return  weight data (for e.g. per sample normally)
	 */	
	public double [] GetWeights(){
		if (weights==null){
			throw new NullObjectException(" double weights' data is null");
		}
		return weights;	
	}
	/**
	 * 
	 * @return fsmatrix (Fixed-Size Matrix)
	 */
	public fsmatrix  Getfsmatrix(){
		if (fm==null){
			throw new NullObjectException(" fsmatrix is null");
		}
		return fm;
	}	
	/**
	 * 
	 * @return smatrix (Fixed-Size Matrix)
	 */
	public smatrix  Getsmatrix(){
		if (sm==null){
			throw new NullObjectException(" smatrix is null");
		}
		return sm;
	}
	/**
	 * 
	 * @return double data 
	 */
	public double [][]  GetData(){
		if (data==null){
			throw new NullObjectException(" 2d data  is null");
		}
		return data;
	}
	/**
	 * 
	 * @return String data 
	 */
	public String [][]  GetDataString(){
		if (dataString==null){
			throw new NullObjectException(" 2d data String");
		}
		return dataString;
	}	
	/**
	 * 
	 * @return single column
	 */
	public double []  GetSingleColumn(){
		if (single_column==null){
			throw new NullObjectException(" single column instance is null");
		}
		return single_column;	
	}	

	/**
	 * 
	 * @param file : file to read
	 * @param hasheaders : True if it has headers
	 * @return The number of rows in the file
	 * <p> Fast and efficient way to get the number of counts
	 */
	public static int GetRowCount(String file, boolean hasheaders){
		int lines = 0;
		try{
			
		FileInputStream fis = new FileInputStream(file);
		BufferedReader reader = new BufferedReader(new InputStreamReader(fis,charset));
		
		if (hasheaders){
			reader.readLine();
		}
		while (reader.readLine() != null) {lines++;};
		reader.close();
		} catch (Exception e){
			System.out.println(e.getMessage());
			//System.out.println("ha");
			return -1;}
		return lines;	
	}
	/**
	 * 
	 * @param file : file to read
	 * @param delimeter : delimiter to use
	 * @return The number of columns in the file
	 * <p> Fast and efficient way to get the number of columns
	 */
	public static int GetColumnCount(String file, String delimeter){
		int columns = 0;
		try{
			FileInputStream fis = new FileInputStream(file);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis,charset));
            columns = br.readLine().split(delimeter +"+",-1).length;
            br.close();
		} catch (Exception e){return -1;}
		return columns;
	}	
	
	
	/**
    * @param n : The name of file to 'Open'.
	* @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
	* @param xfirstlines : How many lines to read prior making judgements about each variable.
	* @param hasheaders : True if we want the first line to be regarded as header
	* @param hasconsecutivedel : Regards consecutive delimiters as one if true.
	* <p> This method reads a file and print various information about the columns
	*/

			public  void getfileinfo (String n, String delimeter, int xfirstlines, boolean hasheaders, boolean hasconsecutivedel) {

	            File x= new File(n);
	                String line;
	            double line_count=0;
	            int colcount=0;
	            int columncount=0;
	            int rowcount=input.GetRowCount(n,hasheaders);           
	            Vector<String>   columns = new Vector<String>();
	            try {
	                    FileInputStream fis = new FileInputStream(x);
	                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,input.charset));
	                   // Map the first Row
	                    if (hasheaders==true){
	                        if (hasconsecutivedel){
	                    coulmnames = br.readLine().split(delimeter +"+",-1);
	                        } else{
	                                     coulmnames = br.readLine().split(delimeter,-1);
	                        }
	                    } else{
	                        coulmnames= new String[columns.size()];
	                        for (int i=0; i<coulmnames.length;i++ ){
	                                    coulmnames[i]=i + "";
	                        }
	                        fis = new FileInputStream(x);
	                        br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
	                    }

	                    colcount=coulmnames.length;
	                    columncount=colcount;
	                    dataString = new String [rowcount][columncount]  ;

	                    int ro=0;

	                    while ((line = br.readLine()) != null && line_count<xfirstlines ) {
	                            String[] tokens = line.split(delimeter,-1);
	                            for  (int i =0 ; i<coulmnames.length; i ++ ) {
	                                    try{
	                                    if (tokens[i].equals("")){
	                                                tokens[i]=Double.NaN + "";
	                                    }
	                                    dataString[ro][i]=tokens[i];
	                            } catch (Exception e){
	                            	   dataString[ro][i]=Double.NaN + "";}
	                            }
	                            ro++;  }

	              //Close the buffer reader
	                    br.close();
	            } catch (Exception e) {
	                    e.printStackTrace();
	            }
	             System.out.println(" The file" + x + " was loaded successfully with :");
	             System.out.println(" Rows read: " + rowcount);
	             System.out.println(" Columns included? : " + hasheaders);
	             System.out.println(" Columns : " + columncount);
	             System.out.println(" Delimeter was  : " + delimeter);
	             ArrayList<Integer> num= new  ArrayList<Integer>();
	             ArrayList<Integer> nonnum= new  ArrayList<Integer>();
	             for (int j=0; j < coulmnames.length; j++){
	                double max=Double.MIN_VALUE;
	                double min=Double.MAX_VALUE;
	                int countnulls=0;
	                String maxs=dataString[0][j];
	                String mins=dataString[0][j];
	                boolean isdouble =true;
	                for (int i=0; i < dataString.length; i++){
	                          try{
									Double.parseDouble(dataString[i][j]);
	                          }catch (Exception e){
	                                      isdouble=false;
	                                      break;
	                          }
	              }
	              if (isdouble==true){
	                          num.add(j);
	              for (int i=0; i < dataString.length; i++){
	                          Double v= Double.parseDouble(dataString[i][j]);
	                          if (v >max){
	                                      max=v; 
	                          }
	                          if (v <min){
	                                    min=v; 
	                          }
	                          if ( Double.isNaN(v)){
	                                      countnulls++; 
	                          }
	              }       
	             System.out.println("Variable: " + coulmnames[j] + " Type: Numeric Max: " + max + " Min: " + min+ " Nulls " + countnulls);
	              }
	              else {
	                          nonnum.add(j);
	                          for (int i=0; i < dataString.length; i++){
	                                      String v= dataString[i][j];
	                                      if (v.compareTo(maxs)==1){
	                                                  maxs=v; 
	                                      }
	                                      if (v.compareTo(mins)==-1){
	                                                  mins=v; 
	                                      }
	                                      if (v.equals(Double.NaN+"")){
	                                                  countnulls++; 
	                                      } 
	              }
	                          System.out.println("Variable: " + coulmnames[j] + " Type: Letter Max: " + maxs + " Min: " + mins+ " Nulls " + countnulls);
	          }
	          }

	          numericcols=  new int [num.size()]  ;
	          nonnumericcols=  new int [nonnum.size()]  ;
	          for (int i=0; i < numericcols.length; i++){
	              numericcols[i]=num.get(i);
	          }
	          for (int i=0; i < nonnumericcols.length; i++){
	              nonnumericcols[i]=nonnum.get(i);
	          }
	          System.out.println("Numeric columns: ") ;
	          System.out.println(Arrays.toString(numericcols));
	          System.out.println("Letter columns: ") ;
	          System.out.println(Arrays.toString(nonnumericcols));
	          System.out.println("The type of columns can be retrieved through GetTnumericcols() and GetTnonnumericcols() methods ");

	        }
	        
	        /**
	         * 
	         * @param file : file to read (full path)
	         * @return : an fsmatrix.
	         * <p> Method to read a fixed size matrix. All rows in the file are assumed to have the same size as the first one.
	         */
	        
	        public fsmatrix Readfmatrix(String file){
	        	
	        	 
	        	// get the number of rows
	        	int n_rows=input.GetRowCount(file,this.HasHeader);
	        	if (n_rows<=0){
	        		throw new IllegalStateException(" File " + file + " appears to be empty ");
	        	}
	        	// get the number of rows
	        	int n_columns=input.GetColumnCount(file, this.delimeter);
	        	// check number of columns
	        	if (n_columns<=0){
	        		throw new IllegalStateException(" File " + file + " has no columns with current delimeter  ");
	        	}
	          	//fix end if it exceeds current columns
	          	if ( end>n_columns ){
	          		end=n_columns;
	          	}
	        	if ( start<0 && end <0 ){
	          		start=-2;
	          		end=-1;
	          	}
	           //throw error if start exceeds number of cilumns
	          	if ( start>n_columns-1 ){
	        		throw new IllegalStateException(" start cannot be larger than the current number of columns  ");	          		
	          	}
	          	if ( start>=end ){
	        		throw new IllegalStateException(" start cannot be larger/equal than end  ");	          		
	          	}
	          	//get the actual number of rows to import
	          	int number_of_rows=0;
	          	if (skiprows<=0){
	          		number_of_rows=n_rows;
	          		skiprows=0;
	          	} else if (skiprows>number_of_rows){
	          		skiprows=0;
	          		throw new IllegalStateException(" Number of skipped rows cannot be larger than the number of rows in the file  ");	  
	          	} else {
	          		number_of_rows=n_rows-skiprows;
	          	}
	          	
	        	//determine the number of valid target columns
	        	int number_ofvalid_target_columns=0;
	        	for (int i=0; i <targets_columns.length;i++ ){
	        		if (targets_columns[i]>=0 && targets_columns[i]<=n_columns-1 ){
	        			number_ofvalid_target_columns++;
	        		}
	        	}
	        	int n=0;
	        	int new_target_cols[]= new int [number_ofvalid_target_columns];
	        	for (int i=0; i <targets_columns.length;i++ ){
	        		if (targets_columns[i]>=0 && targets_columns[i]<=n_columns-1 ){
	        			new_target_cols[n]=targets_columns[i];
	        			n++;
	        		}
	        	}
	        	
	        	targets_columns=new_target_cols; // assign the correct columns
	        	
	        	int count_ofweightcolumn=0;
	        	//n_columns-=targets_columns.length; // check if the weight variable is valie
	        	if (  wcolumn>=0  && wcolumn <=n_columns-1){
	        		count_ofweightcolumn=1;
	        	}else {
	        		wcolumn=-1;
	        	}
	        	// check int and string columns
	        	int count_ofstringcol=0;
	        	int count_ofintcol=0;
	        	int total_number_of_id_cols=0;
	        
	          	if (  idint>=0  && idint <=n_columns-1){
	          		count_ofintcol =1;
	          		total_number_of_id_cols=1;

	        	}else {
	        		idint=-1;
	        	}
	          	if (  idstring>=0  && idstring <=n_columns-1){
	          		count_ofstringcol=1;
	          		total_number_of_id_cols=1;
	        	}	else {
	        		idstring=-1;
	        	}

	          	// check main data columns

	        	if (start<0 && end<=0 && columns_to_import.length<=0){
	        		
	        		// load all the columns that do not fall in all other categories
	        		n=0;
	        		columns_to_import= new int [n_columns-total_number_of_id_cols-count_ofweightcolumn-number_ofvalid_target_columns] ;
	        		for (int i=0; i <n_columns; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				columns_to_import[n]=i;
	        				n++;
	        			}
	        		}
	        		
	        	} else if(start<0 &&  end>0 && columns_to_import.length<=0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=0; i <end; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				temp_list_of_valid_names.add(i);
	        			}
	        		}	   
	        		
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	        		
	        	}else if(start>=0 && end<0 && columns_to_import.length<=0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=start; i <n_columns; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				temp_list_of_valid_names.add(i);
	        				
	        			}
	        			
	        		}	   
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	     	
	        	} else if(start>=0 && end>0 && columns_to_import.length<=0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=start; i <end; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				temp_list_of_valid_names.add(i);
	        			}
	        			
	        		}	   
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	     	// check valid columns for columns_to_import
	        	}else if(columns_to_import.length>0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=0; i <columns_to_import.length; i++){
	        			if (columns_to_import[i]>=0 && 
	        					columns_to_import[i]< n_columns &&
	        					exists(columns_to_import[i],targets_columns)==false &&
	        					wcolumn!=columns_to_import[i] && 
	        					idint!=columns_to_import[i] &&
	        					idstring!=columns_to_import[i]){
	        				temp_list_of_valid_names.add(columns_to_import[i]);
	        			}
	        			
	        		}	   
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	     	
	        	} else {
	        		
	        		throw new IllegalStateException(" No correct combination of columns to import were determined  ");
	        	}
	        	
	        	if (columns_to_import.length<=0){
	        		
	        		throw new IllegalStateException(" No columns to import were determined for fmatrix ");
	        		
	        	}
	        		     
	        	// ---------Start initialising objects-----------//
	        	
	        	// target columns
	        	if (number_ofvalid_target_columns>0){
	        		
	        		if (number_ofvalid_target_columns==1){
	        			target= new double [number_of_rows];
	        		} else {
	        			target2d=new double [number_of_rows][number_ofvalid_target_columns];
	        		}	
	        		
	        	}
	        	// ids columns	
	        	//int
	        	if (count_ofintcol>0){
	        		intidcolumn=new int [number_of_rows];
	          		
	        	}
	        	//String
	        	if (count_ofstringcol>0){
	        		Stringidcolumn=new String [number_of_rows];
	        	}	        			
	        	// weights columns
	        	if (count_ofweightcolumn>0){
	        		weights=new double [number_of_rows];
	        	}
	        	
	        	
	        	// Main data file and columnames
	        	double data_toimport []= new double [number_of_rows*columns_to_import.length];
	        	coulmnames= new String [columns_to_import.length];
	        	int keep_current_row_count=0;
	        	
	        	// ---------Start Reading the file -----------//	 
	        	try{
                FileInputStream fis = new FileInputStream(file);
                @SuppressWarnings("resource")
				BufferedReader br = new BufferedReader(new InputStreamReader(fis,input.charset));
                //The string object to hold the line
                String line="";
                // check for headers
               
                if (this.HasHeader){
                	
                	//assign the headers
                	String colvalues [] = br.readLine().split(delimeter +"+",-1);
                	//import the columns
                	for (int j=0; j <columns_to_import.length; j++ ){
                		coulmnames[j]=colvalues[columns_to_import[j]];
                	}
                }
               
	        	// Skip the rows specified
	        	int number_of_rows_passed=0;
	        	while  (number_of_rows_passed < this.skiprows) {
	        		br.readLine();
	        		number_of_rows_passed++;
	        		
	        	}
	       	
               // Start importing data
	        	 while ((line = br.readLine()) != null  ) {
	        		 String values []=line.split(this.delimeter,-1);
	        		 //System.out.println(values.length);
	 	        	// ---------Start updating the  objects-----------//
	 	        	
	 	        	// target columns
	 	        	if (number_ofvalid_target_columns>0){
	 	        		
	 	        		if (number_ofvalid_target_columns==1){
	 	        			
	 	        			double cvalue=this.to_replace_null;
	 	        			try {
	 	        				cvalue=Double.parseDouble(values[targets_columns[0]]);
	 	        			} catch(Exception e){
	 	        				cvalue=this.to_replace_null;
	 	        			}
	 	        			// check if value exists in stopwords
	 	        			if (cvalue!= this.to_replace_null && exists(values[targets_columns[0]], this.replaceable_values)) {
	 	        				cvalue=this.to_replace_null;
	 	        			}
	 	        			//update the value
	 	        			target[keep_current_row_count]= cvalue;
	 	        			
	 	        		} else {
	 	        			
	 	        			for (int j=0; j <targets_columns.length; j++ ){
	 	        				
		 	        			double cvalue=this.to_replace_null;
		 	        			try {
		 	        				cvalue=Double.parseDouble(values[targets_columns[j]]);
		 	        			} catch(Exception e){
		 	        				
		 	        				cvalue=this.to_replace_null;
		 	        			}
		 	        			// check if value exists in stopwords
		 	        			if (cvalue!= this.to_replace_null && exists(values[targets_columns[j]], this.replaceable_values)) {
		 	        				cvalue=this.to_replace_null;
		 	        			}	 	        				

	 	        			target2d[keep_current_row_count][j]=cvalue;
	 	        			}
	 	        		}	
	 	        		
	 	        	}
	 	        	// ids columns	
	 	        	//int
	 	        	
	 	        	if (count_ofintcol>0){
	 	        		
	 	        		intidcolumn[keep_current_row_count]= (int)Double.parseDouble(values[idint]);
	 	        		
	 	        	}
	 	        	
	 	        	//String
	 	        	if (count_ofstringcol>0){
	 	        		Stringidcolumn[keep_current_row_count]=values[idstring];
	 	        	}	        			
	 	        	// weights columns
	 	        	if (count_ofweightcolumn>0){
	 	        		weights[keep_current_row_count]=Double.parseDouble(values[wcolumn]);
	 	        	}
	 	        	
	 	        	
	 	        	// Main data file and columnames
	 	        	for (int j=0; j <columns_to_import.length; j++ ){
	 	        		
 	        			double cvalue=this.to_replace_null;
 	        			try {
 	        				cvalue=Double.parseDouble(values[columns_to_import[j]]);
 	        			} catch(Exception e){
 	        				cvalue=this.to_replace_null;
 	        			}
 	        			// check if value exists in stopwords
 	        			if (cvalue!= this.to_replace_null && exists(values[columns_to_import[j]], this.replaceable_values)) {
 	        				cvalue=this.to_replace_null;
 	        			}	 	        				

 	        			data_toimport[keep_current_row_count *columns_to_import.length+j ]=cvalue;
                		
                	}
	 	        	
	        		 
	        		 //increment row
	        		 keep_current_row_count++;
	        		 if (this.verbose){
	        			 if (keep_current_row_count%(number_of_rows/20)==0.0) {
	        				 System.out.printf(" Completed: %.2f %% \n",  ((double)keep_current_row_count/(double)number_of_rows)*100.0 );
	    	        	 }
	        			 }
	        		 }
	        		
	        	
                
	        	} catch (Exception e) {
	        		throw new IllegalStateException("File " + file + "  failed to import at bufferreader");
	        	}

	        	fsmatrix f= new fsmatrix(data_toimport,keep_current_row_count,columns_to_import.length);
	        	

	        	if (this.verbose){
	        		System.out.println(" Loaded File: " + file);
	        		System.out.println(" Total rows in the file: " + n_rows);
	        		System.out.println(" Total columns in the file: " + n_columns);	
	        		System.out.println(" Weighted variable : " + wcolumn + " counts: " +count_ofweightcolumn );
	        		System.out.println(" Int Id variable : " + idint + " str id: " + idstring + " counts: " +total_number_of_id_cols );	 
	        		System.out.println(" Target Variables  : " + number_ofvalid_target_columns + " values : " + Arrays.toString(targets_columns) );	 
	        		System.out.println(" Actual columns number  : " + columns_to_import.length  );	 
	        		System.out.println(" Number of Skipped rows   : " + this.skiprows  );	
	        		System.out.println(" Actual Rows (removing the skipped ones)  : " + keep_current_row_count  );		        			
	        	}
	        	
	        	return f;

	        }
	        
	        
	        /**
	         * 
	         * @param file : file to read (full path)
	         * @return : an fsmatrix.
	         * <p> Method to read a fixed size matrix. All rows in the file are assumed to have the same size as the first one.
	         */
	        
	        public double [][] ReadDoubleData(String file){
	        	
	        	 
	        	// get the number of rows
	        	int n_rows=input.GetRowCount(file,this.HasHeader);
	        	if (n_rows<=0){
	        		throw new IllegalStateException(" File " + file + " appears to be empty ");
	        	}
	        	// get the number of rows
	        	int n_columns=input.GetColumnCount(file, this.delimeter);
	        	// check number of columns
	        	if (n_columns<=0){
	        		throw new IllegalStateException(" File " + file + " has no columns with current delimeter  ");
	        	}
	          	//fix end if it exceeds current columns
	          	if ( end>n_columns ){
	          		end=n_columns;
	          	}
	        	if ( start<0 && end <0 ){
	          		start=-2;
	          		end=-1;
	          	}
	           //throw error if start exceeds number of cilumns
	          	if ( start>n_columns-1 ){
	        		throw new IllegalStateException(" start cannot be larger than the current number of columns  ");	          		
	          	}
	          	if ( start>=end ){
	        		throw new IllegalStateException(" start cannot be larger/equal than end  ");	          		
	          	}
	          	//get the actual number of rows to import
	          	int number_of_rows=0;
	          	if (skiprows<=0){
	          		number_of_rows=n_rows;
	          		skiprows=0;
	          	} else if (skiprows>number_of_rows){
	          		skiprows=0;
	          		throw new IllegalStateException(" Number of skipped rows cannot be larger than the number of rows in the file  ");	  
	          	} else {
	          		number_of_rows=n_rows-skiprows;
	          	}
	          	
	        	//determine the number of valid target columns
	        	int number_ofvalid_target_columns=0;
	        	for (int i=0; i <targets_columns.length;i++ ){
	        		if (targets_columns[i]>=0 && targets_columns[i]<=n_columns-1 ){
	        			number_ofvalid_target_columns++;
	        		}
	        	}
	        	int n=0;
	        	int new_target_cols[]= new int [number_ofvalid_target_columns];
	        	for (int i=0; i <targets_columns.length;i++ ){
	        		if (targets_columns[i]>=0 && targets_columns[i]<=n_columns-1 ){
	        			new_target_cols[n]=targets_columns[i];
	        			n++;
	        		}
	        	}
	        	
	        	targets_columns=new_target_cols; // assign the correct columns
	        	
	        	int count_ofweightcolumn=0;
	        	//n_columns-=targets_columns.length; // check if the weight variable is valie
	        	if (  wcolumn>=0  && wcolumn <=n_columns-1){
	        		count_ofweightcolumn=1;
	        	}else {
	        		wcolumn=-1;
	        	}
	        	// check int and string columns
	        	int count_ofstringcol=0;
	        	int count_ofintcol=0;
	        	int total_number_of_id_cols=0;
	        
	          	if (  idint>=0  && idint <=n_columns-1){
	          		count_ofintcol =1;
	          		total_number_of_id_cols=1;

	        	}else {
	        		idint=-1;
	        	}
	          	if (  idstring>=0  && idstring <=n_columns-1){
	          		count_ofstringcol=1;
	          		total_number_of_id_cols=1;
	        	}	else {
	        		idstring=-1;
	        	}

	          	// check main data columns

	        	if (start<0 && end<=0 && columns_to_import.length<=0){
	        		
	        		// load all the columns that do not fall in all other categories
	        		n=0;
	        		columns_to_import= new int [n_columns-total_number_of_id_cols-count_ofweightcolumn-number_ofvalid_target_columns] ;
	        		for (int i=0; i <n_columns; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				columns_to_import[n]=i;
	        				n++;
	        			}
	        		}
	        		
	        	} else if(start<0 &&  end>0 && columns_to_import.length<=0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=0; i <end; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				temp_list_of_valid_names.add(i);
	        			}
	        		}	   
	        		
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	        		
	        	}else if(start>=0 && end<0 && columns_to_import.length<=0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=start; i <n_columns; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				temp_list_of_valid_names.add(i);
	        				
	        			}
	        			
	        		}	   
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	     	
	        	} else if(start>=0 && end>0 && columns_to_import.length<=0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=start; i <end; i++){
	        			if (exists(i,targets_columns)==false && wcolumn!=i &&  idint!=i && idstring!=i){
	        				temp_list_of_valid_names.add(i);
	        			}
	        			
	        		}	   
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	     	// check valid columns for columns_to_import
	        	}else if(columns_to_import.length>0) {
	        		
	        		ArrayList<Integer> temp_list_of_valid_names= new ArrayList<Integer>();
	        		
	        		for (int i=0; i <columns_to_import.length; i++){
	        			if (columns_to_import[i]>=0 && 
	        					columns_to_import[i]< n_columns &&
	        					exists(columns_to_import[i],targets_columns)==false &&
	        					wcolumn!=columns_to_import[i] && 
	        					idint!=columns_to_import[i] &&
	        					idstring!=columns_to_import[i]){
	        				temp_list_of_valid_names.add(columns_to_import[i]);
	        			}
	        			
	        		}	   
	        		columns_to_import= convertIntegers(temp_list_of_valid_names);
	        		temp_list_of_valid_names=null;
	     	
	        	} else {
	        		
	        		throw new IllegalStateException(" No correct combination of columns to import were determined  ");
	        	}
	        	
	        	if (columns_to_import.length<=0){
	        		
	        		throw new IllegalStateException(" No columns to import were determined for fmatrix ");
	        		
	        	}
	        		     
	        	// ---------Start initialising objects-----------//
	        	
	        	// target columns
	        	if (number_ofvalid_target_columns>0){
	        		
	        		if (number_ofvalid_target_columns==1){
	        			target= new double [number_of_rows];
	        		} else {
	        			target2d=new double [number_of_rows][number_ofvalid_target_columns];
	        		}	
	        		
	        	}
	        	// ids columns	
	        	//int
	        	if (count_ofintcol>0){
	        		intidcolumn=new int [number_of_rows];
	          		
	        	}
	        	//String
	        	if (count_ofstringcol>0){
	        		Stringidcolumn=new String [number_of_rows];
	        	}	        			
	        	// weights columns
	        	if (count_ofweightcolumn>0){
	        		weights=new double [number_of_rows];
	        	}
	        	
	        	
	        	// Main data file and columnames
	        	double data_toimport [][]= new double [number_of_rows][columns_to_import.length];
	        	coulmnames= new String [columns_to_import.length];
	        	int keep_current_row_count=0;
	        	
	        	// ---------Start Reading the file -----------//	 
	        	try{
                FileInputStream fis = new FileInputStream(file);
                @SuppressWarnings("resource")
				BufferedReader br = new BufferedReader(new InputStreamReader(fis,input.charset));
                //The string object to hold the line
                String line="";
                // check for headers
                
                if (this.HasHeader){
                	//assign the headers
                	String colvalues [] = br.readLine().split(delimeter +"+",-1);
                	//import the columns
                	for (int j=0; j <columns_to_import.length; j++ ){
                		coulmnames[j]=colvalues[columns_to_import[j]];
                	}
                }
               
	        	// Skip the rows specified
	        	int number_of_rows_passed=0;
	        	while  (number_of_rows_passed < this.skiprows) {
	        		br.readLine();
	        		number_of_rows_passed++;
	        	
	        	}
	       	
               // Start importing data
	        	 while ((line = br.readLine()) != null  ) {
	        		 String values []=line.split(this.delimeter + "+",-1);
	        		 
	 	        	// ---------Start updating the  objects-----------//
	 	        	
	 	        	// target columns
	 	        	if (number_ofvalid_target_columns>0){
	 	        		
	 	        		if (number_ofvalid_target_columns==1){
	 	        			
	 	        			double cvalue=this.to_replace_null;
	 	        			try {
	 	        				cvalue=Double.parseDouble(values[targets_columns[0]]);
	 	        			} catch(Exception e){
	 	        				cvalue=this.to_replace_null;
	 	        			}
	 	        			// check if value exists in stopwords
	 	        			if (cvalue!= this.to_replace_null && exists(values[targets_columns[0]], this.replaceable_values)) {
	 	        				cvalue=this.to_replace_null;
	 	        			}
	 	        			//update the value
	 	        			target[keep_current_row_count]= cvalue;
	 	        			
	 	        		} else {
	 	        			for (int j=0; j <targets_columns.length; j++ ){
	 	        				
		 	        			double cvalue=this.to_replace_null;
		 	        			try {
		 	        				cvalue=Double.parseDouble(values[targets_columns[j]]);
		 	        			} catch(Exception e){
		 	        				cvalue=this.to_replace_null;
		 	        			}
		 	        			// check if value exists in stopwords
		 	        			if (cvalue!= this.to_replace_null && exists(values[targets_columns[j]], this.replaceable_values)) {
		 	        				cvalue=this.to_replace_null;
		 	        			}	 	        				

	 	        			target2d[keep_current_row_count][j]=cvalue;
	 	        			}
	 	        		}	
	 	        		
	 	        	}
	 	        	// ids columns	
	 	        	//int
	 	        	
	 	        	if (count_ofintcol>0){
	 	        		
	 	        		intidcolumn[keep_current_row_count]= (int)Double.parseDouble(values[idint]);
	 	        		
	 	        	}
	 	        	
	 	        	//String
	 	        	if (count_ofstringcol>0){
	 	        		Stringidcolumn[keep_current_row_count]=values[idstring];
	 	        	}	        			
	 	        	// weights columns
	 	        	if (count_ofweightcolumn>0){
	 	        		weights[keep_current_row_count]=Double.parseDouble(values[wcolumn]);
	 	        	}
	 	        	
	 	        	
	 	        	// Main data file and columnames
	 	        	for (int j=0; j <columns_to_import.length; j++ ){
	 	        		
 	        			double cvalue=this.to_replace_null;
 	        			try {
 	        				cvalue=Double.parseDouble(values[columns_to_import[j]]);
 	        			} catch(Exception e){
 	        				cvalue=this.to_replace_null;
 	        			}
 	        			// check if value exists in stopwords
 	        			if (cvalue!= this.to_replace_null && exists(values[columns_to_import[j]], this.replaceable_values)) {
 	        				cvalue=this.to_replace_null;
 	        			}	 	        				

 	        			data_toimport[keep_current_row_count ][j]=cvalue;
                		
                	}
	 	        	
	        		 
	        		 //increment row
	        		 keep_current_row_count++;
	        		 if (this.verbose){
	        			 if (keep_current_row_count%(number_of_rows/20)==0.0) {
	        				 System.out.printf(" Completed: %.2f %% \n",  ((double)keep_current_row_count/(double)number_of_rows)*100.0 );
	    	        	 }
	        			 }
	        		 }
	        		
	        	
                
	        	} catch (Exception e) {
	        		throw new IllegalStateException("File " + file + "  failed to import at bufferreader");
	        	}

	        	
	        	

	        	if (this.verbose){
	        		System.out.println(" Loaded File: " + file);
	        		System.out.println(" Total rows in the file: " + n_rows);
	        		System.out.println(" Total columns in the file: " + n_columns);	
	        		System.out.println(" Weighted variable : " + wcolumn + " counts: " +count_ofweightcolumn );
	        		System.out.println(" Int Id variable : " + idint + " str id: " + idstring + " counts: " +total_number_of_id_cols );	 
	        		System.out.println(" Target Variables  : " + number_ofvalid_target_columns + " values : " + Arrays.toString(targets_columns) );	 
	        		System.out.println(" Actual columns number  : " + columns_to_import.length  );	 
	        		System.out.println(" Number of Skipped rows   : " + this.skiprows  );	
	        		System.out.println(" Actual Rows (removing the skipped ones)  : " + keep_current_row_count  );		        			
	        	}
	        	
	        	return data_toimport;

	        }
	        

	        

	        /**
	         * 
	         * @param file : File to read
	         * @param second_delimiter : second delimiter that separates the column from the value
	         * @param hashead : True if the files has a header ( that needs to be skipped)
	         * @param has_target : True if it contains a target variable in the first column 
	         * @return an smatrix
	         * <p> read data in sparse (@see smatrix) format
	         * 
	         */
	        public smatrix readsmatrixdata(String file, String second_delimiter, boolean hashead, boolean has_target){
	        	
	        	// first we need to count the number of elements in the file (rows and columns)
	        	// this file cannot have headers
 
	        	int stats [] =GetTotalNumberOfElementsandrows(file, this.delimeter, hashead,has_target);
	        	System.out.println(Arrays.toString(stats));
	        	int elements=stats[0];
	        	int rowsize=stats[1];
	        	//initialize the elements of the smatrix
	        	int rows []= new int [rowsize+1];
	        	int cols []= new int [elements];
	        	double values []= new double [elements];
	        	int element_counter=0;
	        	int row_counter=0;
	        	int column_counter=0;
	        	String line="";
	            try {
	                    FileInputStream fis = new FileInputStream(file);
	                    @SuppressWarnings("resource")
						BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
	                    if ( hashead){
	                    	
                        	br.readLine();
                        }
	                   
	                    while ((line = br.readLine()) != null  ) {
	                    	String element []=line.split(this.delimeter + "+",-1);	
	                    	int len=element.length;
	                    	int hh=0;
	                    	if (has_target){
	                    		hh=1;
	                    		
	                    	}
	                    	if (len>1 && !element[1].equals((""))){
	                    		for (int h=hh; h < len; h++) {
	                    			String this_value[]=element[h].split(second_delimiter + "+",-1);
	                    			try {
	                    			int columns=Integer.parseInt(this_value[0]);
	                    			double val=-9999.99;
	                    			try {
	                    				val=Double.parseDouble(this_value[1]);
	                    			} catch (Exception e) {
	                    			}
	                    			if (val==0.0){
	                    				continue;
	                    			}
	                    			//rows[element_counter]=row_counter;
	                    			cols[element_counter]=columns;
	                    			if (columns>column_counter){
	                    				column_counter=columns;
	                    			}
	                    			values[element_counter]=val;
	                    			element_counter++;
	                    			} catch (Exception e) {
	                    				throw new IllegalStateException("Could not parse elemnt : " + h + " at row: " + row_counter + " and specifically the value of: " +element[h] );	
	                    			}
	                    		}


	                    	} else if (len==1 && has_target==false){
	                    		int nlean=element[0].length();
	                    		if (nlean>=3){
	                    			String this_value[]=element[0].split(second_delimiter + "+",-1);
	                    			try {
	                    			int columns=Integer.parseInt(this_value[0]);
	                    			double val=-9999.99;
	                    			try {
	                    				val=Double.parseDouble(this_value[1]);
	                    			} catch (Exception e) {
	                    			
	                    			}
	                    			//rows[element_counter]=row_counter;
	                    			cols[element_counter]=columns;
	                    			if (columns>column_counter){
	                    				column_counter=columns;
	                    			}
	                    			values[element_counter]=val;
	                    			element_counter++;
	                    			} catch (Exception e) {
	                    				throw new IllegalStateException("Could not parse elemnt : " + 0 + " at row: " + row_counter + " and specifically the value of: " +element[0] );	
	                    			}
	                    			

	                    		}
	                    	}
	   
	                    	
	                    	row_counter++;	
	                    	rows[row_counter]=element_counter;
	   	        		 if (this.verbose){
		        			 if (element_counter%(elements/20)==0.0) {
		        				 System.out.printf(" Completed: %.2f %% \n",  ((double)element_counter/(double)elements)*100.0 );
		    	        	 }
		        			 }
	                    }
            	} catch (Exception e) {
	        		throw new IllegalStateException("File " + file + "  failed to import at bufferreader");
	        	}
	            
	        	if (this.verbose){
	        		System.out.println(" Loaded File: " + file);
	        		System.out.println(" Total rows in the file: " + row_counter);
	        		System.out.println(" Total columns in the file: undetrmined-Sparse");		
	        		System.out.println(" Number of elements : " + elements  );		        			
	        	}
	            
	            return  new smatrix(values, cols, rows,  row_counter,column_counter+1, element_counter,true);
	          	
	        }
	            
	            /**
	             * @param file : File to read
	             * @param delimiter :  delimiter 
	             * @param has_headers :  True if the files has a header that needs to be skipped
	             * @return : the total number of elements
	             */
	            @SuppressWarnings("resource")
				public static int GetTotalNumberOfElements(String file, String delimiter, boolean has_headers){
	            	int elements=0;
	            	
	            	   try {
		                    FileInputStream fis = new FileInputStream(file);
		                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                            String line;
                            if (has_headers){
                            	br.readLine();
                            }
		                    while ((line = br.readLine()) != null  ) {
		                    	String element []=line.split(delimiter + "+",-1);
		                    	int len=element.length;
		                    	if (len>1){
		                    	elements+=element.length;
		                    	} else if (len==1){
		                    		int nlean=element[0].length();
		                    		if (nlean>=3){
		                    			elements+=1;
		                    		}
		                    	}
		                    }
		                    	
		                	} catch (Exception e) {
		    	        		throw new IllegalStateException("File " + file + "  failed to import at bufferreader");
		    	        	}
	            	   return elements;
	            }
	            /**
	             * @param file : File to read
	             * @param delimiter :  delimiter 
	             * @param has_headers : True if the files has a header ( that needs to be skipped)
	             * @param hasttarget : True if it contains a target variable in the first column 
	             * @return : the total number of elements [0] and total rows in [1]
	             */
	            @SuppressWarnings("resource")
				public static int [] GetTotalNumberOfElementsandrows(String file, String delimiter, boolean has_headers, boolean hasttarget){
	            	int elements=0;
	            	int rows=0;
	            	 int ss=0;
	            	 if (hasttarget){
	            		 ss=1;
	            	 }
	            	   try {
		                    FileInputStream fis = new FileInputStream(file);
		                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                            String line;
                            if (has_headers){
                            	br.readLine();
                            }
                          
		                    while ((line = br.readLine()) != null  ) {
		                    	rows+=1;
		                    	String element []=line.split(delimiter + "+",-1);
		                    	int len=element.length;
		                    	if ( len>1  && !element[1].equals("") || (len==1  && hasttarget==false)){
		                    	for (int j=ss; j <element.length;j++ ){
		                    		String this_value[]=element[j].split(":" + "+" ,-1);
	                    			double val=-9999.99;
	                    			try {
	                    				val=Double.parseDouble(this_value[1]);
	                    			} catch (Exception e) {
	                    			}
	                    			if (val!=0.0){
	                    				elements+=1;
	                    			}
		                    	}
		                    	} 
			                    if (rows%100000==0){
			                    	System.out.println(rows + " " + elements);
			                    }		                    	
		                    }

		                	} catch (Exception e) {
		    	        		throw new IllegalStateException("File " + file + "  failed to import at row " + rows  + " bufferreader");
		    	        	}
	            	  
	            	   return new int [] {elements,rows};
	            }
	            
	            	
	            /**
	            *
	            * @param n : The file to 'Open'.
	            * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
	            * @param col : The column to retrieve starting from 0. Ay value lower than zero gives 0.
	            * @param nullvalue : Replace null values with the double here.
	            * @param hasheaders : True if we want the first line to be regarded as header
	            * @param verbose : Print details about the imported file.
	             * @return : The the specified column as double array
	            * <p> Method to read a file and retrieve the specified column as double array
	            */    

	           

	           public static double[]Retrievecolumn(String n, String delimeter, int col,

	                           double nullvalue,boolean hasheaders, boolean verbose) {

	               String line;
	               File x= new File(n);
	               int rowcount=GetRowCount(n,hasheaders);
	               //If less than zero, ammend to zzero
	               if (col<0){
	                           col=0;
	               }   
	               //System.out.println(rowcount);
	               double column [] = new double [rowcount]  ;
	               try {
	                       FileInputStream fis = new FileInputStream(x);
	                       BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
	                       if (hasheaders){
	                       br.readLine();
	                       }

	                       int ro=0;
	                       while ((line = br.readLine()) != null) {
	                               String[] tokens = line.split(delimeter,-1);
	                                       if (tokens[col].equals("") ||tokens[col].equals("NA")  ||tokens[col].equals("nan") ){
	                                                   tokens[col]=nullvalue +"";
	                                       }
	                                       column[ro]=((Double.parseDouble(tokens[col])));
	                                       ro++; }

	                 //Close the buffer reader
	                       br.close();
	               } catch (Exception e) {
	                       e.printStackTrace();
	               }

	    

	             if ( verbose==true){
	               System.out.println(" The file" + x + " was loaded successfully with :");
	               System.out.println(" Rows : " + rowcount);
	               System.out.println(" Columns (excluding target) : 1");
	               System.out.println(" Delimeter was  : " + delimeter);
	               } 

	             return column;

	       }

	            /**
	            *
	            * @param n : The file to 'Open'.
	            * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
	            * @param col : The column to retrieve starting from 0. Ay value lower than zero gives 0.
	            * @param nullvalue : Replace null values with the double here.
	            * @param hasheaders : True if we want the first line to be regarded as header
	            * @param verbose : Print details about the imported file.
	             * @return : The the specified column as int array
	            * <p> Method to read a file and retrieve the specified column as int array
	            */    

	           

	           public static int[]Retrievecolumnint(String n, String delimeter, int col,

	                           int nullvalue,boolean hasheaders, boolean verbose) {

	               String line;
	               File x= new File(n);
	               int rowcount=GetRowCount(n,hasheaders);
	               //If less than zero, ammend to zzero
	               if (col<0){
	                           col=0;
	               }                     
	               int column [] = new int [rowcount]  ;
	               try {
	                       FileInputStream fis = new FileInputStream(x);
	                       BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
	                       if (hasheaders){
	                       br.readLine();
	                       }

	                       int ro=0;
	                       while ((line = br.readLine()) != null) {
	                               String[] tokens = line.split(delimeter,-1);
	                                       if (tokens[col].equals("") ||tokens[col].equals("NA") ){
	                                                   tokens[col]=nullvalue +"";
	                                       }
	                                       column[ro]=((Integer.parseInt(tokens[col])));
	                                       ro++; }

	                 //Close the buffer reader
	                       br.close();
	               } catch (Exception e) {
	                       e.printStackTrace();
	               }

	    

	             if ( verbose==true){
	               System.out.println(" The file" + x + " was loaded successfully with :");
	               System.out.println(" Rows : " + rowcount);
	               System.out.println(" Columns (excluding target) : 1");
	               System.out.println(" Delimeter was  : " + delimeter);
	               } 

	             return column;

	       }	    
	           /**
	            *
	            * @param n : The file to 'Open'.
	            * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
	            * @param col : The column to retrieve starting from 0. Ay value lower than zero gives 0.
	            * @param nullvalue : Replace null values with the String here.
	            * @param hasheaders : True if we want the first line to be regarded as header
	            * @param verbose : Print details about the imported file.
	            * <p> Method to read a file and retrieve the specified column as String array
	            */    
	           public static String[] Retrievecolumn(String n, String delimeter, int col, String nullvalue
	                           ,boolean hasheaders, boolean verbose) {
	               String line;
	               File x= new File(n);
	               int rowcount=GetRowCount(n,hasheaders);
	               //If less than zero, ammend to zero
	               if (col<0){
	                           col=0;
	               }
	               String column [] = new String [rowcount]  ;          
	               try {
	                       FileInputStream fis = new FileInputStream(x);
	                       BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
	                       if (hasheaders){
	                           br.readLine();
	                           }
	                       int ro=0;
	                       while ((line = br.readLine()) != null) {
	                               String[] tokens = line.split(delimeter,-1);
	                                       if (tokens[col].equals("")){
	                                                  tokens[col]=nullvalue;
	                                       }
	                                       column[ro]=(tokens[col]);
	                                       ro++;
	              }
	                 //Close the buffer reader
	                      br.close();
	               } catch (Exception e) {
	                       e.printStackTrace();
	               }

	              
	             if ( verbose==true){
	               System.out.println(" The file" + x + " was loaded successfully with :");
	               System.out.println(" Rows : " + rowcount);
	               System.out.println(" Columns (excluding target) : 1");
	               System.out.println(" Delimeter was  : " + delimeter);
	               } 
	             return column;

	       }

	           
	           
	           
		        /**
		         * 
		         * @param file : File to read
		         * @return a 2-dimensional String array for layers and models. 
		         * <p> read a configuration file that lists the models to be used for a StackNet Model . 
		         * 
		         */
		        public static String[][] StackNet_Configuration(String file){
		        	
		        	ArrayList<ArrayList<String>> models_list= new ArrayList<ArrayList<String>>();
		        	
		        	// first we need to count the number of elements in the file (rows and columns)
		        	// this file cannot have headers
	 
		        	int row_counter=0;
		        	String line="";
		        	ArrayList<String> level_list= new ArrayList<String> ();
		        	models_list.add(level_list);
		        	
		            try {
		                    FileInputStream fis = new FileInputStream(file);
		                    @SuppressWarnings("resource")
							BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		             
		                    while ((line = br.readLine()) != null  ) {
		                    	//measure line length
		                    	line=line.replace("\n", "");
		                    	if (line.contains("#")){
		                    		String splits [] =line.split("#");
		                    		line=splits[0];
		                    	}
		                    	int line_length=line.length();
		                    	
		                    	// if size is big enough to be considered an 'algorithm'
		                    	if (line_length > 9){
		                    		boolean model_OK=check_model_status(line);
		                    		if (model_OK){
		                    			models_list.get(models_list.size()-1).add(line);
		                    		} else {
		                    			throw new IllegalStateException("Line " + row_counter + " : " +  line + " does not contain a valid StackNet input model, please check the spelling - It is case sensitive. ");
		                    		}
		                    	} else {
		                    		if (models_list.get(models_list.size()-1).size()>0){
		                    		// we add a new list - e.g. new modelling layer
			                    		ArrayList<String> temp_level_list= new ArrayList<String> ();
			        		        	models_list.add(temp_level_list);
		                    		}
		                    		
		                    	}
		                    	row_counter++;
		                    }      
	            	} catch (Exception e) {
		        		throw new IllegalStateException("File " + file + "  failed to import at bufferreader " + e.getMessage());
		        	}
		            
		            if (models_list.get(models_list.size()-1).size()==0){
		            	models_list.remove(models_list.size()-1);
		            }
		            if (models_list.size()<=0){
		            	throw new IllegalStateException("There is no valid model in " + file);
		            }
		            
		            String[][] level_models= new String[models_list.size()][];
		            int cc=0;
		            for (ArrayList<String> temp_level_list : models_list){
		            	String models []= new String [temp_level_list.size()];
		            	for (int i=0; i <temp_level_list.size();i++ ){
		            		models[i]=temp_level_list.get(i);
		            	}
		            	level_models[cc]=models;
		            	cc++;
		            }
		            return  level_models;
		          	
		        }
	        
		        /**
		         * 
		         * @param str_estimator : model parameters
		         * @return if it is exists or not
		         */
			private static  boolean check_model_status(String str_estimator){
	        		boolean is_valid=false;
					
					if (str_estimator.contains("AdaboostForestRegressor")) {
						is_valid=true;
					} else if (str_estimator.contains("AdaboostRandomForestClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("DecisionTreeClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("DecisionTreeRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("GradientBoostingForestClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("GradientBoostingForestRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("RandomForestClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("RandomForestRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("Vanilla2hnnregressor")) {
						is_valid=true;
					}else if (str_estimator.contains("Vanilla2hnnclassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("softmaxnnclassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("multinnregressor")) {
						is_valid=true;
					}else if (str_estimator.contains("NaiveBayesClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("LSVR")) {
						is_valid=true;
					}else if (str_estimator.contains("LSVC")) {
						is_valid=true;
					}else if (str_estimator.contains("OriginalLibFMClassifier")) {
						is_valid=true;						
					}else if (str_estimator.contains("OriginalLibFMRegressor")) {
						is_valid=true;							
					}else if (str_estimator.contains("LogisticRegression")) {
						is_valid=true;
					}else if (str_estimator.contains("LinearRegression")) {
						is_valid=true;
					}else if (str_estimator.contains("LibFmRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("LibFmClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("knnClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("knnRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("KernelmodelClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("KernelmodelRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("XgboostClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("XgboostRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("LightgbmClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("LightgbmRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("H2OGbmClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("H2OGbmRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("H2ODeepLearningClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("H2ODeepLearningRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("H2ODrfClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("H2ODrfRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("H2OGlmClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("H2OGlmRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("H2ONaiveBayesClassifier")) {
						is_valid=true;					
					}else if (str_estimator.contains("SklearnAdaBoostClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnAdaBoostRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnDecisionTreeClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnDecisionTreeRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnExtraTreesClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnExtraTreesRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnknnClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnknnRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnMLPClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnMLPRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnRandomForestClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnRandomForestRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnSGDClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnSGDRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnsvmClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("SklearnsvmRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("KerasnnRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("KerasnnClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("PythonGenericClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("PythonGenericRegressor")) {
						is_valid=true;
					}else if (str_estimator.contains("FRGFClassifier")) {
						is_valid=true;
					}else if (str_estimator.contains("FRGFRegressor")) {
						is_valid=true;					
					}else if (str_estimator.contains("libffmClassifier")) {
						is_valid=true;						
					}else if (str_estimator.contains("VowpaLWabbitClassifier")) {
						is_valid=true;							
					}else if (str_estimator.contains("VowpaLWabbitRegressor")) {
						is_valid=true;	
					}
						
						
					
					return is_valid;
	        }
	        /**
	         * 
	         * @param a : int to check if exists in array b
	         * @param b : int array
	         * @return true if the element exists
	         */
	        public boolean exists(int a, int b[]) {
	        	boolean exists=false;
	        	for (int i=0; i < b.length; i++){
	        		if (a==b[i]){
	        			exists=true;
	        			break;
	        		}
	        	}
	        	return exists;
	        }
	        /**
	         * 
	         * @param a : v to check if exists in array b
	         * @param b : String array
	         * @return true if the element exists
	         */
	        public boolean exists(String a, String b[]) {
	        	boolean exists=false;
	        	for (int i=0; i < b.length; i++){
	        		if (a.equals(b[i])){
	        			exists=true;
	        			break;
	        		}
	        	}
	        	return exists;
	        }	        
	        /**
	         * 
	         * @param integers : an arraylist of integers
	         * @return and int array of ints
	         */
	        public static int[] convertIntegers(ArrayList<Integer> integers)
	        {
	            int[] ret = new int[integers.size()];
	            for (int i=0; i < ret.length; i++)
	            {
	                ret[i] = integers.get(i).intValue();
	            }
	            return ret;
	        }
	        
	        
	        
	        
}
