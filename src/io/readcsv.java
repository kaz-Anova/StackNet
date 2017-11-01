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

import java.io.*;
import java.util.*;
import java.io.File; 
import java.net.URL;
import java.nio.charset.Charset;
import Stats.DescriptiveStatistics;
import utilis.map.intint.IntIntMapminus4a;

/**
 * 
 * @author mariosm
 *
 * Unstructured class for loading .csv data. It is not recommended for usage as it was built at 
 * a very primitive stage for quick loading of data It still may be of some use. 
 */

public class readcsv   {
	
	
/**
 * String data holder
 */
String datas [][];
/**
 * double data holder
 */
double datasdouble [][];
/**
 * double array reserved for the first column of the data
 */
double firstcol [];
/**
 * double array reserved for the target variable
 */
double target [];
/**
 * String array with the column names of the file
 */
static String coulmnames[];
/**
 * int Array with indices of the numerical columns
 */
public static int numericcols[];
/**
 * int Array with indices of the non-numerical columns
 */
public static int nonnumericcols[];
/**
 * default delimeter of the files
 */
public String delimiter=",";
/**
 * another double array reserved for the target variable
 */
public static double [] label ;


/**
 * 
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
            int rowcount=readcsv.getrowcount(n);           
            if (hasheaders==false){
            	rowcount++;
            }
           
           Vector<String>   columns = new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    new StringTokenizer(br.readLine(), delimeter);
                  
                  
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
                    datas = new String [rowcount][columncount]  ;
                    int ro=0;
                    while ((line = br.readLine()) != null && line_count<xfirstlines ) {
                            String[] tokens = line.split(delimeter,-1);
                            for  (int i =0 ; i<coulmnames.length; i ++ ) {
                            	try{
                            	if (tokens[i].equals("")){
                            		tokens[i]=Double.NaN + "";
                            	}
                            	datas[ro][i]=tokens[i];
                            } catch (Exception e){
                            	datas[ro][i]=Double.NaN + "";}
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
        	  String maxs=datas[0][j];
        	  String mins=datas[0][j];
        	  boolean isdouble =true;
        	  for (int i=0; i < datas.length; i++){
        		 
        		  try{
        			  Double.parseDouble(datas[i][j]);
        		  }catch (Exception e){
        			  isdouble=false;
        			  break;
        		  }
        		  
        	  }
        	  if (isdouble==true){
        		  num.add(j);
        	  for (int i=0; i < datas.length; i++){
        		  Double v= Double.parseDouble(datas[i][j]);
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
        		  for (int i=0; i < datas.length; i++){
            		  String v= datas[i][j];
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
          System.out.println("The type of columns can be retrieved through getnumericcolumns() and getnonnumericcolumns() methods ");
        	
        	
        }
        
        /**
         * 
         * @param filePath : Path to the file
         * @return : file as string
         * @throws IOException
         */
        public static String readFileAsString(String filePath) throws IOException {
            StringBuffer fileData = new StringBuffer();
            BufferedReader reader = new BufferedReader(
                    new FileReader(filePath));
            char[] buf = new char[1024];
            int numRead=0;
            while((numRead=reader.read(buf)) != -1){
                String readData = String.valueOf(buf, 0, numRead);
                fileData.append(readData);
            }
            reader.close();
            return fileData.toString();
        }
        
        /**
         * 
         * @return The list of numeric columns in the file read through getfileinfo method.
         */
        public static int [] getnumericcolumns(){
        	return numericcols;
        }
        
        /**
         * 
         * @return The list of non-numeric columns in the file read through getfileinfo method.
         */
        public static int [] getnonnumericcolumns(){
        	return nonnumericcols;
        }

/**
 * 
 * @param n : The name of file to 'Open'.
 * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
 * @param cols : Array that holds the arrays we want to import.
 * @param nullvalue : Replace null values with the String here.
 * @param hasheaders : True if we want the first line to be regarded as header
 * @param hasconsecutivedel : Regard consecutive delimiters as one.
 * @param verbose : Print details about the imported file.
 * <p> Method to read a file as a String array.
 */


		public void putfiletoarrayString(String n, String delimeter, int [] cols, String nullvalue,
        		boolean hasheaders,boolean hasconsecutivedel,boolean verbose) {
        	File x= new File(n);
        	 String line;
             int colcount=0;
             int columncount=0;
             int rowcount=readcsv.getrowcount(n);   
             Arrays.sort(cols); 
             if (hasheaders==false){
             	rowcount++;
             }
            new Vector<String>();
             try {
                     FileInputStream fis = new FileInputStream(x);
                     BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    // Map the first Row
                  
                     if (hasheaders==true){
                    	 String [] coulmnamess;
                    	 
                    	  	if (hasconsecutivedel){
                    	  		coulmnamess = br.readLine().split(delimeter +"+",-1);
                                	} else{
                                		coulmnamess = br.readLine().split(delimeter,-1);
                                	}
                    	 
                    	 ArrayList<String> coll= new ArrayList<String>();
                         for (int i=0; i<cols.length; i++){
                      	   coll.add(coulmnamess[cols[i]]);
                         }
                         
                         coulmnames = (String[])coll.toArray(new String[coll.size()]);
                         } else{
                         	coulmnames= new String[cols.length];
                         	for (int i=0; i<coulmnames.length;i++ ){
                         		coulmnames[i]=cols[i] + "";
                         	}
                         	fis = new FileInputStream(x);
                         	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));

                         	
                         }
       
                     colcount=coulmnames.length;
                      columncount=colcount;
                      datas = new String [rowcount][columncount]  ;
                     int ro=0;
                     while ((line = br.readLine()) != null) {
                             String[] tokens = line.split(delimeter,-1);
                           
                             for  (int i =0 ; i<cols.length; i ++ ) {
                             	if (tokens[cols[i]].equals("")){
                             		tokens[cols[i]]=nullvalue;
                             	}
                             	datas[ro][i]=(tokens[cols[i]]);
                             ;}
                             ro++;  }
               //Close the buffer reader
                     br.close();
             } catch (Exception e) {
                     e.printStackTrace();
             }
   
   
           if ( verbose==true){
             	 System.out.println(" The file" + x + " was loaded successfully with :");
             	 System.out.println(" Rows : " + rowcount);
             	 System.out.println(" Columns : " + columncount);
             	 System.out.println(" Delimeter was  : " + delimeter);
              }    
           
     }
        
        /**
         * 
         * @param x : The name of file to 'Open'.
         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param cols : Array that holds the arrays we want to import.
         * @param nullvalue : Replace null values with the String here.
         * @param hasheaders : If true headers exists in the file.
         * @param verbose : Print details about the imported file.
         * @return Data as string 2d array
         * <p> Static Method to read a file as a String array. It ignores the first row.
         */

                public static String [][] putfiletoarrayStringstatic(String x, String delimeter, int [] cols, String nullvalue,boolean hasheaders, boolean verbose) {
                	readcsv reads= new readcsv();
                	reads.putfiletoarrayString(x, delimeter, cols, nullvalue,hasheaders,true, verbose);
                	return reads.getdatasetString();
                }
                
                /**
                 * 
                 * @param x : The file to 'Open'.
                 * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
                 * @param cols : Array that holds the arrays we want to import.
                 * @return Data as string 2d array
                 * <p> Static Method to read a file as a String array. It ignores the first row. It replaces null values with 'null'.
                 */

                        public static String [][] putfiletoarrayStringstatic(String x, String delimeter, int [] cols) {
                        	readcsv reads= new readcsv();
                        	reads.putfiletoarrayString(x, delimeter, cols, "null", false,true,false);
                        	return reads.getdatasetString();
                        } 
                        
                        /**
                         * 
                         * @param x : The file to 'Open'.
                         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
                         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
                         * @param last : Import until the number of the column specified here, where 0 refers to first column
                         * @param nullvalue : Replace null values with the String here.
                         * @param hasheaders : If true headers exists in the file.
                         * @param verbose : Print details about the imported file.
                         * @return Data as string 2d array
                         * <p> Static Method to read a file as a String array.
                         */


                   public static String [][] putfiletoarrayStringstatic(String x,String delimeter, int start, int last, String nullvalue,boolean hasheaders, boolean verbose) {
                     readcsv reads= new readcsv();
                     reads.putfiletoarrayString(x, delimeter, start, last, nullvalue, hasheaders,true,verbose);
                     return reads.getdatasetString();
                     }   
                   
                   /**
                    * 
                    * @param x : The file to 'Open'.
                    * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
                    * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
                    * @param last : Import until the number of the column specified here, where 0 refers to first column
                    * @return Data as string 2d array
                    * <p> Static Method to read a file as a String array.It ignores the first row. It replaces null values with 'null'.
                    */


              public static String [][] putfiletoarrayStringstatic(String x,String delimeter, int start, int last) {
                readcsv reads= new readcsv();
                reads.putfiletoarrayString(x, delimeter, start, last, "null",false, true,false);
                return reads.getdatasetString();
                }  
              
              /**
               * 
               * @param x : The file to 'Open'.
               * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
               * @return Data as string 2d array               
               * <p> Static Method to read a file as a String array.It ignores the first row, imports all columns and replaces null values with 'null'.
               */


         public static String [][] putfiletoarrayStringstatic(String x,String delimeter) {
           readcsv reads= new readcsv();
           reads.putfiletoarrayString(x, delimeter);
           return reads.getdatasetString();
           } 
         
         /**
          * 
          * @param x : The file to 'Open'.
          * @return Data as string 2d array         
          * <p> Static Method to read a file as a String array.The delimiter is assumed to be comma ',', it ignores the first row, imports all columns and replaces null values with 'null'.
          */


    public static String [][] putfiletoarrayStringstatic(String x) {
      readcsv reads= new readcsv();
      reads.putfiletoarrayString(x);
      return reads.getdatasetString();
    } 
 
    /**
     * 
     * @param x : The file to 'Open'.
     * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
     * @param cols : Array that holds the arrays we want to import.
     * @param nullvalue : Replace null values with the String here.
     * @param hasheaders : If true headers exists in the file.
     * @param verbose : Print details about the imported file.
     * @return Data as double 2d array
     * <p> Static Method to read a file as a double array. It ignores the first row.
     */

            public static double [][] putfiletoarraydoublestatic(String x, String delimeter, int [] cols, String nullvalue,boolean hasheaders, boolean verbose) {
            	readcsv reads= new readcsv();
            	reads.putfiletoarraydouble(x, delimeter, cols, nullvalue, hasheaders,true, verbose);
            	return reads.getdatasetdouble();
            }
            
            /**
             * 
             * @param x : The file to 'Open'.
             * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
             * @param cols : Array that holds the arrays we want to import.
             * @return Data as double 2d array 
             * <p> Static Method to read a file as a double array. It ignores the first row. It replaces null values with 0'.
             */

                    public static double [][] putfiletoarraydoublestatic(String x, String delimeter, int [] cols) {
                    	readcsv reads= new readcsv();
                    	reads.putfiletoarraydouble(x, delimeter, cols, "0",false,true, false);
                    	return reads.getdatasetdouble();
                    } 
                    
                    /**
                     * 
                     * @param x : The file to 'Open'.
                     * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
                     * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
                     * @param last : Import until the number of the column specified here, where 0 refers to first column
                     * @param nullvalue : Replace null values with the String here.
                     * @param verbose : Print details about the imported file.
                     * @return Data as double 2d array
                     * <p> Static Method to read a file as a double array.
                     */


               public static double [][] putfiletoarraydoublestatic(String x,String delimeter, int start, int last, double nullvalue, boolean verbose) {
                 readcsv reads= new readcsv();
                 reads.putfiletoarraydouble(x, delimeter, start, last, nullvalue,false,true, verbose);
                 return reads.getdatasetdouble();
                 }   
               
               /**
                * 
                * @param x : The file to 'Open'.
                * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
                * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
                * @param last : Import until the number of the column specified here, where 0 refers to first column
                * @return Data as double 2d array              
                * <p> Static Method to read a file as a double array.It ignores the first row. It replaces null values with 0.
                */


          public static double [][] putfiletoarraydoublestatic(String x,String delimeter, int start, int last) {
            readcsv reads= new readcsv();
            reads.putfiletoarraydouble(x, delimeter, start, last, 0,false,true, false);
            return reads.getdatasetdouble();
            }  
          
          /**
           * 
           * @param x : The file to 'Open'.
           * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
           * @return Data as double 2d array
           * <p> Static Method to read a file as a double array.It ignores the first row, imports all columns and replaces null values with '0'.
           */


     public static double [][] putfiletoarraydoublestatic(String x,String delimeter) {
       readcsv reads= new readcsv();
       reads.putfiletoarraydouble(x, delimeter);
       return reads.getdatasetdouble();
       } 
     
     /**
      * 
      * @param x : The file to 'Open'.
      * @return Data as double 2d array
      * <p> Static Method to read a file as a double array.The delimeter is assumed to be comma ',', it ignores the first row, imports all columns and replaces null values with '0'.
      */


public static double [][] putfiletoarraydoublestatic(String x) {
  readcsv reads= new readcsv();
  reads.putfiletoarraydouble(x);
  return reads.getdatasetdouble();
} 
        	
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param cols : Array that holds the arrays we want to import.
         * @param nullvalue : Replace null values with the String here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one. 
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a double array.
         */


				public void putfiletoarraydouble(String n, String delimeter, int [] cols, String nullvalue, 
                		boolean hasheaders,boolean hasconsecutivedel,boolean verbose) {
                	File x= new File(n);
                	 String line;
                     int columncount=0;
                     int rowcount=readcsv.getrowcount(n);   
                     Arrays.sort(cols);
                     if (hasheaders==false){
                     	rowcount++;
                     }                 
                    new Vector<String>();
                     try {
                             FileInputStream fis = new FileInputStream(x);
                             BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));

                            // Map the first Row
                 
                             
                             if (hasheaders==true){
                            	 

                            	 String [] coulmnamess;
                            	                    	 
                            	  if (hasconsecutivedel){
                            	  coulmnamess = br.readLine().split(delimeter +"+",-1);
                            	 } else{
                            		 coulmnamess = br.readLine().split(delimeter,-1);
                            	}
                            	 ArrayList<String> coll= new ArrayList<String>();
                                 for (int i=0; i<cols.length; i++){
                              	   coll.add(coulmnamess[cols[i]]);
                                 }
                                 coulmnames = (String[])coll.toArray(new String[coll.size()]);
                                 } else{
                                 	coulmnames= new String[cols.length];
                                 	for (int i=0; i<coulmnames.length;i++ ){
                                 		coulmnames[i]=cols[i] + "";
                                 	}
                                 	fis = new FileInputStream(x);
                                 	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                                 	
                                 }
                            
                            int colcount=coulmnames.length;
                              columncount=colcount;
                              datasdouble = new double [rowcount][columncount]  ;
                              
                              int ro=0;
                             while ((line = br.readLine()) != null) {
                                     String[] tokens = line.split(delimeter,-1);
                                     for  (int i =0 ; i<cols.length; i ++ ) {
                                     	if (tokens[cols[i]].equals("") || tokens[cols[i]].equals("NA")){
                                     		tokens[cols[i]]=nullvalue;
                                     	}
                                     	datasdouble[ro][i]=Double.parseDouble(tokens[cols[i]]);
                                     }
                                     ro++;  }
                       //Close the buffer reader
                             br.close();
                             fis.close();
                     } catch (Exception e) {
                             e.printStackTrace();
                     }
   
                    
                   if ( verbose==true){
                     	 System.out.println(" The file" + x + " was loaded successfully with :");
                     	 System.out.println(" Rows : " + rowcount);
                     	 System.out.println(" Columns : " + columncount);
                     	 System.out.println(" Delimeter was  : " + delimeter);
                      }   
                   
             }     

		        /**
		         * 
		         * @param n : The file to 'Open'.
		         * <p> Method read and return vw predictions
		         */

						public static double [][]  getvowpalpreds(String n) {
		                	File x= new File(n);
		                	 String line;
		                	 double datasdoubles [][]= null;
		                     int rowcount=readcsv.getrowcount(n);   
		                     rowcount++;
	                
		                    new Vector<String>();
		                     try {
		                             FileInputStream fis = new FileInputStream(x);
		                             BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
		                             int ro=0;
		                             line = br.readLine();
		                             String[] tokens = line.split(" ",-1);
		                             int columncount=tokens.length;
		                             datasdoubles = new double [rowcount][columncount]  ;
		                             for  (int i =0 ; i<columncount; i ++ ) {
		                            	 datasdoubles[ro][i]=Double.parseDouble(tokens[i].split(":")[1]);
		                             }
		                             ro++;
		                            // Map the first Row
		                 

		                             while ((line = br.readLine()) != null) {
		                                      tokens = line.split(" ",-1);
		 		                             for  (int i =0 ; i<columncount; i ++ ) {
				                            	 datasdoubles[ro][i]=Double.parseDouble(tokens[i].split(":")[1]);
		                                     }
		                                     ro++;  }
		                       //Close the buffer reader
		                             br.close();
		                             fis.close();
		                     } catch (Exception e) {
		                             e.printStackTrace();
		                     }
		   
		                    
		                     return datasdoubles;
		                   
		             } 

/**
 * 
 * @param n : The file to 'Open'.
 * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
 * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
 * @param last : Import until the number of the column specified here, where 0 refers to first column
 * @param nullvalue : Replace null values with the String here.
 * @param hasheaders : True if we want the first line to be regarded as header
 * @param hasconsecutivedel : Regard consecutive delimiters as one.
 * @param verbose : Print details about the imported file.
 * <p> Method to read a file as a String array.
 */


		public void putfiletoarrayString(String n, String delimeter, int start, int last, String nullvalue,
        		boolean hasheaders,boolean hasconsecutivedel,boolean verbose) {
                String line="";
            	File x= new File(n);
                int columncount=0;
                int rowcount=readcsv.getrowcount(n);
                if (hasheaders==false){
                	rowcount++;
                }
                if (last==0){
                	last=Integer.MAX_VALUE;
                } 
                
               new Vector<String>();
                try {
                        FileInputStream fis = new FileInputStream(x);
                        BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                       // Map the first Row
                        
                        String [] coulmnamess;
                   	 
                	  	if (hasconsecutivedel){
                	  		coulmnamess = br.readLine().split(delimeter +"+",-1);
                            	} else{
                            		coulmnamess = br.readLine().split(delimeter,-1);
                            	}
                	  	
                       	if (last>coulmnamess.length){
                           	last=coulmnamess.length;
                           }
                        ArrayList<String> coll= new ArrayList<String>();
                        for (int i=start; i<last; i++){
                     	   coll.add(coulmnamess[i]);
                        }
                        if (hasheaders==true){
              
                        	
                            coulmnames = (String[])coll.toArray(new String[coll.size()]);
                            
                            
                            
                            } else{
                            	coulmnames= new String[coll.size()];
                            	for (int i=0; i<coll.size();i++ ){
                            		coulmnames[i]=(start+i) + "";
                            	}
                            	fis = new FileInputStream(x);
                            	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                            	
                            }
                        
                         columncount=coulmnames.length;
                         datas = new String [rowcount][columncount]  ;
                          int ro=0;
                        while ((line = br.readLine()) != null) {
                                String[] tokens = line.split(delimeter,-1);
                                int co=0;
                                for  (int i =start ; i<last; i ++ ) {
                                	if (tokens[i].equals("")){
                                		tokens[i]=nullvalue;
                                	}
                                	datas[ro][co]=(tokens[i]);
                                        co++;
                                }
                                ro++;  }
                  //Close the buffer reader
                        br.close();
                } catch (Exception e) {
                        e.printStackTrace();
                }
                

              if ( verbose==true){
                	 System.out.println(" The file" + x + " was loaded successfully with :");
                	 System.out.println(" Rows : " + rowcount);
                	 System.out.println(" Columns : " + columncount);
                	 System.out.println(" Delimeter was  : " + delimeter);
                 }  
              
        }
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param nullvalue : Replace null values with the String here.
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a String array.
         */

                public static String [] RetrieveString (String n, String delimeter,  boolean verbose) {
                        String line="";
                        ArrayList <String> arraysss= new   ArrayList <String>();
                    	File x= new File(n);
                    	
                        try {
                                FileInputStream fis = new FileInputStream(x);
                                BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                               
                                while ((line = br.readLine()) != null) {
                                	 String[] tokens = line.split(delimeter,-1);
                                     for  (int i =0 ; i<tokens.length; i ++ ) {
                                    	 arraysss.add(tokens[i]);
                                     }
                                        }
                          //Close the buffer reader
                                br.close();
                        } catch (Exception e) {
                                e.printStackTrace();
                        }
                        

                      if ( verbose==true){
                        	 System.out.println(" The file" + x + " was loaded successfully with :");
                         }  
                      
                      return arraysss.toArray(new String [arraysss.size()]);
                      
                }   
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one. 
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a double array.
         */     
        

		public void putfiletoarraydouble(String n, String delimeter, int start, int last, double nullvalue, 
        		boolean hasheaders, boolean hasconsecutivedel,boolean verbose) {
            String line;
            File x= new File(n);
            int columncount=0;
            int rowcount=readcsv.getrowcount(n);  
            if (hasheaders==false){
            	rowcount++;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
                        
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    
                    String [] coulmnamess;
               	 
            	  	if (hasconsecutivedel){
            	  		coulmnamess = br.readLine().split(delimeter +"+",-1);
                        	} else{
                        		coulmnamess = br.readLine().split(delimeter,-1);
                        	}
                    
                  	// String [] coulmnamess = (String[])columns.toArray(new String[columns.size()]);
                  	if (last>coulmnamess.length){
                      	last=coulmnamess.length;
                      }
                   ArrayList<String> coll= new ArrayList<String>();
                   for (int i=start; i<last; i++){
                	   coll.add(coulmnamess[i]);
                   }
                   if (hasheaders==true){
                       coulmnames = (String[])coll.toArray(new String[coll.size()]);
                       } else{
                       	coulmnames= new String[coll.size()];
                       	for (int i=0; i<coll.size();i++ ){
                       		coulmnames[i]=(start +i)+ "";
                       	}
                       	fis = new FileInputStream(x);
                       	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                       	
                       }
                   columncount=coulmnames.length;
                   datasdouble = new double [rowcount][columncount]  ;
                    int ro=0;
                    while ((line = br.readLine()) != null) {
                    	    line=line.replaceAll(delimeter +"+", delimeter);
                            String[] tokens = line.split(delimeter,-1);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datasdouble[ro][co]=Double.parseDouble(tokens[i]);
                            	co++;
                            	
                            }
                            ro++;  }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
   
          if ( verbose==true){
           	 System.out.println(" The file" + x + " was loaded successfully with :");
           	 System.out.println(" Rows : " + rowcount);
           	 System.out.println(" Columns : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
    }
        
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param column : column that represents the target
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a double array.
         */     
        

		public void putfractiontoarraydouble(String n, String delimeter, int start, int last,int column,double split, 
        		double nullvalue,boolean hasheaders,boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int columncount=0;
            int rowcount=readcsv.getrowcount(n);  
            if (hasheaders==false){
            	rowcount++;
            }
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
                        
           double fraction []= new double [rowcount];
           
           for (int i=0; i <fraction.length; i++ ){
        	   if (Math.random()<=split) {
        		   fraction [i]=1.0;
        	   }
           }
           double realcount=DescriptiveStatistics.getSum(fraction);
           
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
          
                   // Map the first Row

                    String [] coulmnamess;
                                       	 
                   if (hasconsecutivedel){
                   coulmnamess = br.readLine().split(delimeter,-1);
                  } else{
                 coulmnamess = br.readLine().split(delimeter,-1);
               }
                    
                    
                    
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	fis = new FileInputStream(x);
                      	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                   columncount=coulmnames.length;
                    datasdouble = new double [(int) realcount][columncount]  ;
                    target = new double [(int) realcount];
                    int ro=0;
                    int ko=0;
                    while ((line = br.readLine()) != null) {
                    	if   (fraction [ko]==1.0){
                            String[] tokens = line.split(delimeter,-1);
                            target[ro]=(Double.parseDouble(tokens[column]));
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datasdouble[ro][co]=((Double.parseDouble(tokens[i])));
                            	co++;
                            ;}
                            ro++;  }
                    	ko++;
                    	
                    }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
          if ( verbose==true){
           	 System.out.println(" The file" + x + " was loaded successfully with :");
           	 System.out.println(" Rows : " + (int)realcount);
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
    }
     
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a String array.
         */     
        

		public void putfractiontoarrayString(String n, String delimeter, int start, int last,double split, int seed,
        		String nullvalue,boolean hasheaders,boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int columncount=0;
            int rowcount=readcsv.getrowcount(n);  
            if (hasheaders==false){
            	rowcount++;
            }
            if (verbose){
            System.out.println("RowCount: " + rowcount);
            }
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
                        
           double fraction []= new double [rowcount];
           Random rand= new Random(seed);
           for (int i=0; i <fraction.length; i++ ){
        	   if (rand.nextDouble()<=split) {
        		   fraction [i]=1.0;
        	   }
           }
           double realcount=DescriptiveStatistics.getSum(fraction);
           
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                   // Map the first Row

                    String [] coulmnamess;
                                       	 
                   if (hasconsecutivedel){
                   coulmnamess = br.readLine().split(delimeter +"+",-1);
                  } else{
                 coulmnamess = br.readLine().split(delimeter,-1);
               }
                   
                    
                    
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	fis = new FileInputStream(x);
                      	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                   columncount=coulmnames.length;
                   datas = new String [(int) realcount][columncount]  ;
                    int ro=0;
                    int ko=0;
                    while ((line = br.readLine()) != null) {
                    	if   (fraction [ko]==1.0){
                            String[] tokens = line.split(delimeter,-1);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datas[ro][co]=(tokens[i]);
                            	co++;
                            ;}
                            ro++;  }
                    	ko++;
                    	
                    }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
          if ( verbose==true){
           	 System.out.println(" The file" + x + " was loaded successfully with :");
           	 System.out.println(" Rows : " + (int)realcount);
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
    }
        
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param linedelimeter : It accounts for line delimiter as well 
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a String array.
         */     
        
        @SuppressWarnings("resource")
		public void putfractiontoarrayString(String n, String delimeter, String linedelimeter, int start, int last,double split, int seed,
        		String nullvalue,boolean hasheaders,boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int columncount=0;
            int rowcount=0;
            try{
            FileInputStream fiss = new FileInputStream(x);
            BufferedReader brs = new BufferedReader(new InputStreamReader(fiss,"UTF-8"));
          
            int ch;
            while((ch = brs.read()) >= 0) {
                if(ch == '\r' && verbose) {
                	rowcount++;
                	if (((double)rowcount)%100000.00==0){ 
                	System.out.println(rowcount);
                	}
                }
            }
            }catch (Exception e){}
            rowcount--;
            
          
            if (hasheaders==false){
            	rowcount++;
            }
            if (verbose){
            System.out.println("RowCount: " + rowcount);
            }
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
                        
           double fraction []= new double [rowcount];
           Random rand= new Random(seed);
           for (int i=0; i <fraction.length; i++ ){
        	   if (rand.nextDouble()<=split) {
        		   fraction [i]=1.0;
        	   }
           }
           double realcount=DescriptiveStatistics.getSum(fraction);
           
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                   // Map the first Row

                    String [] coulmnamess;
                                       	 
                   if (hasconsecutivedel){
                       int ch;
                       StringBuilder sb = new StringBuilder();
                       while((ch = br.read()) >= 0) {
                           if(ch == '\r') {
                           break;
                           }
                           sb.append((char) ch);
                       } 
                       String kati=sb.toString().replaceAll("\n", " ");
                       kati=kati.replaceAll("\t", " ");
                       kati=kati.replaceAll("\"", " ");
                       coulmnamess=kati.split(delimeter +"+",-1);
                	   
 
                  } else{
                      int ch;
                      StringBuilder sb = new StringBuilder();
                      while((ch = br.read()) >= 0) {
                          if(ch == '\r') {
                          break;
                          }
                          sb.append((char) ch);
                      } 
                      String kati=sb.toString().replaceAll("\n", " ");
                      kati=kati.replaceAll("\t", " ");
                      kati=kati.replaceAll("\"", " ");
                      coulmnamess=kati.split(delimeter,-1);
               }
                   
                    
                    
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	fis = new FileInputStream(x);
                      	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                   columncount=coulmnames.length;
                   datas = new String [(int) realcount][columncount]  ;
                    int ro=0;
                    int ko=0;
                    int ch;
                    StringBuilder sb = new StringBuilder();
                    while((ch = br.read()) >= 0) {
                    	 sb.append((char) ch);
                    	 if(ch == '\r') {
                    		 line=sb.toString();
                    		 line=line.replaceAll("\n", " ");
                    		 line= line.replaceAll("\t", " ");
                    		 line= line.replaceAll("\"", "");
                    	if   (fraction [ko]==1.0){
                            String[] tokens = line.split(delimeter,-1);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datas[ro][co]=(tokens[i]);
                            	co++;
                            ;}
                            ro++;  }
                    	ko++;
                    	sb = new StringBuilder();
                    	 }
                    	
                    }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
          if ( verbose==true){
           	 System.out.println(" The file" + x + " was loaded successfully with :");
           	 System.out.println(" Rows : " + (int)realcount);
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
    }
        
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param filetoprint : File to print
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a String array.
         */     
        
		public void putfractiontoarrayStringandprint(String n, String filetoprint,String delfile, int start, int last,double split, int seed,
        		String nullvalue,boolean hasheaders,boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int columncount=0;
            int ro=0;
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
                        
           
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                   // Map the first Row

                    String [] coulmnamess;
                                       	 
                   if (hasconsecutivedel){
                       int ch;
                       StringBuilder sb = new StringBuilder();
                       while((ch = br.read()) >= 0) {
                           if(ch == '\r') {
                           break;
                           }
                           sb.append((char) ch);
                       } 
                       String kati=sb.toString().replaceAll("\n", " ");
                       kati=kati.replaceAll("\t", " ");
                       kati=kati.replaceAll("\"", " ");
                       coulmnamess=kati.split("," +"+",-1);
                	   
 
                  } else{
                      int ch;
                      StringBuilder sb = new StringBuilder();
                      while((ch = br.read()) >= 0) {
                          if(ch == '\r') {
                          break;
                          }
                          sb.append((char) ch);
                      } 
                      String kati=sb.toString().replaceAll("\n", " ");
                      kati=kati.replaceAll("\t", " ");
                      kati=kati.replaceAll("\"", " ");
                      coulmnamess=kati.split(",",-1);
               }
                   

					FileWriter writer = new FileWriter(filetoprint);
			
					 /*
				//import the rest of the data-------------------	 
					 for (int j= 0; j<all_data.length;j++){
						 writer.append(all_data[j][0] +"");
						// System.out.println(all_data[j][0] + " " + j + " " + 0);
					 for (int i= 1; i<all_data[0].length;i++){
						 writer.append(delimeter);
						 writer.append(all_data[j][i] +"");
						 */
                    
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	fis = new FileInputStream(x);
                      	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                   columncount=coulmnames.length;
                    int ch;
                    StringBuilder sb = new StringBuilder();
                    while((ch = br.read()) >= 0) {
                    	 sb.append((char) ch);
                    	 if(ch == '\r') {
                    		 //System.out.println("row: " + ro);
                    		 line=sb.toString();
                    		 line= line.replaceAll("\t", " ");
                    		 line=line.replaceAll("\n", " ");
                    		 line= line.replaceAll("\"", "");
                            String[] token = line.split(",",-1);
                            String tokens []= new String [3];
                            tokens[0]=token[0];
                            tokens[2]=token[token.length-1];
                            for(int i=1; i<token.length-1; i++ ){
                            	tokens[1]=tokens[1]+token[i];
                            }
                            //System.out.println(tokens[tokens.length-1]);
                            writer.append(tokens[0]);
                            for  (int i =1 ; i<3; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	writer.append(delfile);
                            	writer.append(tokens[i]);
                            ;}
                            ro++;  
                            if( verbose) {
                            	if (((double)ro)%100000.00==0){ 
                            	System.out.println(ro);
                            	}
                            }
                    	writer.append("\n");
                    	sb = new StringBuilder();
                    	 }
                    	
                    }
              //Close the buffer reader
                    br.close();
                    writer.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
          if ( verbose==true){
           	 System.out.println(" The file" + x + " was loaded successfully with :");
           	 System.out.println(" Rows : " + ro+"");
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + ",");
            }  
            
    }
        
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param filetoprint : File to print
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a String array.
         */     

		public void putfractiontoarrayStringandprint2(String n, String filetoprint,String delfile, int start, int last,double split, int seed,
        		String nullvalue,boolean hasheaders,boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int columncount=0;
            int ro=0;
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
                        
           
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                   // Map the first Row

                    String [] coulmnamess;
                                       	 
                   if (hasconsecutivedel){
                       int ch;
                       StringBuilder sb = new StringBuilder();
                       while((ch = br.read()) >= 0) {
                           if(ch == '\r') {
                           break;
                           }
                           sb.append((char) ch);
                       } 
                       String kati=sb.toString().replaceAll("\n", " ");
                       kati=kati.replaceAll("\t", " ");
                       kati=kati.replaceAll("\"", " ");
                       coulmnamess=kati.split("," +"+",-1);
                	   
 
                  } else{
                      int ch;
                      StringBuilder sb = new StringBuilder();
                      while((ch = br.read()) >= 0) {
                          if(ch == '\r') {
                          break;
                          }
                          sb.append((char) ch);
                      } 
                      String kati=sb.toString().replaceAll("\n", " ");
                      kati=kati.replaceAll("\t", " ");
                      kati=kati.replaceAll("\"", " ");
                      coulmnamess=kati.split(",",-1);
               }
                   

					FileWriter writer = new FileWriter(filetoprint);
			
					 /*
				//import the rest of the data-------------------	 
					 for (int j= 0; j<all_data.length;j++){
						 writer.append(all_data[j][0] +"");
						// System.out.println(all_data[j][0] + " " + j + " " + 0);
					 for (int i= 1; i<all_data[0].length;i++){
						 writer.append(delimeter);
						 writer.append(all_data[j][i] +"");
						 */
                    
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	fis = new FileInputStream(x);
                      	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                   columncount=coulmnames.length;
                    int ch;
                    StringBuilder sb = new StringBuilder();
                    while((ch = br.read()) >= 0) {
                    	 sb.append((char) ch);
                    	 if(ch == '\r') {
                    		 //System.out.println("row: " + ro);
                    		 line=sb.toString();
                    		 line= line.replaceAll("\t", " ");
                    		 line=line.replaceAll("\n", " ");
                    		 line= line.replaceAll("\"", "");
                            String[] token = line.split(",",-1);
                            String tokens []= new String [2];
                            tokens[0]=token[0];
                            for(int i=1; i<token.length; i++ ){
                            	tokens[1]=tokens[1]+token[i];
                            }
                            //System.out.println(tokens[tokens.length-1]);
                            writer.append(tokens[0]);
                            for  (int i =1 ; i<2; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	writer.append(delfile);
                            	writer.append(tokens[i]);
                            ;}
                            ro++;  
                            if( verbose) {
                            	if (((double)ro)%100000.00==0){ 
                            	System.out.println(ro);
                            	}
                            }
                    	writer.append("\n");
                    	sb = new StringBuilder();
                    	 }
                    	
                    }
              //Close the buffer reader
                    br.close();
                    writer.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
          if ( verbose==true){
           	 System.out.println(" The file" + x + " was loaded successfully with :");
           	 System.out.println(" Rows : " + ro+"");
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + ",");
            }  
            
    }
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param column : column that represents the target
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.  
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a double array.
         */     

		public void putfwholefile(String n, String delimeter, int start, int last,int column,
        		double nullvalue,boolean hasheaders, boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int colcount=0;
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
            int columncount=0;            
    
            int rowcount=readcsv.getrowcount(n);

            if (hasheaders==false){
                           	rowcount++;
                           }
            
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                 
                   // Map the first Row
                    String [] coulmnamess;
               	 
            	  	if (hasconsecutivedel){
            	  		coulmnamess = br.readLine().split(delimeter +"+",-1);
                        	} else{
                        		coulmnamess = br.readLine().split(delimeter,-1);
                        	}
            	  	
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	fis = new FileInputStream(x);
                      	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                    colcount=coulmnames.length;
                     columncount=colcount;
                    datasdouble = new double [rowcount][columncount]  ;
                    target = new double [rowcount];
                    int ro=0;
                    while ((line = br.readLine()) != null) {
                            String[] tokens = line.split(delimeter,-1);
                            target[ro]=Double.parseDouble(tokens[column]);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datasdouble[ro][co]=Double.parseDouble(tokens[i]);
                            	co++;
                            }
                          
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
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
    }  
        
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param column : column that represents the target
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.  
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a double array.
         */     

		public void putfwholefileString(String n, String delimeter, int start, int last,int column,
        		double nullvalue,boolean hasheaders, boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int colcount=0;
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }

            if (last==0){
            	last=Integer.MAX_VALUE;
            }
            int columncount=0;            
    
            int rowcount=readcsv.getrowcount(n);

            if (hasheaders==false){
                           	rowcount++;
                           }
            
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                 
                   // Map the first Row
                    String [] coulmnamess;
               	 
            	  	if (hasconsecutivedel){
            	  		coulmnamess = br.readLine().split(delimeter +"+",-1);
                        	} else{
                        		coulmnamess = br.readLine().split(delimeter,-1);
                        	}
            	  	
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	fis = new FileInputStream(x);
                      	br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                    colcount=coulmnames.length;
                     columncount=colcount;
                     datas = new String [rowcount][columncount]  ;
                    target = new double [rowcount];
                    int ro=0;
                    while ((line = br.readLine()) != null) {
                            String[] tokens = line.split(delimeter,-1);
                            target[ro]=Double.parseDouble(tokens[column]);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datas[ro][co]=tokens[i];
                            	co++;
                            }
                          
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
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
    }  
              
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param column : column that represents the target
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.  
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a double array.
         */     
        
        public void putfwholefilefromurl(  URL url, String delimeter, int start, int last,int column,
        		double nullvalue,boolean hasheaders, boolean hasconsecutivedel, boolean verbose) {
            String line;
            int colcount=0;
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
            int columncount=0;            
    
            int rowcount=readcsv.getrowcount(url);

            if (hasheaders==false){
                           	rowcount++;
                           }
            
           new Vector<String>();
            try {
                    BufferedReader br = new BufferedReader(new InputStreamReader(url.openStream(),Charset.forName("ISO-8859-7")));
                
                   // Map the first Row
                    String [] coulmnamess;
               	 
            	  	if (hasconsecutivedel){
            	  		coulmnamess = br.readLine().split(delimeter +"+",-1);
                        	} else{
                        		coulmnamess = br.readLine().split(delimeter,-1);
                        	}
            	  	
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
                	  byte ptext[] = coulmnamess[i].getBytes();
                	  String value = new String(ptext, "UTF-8");
               	   coll.add(value);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}
                      	br = new BufferedReader(new InputStreamReader(url.openStream(),"UTF-8"));
                      	
                      }
                    colcount=coulmnames.length;
                     columncount=colcount;

                    datas= new String [rowcount][columncount]  ;
                    int ro=0;
                    while ((line = br.readLine()) != null) {
                            String[] tokens = line.split(delimeter,-1);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datas[ro][co]=tokens[i];
                            	co++;
                            }
                          
                    	ro++;
                    }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            

          if ( verbose==true){
           	 System.out.println(" The url " + url.toString() + " was loaded successfully with :");
           	 System.out.println(" Rows : " + rowcount);
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
    }  
    
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param column : column that represents the target
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param hasconsecutivedel : Regard consecutive delimiters as one.
         * @param verbose : Print details about the imported file.
         * <p> Method to read a file as a double array.
         */     

		public static double [][] putfwholefiletodouble(String n, String delimeter, int start, int last,int column,
        		double nullvalue,boolean hasheaders,boolean hasconsecutivedel, boolean verbose) {
            String line;
            File x= new File(n);
            int colcount=0;
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }
            if (last==0){
            	last=Integer.MAX_VALUE;
            }
            int columncount=0;            
    
            int rowcount=readcsv.getrowcount(n);

            if (hasheaders==false){
                           	rowcount++;
                           }
            double [][] datasdoubles = new double [1][1];
            
           new Vector<String>();
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                   // Map the first Row
                    

                    String [] coulmnamess;
                                       	 
                   if (hasconsecutivedel){
                   coulmnamess = br.readLine().split(delimeter +"+",-1);
                  } else{
                  coulmnamess = br.readLine().split(delimeter,-1);
                  }
                    
                 	if (last>coulmnamess.length){
                     	last=coulmnamess.length;
                     }
                  ArrayList<String> coll= new ArrayList<String>();
                  for (int i=start; i<last; i++){
               	   coll.add(coulmnamess[i]);
                  }
                  if (hasheaders==true){
                      coulmnames = (String[])coll.toArray(new String[coll.size()]);
                      } else{
                         	coulmnames= new String[coll.size()];
                        	for (int i=0; i<coll.size();i++ ){
                        		coulmnames[i]=(start+i) + "";
                        	}

                     fis = new FileInputStream(x);
                      br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                      	
                      }
                    colcount=coulmnames.length;
                     columncount=colcount;
                     datasdoubles = new double [rowcount][columncount]  ;
                    label = new double [rowcount];
                    int ro=0;
                    while ((line = br.readLine()) != null) {
                    	//System.out.println("row: " + ro);
                            String[] tokens = line.split(delimeter,-1);
                            label[ro]=Double.parseDouble(tokens[column]);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("") ||tokens[i].equals("NA") ||tokens[i].isEmpty()){
                            		tokens[i]=nullvalue +"";
                            	}
                            	try{
                            	datasdoubles[ro][co]=Double.parseDouble(tokens[i]);
                            	} catch (Exception e){
                            		datasdoubles[ro][co]=Double.parseDouble(nullvalue +"");
                            	}
                            	co++;
                            }
                          
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
           	 System.out.println(" Columns (excluding target) : " + columncount);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          
          return datasdoubles;
          
    }          
      
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param start : Import after the number of the column specified here (including this one) , where 0 refers to first column
         * @param last : Import until the number of the column specified here, where 0 refers to first column
         * @param row : row to bring back 
         * @param column : column that represents the target
         * @param split : Refers to the percentage of file that we want to retrieve (e.g. 0.1= 10% of the file)
         * @param nullvalue : Replace null values with the double here.
         * @param verbose : Print details about the imported file.
         * @return A line as double array 
         * <p> Method to read a file and bring a specific line back as double array[].
         */     
        
        public static double [] Importsingleline(String n, String delimeter, int start, int last,int row, double nullvalue, boolean verbose) {
            String line = "";
            File x= new File(n);
            //start cannot be 0 as this is the location of the target
            if (start>=last){
            	start=0;
            }    
            
            double [] datarow = new double [last-start];
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    br.readLine();
                   // Map the first Row
                    int ro=-1;
                    while (ro !=row) {
                    	line= br.readLine();
                    	 ro++;
                    }
                            String[] tokens = line.split(delimeter,-1);
                            int co=0;
                            for  (int i =start ; i<last; i ++ ) {
                            	if (tokens[i].equals("")){
                            		tokens[i]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	datarow[co]=Double.parseDouble(tokens[i]);
                            	co++;
                            }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
          if ( verbose==true){
           	 System.out.println(" The file" + x + " was loaded successfully with :");
           	 System.out.println(" Row : " + row);
           	 System.out.println(" Delimeter was  : " + delimeter);
            }  
          return datarow; 
    }         
        
        
        
        /**
         * 
         * @return The label variable in a static context
         */
        public double [] getlabelstatic(){
        	return label;
        }
        

  
        
        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param col : The column to retrieve starting from 0. Ay value lower than zero gives 0.
         * @param nullvalue : Replace null values with the double here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param verbose : Print details about the imported file.
         * @return A column as double array 
         * <p> Method to read a file and retrieve the specified column as double array
         * . We assume the first row is the headers and is excluded.
         */     
        
        public static double[]Retrievecolumn(String n, String delimeter, int col,
        		double nullvalue,boolean hasheaders, boolean verbose) {
            String line;
            File x= new File(n);
            
            int rowcount=readcsv.getrowcount(n); 
            if (hasheaders==false){
            	rowcount++;
                }
            //If less than zero, ammend to zzero
            if (col<0){
            	col=0;
            }     
            int ro=0;
            double column [] = new double [rowcount]  ;
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    if (hasheaders){
                    br.readLine();
                    }
                    
                    while ((line = br.readLine()) != null) {
                            String[] tokens = line.split(delimeter,-1);

                            	if (tokens[col].equals("") ||tokens[col].equals("NA") ){
                            		tokens[col]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	column[ro]=((Double.parseDouble(tokens[col])));
                            
                                    ro++; }

              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("error at: " + ro);
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
         * @return A column as double array 
         * <p> Method to read a file and retrieve the specified column as int array
         */     
        
        public static int[]Retrievecolumnint(String n, String delimeter, int col,
        		int nullvalue,boolean hasheaders, boolean verbose) {
            String line;
            File x= new File(n);
            
            int rowcount=readcsv.getrowcount(n); 
            if (hasheaders==false){
            	rowcount++;
                }
            //If less than zero, ammend to zzero
            if (col<0){
            	col=0;
            }     
            int ro=0;
            int column [] = new int [rowcount]  ;
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    if (hasheaders){
                    br.readLine();
                    }
                    
                    while ((line = br.readLine()) != null) {
                            String[] tokens = line.split(delimeter,-1);

                            	if (tokens[col].equals("") ||tokens[col].equals("NA") ){
                            		tokens[col]=nullvalue +"";
                            		//System.out.println("change at " + line_count +" line at " + i);
                            	}
                            	column[ro]=(int)((Double.parseDouble(tokens[col])));
                            
                                    ro++; }

              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("error at: " + ro);
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
         * <p> Method to read a file and retrieve Kfolds
         */     
        
        public static int[][][] get_kfolder(String n) {
            String line;
            File x= new File(n);
            
            int rowcount=readcsv.getrowcount(n); 
            rowcount++;
            
    		//IntIntMapminus4a row_Values=new IntIntMapminus4a(rowcount, 0.5F);
			// check if values only 1 and zero
			HashSet<Integer> row_Values= new HashSet<Integer> ();
			/*
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
    		*/
    		/*
    		for (int b=start; b < end ;b++ ){
    			column_Values.put(data.mainelementpile[b], b);
    		}
    		*/
            int ro=0;
            int all_cases [] = new int [rowcount]  ;
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    
                    while ((line = br.readLine()) != null) {
                    	int cv_index=0;
        	    		try{
        	    			cv_index=Integer.parseInt(line);
        	    		}catch (Exception e){
        	    			throw new IllegalStateException(" file  " + n + " needs to contain only integers");	
        	    		}
        	    		row_Values.add(cv_index);
        	    		all_cases[ro]=cv_index;
        	    		ro++;
        	    		}
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("error at: " + ro);
            }
            if (row_Values.size()<=1){
            	throw new IllegalStateException(" There needs to be more than 1 unique values to generated train and test splits."  );	
            }
			int uniquevalues[]= new int[row_Values.size()];
			
			int k=0;
		    for (Iterator<Integer> it = row_Values.iterator(); it.hasNext(); ) {
		    	uniquevalues[k]= it.next();
		    	k++;
		    	}
		    // sort values
		    Arrays.sort(uniquevalues); 
		    int kfolder [][][]= new int [uniquevalues.length][2][];
		    for (int s=0; s < uniquevalues.length; s++){
		    	int unique_value=uniquevalues[s];
		    	int train_count=0;
		    	int test_count=0;
		    	for (int i=0; i < all_cases.length; i++){
		    		if (all_cases[i]==unique_value){
		    			test_count++;
		    		}else {
		    			train_count++;
		    		}
		    	}
		    	int train_cases [] = new int [train_count];		
		    	int test_cases [] = new int [test_count];	
		    	train_count=0;
		    	test_count=0;		
		    	for (int i=0; i < all_cases.length; i++){
		    		if (all_cases[i]==unique_value){
		    			test_cases[test_count]=i;
		    			test_count++;
		    		}else {
		    			train_cases[train_count]=i;
		    			train_count++;
		    		}		    		    		
		    	
		    	}
	    		kfolder[s][0]=train_cases;
	    		kfolder[s][1]=test_cases;	
            
		    }
          
          return kfolder;
    }

        /**
         * 
         * @param n : The file to 'Open'.
         * @param delimeter : the type of delimiter to split the file in different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param col : The column to retrieve starting from 0. if value lower than zero, gives 0.
         * @param nullvalue : Replace null values with the String here.
         * @param hasheaders : True if we want the first line to be regarded as header
         * @param verbose : Print details about the imported file.
         * @return A column as double array 
         * <p> Method to read a file and retrieve the specified column as double array
         * . We assume the first row is the headers and is excluded.
         */     
        
        public static String[] Retrievecolumn(String n, String delimeter, int col, String nullvalue
        		,boolean hasheaders, boolean verbose) {
            String line;
            File x= new File(n);
            int rowcount=readcsv.getrowcount(n); 
            if (hasheaders==false){
            	rowcount++;
                }
            //If less than zero, ammend to zzero
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
                            		//System.out.println("change at " + line_count +" line at " + i);
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
         * <p> Default constructor for importing a file as String array  that reads all the file
         * @param x : The file to 'Open'.
         * @param delimeter : the type of delimeter to slit the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         *
         */
     
        public void putfiletoarrayString(String x, String delimeter) {
        	this.putfiletoarrayString( x,  delimeter, 0, 0, "null",false,true, false);
        }
        
        /**
         * <p> Default constructor for importing a file as String array  that reads all the file and assumes the delimeter to be comma ','.
         * @param x : The file to 'Open'.
         */
     
        public void putfiletoarrayString(String x) {
        	this.putfiletoarrayString( x,  ",", 0, 0, "null",false,true, false);
        }
        
        /**
         * <p> Default constructor for importing a file as double array  that reads all the file and assumes the delimeter to be comma ','.
         * @param x : The file to 'Open'.
         */
     
        public void putfiletoarraydouble(String x) {
        	this.putfiletoarraydouble( x,  ",", 0, 0, 0.0,false,true, false);
        }
        
        /**
         * <p> Default constructor for importing a file double array that reads all the file
         * @param x : The file to 'Open'.
         * @param delimeter : the type of delimeter to slit the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * <p> Method to read a file as a double array.
         */     
        
        public void putfiletoarraydouble(String x, String delimeter) {
        	
        	this.putfiletoarraydouble( x,  delimeter, 0, 0, 0.0,false,true, false);
        	
        }
        
        
       
 /**
  * 
  * @return : The String Array if the file was loaded as String.
  */
        public String [][] getdatasetString(){
        	return datas;
        }
        
        /**
         * 
         * @return : The double Array if the file was loaded as String.
         */
               public double [][] getdatasetdouble(){
               	return datasdouble;
               }
        /**
         * 
         * @return : The column names as an array.
         */
        public static String [] getcolumnnames(){
        	return coulmnames;
        }
        
        /**
         * 
         * @return the table array representing the target of a file using the putfractiontoarraydouble method
         */
        public double[] gettarget(){
        	return target;
        }
            
        /**
         * <p> Returns a specific column as double from reading a file
         * @param m : The file to import.
         * @param delimeter : the type of delimeter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param x : The column we want to import as double array.
         * @param nullvalue : Replace null values with the String here.
         * @return : Single double array.
         */
        public static double[] getcollumnasdouble(String m, String delimeter ,int x, String nullvalue) {
        	int column []={x};
        	readcsv read= new readcsv();
        	read.putfiletoarraydouble(m, delimeter, column, nullvalue,false,true, false);
        	double arraydouble [][]=read.getdatasetdouble();
        		double xcoll[]= new double [arraydouble.length];
        		for (int i=0; i < xcoll.length; i++){
        			xcoll[i]=arraydouble[i][0];
        		}
        		return xcoll;
        }
      
        /**
         * <p>Default Constructor that returns a specific column as double from reading a file
         * @param m : The file to import.
         * @param x : The column we want to import as double array.
         * @return : Single double array.
         */
        public static double[] getcollumnasdouble(String m ,int x) {
        	int column []={x};
        	readcsv read= new readcsv();
        	read.putfiletoarraydouble(m, ",", column, "null",false,true, false);
        	double arraydouble [][]=read.getdatasetdouble();
        		double xcoll[]= new double [arraydouble.length];
        		for (int i=0; i < xcoll.length; i++){
        			xcoll[i]=arraydouble[i][0];
        		}
        		return xcoll;
        }
        
        
        
        
        /**
         * <p> Returns a specific column as String from reading a file
         * @param m : The file to import.
         * @param delimeter : the type of delimeter to split the file int different columns. Typical ones would be commas, tabs, semicolons etc.
         * @param x : The column we want to import as double array.
         * @param nullvalue : Replace null values with the String here.
         * @return : Single String array.
         */
        public static String[] getcollumasString(String m, String delimeter ,int x, String nullvalue) {
        	int column []={x};
        	readcsv read= new readcsv();
        	read.putfiletoarrayString(m, delimeter, column, nullvalue,false,true, false);
        	String arraystring [][]=read.getdatasetString();
        		String xcoll[]= new String [arraystring.length];
        		for (int i=0; i < xcoll.length; i++){
        			xcoll[i]=arraystring[i][0];
        		}
        		return xcoll;
        }
        
        /**
         * <p>Default Constructor that returns a specific column as String from reading a file
         * @param m : The file to import.
         * @param x : The column we want to import as double array.
         * @return : Single String array.
         */
        public static String[] getcollumasString(String m ,int x) {
        	int column []={x};
        	readcsv read= new readcsv();
        	read.putfiletoarrayString(m, ",", column, "null",false,true, false);
        	String arraystring [][]=read.getdatasetString();
    		String xcoll[]= new String [arraystring.length];
    		for (int i=0; i < xcoll.length; i++){
    			xcoll[i]=arraystring[i][0];
    		}
    		return xcoll;

        }
        
        /**
         * @param File  The file to Open
         * @return Scans the given file and brings back the row count
         */
        public static int getrowcount(String File){
        	File x= new File(File);
            int line_count=0;
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    br.readLine();
                    while ((br.readLine()) != null) {
                            line_count=line_count+1;  }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
        return line_count;  
    }
        /**
         * 
         * @param url  The file to Open
         * @return Scans the given file and brings back the row count
         */
        public static int getrowcount(URL url){
        	int line_count=0;
            try {

                    BufferedReader br = new BufferedReader(new InputStreamReader(url.openStream(),"UTF-8"));
                    br.readLine();
                    while ((br.readLine()) != null) {
                            line_count=line_count+1;  }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
        return line_count ;
        
        }
        /**
         * 
         * @param File  The file to Open
         * @return Scans the given file and brings back the row count
         */
        public static int getrowcountnocol(String File){
        	File x= new File(File);
            int line_count=0;
            try {
                    FileInputStream fis = new FileInputStream(x);
                    BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    while ((br.readLine()) != null) {
                            line_count=line_count+1;  }
              //Close the buffer reader
                    br.close();
            } catch (Exception e) {
                    e.printStackTrace();
            }
            
 
        return line_count;  
    }
           
        /* End of class*/
        }

