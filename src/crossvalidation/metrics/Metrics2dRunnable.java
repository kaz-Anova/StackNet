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


package crossvalidation.metrics;
import java.util.Arrays;

import exceptions.IllegalStateException;

/**
 * 
 * @author marios
 *<p> Runnable methods for metrics </p>
 */
public class Metrics2dRunnable implements Runnable {
    /**
     * This refers to the actual or label of the data 
     */
	public double actual [][];
	/**
	 * predictions' data
	 */
	public double pred [][];
	/**
	 * array where to hold the results
	 */
	public double  results [];
    /**
    * column to use 
    */
	int column=-1;
	/**
	 * The type in rmse,mae,auc,log,cat,pre,rec,f,cor
	 */
	String type="";
  /**
   * 
   * @param actual : the actual values
   * @param pred : the predicted values
   * @param results  : where to put the metric
   * @param column  : column to use for the array
   * @param type  : the type in  rmse,mae,auc,log,cat,pre,rec,f,cor
   */
	public Metrics2dRunnable(double actual [][],  double pred [][],double results [] , int column,String type) {
		this.actual=actual;
		this.pred=pred;
		this.results=results;
		this.type=type;
		this.column=column;
		
	}
	
	@Override
	public void run() {
		//sensible checks
		if (actual==null || pred==null ||
				actual.length==0 || pred.length==0 || 
				actual.length!=pred.length ||
				column<0 || column>=actual[0].length ||
				results==null || results.length!=actual[0].length){
			throw new IllegalStateException(" There is an error with the state of actual and pred in terms of length ");
		}

	     //------------------Root Mean Squared Error rmse------------------//
         if (type.equals("rmse")){ 
            
     		double errorrmse=0;
    		for (int i=0; i <pred.length; i++ ) {
    			errorrmse+=(actual[i][column]-pred[i][column])*(actual[i][column]-pred[i][column]);
    		}
    		errorrmse=Math.sqrt(errorrmse/pred.length);
    		results[column]=errorrmse;
         }
         else if (type.equals("mae")){  
        	
     		double errormae=0;
    		for (int i=0; i <pred.length; i++ ) {
    			errormae+=Math.abs(actual[i][column]-pred[i][column]);
    		}
    		results[column]=errormae/actual.length;
         } else if (type.equals("auc")){      	 
			 // find distinct values
		    double [] distinct_values =manipulate.distinct.distinct.getdoubleDistinctset(actual,column);
		    int Targetsize=  distinct_values.length;
		    // Throw exception if the size is not 2
		    if (Targetsize!=2) {
					throw new IllegalStateException("Your array needs to be binary (e.g to have 2 disticnt values like 0 and 1).");
				}	 

		    //find smallest and highest values
		    Arrays.sort(distinct_values);
		    double low=distinct_values[0];
		    double high=distinct_values[1];	
		    //make compies
		    double predictedvalues []=manipulate.select.columnselect.ColumnSelect(pred,column);
		    double target []=manipulate.select.columnselect.ColumnSelect(actual,column);
		    // Sort based on predictive value
		    manipulate.sort.mergesorts.mergesort(predictedvalues,target,0,predictedvalues.length-1 );
		    double Sum_bad=0;
		    double Sum_good=0;	
		    double firstc=0;
		    double secon=0;
		    for (int i=0; i< predictedvalues.length; i++) {
       	         if(low==target[i]) 
       	         { firstc++;} 
       	         else 
       	         {secon++;}
       	         }
		    //Calculate total combinations for the AUC
		    double combination= firstc*secon;
		    double good= (int)secon;
		    //counters
		    int counter=0;
		    int counter2=0;
		    //sum-ers
		    double goodss= good;
		    double big_sum=0;		
		    // here starts the big while
		    while (counter2<target.length) {
		    	double sum=0;
		    	int other_count=0;
		    	while (counter2<target.length && predictedvalues[counter]==predictedvalues[counter2] ){
		    		other_count=other_count+1;
		    		if (target[counter2]==high){
		    			sum+=1;	
		    		}
		    		counter2++;
		    	}			
		    	if (sum==0) {
		    		big_sum=big_sum+ (other_count * goodss);
		    		Sum_bad=Sum_bad+other_count;
		    	} else {
		    		Sum_bad= (Sum_bad+ other_count- sum);
		    		Sum_good= (Sum_good +sum);
		    		big_sum=big_sum+ ( (other_count-sum) * (goodss-sum) ) + (((other_count-sum) *(sum)) /2);	
		    	}

		    	goodss=goodss-sum;
		    	counter=counter2;
		    }
		    // the AUC
		    distinct_values=null;
		    predictedvalues=null;
		    target=null;
		    
		    results[column]=(big_sum/combination);
		    
         }
		 //deviance
         else if (type.equals("log")){    
     		double errorlog=0;
    		double len=pred.length;
    		double [] distinct_valueslog =manipulate.distinct.distinct.getdoubleDistinctset(actual,column);
		    int Targetsizelog=  distinct_valueslog.length;
		    // Throw exception if the size is not 2
		    if (Targetsizelog!=2) {
					throw new IllegalStateException("Your array needs to be binary (e.g to have 2 disticnt values like 0 and 1).");
				}	
    		if (distinct_valueslog[0]!=0.0 && distinct_valueslog[1]!=1.0){
    			throw new IllegalStateException("Your array needs to be binary and have the values of 0 and 1");
    		}
    		for (int i=0; i <pred.length; i++ ) {
    			double value=pred[i][column];
    			if (value>1.0-(1E-14)){
    				value=1.0-(1E-14);
    			} else if (value<0+(1E-14)){
    				value=0.0+(1E-14);
    			}
    			if (actual[i][column]==0){
    				errorlog-=(1-actual[i][column]) * Math.log(1-value);
    			} else {
    				errorlog-=actual[i][column]*Math.log(value) ;
    			}
    		
    		}
		    distinct_valueslog=null;

    		  results[column]=errorlog/len; 
    	 //categorisation accuracy	  
            }
    		else if (type.equals("cat")){    
     		double classification_accuracy=0;
    		
    		for (int i=0; i <pred.length; i++ ) {
    			if (pred[i]==actual[i]){
    				classification_accuracy++;
    			}
    		}
    		results[column]=classification_accuracy/pred.length;    
    		}
    	   //precision	
    		else if (type.equals("pre")){
      		double precision=0;
      		double totalprecision=0.0;
     		double [] distinct_valuesprecision =manipulate.distinct.distinct.getdoubleDistinctset(actual,column);
 		    int Targetsizeprecision=  distinct_valuesprecision.length;
 		    // Throw exception if the size is not 2
 		    if (Targetsizeprecision!=2) {
 					throw new IllegalStateException("Your array needs to be binary (e.g to have 2 disticnt values like 0 and 1).");
 				}	
     		if (distinct_valuesprecision[0]!=0.0 && distinct_valuesprecision[1]!=1.0){
     			throw new IllegalStateException("Your array needs to be binary and have the values of 0 and 1");
     		}
    		for (int i=0; i <pred.length; i++ ) {
    			if (pred[i][column]==1.0){
    				totalprecision++;
    				if (actual[i][column]==1.0){
    					precision++;
    				}
    			}
    		}     		
    		results[column]=precision/totalprecision;    
		    distinct_valuesprecision=null;

    		}
    	 // recall	
    	else if (type.equals("rec")){
      		double recall=0;
      		double totalrecall=0.0;
     		double [] distinct_valuesrecall =manipulate.distinct.distinct.getdoubleDistinctset(actual,column);
 		    int Targetsizerecall=  distinct_valuesrecall.length;
 		    // Throw exception if the size is not 2
 		    if (Targetsizerecall!=2) {
 					throw new IllegalStateException("Your array needs to be binary (e.g to have 2 disticnt values like 0 and 1).");
 				}	
     		if (distinct_valuesrecall[0]!=0.0 && distinct_valuesrecall[1]!=1.0){
     			throw new IllegalStateException("Your array needs to be binary and have the values of 0 and 1");
     		}
    		for (int i=0; i <pred.length; i++ ) {
    			if (actual[i][column]==1.0){
    				totalrecall++;
    				if (pred[i][column]==1.0){
    					recall++;
    				}
    			}
    		}
		    distinct_valuesrecall=null;

    		results[column]=recall/totalrecall;  
    	}
    	// f1
      	else if (type.equals("f")){
      		double true_positive=0.0;
      		double false_positive=0.0;
      		double false_negative=0.0;
      		
     		double [] distinct_valuesf=manipulate.distinct.distinct.getdoubleDistinctset(actual,column);
 		    int Targetsizef = distinct_valuesf.length;
 		    // Throw exception if the size is not 2
 		    if (Targetsizef!=2) {
 					throw new IllegalStateException("Your array needs to be binary (e.g to have 2 disticnt values like 0 and 1).");
 				}	
     		if (distinct_valuesf[0]!=0.0 && distinct_valuesf[1]!=1.0){
     			throw new IllegalStateException("Your array needs to be binary and have the values of 0 and 1");
     		}
    		for (int i=0; i <pred.length; i++ ) {
    			if (pred[i][column]==1.0){
    				if (actual[i][column]==1.0){
    					true_positive++;
    				} else {
    					false_positive++;
    				}
    			} else {
    				if (actual[i][column]==1.0){
    					false_negative++;
    				}
				}
    		}    
		    distinct_valuesf=null;

    		results[column]= 2 * true_positive / (2 *true_positive  + false_positive + false_negative);      
      	}
    	 //r-squared
         
       	else if (type.equals("cor")){
  		   double sx=0.0;
  		   double sy=0.0;
  		   double sx_2=0.0;
  		   double sy_2=0.0;
  		   double sxy=0.0;
  		   double n=pred.length;
  		   
  		   for (int i=0; i < pred.length;i++){
  			   sx=sx+pred[i][column];
  			   sy=sy+actual[i][column];
  			   sx_2=sx_2+(pred[i][column]*pred[i][column]);
  			   sy_2=sy_2+(actual[i][column]*actual[i][column]);
  			   sxy=sxy+(pred[i][column]*actual[i][column]); 
  		   }
  		   double cor=(n*sxy - sx*sy)/(Math.sqrt(Math.abs(n*sx_2-(sx*sx))*Math.abs(n*sy_2-(sy*sy))));
  		 results[column]=cor*cor;
  		       	 
       	}	 
  		 else {
              throw new IllegalStateException(" Type was not recognized and has to be one of (rmse,mae,auc,log,cat,pre,rec,f,cor)");
     }
		
	}

}
