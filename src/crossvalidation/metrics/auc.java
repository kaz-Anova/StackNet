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
 *<p> Class used to calculate the Area Under the Roc Curve defined as:
 * <p> This class calculates the area under the ROC curve commonly used to asses
 * the predictive power of a score (double array) versus a binary event .
 * <p> Generally the ROC (Receiver Operator Characteristics) curve was first introduced by Green & swets (1966) and 
 * it maps the confusion matrix of sensitivity and 1-specificity for all possible cut-offs of the scoring array.
 * <p>The area under ROC will be calculated by summing the the number of cases with a positive score that are
 * higher of the total of those with a negative score for each different distinct score divided by the total number of 'good' and 'bad' combinations.
 */
public class auc implements Metric {
    /**
     * The current_metrics' value
     */
	private double metric=0.0;

	public double GetValue(double[] predicted, double[] actual) {
		//sensible checks
		if (actual==null || predicted==null || actual.length==0 || predicted.length==0 || actual.length!=predicted.length){
			throw new IllegalStateException(" There is an error with the state of actual and pred in terms of length ");
		}
		 // find distinct values
	    double [] distinct_values =manipulate.distinct.distinct.getdoubleDistinctset(actual);
	    int Targetsize=  distinct_values.length;
	    // Throw exception if the size is not 2
	    if (Targetsize!=2) {
				throw new IllegalStateException("Your array needs to be binary (e.g to have 2 disticnt values like 0 and 1).");
			}	 
	    //find smallest and highest values
	    Arrays.sort(distinct_values);
	    double low=distinct_values[0];
	    double high=distinct_values[1];	
	    //make compies to respect order
	    double predictedvalues []=manipulate.copies.copies.Copy(predicted);
	    double target []=manipulate.copies.copies.Copy(actual);
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
	    predictedvalues=null;
	    target=null;
	    distinct_values=null;
		metric=(big_sum/combination);
		return (big_sum/combination);
	}

	@Override
	public double[] GetValue(double[][] predicted, double[][] actual,int threads) {
		
		//sensible checks
		if (actual==null || predicted==null || actual.length==0 || predicted.length==0 || actual.length!=predicted.length || predicted[0].length!=actual[0].length){
			throw new IllegalStateException(" There is an error with the state of actual and pred in terms of dimensions ");
		}		
		// thread checks
		
		if (threads<=0) {
			threads=1;
		}
		
		if (threads>predicted[0].length) {
			threads=predicted[0].length;
		}	
		// initialize the new arry to copy to
		double new_array_to_copy_to []= new double [predicted[0].length];
	
		// the threads of operations
		
		Thread[] thread_array= new Thread[threads];
		int count_of_live_threads=0;
		
		for (int n=0; n <new_array_to_copy_to.length; n++ ){
			thread_array[count_of_live_threads]= new Thread(new Metrics2dRunnable(actual,predicted, new_array_to_copy_to, n,"auc"));
			thread_array[count_of_live_threads].start();
			count_of_live_threads++;
			if (count_of_live_threads==threads || n==new_array_to_copy_to.length-1){
				for (int s=0; s <count_of_live_threads;s++ ){
					try {
						thread_array[s].join();
					} catch (InterruptedException e) {
					   System.out.println(e.getMessage());
					}
				}
				count_of_live_threads=0;
			}
		}
		
	    metric=Stats.DescriptiveStatistics.getMean(new_array_to_copy_to);
	    
		return  new_array_to_copy_to ;
	}

	@Override
	public double GetValue() {
		return metric;
	}

	@Override
	public boolean IsBetter(double value) {
		if (value>metric){
			return true;
		}else {
		return false;
		}
	}

	@Override
	public void UpdateValue(double value) {
		metric=value;
		
	}

	@Override
	public boolean IsBetter(Metric m) {
		if ( !m.Gettype().equals(this.Gettype())){
			throw new IllegalStateException(" Metrics are not comparable as: " + m.Gettype() + " <> " + this.Gettype());
		}
		if (m.GetValue()>metric){
			return true;
		}else {
		return false;
		}
	}

	@Override
	public String Gettype() {

		return "auc";
	}

}
