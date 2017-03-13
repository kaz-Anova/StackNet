package Stats;

import java.util.Arrays;
import exceptions.IllegalStateException;


/**
 * <p> This class is meant to reproduce all the basic univariate statistics
 * commonly found in statistics and many other statistical packages. Generally the form of this class
 * will be similar to Apache's common Math package simply because this library started
 * with those.
 * 
 * <p> This class will compute the following statistics :
 * <ol>
 * <li> Mean </li>
 * <li> Variance </li>
 * <li> Standard Deviation </li>
 * <li> Sum </li>
 * <li> Product </li>
 * <li> Min </li>
 * <li> Max </li>
 * <li> Count </li>
 * <li> Median </li>
 * <li> Quantiles </li>
 * <li> Percentiles </li>
 * <li> Skewness </li>
 * <li> Kurtosis </li>
 * <li> Range </li>
 * </ol>
 *
 */

public class DescriptiveStatistics {

	/**
	 * 
	 * @param array : The array to calculate the count
	 * @return : the count
	 */
    public static int getN(double array[]){
    	return array.length;	
    }    
    /**
     * Returns the average or mean namely:
     * @param main : The array to calculate the mean
     * <pre> m=S<sub>i=1</sub>x<sub>i</sub> / N </pre>
     * @return the mean as a double value
     */
    public static double getMean(double main []){
    	
    	if ( main==null ){
    			throw new IllegalStateException("The array is empty");
    	}
    	double mean=0;
    	for (int i=0; i < main.length; i++) {
    		mean+=main[i];
    	}
    	mean= mean/((double)(main.length));
		return mean;
    }
      
    
    /**
     * Returns the average or mean of the absolute values namely:
     * <pre> m=S<sub>i=1 Abs</sub>x<sub>i</sub> / N </pre>
     * @param array : Array to get the data from.
     * @return the absolute mean as a double value
     */
    public static double getAbsMean(double array[]){
    	if ( array==null ){
			throw new IllegalStateException("The array is empty");
	}
    	
    	double mean=0;
    	for (int i=0; i < array.length; i++) {
    		mean=mean+Math.abs(array[i]);
    	}
    	mean= mean/((double)(array.length));
		return mean;
    }

    /**
     * <p> This method returns the variance as static:
     * <pre> var=S<sub>i=1</sub>(x<sub>i</sub>-m)<sup>2</sup> / N-1 </pre>
     * @param array : The array to compute the variance
     * @return the average as double value
     */
    public static double getVariance( double array[]){
    	if ( array==null ){
			throw new IllegalStateException("The array is empty");
	}
    	     double sum=0.0;
    	     for (int i=0; i < array.length; i++){
    		sum+=array[i];
    	       }
    	double mean= sum/array.length;
    			
    			double variances=0;
    	    	for (int i=0; i < array.length; i++) {
    	    		variances=variances + (mean-array[i])*(mean-array[i]);
    	    	}
    	    	variances=variances/(double)(array.length-1);
    		return variances;
    		
    		}
    
    /**
     * <p> This method returns the standard deviation as static:
     * <pre> std=sqrt(var)</pre>
     * @param array : The array to compute the std
     * @return the average as double value
     */
    public static double getstd( double array[]){
    	if ( array==null ){
			throw new IllegalStateException("The array is empty");
	}
    	     double sum=0.0;
    	     for (int i=0; i < array.length; i++){
    		sum+=array[i];
    	       }
    	double mean= sum/array.length;
    			
    			double std=0;
    	    	for (int i=0; i < array.length; i++) {
    	    		std=std + (mean-array[i])*(mean-array[i]);
    	    	}
    	    	std=std/(double)(array.length-1);
    		return Math.sqrt(std);
    		
    		}
    
    /**
     * <p> Fast (and <b>DANGEROUS</b>) way to retrieve the Variance - still useful in some situations as :
     * <pre> var=S<sub>i=1</sub>(x<sub>i</sub><sup>2</sup>)-(S<sub>i=1</sub>(x<sub>i</sub>)<sup>2</sup>)/N /( N-1) </pre>
     * @param array : The array to compute the variance
     * @return the Variance in an one-pass manner
     */
    public static double getVarianceFast( double array[]){
    	if ( array==null ){
			throw new IllegalStateException("The array is empty");
	}
               double  n = 0;
               double Sum = 0;
               double Sum_sqr = 0;
        	 
        	    for (double x : array){
        	        n += 1;
        	        Sum += x;
        	        Sum_sqr += x*x;
        	    }

    		return (Sum_sqr - (Sum*Sum)/n)/(n - 1);
    		
    		}      
    /**
     * <p> Fast (and <b>DANGEROUS</b>) way to retrieve the Standard Deviation - still useful in some situations as :
     * <pre> std= sqrt(S<sub>i=1</sub>(x<sub>i</sub><sup>2</sup>)-(S<sub>i=1</sub>(x<sub>i</sub>)<sup>2</sup>)/N /( N-1) ) </pre>
     * @param array : The array to compute the variance
     * @return the STD as double value
     */
    public static double getStdFast( double array[]){
    	if ( array==null ){
			throw new IllegalStateException("The array is empty");
	}
    		return Math.sqrt(getVarianceFast(array));
    		
    		}         
        /**
         * <p> This method returns the Sum as :
         *  <pre> Sum=S<sub>i=1</sub>x<sub>i</sub> </pre>
         * @param array : The array to compute the sum
         * @return the sum as double value
         */
        public static double getSum(double array[]){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }

        			double Sum= 0.0;
        	    	for (int i=0; i < array.length; i++) {
        	    		Sum+=array[i];
        	    	}
        			return Sum;
        		}	
    
        /**
         * <p> Thus method returns the sum of squares namely :
         * <pre> sumofsquares =S<sub>i=1</sub>X<sub>i=1</sub><sup>2</sup> </pre>
         * @param array : The array to compute the sum of squares
         * @return The Sum of squares as double value
         */
        public static double getSumsq(double array []){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        	
        			double SumofSquare= 0.0;
        	    	for (int i=0; i <array.length; i++) {
        	    		SumofSquare=SumofSquare+(array[i]*array[i]);
        	    	}
        			return SumofSquare;
        		}	
        
        /**
         * <p> Thus method returns the sum of deviations squares (also known as second moment) namely :
         * <pre> sumDevsquares = S<sub>i=1</sub>(X<sub>i=1</sub>-m)<sup>2</sup> </pre>
         * @param array : The array to compute the Deviation of sum of squares
         * @return The Sum of deviations' squares as double value
         */
        public static double getDevSumsq(double array []){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        	
        			double SumDevSquare= 0.0;
        			double means=getMean(array);
        	    	for (int i=0; i < array.length; i++) {
        	    		SumDevSquare=SumDevSquare+((array[i]-means) * (array[i]-means));
        	    	}
        			return SumDevSquare;			
    }        
        /**
         * <p> Thus method returns the Absolute (the mean) sum of deviations squares namely :
         * <pre> sumDevsquares = S<sub>i=1</sub>(X<sub>i=1</sub>-Abs(m))<sup>2</sup> </pre>
         * @param array : The array to compute the Deviation of sum of squares
         * @return The Sum of absolute deviations' squares as double value
         */
        public static double getAbsDevSumsq(double array []){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        			double SumDevSquare= 0.0;
        			double means=getAbsMean(array);
        	    	for (int i=0; i < array.length; i++) {
        	    		SumDevSquare=SumDevSquare+((array[i]-means) * (array[i]-means));
        	    	}
        			return SumDevSquare;
        			
    }
        
        
        /**
         * <p> This method returns the Product as :
         *  <pre> Sum=P<sub>i=1</sub>x<sub>i</sub> </pre>
         *  Warning as it may be too big on big sets.
         * @param array : The array to compute the Deviation of sum of squares
         * @return the Product as double value
         */
        public static double getProduct(double array []){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        			double Product= 1.0;
        	    	for (int i=0; i <array.length; i++) {
        	    		Product*=array[i];
        	    	}
        			return Product;  		
    }      

        
        /**
         * <p> This method returns the Max 
         * @param a : the array to use
         * @return the maximum value as double value
         */
        public static double getMax(double a[]){
        	if ( a==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        		double Max=Double.NEGATIVE_INFINITY;
        		
        		for (int i=0; i< a.length; i++){
        			if (a[i]>=Max){
        				Max=a[i];
        			}	
        		}
        		return Max;
        		}	
        /**
         * <p> This method returns the Min
         * @param a : the array to use
         * @return the minimum value as double value
         */
        public static double getMin(double a[]){
        	if ( a==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        		double Min=Double.POSITIVE_INFINITY;
        		
        		for (int i=0; i< a.length; i++){
        			if (a[i]<Min){
        				Min=a[i];
        			}	
        		}
        		return Min;
        		}
        
        /**
         * <p> This method returns the Max index 
        * @param a : the array to use
         * @return the index with the maximum value as double value
         */
        public static int getMaxindex(double a[]){
        	if ( a==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        		double Max=Double.NEGATIVE_INFINITY;
        		int k=-1;
        		for (int i=0; i< a.length; i++){
        			if (a[i]>=Max){
        				Max=a[i];
        				k=i;
        			}	
        		}
        		return k;
        		}	   


        	
        	  /**
             * <p> Returns the range which stands for the MAX-MIN.
             * @param array : the array to compute the range
             * @return The Range as a double value
             */
            public static double getRange (double array [] ) {
            	if ( array==null ){
        			throw new IllegalStateException("The array is empty");
        	                }
            			double Ranges=0;
            			double Mins=Double.POSITIVE_INFINITY;
            			double Maxs=Double.NEGATIVE_INFINITY;
            			
            	     	for (int i=0; i< array.length; i++){
            			if (array[i]<Mins){
            				Mins=array[i];
            			}	
            			if (array[i]>Maxs){
            				Maxs=array[i];
            			}
            		}
            	     	Ranges=	Maxs-Mins;
            	
            	return Ranges;
            }
              

        
        
        /**
         * <p> This method returns the absolute Max 
         * @param array : The double array 
         * @return the absolute maximum value as double value
         */
        public static double getabsMax( double array[]){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        		double absMax=Double.MIN_VALUE;
        		for (int i=0; i< array.length; i++){
        			if (Math.abs(array[i])>absMax){
        				absMax=Math.abs(array[i]);
        			}	
        		}
        		return absMax;
        			
        }
        
        
        
        /**
         * <p> This method returns the absolute Min 
         * @param array : The double array 
         * @return the  absolute minimum value as double value
         */
        public static double getabsMin(double array []){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        			double absMin=Double.POSITIVE_INFINITY;
        		for (int i=0; i<array.length; i++){
        			if (Math.abs(array[i])<absMin){
        				absMin=Math.abs(array[i]);
        			}	
        		}
        		return absMin;
        		
        }
        
        /**
         * <p> This method returns the kurtosis of the variable, same as SPSS does :
         * <pre>  kurtosis = { [n(n+1)sum(x<sub>i</sub> - m)<sup>4</sup>- 3(n-1)<sup>3</sup>st.dev<sup>4</sup>]  /[ st.dev<sup>4</sup> (n-1)(n-2)(n-3)]} </pre>
         * @param array : The double array 
         * @return the  kurtosis as a double value
         */
        public static double getKurtosis(double array []){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }	
        		double variance= getVariance(array);
        		double mean=getMean(array);
        		double n =array.length;
        		double xi_minus_mean_four=0;
        		for (int i=0; i<array.length; i++) {
        			xi_minus_mean_four=xi_minus_mean_four+Math.pow((array[i]-mean),4);
        		}
        		double Kurtosis=(((n*(n+1)*xi_minus_mean_four)-(3*(n-1)*(n-1)*(n-1) *variance*variance))
        				/((n-1)*(n-2)*(n-3)*variance*variance));      		
        		return Kurtosis;
        			
        }
   
        /**
         * <p> This method returns the kurtosis of the variable, same as many statistical packages do :
         * <pre>  skewness = [n / (n -1) (n - 2)] sum[(x<sub>i</sub>- mean)<sup>3</sup>] / std<sup>3<sup>  </pre>
         * @param array : The double array         
         * @return the  kurtosis as a double value
         */
        public static double getSkewness(double array []){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }	
        		double Skewness=0.0;	
        		double variance= getVariance(array);
        		double st_dev=Math.sqrt(variance);
        		double mean=getMean(array);
        		double n = array.length;
        		double xi_minus_mean_three=0;
        		for (int i=0; i<array.length; i++) {
        			xi_minus_mean_three=xi_minus_mean_three+Math.pow((array[i]-mean),3);
        		}
        		Skewness=(n*xi_minus_mean_three)/
        				((n-1)*(n-2)*variance*st_dev);      		
        		return Skewness;
        			
        }
        
        /**
         * <p> This method returns the percentile given a provided double number.
         * It would have made more sense to have the data already sorted prior to running this,
         *  however you will have the option to choose in the
         * beginning on whether the data is sorted or not. Then this value is kept for
         * future calculations, otherwise it needs to be re-specified
         * @param array : The double array      
         * @param value : a double value between 0 and 1 to reflect the desired percentage
         * @param sorted : a boolean flag on whether the data is sorted or not  (false not sorted which is the default)
         * @return the  percentile as a double value
         */
        public static double  getPercentile(double array [],double value, boolean sorted) {
        	boolean sort=sorted;
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        	if ( value<0 || value>100){
            		throw new IllegalStateException("The value provided needs to be between 0 and 1");
        	}
             if( array.length==1) {
        			return array[0];}
        	else {
        		
               if(sort==false){
            	   Arrays.sort(array);
            	   }
               
               double position= value*(array.length+1)/100;
               double floor = Math.floor(position);
               int floor_pos=(int)floor;
               double difference =position-floor;
               if (position < 1) {
                   return array[0];
               } else if (position >= array.length) {
                   return array[array.length - 1];
               }else{ 
        		return ( array[floor_pos-1] + difference * (array[floor_pos] - array[floor_pos-1]));	
               }
        		}		
        }
        
        
        /**
         * <p> This method returns the median
         * @param array : The double array     
         * @param sorted : a boolean flag on whether the data is sorted or not  (false not sorted which is the default)
         * @return the median as a double value
         */
        public static double getMedian(double array [], boolean sorted){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        	double Median=getPercentile(array, 50.00, sorted);
        	return Median;		
        }    
        
        /**
         * <p> This method returns the first Quantile (25%)
         * @param array : The double array   
         * @param sorted : a boolean flag on whether the data is sorted or not  (false not sorted which is the default)
         * @return the first quantile as a double value
         */
        public static double getQuantile1(double array [],boolean sorted){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        	double quantile1=getPercentile(array,25.00, sorted);
        	return quantile1;		
        }  
        /**
         * <p> This method returns the thrid Quantile (75%)
         * @param array : The double array   
         * @param sorted : a boolean flag on whether the data is sorted or not  (false not sorted which is the default)
         * @return the first quantile as a double value
         */
        public static double getQuantile3(double array [],boolean sorted){
        	if ( array==null ){
    			throw new IllegalStateException("The array is empty");
    	                }
        	double quantile1=getPercentile(array,75.00, sorted);
        	return quantile1;		
        }  
        
        /**
         * 
         * @param array a double array
         * @param divide the value we want to divide all elements of the array
         * @return the new array
         */
        public static double[] dividearray(double array[], double divide){
        	double ar[]= new double [array.length];
        	for (int i=0; i < array.length; i++){
        		ar[i]=array[i]/divide;
        	}
        	return ar;
        }
        
        /**
         * 
         * @param array a double array
         * @param multiply the value we want to multiply all elements of the array
         * @return the new array
         */
        public  static double[] multiplyarray(double array[], double multiply){
        	double ar[]= new double [array.length];
        	for (int i=0; i < array.length; i++){
        		ar[i]=array[i]*multiply;
        	}
        	return ar;
        }
        
        /**
         * 
         * @param array a double array
         * @param add the value we want to add all elements of the array
         * @return the new array
         */
        public  static double[] addarray(double array[], double add){
        	double ar[]= new double [array.length];
        	for (int i=0; i < array.length; i++){
        		ar[i]=array[i]+add;
        	}
        	return ar;
        }
        /**
         * 
         * @param array a double array
         * @param substract the value we want to add all elements of the array
         * @return the new array
         */
        public   static double[] substractarray(double array[], double substract){
        	double ar[]= new double [array.length];
        	for (int i=0; i < array.length; i++){
        		ar[i]=array[i]-substract;
        	}
        	return ar;
        }   
         
// end of class	
}
