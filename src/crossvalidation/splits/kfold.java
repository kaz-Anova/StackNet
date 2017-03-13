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

/**
 * 
 * @author mariosm
 * <p> Method to perform kfold and provide indices (as in rows) 
 */
public class kfold {
	
	/**
	 * The number of rows , kfold is constructed for
	 */
	private static int sizes=0;
	
	/**
	 * number of folds
	 */
	private static int nfolds=5;
	
	/**
	 *  target variable to make stratified kfold
	 */
	public double doubletargetvar [];
	
	/**
	 *  target variable to make stratified kfold
	 */
	public String stringtargetvar [];
	
	/**
	 *   seed to replicate results
	 */	
	public static int seed=1;
	
	/**
	 * 
	 * @param size : the desired size to make indices for (as in rows)
	 * @param folds : number of folds
	 * @return a 3-dimensional array of [folds][2 samples] [indices]. 
	 * The first sample is the big one (as in the training) and the second is the test ( e.g. the small one )
	 */
	public static int [] [] [] getindices(int size, int folds){
		
		if (folds<2){
			throw new IllegalStateException(" folds cannot be less than 2");
		}
		
		if (size<folds) {
			throw new IllegalStateException(" Size cannot be less than the given folds");
		}
		nfolds=folds;
		sizes=size;
		//set seed
		Random ran= new Random();
		ran.setSeed(seed);
		
		// we construct the main object
		int [] [] [] holder= new int [nfolds] [2] [];
		
		int allcases[]= new int [sizes];
		for (int i=0; i < allcases.length; i++){
			 allcases[i]=i;
		}
		
		// shuffle the indices
		shuffleArray(allcases,  ran);
		
		int hold_sizes []= new int[nfolds+1];
		int minibatch=(int)((double) size / (double) nfolds);
		
		if (minibatch<1){
			minibatch=1;
		}
		int sumofcasessofar=0;
		for (int j=1; j <nfolds; j++){
			sumofcasessofar+=minibatch;
			hold_sizes[j]=sumofcasessofar;
		}
		hold_sizes[nfolds]= size;
		//System.out.println(Arrays.toString(hold_sizes));
		if (hold_sizes[nfolds]<=0){
			throw new IllegalStateException(" The last sample cannot be negative, there is an internal computationl problem");
		}
		
		//populate the arrays
		
		for (int f=0; f < nfolds ; f++){
			
			int train  []= new int [size - (hold_sizes[f+1]-hold_sizes[f])];
			int test []=new int [(hold_sizes[f+1]-hold_sizes[f])];
			int tr=0;
			int te=0;
			
			for (int i=0; i < size; i++){
				
				if (i >= hold_sizes[f] && i < hold_sizes[f+1]){
					test[te]=allcases[i];
					te++;
				} else {
					train[tr]=allcases[i];
					tr++;
				}
			}
			holder[f][0]=train;
			holder[f][1]=test;			
			
			
			//end of main loop
		}
		
		return holder;
		
		
	}
	
	/**
	 * 
	 * @param ar :  int array for shuffling
	 * @param rand :Random number generator object
	 */

	static void shuffleArray(int[] ar, Random rand)
	{ 
	  for (int i = ar.length - 1; i > 0; i--)
	  {
	    int index = rand.nextInt(i + 1);
	    // Simple swap
	    int a = ar[index];
	    ar[index] = ar[i];
	    ar[i] = a;
	  }
	}
	
	
	

}
