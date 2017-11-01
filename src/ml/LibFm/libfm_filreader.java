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

package ml.LibFm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

import io.readcsv;

/**
 * 
 * Reads data from the 'save_model' command of the original libFM software implementation
 *
 */
public class libfm_filreader {
	
	/**
	 * This will hold the latent features to encapsulate the 2d interactions among the variables
	 */
	private double[] latent_features;
	/**
	 * The 1-way coefficient values
	 */
	private double[] betas;
	/**
	 * The constant value
	 */
	private double[] constant;
	/**
	 * 
	 * @return the HashMap of that holds the latent features
	 */
	public  double[] GetLatentFeatures(){
		
		if (latent_features==null || latent_features.length<=0){
			throw new IllegalStateException(" load_libfm needs to be run first" );
		}
		return manipulate.copies.copies.Copy(this.latent_features.clone());
	}	
	/**
	 * 
	 * @return the betas
	 */
	public double [] Getbetas(){
		if (betas==null || betas.length<=0){
			throw new IllegalStateException(" load_libfm needs to be run first" );
		}
		return manipulate.copies.copies.Copy(betas);
	}
	/**
	 * @return the constant of the model
	 */
	public double  Getcosntant(){
		
		if (constant==null || constant.length<=0){
			throw new IllegalStateException(" load_libfm needs to be run first" );
		}
		return constant[0];
	}	
	

    /**
     * 
     * @param n : The file to 'Open' with libFM parameters.
     * @param latent : Number of latent features in the file
     * <p> all parameters are inserted in global objects (constant, betas, latent_features)
     */

			public void load_libfm(String n, int latent) {
            	 
				 File x= new File(n);
            	 String line;
                 int rowcount=readcsv.getrowcount(n);   
                 rowcount++;
                 rowcount=rowcount-2; //remove the 3 sentences for constant, beta and latent features
                 try {
                     FileInputStream fis = new FileInputStream(x);
                     @SuppressWarnings("resource")
					BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
                    line = br.readLine();
                    if (line.contains("#global bias")){
                    	 rowcount=rowcount-2;
                    }
             //Close the buffer reader
                   br.close();
                   fis.close();
           } catch (Exception e) {
                   e.printStackTrace();
                   
           }                   
                 constant= new double []{0.0};
                
                 //sanity check
                 if (rowcount%2!=0){
                	 throw new IllegalStateException(" The size of betas plus the latent feature entries in the 'save_model' file are not of crrect size because " + rowcount + " %2 <> 0");
                 }
                 betas= new double [rowcount/2];
                 latent_features=new double [(rowcount/2) * latent];
                 int constant_counter=0;
                 int beta_counter=0;
                 int latent_counter=0;
                 int type=0;//type of feature in the file. 0 is constant, 1 is betas, 2 are the latent features.              

   
                         
                 try {
                         FileInputStream fis = new FileInputStream(x);
                         @SuppressWarnings("resource")
						BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));

                         while ((line = br.readLine()) != null) {
                        	 if (line.contains("#global bias")){
                        		 type=0;
                        		 continue;
                        	 }
                        	 if (line.contains("#unary interactions")){
                        		 type=1;
                        		 
                        		 continue;
                        	 } 
                        	 if (line.contains("#pairwise interactions")){
                        		 type=2;
                        		 continue;
                        	 }    
                        	 
                        	 if(type==0){
                        		 if (constant_counter>=1){
                        			 throw new IllegalStateException("More than one constant value has been found when reading libFM save_model file"); 
                        		 }
                        		 constant[constant_counter]=Double.parseDouble(line);
                        		 constant_counter++;
                        		 
                        	 } else if (type==1){
                        		 if (beta_counter>=betas.length){
                        			 throw new IllegalStateException("More betas have been found when reading libFM save_model file"); 
                        		 }
                        		 betas[beta_counter]=Double.parseDouble(line);
                            	 beta_counter++; 
                        	 }else { //type is 2
   
                        		 String tokens [] = line.split(" ",-1);
                        		 if (tokens.length!=latent){
                        			 throw new IllegalStateException("There are " +  tokens.length + " latent features in a row while expecting " + latent + "."); 
                        		 }
                        		 for (int j=0; j <tokens.length;j++ ){
                        			 
                            		 if (latent_counter>=latent_features.length){
                            			 throw new IllegalStateException("More latent feature values have been found when reading libFM save_model file"); 
                            		 }  
                            		 latent_features[latent_counter]=Double.parseDouble(tokens[j]);
                        			 latent_counter++; 
                        		 }
 
                        	 }
                        	}
                   //Close the buffer reader
                         br.close();
                         fis.close();
                 } catch (Exception e) {
                         e.printStackTrace();
                         
                 }


               
         } 	
	
	
}
