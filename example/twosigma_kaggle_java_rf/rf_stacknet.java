
import java.io.FileWriter;
import matrix.fsmatrix;
import ml.Tree.RandomForestClassifier;
import crossvalidation.splits.kfold;



/*
 * Example of using Random Forest with The STackNet Library using data from
 * https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data
 * and then following the same instructions in :
 * https://github.com/kaz-Anova/StackNet/blob/master/example/twosigma_kaggle/EXAMPLE.MD
 * until 6.
 */

public class rf_stacknet {
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		// The Directory where the files are
		String directory="";
		// output name (prefix) for the submission file
		String outputname="rf";
		//model parameters for an RandomForestClassifier models
		String model_params="bootsrap:false estimators:100 threads:3 offset:0.00001 max_depth:6 max_features:0.4 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1 verbose:false";
		//specify number of folds
		int folds=5;
		
		String file=directory + "train_stacknetv1.csv";		
		io.input in = new io.input(); //open a reader-type of class
		in.delimeter=","; // set delimiter
        	in.HasHeader=false; // it does not have headers
		in.targets_columns= new int[] {0}; // the first column is the target
		in.start=1; //we load the predictors from (1,onward). so we set for the main data everything apart from the target (column 0)
		in.end=5000;
		fsmatrix X= in.Readfmatrix(file); // we read the data as fixed-size matrix
		double label []= in.GetTarget(); // we retrieve the label. If we had not set target_columns this would have been null
		System.out.println("Loaded train data with " + X.GetRowDimension() + " and columns " + X.GetColumnDimension() );		
		
		//specify test file
		file=directory + "test_stacknetv1.csv";		
		in = new io.input();
		in.delimeter=",";
        	in.HasHeader=false;
       		in.idint=0; // the first column is the id and we set it as int, this has to be int value, not array
		in.start=1;
		in.end=5000;
		fsmatrix X_test=in.Readfmatrix(file); // we read the data as fixed-size matrix
		int[] id=in.GetIntid();// we retrieve the id as int array
			
		System.out.println("Loaded test data with " + X_test.GetRowDimension() + " and columns " + X_test.GetColumnDimension() );
		
		double mean_logloss=0.0; // the metric;
		//set a kfolder object
		int kfolder [][][]=kfold.getindices(label.length, folds);
		
		// begin cross validation
		System.out.println(" begin cross validation");
		for (int f=0; f <folds; f++){
			
				int train_indices[]=kfolder[f][0]; // train indices
				int test_indices[]=kfolder[f][1]; // test indices	
				// create train an cv data based on array's indices
				fsmatrix X_train= X.makerowsubset(train_indices);
				fsmatrix X_cv= X.makerowsubset(test_indices);
				//also slice the target
				double [] y_train=manipulate.select.rowselect.RowSelect(label, train_indices);
				double [] y_cv=manipulate.select.rowselect.RowSelect(label, test_indices);			
				
				// the modelling object

				RandomForestClassifier model = new RandomForestClassifier();
				model.set_params(model_params); //put the parameters based on the string object on the top

				/* alternative way to put parameters one by one
					model.estimators=100;
					model.max_depth=15;
					model.seed=1;
					etc....
				 */
				
				//Set target
				model.target=y_train;
					
				//fit model
				model.fit(X_train);
				//make predictions in probabilities
				double preds[][]=model.predict_proba(X_cv);
				double logloss=computelogloss(y_cv ,preds); // compute logloss for the current fold	based on the method at the bottom		
				mean_logloss+=logloss;
				System.out.printf(" cv fold: %d/%d  training size: %d test size: %d logloss-----> ( %.4f) \n", f+1,folds, X_train.GetRowDimension(),X_cv.GetRowDimension(),logloss);

		}
		mean_logloss/=folds; // average logloss
		System.out.printf(" Final logloss-----> %.4f <-----\n",mean_logloss);
		System.out.println(" Beginning test modelling");

			
		// the modelling object

		RandomForestClassifier model = new RandomForestClassifier();		
		model.set_params(model_params);	
		model.target=label;	
		model.fit(X);
			
		double preds[][]=model.predict_proba(X_test);


		System.out.println(" Finsihed predictions for test");

		System.out.println(" PRINTING ");
		
		try{  // Catch errors in I/O if necessary.
			  // Open a file to write to.
				String saveFile = directory + outputname +"_Submission_" + mean_logloss + ".csv";
				//open a writter
				FileWriter writer = new FileWriter(saveFile);
				//set first lime-headers
				writer.append("listing_id,high,medium,low\n");
				//print for every row the id and then the 3 values corresponding to probabilities
				for (int i=0; i < preds.length;i++){
					writer.append( id[i] +"");
					for (int j=0; j < 3;j++){						
						 writer.append( "," + preds[i][j]);
					}
					 writer.append("\n");
				}
				
				writer.close();
				System.out.println(" Done! ");
				
		}catch (Exception e) {}
		
		

	}

	/**
	 * @param target : the target variable
	 * @param probas : rows/columns probability predictions
	 * @return the logloss value
	 */
	public static double computelogloss(double target[], double probas[][]){
		
		double logloss=0.0;
		
		double epsilon= 1e-15;
	        
		for (int i=0; i < target.length; i++){
			logloss-=Math.log(Math.min(Math.max(probas[i][(int) target[i]],epsilon ),1-epsilon));
		}
		return logloss/(double)((target.length)) ;
		
	}

}
