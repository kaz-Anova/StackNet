# Versions

## V0.1 version 01/04/2017

1.	Made significant **changes to the ml.tree** module. Previously it was built assuming a dense format. Whereas now it is built having a sparse format in mind. Significant changes have been made to enable hashing to be more efficient (but with more memory). Dense format might have suffered a bit, but it will be improved as I will later try to make adjustments to leverage this. 
2.	Improved multithreading in ml.tree
3.	Included a **indices_name** in the command line to print a.csv for each fold file with the corresponding train(0) and valiation(1) indices indices .The format it “row_index,[0 if training else 1]”. 
4.	Fixed a bug that was printing the same prediction file twice in last level of the training process.
5.	Fixed a bug when there was not a target variable (or an extra column anyway)  in the test file
6.	Fixed a bug with reading data in Sparse format
7.	Added standardscaler in code, BUT it is not available in StackNet yet (only in source code)  

