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

package misc;
/**
 * 
 * @author marios
 * <p> class for printing </p>
 */
public class print {

	/**
	 * 
	 * @param array : array to print
	 * @param cases : number of cases from the beginning to print
	 */
	public static void Print(double [] array, int cases){
		
		if (cases <=0){
			cases=array.length;
		}
		if (cases >100){
			cases=100;
		}
		if (cases >array.length){
			cases=array.length;
		}
		// printing
		for (int i=0; i < cases; i++) {
			System.out.println("row: " + i + " value: " + array[i]);
		}
		
	}
	
	
	/**
	 * 
	 * @param array : array to print
	 * @param cases : number of cases from the beginning to print
	 */
	public static void Print(String [] array, int cases){
		
		if (cases <=0){
			cases=array.length;
		}
		if (cases >100){
			cases=100;
		}
		if (cases >array.length){
			cases=array.length;
		}
		// printing
		for (int i=0; i < cases; i++) {
			System.out.println("row: " + i + " value: " + array[i]);
		}
		
	}	
	 
		/**
		 * 
		 * @param array : array to print
		 * @param cases : number of cases from the beginning to print
		 */
	public static void Print(int [] array, int cases){
			
			if (cases <=0){
				cases=array.length;
			}
			if (cases >1000){
				cases=1000;
			}
			if (cases >array.length){
				cases=array.length;
			}
			// printing
			for (int i=0; i < cases; i++) {
				System.out.println("row: " + i + " value: " + array[i]);
			}
			
		}	 
	 
			/**
			 * 
			 * @param array : array to print
			 * @param start : start of the loop
			 * @param end : send of the loop
			 */
	public static void Print(double [] array, int start, int end){
				
				if (start <=0){
					start=0;
				}
				if (start >=array.length){
					throw new IllegalStateException(" start of the loop exceeds current length for printing");
				}				
				if (end <=0){
					end=array.length;
				}
				if (end >array.length){
					throw new IllegalStateException(" end of the loop exceeds current length for printing");
				}	
				if (start> end){
					throw new IllegalStateException(" end of the loop is smaller than the start length for printing");
				}
				// printing
				for (int i=start; i < end; i++) {
					System.out.println("row: " + i + " value: " + array[i]);
				}
				
			}
			
			/**
			 * 
			 * @param array : array to print
			 * @param start : start of the loop
			 * @param end : send of the loop
			 */
	public static void Print(String [] array, int start, int end){
				
					if (start <=0){
						start=0;
					}
					if (start >=array.length){
						throw new IllegalStateException(" start of the loop exceeds current length for printing");
					}				
					if (end <=0){
						end=array.length;
					}
					if (end >array.length){
						throw new IllegalStateException(" end of the loop exceeds current length for printing");
					}	
					if (start> end){
						throw new IllegalStateException(" end of the loop is smaller than the start length for printing");
					}
					// printing
					for (int i=start; i < end; i++) {
						System.out.println("row: " + i + " value: " + array[i]);
					}
				
			}	
			 
			/**
			* 
			* @param array : array to print
			* @param start : start of the loop
			* @param end : send of the loop
			*/
	public static void Print(int [] array, int start, int end){
					
						if (start <=0){
							start=0;
						}
						if (start >=array.length){
							throw new IllegalStateException(" start of the loop exceeds current length for printing");
						}				
						if (end <=0){
							end=array.length;
						}
						if (end >array.length){
							throw new IllegalStateException(" end of the loop exceeds current length for printing");
						}	
						if (start> end){
							throw new IllegalStateException(" end of the loop is smaller than the start length for printing");
						}
						// printing
						for (int i=start; i < end; i++) {
							System.out.println("row: " + i + " value: " + array[i]);
					}
					
				}	
				 
				 
					/**
					 * 
					 * @param array : array to print
					 * @param cases : number of cases from the beginning to print
					 */
	public static void Print(double [][] array, int cases){
						
						if (cases <=0){
							cases=array.length;
						}
						if (cases >100){
							cases=100;
						}
						if (cases >array.length){
							cases=array.length;
						}
						// printing
						for (int i=0; i < cases; i++) {
							System.out.print("row: " + i);
							for (int j=0; j < array[i].length; j++) {							
							   System.out.print( "  col" + j +":"  + array[i][j]);
							}
							System.out.println("");
						}
						
					}
					
					/**
					 * 
					 * @param array : array to print
					 * @param cases : number of cases from the beginning to print
					 */
	public static void Print(String [][] array, int cases){
						
						if (cases <=0){
							cases=array.length;
						}
						if (cases >100){
							cases=100;
						}
						if (cases >array.length){
							cases=array.length;
						}
						// printing
						for (int i=0; i < cases; i++) {
							System.out.print("row: " + i);
							for (int j=0; j < array[i].length; j++) {							
							   System.out.print( "  col" + j +":"  + array[i][j]);
							}
							System.out.println("");
						}
						
					}	
					 
						/**
						 * 
						 * @param array : array to print
						 * @param cases : number of cases from the beginning to print
						 */
	public static void Print(int [][] array, int cases){
							
							if (cases <=0){
								cases=array.length;
							}
							if (cases >100){
								cases=100;
							}
							// printing
							if (cases >array.length){
								cases=array.length;
							}
							// printing
							for (int i=0; i < cases; i++) {
								System.out.print("row: " + i);
								for (int j=0; j < array[i].length; j++) {							
								   System.out.print( "  col" + j +":"  + array[i][j]);
								}
								System.out.println("");
							}
							
						}	 
					 
							/**
							 * 
							 * @param array : array to print
							 * @param start : start of the loop
							 * @param end : send of the loop
							 */
	public static void Print(double [][] array, int start, int end){
								
								if (start <=0){
									start=0;
								}
								if (start >=array.length){
									throw new IllegalStateException(" start of the loop exceeds current length for printing");
								}				
								if (end <=0){
									end=array.length;
								}
								if (end >array.length){
									throw new IllegalStateException(" end of the loop exceeds current length for printing");
								}	
								if (start> end){
									throw new IllegalStateException(" end of the loop is smaller than the start length for printing");
								}
			
								// printing
								for (int i=start; i < end; i++) {
									System.out.print("row: " + i);
									for (int j=0; j < array[i].length; j++) {							
									   System.out.print( "  col" + j +":"  + array[i][j]);
									}
									System.out.println("");
								}
								
							}
							
							/**
							 * 
							 * @param array : array to print
							 * @param start : start of the loop
							 * @param end : send of the loop
							 */
	public static void Print(String [][] array, int start, int end){
								
									if (start <=0){
										start=0;
									}
									if (start >=array.length){
										throw new IllegalStateException(" start of the loop exceeds current length for printing");
									}				
									if (end <=0){
										end=array.length;
									}
									if (end >array.length){
										throw new IllegalStateException(" end of the loop exceeds current length for printing");
									}	
									if (start> end){
										throw new IllegalStateException(" end of the loop is smaller than the start length for printing");
									}
									// printing
									for (int i=start; i < end; i++) {
										System.out.print("row: " + i);
										for (int j=0; j < array[i].length; j++) {							
										   System.out.print( "  col" + j +":"  + array[i][j]);
										}
										System.out.println("");
									}
								
							}	
							 
							/**
							* 
							* @param array : array to print
							* @param start : start of the loop
							* @param end : send of the loop
							*/
	public static void Print(int [][] array, int start, int end){
									
										if (start <=0){
											start=0;
										}
										if (start >=array.length){
											throw new IllegalStateException(" start of the loop exceeds current length for printing");
										}				
										if (end <=0){
											end=array.length;
										}
										if (end >array.length){
											throw new IllegalStateException(" end of the loop exceeds current length for printing");
										}	
										if (start> end){
											throw new IllegalStateException(" end of the loop is smaller than the start length for printing");
										}
										// printing
										for (int i=start; i < end; i++) {
											System.out.print("row: " + i);
											for (int j=0; j < array[i].length; j++) {							
											   System.out.print( "  col" + j +":"  + array[i][j]);
											}
											System.out.println("");
										}
									
								}	
								 
								 
				 
				 

}
