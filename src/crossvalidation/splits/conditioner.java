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

import exceptions.IllegalStateException;

/**
 * 
 * @author marios
 * <p> class to assist in evaluations an comparisons in the splitting methods </p>
 */
public class conditioner {
	/**
	 * What to compare, could be one of double, int or string
	 */
	private String type="double";
	/**
	 * How to compare. It can be one of =,<,>,!=,<=,>= . The meaning is self-explanatory
	 */
	private String comarison_type="=";
	/**
	 * This is the value we always compare against as double
	 */
	private double dvalue;
	/**
	 * This is the value we always compare against as int
	 */
	private int ivalue;
	/**
	 * This is the value we always compare against as String
	 */
	private String svalue;		
	
	
	/**
	 * 
	 * @param comarison_type : It can be one of =,<,>,!=,<=,>= . The meaning is self-explanatory
	 * @param value : the double value to compare against
	 */
	public conditioner( String comarison_type, double value ){
		if (!comarison_type.equals("=") && !comarison_type.equals("<")&& !comarison_type.equals(">")
				&& !comarison_type.equals("!=")&& !comarison_type.equals("<=")&& !comarison_type.equals(">=")) {
			throw new IllegalStateException("Type has to be one of =,<,>,!=,<=,>=");
		}
		dvalue=value;
		type="double";
		this.comarison_type=comarison_type;	
	}
	/**
	 * 
	 * @param comarison_type : It can be one of =,<,>,!=,<=,>= . The meaning is self-explanatory
	 * @param value : the int value to compare against
	 */
	public conditioner( String comarison_type, int value  ){
		if (!comarison_type.equals("=") && !comarison_type.equals("<") && !comarison_type.equals(">")
				&& !comarison_type.equals("!=") && !comarison_type.equals("<=") && !comarison_type.equals(">=")) {
			throw new IllegalStateException("Type has to be one of =,<,>,!=,<=,>=");
		}
		ivalue=value;
		type="int";
		this.comarison_type=comarison_type;	
	}
	
	/**
	 * 
	 * @param comarison_type : It can be one of =,<,>,!=,<=,>= . The meaning is self-explanatory
	 * @param value : the String value to compare against
	 */
	public conditioner( String comarison_type, String value  ){
		if (!comarison_type.equals("=") && !comarison_type.equals("<")&& !comarison_type.equals(">") && !comarison_type.equals("!=")&& !comarison_type.equals("<=")&& !comarison_type.equals(">=") ) {
			
			throw new IllegalStateException("Type has to be one of =,<,>,!=,<=,>=");
		}
		svalue=value ;
		type="String";
		this.comarison_type=comarison_type;

	}

	/**
	 * 
	 * @param Value : the value to compare with the cvalue (constructor)
	 * @return 1 if the comparison with the object is true
	 */
	public int compare(Object Value){
		
	
		if (type.equals("double")) {
			
			double val=((Number)Value).doubleValue();
			if (comarison_type.equals("=")){
				if (dvalue==val) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals("<")){
				if (val<dvalue) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals(">")){
				if (val>dvalue) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals("!=")){
				if (val!=dvalue) {
					return 1;
				} else {
					return 0;
				}	
			} else if (comarison_type.equals("<=")){
				if (val<=dvalue) {
					return 1;
				} else {
					return 0;
				}					
			} else if (comarison_type.equals(">=")){
				if (val>=dvalue) {
					return 1;
				} else {
					return 0;
				}				
			}	else {
				throw new IllegalStateException("Comparison operator not recognised");	
			}
		}else if (type.equals("int")) {
			int vali=((Number)Value).intValue();
			if (comarison_type.equals("=")){
				if (ivalue==vali) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals("<")){
				if (vali<ivalue) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals(">")){
				if (vali>ivalue) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals("!=")){
				if (vali!=ivalue) {
					return 1;
				} else {
					return 0;
				}	
			} else if (comarison_type.equals("<=")){
				if (vali<=ivalue) {
					return 1;
				} else {
					return 0;
				}					
			} else if (comarison_type.equals(">=")){
				if (vali>=ivalue) {
					return 1;
				} else {
					return 0;
				}			
			
			}else {
				throw new IllegalStateException("Comparison operator not recognised");	
			}
		}else if (type.equals("String")) {
			String vals=Value.toString();
			if (comarison_type.equals("=")){
				if (svalue.equals(vals)) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals("<")){
				if (vals.compareTo(svalue)<0) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals(">")){
				if (vals.compareTo(svalue)>0) {
					return 1;
				} else {
					return 0;
				}
			} else if (comarison_type.equals("!=")){
				if (!svalue.equals(vals)) {
					return 1;
				} else {
					return 0;
				}	
			} else if (comarison_type.equals("<=")){
				if (vals.compareTo(svalue)<=0) {
					return 1;
				} else {
					return 0;
				}					
			} else if (comarison_type.equals(">=")){
				if (vals.compareTo(svalue)>=0) {
					return 1;
				} else {
					return 0;
				}			
			
			}else {
					throw new IllegalStateException("Comparison operator not recognised");	
				}
	
		} else {
			throw new IllegalStateException("Type was not recognised, it has to be one of double, int, String");	
		}

	}

	
}
