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

package utilis;

import java.util.ArrayList;

public class Uniquerowidentifier {
	
	/**
	 * This will store the unique values
	 */
	private ArrayList<Integer> pile [] ; 
	
	/**
	 * size
	 */
	public int size =0;
	/**
	 * Capacity of ArrayLists
	 */
	public int initial_capacity_oflists=10;
	
	/**
	 * Keep current size at iteration count
	 */
	public int iterator_count=0;
	
	
	@SuppressWarnings("unchecked")
	public Uniquerowidentifier(int capacity){
		
		pile=   (ArrayList<Integer>[]) new ArrayList [capacity] ;
	}
	/**
	 * @param x value to add
	 */
	public void add (int x){
		int index=hash(x);
		if (this.pile[index]==null){
			// we definately add it
			this.pile[index]= new ArrayList<Integer>(this.initial_capacity_oflists);
			this.pile[index].add(x);
			size++;
		}else {
			boolean was_not_found=true;
			for (int maybex: this.pile[index]){
					if (maybex==x){
						was_not_found=false;
						return;
					}
			}
			if (was_not_found){
				this.pile[index].add(x);
				size++;
			}
		}
	}
	
	/**
	 * check if it exists
	 * @param f : index to check
	 * @return true if exists
	 */
	public boolean contains(int f){
		
		int index=hash(f);
		if (this.pile[index]==null){
			return false;
		}else {
			for (int maybex: this.pile[index]){
					if (maybex==f){
						return true;
					}
			}
			return false;
		}
	
	}
	
	public int size(){
		return size;
	}

	/**
	 * 
	 * @param h : value to hash
	 * @return : hashed value
	 */
    public int hash(int h) {
    	

        return h % pile.length;
        /*      
    	int k=h;
    	k ^= (k >>> 20) ^ (k >>> 12);
        k= k ^ (k >>> 7) ^ (k >>> 4);

        return k & (pile.length-1);
         */
    }	

}
