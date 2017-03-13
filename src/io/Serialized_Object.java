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

package io;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;


public class Serialized_Object implements Serializable {
	
	private static final long serialVersionUID = 1L;

	/**
	 * Load an object.
	 * 
	 * @param filename: The name of the file.
	 * @return The loaded object.
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static Serializable load(final String filename) throws IOException,
			ClassNotFoundException {
		Serializable object;
		FileInputStream fis = null;
		ObjectInputStream in = null;
		fis = new FileInputStream(filename);
		in = new ObjectInputStream(fis);
		object = (Serializable) in.readObject();
		in.close();
		return object;
	}

	/**
	 * Save the specified object. It needs to implement serialised
	 * @param filename : The filename to save.
	 * @param object : The object to save.
	 * @throws IOException : 
	 */
	public static void save(final String filename, final Serializable object)
			throws IOException {
		FileOutputStream fos = null;
		ObjectOutputStream out = null;

		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(object);
		out.close();
	}

}
