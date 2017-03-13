
/**
 * <p> package to implement Logistic Regression .
 * Different implementations will be supported:  </p>
 * <ol>
 * <li> Routine method with Linear Algebra NewtonR optimisation method</li>
 * <li> SGD Method</li>
 * <li> LibLinear Method (coordinate Ascent)</li>
 * <li> FTRL</li>
 * </ol>
 * <p>The (binary) logistic regression is a model commonly used to predict the chances
 *  of a dependent variable resulting into Good=1 or Bad=0. The general function is the following:
 *  <pre>f(x)=e<sup>(X0+X1+X2+...Xn)</sup>/(1+e<sup>(X0+X1+X2+...Xn)</sup>)
 *  
 *   Where,f(x) can take any value between 0 and 1,
 *   
 *   while xn represents the characteristics of the model and bn their coefficients. 
 *   
 *  (Samprit et al.2000/p:321)
 *    </pre>
 *  <p>  The equation is non-linear for b0…bn, however it can be transformed
 *     by replacing the ð value with the <pre>p/(1-p)</pre>, which represents the probability of an event to happen, 
 *    divided with the probability of not happening and it is called the odds ratio. </p> 
 *   <p> By using natural logarithm on both sides the initial equation is transformed to :
 *   <pre> log(p/1-p)=b0 + b1x1 + b2x2 +....bnxn
 *   
 *   (Samprit et al.2000/p:321)</pre>
 *<p> The coefficients in the logistic regression are (most commonly) calculated by
 *  using the method of maximum likelihood. 
 *  According to Thomas R. (1997), “the likelihood function is in general
 *  , defined as the joint probability function of the random variables whose 
 *  realizations constitute the sample”. 
 *  For Yn variables, the joint probability function can be written as:
 *   <pre>g(Y<sub>ð</sub>)=Ð<sub>i=1</sub>f<sub>i</sub>(Y<sub>i</sub>)=Ð<sub>i=1</sub>ð<sub>i</sub><sup>Y<sub>i</sub></sup>(1-ð<sub>i</sub>)<sup>1-Y<sub>i</sub></sup>
 *   (Thomas R.,1997:258)</pre>
 *   <p>In other words it expresses the probability of particular sequences of 0s and 1s.
 *   The log of this function is frequently used to access how much the model has explained
 *   <p> The multinomial model that accepts more outcomes than 2, gets reduced to an 1 vs all Binary loguistic problems</p>
 *   <p> The logic to run the algorithm will be to create K-1 Binary logistic regression models (where
 *K is the number of distinct values of our target category) by converting the target category into binary
 *(e.g. if it is the category we want then 1 else 0). We already know that the sum of the probabilities in 
 *all categories  must be 1, therefore, by knowing the probabilities of (k-1), we can easily infer the probability of K.
 *
 * <p> The probabilities of any of the (k-1) factors-categories can be seen as the odds of an event occurring <em> more or less </em>
 * than the Kth event of the factor-category that will be left aside. Therefore the probability of any of the K factors can be computed as:
 * <pre>P(y<sub>i</sub>=K) = e<sup>X<sub>i</sub></sup><sup>B<sub>K</sub></sup>/ (1+ S e<sup>X<sub>i</sub></sup><sup>B<sub>aside</sub></sup>)
 * 
 *While the the probability of the K factor is :
 *
 *P(y<sub>i</sub>=K) = 1/ (1+ S e<sup>X<sub>i</sub></sup><sup>B<sub>K</sub></sup>)
 *
 *Where B is the regerssion's coefficient , X the data attribute and i the observation.
 * </pre>(William H. Greene, 2003/p.720-722)
 *   
 */

package ml.LogisticRegression;