**LINEAR ALGEBRA**

*Linear algebra is built on two basic elements, the matrix and the vector.*

What is a vector:

Physics student: Arrow with a length and direction  →

CS student: A list consisting of numbers [x y z]

Math student: Well it's both of the above.Both the forms are interconvertible.

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.001.png)





**VECTOR ADDITION:**

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.002.png)

It can be realized as joining head of one vector(1st) to tail of other(2nd) and joining tail of 1st to head of 2nd

OR adding like terms of two vectors

**VECTOR MULTIPLICATION:**

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.003.png)

Scalar multiplication can be realized as scaling the length of arrow by a number OR multiplying each term of vector by the number.


**BASIS:** Set of vectors of unit magnitude which are scaled by the scalars .Any vector in plane or space can be represented by linear combination of scaled basis.

For v=xi+yj here i and j are basis and scaled by x and y respectively to form vector v

**SPAN:** Set of vectors that can be obtained by linear combination of  basis.

For any 2 basis that are not parallel the span is whole 2D space

For any 3 basis such that they are linearly independent is whole 3D space

*NOW WHAT DOES LINEARLY INDEPENDENT MEANS??*

If three basis are such that no vector can be expressed as linear combination of two or doesnt le in the plane formed by the other two,they are saidd to be linearly independent.

Else linearly dependent.

Linearly dependent vectors: Any one vector can bee removed without reducing the span of the system.

**If two vectors line up their span is just a line.**

**If three vectors are linearly dependent,their span is a 2D plane.**

So now, **BASIS:** Set of linearly independent vectors that span the full space.

**LINEAR                                    TRANSFORMATIONS:**

`     `**|                                                      |**
**
` `Lines must remain**                   Moving the input to output

evenly spaced lines without

` `change of origin
**
` `In context of linear algebra,it is a function that takes one vector as input and give one vector as output.

**TRANSFORMATION OF ANY VECTOR CAN BE DESCRIBED BY THE TRANSFORMATION OF THE BASIS OF THE SYSTEM ONLY** I.E. where i cap and j cap has landed after the transformation

For example:

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.004.png)

**2D→ 4 NUMBERS     3D→ 9  NUMBERS**

**Transformation of the basis vectors is represented by a matrix each column correspond to co ordinates of resultant basis.** 

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.005.png)

` `***MATRICES GIVES A LANGUAGE TO DESCRIBE THESE TRANSFORMATIONS***


**MATRIX MULTIPLICATION**

When a system undergoes two transformations one after another,the overall effect(final position of basis) can be considered as the product of two matrices indicating co ordinates of basis after each transformation. Also (AB)C=A(BC)

*APPLYING MATRIX MULTIPLICATION HAVE GEOMETRIC MEANING OF APPLYING ONE TRANSFORMATION AFTER ANOTHER.*

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.006.png)

\* **THE DETERMINANT:**

It indicates how much the area(in 2D) and volume(in 3D) of a considered system is scalar by a transformation

Determinant >1 indicates area is increased and <1 indicates the area has decreased

And negative determinant indicates that the area squeezes to 0 and then increases in other direction or in other way the basis vectors crosses each other or the direction of area/volume is reversed. 

ZERO DETERMINANT indicates that the spn of output is squeezed to a smaller dimension or the columns of matrix are linearly dependent.

**det(AB)=det(A).det(B)** as total area scaled will be product of area scaled during two transformations

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.007.png)

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.008.png)

Linear algebra can be used for solving equations:

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.009.png)

**INVERSE TRANSFORMATION:**Unique transformation that when applied to original transformation brings the system back to initial state.i and j cap will be unmoved finally.

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.010.png)

In case the inverse of matrix is not zero,it can be multiplied with v to find out x.

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.011.png)

When A inverse is 0,solution can exist if v vector lies on the line on which the span squeezes.

**RANK:**It is defined as the number of dimensions the span has after transformation.



**COLUMN SPACE:**Set of all possible outputs of a transformation(span of columns of matrix).

SO, ***rank is the number of dimensions in column space.***

FULL RANK MATRIX: Rank equals number of columns.

**NULL SPACE/KERNEL OF MATRIX:** Set of all vectors that becomes zero vector after transformation.

*The idea of column space helps in understanding when a solution exists and null space helps us understand what the set of all possible solutions can look like.*

GEOMETRIC INTERPRETETION:

3X2 MATRIX → 2D input gives 3D output of i and j

2x3 MATRIX → 3D input of i j k gives 2D output of i j k 

1x2 MATRIX → 2D input of i and j squeezes to a single number line  

**DOT PRODUCT:** Projection of one vector over another 

OR product of length of projection and length of vector on which projection occurs. 

`                                          `**A.B=B.A**

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.012.png)

Dot product can be realized as a linear transformation of a space into one dimension.

The 1x2 matrix representing the linear transformation of space into a number line along u cap consists of the co ordinates of u cap.(ux , uy). 

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.013.png)

Applying rank 1 linear transformation to any vector is same as taking its dot product.with a vector of cor ordinates of i and j after transformation.

**DUALITY:** having a correspondence between 2 mathematical things 

Vector in space ←→ linear transformation of that space in one dimension.

It means that linear transformation of any vector v into a number line is same as the dot product of v with dual vector of that transformation

DOT PRODUCT USEFULL FOR: 1.understanding projections 2.testing whether two vectors are in same direction

**CROSS PRODUCT:** Area of the parallelogram spanned by two vectors.

Resultant of cross product is a vector with direction perpendicular to two vectors and magnitude equal to area of parallelogram.

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.014.png)


**GEOMETRIC INTERPRETATION OF CRAMERS RULE:**

x=Dx/D      y=Dy/D     z=Dz/D 

Here x co ordinate is the volume spanned by the vector v with j x k area which is the volume spanned by the same after transformation(can be computed by values) divided by the determinant of matrix as determinant shows the scaling of the original volume.

**CHANGE OF BASIS:** 

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.015.png)

The above example shows that what would the v -1,2 from jennifers co ordinate be in our co ordinate system given the coordinates of basis of her system wrt to ours.

The matrix A represents change from our grid to her grid (-1,2) represents v in her language and answer -4,1 represents v in our language.

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.016.png)

The below example shows v in her language when v is given in our lang

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.017.png)

Consider v given in her language and a transformation matrix in our lang and we have to find final v in her lang so the steps are:

1. Convert v in our lang
1. Transform v to final position in our lang
1. Use A inverse to convert final v to her lang

`    `![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.018.png)

.**EIGEN VECTORS:**They are defined as the vectors which remain on their span(line of action) during the transformation OR in simpler words, during transformation its magnitude can change but its direction remains the same

**EIGEN VALUES:**The factor by which eigenvectors are scaled/sized is known as eigen values 

` `For example v becomes (lambda)v 

**EIGEN BASIS:**The eigen vectors that can be uses as basis for simpler matrix operations are known as eigen basis.

When all basis are eigen, all vectors in the plane will be eigen and the transformation matrix can be represented by a diagonal matrix,each element will represent eigen value of its basis.

In 3D,eigen vectors act as axis of rotation and can conveniently describe a transformation and eigen value will be 1

If v is eigen vector and A is transformation matrix: 

Also (A-lambda I)v=0

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.019.png)

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.020.png)


If we have to compute high powers of a matrix,convert it into eigen basis and compute and again take inverse of conversion for convenience

Complex eigen values(i) represent transformation consists rotation

**For finding eigen values for any matrix:**

Let half of trace(sum of diagonal elements) is m(mean of eigen values) and determinant of matrix is p(product of eigen values) 

**Eigen values= m +/- sqrt(m^2 - p)**

**VECTOR SPACE:** In mathematics and physics, a vector space is a set whose elements, often called vectors, may be added together and multiplied by numbers called scalars

*Vectors can also be functions as they can be added or scaled.*

Derivatives are linear operators(transformations)

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.021.png).

![](Aspose.Words.f2c10b5a-7c40-4f00-9ee9-43ca728ce163.022.png)


` `“ *After learning all phy cs math, if someone asks you what is a vector*

*Say idk and run as fast as possible “*

