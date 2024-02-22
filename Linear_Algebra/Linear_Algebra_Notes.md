
Vectors are generally *Lists*
Dimensions are *number of members*
**Origin**- intersection of x & y axis

**Coordinates of vectors** - pair of numbers that gets instruction on how to get from tail to tip
1st number corresponds to x axis
2nd number corresponds to y axis

---
**Vector addition**
--
![](https://lh4.googleusercontent.com/zp9MJv4_p2gFq7brFBwH8XN8o2rJGzAEKGPS2ztRPoKSKdwd4YkfzZ1v7M8-_LDVs0lYiXoFu60Ipf74LBVo31wJcaxdFealA2jzCy_F35T8xk3V8woQTuFMQYtws2B7f42a5Rk1BT0FEja3zaqzyZk)

**Scaling**- 
Stretching / squishing / reversing the direction of vectors
i-hat and j-hat are basis vectors

---
**Linear combination** of v & w is given by 
**a v+ b w**
where a & b are scalars

**Span** of v & w are set of linear combination of a v+ b w in an 
- entire 2-D sheet in space 
- sometimes its just a line
- In 3-D all possible combination of those two linear dependent vector  is just a flat space through origin
---
**Linearly dependent** 
When 3rd vector sit on the span of first two vectors

**Linearly independent** 
When 3rd vector moves the span of first two vectors throughout the space

---
**Linear Transformation**
--
- It is a function with which vector is given as input and vector as output is formed
- After transformation, origin remains fixed, grid lines remain parallel and evenly spaced.
- shear x rotation = composition

Linear transformation is determined by where it takes the basis vectors i-hat, j-hat because any vectors can be described as linear combination of those basis vectors

After linear transformation
- x co-ordinate is x times transformed version of i-hat
- y co-ordinate is y times transformed version of j-hat
![](https://lh4.googleusercontent.com/BBbVV0CtP_L1pbId6vsiIpANE4CY5GjydZkPuulDqVPuhNmXOoyYl6TcmhN9DGTi0hZaazjkzPTtEftIdbyM1YkWsX5ZN4otORHI3GoJBhCAkLp4fQXgJQqWNjms0aEmZxYbFz5hdaUBco0Fyg5fvWo)

---
Matrix vector multiplication
![](https://lh4.googleusercontent.com/b4r0ysGrSOT8y6RK_JiWsa_6FIOpdDmv2B597vP0uhl3xujZbQE72kSyTgsLr6vFRPxf2gS0q3QpJxbY7px8w9qrsCsF7zOVIIxP2KlMp6rCp8RIzm25ZFfWMoW122G0i31s5K3QkvErKPLWn2lVM_Y)In 3-D 
Squishing around all points as represented by a grid in such a way that grid lines are parallel , evenly spaced, with a fixed origin
it is represented by a 3 x 3 matrix ![](https://lh5.googleusercontent.com/jI3fh6ElI_2k2-fJNgbkFL1tkxKSFFiUv6pTiTHzL0ja8COFzJ3hviGcRuhoAkTD6DAUDhensCuwCqp1GylrjWcm6uKB5A4Gccv3wovHJRq0sO_9d5tPe3wkqUyoRrjNqjAm2siKuB_emWczY4RsCkc)

---
First column of matrix in right shows where i- hat initially lands. 
Multiplying that column with linear transformation matrix gives final location for i-hat.
same for j-hat
Order with which the matrix is multiplied matters

![](https://lh4.googleusercontent.com/xTDs5M2VrZ8Ujp9YV5khNvYjqd-cAXdZWhfM7AJRDdYeQszUgfPHvMHD2heWIl5JYMlEba6YQnktKlWzp0R1Vx1UWMskPf_239ZGavZv95jnn-VazdmBVVY-l-W_aD_yYXPF_yUFYg9kJXgejsjwduc)

---
Matrix multiplication

![](https://lh6.googleusercontent.com/6C4kSrsngYRoOyV1v7NVPgy6fuRsGMnwY_ti_ruiulrVQ7xcPpM7GUj9yQoaBr41ZBxD-YoFUzHFmmxw3TaV-8BLxs_i1dbeSfx_PwDYh--9V0fQ6SoFO0OxvzLzTph6mrKXIbZ4-0A0XEsFl0HhF24)

**Determinants**
--

It is a scaling factor- a factor by which linear transformation changes any area

- when determinant=0 2-D area squishes into 1-D 
- when determinant<0,  orientation of space is inverted
- In 3-D determinant is the volume of space

Determinant can be calculated as follows
- For 2-D matrices

![](https://lh4.googleusercontent.com/EgX0n1C8CQKF81TYJIZslvzHRdhcbxdWDfJBBDkDZcLW9781TEI8uxSezdbXmcD8VS15zJSWnw89I-wH_vKZH_qy4rSDV-N0REhz51w5-Ykloj3trW3_uv9ZwM88tLkoN509JBgey01l6Ac403QNAic)

- For 3-D matrices

![](https://lh4.googleusercontent.com/bxQcS3Y-tC46fXT5ram3kZMgFb4SsU8MGLjlLWRZlbYxRq3Sx04YZWFXGfynkDYP72KWgNQcNKR6M5ROIiFYiNWub5nGqgJJT5cYBAu3Tr3ygzbZDyBQpGOCB1f-gn6w_2C8n1C66U9sxVpDfhycCUk)**Inverse matrices**
--
vector x on linear transformation with A lands on vector v
![](https://lh5.googleusercontent.com/hWBndMqahJXMY8XbLYelfuTNDoUGPIYN4dAReej8njezP2WfaEhOoXdd4-vxdwFYTxMCF7RkLZYIIjIvO97Bggz69qyPDOfay1HuHlBTuwkQmx9aq7EjuhrNy0Kh2BEtdxQ7KoqdiZQm0Jv29dc9TII) 

---
Reversing linear transformation
If we apply transformation A then apply transformation A-1 , we get beck to initial position
It is called identity transformation

![](https://lh3.googleusercontent.com/2uiqAiAgIfEuwEN6K3c-zmOOXec75uNALQhnOKJX9amgBuZFhVZrzXJaAsDsR52UDdSAyl7UNv705eqBYHJauAzJa8Dp1G0T1Hpx5wb-qGjE8_dewV9PJH_avWb7SZF6GX8fYiTFdu-x9CTgMmLH_ug)If the determinant is zero, inverse of matrix is possible only when vector v is coincident on the line of vector A x.
Otherwise the inverse is not possible. Since determinant zero represents a line and line cannot be squished into a plane

![](https://lh5.googleusercontent.com/jhI2QDxEHkM-74uC7zOLd2nXXiJDglK4nviN8jBOInJcc5v3nxWH0c6HSYhlVon8GVsBGMZIZKqjhIQF-_-MokcwX34yBu7G_2_fnPM9kS3AulvDbDLuVRyZguHDLbWBbjKbRga9ZpJHz0ypx2BENgk)**Rank of a matrix**
number of dimension in a column space

---

**Column space** - tells if a solution even exists
Column of matrix tell where basis vectors usually lands. Span off columns is repesented by column space. so column represents all possible outputs

---

**Full rank**
rank of matrix = number of column

---

**Null space** - what a set of possible solution would look like
When a linear transformation squishes to a smaller dimension, there can be many vector (line of vectors, plane of vectors) that land on origin.
Set of vectors that land on origin is null space/ kernel

![](https://lh3.googleusercontent.com/VoS-dkvzywJO9kwmLkwlXkgwoWLAvrGkqv-uXUbz8PMRu7NJx5tEoKNEpN9AhEYJe7NkallNyEm3Fq_q7iFV_NBlamtrRC_Vp9oGh9Sr0zI7zcMIkNHAYGp2nvYM3l6_aCF_vIVC60Kq5r0XaPJUqNQ)
![](https://lh4.googleusercontent.com/_h4z5caRNQgLolNawdGDPPg3DMucuJ9OQsL7uC8F7TPrwzRfpkxbTCVbl_dKYU_D80DWa6kVabAY0OU5oAuEV18KUA9RbBkCUzQYY1dw2yzQ7IH-_ozX5I_dPorrtQ3SAAbwSWxaPXiy3KEGv_KeKXM)

**Non- square matrix**
--
When  function takes input in 2-D and  gives out output in 3-D
3 X 2 MATRIX → 2 D input gives 3 D output of i and j 

2 x 3 MATRIX → 3 D input of i j k gives 2 D output of i j k 

1 x 2 MATRIX → 2 D input of i and j squeezes to a single number line


![](https://lh3.googleusercontent.com/kOeFGdn8U9gh_JiHURTfFVuPw3nH7PWYUH2d4uWr3fwtxPwf0nUepYZgGv63h_9IU0iKce4QvRZP1usHlVun6PXUXg38Pl_XKFWMF2C4wA3ozNTdMUxse4e_XB7zORLXJw4kRyG7gbRQqgFNSO67PwU)

**Dot products and duality**
--

For two vectors in same direction,dot product is length of projection of w on line passing through vector v and length of v or vice versa
- for same direction, dot product is positive
- for opposite direction, dot product is negative

Duality is correspondence between two different type of math things

![](https://lh5.googleusercontent.com/vFB49O84w1-CnpUfNRgJG-2sicaRECTAbYiOUGW36juWUFMKKjhAvwmL1ZIAzwosM6sG4QDne4ro8Iy8U6n81S5L2MF2cNJHtiFljFGkJTYe4XmWUQ4IeDS1NkihAAxV1yICB5i66Joo2B7DcHWtlWo)

**Cross Product**
--
It combines  two different 3-D vectors to a new 3-D vector. Its direction can be found using right hand rule
![](https://lh4.googleusercontent.com/G3tXCe_Pje7I3w-EIp8Vcxpa2YQj1ljKjz3NiJqVVGTor_FdcKpPZpl4cNhYsZoCmWYKNTGcV4cJjEEzcdkio8tWiln32cx9Da8UGstWfKHJCLBPxEcn0Op4kSTDYZSDExxSFt6Nao_GcRZ1NS5JbKA)Cross product can be found out using determination as it scales the area of the matrix similar to cross product of a matrix
![](https://lh6.googleusercontent.com/2vE40i3T5gve8xhYfvePvsB-huOc3SmT1w29lRNXV5uxPFofBsXVSUnm1bmJgQ-rkKRHJGLG2e6wn6rsDXE0jer4ooaFclyjLgD8Flj0kSWY7cPlul7qNxA1KhR2i97B59yu3dIVQXYklahlWPxyd2o)One of the way to find out cross product is as follows
![](https://lh5.googleusercontent.com/I8A-dHmSP4aCxMJlxG5eZl9aa5UCUp4EpiAxGFwKVGFjXa9kTqFKiwpeS1gxds1fwUDNb8aluR6mAweXcS8zYHmLaXn3czvb3oC9Ze6H4k4-PSRdKwjZW8VLLnFhDCozdWtW97FFQuH_liAPKQPQVvg)**Cramer's Rule**

It is used to find coordinates, which on multiplication with a linear transformation matrix, bring out the given product vector
it can be found by
x = Dx/D
y= Dy/D
z= Dz/D

---
**Change of basis**
--
To find out the coordinates of a vector in another basis, following steps are to be followed
the vector is
- multiplied with change of basis
- Linear transformation is taken place
- multiplied with inverse change of basis
![](https://lh3.googleusercontent.com/324OmXzTtNsYM26SL2NKMPB5UXmKRG1jZvTwRSgH5kkRazG3KQ17tyE7GmCA6Y2a01iv1IAUbVK4cF8ERDpqyYj-_B-ttSL4jIXpz0WPnSbFfcOij5rYC9M9ekUedxsxlbzeMvBUAUKz4tG8OsEWHKI)


**Eigenvectors**
--
 When linear transformation represented by a matrix is applied on a vector, it coincides with its span
 it is a better way to understand what linear transformation does without being dependent on coordinate system
 (vector multiplication)A v = xI v(Scalar multiplication)
 A is eigenvector, x is eigenvalue, I is unit matrix
**Det(Av-xv) = 0** transformation squishes space to lower dimension

in 3 D , eigenvector are the axis of rotation
eigenvalue = 1 : no stretching or squishing happens

 ![](https://lh3.googleusercontent.com/pOTnF18qek71AHGQ5Rxo_PbqJZXPSq0WuMHPtXkwCLkxH5cs3RcVSABusDjwIW32bOTIPkpVT_JyUncanDdzErLjpr9ukolo8XFYvaqQdFxgAQ3uDvYQQAMKoGn9be4-pC1B9-RaTyNvDES2s3OnyXU)One of the way to find eigenvalues ![](https://lh5.googleusercontent.com/KJTd3I8WfeA_VUFknRj9e4TxG9ktj7Qm378fqyq-F42pndmHyhHTXzQi2I3KBniovF8RZEYcINx8HnO0K1ZIS6_BW4yEOHaLtfJ6QPUnMYIaQSUuqgPXHT7AdEjyYHivWLVj_ZCCMhg9w-3KlSrpc2A)An easier way to find eigenvalue![](https://lh5.googleusercontent.com/7bIWuOMlnpLN_6oXAMjg-F8Qw99vqk6kTWxXuSL9yK5sCTnfSn_CTNRU4l5Jt1XcYvtXwYXURqIm0WDVtgbDWnmqp1RvFPnPegE37ofWetf9XtrdIyxq4JKOwb81yIGtwq704z9LcWW6Z00kBgmY6s0)
 
diagonal matrix- all the basis vectors are eigenvectors , diagonal entries as eigenvalues

---

**VECTOR SPACE**:
In mathematics and physics, a vector space is a set whose elements, often called vectors, may be added together and multiplied by numbers called scalars 
- Vectors can also be functions as they can be added or scaled. 
- Vector derivatives are linear operators(transformations)