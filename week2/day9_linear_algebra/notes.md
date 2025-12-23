\# Day 9 - Linear algebra intuition for machine learning



\## What a vector represents in ML

In machine learning , a vector represents \*\*one data point\*\*.

Each value in vector is a \*\*feature\*\*.



EXAMPLE:

a student can be represented as 

\[height, weight, age]



So a vector is just a way to store features of one sample in numbers.





---



\## What a matrix represents

A matrix represents \*\* many data points together\*\*.



-Each \*\*row\*\* = one sample (one data point)

-Each \*\*column\*\*=one feature



EXAMPLE:

if we have 100 students and 3 features, the dataset becomes a matrix of shape (100,3).



This is why ML always works with matrices instead of single vectors.



---



\## Dot product - intuition

The dot product measures \*\*how much two vectors point in the same direction\*\*.



-Large dot product -> vectors are similar

-Zero dot product -> vectors are unrelated (perpendicular)

-Negative dot product -> vectors point in opposite directions



In ML , dot product is used to:

-Combine input features with weights

-Compute predictions in linear models



---



\## Why transpose exists 

Transpose switches \*\*rows into columns \*\* and \*\*columns into rows\*\*.



Transpose is needed because:

-Matrix multiplication has strict shape rules

-Sometimes data is in the wrong orientation 

-ML math requires correct alignment of dimensions



Without transpose, many matrix operations would fail.



---



\## Why ML models prefer matrix form 

ML models use matrix equations because :



-Matrix operations are very fast 

-Entire datasets can be processed at once 

-Code becomes shorter and cleaner 

-It scales to large datasets



That is why ML equations look like :

y=XW+b



Instead of loops over each data point.

