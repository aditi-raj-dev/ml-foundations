\## What is a derivative(Intuitive meaning)?



A derivative tells how fast something is changing. It is simply:

-slope of curve at a point

-"if I move a little in x , how much does y change?"



If slope > 0  -> function is increasing

If slope < 0  -> function is decreasing

If slope = 0  -> flat region(possible minimum/ maximum)



In ML , derivative tells: -> "If I change weights slightly, will loss increase or decrease?"





\## Why slope matters in machine learning?



Weights in ML decide predictions. Loss tells how bad predictions are . We want to change weights in the direction that reduces loss.



Slope tells direction:



* positive slope -> move left
* negative slope -> move right
* zero slope -> stop(we reached optimum)



Without slope= you are walking blind.





\## What is a loss function?



Loss measures how wrong the model is.

Smaller loss= better learning.



Examples:

* Mean absolute error -> punishment linear
* Mean squared error -> punishment stronger for big errors 



Loss gives feedback:

->"Model , you are this bad. Fix yourself." 





\## What is gradient descent?

Gradient descent is a method to reduce loss by updating weights step-by-step.



It does:

* Look at slope (gradient)
* Move opposite to slope 
* Repeat again and again
* Eventually reach minimum loss



Formula idea:

new weight = old weight - learning\_rate \* slope





\## Learning rate (very important)

Learning rate=Step size.



Too small->

* model learns slow
* takes forever



Too big ->

* jumps over minimum
* never settles
* training fails



Right learning rate ->

* smooth convergence





\# FINAL UNDERSTANDING

ML "learning" is nothing magical.

It is simply:

Adjusting weights using slope to reduce loss.







