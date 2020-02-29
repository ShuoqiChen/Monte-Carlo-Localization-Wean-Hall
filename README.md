# Monte Carlo Localization of Wean Hall
A simple particle filter demo in cooperation with Mosam Dhabi and Sara Misra

**What is Monte Carlo Localization?**

Monte Carlo Localization (MCL), is a popular localization algorithm and known to be an effective application of particle filter for mobile robot localization. To localize the robot, the MCL algorithm uses a particle filter to estimate its position. Each of particle represents a possible robot state, and the particles converge around a single location as the robot moves in the environment as it gets more and more confortable in its state estimation. 

Essentially, the sampling-based algorithm requires knowledge of 3 major parts: the robot motion model, the sensor models, and the method resampling process. 

For more information on MCL, please refere to RI paper: https://www.ri.cmu.edu/pub_files/pub1/dellaert_frank_1999_2/dellaert_frank_1999_2.pdf

**Demo**

![](robotmovie1.gif)
![](robotmovie2.gif)

