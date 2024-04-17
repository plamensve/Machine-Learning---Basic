import numpy
import stats
import matplotlib.pyplot as plt

"""Data Set"""

data_set = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

"""Machine Learning - Mean Median Mode"""

"""Mean --> (99 + 86 + 87 + 88 + 111 + 86 + 103 + 87 + 94 + 78 + 77 + 85 + 86) / 13 = 89.77"""
speed_mean = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

x_mean = numpy.mean(speed_mean)
print(f"Mean: {x_mean:.1f}")

"""Median    --> The median value is the value in the middle, after you have sorted all the values / If there are two 
numbers in the middle, divide the sum of those numbers by two."""
speed_meadian = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

x_median = numpy.median(speed_meadian)
print(f"Median: {x_median:.1f}")

"""Mode --> The Mode value is the value that appears the most number of times"""
speed_mode = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

x_mode = stats.mode(speed_mode)
print(f"Mode: {x_mode:.1f}")

"""Standard Deviation --> Standard deviation is a number that describes how spread out the values are. Meaning that 
most of the values are within the range of 0.9 from the mean value, which is 86.4"""
speed_std = [86, 87, 88, 86, 87, 85, 86]

x_std = numpy.std(speed_std)

print(f"Standart deviation: {x_std:.1f}")

"""Variance --> Variance is another number that indicates how spread out the values are.
In fact, if you take the square root of the variance, you get the standard deviation!
Or the other way around, if you multiply the standard deviation by itself, you get the variance!"""
speed_variance = [32, 111, 138, 28, 59, 77, 97]

"""1. Find the mean: (32+111+138+28+59+77+97) / 7 = 77.4"""
x_variance = numpy.mean(speed_variance)
print(f"Variance: {x_variance:.1f}")

"""2. For each value: find the difference from the mean:"""
speed_variance_list = []
for num in speed_variance:
    print(f"{num} - {x_variance} = {(num - x_variance):.1f}")
    speed_variance_list.append(num - x_variance)

"""3. For each difference: find the square value"""
square_value_variance = []
for num in speed_variance_list:
    print(f"({num})^2 = {num ** 2:.2f}")
    square_value_variance.append(num ** 2)

"""4. The variance is the average number of these squared differences: --> (
2061.16+1128.96+3672.36+2440.36+338.56+0.16+384.16) / 7 = 1432.2"""
result = sum(square_value_variance) / len(square_value_variance)
print(f"{result:.2f}")

"""Example"""
speed = [32, 111, 138, 28, 59, 77, 97]
x = numpy.var(speed)
print(f"{x:.2f}")

"""Percentiles - Percentiles are used in statistics to give you a number that describes the value that a given 
percent of the values are lower than."""
ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]

x = numpy.percentile(ages, 75)
print(f"Percentile: {x}")

"""Data Distribution"""
x = numpy.random.uniform(0.0, 5.0, 250)

print(x)

"""Histogram"""
x = numpy.random.uniform(0.0, 5.0, 250)

plt.hist(x, 5)
plt.show()

"""Histogram Explained
We use the array from the example above to draw a histogram with 5 bars.

The first bar represents how many values in the array are between 0 and 1.

The second bar represents how many values are between 1 and 2.

Etc.

Which gives us this result:

52 values are between 0 and 1
48 values are between 1 and 2
49 values are between 2 and 3
51 values are between 3 and 4
50 values are between 4 and 5"""

"""Normal Data Distribution -> Note: A normal distribution graph is also known as the bell curve because of it's 
characteristic shape of a bell."""
x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

"""Scatter Plot --> A scatter plot is a diagram where each value in the data set is represented by a dot."""
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]  # The x array represents the age of each car.
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]  # The y array represents the speed of each car.

plt.scatter(x, y)
plt.show()

"""Scatter Plot Explained
The x-axis represents ages, and the y-axis represents speeds.

What we can read from the diagram is that the two fastest cars were both 2 years old, and the slowest car was 12 
years old.

Note: It seems that the newer the car, the faster it drives, but that could be a coincidence, after all we only 
registered 13 cars."""

"""Random Data Distributions"""
"""In Machine Learning the data sets can contain thousands-, or even millions, of values.
You might not have real world data when you are testing an algorithm, you might have to use randomly generated values.
As we have learned in the previous chapter, the NumPy module can help us with that!
Let us create two arrays that are both filled with 1000 random numbers from a normal data distribution.
The first array will have the mean set to 5.0 with a standard deviation of 1.0.
The second array will have the mean set to 10.0 with a standard deviation of 2.0:"""
x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()

"""Scatter Plot Explained
We can see that the dots are concentrated around the value 5 on the x-axis, and 10 on the y-axis.
We can also see that the spread is wider on the y-axis than on the x-axis."""

"""Linear Regression"""
"""Linear regression uses the relationship between the data-points to draw a straight line through all them.
This line can be used to predict future values."""

"""In Machine Learning, predicting the future is very important."""
"""How Does it Work? Python has methods for finding a relationship between data-points and to draw a line of linear 
regression. We will show you how to use these methods instead of going through the mathematic formula.

In the example below, the x-axis represents age, and the y-axis represents speed. We have registered the age and 
speed of 13 cars as they were passing a tollbooth. Let us see if the data we collected could be used in a linear 
regression:"""


x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

plt.scatter(x, y)
plt.show()











































