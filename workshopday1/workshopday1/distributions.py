# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 22:07:21 2017

@author: Dr.Srinivas
"""
import numpy as np
import scipy as stats
from scipy.stats import binom
import matplotlib.pyplot as plt 

#Normal Distribution
#Theory - let us look at http://www.di.fc.ul.pt/~jpn/r/distributions/index.html 

mu =-1
sigma =1
x = np.arange(-5,5,0.1)

from scipy.stats import norm
y= norm.pdf(x,mu,sigma)
plt.plot(x,y)
plt.title('Normal: $\mu$=%.1f, $\sigma^2$=%.1f' % (mu, sigma))
plt.xlabel('x')
plt.ylabel('Probability density') # probability of observing each of these observations
plt.show()



#Binomial distribution
n = 20
p = 0.4
k = np.arange(0,21)
binomial = binom.pmf(k, n, p)
binomial

plt.plot(k, binomial, 'o-')
plt.title('Binomial: n=%i, p=%.2f' % (n,p), fontsize=15)
plt.xlabel('Number of Successes')
plt.ylabel('Probability of Successes', fontsize=15)
plt.show()

#Simulating a binomial random variable using .rvs
binom_sim = data = binom.rvs(n = 10, p =0.3, size = 10000)
print ("Mean: %g" % np.mean(binom_sim))
print ("SD: %g" % np.std(binom_sim, ddof=1))
plt.hist(binom_sim, bins=10, normed=True) #normed
plt.xlabel("x")
plt.ylabel("density")
plt.show()

from scipy.stats import poisson
#Poisson Distribution
rate = 2
n = np.arange(0,10)
y = poisson.pmf(n,rate)
y
plt.plot(n, y, 'o-')
plt.title('Poisson: $\lambda$ =%i' % rate)
plt.xlabel('Number of Accidents')
plt.ylabel('Probability of number of accidents')
plt.show()

#Simulating 1000 random variables from a Poisson distribution

data = poisson.rvs(mu = 2, loc = 0, size=1000)
print ("Mean: %g" % np.mean(data))
print ("SD: %g" % np.std(data, ddof=1))

plt.figure()
plt.hist(data, bins=9, normed = True)
plt.xlim(0,10)
plt.xlabel("Number of Accidents")
plt.title("Simulating Poisson Random Variables")
plt.show()





#Beta Distribution
a = 0.5
b = 0.5
x = np.arange(0.01,0.01)
from scipy.stats import beta
y = beta.pdf(x,a,b)
plt.plot(x, y)
plt.title('Beta: a=%.1f, b=%.1f' % (a,b))
plt.xlabel('x')
plt.ylabel('Probability density')
plt.show()
#uniform distribution




#Exponential Distribution
lambd = 0.5
x = np.arange(0, 15, 0.1)
y = lambd * np.exp(-lambd * x)
plt.plot(x,y)
plt.title("Exponential: $\lambda$ = %.2f" % lambd)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.show()

#Simulating 1000 random variables from an exponential distribution
#.scale is the inverse of lambda
from scipy.stats import expon
data = expon.rvs(scale = 2, size = 1000)
print ("Mean: %g" % np.mean(data))
print ("SD: %g" % np.std(data, ddof=1))

plt.figure()
plt.hist(data, bins=20, normed=True)
plt.xlim(0,15)
plt.title("Simulating Exponential Random Variables")
plt.show()