# PYMLLIB ™ 

[![Build Status](https://travis-ci.org/stfnwong/pymllib.svg?branch=master)](https://travis-ci.org/stfnwong/pymllib)

## A machine learning library built on Numpy

This is a simple machine learning library written in Python. I wrote this while following the 2015/2016 version of CS231n. In effect, this is a solution to that version of the class but packaged like a library. 

There is a little bit of Cython, but otherwise its quite slow and janky.


## Requirements 
- Developed and Tested on Fedora. I had Fedora 27 or 28 at the start, but seems to work on Fedora 30. This should work on any reasonable distro configuration, and also on the majority of non reasonable distro configurations, BSD, etc.
- Should probably work on OSX but not tested.
- Python 3.6
- Numpy 1.12.x
- See requirements.txt for full list of libraries. There is nothing fancy used, most Python 3.x installations will be fine.


## Setup 
To setup the cython solver, run

"python3 setup.py build_ext --inplace"

## Guides 
Haven't written any yet. For the sake of completeness I ought to do that, as well as merge in the other branches that I have left dangling and unfinished. I mostly dropped this when I started working on LERNOMATIC.
