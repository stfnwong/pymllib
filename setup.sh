# Setup the Cython modules
python3 setup.py build_ext --inplace

# Get dataasets 
cd datasets/

# CIFAR-10 dataset
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
