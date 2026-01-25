"""Julia set generator without optional PIL-based image drawing"""
import time
from timeit import default_timer as timer
from functools import wraps
import numpy as np
import psutil
import matplotlib.pyplot as plt


# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193

def calc_pure_python(desired_width, max_iterations):
	"""Create a list of complex coordinates (zs) and complex parameters (cs),
	build Julia set"""
	x_step = (x2 - x1) / desired_width
	y_step = (y1 - y2) / desired_width
	x = []
	y = []
	ycoord = y2
	while ycoord > y1:
		y.append(ycoord)
		ycoord += y_step
	xcoord = x1
	while xcoord < x2:
		x.append(xcoord)
		xcoord += x_step
	# build a list of coordinates and the initial condition for each cell.
	# Note that our initial condition is a constant and could easily be removed,
	# we use it to simulate a real-world scenario with several inputs to our
	# function
	zs = []
	cs = []
	for ycoord in y:
		for xcoord in x:
			zs.append(complex(xcoord, ycoord))
			cs.append(complex(c_real, c_imag))

	print("Length of x:", len(x))
	print("Total elements:", len(zs))
	[output,m] = calculate_z_serial_purepython(max_iterations, zs, cs)
	return m

	# This sum is expected for a 1000^2 grid with 300 iterations
	# It ensures that our code evolves exactly as we'd intended
	#assert sum(output) == 33219980

def calculate_z_serial_purepython(maxiter, zs, cs):
	"""Calculate output list using Julia update rule"""
	output = [0] * len(zs)
	n_cores = psutil.cpu_count(logical=True) or 1
	memory = np.empty(n_cores+1)
	t0 = timer()
	for i in range(len(zs)):
		if i%10==0:
			t = timer()-t0
			m = psutil.cpu_percent(interval=1,percpu=True)
			n = np.array(m)
			n = np.insert(n,0,t)
			print("At time:",t," CPU usage:",m)
			memory = np.vstack([memory,n])
		n = 0
		z = zs[i]
		c = cs[i]
		while abs(z) < 2 and n < maxiter:
			z = z * z + c
			n += 1
		output[i] = n
	return output, memory

def Export_memory_stat(m,s1,s2):
	plt.figure()
	for i in range(m.shape[1]-1):
		plt.plot(m[:,0],m[:,i+1],label="Processor - "+str(i))
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Usage (%)")
	plt.title("Memory Usage Patterns")
	np.savetxt(s2, m, delimiter=",", comments="")
	plt.tight_layout()
	plt.savefig(s1)


if __name__ == "__main__":
	# Calculate the Julia set using a pure Python solution with
	# reasonable defaults for a laptop
	m = calc_pure_python(desired_width=100, max_iterations=300)
	Export_memory_stat(m,"Memory Usage Evolution.jpg","Memory Usage.csv")