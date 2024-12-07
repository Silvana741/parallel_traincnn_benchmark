from mpi4py import MPI
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
import time

def load_data(file_path):
    file_path = "mnist.npz" 
    mnist_data = np.load(file_path)

    # Extract train and test sets
    x_train = mnist_data['x_train']
    y_train = mnist_data['y_train']
    x_test = mnist_data['x_test']
    y_test = mnist_data['y_test']

    return (x_train, y_train), (x_test, y_test)

# Define convolution function using JAX
def convolution_2d(x, kernel):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Pad the input array by adding extra pixel
    padded_x = jnp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output matrix
    output_data = jnp.zeros_like(x)

    # Perform the convolution operation
    for i in range(input_height):
        for j in range(input_width):
            # Extract the region of interest
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and summation
            output_data = output_data.at[i, j].set(jnp.sum(region * kernel))

    return output_data

# Define loss function
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d(x, kernel)
    return jnp.mean((y_pred - y_true) ** 2)  # Mean squared error

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load the MNIST dataset
if rank == 0:
    (x_train, y_train), (x_test, y_test) = load_data('mnist.npz')
    x = x_train[0]
    y_true = x.copy()

    # Add salt-and-pepper noise
    num_corrupted_pixels = 100
    for _ in range(num_corrupted_pixels):
        i, j = np.random.randint(0, x.shape[0]), np.random.randint(0, x.shape[1])
        x[i, j] = np.random.choice([0, 255])

    # Normalize images
    y_true = y_true.astype(np.float32) / 255.0
    x = x.astype(np.float32) / 255.0
else:
    x = None
    y_true = None

# Broadcast the data to all processes
x = comm.bcast(x, root=0)
y_true = comm.bcast(y_true, root=0)

# Initialize kernel
kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])  # Random kernel for horizontal edge detection

# Gradient of the loss function w.r.t. the kernel
loss_grad = grad(loss_fn)

# Benchmarking
learning_rate = 0.01
num_iterations = 10
benchmark_runs = 30

execution_times = []
cpu_times = []

for _ in range(benchmark_runs):
    # Start timing
    start_wall_time = time.perf_counter()
    start_cpu_time = time.process_time()

    # Divide iterations among processes
    iterations_per_process = num_iterations // size
    start_iteration = rank * iterations_per_process
    end_iteration = start_iteration + iterations_per_process

    for i in range(start_iteration, end_iteration):
        gradients = loss_grad(kernel, x, y_true)
        kernel -= learning_rate * gradients  # Update kernel with gradient descent

    # End timing
    end_wall_time = time.perf_counter()
    end_cpu_time = time.process_time()

    execution_times.append(end_wall_time - start_wall_time)
    cpu_times.append(end_cpu_time - start_cpu_time)

# Gather timing data
all_execution_times = comm.gather(execution_times, root=0)
all_cpu_times = comm.gather(cpu_times, root=0)

if rank == 0:
    # Compute averages
    avg_execution_time = np.mean([time for sublist in all_execution_times for time in sublist])
    avg_cpu_time = np.mean([time for sublist in all_cpu_times for time in sublist])

    # Log results
    print(f"Processes: {size}, Average Execution Time: {avg_execution_time:.4f} s, Average CPU Time: {avg_cpu_time:.4f} s")
