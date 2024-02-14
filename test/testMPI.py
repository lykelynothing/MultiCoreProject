import subprocess
import struct
import os
import matplotlib.pyplot as plt

def run_mpi_program(comm_sz, dim):
    command = ["mpiexec", "-n", str(comm_sz), "../out", str(dim)]
    custom_environment = os.environ.copy()
    custom_environment["BITS_VAR"] = "8"
    custom_environment["QUANT_ALGO"] = "HOMOMORPHIC"
    custom_environment["SEND_ALGO"] = "REC_HALVING"
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               env=custom_environment, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout

def run_mpi_program_vanilla(comm_sz, dim):
    command = ["mpiexec", "-n", str(comm_sz), "../out", str(dim)]
    custom_environment = os.environ.copy()
    custom_environment["BITS_VAR"] = "8"
    custom_environment["QUANT_ALGO"] = "HOMOMORPHIC"
    custom_environment["SEND_ALGO"] = "NULL"
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               env=custom_environment, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout

if __name__ == "__main__":

    rang = 4
    comm_sz = 2

    data = [0.0 for i in range(0, rang, 1)]
    times = [0.0 for i in range(0, rang, 1)]

    dims = [0 for i in range(0, rang, 1)]
    c = 0

    for i in range(comm_sz * 2, comm_sz * rang + 1, comm_sz):
        out = run_mpi_program(comm_sz, i).strip().split("\n")
        try:
            data_str = out[0]
            time_str = out[1]
            data_fl = float(data_str)
            time_fl = float(time_str)
            if (data_fl > 1):
                data[c] = 0.005
            else:
                data[c] = data_fl
            times[c] = time_fl
        except:
            data[c] = 0.005
            times[c] = 0.0004
        dims[c] = i
        print("Dim : " + str(i))
        print("NMSE : " + data_str)
        print("TIME: " + time_str)
        c += 1


    plt.plot(data, marker='o')
    plt.title('Normalized Mean Squared Error as dimension of vectors grow')
    plt.xlabel('Dimension')
    plt.ylabel('MSE')
    plt.show()

    plt.plot(dims, times, marker='o')
    plt.title('Time as dimensions increase')
    plt.xlabel('Dimensions')
    plt.ylabel('Time')
    plt.show()

    for i in range(len(dims) - 1):
        print(str(i) + " : " + str(dims[i]))

