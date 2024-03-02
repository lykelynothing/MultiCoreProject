import subprocess
import struct
import os
import matplotlib.pyplot as plt

def run_mpi_program(comm_sz, dim):
    command = ["mpiexec", "-n", str(comm_sz), "../out", str(dim)]
    custom_environment = os.environ.copy()
    custom_environment["BITS_VAR"] = "8"
    custom_environment["QUANT_ALGO"] = "LLOYD"
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

    # power of 10 x up to which (dimension = comm_sz * 10^x) gets to
    rang = 6
    comm_sz = 2
    factor = 100000


    data = [0 for i in range((comm_sz * 2) * pow(10, 2), (comm_sz * 2) * pow(10, rang), (comm_sz * 2) * factor)]
    times = [0 for i in range((comm_sz * 2) * pow(10, 2), (comm_sz * 2) * pow(10, rang), (comm_sz * 2) * factor)]
    times_v = [0 for i in range((comm_sz * 2) * pow(10, 2), (comm_sz * 2) * pow(10, rang), (comm_sz * 2) * factor)]


    dims = [0 for i in range((comm_sz * 2) * pow(10, 2), (comm_sz * 2) * pow(10, rang), (comm_sz * 2) * factor)]
    c = 0

    for dim in range((comm_sz * 2) * pow(10, 2), (comm_sz * 2) * pow(10, rang), (comm_sz * 2) * factor):
        out = run_mpi_program(comm_sz, dim).strip().split("\n")
        try:
            data_str = out[0]
            time_str = out[1]
            time_str_v = out[2]
            data_fl = float(data_str)
            time_fl = float(time_str)
            time_fl_v = float(time_str_v)

            data[c] = data_fl
            times[c] = time_fl
            times_v[c] = time_fl_v
        except:
            print('Failure')
            data[c] = 0.005
            times[c] = 0.0004
        dims[c] = dim
        print("Dim : " + str(dim))
        print("NMSE : " + data_str)
        print("TIME: " + time_str)
        print("TIME VANILLA: " + time_str_v)
        c += 1


    plt.plot(dims, data, marker='o')
    plt.title('Normalized Mean Squared Error as dimension of vectors grow')
    plt.xlabel('Dimension')
    plt.ylabel('MSE')
    plt.show()

    plt.plot(dims[:-1], times[:-1], marker='o', label='Custom all reduce', markersize=4)
    plt.plot(dims[:-1], times_v[:-1], marker='o', label='Vanilla all reduce', markersize=4)
    plt.title('Time as dimensions increases (n = ' + str(comm_sz) + ')')
    plt.xlabel('Dimensions')
    plt.ylabel('Time')
    plt.legend()
    plt.show()


