from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

size = MPI.COMM_WORLD.Get_size()
print(f"[rank {rank}/{size}] hello from node")
