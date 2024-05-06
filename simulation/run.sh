#!/bin/bash

#PBS -I -N GPUrun -q muthcomp -l select=1:ncpus=16:ngpus=2:mem=16gb:interconnect=fdr,walltime=2:00:00
#PBS -j oe

# cd $PBS_O_WORKDIR

# mpirun -np 1 lmp_meso_gpu -i in.dpdmeso -meso off : -np 1 lmp_meso_gpu -in in.mdpd -meso on > log.out &


mpirun -np 1 lmp_meso_gpu -meso on -in in.mdpd : -np 1 lmp_mpi_dpd -i in.dpd > log.out &

echo "waitting for process to finish"
wait
