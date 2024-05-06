cd ../src_gpu

# make clean-all
# make yes-molecule
# make yes-user-meso
make meso ARCH=sm_70 -j8

cd ../dpd_coupling



cd ../src_dpd

make mpi -j8

cd ../dpd_coupling