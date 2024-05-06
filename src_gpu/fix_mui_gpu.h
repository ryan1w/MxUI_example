/* ----------------------------------------------------------------------
	 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
	 http://lammps.sandia.gov, Sandia National Laboratories
	 Steve Plimpton, sjplimp@sandia.gov

	 Copyright (2003) Sandia Corporation.	Under the terms of Contract
	 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
	 certain rights in this software.	This software is distributed under
	 the GNU General Public License.

	 See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(mui/meso/gpu,FixMUIGPU)

#else

#ifndef LMP_FIX_MUI_GPU_H
#define LMP_FIX_MUI_GPU_H

#include "fix.h"
#include "mui/mui.h"
#include "pointers_meso.h"
#include <vector>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

namespace LAMMPS_NS {

class FixMUIGPU : public Fix, protected MesoPointers {
public:
	FixMUIGPU(class LAMMPS *, int, char **);
	virtual ~FixMUIGPU();
	int setmask();
	virtual void init();
	virtual void post_integrate();
	virtual void end_of_step();

protected:
    mui::uniface<mui::default_config> *interface;
	double send_upper, send_lower, recv_upper, recv_lower, sample_rc;
	std::vector<double4> gpu_push();
	pair<vector<int>, vector<double4> > gpu_fetch_predicate();
	void gpu_fetch( pair<vector<int>, vector<double> > );
};

}

#endif
#endif
