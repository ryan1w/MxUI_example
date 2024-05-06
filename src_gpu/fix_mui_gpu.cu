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
#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"

#include "atom_vec_meso.h"
#include "fix_mui_gpu.h"
#include "engine_meso.h"
#include "atom_meso.h"
#include "comm_meso.h"

#include "atom.h"
#include "comm.h"
#include "input.h"
#include "variable.h"

#include "mui/mui.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

__global__ void gpu_push_gather(
	double4* __restrict push_buffer,
	uint* __restrict push_count,
	r64* __restrict coord_x,
	r64* __restrict coord_y,
	r64* __restrict coord_z,
	r64* __restrict veloc_x,
	r64* __restrict veloc_y,
	r64* __restrict veloc_z,
	int* __restrict mask,
	const r64 send_upper,
	const r64 send_lower,
	const int  groupbit,
	const int  n_atom )
{
	for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
		if ( ( mask[i] & groupbit ) && coord_z[i] >= send_lower && coord_z[i] <= send_upper ) {
			uint p = atomicInc( push_count, 0xFFFFFFFF );
			double4 info;
			info.x = coord_x[i];
			info.y = coord_y[i];
			info.z = coord_z[i];
			info.w = veloc_x[i];
			push_buffer[p] = info;
		}
	}
}

vector<double4> FixMUIGPU::gpu_push() {
	static int2 grid_cfg;
	static HostScalar<double4> hst_push_buffer(this->lmp,"FixMUI::push_buffer");
	static DeviceScalar<uint>  dev_push_count (this->lmp,"FixMUI::push_count");

	if ( !grid_cfg.x )
	{
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_push_gather, 0, cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_push_gather, cudaFuncCachePreferL1 );
		dev_push_count.grow(1);
	}
	if ( hst_push_buffer.n_elem() < atom->nlocal ) {
		hst_push_buffer.grow( atom->nlocal );
	}

	dev_push_count.set( 0, meso_device->stream() );
	gpu_push_gather<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>>(
		hst_push_buffer,
		dev_push_count,
		meso_atom->dev_coord(0),
		meso_atom->dev_coord(1),
		meso_atom->dev_coord(2),
		meso_atom->dev_veloc(0),
		meso_atom->dev_veloc(1),
		meso_atom->dev_veloc(2),
		meso_atom->dev_mask,
		send_upper,
		send_lower,
		groupbit,
		atom->nlocal );

	uint n;
	dev_push_count.download( &n, 1 );
	meso_device->sync_device();
	vector<double4> result;
	for(int i=0;i<n;i++) result.push_back(hst_push_buffer[i]);
	return result;
}

__global__ void gpu_fetch_pred(
	int* __restrict pred,
	double4* __restrict loc,
	r64* __restrict coord_x,
	r64* __restrict coord_y,
	r64* __restrict coord_z,
	int* __restrict mask,
	const r64 recv_upper,
	const r64 recv_lower,
	const int  groupbit,
	const int  n_atom )
{
	for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
		if ( ( mask[i] & groupbit ) && coord_z[i] >= recv_lower && coord_z[i] <= recv_upper ) {
			pred[i] = 1;
			loc[i].x = coord_x[i];
			loc[i].y = coord_y[i];
			loc[i].z = coord_z[i];
		}
		else
			pred[i] = 0;
	}
}

pair<vector<int>, vector<double4> > FixMUIGPU::gpu_fetch_predicate() {
	static int2 grid_cfg;
	static HostScalar<int>     hst_fetch_pred(this->lmp,"FixMUI::fetch_pred");
	static HostScalar<double4> hst_fetch_loc(this->lmp,"FixMUI::fetch_coord");
	static vector<int> host_buffer;

	if ( !grid_cfg.x )
	{
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fetch_pred, 0, cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_fetch_pred, cudaFuncCachePreferL1 );
	}
	if ( hst_fetch_pred.n_elem() < atom->nlocal ) {
		hst_fetch_pred.grow( atom->nlocal );
		hst_fetch_loc.grow( atom->nlocal );
	}

	gpu_fetch_pred<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>>(
		hst_fetch_pred,
		hst_fetch_loc,
		meso_atom->dev_coord(0),
		meso_atom->dev_coord(1),
		meso_atom->dev_coord(2),
		meso_atom->dev_mask,
		recv_upper,
		recv_lower,
		groupbit,
		atom->nlocal );

	meso_device->sync_device();
	vector<int> result_first;
	vector<double4> result_second;
	for(int i=0;i<hst_fetch_pred.n_elem();i++) {
		result_first.push_back( hst_fetch_pred[i] );
		result_second.push_back( hst_fetch_loc[i] );
	}
	return make_pair(result_first,result_second);
}

__global__ void gpu_scatter_fetch(
	int* __restrict pred,
	double* __restrict vres,
	r64* __restrict veloc_x,
	r64* __restrict veloc_y,
	r64* __restrict veloc_z,
	int* __restrict mask,
	const int  groupbit,
	const int  n_atom )
{
	for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
		if ( pred[i] ) veloc_x[i] += ( vres[i] - veloc_x[i] ) * 1.00;
	}
}

void FixMUIGPU::gpu_fetch( pair<vector<int>, vector<double> > result ) {
	static int2 grid_cfg;
	static HostScalar<int>    hst_pred(this->lmp,"FixMUI::dev_pred");
	static HostScalar<double> hst_vres(this->lmp,"FixMUI::dev_r");

	if ( !grid_cfg.x )
	{
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_scatter_fetch, 0, cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_scatter_fetch, cudaFuncCachePreferL1 );
	}
	if ( hst_pred.n_elem() < atom->nlocal ) {
		hst_pred.grow( atom->nlocal );
		hst_vres.grow( atom->nlocal );
	}

	for(int i=0;i<result.first.size();i++) {
		hst_pred[i] = result.first[i];
		hst_vres[i] = result.second[i];
	}
	gpu_scatter_fetch<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>>(
		hst_pred,
		hst_vres,
		meso_atom->dev_veloc(0),
		meso_atom->dev_veloc(1),
		meso_atom->dev_veloc(2),
		meso_atom->dev_mask,
		groupbit,
		atom->nlocal );
}

mui::point3d point( double4 x ) {
	mui::point3d p;
	p[0] = x.x;
	p[1] = x.y;
	p[2] = x.z;
	return p;
}

FixMUIGPU::FixMUIGPU(LAMMPS *lmp, int narg, char **arg) :
	Fix(lmp, narg, arg),
	MesoPointers(lmp)
{
	// if (narg != 9) error->all(FLERR,"Illegal fix mui arguments");
	interface = new mui::uniface<mui::default_config>( arg[3] );
	send_upper = atof(arg[4]);
	send_lower = atof(arg[5]);
	recv_upper = atof(arg[6]);
	recv_lower = atof(arg[7]);
	sample_rc  = atof(arg[8]);
}

FixMUIGPU::~FixMUIGPU()
{
	if ( interface ) delete interface;
}

int FixMUIGPU::setmask()
{
	int mask = 0;
	mask |= POST_INTEGRATE;
	mask |= END_OF_STEP;
	return mask;
}

void FixMUIGPU::init()
{
}

void FixMUIGPU::post_integrate()
{
	vector<double4> info = gpu_push();

	for (int i = 0; i < info.size(); i++) {
	    // fprintf(screen, "<<<debug gpu push>>> before #%d point loc = %f, %f, %f  velocity = %f\n", info[i].x, info[i].y, info[i].z, info[i].w);
		// fprintf(logfile, "<<<debug gpu push>>> before #%d point loc = %f, %f, %f  velocity = %f\n", info[i].x, info[i].y, info[i].z, info[i].w);
		interface->push( "velocity_x", point(info[i]), info[i].w );
	}

	double time = update->ntimestep * update->dt;
	interface->commit( time );
	interface->barrier( time - 1);
	interface->forget( time - 1 );

}

void FixMUIGPU::end_of_step()
{
	int nlocal = atom->nlocal;

	mui::sampler_shepard_quintic <> quintic(sample_rc);
	mui::temporal_sampler_exact<>       texact(0);

	pair<vector<int>, vector<double4> > pred = gpu_fetch_predicate();
	pair<vector<int>, vector<double> > result;

	double t = update->ntimestep * update->dt;

	// mui::point3d testP;
	// testP[0] = 0.0;
	// testP[1] = 0.0;
	// testP[2] = 0.0;
	
	// double testVal = interface->fetch( "velocity_x", testP, t, quintic, texact );
			
    // fprintf(screen, "<<< GPU debug >>> value is %f at time %f\n", testVal, t);
	// fprintf(logfile, "<<< GPU debug >>> value is %f at time %f\n", testVal, t);

	for (int i = 0; i < nlocal; i++) {
		if ( pred.first[i] ) {
			double res = interface->fetch( "velocity_x", point(pred.second[i]), t, quintic, texact );
			result.second.push_back( res );

			// if (screen)
			// 	fprintf(screen, "<<<debug gpu fetch>>> #%d point loc = %f, %f, %f  velocity = %f\n", pred.second[i].x, pred.second[i].y, pred.second[i].z, res);
			// if (logfile)
			// 	fprintf(logfile, "<<<debug gpu fetch>>> #%d point loc = %f, %f, %f  velocity = %f\n", pred.second[i].x, pred.second[i].y, pred.second[i].z, res);

		} else
			result.second.push_back( 0 );
	}
	result.first = pred.first;

	gpu_fetch( result );
}

