/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------------*/ 
#include "stdio.h"
#include "string.h"
#include "fix_micro_mui.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "input.h"
#include "variable.h"
#include "error.h"
#include "force.h"
#include "region.h"
#include "stdlib.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */

FixMicroMUI::FixMicroMUI(LAMMPS *lmp, int narg, char **arg):
  Fix(lmp, narg, arg),
  interface(NULL)
{
  if (narg != 12) error->all(FLERR,"Illegal fix micro/mui command"); 
 
//  printf("fix is ready\n");
  interface = new mui::uniface<config>( arg[3] );
//  printf("here is ok\n");
  iregion1=domain->find_region(arg[4]);
  iregion2=domain->find_region(arg[5]);
  sample_rc  = atof( arg[6] );
  step_ratio=atof(arg[7]);
  l_ratio=atof(arg[8]);
  v_ratio=atof(arg[9]);
  t_ratio=atof(arg[10]);
  tol=atof(arg[11]);
  
//   printf("MD ");
//  printf("step=%f",l_ratio);
  // printf("step=%f\n",v_ratio);
}

/* ---------------------------------------------------------------------- */

FixMicroMUI::~FixMicroMUI()
{
   if ( interface ) delete interface;
}

/* ---------------------------------------------------------------------- */

int FixMicroMUI::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMicroMUI::init()
{
}

/* ---------------------------------------------------------------------- */

void FixMicroMUI::post_integrate()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nall = atom->nlocal;
  Region *region1=domain->regions[iregion1];
  Region *region2=domain->regions[iregion2];
  double vave;
  int ck;
  double fake[3]={0.0,0.0,0.0};
  
  
  int step_coupling=((update->ntimestep-1)-(update->ntimestep-1)%step_ratio)/step_ratio;
  int next_coupling=(step_coupling+1)*step_ratio;
  double bound=abs(update->ntimestep-next_coupling);
 
//  printf("timestep = %d , bound = %f\n",update->ntimestep, bound*update->dt);    

//  if(bound*update->dt <= 0.2){
 // if(update->ntimestep%step_ratio == 0 ||update->ntimestep%step_ratio>=(step_ratio-10) ){
   vave=0.0;
   ck=0;
    for (int i = 0; i < nall; i++){
     if ( (mask[i] & groupbit) && region1->match(x[i][0],x[i][1],x[i][2]) ){
      
       vave+=v[i][0];
       ck++;
     //  printf("k=%d\n",k);
       // interface->push( "v_x", point(x[i]), real(v[i][0]) );
      }
    }
    if(ck!=0){
    	
      vave=vave/ck;
      interface->push("v_x",point(fake),real(vave));
     }
   double time = update->ntimestep * update->dt*t_ratio;
   interface->commit( time );
   interface->barrier( time-1);
   interface->forget( time - 1 );
 
   double outtime=update->ntimestep;
}  


void FixMicroMUI::end_of_step()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  Region *region1=domain->regions[iregion1];
  Region *region2=domain->regions[iregion2];
  
  int step_coupling=update->ntimestep-update->ntimestep%step_ratio;
  double time_coupling=step_coupling*update->dt*t_ratio;
  double dt=update->dt;
  
  double sample_rc2=sample_rc/l_ratio;
  mui::sampler_shepard_quintic <config,real,real> quintic(sample_rc2);
  mui::temporal_sampler_exact <config> texact(tol);

  double time = update->ntimestep * update->dt*t_ratio;
  point p;
  	p[0]=0.0;
  	p[1]=0.0;
  	p[2]=0.0;
   double recv0 = interface->fetch( "v_x", p, time_coupling, quintic, texact);
   double recv=recv0*v_ratio;
   

  for (int i = 0; i < nlocal; i++){
  if ( ( mask[i] & groupbit ) && region2->match(x[i][0],x[i][1],x[i][2])  ) {

     v[i][0] += ( recv - v[i][0] ) * 1.0;
 }
   
  }
 
double outtime=update->ntimestep;

}
