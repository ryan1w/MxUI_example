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
#include "fix_meso_mui.h"
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

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */

FixMesoMUI::FixMesoMUI(LAMMPS *lmp, int narg, char **arg):
    Fix(lmp, narg, arg),
    interface(NULL)
{
    if (narg != 12) error->all(FLERR,"Illegal fix meso/mui command"); 

    interface = new mui::uniface<config>( arg[3] );
    iregiona=domain->find_region(arg[4]);
    iregionb=domain->find_region(arg[5]);
    sample_rc  = atof( arg[6] );
    step_ratio = atof( arg[7] );
    l_ratio=atof(arg[8]);
    v_ratio=atof(arg[9]);
    t_ratio=atof(arg[10]);
    tol=atof(arg[11]);
}
  
/* ---------------------------------------------------------------------- */

FixMesoMUI::~FixMesoMUI()
{
    if ( interface ) delete interface;
}

/* ---------------------------------------------------------------------- */

int FixMesoMUI::setmask()
{
    int mask = 0;
    mask |= POST_INTEGRATE;
    mask |= END_OF_STEP;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixMesoMUI::init()
{
}

/* ---------------------------------------------------------------------- */

void FixMesoMUI::post_integrate()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nall = atom->nlocal;
    Region *regiona=domain->regions[iregiona];
    Region *regionb=domain->regions[iregionb];

    double time = update->ntimestep * update->dt * t_ratio;

    double vave = 0.0;
    int ck = 0;
  
    double fake[3]={0.0, 0.0, 0.0};
    for (int i = 0; i < nall; i++)
    {
        if ( (mask[i] & groupbit) &&  regiona->match(x[i][0],x[i][1],x[i][2]) ){
        vave+=v[i][0];
        ck++;
        interface->push( "velocity_x", point(x[i]), real(v[i][0]) );
        }
    }
        if(ck!=0){ 
            vave=vave/ck;

        // interface->push("velocity_x",point(fake),real(9.9));
        }

    interface->commit( time );
    interface->barrier( time - 1);
    interface->forget( time - 1);
}

void FixMesoMUI::end_of_step()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    Region *regiona=domain->regions[iregiona];
    Region *regionb=domain->regions[iregionb];
  
    double time=update->ntimestep * update->dt * t_ratio;
    
    double sample_rc2 = sample_rc * l_ratio;
    mui::sampler_shepard_quintic <config,real,real> quintic(sample_rc2);
    mui::temporal_sampler_exact <config> texact(tol);
  	
    double recv0 = 0.0;
    double recv = 0.0;
    for (int i = 0; i < nlocal; i++){
        recv0 = interface->fetch( "velocity_x", point(x[i]), time, quintic, texact);
        recv =recv0/v_ratio;

        if ( ( mask[i] & groupbit ) && regionb->match(x[i][0],x[i][1],x[i][2])  ) {
            v[i][0] += ( recv - v[i][0] ) * 1.0;
            }
    }
}

