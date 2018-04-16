#include "epsilon.h"
#include <math.h>
#include "sys.h"

namespace oph {

real epsilon = 1.0e-8;
real user_epsilon = 1.0e-8;
real intersection_epsilon = 1e-6;
real sqrt_epsilon =  1.490116119385000000e-8;
real unset_value = -1.23432101234321e+308;
real zero_tolerance = 1.0e-12;
real zero_epsilon = 1.0e-12;
real angle_tolerance = M_PI/180.0;
real save_zero_epsilon = 1.0e-12;


/*|--------------------------------------------------------------------------*/
/*| Set user epsilon : Throughout the running program we could use the same  */
/*| user epsilon defined here. Default user_epsilon is always 1e-8.          */
/*|--------------------------------------------------------------------------*/
void set_u_epsilon(real a)
{
    user_epsilon = a;
}

void reset_u_epsilon()
{
    user_epsilon = epsilon;
}
void set_zero_epsilon(real a)
{
	save_zero_epsilon = zero_epsilon;
	zero_epsilon = a;
}

void reset_zero_epsilon()
{
	zero_epsilon = save_zero_epsilon;
}

/*|--------------------------------------------------------------------------*/
/*| Approximated version of checking equality : using epsilon                */
/*|--------------------------------------------------------------------------*/
int apx_equal(real x, real y)
{
    int c;
    real a;

    a = real(fabsf(((float)x) - ((float)y)));

    if (a < user_epsilon) c = 1;
    else c = 0;

    return c;
}

/*|--------------------------------------------------------------------------*/
/*| Approximated version of checking equality : using epsilon                */
/*|--------------------------------------------------------------------------*/
int apx_equal(real x, real y, real eps)
{
    int c;
    real a;

    a = real(fabsf(((float)x) - ((float)y)));

    if (a < eps) c = 1;
    else c = 0;

    return c;
}
}; // namespace graphics