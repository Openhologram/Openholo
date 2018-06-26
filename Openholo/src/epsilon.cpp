#include "epsilon.h"
#include <math.h>
#include "sys.h"
#include "define.h"

namespace oph {

Real epsilon = 1.0e-8;
Real user_epsilon = 1.0e-8;
Real intersection_epsilon = 1e-6;
Real sqrt_epsilon =  1.490116119385000000e-8;
Real unset_value = -1.23432101234321e+308;
Real zero_tolerance = 1.0e-12;
Real zero_epsilon = 1.0e-12;
Real angle_tolerance = M_PI/180.0;
Real save_zero_epsilon = 1.0e-12;


/*|--------------------------------------------------------------------------*/
/*| Set user epsilon : Throughout the running program we could use the same  */
/*| user epsilon defined here. Default user_epsilon is always 1e-8.          */
/*|--------------------------------------------------------------------------*/
void set_u_epsilon(Real a)
{
    user_epsilon = a;
}

void reset_u_epsilon()
{
    user_epsilon = epsilon;
}
void set_zero_epsilon(Real a)
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
int apx_equal(Real x, Real y)
{
    int c;
    Real a;

    a = Real(fabsf(((float)x) - ((float)y)));

    if (a < user_epsilon) c = 1;
    else c = 0;

    return c;
}

/*|--------------------------------------------------------------------------*/
/*| Approximated version of checking equality : using epsilon                */
/*|--------------------------------------------------------------------------*/
int apx_equal(Real x, Real y, Real eps)
{
    int c;
    Real a;

    a = Real(fabsf(((float)x) - ((float)y)));

    if (a < eps) c = 1;
    else c = 0;

    return c;
}
}; // namespace graphics