#ifndef __epsilon_h
#define __epsilon_h

#include "typedef.h"

namespace oph {

extern Real epsilon;
extern Real user_epsilon;
extern Real intersection_epsilon;

extern Real sqrt_epsilon;
extern Real unset_value;
extern Real zero_tolerance;
extern Real angle_tolerance;
extern Real zero_epsilon;

#ifndef M_PI
#define M_PI	3.141592653589793238462643383279502884197169399375105820974944592308
#endif


/*|--------------------------------------------------------------------------*/
/*| Set user epsilon : Throughout the running program we could use the same  */
/*| user epsilon defined here. Default user_epsilon is always 1e-8.          */
/*|--------------------------------------------------------------------------*/
void set_u_epsilon(Real a);

void reset_u_epsilon();


void set_zero_epsilon(Real a);

void reset_zero_epsilon();
/*|--------------------------------------------------------------------------*/
/*| Approximated version of checking equality : using epsilon                */
/*|--------------------------------------------------------------------------*/
int apx_equal(Real x, Real y);

int apx_equal(Real x, Real y, Real eps);

}; // namespace oph
#endif // !__epsilon_h
