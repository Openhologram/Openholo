#ifndef __epsilon_h
#define __epsilon_h

#include "real.h"

namespace oph {

extern real epsilon;
extern real user_epsilon;
extern real intersection_epsilon;

extern real sqrt_epsilon;
extern real unset_value;
extern real zero_tolerance;
extern real angle_tolerance;
extern real zero_epsilon;

#ifndef M_PI
#define M_PI	3.141592653589793238462643383279502884197169399375105820974944592308
#endif


/*|--------------------------------------------------------------------------*/
/*| Set user epsilon : Throughout the running program we could use the same  */
/*| user epsilon defined here. Default user_epsilon is always 1e-8.          */
/*|--------------------------------------------------------------------------*/
void set_u_epsilon(real a);

void reset_u_epsilon();


void set_zero_epsilon(real a);

void reset_zero_epsilon();
/*|--------------------------------------------------------------------------*/
/*| Approximated version of checking equality : using epsilon                */
/*|--------------------------------------------------------------------------*/
int apx_equal(real x, real y);

int apx_equal(real x, real y, real eps);

}; // namespace oph
#endif // !__epsilon_h
