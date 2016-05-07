#define NSPECIES 2
#define NPARAM 7
#define NREACT 4

#define leq(a,b) a<=b
#define neq(a,b) a!=b
#define geq(a,b) a>=b
#define lt(a,b) a<b
#define gt(a,b) a>b
#define eq(a,b) a==b
#define and_(a,b) a&&b
#define or_(a,b) a||b

__constant__ int smatrix[]={
	//  S1   , S2
	   -1.0,   0.0,
	    1.0,   0.0,
	    0.0,  -1.0,
	    0.0,   1.0 };

__device__ void hazards(int *y, float *h, float t, int tid){

    float a = 0;
    if(t>= 5){
      a = tex2D(param_tex,6,tid);
    }
    h[0] = y[0];
    h[1] = tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)/(1 + tex2D(param_tex,2,tid) + pow( y[1], tex2D(param_tex,1,tid)));
    h[2] = (1 + a)*y[1];
    h[3] = tex2D(param_tex,3,tid)*tex2D(param_tex,5,tid)/(1 + tex2D(param_tex,5,tid) + pow( y[0], tex2D(param_tex,4,tid)));
}
