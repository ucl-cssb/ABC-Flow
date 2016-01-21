#define NSPECIES 1
#define NPARAM 2
#define NREACT 2

#define leq(a,b) a<=b
#define neq(a,b) a!=b
#define geq(a,b) a>=b
#define lt(a,b) a<b
#define gt(a,b) a>b
#define eq(a,b) a==b
#define and_(a,b) a&&b
#define or_(a,b) a||b

__constant__ int smatrix[]={
	//  S1
	   1.0,  
	   -1.0 };

__device__ void hazards(int *y, float *h, float t, int tid){

    h[0] = tex2D(param_tex,0,tid);
    h[1] = tex2D(param_tex,1,tid)*y[0];
}
