#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include "3d.cl/3d_funcs.h"

#define  CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include "cl_tools.h"

#define  fnm "mgpu.nc"

#define  ond 4 // number of dimension of output variable

#define  err(e)         {printf("error : %s\n",nc_strerror(e)); return (2);}
#define  dbg(line)      {printf("here ? line : %d\n",line);}
#define  clerr(arg,e)   {printf("   %-40s : %s\n",arg,getclerr(e));}

int main()
{
int      t,ctr=0;

int      nt =  100,
         snt=  200,
         nz =  1,
         ny =  1024,
         nx =  1024,
         sz =  nz,
         sy =  ny,
         sx =  nx/2,
         pz =  sz+2,
         py =  sy+2,
         px =  sx+2,
         iz =  1,
         iy =  256,
         ix =  256,
         oz =  0,
         oy =  256,
         ox =  256;

float    u  =  2.0,
         v  =  2.0,
         w  =  0.0,
         c  =  0.05;

float    *an0   =  malloc(sizeof(float) * nz * ny * nx),
         *pn0p0 =  malloc(sizeof(float) * pz * py * px), 
         *pn0p1 =  malloc(sizeof(float) * pz * py * px),
         *pn1p0 =  malloc(sizeof(float) * pz * py * px), 
         *pn1p1 =  malloc(sizeof(float) * pz * py * px);


// netcdf variables
int      ec, nid, z_did, y_did, x_did, t_did, ovid,
         dids[ond];

size_t   start[ond]  =  {0, 0, 0, 0},
         count[ond]  =  {1,nz,ny,nx};

// opencl related variables
size_t   psiz           =  sizeof(float) * pz * py * px,
         nsbsiz         =  sizeof(float) * sy * sx,
         websiz         =  sizeof(float) * sz * sy,
         udbsiz         =  sizeof(float) * sz * sx,
         glwksiz[3]     =  {sx,sy,sz},
         lcwksiz[3]     =  {4,8,8},
         advgwsiz[3]    =  {sx,sy,sz},
         advofset[3]    =  {1,1,1};

cl_int            cle;
cl_uint           nplf, ndev;
cl_context        ctx;
cl_device_id      devs[4];
cl_platform_id    plf[1];
cl_command_queue  cq[2];

cl_program        getb_pg[2], bexchg_pg[2], adv_pg[2];
cl_kernel         getb_kn[4], bexchg_kn[4], adv_kn[4];
cl_event          getb_ev[4], mim_ev[4], bexchg_ev[4], adv_ev[4];
cl_mem            brr[4], b0[6], b1[6];


getb_ev[0]  =  0;
getb_ev[1]  =  0;
getb_ev[2]  =  0;
getb_ev[3]  =  0;

mim_ev[0]   =  0;
mim_ev[1]   =  0;
mim_ev[2]   =  0;
mim_ev[3]   =  0;

bexchg_ev[0]   =  0;
bexchg_ev[1]   =  0;
bexchg_ev[2]   =  0;
bexchg_ev[3]   =  0;

adv_ev[0]   =  0;
adv_ev[1]   =  0;
adv_ev[2]   =  0;
adv_ev[3]   =  0;


// create nc file
if ((ec  =  nc_create(fnm, NC_NETCDF4, &nid))) err(ec);

// create dimensions
if ((ec  =  nc_def_dim(nid, "t", NC_UNLIMITED  , &t_did))) err(ec);
if ((ec  =  nc_def_dim(nid, "z", nz            , &z_did))) err(ec);
if ((ec  =  nc_def_dim(nid, "y", ny            , &y_did))) err(ec);
if ((ec  =  nc_def_dim(nid, "x", nx            , &x_did))) err(ec);

dids[0]  =  t_did;
dids[1]  =  z_did;
dids[2]  =  y_did;
dids[3]  =  x_did;

// create variables
if ((ec  =  nc_def_var(nid, "data", NC_FLOAT, ond, dids, &ovid))) err(ec);

if ((ec  =  nc_def_var_deflate(nid, ovid, 1, 1, 5))) err(ec);

if ((ec  =  nc_enddef(nid))) err(ec);


// initializing OpenCL things : platforms and devices
// assuming 1 platform
cle   =  clGetPlatformIDs(0, NULL, &nplf);
cle   =  clGetPlatformIDs(nplf, plf, NULL);
cle   =  clGetDeviceIDs(plf[0], CL_DEVICE_TYPE_GPU, 0, NULL, &ndev);
cle   =  clGetDeviceIDs(plf[0], CL_DEVICE_TYPE_GPU, ndev, devs, NULL);

// initializing OpenCL things : context and command queues
ctx   =  clCreateContext(NULL, ndev, devs, NULL, NULL, &cle);
cq[0] =  clCreateCommandQueueWithProperties(ctx, devs[0], 0, &cle);
cq[1] =  clCreateCommandQueueWithProperties(ctx, devs[1], 0, &cle);

// buffers
brr[0]=  clCreateBuffer(ctx, CL_MEM_READ_WRITE, psiz , NULL, &cle); // bn0p0
brr[1]=  clCreateBuffer(ctx, CL_MEM_READ_WRITE, psiz , NULL, &cle); // bn1p0
brr[2]=  clCreateBuffer(ctx, CL_MEM_READ_WRITE, psiz , NULL, &cle); // bn0p1
brr[3]=  clCreateBuffer(ctx, CL_MEM_READ_WRITE, psiz , NULL, &cle); // bn1p1

b0[0] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, nsbsiz, NULL, &cle);
b0[1] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, nsbsiz, NULL, &cle);
b0[2] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, websiz, NULL, &cle);
b0[3] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, websiz, NULL, &cle);
b0[4] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, udbsiz, NULL, &cle);
b0[5] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, udbsiz, NULL, &cle);

b1[0] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, nsbsiz, NULL, &cle);
b1[1] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, nsbsiz, NULL, &cle);
b1[2] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, websiz, NULL, &cle);
b1[3] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, websiz, NULL, &cle);
b1[4] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, udbsiz, NULL, &cle);
b1[5] =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, udbsiz, NULL, &cle);


// initialize array
arr_init_3d(an0,nz,ny,nx,iz,iy,ix,oz,oy,ox);
dcmp_2pc_3d(an0,pn0p0,pn0p1,nz,ny,nx);
dcmp_2pc_3d(an0,pn1p0,pn1p1,nz,ny,nx);


start[0] =  0;
if ((ec = nc_put_vara_float(nid, ovid, start, count, &an0[0]))) err(ec);



// write main arrays into buffers
// device 0
cle   =  clEnqueueWriteBuffer(cq[0], brr[0], CL_TRUE, 0, psiz, pn0p0, 0, NULL, NULL);
cle   =  clEnqueueWriteBuffer(cq[0], brr[1], CL_TRUE, 0, psiz, pn1p0, 0, NULL, NULL);

// device 1
cle   =  clEnqueueWriteBuffer(cq[1], brr[2], CL_TRUE, 0, psiz, pn0p1, 0, NULL, NULL);
cle   =  clEnqueueWriteBuffer(cq[1], brr[3], CL_TRUE, 0, psiz, pn1p1, 0, NULL, NULL);


// building opencl program from .cl files
getb_pg[0]   =  createprogram(ctx, ndev, devs, "3d.cl/get_binfo_3d.cl" , 0);
getb_pg[1]   =  createprogram(ctx, ndev, devs, "3d.cl/get_binfo_3d.cl" , 1);
bexchg_pg[0] =  createprogram(ctx, ndev, devs, "3d.cl/bexchg_3d.cl"    , 0);
bexchg_pg[1] =  createprogram(ctx, ndev, devs, "3d.cl/bexchg_3d.cl"    , 1);
adv_pg[0]    =  createprogram(ctx, ndev, devs, "3d.cl/upstream_3d.cl"  , 0);
adv_pg[1]    =  createprogram(ctx, ndev, devs, "3d.cl/upstream_3d.cl"  , 1);

// device 0
getb_kn[0]   =  clCreateKernel(getb_pg[0],  "get_binfo_3d", &cle);
getb_kn[1]   =  clCreateKernel(getb_pg[0],  "get_binfo_3d", &cle);

// device 1
getb_kn[2]   =  clCreateKernel(getb_pg[1],  "get_binfo_3d", &cle);
getb_kn[3]   =  clCreateKernel(getb_pg[1],  "get_binfo_3d", &cle);

// device 0
bexchg_kn[0] =  clCreateKernel(bexchg_pg[0],"bexchg_3d"   , &cle);
bexchg_kn[1] =  clCreateKernel(bexchg_pg[0],"bexchg_3d"   , &cle);

// device 1
bexchg_kn[2] =  clCreateKernel(bexchg_pg[1],"bexchg_3d"   , &cle);
bexchg_kn[3] =  clCreateKernel(bexchg_pg[1],"bexchg_3d"   , &cle);

// device 0
adv_kn[0]    =  clCreateKernel(adv_pg[0],   "upstream_3d" , &cle);
adv_kn[1]    =  clCreateKernel(adv_pg[0],   "upstream_3d" , &cle);

// device 1
adv_kn[2]    =  clCreateKernel(adv_pg[1],   "upstream_3d" , &cle);
adv_kn[3]    =  clCreateKernel(adv_pg[1],   "upstream_3d" , &cle);



// setting kernel arguments

// get boundary info (west and east only) kernel
// device 0
// get_binfo n0 part 0
cle   =  clSetKernelArg(getb_kn[0], 0, sizeof(cl_mem), &brr[0]); // bn0p0
cle   =  clSetKernelArg(getb_kn[0], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(getb_kn[0], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(getb_kn[0], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(getb_kn[0], 4, sizeof(cl_mem), &b0[0]);
cle   =  clSetKernelArg(getb_kn[0], 5, sizeof(cl_mem), &b0[1]);
cle   =  clSetKernelArg(getb_kn[0], 6, sizeof(cl_mem), &b0[2]);
cle   =  clSetKernelArg(getb_kn[0], 7, sizeof(cl_mem), &b0[3]);
cle   =  clSetKernelArg(getb_kn[0], 8, sizeof(cl_mem), &b0[4]);
cle   =  clSetKernelArg(getb_kn[0], 9, sizeof(cl_mem), &b0[5]);

// get_binfo n1 part 0
cle   =  clSetKernelArg(getb_kn[1], 0, sizeof(cl_mem), &brr[1]); // bn1p0
cle   =  clSetKernelArg(getb_kn[1], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(getb_kn[1], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(getb_kn[1], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(getb_kn[1], 4, sizeof(cl_mem), &b0[0]);
cle   =  clSetKernelArg(getb_kn[1], 5, sizeof(cl_mem), &b0[1]);
cle   =  clSetKernelArg(getb_kn[1], 6, sizeof(cl_mem), &b0[2]);
cle   =  clSetKernelArg(getb_kn[1], 7, sizeof(cl_mem), &b0[3]);
cle   =  clSetKernelArg(getb_kn[1], 8, sizeof(cl_mem), &b0[4]);
cle   =  clSetKernelArg(getb_kn[1], 9, sizeof(cl_mem), &b0[5]);


// device 1
// get_binfo n0 part 1
cle   =  clSetKernelArg(getb_kn[2], 0, sizeof(cl_mem), &brr[2]); // bn0p1
cle   =  clSetKernelArg(getb_kn[2], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(getb_kn[2], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(getb_kn[2], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(getb_kn[2], 4, sizeof(cl_mem), &b1[0]);
cle   =  clSetKernelArg(getb_kn[2], 5, sizeof(cl_mem), &b1[1]);
cle   =  clSetKernelArg(getb_kn[2], 6, sizeof(cl_mem), &b1[2]);
cle   =  clSetKernelArg(getb_kn[2], 7, sizeof(cl_mem), &b1[3]);
cle   =  clSetKernelArg(getb_kn[2], 8, sizeof(cl_mem), &b1[4]);
cle   =  clSetKernelArg(getb_kn[2], 9, sizeof(cl_mem), &b1[5]);

// get_binfo n1 part 1
cle   =  clSetKernelArg(getb_kn[3], 0, sizeof(cl_mem), &brr[3]); // bn1p1
cle   =  clSetKernelArg(getb_kn[3], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(getb_kn[3], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(getb_kn[3], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(getb_kn[3], 4, sizeof(cl_mem), &b1[0]);
cle   =  clSetKernelArg(getb_kn[3], 5, sizeof(cl_mem), &b1[1]);
cle   =  clSetKernelArg(getb_kn[3], 6, sizeof(cl_mem), &b1[2]);
cle   =  clSetKernelArg(getb_kn[3], 7, sizeof(cl_mem), &b1[3]);
cle   =  clSetKernelArg(getb_kn[3], 8, sizeof(cl_mem), &b1[4]);
cle   =  clSetKernelArg(getb_kn[3], 9, sizeof(cl_mem), &b1[5]);


// boundary exchange kernel
// device 0
// bexchg n0p0
cle   =  clSetKernelArg(bexchg_kn[0], 0, sizeof(cl_mem), &brr[0]);
cle   =  clSetKernelArg(bexchg_kn[0], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(bexchg_kn[0], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(bexchg_kn[0], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(bexchg_kn[0], 4, sizeof(cl_mem), &b0[0]);
cle   =  clSetKernelArg(bexchg_kn[0], 5, sizeof(cl_mem), &b0[1]);
cle   =  clSetKernelArg(bexchg_kn[0], 6, sizeof(cl_mem), &b1[2]);
cle   =  clSetKernelArg(bexchg_kn[0], 7, sizeof(cl_mem), &b1[3]);
cle   =  clSetKernelArg(bexchg_kn[0], 8, sizeof(cl_mem), &b0[4]);
cle   =  clSetKernelArg(bexchg_kn[0], 9, sizeof(cl_mem), &b0[5]);

// bexchg n1p0
cle   =  clSetKernelArg(bexchg_kn[1], 0, sizeof(cl_mem), &brr[1]);
cle   =  clSetKernelArg(bexchg_kn[1], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(bexchg_kn[1], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(bexchg_kn[1], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(bexchg_kn[1], 4, sizeof(cl_mem), &b0[0]);
cle   =  clSetKernelArg(bexchg_kn[1], 5, sizeof(cl_mem), &b0[1]);
cle   =  clSetKernelArg(bexchg_kn[1], 6, sizeof(cl_mem), &b1[2]);
cle   =  clSetKernelArg(bexchg_kn[1], 7, sizeof(cl_mem), &b1[3]);
cle   =  clSetKernelArg(bexchg_kn[1], 8, sizeof(cl_mem), &b0[4]);
cle   =  clSetKernelArg(bexchg_kn[1], 9, sizeof(cl_mem), &b0[5]);


// device 1
// bexchg n0p1
cle   =  clSetKernelArg(bexchg_kn[2], 0, sizeof(cl_mem), &brr[2]);
cle   =  clSetKernelArg(bexchg_kn[2], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(bexchg_kn[2], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(bexchg_kn[2], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(bexchg_kn[2], 4, sizeof(cl_mem), &b1[0]);
cle   =  clSetKernelArg(bexchg_kn[2], 5, sizeof(cl_mem), &b1[1]);
cle   =  clSetKernelArg(bexchg_kn[2], 6, sizeof(cl_mem), &b0[2]);
cle   =  clSetKernelArg(bexchg_kn[2], 7, sizeof(cl_mem), &b0[3]);
cle   =  clSetKernelArg(bexchg_kn[2], 8, sizeof(cl_mem), &b1[4]);
cle   =  clSetKernelArg(bexchg_kn[2], 9, sizeof(cl_mem), &b1[5]);

// bexchg n0p1
cle   =  clSetKernelArg(bexchg_kn[3], 0, sizeof(cl_mem), &brr[3]);
cle   =  clSetKernelArg(bexchg_kn[3], 1, sizeof(cl_int), &pz);
cle   =  clSetKernelArg(bexchg_kn[3], 2, sizeof(cl_int), &py);
cle   =  clSetKernelArg(bexchg_kn[3], 3, sizeof(cl_int), &px);
cle   =  clSetKernelArg(bexchg_kn[3], 4, sizeof(cl_mem), &b1[0]);
cle   =  clSetKernelArg(bexchg_kn[3], 5, sizeof(cl_mem), &b1[1]);
cle   =  clSetKernelArg(bexchg_kn[3], 6, sizeof(cl_mem), &b0[2]);
cle   =  clSetKernelArg(bexchg_kn[3], 7, sizeof(cl_mem), &b0[3]);
cle   =  clSetKernelArg(bexchg_kn[3], 8, sizeof(cl_mem), &b1[4]);
cle   =  clSetKernelArg(bexchg_kn[3], 9, sizeof(cl_mem), &b1[5]);



// advection kernel
// adv for the first set of array n 0 part 0
cle   =  clSetKernelArg(adv_kn[0], 0, sizeof(cl_mem),   &brr[0]);
cle   =  clSetKernelArg(adv_kn[0], 1, sizeof(cl_mem),   &brr[1]);
cle   =  clSetKernelArg(adv_kn[0], 2, sizeof(cl_int),   &py);
cle   =  clSetKernelArg(adv_kn[0], 3, sizeof(cl_int),   &px);
cle   =  clSetKernelArg(adv_kn[0], 4, sizeof(cl_float), &u);
cle   =  clSetKernelArg(adv_kn[0], 5, sizeof(cl_float), &v);
cle   =  clSetKernelArg(adv_kn[0], 6, sizeof(cl_float), &w);
cle   =  clSetKernelArg(adv_kn[0], 7, sizeof(cl_float), &c);

// adv for the first set of array n 1 part 0
cle   =  clSetKernelArg(adv_kn[1], 0, sizeof(cl_mem),   &brr[1]);
cle   =  clSetKernelArg(adv_kn[1], 1, sizeof(cl_mem),   &brr[0]);
cle   =  clSetKernelArg(adv_kn[1], 2, sizeof(cl_int),   &py);
cle   =  clSetKernelArg(adv_kn[1], 3, sizeof(cl_int),   &px);
cle   =  clSetKernelArg(adv_kn[1], 4, sizeof(cl_float), &u);
cle   =  clSetKernelArg(adv_kn[1], 5, sizeof(cl_float), &v);
cle   =  clSetKernelArg(adv_kn[1], 6, sizeof(cl_float), &w);
cle   =  clSetKernelArg(adv_kn[1], 7, sizeof(cl_float), &c);
   
// adv for the second set of array n 0 part 1
cle   =  clSetKernelArg(adv_kn[2], 0, sizeof(cl_mem),   &brr[2]);
cle   =  clSetKernelArg(adv_kn[2], 1, sizeof(cl_mem),   &brr[3]);
cle   =  clSetKernelArg(adv_kn[2], 2, sizeof(cl_int),   &py);
cle   =  clSetKernelArg(adv_kn[2], 3, sizeof(cl_int),   &px);
cle   =  clSetKernelArg(adv_kn[2], 4, sizeof(cl_float), &u);
cle   =  clSetKernelArg(adv_kn[2], 5, sizeof(cl_float), &v);
cle   =  clSetKernelArg(adv_kn[2], 6, sizeof(cl_float), &w);
cle   =  clSetKernelArg(adv_kn[2], 7, sizeof(cl_float), &c);

// adv for the second set of array n 1 part 1
cle   =  clSetKernelArg(adv_kn[3], 0, sizeof(cl_mem),   &brr[3]);
cle   =  clSetKernelArg(adv_kn[3], 1, sizeof(cl_mem),   &brr[2]);
cle   =  clSetKernelArg(adv_kn[3], 2, sizeof(cl_int),   &py);
cle   =  clSetKernelArg(adv_kn[3], 3, sizeof(cl_int),   &px);
cle   =  clSetKernelArg(adv_kn[3], 4, sizeof(cl_float), &u);
cle   =  clSetKernelArg(adv_kn[3], 5, sizeof(cl_float), &v);
cle   =  clSetKernelArg(adv_kn[3], 6, sizeof(cl_float), &w);
cle   =  clSetKernelArg(adv_kn[3], 7, sizeof(cl_float), &c);


while(ctr < nt)
{
   for (t=0;t<snt;t++)
   {
      // get boundary info for first set of array
      cle   =  clEnqueueNDRangeKernel(cq[0], getb_kn[0], 3, NULL, glwksiz, NULL, 0, NULL, &getb_ev[0]);
      cle   =  clEnqueueNDRangeKernel(cq[1], getb_kn[2], 3, NULL, glwksiz, NULL, 0, NULL, &getb_ev[2]);

      // boundary exchange for the first set of array
      cle   =  clEnqueueNDRangeKernel(cq[0], bexchg_kn[0], 3, NULL, glwksiz, NULL, 1, &getb_ev[0], &bexchg_ev[0]);
      cle   =  clEnqueueNDRangeKernel(cq[1], bexchg_kn[2], 3, NULL, glwksiz, NULL, 1, &getb_ev[2], &bexchg_ev[2]);


      // get boundary info for second set of array
      cle   =  clEnqueueNDRangeKernel(cq[0], getb_kn[1], 3, NULL, glwksiz, NULL, 1, &bexchg_ev[0], &getb_ev[1]);
      cle   =  clEnqueueNDRangeKernel(cq[1], getb_kn[3], 3, NULL, glwksiz, NULL, 1, &bexchg_ev[2], &getb_ev[3]);

      // boundary exchange for the second set of array
      cle   =  clEnqueueNDRangeKernel(cq[0], bexchg_kn[1], 3, NULL, glwksiz, NULL, 1, &getb_ev[1], &bexchg_ev[1]);
      cle   =  clEnqueueNDRangeKernel(cq[1], bexchg_kn[3], 3, NULL, glwksiz, NULL, 1, &getb_ev[3], &bexchg_ev[3]);


      cle   =  clEnqueueNDRangeKernel(cq[0], adv_kn[0], 3, advofset, advgwsiz, NULL, 1, &bexchg_ev[0], &adv_ev[0]);
      cle   =  clEnqueueNDRangeKernel(cq[0], adv_kn[1], 3, advofset, advgwsiz, NULL, 1, &bexchg_ev[1], &adv_ev[1]);
      cle   =  clEnqueueNDRangeKernel(cq[1], adv_kn[2], 3, advofset, advgwsiz, NULL, 1, &bexchg_ev[2], &adv_ev[2]);
      cle   =  clEnqueueNDRangeKernel(cq[1], adv_kn[3], 3, advofset, advgwsiz, NULL, 1, &bexchg_ev[3], &adv_ev[3]);
   }

   cle   =  clEnqueueReadBuffer(cq[0], brr[0], CL_TRUE, 0, psiz, pn1p0, 4, adv_ev, NULL);
   cle   =  clEnqueueReadBuffer(cq[1], brr[2], CL_TRUE, 0, psiz, pn1p1, 4, adv_ev, NULL);

   dpad0_cat_3d(an0,pn1p0,pn1p1,nz,ny,nx);

   ctr =  ctr + 1;

   start[0] =  ctr;
   if ((ec = nc_put_vara_float(nid, ovid, start, count, &an0[0]))) err(ec);


   printf("time step: %d\n",ctr * snt);
}


printf("    output : %s\n",fnm);

if ((ec = nc_close(nid))) err(ec);


free(an0  );
free(pn0p0);
free(pn0p1);
free(pn1p0);
free(pn1p1);

clFinish(*cq);

clReleaseEvent(*getb_ev);
clReleaseEvent(*mim_ev);
clReleaseEvent(*bexchg_ev);
clReleaseEvent(*adv_ev);

clReleaseMemObject(*brr);
clReleaseMemObject(*b0);
clReleaseMemObject(*b1);

clReleaseKernel(*adv_kn);
clReleaseKernel(*bexchg_kn);
clReleaseKernel(*getb_kn);

clReleaseProgram(*adv_pg);
clReleaseProgram(*bexchg_pg);
clReleaseProgram(*getb_pg);

clReleaseCommandQueue(*cq);
clReleaseContext(ctx);


return 0;
}
