kernel void bexchg_3d(global float *in_arr, const int in_pz, const int in_py, const int in_px, global float *n, global float *s, global float *w, global float *e, global float *u, global float *d)
{
   // input 3d array should be 0 padded 3d array

   // indexing variables
   uint  i  =  get_global_id(0),
         j  =  get_global_id(1),
         k  =  get_global_id(2);

   // indexes
   uint  nsidx, nsidx0, nsidx1,
         weidx, weidx0, weidx1,
         udidx, udidx0, udidx1;

   // frequently used constants for indexing
   uint  px0  =  0,
         px1  =  in_px-1,
         px2  =  in_px-2,
         py0  =  0,
         py1  =  in_py-1,
         py2  =  in_py-2,
         pz0  =  0,
         pz1  =  in_pz-1,
         pz2  =  in_pz-2,
         pxy  =  in_px*in_py;


   // 3D onorth is isouth (size : sy * sx)
   // 3D osouth is inorth (size : sy * sx)
   nsidx           =  i + j * in_px;
   nsidx0          =  i + j * in_px + pz1 * pxy; // out north idx
   nsidx1          =  i + j * in_px + pz0 * pxy; // out south idx
   in_arr[nsidx0]  =  s[nsidx];                          // out north data
   in_arr[nsidx1]  =  n[nsidx];                          // out south data


   // 3D owest is ieast (size : sy * sz)
   // 3D oeast is iwest (size : sy * sz)
   weidx           =  k + j * in_pz;
   weidx0          =  px0 + j * in_px + k * pxy; // out west idx
   weidx1          =  px1 + j * in_px + k * pxy; // out east idx
   in_arr[weidx0]  =  e[weidx];                          // out west data
   in_arr[weidx1]  =  w[weidx];                          // out east data


   // 3D oup   is idown (size : sz * sx)
   // 3D odown is iup   (size : sz * sx)
   udidx           =  i + k * in_px;
   udidx0          =  i + py0 * in_px + k * pxy; // out up   idx
   udidx1          =  i + py1 * in_px + k * pxy; // out down idx
   in_arr[udidx0]  =  d[udidx];                          // out up   data
   in_arr[udidx1]  =  u[udidx];                          // out down data
}
