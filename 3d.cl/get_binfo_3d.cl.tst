kernel void get_binfo_3d(global float *in_arr, const int in_pz, const int in_py, const int in_px, global float *n, global float *s, global float *w, global float *e, global float *u, global float *d)
{
   // input 3d array should be 0 padded 3d array
   // global work size : {sx,sy,sz}

   // indexing variables
   uint  i  =  get_global_id(0),
         j  =  get_global_id(1),
         k  =  get_global_id(2);

   // indexes for input & output
   uint  nsidx, nsidx0, nsidx1,
         weidx, weidx0, weidx1,
         udidx, udidx0, udidx1;

   // frequently used constants for indexing
   uint  pxp1  =  1,
         pxm2  =  in_px-2,
         pyp1  =  1,
         pym2  =  in_py-2,
         pzp1  =  1,
         pzm2  =  in_pz-2,
         pxy   =  in_px*in_py;


   // 3D west & east (size : sy * sz)
   weidx     =  k + j * in_pz;
   weidx0    =  pxp1 + j * in_px + k * pxy; // 3d west idx
   weidx1    =  pxm2 + j * in_px + k * pxy; // 3d east idx
   w[weidx]  =  in_arr[weidx0];                    // 3d west data
   e[weidx]  =  in_arr[weidx1];                    // 3d east data


   // 3D north & south (size : sy * sx)
   nsidx     =  i + j * in_px;
   nsidx0    =  i + j * in_px + pzp1 * pxy; // 3d south idx
   nsidx1    =  i + j * in_px + pzm2 * pxy; // 3d north idx
   s[nsidx]  =  in_arr[nsidx0];                    // 3d south data
   n[nsidx]  =  in_arr[nsidx1];                    // 3d north data


   // 3D up & down (size : (sz * sx)
   udidx     =  i + k * in_px;
   udidx0    =  i + pyp1 * in_px + k * pxy; // 3d up   idx
   udidx1    =  i + pym2 * in_px + k * pxy; // 3d down idx
   u[udidx]  =  in_arr[udidx0];                    // 3d up   data
   d[udidx]  =  in_arr[udidx1];                    // 3d down data
}
