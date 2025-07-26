kernel void upstream_3d(
                        global float *in_p_tn,
                        global float *in_p_tf,
                               int ny,
                               int nx,
                               float u_vel,
                               float v_vel,
                               float w_vel,
                               float c
                       )
{
   uint  i    =  get_global_id(0),
         j    =  get_global_id(1),
         k    =  get_global_id(2);

   uint  idx, idx_i, idx_j, idx_k;

   // indexes dealing with neighbors
   uint  c_xi = i, 
         c_yi = j, 
         c_zi = k,
         m_xi = (i-1), 
         m_yi = (j-1), 
         m_zi = (k-1);

         float uc =  u_vel * c,
               vc =  v_vel * c,
               wc =  w_vel * c;
   

   // full 3d indexes
   idx      =  c_xi + c_yi * nx + c_zi * nx * ny;

   idx_i    =  m_xi + c_yi * nx + c_zi * nx * ny;
   idx_j    =  c_xi + m_yi * nx + c_zi * nx * ny;
   idx_k    =  c_xi + c_yi * nx + m_zi * nx * ny;

   // upstream biased advection scheme
   in_p_tf[idx]  = in_p_tn[idx] 
                 - uc * (in_p_tn[idx] - in_p_tn[idx_i])
                 - vc * (in_p_tn[idx] - in_p_tn[idx_j])
                 - wc * (in_p_tn[idx] - in_p_tn[idx_k]);
}
