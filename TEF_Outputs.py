import numpy as np
from glob import glob
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr

def get_roms(path):
    ds = xr.open_mfdataset(path,
                           concat_dim='ocean_time',
                           combine='nested',
                           data_vars='minimal',
                           coords='minimal',
                           parallel=True,
                           chunks={'ocean_time': 1,})
    ds = ds.rename({'eta_u': 'eta_rho', 'xi_v': 'xi_rho', 'xi_psi': 'xi_u', 'eta_psi': 'eta_v'})

    coords = {'X': {'center': 'xi_rho', 'inner': 'xi_u'},
              'Y': {'center': 'eta_rho', 'inner': 'eta_v'},
              'Z': {'center': 's_rho', 'outer': 's_w'}}

    grid = Grid(ds, coords=coords, periodic=[])

    Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    z_rho = Zo_rho * (ds.zeta + ds.h) + ds.zeta
    Zo_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
    z_w = Zo_w * (ds.zeta + ds.h) + ds.zeta

    ds.coords['z_w'] = z_w.where(ds.mask_rho, 0).transpose('ocean_time', 's_w', 'eta_rho', 'xi_rho')
    ds.coords['z_rho'] = z_rho.where(ds.mask_rho, 0).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')

    ds['pm_v'] = grid.interp(ds.pm, 'Y')
    ds['pn_u'] = grid.interp(ds.pn, 'X')
    ds['pm_u'] = grid.interp(ds.pm, 'X')
    ds['pn_v'] = grid.interp(ds.pn, 'Y')
    ds['pm_psi'] = grid.interp(grid.interp(ds.pm, 'Y'), 'X')  # at psi points (eta_v, xi_u)
    ds['pn_psi'] = grid.interp(grid.interp(ds.pn, 'X'), 'Y')  # at psi points (eta_v, xi_u)

    ds['dx'] = 1 / ds.pm
    ds['dx_u'] = 1 / ds.pm_u
    ds['dx_v'] = 1 / ds.pm_v
    ds['dx_psi'] = 1 / ds.pm_psi

    ds['dy'] = 1 / ds.pn
    ds['dy_u'] = 1 / ds.pn_u
    ds['dy_v'] = 1 / ds.pn_v
    ds['dy_psi'] = 1 / ds.pn_psi

    ds['dz'] = grid.diff(ds.z_w, 'Z', boundary='fill')
    ds['dz_w'] = grid.diff(ds.z_rho, 'Z', boundary='fill')
    ds['dz_u'] = grid.interp(ds.dz, 'X')
    ds['dz_w_u'] = grid.interp(ds.dz_w, 'X')
    ds['dz_v'] = grid.interp(ds.dz, 'Y')
    ds['dz_w_v'] = grid.interp(ds.dz_w, 'Y')

    ds['dA'] = ds.dx * ds.dy

    metrics = {
        ('X',): ['dx', 'dx_u', 'dx_v', 'dx_psi'],  # X distances
        ('Y',): ['dy', 'dy_u', 'dy_v', 'dy_psi'],  # Y distances
        ('Z',): ['dz', 'dz_u', 'dz_v', 'dz_w', 'dz_w_u', 'dz_w_v'],  # Z distances
        ('X', 'Y'): ['dA']  # Areas
    }
    grid = Grid(ds, coords=coords, metrics=metrics, periodic=[])
    return ds, grid


salt_bins = np.linspace(25, 40, 101)
dsalt = salt_bins[1]-salt_bins[0]

xislice=slice(284,350)
etaslice=slice(30,118)

months = np.arange(1,13)
for m in months:
    path = '/d1/shared/TXLA_ROMS/output_20yr_obc/2010/ocean_his_00*.nc'
    ds, grid = get_roms(path)
    ds = ds.sel(ocean_time='2010-%02d' %m)
    salt_u = grid.interp(ds.salt, 'X')

    dA_u = ds.dy_u*ds.dz_u

    q_u_total = salt_u*ds.u*dA_u
    q_u_box = q_u_total.isel(xi_u = xislice, eta_rho=etaslice)
    salt_u_box = salt_u.isel(xi_u = xislice, eta_rho=etaslice)

    q_u_right = q_u_box.isel(xi_u = -1)
    q_u_right.name = 'q_right'
    salt_u_right = salt_u_box.isel(xi_u = -1)
    salt_u_right.name = 'salt_right'
    
    q_u_left = q_u_box.isel(xi_u = 0)
    q_u_left.name = 'q_left'
    salt_u_left = salt_u_box.isel(xi_u = 0)
    salt_u_left.name = 'salt_left'

    qlefthist = histogram(salt_u_left, bins = [salt_bins], weights = q_u_left, 
                          dim = ['s_rho', 'eta_rho'])
    qleft = qlefthist.sum(axis = 0) 
    Qleft = qlefthist.sum(axis = 0)*dsalt
    Qleftin = qlefthist.where(qleft>0).sum(axis = 1)*dsalt
    Qlefttout = qlefthist.where(qleft<0).sum(axis = 1)*dsalt
    
    qleft.to_netcdf('qleft_%02d.nc' %m)
    print('qleft_%02d.nc DONE!' %m, flush=True)
    Qleft.to_netcdf('Qleft_%02d.nc' %m)
    Qleftin.to_netcdf('Qleftin_%02d.nc' %m)
    Qleftout.to_netcdf('Qleftout_%02d.nc' %m)
    print('Qleftout_%02d.nc' %m, flush=True)
    
    qrighthist = histogram(salt_u_right, bins = [salt_bins], weights = q_u_right, 
                      dim = ['s_rho', 'eta_rho'])
    
    qright = qrighthist.sum(axis = 0) 
    print('qright_%02d.nc DONE!' %m, flush=True)
    Qright = qrighthist.sum(axis = 0)*dsalt
    Qrightin = qrighthist.where(qright>0).sum(axis = 1)*dsalt
    Qrightout = rightthist.where(qright<0).sum(axis = 1)*dsalt
    print('Qrightout_%02d.nc' %m, flush=True)
    
