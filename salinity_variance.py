def salt_variance(ds, xislice, etaslice, tslice):
    '''Returns the three volume-averaged terms in the salinity variance budget: total, vertical,
    and horizontal salinity variance. See Li et al. (2018) JPO for details.

    Input
    ----
    ds: xarray dataset
    xislice: slice object of xi points
    etaslice: slice object of eta points
    tslice: slice object of ocean_time

    Output
    ----
    svar: total salinity variance
    svert: vertical salinity variance
    shorz: horizontal salinity variance
    '''

    if xislice == None:
        xi_rho = ds.xi_rho[:]
    if etaslice == None:
        eta_rho = ds.eta_rho[:]
    if tslice == None:
        tslice = ds.ocean_time[:]


    salt = ds.salt.sel(ocean_time = tslice, xi_rho = xislice, eta_rho=etaslice)
    dV = ds.dx.sel(xi_rho = xislice, eta_rho=etaslice)* \
         ds.dy.sel(xi_rho = xislice, eta_rho=etaslice)* \
         ds.dz.sel(xi_rho = xislice, eta_rho=etaslice)
    z = ds.dz.sel(ocean_time = tslice, xi_rho = xislice, eta_rho=etaslice).sum(dim = 's_rho')

    sbar =(salt*dV).sum(dim = ['eta_rho','xi_rho','s_rho'])/(dV.sum(dim = ['eta_rho','xi_rho','s_rho']))

    svar = (((salt-sbar)**2*dV).sum(dim = ['eta_rho','xi_rho','s_rho']))/(dV.sum(dim = ['eta_rho','xi_rho','s_rho']))

    svbar = ((salt*ds.dz.sel(xi_rho = xislice, eta_rho=etaslice)).sum(dim = ['s_rho']))/z
    svert = (((salt-svbar)**2*dV).sum(dim = ['eta_rho','xi_rho','s_rho']))/(dV.sum(dim = ['eta_rho','xi_rho','s_rho']))
    shorz = svar-svert

    return(svar, svert, shorz)
