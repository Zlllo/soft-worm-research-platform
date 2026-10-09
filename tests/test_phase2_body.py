"""Independent mechanical contracts for the prescribed, nonelastic body."""
from dataclasses import replace
import numpy as np
import pytest
from thermotaxis.phase2_body import BodyConfig, DriveState, shape_geometry, solve_motion, simulate


def dense_reference(config, drive, count=12001):
    """Independent uniform-grid trapezoids, including the material COM rate."""
    s = np.linspace(0., config.length_m, count)
    ds = s[1] - s[0]
    z = 2*np.pi*config.waves*s/config.length_m-drive.phase_rad
    angle = drive.amplitude_rad*np.sin(z)+drive.bias_rad*(s/config.length_m-.5)
    rate = drive.amplitude_rate_rad_s*np.sin(z)-drive.amplitude_rad*drive.phase_rate_rad_s*np.cos(z)+drive.bias_rate_rad_s*(s/config.length_m-.5)
    tangent = np.column_stack((np.cos(angle), np.sin(angle)))
    tangent_rate = np.column_stack((-np.sin(angle)*rate, np.cos(angle)*rate))
    def primitive(y):
        return np.vstack((np.zeros(2), np.cumsum((y[:-1]+y[1:])*ds/2, axis=0)))
    r, v = primitive(tangent), primitive(tangent_rate)
    w = np.full(count, ds); w[[0,-1]] *= .5
    r -= w@r/config.length_m
    v -= w@v/config.length_m
    normal = np.column_stack((-tangent[:,1], tangent[:,0]))
    D = config.drag_parallel_N_s_m2*(tangent[:,:,None]*tangent[:,None,:]+config.drag_ratio*normal[:,:,None]*normal[:,None,:])
    B = np.zeros((count, 2, 3)); B[:,0,0] = 1; B[:,1,1] = 1
    B[:,:,2] = np.column_stack((-r[:,1],r[:,0]))/config.length_m
    M = np.zeros((3,3)); b = np.zeros(3)
    for i in range(3):
        b[i] = -np.sum(w*np.sum(B[:,:,i]*np.einsum('nij,nj->ni',D,v),axis=1))
        for j in range(3):
            M[i,j] = np.sum(w*np.sum(B[:,:,i]*np.einsum('nij,nj->ni',D,B[:,:,j]),axis=1))
    q = np.linalg.solve(M,b)
    vel = v+np.einsum('nij,j->ni',B,q)
    power = np.sum(w*np.sum(vel*np.einsum('nij,nj->ni',D,vel),axis=1))
    return s, r, v, q, power


DYNAMIC = DriveState(.73,.31,.81,5.3,.27,-.42)


def test_material_geometry_and_centred_reference():
    c = BodyConfig()
    g = shape_geometry(c,0,drive=DYNAMIC)
    assert np.sum(g.weights_m) == pytest.approx(c.length_m,abs=1e-17)
    np.testing.assert_allclose(np.linalg.norm(g.tangent,axis=1),1,atol=3e-16)
    np.testing.assert_allclose(g.weights_m@g.position_m,0,atol=1e-21)
    np.testing.assert_allclose(g.weights_m@g.deformation_velocity_m_s,0,atol=1e-21)
    s = np.linspace(0,c.length_m,401)
    h = 1e-9
    inner = s[1:-1]
    derivative = (shape_geometry(c,0,inner+h,drive=DYNAMIC).position_m-shape_geometry(c,0,inner-h,drive=DYNAMIC).position_m)/(2*h)
    np.testing.assert_allclose(derivative,shape_geometry(c,0,inner,drive=DYNAMIC).tangent,atol=1e-9)


@pytest.mark.parametrize('which',['amplitude','phase','bias','all'])
def test_complete_shape_velocity_by_finite_difference(which):
    c = BodyConfig()
    rates = [.31,5.3,-.42]
    keys = ['amplitude','phase','bias']
    d = replace(DYNAMIC,**{k+'_rate_rad_s':rates[i] if which in (k,'all') else 0. for i,k in enumerate(keys)})
    eps = 1e-6
    def advanced(sign):
        return replace(d,**{k+'_rad':getattr(d,k+'_rad')+sign*eps*getattr(d,k+'_rate_rad_s') for k in keys})
    s = np.linspace(0,c.length_m,73)
    fd = (shape_geometry(c,0,s,drive=advanced(1)).position_m-shape_geometry(c,0,s,drive=advanced(-1)).position_m)/(2*eps)
    np.testing.assert_allclose(fd,shape_geometry(c,0,s,drive=d).deformation_velocity_m_s,atol=3e-13,rtol=2e-8)


def test_dense_independent_geometry_and_force_balance_solution():
    c = BodyConfig()
    s,r,v,q,power = dense_reference(c,DYNAMIC)
    g = shape_geometry(c,0,s[::100],drive=DYNAMIC)
    np.testing.assert_allclose(g.position_m,r[::100],atol=2e-11)
    np.testing.assert_allclose(g.deformation_velocity_m_s,v[::100],atol=2e-10)
    motion = solve_motion(c,0,drive=DYNAMIC)
    np.testing.assert_allclose(motion.translation_m_s,q[:2],atol=2e-11)
    assert motion.angular_velocity_rad_s == pytest.approx(q[2]/c.length_m,abs=2e-7)
    assert motion.power_W == pytest.approx(power,rel=2e-7)


def test_force_torque_residual_and_shape_work_identity():
    c = BodyConfig(); m = solve_motion(c,0,drive=DYNAMIC); g=m.geometry
    np.testing.assert_allclose(m.force_N,0,atol=1e-21)
    assert abs(m.torque_N_m)<1e-24
    t=g.tangent; n=np.column_stack((-t[:,1],t[:,0])); vel=m.material_velocity_m_s
    force = -c.drag_parallel_N_s_m2*(np.sum(vel*t,axis=1)[:,None]*t+c.drag_ratio*np.sum(vel*n,axis=1)[:,None]*n)
    shape_work = -np.sum(g.weights_m*np.sum(force*g.deformation_velocity_m_s,axis=1))
    assert m.power_W>0
    assert m.power_W == pytest.approx(shape_work,rel=2e-14)
    # A shifted torque origin leaves torque unchanged because the net force vanishes.
    shifted_r = g.position_m+np.array([.013,-.021])
    torque = np.sum(g.weights_m*(shifted_r[:,0]*force[:,1]-shifted_r[:,1]*force[:,0]))
    assert abs(torque)<1e-23


def test_rotation_covariance_and_drag_scaling():
    c=BodyConfig(); angle=.71; R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    m=solve_motion(c,0,drive=DYNAMIC); rotated=solve_motion(c,0,angle,drive=DYNAMIC)
    np.testing.assert_allclose(rotated.translation_m_s,R@m.translation_m_s,atol=1e-17)
    np.testing.assert_allclose(rotated.material_velocity_m_s,m.material_velocity_m_s@R.T,atol=1e-17)
    assert rotated.angular_velocity_rad_s==pytest.approx(m.angular_velocity_rad_s,abs=2e-14)
    assert rotated.power_W==pytest.approx(m.power_W,rel=2e-14)
    scaled=solve_motion(replace(c,drag_parallel_N_s_m2=7*c.drag_parallel_N_s_m2),0,drive=DYNAMIC)
    np.testing.assert_allclose(scaled.translation_m_s,m.translation_m_s,atol=1e-17)
    assert scaled.power_W==pytest.approx(7*m.power_W,rel=2e-14)


def test_reflection_reverses_chirality():
    d=replace(DYNAMIC,amplitude_rad=-DYNAMIC.amplitude_rad,amplitude_rate_rad_s=-DYNAMIC.amplitude_rate_rad_s,bias_rad=-DYNAMIC.bias_rad,bias_rate_rad_s=-DYNAMIC.bias_rate_rad_s)
    m=solve_motion(BodyConfig(),0,drive=DYNAMIC); mirror=solve_motion(BodyConfig(),0,drive=d)
    np.testing.assert_allclose(mirror.translation_m_s,m.translation_m_s*[1,-1],atol=1e-17)
    assert mirror.angular_velocity_rad_s==pytest.approx(-m.angular_velocity_rad_s,abs=2e-14)
    assert mirror.power_W==pytest.approx(m.power_W,rel=2e-14)


def test_isotropic_drag_zero_com_translation_with_nonzero_rotation_and_work():
    m=solve_motion(BodyConfig(drag_ratio=1),0,drive=DYNAMIC)
    np.testing.assert_allclose(m.translation_m_s,0,atol=1e-17)
    assert abs(m.angular_velocity_rad_s)>.01
    assert m.power_W>0


def test_no_shape_change_has_zero_motion_and_dissipation():
    d=replace(DYNAMIC,amplitude_rate_rad_s=0.,phase_rate_rad_s=0.,bias_rate_rad_s=0.)
    m=solve_motion(BodyConfig(),0,drive=d)
    np.testing.assert_array_equal(m.translation_m_s,np.zeros(2))
    assert m.angular_velocity_rad_s==0 and m.power_W==0


def test_mesh_convergence_resolves_more_complex_curve():
    c=BodyConfig(waves=2.5,amplitude_rad=1.4,bias_rad=.5)
    solutions=[solve_motion(replace(c,quadrature_order=n,integration_order=n),.137) for n in [12,24,48,96]]
    def vector(m): return np.r_[m.translation_m_s,c.length_m*m.angular_velocity_rad_s,m.power_W/1e-6]
    errors=[np.linalg.norm(vector(m)-vector(solutions[-1])) for m in solutions[:-1]]
    assert errors[1]<errors[0]*.1
    assert errors[2]<1e-11


def test_time_convergence_and_final_remainder():
    c=BodyConfig(amplitude_modulation_rad=.12,phase_modulation_rad=.13,bias_modulation_rad=.2,duration_s=1.037,quadrature_order=24,integration_order=24)
    finals=[simulate(replace(c,dt_s=h)) for h in [.04,.02,.01,.005]]
    def state(rows): return np.r_[rows[-1,1:3],rows[-1,3]*c.length_m,rows[-1,4]/1e-6]
    errors=[np.linalg.norm(state(rows)-state(finals[-1])) for rows in finals[:-1]]
    assert errors[1]<errors[0]*.4
    assert errors[2]<errors[1]*.4
    for rows in finals:
        assert rows[-1,0]==c.duration_s
        assert np.all(np.diff(rows[:,4])>=0)
    rotated=simulate(replace(c,dt_s=.02,initial_orientation_rad=.7))
    R=np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
    np.testing.assert_allclose(rotated[:,1:3],finals[1][:,1:3]@R.T,atol=1e-16)


def test_numerically_ill_conditioned_rft_is_reported():
    c=BodyConfig(drag_ratio=1e-16,amplitude_rad=0)
    with pytest.raises((ValueError,np.linalg.LinAlgError),match='condition|singular|finite'):
        solve_motion(c,0)


def test_zero_wave_frequency_still_dissipates_when_amplitude_changes():
    c=BodyConfig(frequency_Hz=0.,amplitude_modulation_rad=.2,bias_modulation_rad=.1)
    m=solve_motion(c,0.)
    assert m.power_W>0
    s=np.linspace(0,c.length_m,31); h=1e-6
    velocity=(shape_geometry(c,h,s).position_m-shape_geometry(c,-h,s).position_m)/(2*h)
    np.testing.assert_allclose(velocity,shape_geometry(c,0,s).deformation_velocity_m_s,atol=1e-13)


def test_invalid_parameters_and_unknown_schema_fail_explicitly():
    with pytest.raises(ValueError): BodyConfig(schema_version=1.0)
    with pytest.raises(ValueError): BodyConfig(drag_parallel_N_s_m2=-1.)
    with pytest.raises(ValueError): shape_geometry(BodyConfig(),0,[-.01])
    with pytest.raises(ValueError): solve_motion(BodyConfig(),0,drive=replace(DYNAMIC,bias_rate_rad_s=float('nan')))
